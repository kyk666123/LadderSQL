#!/bin/bash
# =============================================================================
# DLC 8 卡 · 只测【模块 2 = SQL 生成】的时间与 token · 一键运行
#
#   SUB300=1 bash run_gen_8gpu.sh     # 【推荐】跑 300 题, 与本地三个模块【同一批题】
#   bash run_gen_8gpu.sh              # 全量 1534 题(约 5 倍时间)
#
# 不需要任何 API key(纯本地 vLLM)。产物全部落 /mnt/data, DLC 释放也不丢。
#
# SUB300 用的是共享盘上已备好的题目子集(本地那三个模块跑的就是这 300 题),
# 所以两边不只平均值能相加, 逐题也对得上。注意别改用 LIMIT=300 —— 那走的是脚本
# 内置的 int(i*1534/300) 抽样, 与这份子集后半段会错开, 就不是同一批题了。
#
# 【并发口径】总并发 = 8 卡 x 每卡 4 路 = 32。
#   题【内】32 个候选同时生成(摊在 8 张卡上, 每卡 4 路 -> KV 充裕);
#   题【间】串行, 所以每题墙钟干净, 不含别的题排队。
#
# 【跑完看什么】脚本结尾会直接打印三个数:
#   平均每题输入 token / 平均每题输出 token / 平均每题耗时(两种口径都给)
#   —— 拿这三个数去和本地三个模块的对应数字相加, 就是平均端到端。
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export STAGES=gen                 # 只跑模块 2, 不碰 GLM API
export NUM_GPUS="${NUM_GPUS:-8}"  # 8 个 vLLM 实例, 每卡一个(端口 9000..9007)
export PER_GPU="${PER_GPU:-4}"    # 每实例 4 路 -> 8x4 = 32 总并发
export ISOLATE_SAMPLES=1          # 题间串行(每题墙钟干净)
export LIMIT="${LIMIT:-0}"

if [ "${SUB300:-0}" = "1" ]; then
    _SUB=/mnt/data/agl_20260626/_profile_local300/dev_sub300.json
    [ -s "${_SUB}" ] || { echo "[FATAL] 找不到题目子集 ${_SUB}"; exit 1; }
    export TEST_JSON="${_SUB}"
    export WORK_DIR="${WORK_DIR:-/mnt/data/agl_20260626/_profile_gen300}"
    export LIMIT=0                # 子集本身就是 300 题, 不能再让脚本二次抽样
    _SCALE="300 题 (与本地三模块同一批)"
elif [ "${LIMIT}" = "0" ]; then
    _SCALE="全量 1534 题"
else
    _SCALE="${LIMIT} 题"
fi

echo "=============================================================="
echo " 模块 2 (SQL 生成) 成本测量 · ${NUM_GPUS} 卡 x ${PER_GPU} 路 = $(( NUM_GPUS * PER_GPU )) 并发"
echo " 规模: ${_SCALE}"
echo "=============================================================="

bash run_profile_140g.sh

# ---------------------------------------------------------------------------
# 结果汇总: 直接把要上报的三个平均数算出来, 免得再翻 json
# ---------------------------------------------------------------------------
TAG="dev_full"; [ "${LIMIT}" != "0" ] && TAG="dev_pilot${LIMIT}"
# 显式传了 WORK_DIR 就用它, 否则按 TAG 推 —— 不然 SUB300 模式下会去空目录找池子
W="${WORK_DIR:-${OUT_ROOT:-/mnt/data/agl_20260626}/_profile_${TAG}}"
POOL="${W}/candidates_test_32.json"

[ -s "${POOL}" ] || { echo "[WARN] 找不到候选池 ${POOL}, 跳过汇总"; exit 0; }

python3 - "${POOL}" "${W}/logs/stage_wall.json" <<'PY'
import json, sys, statistics as st, os
pool = json.load(open(sys.argv[1]))
wall = {}
if os.path.exists(sys.argv[2]):
    try: wall = json.load(open(sys.argv[2]))
    except Exception: pass

q_in, q_out, q_lat, q_sum = [], [], [], []
for it in pool:
    tr = it.get("trials") or []
    # 每题 = 32 个候选的合计(这才是一道题真实花掉的 token)
    q_in.append(sum((t.get("in_tok") or t.get("prompt_tokens") or 0) for t in tr))
    q_out.append(sum((t.get("out_tok") or t.get("gen_tokens") or 0) for t in tr))
    lt = [t.get("llm_time") or 0 for t in tr if t.get("llm_time")]
    q_sum.append(sum(lt))
    # 题内并发 -> 这题的墙钟由最慢的那个候选决定
    q_lat.append(it.get("wall_sec") or (max(lt) if lt else 0))

n = len(pool)
print("\n" + "=" * 62)
print(f" 模块 2 (SQL 生成) 结果 · {n} 题")
print("=" * 62)
print(f"  平均每题输入 token : {st.mean(q_in):>12,.0f}    (中位 {st.median(q_in):,.0f})")
print(f"  平均每题输出 token : {st.mean(q_out):>12,.0f}    (中位 {st.median(q_out):,.0f})")
print(f"\n  平均每题耗时 —— 两种口径, 跟其它模块相加时【必须用同一种】:")
print(f"    (a) 单题延迟   : {st.mean(q_lat):>10.2f} s   <- 题间串行下这题自己的墙钟")
gw = wall.get("generation")
if gw:
    print(f"    (b) 吞吐摊薄   : {float(gw)/max(n,1):>10.2f} s   <- 阶段总墙钟 {float(gw):.0f}s / {n} 题")
else:
    print(f"    (b) 吞吐摊薄   :        n/a   <- 缺 logs/stage_wall.json")
print(f"\n  参考: 32 候选的 LLM 时间之和 平均 {st.mean(q_sum):.1f} s/题"
      f" (纯串行才会是这个数, 别用它相加)")
print(f"  全量合计: 输入 {sum(q_in)/1e6:.1f}M  输出 {sum(q_out)/1e6:.1f}M")
print("=" * 62)
PY

echo
echo " 产物目录: ${W}"
echo "   候选池           : ${POOL}"
echo "   逐题时间/token   : ${W}/logs/per_sample_cost.csv"
echo "   阶段总墙钟       : ${W}/logs/stage_wall.json"
echo
echo " 把上面三个平均数发回去, 和本地三个模块的数相加即平均端到端。"
