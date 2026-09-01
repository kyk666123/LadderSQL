#!/bin/bash
# =============================================================================
# LadderSQL — BIRD-dev 全量成本测量  ·  单卡 140GB  ·  一键运行
#
#   目的: 在 BIRD old-dev (1534 题) 上完整跑一遍四模块链路, 同时采集
#         (1) 最终 EX (期望 ~73.66)
#         (2) 每题 / 每模块 的端到端耗时
#         (3) 每题 / 每模块 的输入 / 输出 token 数
#
#   并发口径(与 PROFILE_GUIDE.md §3 一致):
#     模块 1 schema link : 32 个【进程】跑样本队列 (APEX_PROC_WORKERS=32)
#     模块 2 SQL 生成    : 样本间【串行】, 每题 32 候选并发 (ISOLATE_SAMPLES=1)
#     模块 3 SQL 修复    : 32 workers 并发
#     模块 4 SQL 选择    : 样本间【串行】, 每题的成对裁决并发 (同模块 2 口径)
#                          不过一题的裁决数本身就常不足 32 -> 32 是并发上限
#   四个模块【严格串行】: 上一个模块 1534 题全部跑完, 才进入下一个。
#
#   用法:
#     export JUDGE_API_KEY=<glm-5.2 的 key>
#     bash run_profile_140g.sh              # 全量 1534 题
#     LIMIT=30 bash run_profile_140g.sh     # 先跑 30 题试点(强烈建议)
#
#   详细说明 / 验收 / 排错: 见同目录 PROFILE_GUIDE.md
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
HERE="$(pwd)"

ASSETS="${ASSETS:-/mnt/data}"                     # 资产根(需挂载同一个 OSS)
OUT_ROOT="${OUT_ROOT:-${ASSETS}/agl_20260626}"    # 所有产物落这里(必须是持久存储)
LIMIT="${LIMIT:-0}"                               # >0 = 只跑前 N 题(试点)
CONC="${CONC:-32}"                                # 统一并发数

TAG="dev_full"; [ "${LIMIT}" != "0" ] && TAG="dev_pilot${LIMIT}"
export WORK_DIR="${WORK_DIR:-${OUT_ROOT}/_profile_${TAG}}"
mkdir -p "${WORK_DIR}/logs"

echo "=============================================================="
echo " LadderSQL BIRD-dev 成本测量"
echo "   资产根   : ${ASSETS}"
echo "   产物目录 : ${WORK_DIR}   ← 跑完把整个目录交给分析方"
# LIMIT=0 不一定就是 1534 题 —— 外部可以把 TEST_JSON 指向一份题目子集, 那时候
# "全量" 指的是那份子集的全部。写死 1534 会和调用方的日志对不上, 让人以为跑错了。
echo "   规模     : $([ "${LIMIT}" = 0 ] && echo 'TEST_JSON 里的全部题目' || echo "试点 ${LIMIT} 题")"
echo "   并发     : ${CONC}"
echo "=============================================================="

# ============================================================================
# 0. 前置硬检查 —— 任何一项不过就立即退出, 绝不空跑几小时
# ============================================================================
fail() { echo "[FATAL] $*" >&2; exit 1; }

# ---- 0.0 本次跑哪些阶段 —— 必须最先定, 后面的硬检查要靠它分支 ----
#   all (默认) : 四个模块都跑, 要 GPU + 要公网 API
#   gen        : 只跑模块 2(SQL 生成, 本地 vLLM) -> 要 GPU, 不要 API
#   api        : 跑模块 1/3/4(APEX 链接、修复、选择) -> 全是纯 API 调用,
#                【完全不需要 GPU】。拆机跑时这一半可以在任意有公网的机器上跑。
STAGES="${STAGES:-all}"
export STAGES
case "${STAGES}" in
    all|gen|api) ;;
    *) fail "STAGES=${STAGES} 不认识。只能是 all / gen / api。" ;;
esac
NEED_API=1; [ "${STAGES}" = "gen" ] && NEED_API=0
NEED_GPU=1; [ "${STAGES}" = "api" ] && NEED_GPU=0
echo "==> 阶段: STAGES=${STAGES}  (需 GPU: ${NEED_GPU} / 需公网 API: ${NEED_API})"

# ---- 0.1 GPU: 只在要跑生成模块时检查 ----
if [ "${NEED_GPU}" = "1" ]; then
# 显存门槛按【每卡实际承担的并发数】算, 而不是写死 60GB —— 多卡时一题的 32 个
# 候选摊到多张卡上, 每卡只需装 PER_GPU 条序列的 KV, 40G 卡完全够。写死 60GB 会误拒。
#   14B bf16 权重 28GB + 每条序列约 4k token x 192 KB/token = 768 MB + 4GB 余量。
_PER_GPU_CHK="${PER_GPU:-${CONC:-32}}"
MIN_GPU_MB="${MIN_GPU_MB:-$(( 28672 + _PER_GPU_CHK * 768 + 4096 ))}"
command -v nvidia-smi >/dev/null 2>&1 || fail "找不到 nvidia-smi, 本机无 GPU"
# ★ 只调一次并把整份输出存进变量, 不要写成 $(nvidia-smi ... | head -1) ——
#   多卡机上 head 读够第一行就关掉管道, nvidia-smi 还在写剩下几行, 于是收到
#   SIGPIPE 以 141 退出; set -euo pipefail 的 pipefail 把 141 传出来, set -e 让整个
#   脚本【静默退出、连 FATAL 都不打印】, 日志正好停在上一行 echo。单卡时输出只有
#   一行、恰好不触发, 所以这个坑只在多卡机器上暴露, 本机怎么测都是好的。
_SMI="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)" \
    || fail "nvidia-smi 存在但执行失败(退出码非 0)。
  常见原因: 驱动与容器镜像的 CUDA 版本不匹配, 或设备未透传进容器。"
GPU_CNT=$(printf '%s\n' "${_SMI}" | wc -l)
_SMI_FIRST="${_SMI%%$'\n'*}"                              # 第一行, 形如 "NVIDIA L20X, 143771"
GPU_NAME="${_SMI_FIRST%%,*}"
GPU_MB="$(printf '%s' "${_SMI_FIRST##*,}" | tr -dc '0-9')"  # nounits 下仍带前导空格, 去掉
echo "==> GPU: ${GPU_NAME}  ${GPU_MB} MiB x ${GPU_CNT} 张"
echo "    计划: ${NUM_GPUS:-1} 个 vLLM 实例 x 每实例 ${_PER_GPU_CHK} 路 = 总并发 $(( ${NUM_GPUS:-1} * _PER_GPU_CHK ))"
# 卡数不够会在起服时才报错(每卡一个实例, CUDA_VISIBLE_DEVICES=i), 在这里先拦下来。
[ "${GPU_CNT}" -ge "${NUM_GPUS:-1}" ] || fail "NUM_GPUS=${NUM_GPUS:-1} 但本机只有 ${GPU_CNT} 张卡。
  起服脚本每卡起一个实例(端口 9000+i), 卡数不够会有实例起不来。
  把 NUM_GPUS 调到 <= ${GPU_CNT}, 并相应提高 PER_GPU 保持总并发不变。"
[ "${GPU_MB}" -ge "${MIN_GPU_MB}" ] || fail "单卡显存 ${GPU_MB} MiB < 所需 ${MIN_GPU_MB} MiB。
  每卡要装 14B bf16 权重 28GB + ${_PER_GPU_CHK} 条序列的 KV(每条约 4k token x 192 KB)。
  当前卡会让这 ${_PER_GPU_CHK} 个候选分批排队, 测出的并发数名不副实。两个出路:
    (a) 多卡摊载(推荐): NUM_GPUS=${GPU_CNT} PER_GPU=$(( 32 / (GPU_CNT>0?GPU_CNT:1) )) bash run_profile_140g.sh
    (b) 接受更低真并发: MIN_GPU_MB=${GPU_MB} bash run_profile_140g.sh
  选 (b) 必须在报告里写明卡型与实际真并发数, 不能声称 32 并发。"
if [ -n "${MIN_GPU_MB_USER:-}" ] || [ "${MIN_GPU_MB}" -lt "$(( 28672 + _PER_GPU_CHK * 768 ))" ]; then
    echo "    [WARN] 显存门槛被手动降到 ${MIN_GPU_MB} MiB —— 每卡 ${_PER_GPU_CHK} 路可能不是真并发, 报告需如实说明"
fi
else
    echo "==> STAGES=api: 模块 1/3/4 全是 API 调用, 跳过 GPU 检查"
fi

# ---- 0.2 磁盘 ----
# 要跑生成就得落 28GB 权重, 门槛 80GB; 只跑 API 阶段不落权重, 但仍要解压 dev 库。
if [ "${NEED_GPU}" = "1" ]; then _MIN_DISK=80; else _MIN_DISK=40; fi
FREE_GB=$(df -BG --output=avail /root 2>/dev/null | tail -1 | tr -dc '0-9')
echo "==> /root 可用 ${FREE_GB} GB (本阶段需 ${_MIN_DISK} GB)"
[ "${FREE_GB:-0}" -ge "${_MIN_DISK}" ] || fail "本地盘可用 ${FREE_GB}GB < ${_MIN_DISK}GB$([ "${NEED_GPU}" = 1 ] && echo '(权重 28G + 库 + 中间产物)' || echo '(dev 库 + 中间产物)')"

# ---- 0.3 资产: 缺任何一个都跑不起来 ----
export MODEL_SRC="${MODEL_SRC:-${ASSETS}/training_outputs/ReAct_agent_bird_redundant/20260711_235128/checkpoints/global_step_192/actor/huggingface}"
export TEST_JSON="${TEST_JSON:-${ASSETS}/nl2sql_dataset/bird/dev_20240627/dev.json}"
export BIRD_ZIP="${BIRD_ZIP:-${ASSETS}/nl2sql_dataset/bird/dev_20240627/dev_databases.zip}"
export SCHEMA_CACHE="${SCHEMA_CACHE:-${OUT_ROOT}/apex-sql-schema-link-results/bird_old_dev/schema_cache_old_dev_rendered.json}"
export APEX_DIR="${APEX_DIR:-${ASSETS}/APEX-SQL-Project-main/APEX-SQL-Project-main/BIRD}"
VENV_TAR="${VENV_TAR:-${ASSETS}/venv_backup.tar.gz}"
UV_PY_TAR="${UV_PY_TAR:-${ASSETS}/uv_python_3.12.12.tar}"

echo "==> 检查资产..."
for p in "${MODEL_SRC}" "${TEST_JSON}" "${BIRD_ZIP}" "${SCHEMA_CACHE}" \
         "${APEX_DIR}/schema_linking.py" "${VENV_TAR}"; do
    [ -e "${p}" ] || fail "缺资产: ${p}
  新机器必须挂载同一个 OSS 到 ${ASSETS}(见 PROFILE_GUIDE.md §2)。"
    echo "    OK  ${p}"
done

# ---- 0.4 凭据 ----
# STAGES=gen 只跑模块 2(纯本地 vLLM), 不碰闭源 API -> 不强制要 key。
# 这是无公网机器上的唯一可行路径(见 PROFILE_GUIDE.md §2.2)。

if [ "${NEED_API}" = "1" ]; then
    [ -n "${JUDGE_API_KEY:-}" ] || fail "JUDGE_API_KEY 未设置。模块 1/3/4 都要调 glm-5.2。
  用法: export JUDGE_API_KEY=<key>  然后重新运行本脚本。
  若本机无公网, 只跑生成模块: STAGES=gen bash run_profile_140g.sh"
else
    echo "==> STAGES=gen: 只跑模块 2(生成), 跳过 API 凭据与公网检查"
fi
export JUDGE_API_BASE="${JUDGE_API_BASE:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
# 模型名因供商而异, 根据 base 自动推断(可用 GLM_MODEL 显式覆盖):
#   dashscope   -> glm-5.2
#   SiliconFlow -> zai-org/GLM-5.2   (已实测; 写成 glm-5.2 会报模型不存在)
if [ -z "${GLM_MODEL:-}" ]; then
    case "${JUDGE_API_BASE}" in
        *siliconflow*) GLM_MODEL="zai-org/GLM-5.2" ;;
        *)             GLM_MODEL="glm-5.2" ;;
    esac
fi
export GLM_MODEL
# 模块 3(修复)读 REPAIR_MODEL/MULTIREP_MODEL, 模块 4(判别)读 MULTIREP_MODEL ——
# 三个模块必须用同一个模型, 否则成本口径对不上。
export MULTIREP_MODEL="${MULTIREP_MODEL:-${GLM_MODEL}}"
export REPAIR_MODEL="${REPAIR_MODEL:-${GLM_MODEL}}"
# APEX 的 chat.py 读 OPENAI_API_KEY/OPENAI_API_BASE, 必须显式同步(否则空 key -> 401 逐题失败)
export OPENAI_API_KEY="${JUDGE_API_KEY:-}"
export OPENAI_API_BASE="${JUDGE_API_BASE}"
export DASHSCOPE_API_KEY="${JUDGE_API_KEY:-}"
if [ "${NEED_API}" = "1" ]; then
    echo "==> 闭源 LLM: model=${GLM_MODEL}  base=${JUDGE_API_BASE}"
fi

# ============================================================================
# 1. 环境: 还原打包好的 venv(含 vllm/torch), 补齐 APEX 依赖
# ============================================================================
VENV_ROOT="${VENV_ROOT:-/root/laddersql_env}"
VENV_DIR="${VENV_ROOT}/.venv"
if [ ! -x "${VENV_DIR}/bin/python" ]; then
    echo "==> 还原 venv (约 5-10 分钟, 5.7GB)..."
    mkdir -p /root/.local/share/uv/python/ "${VENV_ROOT}"
    [ -f "${UV_PY_TAR}" ] && tar -xf "${UV_PY_TAR}" -C /root/.local/share/uv/python/ || true
    tar -xzf "${VENV_TAR}" -C "${VENV_ROOT}/"
    echo "  ✓ venv -> ${VENV_DIR}"
else
    echo "==> venv 已存在, 跳过: ${VENV_DIR}"
fi
export PATH="${VENV_DIR}/bin:${PATH}"
export PY="${VENV_DIR}/bin/python"
export VENV_BIN="${VENV_DIR}/bin"      # start_vllm_redundant.sh 用真实 venv 的 vllm

# APEX 依赖(uv 建的 venv 默认无 pip, 故先 ensurepip)
if ! "${PY}" -c "import sqlglot, json_repair, func_timeout, sqlparse" >/dev/null 2>&1; then
    echo "==> 补装 APEX 依赖..."
    "${PY}" -m pip --version >/dev/null 2>&1 || "${PY}" -m ensurepip --upgrade >/dev/null 2>&1 || true
    APEX_WHEELS="${APEX_WHEELS:-${OUT_ROOT}/_others_repo/_apex_wheels}"
    if [ -d "${APEX_WHEELS}" ] && ls "${APEX_WHEELS}"/*.whl >/dev/null 2>&1; then
        "${PY}" -m pip install -q --no-index --find-links "${APEX_WHEELS}" \
            sqlglot json_repair func_timeout sqlparse
    else
        "${PY}" -m pip install -q sqlglot json_repair func_timeout sqlparse
    fi
    "${PY}" -c "import sqlglot, json_repair, func_timeout, sqlparse" \
        || fail "APEX 依赖装不上(离线 wheel 目录: ${APEX_WHEELS})"
fi
"${PY}" -c "import vllm, torch; print('  [env] vllm', vllm.__version__, 'torch', torch.__version__)" \
    || fail "venv 里没有 vllm/torch, 跑不了生成模块"

# ---- 1.1 公网 + glm-5.2 凭据实测: 真发一次调用, 不通立即停 ----
if [ "${NEED_API}" = "1" ]; then
echo "==> preflight: 实测 ${GLM_MODEL} ..."
"${PY}" - <<'PYPF' || fail "preflight 失败。四种可能:
  (a) key 无效/过期/额度耗尽 —— 401 invalid_api_key 就是这类; 换 key 后重试。
  (b) 本机无公网出口 —— 这类是连接超时/DNS 失败, 不是 401。
      此时只能先跑模块 2: STAGES=gen bash run_profile_140g.sh
  (c) 模型名不对 —— 不同供商叫法不同(dashscope=glm-5.2, SiliconFlow=zai-org/GLM-5.2)。
      列可用模型: curl -s \$JUDGE_API_BASE/models -H \"Authorization: Bearer \$JUDGE_API_KEY\"
      然后用 GLM_MODEL=<正确名字> 覆盖。
  (d) 该模型不支持关 thinking —— 看下面的 reasoning_tokens 提示。
  绝不要跳过这一步硬跑: 会逐题失败空转几小时且产出全是空结果。"
import os
from openai import OpenAI
c = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_API_BASE"],
           timeout=60, max_retries=0)
m = os.environ["GLM_MODEL"]
r = c.chat.completions.create(model=m,
                              messages=[{"role": "user", "content": "reply with OK"}],
                              max_tokens=16, temperature=0,
                              extra_body={"enable_thinking": False})
assert r.choices[0].message.content, "empty response"
u = r.usage
assert u is not None and u.prompt_tokens, "端点没返回 usage —— token 计量会全是 0, 不能用"
rt = getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", None)
print(f"  [preflight] {m} OK -> {r.choices[0].message.content!r} "
      f"(usage in={u.prompt_tokens} out={u.completion_tokens})")
# thinking 必须真的关掉: GLM-5.2 开着思考时 reasoning 占输出的绝大部分,
# 实测同一个提问 129 out_tok(其中 128 是 reasoning) vs 关掉后 1 out_tok。
if rt:
    print(f"  [preflight][WARN] reasoning_tokens={rt} —— thinking 没关掉! "
          f"输出 token 与耗时会数倍放大, 成本测量不可信。请检查供商是否支持 enable_thinking。")
else:
    print("  [preflight] thinking 已关(reasoning_tokens=0), 成本口径干净")
PYPF
fi

# ---- 1.2 自检: 多进程 token 归属(5 秒, 不花钱) ----
# 模块 1 用 32 个进程跑样本队列, 每题的 token 归属靠进程隔离 + 分片账本合并。
# 这里先用 40 题/8 进程/嵌套线程 验证机制, 避免跑了几小时才发现账算串题。
echo "==> 自检: 多进程 token 归属 ..."
"${PY}" tools/_smoke_multiproc_usage.py \
    || fail "多进程记账自检未通过。这是整份成本数据可信的前提(见 PROFILE_GUIDE.md §4.0),
  不要带着它往下跑 —— 否则模块 1 的每题 token 会归错题。"

# ---- 1.3 自检: 逐样本成本报表(秒级, 不花钱) ----
# 验证四模块按题 join 与 CSV 导出都正确, 包括修复改写过的 SQL 其 repair token
# 能否通过 keymap 回查归到题上(不能则每题 repair token 会漏计)。
echo "==> 自检: 逐样本成本报表 ..."
"${PY}" tools/_smoke_per_sample_report.py \
    || fail "逐样本成本报表自检未通过。跑完也拿不到可信的每题时间/token, 先修再跑。"

# ---- 1.4 自检: 模块 4 的并发口径(秒级, 不花钱) ----
# 模块 4 要求「样本间串行 + 题内并发」(同模块 2)。这依赖 ISOLATE_SAMPLES=1 且
# JUDGE_SAMPLE_CONC=0 —— 后者非 0 会静默地把口径翻成「题间并发 + 题内串行」,
# 跑出来的每题墙钟含义就变了。用桩验证调度骨架而不发真请求。
echo "==> 自检: 模块 4 题间串行/题内并发 ..."
"${PY}" tools/_smoke_judge_isolate.py \
    || fail "模块 4 并发口径自检未通过。每题判别墙钟会不是要求的口径, 先修再跑。"

# ============================================================================
# 2. 并发 / 计时口径  —— 改这里之前先读 PROFILE_GUIDE.md §3
# ============================================================================
# 模块 2 (GPU): 起服脚本每卡一个独立 vLLM 实例(端口 9000+i),
#   总并发 = NUM_GPUS x PER_GPU。三个值必须自己配平:
#   • 单卡 140G (默认): NUM_GPUS=1 PER_GPU=32 -> 一题的 32 个候选全在一张卡上。
#   • 多卡 (例: 8 卡): NUM_GPUS=8 PER_GPU=4 -> 总并发仍是 32, 但一题的 32 个
#     候选摊到 8 张卡、每卡只 4 路 -> KV 充裕、单题反而比单卡 32 路更快。
#   注: 改了 NUM_GPUS 就要反向改 PER_GPU, 否则总并发不再是 32, 测出的数没法跟别人比。
export NUM_GPUS="${NUM_GPUS:-1}"
export PER_GPU="${PER_GPU:-${CONC}}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-${PER_GPU}}"   # 单实例并发上限, 与该实例承担的路数相等 -> vLLM 内部零排队
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"

# 模块 2 与模块 4 的【题间】口径:
#   =1 (默认): 样本间串行, 题内并发 -> 每题墙钟不含跨样本排队, 是干净的【单题延迟】。
#   =0        : 样本间也并发 -> 全量快数倍, 但每题墙钟含跨样本挤占, 只能当
#             【吞吐量口径】用。profile_report 会根据候选池里的 isolate_samples
#             标记自动换字段(wall_sec / wall_exec_max), 不需要你手动算。
#   注: token 计量完全不受这个开关影响, 两种跑法的 token 数都是准确的。
export ISOLATE_SAMPLES="${ISOLATE_SAMPLES:-1}"

# 模块 1 (APEX): 32 进程样本队列。用进程而非线程, 因为 APEX 内部有嵌套线程池,
#   子线程只能读模块全局 _GLOBAL_UNIT, 线程池下 32 道题会互相串 token 归属。
export APEX_PROC_WORKERS="${CONC}"
export RUN_SCHEMA_LINK=1

# 模块 4 (判别): 与模块 2 同口径 —— 题间串行, 一题的所有成对裁决同时跑。
#   靠上面的 ISOLATE_SAMPLES=1 生效, 所以 JUDGE_SAMPLE_CONC 必须是 0:
#   它非 0 时会反过来接管成「题间并发 + 题内串行」(tournament_blind.py 的分支优先级)。
#   注: 一题的裁决数 = C(簇数,2) x 写法数^2 x 2, 实际 2~54, 大多数题不足 32,
#   所以 32 是【并发上限】而非 32 路真并发(见 PROFILE_GUIDE.md §3c)。
export JUDGE_SAMPLE_CONC="${JUDGE_SAMPLE_CONC:-0}"
export MULTIREP_WORKERS="${CONC}"   # 题内并发上限 = min(32, 此值), 必须不小于 32

# 本地盘落权重与 sqlite(OSS 上直读 sqlite 会随机 I/O error 且极慢)
export LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-/root/local_models/LadderSQL-14B}"
export LOCAL_DB_ROOT="${LOCAL_DB_ROOT:-/root/local_bird_db}"
export DB_SUBDIR=dev_databases
export DB_MIN_COUNT=10

export OUT_PREDICT="${WORK_DIR}/predict_dev.json"
export SCORE_GOLD="${TEST_JSON}"          # dev 有 gold -> 跑完自动算 EX
export PYTHONUNBUFFERED=1

# 试点模式: 抽 N 题成一个小 dev.json(APEX 与生成都按它跑, 口径完全一致)
# BIRD dev 是按数据库排序的, 取前 N 题只会落在 1 个库里 —— 那样估出的速率
# 不代表全量(单库的 schema 大小 / 缓存命中都有偏)。所以默认均匀跨库抽样。
#   PILOT_MODE=spread (默认) 等间隔抽样, 跨多个库, 适合估全量耗时
#   PILOT_MODE=head          取前 N 题, 适合只想最快验证流程跑不跑得通
PILOT_MODE="${PILOT_MODE:-spread}"
if [ "${LIMIT}" != "0" ]; then
    SMALL="${WORK_DIR}/dev_pilot${LIMIT}_${PILOT_MODE}.json"
    "${PY}" - "$TEST_JSON" "$SMALL" "$LIMIT" "$PILOT_MODE" <<'PYCUT'
import json, sys
src, dst, n, mode = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
full = json.load(open(src, encoding="utf-8"))
if mode == "head" or n >= len(full):
    d = full[:n]
else:
    # 等间隔取样: 保留原顺序, 且尽可能跨库
    step = len(full) / float(n)
    idx = sorted({min(len(full) - 1, int(i * step)) for i in range(n)})
    d = [full[i] for i in idx]
json.dump(d, open(dst, "w", encoding="utf-8"), ensure_ascii=False)
ndb = len({x["db_id"] for x in d})
print(f"  [pilot] mode={mode} 抽 {len(d)} 题 / 跨 {ndb} 个库 -> {dst}")
PYCUT
    export TEST_JSON="${SMALL}"
    export SCORE_GOLD="${SMALL}"
fi

# ============================================================================
# 3. 跑。四模块严格串行, 全部日志实时落 WORK_DIR/logs/
# ============================================================================
echo
echo "==> 开始。实时进度: tail -f ${WORK_DIR}/logs/run.log"
echo "    (可安全中断; 重跑同一条命令会断点续跑)"
echo
T0=$SECONDS
stdbuf -oL -eL bash run_test.sh 2>&1 | stdbuf -oL tee -a "${WORK_DIR}/logs/run.log"
TOTAL=$((SECONDS - T0))

echo
echo "=============================================================="
echo " 完成, 总墙钟 ${TOTAL}s ($((TOTAL / 3600))h $(((TOTAL % 3600) / 60))m)"
echo
echo " 要看的三个文件:"
echo "   1) 成本表   ${WORK_DIR}/logs/cost_report.txt   (每题/每模块 时间+token)"
echo "   2) EX       ${WORK_DIR}/logs/ex_score.txt      (期望 ~73.66)"
echo "   3) 预测     ${WORK_DIR}/predict_dev.json"
echo
echo " 验收(见 PROFILE_GUIDE.md §5):"
echo "   - cost_report 里 end-to-end JOINED coverage 应 = 题数"
echo "   - schema_link / generation / repair / judge 四个模块都要有非零 token"
echo "=============================================================="
