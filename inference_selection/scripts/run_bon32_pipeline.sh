#!/bin/bash
# ============================================================
# 同模型 best-of-32 pipeline
#   采样文件 -> 锦标赛判别 -> 评估取峰值。
#   全程同一模型: MULTIREP_MODEL=$MODEL 驱动锦标赛判别
#     (run_multirep_tournament.MODEL 读 MULTIREP_MODEL)
#   judge enable_thinking=False, temperature=0, reward-blind。
#
#   用法: MODEL=<模型> TAG=<短标签> SRC=<采样json绝对路径> bash run_bon32_pipeline.sh
# ============================================================
set -uo pipefail
cd /path/to/LadderSQL/inference_selection
PY=/path/to/LadderSQL/.venv/bin/python

MODEL="${MODEL:?必须指定 MODEL}"
TAG="${TAG:?必须指定 TAG}"
SRC="${SRC:?必须指定 SRC 采样json}"

export BON_DATASET=bird
export BON_DB_DIR="${BON_DB_DIR:-/root/local_bird_db/dev_databases}"
export MULTIREP_MODEL="$MODEL"      # ★ 驱动判别, 保证全程同模型
export JUDGE_API_BASE="${JUDGE_API_BASE:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export JUDGE_API_KEY="${JUDGE_API_KEY:?set JUDGE_API_KEY to your judge LLM API key}"

CA=/path/to/LadderSQL/inference_selection/cache
SCHEMA=/path/to/schema_link_results/bird_old_dev/schema_cache_old_dev_rendered.json
EC="$CA/exec_cache_${TAG}_32.json"
MC="$CA/multirep_pairs_${TAG}_v1.json"

echo "############################################################"
echo "# best-of-32 pipeline | model=$MODEL  tag=$TAG"
echo "#   SRC=$SRC"
echo "#   开始: $(date '+%F %T')"
echo "############################################################"

echo ""; echo ">>> [1/4] 建执行缓存 ..."
BIRD_FILE="$SRC" EXEC_CACHE="$EC" "$PY" build_exec_cache_generic.py --workers 32 || { echo "[FATAL] step1 失败"; exit 1; }

echo ""; echo ">>> [2/4] 基线 (多数投票 / pass@32 天花板) ..."
BIRD_FILE="$SRC" EXEC_CACHE="$EC" "$PY" report_baseline.py

echo ""; echo ">>> [3/4] 多代表锦标赛 (同模型判别, 可断点续跑) ..."
BIRD_FILE="$SRC" EXEC_CACHE="$EC" MULTIREP_CACHE="$MC" MULTIREP_RMAX=3 SCHEMA_CACHE="$SCHEMA" \
    "$PY" run_multirep_tournament.py || { echo "[FATAL] step3 失败"; exit 1; }

echo ""; echo ">>> [4/4] 评估 best-of-32 ..."
echo "===== RMAX=3 (多代表, 取峰值即 best-of-32) ====="
BIRD_FILE="$SRC" EXEC_CACHE="$EC" MULTIREP_CACHE="$MC" MULTIREP_RMAX=3 "$PY" eval_multirep.py
echo "===== RMAX=1 (单代表对照) ====="
BIRD_FILE="$SRC" EXEC_CACHE="$EC" MULTIREP_CACHE="$MC" MULTIREP_RMAX=1 "$PY" eval_multirep.py

echo ""; echo "===== best-of-32 pipeline 完成: $MODEL ($TAG) @ $(date '+%F %T') ====="
