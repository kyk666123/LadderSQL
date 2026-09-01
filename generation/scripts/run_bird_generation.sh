#!/bin/bash
# ============================================================
# LadderSQL 推理阶段 1：候选生成（BIRD dev 或 test 通用）
#
#   Step 0  把 checkpoint 与 BIRD 数据库拷/解压到【本地盘】
#           （sqlite 跑在网络盘/对象存储上会随机 disk I/O error，污染聚簇）
#   Step 1  每卡起 1 个 vLLM（8 卡 → 8 端点，端口 9000..9007）
#   Step 2  每题采 N_TRIALS 条多轮 ReAct 轨迹，落盘每条最终 SQL
#
# 全部路径/超参用环境变量传入；默认值 = 论文主结果（BIRD-dev 73.7）的配置。
# 幂等、可断点续跑：直接重复执行即可（已完成的题会跳过）。
#
# 用法：
#   # dev（有金标，可自检 EX）
#   SPLIT=dev bash generation/scripts/run_bird_generation.sh
#   # test（无金标，提交用）
#   SPLIT=test NO_GOLD=1 \
#     PARQUET=/path/to/bird_test/test.parquet \
#     BIRD_ZIP=/path/to/bird_test/test_databases.zip \
#     DB_SUBDIR=test_databases \
#     SCHEMA_CACHE=/path/to/schema_link_results/bird_test/schema_cache_test_rendered.json \
#     OUTPUT=/path/to/candidate_pools/bird_test_32trials_candidates.json \
#     bash generation/scripts/run_bird_generation.sh
# ============================================================
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SPLIT="${SPLIT:-dev}"

# ---------------- 权重（产出 BIRD-dev 73.7 的那份，见 docs/WEIGHTS.md）----------------
MODEL_SRC="${MODEL_SRC:-/path/to/training_outputs/ReAct_agent_bird_redundant/20260711_235128/checkpoints/global_step_192/actor/huggingface}"
LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-/root/local_models/laddersql_14b_bird_step192}"
MODEL_MIN_SHARDS="${MODEL_MIN_SHARDS:-6}"      # 14B=7 分片写 6；7B=4 分片写 4
export MODEL_SRC LOCAL_MODEL_DIR MODEL_MIN_SHARDS

# ---------------- 数据 ----------------
BIRD_ZIP="${BIRD_ZIP:-/path/to/nl2sql_dataset/bird/dev_20240627/dev_databases.zip}"
LOCAL_DB_ROOT="${LOCAL_DB_ROOT:-/root/local_bird_db}"
DB_SUBDIR="${DB_SUBDIR:-dev_databases}"
DB_MIN_COUNT="${DB_MIN_COUNT:-10}"             # dev 11 库 → 10；test 库更多可调大
PARQUET="${PARQUET:-/path/to/nl2sql_dataset/bird/dev_20240627/dev.parquet}"
SCHEMA_CACHE="${SCHEMA_CACHE:-/path/to/schema_link_results/bird_old_dev/schema_cache_old_dev_rendered.json}"
OUTPUT="${OUTPUT:-/path/to/candidate_pools/bird_${SPLIT}_32trials_candidates.json}"
export BIRD_ZIP LOCAL_DB_ROOT DB_SUBDIR DB_MIN_COUNT SCHEMA_CACHE

# ---------------- 采样超参（= 论文主结果配置）----------------
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen2.5-coder}"
export N_TRIALS="${N_TRIALS:-32}"              # 每题候选数 n=32
export TEMPERATURE="${TEMPERATURE:-1.1}"       # 与训练采样温度一致
export MAX_TURNS_AGENT="${MAX_TURNS_AGENT:-5}" # 推理时 ReAct 轮数
export MAX_TOKENS="${MAX_TOKENS:-3072}"        # 单轮生成上限（= 训练 max_response_length）
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-24576}" # vLLM 上下文（= 训练 max_model_len）
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
export LLM_TIMEOUT="${LLM_TIMEOUT:-45}"        # 单次 LLM 请求超时
export TRAJ_TIMEOUT="${TRAJ_TIMEOUT:-120}"     # 单条轨迹硬超时
export PER_GPU="${PER_GPU:-10}"                # 每端点并发 rollout 数
export SAVE_TRAJECTORY="${SAVE_TRAJECTORY:-1}" # 存完整 ReAct 中间轨迹（便于复盘）
export NUM_GPUS="${NUM_GPUS:-8}"
export BASE_PORT="${BASE_PORT:-9000}"
export NO_GOLD="${NO_GOLD:-0}"                 # test 提交必须置 1
NO_GOLD_FLAG=""
[ "${NO_GOLD}" = "1" ] && NO_GOLD_FLAG="--no-gold"

PY="${PY:-python}"

echo "############################################################"
echo "# LadderSQL 候选生成  split=${SPLIT}  n=${N_TRIALS}  T=${TEMPERATURE}  max_turns=${MAX_TURNS_AGENT}"
echo "#   权重:   ${MODEL_SRC}"
echo "#   题目:   ${PARQUET}"
echo "#   schema: ${SCHEMA_CACHE}"
echo "#   库:     ${LOCAL_DB_ROOT}/${DB_SUBDIR}"
echo "#   输出:   ${OUTPUT}   (no_gold=${NO_GOLD})"
echo "############################################################"

echo ">>> [1/3] 本地化模型与数据库"
bash "${REPO_ROOT}/sample_selection/scripts/prepare_local_data.sh"

echo ">>> [2/3] 启动 ${NUM_GPUS} 个 vLLM 实例"
MODEL_PATH="${LOCAL_MODEL_DIR}" bash "${REPO_ROOT}/sample_selection/scripts/start_vllm_redundant.sh"

echo ">>> [3/3] 采样 ${N_TRIALS} 条候选/题"
cd "${REPO_ROOT}/generation"
"${PY}" sample_bird_candidates.py \
    --parquet "${PARQUET}" \
    --db-dir "${LOCAL_DB_ROOT}/${DB_SUBDIR}" \
    --output "${OUTPUT}" \
    --n-trials "${N_TRIALS}" \
    --num-gpus "${NUM_GPUS}" \
    --per-gpu "${PER_GPU}" \
    --traj-timeout "${TRAJ_TIMEOUT}" \
    ${NO_GOLD_FLAG}

echo ""
echo "✅ 候选池已生成: ${OUTPUT}"
echo "   下一步: 执行阶段 2-4（scripts/run_bird_${SPLIT}_end2end.sh 的后半段）"
echo "   释放显存: pkill -f 'vllm serve ${LOCAL_MODEL_DIR}'"
