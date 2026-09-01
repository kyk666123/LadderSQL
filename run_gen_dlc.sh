#!/bin/bash
# =============================================================================
# 个人启动脚本（DLC 单卡 · 仅 SQL 生成模块 · 测 1 卡时间+token）
#   只跑 STAGES=gen: 本地 vLLM 生成 32 候选/题, 不调 glm, 不需要公网。
#   产物(候选池 + 每题干净墙钟 + token)写到 OSS 的 WORK_DIR, 供本机合并成本表。
#   直接运行:  bash run_gen_dlc.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# ---- 还原 DLC 打包的 venv(与 run_dev_full.sh 一致; 不注入 agentlightning) ----
VENV_ROOT="${VENV_ROOT:-/root/laddersql_env}"
VENV_DIR="${VENV_ROOT}/.venv"
if [ ! -x "${VENV_DIR}/bin/python" ]; then
    echo "==> 还原 venv (uv python + venv_backup, 首次约几分钟)..."
    mkdir -p /root/.local/share/uv/python/ "${VENV_ROOT}"
    [ -f /mnt/data/uv_python_3.12.12.tar ] && tar -xf /mnt/data/uv_python_3.12.12.tar -C /root/.local/share/uv/python/ || true
    tar -xzf /mnt/data/venv_backup.tar.gz -C "${VENV_ROOT}/"
    echo "  ✓ venv 还原完成 -> ${VENV_DIR}"
else
    echo "==> venv 已存在, 跳过还原: ${VENV_DIR}"
fi
export PATH="${VENV_DIR}/bin:${PATH}"
export PY="${VENV_DIR}/bin/python"
export VENV_BIN="${VENV_DIR}/bin"
"${PY}" -c "import vllm, torch; print('  [env] vllm', vllm.__version__, 'torch', torch.__version__)" \
    || { echo '[FATAL] venv 里没有 vllm/torch, 无法跑生成阶段'; exit 1; }

# ---- 只跑生成; 单卡; 每题干净计时(样本间顺序, 样本内 32 并发) ----
export STAGES=gen
export ISOLATE_SAMPLES=1          # 1 卡下每题时间干净(慢); 想要吞吐口径改 0
export NUM_GPUS=1
export PER_GPU=48
export MAX_NUM_SEQS=48

# ---- 模型（BIRD-dev 73.7 的 14B, 自动从 OSS 拷到本地盘）----
export MODEL_SRC=/mnt/data/training_outputs/ReAct_agent_bird_redundant/20260711_235128/checkpoints/global_step_192/actor/huggingface
export LOCAL_MODEL_DIR=/root/local_models/LadderSQL-14B

# ---- 数据（dev 库自动解压到本地盘）----
export BIRD_ZIP=/mnt/data/nl2sql_dataset/bird/dev_20240627/dev_databases.zip
export LOCAL_DB_ROOT=/root/local_bird_db
export DB_SUBDIR=dev_databases
export DB_MIN_COUNT=10
export TEST_JSON=/mnt/data/nl2sql_dataset/bird/dev_20240627/dev.json
export SCHEMA_CACHE=/mnt/data/agl_20260626/apex-sql-schema-link-results/bird_old_dev/schema_cache_old_dev_rendered.json

# ---- 工作目录（写到 OSS, 与本机 api 分开; 生成的 stage_wall/generation 也落这里）----
export WORK_DIR="${WORK_DIR:-/mnt/data/agl_20260626/_dlc_gen_run}"

mkdir -p "${WORK_DIR}/logs"
export PYTHONUNBUFFERED=1
echo "==> 前台运行。生成日志/候选池写 OSS: ${WORK_DIR}/  (我可跨机读)"
echo "    实时进度(任意机器): tail -f ${WORK_DIR}/logs/run.log"
stdbuf -oL -eL bash run_test.sh 2>&1 | stdbuf -oL tee "${WORK_DIR}/logs/run.log"

echo ""
echo "===================== 生成模块跑完后 ====================="
echo " 候选池: ${WORK_DIR}/candidates_test_32.json  (含每题 1 卡墙钟 + 输入/输出 token)"
echo " 阶段墙钟: ${WORK_DIR}/logs/stage_wall/generation"
echo " 成本(仅生成部分)可单独看; 完整四模块表由本机汇总。"
echo "========================================================"
