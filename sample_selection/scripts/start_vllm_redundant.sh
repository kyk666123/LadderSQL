#!/bin/bash
# ============================================================
# Start NUM_GPUS vLLM instances (one per GPU) for redundant-schema sampling.
# Configurable knobs:
#   MAX_MODEL_LEN    context length (qwen2.5=16384, qwen3=32768)
#   MAX_NUM_SEQS     max concurrent sequences per instance (lower it for thinking models to bound KV)
#   EXTRA_VLLM_ARGS  extra vLLM arguments (qwen3 thinking: "--reasoning-parser qwen3")
#   VLLM_DTYPE       weight dtype, default bfloat16 -- do NOT set this to "auto"
# Ports are BASE_PORT..BASE_PORT+NUM_GPUS-1.
# ============================================================
set -e

MODEL_PATH="${MODEL_PATH:-/root/local_models/Qwen2.5-Coder-14B-Instruct}"
BASE_PORT="${BASE_PORT:-9000}"
NUM_GPUS="${NUM_GPUS:-8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen2.5-coder}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
# Force bf16 instead of vLLM's "auto". "auto" reads torch_dtype from config.json; our config.json
# uses the transformers>=4.56 field name "dtype", so an older transformers finds nothing, falls
# back to float32, and the 28 GB checkpoint needs ~56 GB -> OOM on an 80 GB card once the KV cache
# is added. Being explicit here makes the served dtype independent of the transformers version.
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VENV_BIN="${VENV_BIN:-}"

# Only prepend a virtualenv if one was given and actually exists -- otherwise use the vllm on PATH.
if [ -n "${VENV_BIN}" ] && [ -d "${VENV_BIN}" ]; then
    export PATH="${VENV_BIN}:${PATH}"
fi

echo "=== starting ${NUM_GPUS} vLLM instances ==="
echo "model: ${MODEL_PATH}"
echo "ports: ${BASE_PORT}-$((BASE_PORT + NUM_GPUS - 1))"
echo "served-model-name: ${SERVED_MODEL_NAME}"
echo "max-model-len: ${MAX_MODEL_LEN}  max-num-seqs: ${MAX_NUM_SEQS}  dtype: ${VLLM_DTYPE}  extra: ${EXTRA_VLLM_ARGS}"
echo ""

if [ ! -d "${MODEL_PATH}" ]; then
    echo "[ERROR] model path not found: ${MODEL_PATH}"
    exit 1
fi
if ! command -v vllm >/dev/null 2>&1; then
    echo "[ERROR] 'vllm' not on PATH. Install a torch+vLLM pair matching your CUDA (see requirements.txt),"
    echo "        or set VENV_BIN=/path/to/venv/bin."
    exit 1
fi

echo "cleaning up existing vLLM processes..."
pkill -f "vllm serve ${MODEL_PATH}" 2>/dev/null || true
sleep 2

mkdir -p /tmp/vllm_logs

for i in $(seq 0 $((NUM_GPUS - 1))); do
    PORT=$((BASE_PORT + i))
    echo "[GPU ${i}] starting vLLM on port ${PORT}..."

    CUDA_VISIBLE_DEVICES=${i} vllm serve "${MODEL_PATH}" \
        --host 0.0.0.0 \
        --port ${PORT} \
        --dtype "${VLLM_DTYPE}" \
        --served-model-name "${SERVED_MODEL_NAME}" \
        --max-num-seqs ${MAX_NUM_SEQS} \
        --gpu-memory-utilization ${GPU_MEM_UTIL} \
        --max-model-len ${MAX_MODEL_LEN} \
        --enforce-eager \
        ${EXTRA_VLLM_ARGS} \
        > /tmp/vllm_logs/vllm_gpu${i}.log 2>&1 &

    echo "  PID: $!"
    sleep 2
done

echo ""
echo "=== all ${NUM_GPUS} instances launched, waiting for health checks ==="
echo "logs: /tmp/vllm_logs/vllm_gpu{0..$((NUM_GPUS - 1))}.log"
echo ""

ALL_OK=1
for i in $(seq 0 $((NUM_GPUS - 1))); do
    PORT=$((BASE_PORT + i))
    OK=0
    for attempt in $(seq 1 180); do
        if curl -sf http://localhost:${PORT}/health > /dev/null 2>&1; then
            echo "  [GPU ${i}] port ${PORT} ready"
            OK=1
            break
        fi
        sleep 5
    done
    if [ $OK -eq 0 ]; then
        echo "  [GPU ${i}] port ${PORT} failed to come up (see /tmp/vllm_logs/vllm_gpu${i}.log)"
        ALL_OK=0
    fi
done

echo ""
if [ $ALL_OK -eq 1 ]; then
    echo "[OK] all ${NUM_GPUS} vLLM instances ready. Test: curl http://localhost:${BASE_PORT}/v1/models"
    exit 0
else
    echo "[FAIL] some vLLM instances are not ready, check the logs."
    exit 1
fi
