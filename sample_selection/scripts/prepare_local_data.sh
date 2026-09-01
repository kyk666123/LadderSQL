#!/bin/bash
# ============================================================
# Preparation step: copy / unzip the 14B model and the BIRD databases onto a local disk
#   - model: ModelScope download directory -> local disk (28G, plain cp)
#   - data : train_databases.zip -> unzipped locally (8.8G; it MUST be unzipped,
#            copying the directory is not equivalent)
# Idempotent: skips whatever is already present and complete, so it is safe to re-run.
# ============================================================
set -e

# ---- source paths (network storage / ModelScope) ----
MODEL_SRC="${MODEL_SRC:-/path/to/models/Qwen/Qwen2.5-Coder-14B-Instruct}"
BIRD_ZIP="${BIRD_ZIP:-/path/to/nl2sql_dataset/bird/train/train_databases.zip}"

# ---- local destination paths ----
LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-/root/local_models/Qwen2.5-Coder-14B-Instruct}"
LOCAL_DB_ROOT="${LOCAL_DB_ROOT:-/root/local_bird_db}"
DB_SUBDIR="${DB_SUBDIR:-train_databases}"      # pass dev_databases when sampling on dev
DB_MIN_COUNT="${DB_MIN_COUNT:-60}"             # skip the unzip once this many databases exist (dev: 10)
MODEL_MIN_SHARDS="${MODEL_MIN_SHARDS:-6}"      # safetensors shards that mean "model is complete" (14B has 7, use 6)
LOCAL_DB_DIR="${LOCAL_DB_ROOT}/${DB_SUBDIR}"

echo "=== [Prepare] staging model and data locally ==="

# ---------------- 1. copy the model to the local disk ----------------
# SKIP_MODEL_COPY=1 skips the copy (for API-based sampling, where only the local BIRD
# databases are needed to compute the reward).
if [ "${SKIP_MODEL_COPY:-0}" = "1" ]; then
    echo ""
    echo ">>> model: copy skipped (SKIP_MODEL_COPY=1, sampling through a remote API)"
    NEED_COPY_MODEL=0
fi
echo ""
echo ">>> model: ${MODEL_SRC}"
echo "         -> ${LOCAL_MODEL_DIR}"
NEED_COPY_MODEL="${NEED_COPY_MODEL:-1}"
if [ -f "${LOCAL_MODEL_DIR}/config.json" ]; then
    N=$(ls "${LOCAL_MODEL_DIR}"/*.safetensors 2>/dev/null | wc -l)
    if [ "$N" -ge "$MODEL_MIN_SHARDS" ]; then
        echo "    complete model already present (${N} safetensors), skipping."
        NEED_COPY_MODEL=0
    fi
fi
if [ "$NEED_COPY_MODEL" -eq 1 ]; then
    if [ ! -d "${MODEL_SRC}" ]; then
        echo "[ERROR] model source not found: ${MODEL_SRC}"; exit 1
    fi
    mkdir -p "${LOCAL_MODEL_DIR}"
    echo "    copying (about 28G, this takes a while)..."
    cp -a "${MODEL_SRC}/." "${LOCAL_MODEL_DIR}/"
    echo "    model copied."
fi

# ---------------- 2. unzip the BIRD databases locally ----------------
# SKIP_DB_UNZIP=1 skips the unzip (when only the model needs staging and the
# databases are already local).
if [ "${SKIP_DB_UNZIP:-0}" = "1" ]; then
    echo ""; echo ">>> data: unzip skipped (SKIP_DB_UNZIP=1)"
    NEED_UNZIP=0
fi
echo ""
echo ">>> data: ${BIRD_ZIP}"
echo "         -> ${LOCAL_DB_DIR}"
NEED_UNZIP="${NEED_UNZIP:-1}"
if [ "$NEED_UNZIP" -eq 1 ] && [ -d "${LOCAL_DB_DIR}" ]; then
    N=$(ls -d "${LOCAL_DB_DIR}"/*/ 2>/dev/null | grep -v MACOSX | wc -l)
    if [ "$N" -ge "$DB_MIN_COUNT" ]; then
        echo "    unzipped databases already present (${N} of them), skipping."
        NEED_UNZIP=0
    fi
fi
if [ "$NEED_UNZIP" -eq 1 ]; then
    if [ ! -f "${BIRD_ZIP}" ]; then
        echo "[ERROR] data zip not found: ${BIRD_ZIP}"; exit 1
    fi
    mkdir -p "${LOCAL_DB_ROOT}"
    echo "    unzipping (must be unzipped, not copied)..."
    # the zip's top-level entry is already ${DB_SUBDIR}/, so extract into LOCAL_DB_ROOT
    unzip -q -o "${BIRD_ZIP}" -d "${LOCAL_DB_ROOT}"
    rm -rf "${LOCAL_DB_ROOT}/__MACOSX" 2>/dev/null || true
    echo "    data unzipped."
fi

echo ""
echo "[OK] [Prepare] done"
echo "   local model: ${LOCAL_MODEL_DIR}"
echo "   local data : ${LOCAL_DB_DIR}"
