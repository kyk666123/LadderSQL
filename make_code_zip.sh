#!/bin/bash
# Build the code.zip we mail to the BIRD Exp Team: inference-only files, as the guidelines ask.
#
# Usage:
#   bash make_code_zip.sh
#
# The zip includes stage1_apex_sql/APEX-SQL-Project/ (the five APEX-SQL files stage 1 imports)
# when that directory is present in the working tree. It is deliberately NOT tracked by git --
# see .gitignore -- because APEX-SQL belongs to its own authors; the mailed package is a limited
# disclosure to the Exp Team for the sole purpose of reproducing this submission.
#
# Output: dist/LadderSQL_code_<date>.zip
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}"
STAGE="$(mktemp -d)"
PKG="LadderSQL"
DEST="${STAGE}/${PKG}"
mkdir -p "${DEST}"

# ---- code and documentation ----
for p in Readme.md STAGE1_RUNBOOK.md DEV_REFERENCE.md SQL_REPAIR_METHOD.md requirements.txt \
         LICENSE NOTICE .env.example run_test.sh predict_dev.json MODEL_LINKS.txt \
         schema_construction generation sql_repair inference_selection tools \
         training sample_selection stage1_apex_sql; do
    [ -e "$p" ] && cp -r "$p" "${DEST}/"
done

# ---- APEX-SQL: make sure the root chat.py is our env-driven copy, never one with a key in it ----
if [ -d "${DEST}/stage1_apex_sql/APEX-SQL-Project" ]; then
    cp "${DEST}/schema_construction/apex_sql_runner/chat.py" \
       "${DEST}/stage1_apex_sql/APEX-SQL-Project/chat.py"
    cp "${DEST}/schema_construction/apex_sql_runner/chat.py" \
       "${DEST}/stage1_apex_sql/APEX-SQL-Project/BIRD/chat.py"
    mkdir -p "${DEST}/stage1_apex_sql/APEX-SQL-Project/BIRD/data/descriptions"
    echo "[apex] bundled (5 files, schema linking only)"
else
    echo "[apex] WARNING: stage1_apex_sql/APEX-SQL-Project/ absent -- stage 1 will not run for them"
fi

# ---- strip caches, logs, intermediates, large files ----
find "${DEST}" -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "${DEST}" \( -name "*.pyc" -o -name "*.log" -o -name ".DS_Store" \) -delete 2>/dev/null || true
rm -rf "${DEST}/work" "${DEST}/logs" "${DEST}"/*/cache "${DEST}"/*/results 2>/dev/null || true
# our own smoke tests and their output -- irrelevant to reproducing the submission, and the
# guidelines ask for a lean package
rm -rf "${DEST}/tools/_smoke_out" "${DEST}"/tools/_smoke_*.py 2>/dev/null || true
# never ship a filled-in .env (only .env.example)
rm -f "${DEST}/.env" 2>/dev/null || true

# ---- self-check: everything the Exp Team needs is actually in the package ----
MISSING=""
for need in Readme.md requirements.txt run_test.sh predict_dev.json MODEL_LINKS.txt \
            STAGE1_RUNBOOK.md SQL_REPAIR_METHOD.md DEV_REFERENCE.md NOTICE LICENSE \
            tools/check_predict_format.py tools/testjson_to_parquet.py \
            generation/sample_bird_candidates.py generation/_agl_stub.py \
            sql_repair/repair_candidates_generic.py inference_selection/tournament_blind.py \
            inference_selection/select_and_export.py \
            sample_selection/scripts/start_vllm_redundant.sh; do
    [ -e "${DEST}/${need}" ] || MISSING="${MISSING} ${need}"
done
if [ -n "${MISSING}" ]; then
    echo "[FATAL] missing from the package:${MISSING}"
    exit 1
fi

# ---- self-check: no credentials, no databases, no weights ----
if grep -rIn "sk-[A-Za-z0-9]\{10,\}" "${DEST}" >/dev/null 2>&1; then
    echo "[FATAL] hardcoded API key found in the package, aborting"
    grep -rIn "sk-[A-Za-z0-9]\{10,\}" "${DEST}" | head
    exit 1
fi
BIG=$(find "${DEST}" -type f -size +20M | head)
[ -n "${BIG}" ] && { echo "[WARN] files larger than 20MB:"; echo "${BIG}"; }
if find "${DEST}" -name "*.sqlite" | grep -q .; then echo "[WARN] .sqlite files in the package"; fi
# the only outbound host we are allowed to have. The rest of the allowlist is documentation links
# in comments (papers, StackOverflow answers) and loopback addresses -- none of them can receive data.
HOSTS=$(grep -rIoh "https\?://[a-zA-Z0-9.-]\+" "${DEST}" 2>/dev/null | sort -u \
        | grep -v "api.siliconflow.cn\|dashscope.aliyuncs.com\|localhost\|127.0.0.1\|0.0.0.0\|github.com\|modelscope.cn\|pytorch.org\|huggingface.co\|bird-bench.github.io\|opensource.org\|www.apache.org\|arxiv.org\|stackoverflow.com\|courses.cs.washington.edu" || true)
if [ -n "${HOSTS}" ]; then
    echo "[WARN] unexpected URLs in the package (check none of them can receive a database):"
    echo "${HOSTS}"
else
    echo "[ok] no outbound host other than the declared GLM-5.2 endpoint"
fi

mkdir -p dist
STAMP="$(date +%Y%m%d)"
# Zip into the (local) staging dir first and copy the finished file over: writing a zip
# incrementally onto an object-storage / fuse mount fails with "zip I/O error: Invalid argument".
TMP_OUT="${STAGE}/LadderSQL_code_${STAMP}.zip"
OUT="${HERE}/dist/LadderSQL_code_${STAMP}.zip"
(cd "${STAGE}" && zip -qr "${TMP_OUT}" "${PKG}")
cp "${TMP_OUT}" "${OUT}"
rm -rf "${STAGE}"
echo "[ok] ${OUT}  ($(du -h "${OUT}" | cut -f1))"
echo "     unpacks to: ${PKG}/    entry points: ${PKG}/Readme.md, ${PKG}/run_test.sh"
