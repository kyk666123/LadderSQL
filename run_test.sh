#!/bin/bash
# =============================================================================
# LadderSQL — BIRD test inference, stages 2-4 (single entry point)
#
#   stage 2  32 candidate SQL per question   [GPU: our 14B served by vLLM]
#   stage 2b execution cache                 [CPU only]
#   stage 3  reward-blind SQL repair         [closed LLM]
#   stage 4  tournament + selection          [closed LLM] -> predict_test.json
#
# STAGE 1 (schema construction) MUST BE DONE FIRST — see STAGE1_RUNBOOK.md.
#
# Everything is resumable: re-run this script after any interruption and it
# continues from where it stopped (see Readme.md §6).
#
# Usage:
#   export JUDGE_API_KEY=...  JUDGE_API_BASE=https://api.siliconflow.cn/v1
#   TEST_JSON=/data/test.json TEST_DB_DIR=/local/test_databases \
#   SCHEMA_CACHE=/work/schema_cache_test_rendered.json \
#   MODEL_DIR=/local/models/LadderSQL-14B \
#     bash run_test.sh
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python}"     # inference needs no RL training stack: no extra PYTHONPATH (see Readme.md §2)

run() { echo; echo ">>> $*"; echo; eval "$@"; }
# Soft-failure variant: warn but keep going. Used for the three closing steps (format check /
# cost report / EX scoring). They do not depend on each other yet all sit at the end of the
# pipeline, so a hard `set -e` failure in one would throw away the other two -- e.g. a failing
# format check would mean the cost report from a multi-hour run is never produced.
# Failures are appended to WORK_DIR/logs/soft_failures.txt and reported at the end, never swallowed.
run_soft() {
    local n="$1"; shift
    echo; echo ">>> $*"; echo
    if ! eval "$@"; then
        echo "[WARN] ${n} did not pass (not fatal, see the output above); other outputs still produced"
        mkdir -p "${WORK_DIR}/logs"; echo "${n}" >> "${WORK_DIR}/logs/soft_failures.txt"
    fi
}
# Per-stage wall-clock timer (defined early so stage 0 can use it). Also writes each stage's
# seconds under stage_wall/, so runs split across machines can be merged later (the `gen` half
# writes `generation`, the `api` half writes the rest).
declare -A STAGE_SEC
timed() { local n="$1"; shift; local t0=$SECONDS; run "$@"; STAGE_SEC[$n]=$((SECONDS - t0)); echo "  [stage_wall] ${n} = ${STAGE_SEC[$n]}s"; mkdir -p "${WORK_DIR}/logs/stage_wall"; echo "${STAGE_SEC[$n]}" > "${WORK_DIR}/logs/stage_wall/${n}"; }

# Stage selector: all = everything; gen = GPU generation only (no closed LLM);
# api = closed-LLM stages only (schema linking / repair / judge / selection).
# Use it to split "needs a GPU" from "needs internet" across machines, sharing WORK_DIR.
STAGES="${STAGES:-all}"
want() { case "${STAGES}" in all) return 0;; gen) [ "$1" = gen ];; api) [ "$1" = api ];; *) return 0;; esac; }

# ---------------------------- inputs (edit or export) ------------------------
TEST_JSON="${TEST_JSON:-/path/to/test.json}"                  # BIRD test.json
TEST_DB_DIR="${TEST_DB_DIR:-/path/to/local/test_databases}"   # LOCAL disk, <db_id>/<db_id>.sqlite
SCHEMA_CACHE="${SCHEMA_CACHE:-/path/to/schema_cache_test_rendered.json}"   # stage-1 output
MODEL_DIR="${MODEL_DIR:-/path/to/local/models/LadderSQL-14B}"  # LOCAL checkpoint dir (see STAGE 0)
WORK_DIR="${WORK_DIR:-${HERE}/work}"
OUT_PREDICT="${OUT_PREDICT:-${HERE}/predict_test.json}"

# ---- STAGE 0 (optional) staging from a slow mount (OSS / NAS) to local disk ----
# Reading weights and sqlite files straight off object storage on a single-GPU box is slow and
# produces random I/O errors. When the two source variables below are set, this script first
# `cp`s the 14B weights and `unzip`s the databases to local disk (idempotent -- skipped when
# already there), then points MODEL_DIR / TEST_DB_DIR at the local copies. When unset, the
# existing local MODEL_DIR / TEST_DB_DIR are used as-is.
#   MODEL_SRC : HF weight directory on the slow mount (.../global_step_192/actor/huggingface)
#   BIRD_ZIP  : dev_databases.zip on the slow mount (unzips to a top-level dev_databases/)
MODEL_SRC="${MODEL_SRC:-}"
BIRD_ZIP="${BIRD_ZIP:-}"
LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-/root/local_models/LadderSQL-14B}"
LOCAL_DB_ROOT="${LOCAL_DB_ROOT:-/root/local_bird_db}"
DB_SUBDIR="${DB_SUBDIR:-dev_databases}"

# ---------------------------- knobs -----------------------------------------
# Defaults assume a single large-memory GPU (e.g. 96 GB). For an 8-GPU cluster set NUM_GPUS=8.
# The 14B bf16 weights are ~28 GB, so a 96 GB card leaves plenty of room for KV cache ->
# concurrency (MAX_NUM_SEQS / PER_GPU) can be pushed up.
export NUM_GPUS="${NUM_GPUS:-1}"             # one GPU -> one vLLM instance
export BASE_PORT="${BASE_PORT:-9000}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-48}"    # concurrent sequences a 96 GB card can hold
export PER_GPU="${PER_GPU:-48}"              # concurrent rollouts per endpoint (= total on 1 GPU)
export N_TRIALS="${N_TRIALS:-32}"            # candidates per question — KEEP 32
export TEMPERATURE="${TEMPERATURE:-1.1}"     # sampling temperature — KEEP 1.1
export MAX_TURNS_AGENT="${MAX_TURNS_AGENT:-5}"
export MAX_TOKENS="${MAX_TOKENS:-3072}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-24576}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen2.5-coder}"
export LLM_TIMEOUT="${LLM_TIMEOUT:-45}"       # per LLM request
export TRAJ_TIMEOUT="${TRAJ_TIMEOUT:-120}"    # per rollout (hard kill)
export SQL_TIMEOUT="${SQL_TIMEOUT:-15}"       # SQL executed by the agent; raise to 30-60 for giant DBs
export EXEC_TIMEOUT="${EXEC_TIMEOUT:-15}"     # SQL executed by repair / exec-cache; same advice
export SAVE_TRAJECTORY="${SAVE_TRAJECTORY:-0}"  # 1 = keep full ReAct transcripts (10x larger output)
# Isolation mode: questions run one at a time (32-way concurrency *within* a question), so the
# per-question wall clock is clean and contains no cross-question queueing.
# Set 1 when measuring per-question cost (slower, low GPU utilisation); set 0 for the real
# submission run / maximum throughput. Applies to both generation and the tournament.
export ISOLATE_SAMPLES="${ISOLATE_SAMPLES:-0}"
export MULTIREP_WORKERS="${MULTIREP_WORKERS:-32}"   # tournament concurrency (<= min(32, this) per question in isolation mode)
# ---- Cost-measurement convention (profiling): stage 2 (generation) and stage 4 (judging) must
# both run "questions serially + concurrency inside the question", because the unit of work in
# those two stages is "the N parallel sub-tasks of one question".
# Stage 1 (schema linking) and stage 3 (repair) issue a single chain of calls per question with
# nothing to parallelise inside, so they run questions concurrently instead.
#   ISOLATE_SAMPLES=1   generation and judging go question-serial (32 candidates / all pairwise
#                       verdicts of that question run concurrently)
#   APEX_PROC_WORKERS=N schema linking consumes the question queue with N processes (process
#                       isolation keeps token attribution clean)
#   JUDGE_SAMPLE_CONC=N flips judging to N questions in flight + serial within a question
#                       (takes precedence over ISOLATE_SAMPLES). Default 0 = off, so judging
#                       keeps the same convention as generation.
# The three are independent; repair is already 32-way concurrent and needs no switch.
export APEX_PROC_WORKERS="${APEX_PROC_WORKERS:-0}"
export JUDGE_SAMPLE_CONC="${JUDGE_SAMPLE_CONC:-0}"
export SCHEMA_CACHE
export BON_DATASET=bird
export BON_DB_DIR="${TEST_DB_DIR}"
# Closed-LLM usage is accounted automatically (repair + judge share one file, accumulated across
# resumes) so that we can report dev token counts as the guidelines ask.
export LLM_USAGE_LOG="${WORK_DIR:-${HERE}/work}/logs/llm_usage.json"

# selection hyper-parameters — tuned on dev, DO NOT CHANGE for test
VARIANT=biased; TOPK=3; RMAX=3; KPRIOR=1.02

mkdir -p "${WORK_DIR}" "${WORK_DIR}/logs" "${WORK_DIR}/cache"
# The soft-failure list must reflect THIS run only: without clearing it, a step that failed last
# time and works now would keep producing a WARN.
rm -f "${WORK_DIR}/logs/soft_failures.txt"
PARQUET="${WORK_DIR}/test.parquet"
POOL="${WORK_DIR}/candidates_test_${N_TRIALS}.json"
REPAIRED="${WORK_DIR}/candidates_test_${N_TRIALS}_repaired.json"
EC="${WORK_DIR}/cache/exec_cache.json"
ECR="${WORK_DIR}/cache/exec_cache_repaired.json"
MC="${WORK_DIR}/cache/judge_verdicts.json"

echo "=============================================================="
echo " LadderSQL test inference"
echo "   test.json     : ${TEST_JSON}"
echo "   databases     : ${TEST_DB_DIR}"
echo "   schema cache  : ${SCHEMA_CACHE}"
echo "   model         : ${MODEL_DIR}"
echo "   work dir      : ${WORK_DIR}"
echo "   n candidates  : ${N_TRIALS}   temperature ${TEMPERATURE}   max turns ${MAX_TURNS_AGENT}"
echo "   GPUs          : ${NUM_GPUS} (vLLM ports ${BASE_PORT}..$((BASE_PORT+NUM_GPUS-1)))"
echo "   output        : ${OUT_PREDICT}"
echo "=============================================================="

# ---------------------------------------------------------------- stage 0 ----
# Optional: stage weights / databases from a slow mount to local disk (idempotent).
if [ -n "${MODEL_SRC}" ] || [ -n "${BIRD_ZIP}" ]; then
    echo "########## stage 0: stage weights / databases to local disk ##########"
    STAGE_ARGS_MODEL=1; [ -z "${MODEL_SRC}" ] && STAGE_ARGS_MODEL=0
    STAGE_ARGS_DB=1; [ -z "${BIRD_ZIP}" ] && STAGE_ARGS_DB=0
    run "MODEL_SRC='${MODEL_SRC}' LOCAL_MODEL_DIR='${LOCAL_MODEL_DIR}' \
         MODEL_MIN_SHARDS=6 SKIP_MODEL_COPY=$((1-STAGE_ARGS_MODEL)) SKIP_DB_UNZIP=$((1-STAGE_ARGS_DB)) \
         BIRD_ZIP='${BIRD_ZIP}' LOCAL_DB_ROOT='${LOCAL_DB_ROOT}' DB_SUBDIR='${DB_SUBDIR}' \
         DB_MIN_COUNT='${DB_MIN_COUNT:-10}' \
         bash '${HERE}/sample_selection/scripts/prepare_local_data.sh' 2>&1 | tee '${WORK_DIR}/logs/stage0_prepare.log'"
    # after staging, point at the local copies
    [ -n "${MODEL_SRC}" ] && MODEL_DIR="${LOCAL_MODEL_DIR}"
    [ -n "${BIRD_ZIP}" ]  && TEST_DB_DIR="${LOCAL_DB_ROOT}/${DB_SUBDIR}"
    export BON_DB_DIR="${TEST_DB_DIR}"
    echo "  [stage0] MODEL_DIR -> ${MODEL_DIR}"
    echo "  [stage0] TEST_DB_DIR -> ${TEST_DB_DIR}"
fi

for f in "${TEST_JSON}"; do
    [ -f "$f" ] || { echo "[FATAL] missing input: $f"; exit 1; }
done
[ -d "${TEST_DB_DIR}" ] || { echo "[FATAL] missing database dir: ${TEST_DB_DIR} (set BIRD_ZIP to auto-stage)"; exit 1; }
if want gen; then
    [ -f "${SCHEMA_CACHE}" ] || { echo "[FATAL] missing input: ${SCHEMA_CACHE}"; exit 1; }
    [ -d "${MODEL_DIR}" ]   || { echo "[FATAL] missing model dir: ${MODEL_DIR} (set MODEL_SRC to auto-stage)"; exit 1; }
fi
if want api; then
    [ -n "${JUDGE_API_KEY:-}" ] || { echo "[FATAL] JUDGE_API_KEY is not set (needed by stages 3-4)"; exit 1; }
fi
export DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:-${JUDGE_API_KEY:-}}"

# ---- closed-source LLM endpoint + model name -------------------------------------------------
# GLM-5.2 is the same model everywhere, but the model-name STRING is provider-specific
# (SiliconFlow: zai-org/GLM-5.2 ; DashScope: glm-5.2). Derive it from JUDGE_API_BASE so a wrong
# name never reaches the API, unless the caller pins GLM_MODEL explicitly. Also mirror the
# endpoint/key into the OPENAI_* names that stage 1 (APEX + light schema) reads.
export JUDGE_API_BASE="${JUDGE_API_BASE:-https://api.siliconflow.cn/v1}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-${JUDGE_API_BASE}}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-${JUDGE_API_KEY:-}}"
if [ -z "${GLM_MODEL:-}" ]; then
    case "${JUDGE_API_BASE}" in
        *siliconflow*) GLM_MODEL="zai-org/GLM-5.2" ;;
        *)             GLM_MODEL="glm-5.2" ;;
    esac
fi
export GLM_MODEL
export MULTIREP_MODEL="${MULTIREP_MODEL:-${GLM_MODEL}}"
export REPAIR_MODEL="${REPAIR_MODEL:-${GLM_MODEL}}"

# stage_wall.json path (run()/timed()/STAGE_SEC are defined at the top of this script)
STAGE_WALL="${WORK_DIR}/logs/stage_wall.json"

# ---------------------------------------------------------------- stage 1 ----
# Stage 1 = APEX-SQL schema linking (part of the evaluated system). With RUN_SCHEMA_LINK=1 it
# really runs APEX (glm-5.2, 2 samples); timing and tokens land in llm_usage under
# stage=schema_link, and per-question wall clock in schema_link_sample_wall.jsonl.
# Needs APEX_DIR pointing at the directory that holds BIRD/schema_linking.py.
# With 0 (the default) the existing SCHEMA_CACHE is used directly.
RUN_SCHEMA_LINK="${RUN_SCHEMA_LINK:-0}"
APEX_DIR="${APEX_DIR:-}"       # e.g. ./stage1_apex_sql/APEX-SQL-Project/BIRD
SCHEMA_WALL="${WORK_DIR}/apex_out/schema_link_sample_wall.jsonl"
if want api && [ "${RUN_SCHEMA_LINK}" = "1" ]; then
    [ -n "${APEX_DIR}" ] && [ -f "${APEX_DIR}/schema_linking.py" ] || {
        echo "[FATAL] RUN_SCHEMA_LINK=1 requires APEX_DIR to point at the BIRD/ directory containing schema_linking.py"; exit 1; }
    echo "########## stage 1: APEX schema linking (module 1, measured) ##########"
    # Note: this line only overwrites BIRD/chat.py. What APEX actually imports is the file with
    # the same name one directory UP (that copy has a now-revoked key hardcoded), so this cp alone
    # is not enough -- the real fix is in run_apex_sl.py, which claims sys.modules['chat'] by
    # import order. Both guards are kept.
    cp -f "${HERE}/schema_construction/apex_sql_runner/chat.py" "${APEX_DIR}/chat.py"
    mkdir -p "${APEX_DIR}/data/descriptions" "${WORK_DIR}/apex" "${WORK_DIR}/apex_out"
    # The model name is not hardcoded: providers spell it differently
    # (dashscope = glm-5.2, SiliconFlow = zai-org/GLM-5.2).
    # run_apex_sl.py turns '/' into '_' when it builds output filenames.
    APEX_MODEL="${GLM_MODEL:-glm-5.2}"
    APEX_MODEL_SLUG="${APEX_MODEL//\//_}"
    timed schema_link "cd '${APEX_DIR}' && PYTHONPATH='${HERE}/inference_selection:${APEX_DIR}:${PYTHONPATH:-}' \
        ${PY} '${HERE}/schema_construction/apex_sql_runner/run_apex_sl.py' \
          --data_json '${TEST_JSON}' --db_root '${TEST_DB_DIR}' \
          --work_dir '${WORK_DIR}/apex' --out_dir '${WORK_DIR}/apex_out' --model '${APEX_MODEL}' \
          2>&1 | tee -a '${WORK_DIR}/logs/schema_link.log'"
    echo "  [stage1] APEX linking result -> ${WORK_DIR}/apex_out/results_${APEX_MODEL_SLUG}.json"
    echo "  [stage1] rendering it into SCHEMA_CACHE is a cheap, LLM-free step (see STAGE1_RUNBOOK.md)"
    # Under the process-level question queue each child writes its own token-accounting shard;
    # merge them back into the main ledger, otherwise profile_report sees no stage=schema_link tokens.
    # Use absolute paths: the `cd ${APEX_DIR}` inside `timed` above is eval'd in the current shell
    # and leaves cwd in the APEX directory, so a relative tools/... path would not resolve and the
    # stage-1 token detail would never be merged.
    run "${PY} '${HERE}/tools/merge_llm_usage.py' --parts-dir '${WORK_DIR}/apex/usage_parts' \
            --out-detail '${WORK_DIR}/logs/llm_usage_detail.jsonl' \
            --out-json '${WORK_DIR}/logs/llm_usage.json'"
fi

# ---------------------------------------------------------------- stage 2 ----
if want gen; then
if [ ! -s "${PARQUET}" ]; then
  run "${PY} tools/testjson_to_parquet.py --input '${TEST_JSON}' --output '${PARQUET}'"
fi

echo "########## stage 2a: start ${NUM_GPUS} vLLM instances ##########"
run "MODEL_PATH='${MODEL_DIR}' bash sample_selection/scripts/start_vllm_redundant.sh 2>&1 | tee '${WORK_DIR}/logs/vllm.log'"

echo "########## stage 2b: sample ${N_TRIALS} candidates per question ##########"
timed generation "cd '${HERE}/generation' && ${PY} sample_bird_candidates.py \
        --parquet '${PARQUET}' --db-dir '${TEST_DB_DIR}' --output '${POOL}' \
        --n-trials ${N_TRIALS} --num-gpus ${NUM_GPUS} --per-gpu ${PER_GPU} \
        --traj-timeout ${TRAJ_TIMEOUT} --no-gold 2>&1 | tee -a '${WORK_DIR}/logs/generation.log'"

echo "(GPUs are no longer needed from here on; free them if you like:"
echo "   pkill -f \"vllm serve ${MODEL_DIR}\" )"
fi   # want gen

if want api; then
[ -s "${POOL}" ] || { echo "[FATAL] STAGES=api needs the candidate pool ${POOL} to exist first (run STAGES=gen on the GPU machine)"; exit 1; }

# ---------------------------------------------------------------- stage 2c ---
echo "########## stage 2c: execution cache (drives clustering) ##########"
run "cd '${HERE}/inference_selection' && BIRD_FILE='${POOL}' EXEC_CACHE='${EC}' \
        ${PY} build_exec_cache_generic.py --workers 32 2>&1 | tee -a '${WORK_DIR}/logs/exec_cache.log'"

# ---------------------------------------------------------------- stage 3 ----
echo "########## stage 3: reward-blind SQL verification / repair ##########"
timed repair "cd '${HERE}/sql_repair' && BIRD_FILE='${POOL}' EXEC_CACHE='${EC}' \
        ${PY} repair_candidates_generic.py \
          --out-file '${REPAIRED}' --out-cache '${ECR}' \
          --budget 6 --chain default --workers 32 --no-gold \
          --ckpt '${WORK_DIR}/cache/repair_ckpt.json' 2>&1 | tee -a '${WORK_DIR}/logs/repair.log'"

# ---------------------------------------------------------------- stage 4 ----
echo "########## stage 4a: pairwise adjudication (blind, all multi-cluster questions) ##########"
timed judge "cd '${HERE}/inference_selection' && BIRD_FILE='${REPAIRED}' EXEC_CACHE='${ECR}' \
        MULTIREP_CACHE='${MC}' MULTIREP_RMAX=${RMAX} MULTIREP_TOPK=${TOPK} \
        ${PY} tournament_blind.py 2>&1 | tee -a '${WORK_DIR}/logs/judge.log'"

echo "########## stage 4b: final selection + export ##########"
run "cd '${HERE}/inference_selection' && BIRD_FILE='${REPAIRED}' EXEC_CACHE='${ECR}' \
        ${PY} select_and_export.py --multirep-cache '${MC}' \
          --variant ${VARIANT} --topk ${TOPK} --k ${KPRIOR} --rmax ${RMAX} --n ${N_TRIALS} \
          --out-selected '${WORK_DIR}/selected_test.json' \
          --out-predict  '${OUT_PREDICT}' 2>&1 | tee -a '${WORK_DIR}/logs/select.log'"

# ---------------------------------------------------------------- validate ---
echo "########## validation ##########"
run_soft check_predict_format "cd '${HERE}' && ${PY} tools/check_predict_format.py '${OUT_PREDICT}' \
        --db-dir '${TEST_DB_DIR}' --questions '${TEST_JSON}'"

# ---------------------------------------------------------------- cost report -
# Per-question / per-stage wall clock and tokens (avg/min/max). Generation's true per-question
# wall clock comes from the pool's wall_sec field; per-stage totals from stage_wall.json (the
# timer above); repair/judge tokens from the llm_usage detail log.
# Important: seconds of concurrent calls are never summed and called wall clock.
# Collect per-stage wall clock from stage_wall/ (mergeable across machines: `generation` written
# by the GPU box, schema_link/repair/judge by the API box, all under the shared
# WORK_DIR/logs/stage_wall/).
mkdir -p "${WORK_DIR}/logs/stage_wall"
# Skip empty files and non-numeric content when assembling the JSON: an interruption or a full
# disk can leave an empty stage_wall shard, and splicing it in would yield invalid JSON such as
# {"generation": } -- which makes the whole cost report unreadable.
{ printf '{'; _f1=1; for _sf in "${WORK_DIR}"/logs/stage_wall/*; do
    [ -s "$_sf" ] || continue
    _k=$(basename "$_sf"); _v=$(tr -dc '0-9.' < "$_sf")
    [ -n "$_v" ] || continue
    [ $_f1 -eq 1 ] && _f1=0 || printf ','
    printf '"%s": %s' "$_k" "$_v"
  done; printf '}\n'; } > "${STAGE_WALL}"
echo "########## cost report (per-question / per-module) ##########"
run_soft profile_report "cd '${HERE}' && ${PY} tools/profile_report.py \
        --pool '${REPAIRED}' \
        --detail '${WORK_DIR}/logs/llm_usage_detail.jsonl' \
        --stage-wall '${STAGE_WALL}' \
        --judge-wall '${MC%.json}_sample_wall.jsonl' \
        --schema-wall '${SCHEMA_WALL}' \
        --out-per-sample '${WORK_DIR}/logs/per_sample_cost.csv' \
        --out '${WORK_DIR}/logs/cost_report.json' 2>&1 | tee '${WORK_DIR}/logs/cost_report.txt'"

# ---------------------------------------------------------------- EX re-check -
# Only dev has gold. When SCORE_GOLD points at a dev.json carrying gold SQL, EX is computed
# automatically (used to verify 73.66). Test has no gold: leave the variable unset and this is
# skipped.
SCORE_GOLD="${SCORE_GOLD:-}"
if [ -n "${SCORE_GOLD}" ] && [ -f "${SCORE_GOLD}" ]; then
    echo "########## EX check (dev only) ##########"
    run_soft score_predict_dev "cd '${HERE}' && ${PY} tools/score_predict_dev.py \
            --predict '${OUT_PREDICT}' --gold '${SCORE_GOLD}' \
            --db-dir '${TEST_DB_DIR}' --workers 16 \
            2>&1 | tee '${WORK_DIR}/logs/ex_score.txt'"
fi
fi   # want api

echo
echo "=============================================================="
# Soft-failure summary: if any of the closing steps above failed, list it explicitly here --
# otherwise it is only one WARN line in the log and the run looks like it fully succeeded.
if [ -s "${WORK_DIR}/logs/soft_failures.txt" ]; then
  echo " [WARN] these closing steps did not pass (intermediates are unaffected; each can be re-run alone):"
  sed 's/^/        - /' "${WORK_DIR}/logs/soft_failures.txt" | sort -u
  echo
fi
if want gen && ! want api; then
  echo " [gen] candidate pool written: ${POOL}"
  echo " Next: run STAGES=api on a host with internet access (schema linking / repair / judge / selection)."
else
  echo " DONE. Submission file: ${OUT_PREDICT}"
fi
echo " Logs: ${WORK_DIR}/logs/   Intermediates: ${WORK_DIR}/"
echo " Cost report: ${WORK_DIR}/logs/cost_report.txt  (+ llm_usage.json for closed-LLM totals)"
echo "=============================================================="
