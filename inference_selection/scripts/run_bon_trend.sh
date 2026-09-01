#!/bin/bash
# ============================================================
# BIRD old dev best-of-N 趋势 · 一次完成 采样 + best-of-N + token/耗时分析
#
#   N ∈ {1(greedy), 8, 16, 24, 32}; 效果用现有 glm-5.2 多代表锦标赛判别器,
#   成本按每题累计(输入/输出 token、LLM 耗时、轮数)。
#   N=8/16/24 从 32 采样池【子采样】(BON_NMAX), 零额外 GPU;
#   裁决走【文本键裁决库】复用已有 32 锦标赛结果, 几乎不新增 glm 调用。
#
# 用法:
#   bash run_bon_trend.sh 14b        # 只跑 14B
#   bash run_bon_trend.sh 7b         # 只跑 7B
#   bash run_bon_trend.sh both       # 两个都跑 (默认)
#   STRICT_NO_NEW_CALLS=1 bash run_bon_trend.sh both   # 严格零新增 glm 调用
#   RUN_SAMPLING=1 ...               # 32 池缺失时才允许触发 DLC/GPU 采样
# ============================================================
set -eo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # -> bon_structured
PYBIN="${PYBIN:-python3}"
PROFILE="${1:-both}"

export BON_DATASET=bird
export BON_DB_DIR="${BON_DB_DIR:-/tmp/bird_dev_databases}"
SCHEMA="${SCHEMA:-/path/to/schema_link_results/bird_old_dev/schema_cache_old_dev_rendered.json}"
OSS_DB="/path/to/nl2sql_dataset/bird/dev_20240627/dev_databases"
NS="${NS:-8,16,24,32}"
NS_DESC="${NS_DESC:-32 24 16 8}"   # 跑锦标赛的顺序(降序: 先跑 32 覆盖最广)
DF_DIR="/path/to/LadderSQL/training/dataset_filter"

# ---------------- DB 本地化到 /tmp (避免 OSS 盘 sqlite I/O error) ----------------
localize_db() {
    if [ ! -d "${BON_DB_DIR}" ] || [ -z "$(ls -A "${BON_DB_DIR}" 2>/dev/null)" ]; then
        echo "[db] 本地化 ${OSS_DB} -> ${BON_DB_DIR}"
        mkdir -p "${BON_DB_DIR}"
        cp -ru "${OSS_DB}/"* "${BON_DB_DIR}/" 2>/dev/null || true
    else
        echo "[db] ${BON_DB_DIR} 已存在, 跳过本地化"
    fi
}

# ---------------- 单模型全流程 ----------------
run_profile() {
    local TAG SRC GREEDY EC SEED TOKENIZER SAMPLE_SH
    TAG="$1"; SRC="$2"; GREEDY="$3"; EC="$4"; SEED="$5"; TOKENIZER="$6"; SAMPLE_SH="$7"
    echo ""
    echo "############################################################"
    echo "# BoN 趋势 · ${TAG}"
    echo "#   32池:     ${SRC}"
    echo "#   greedy:   ${GREEDY}"
    echo "#   种子裁决: ${SEED}"
    echo "############################################################"

    # ---- Stage A: 采样(缺 32 池才触发, 需 DLC/GPU) ----
    if [ ! -f "${SRC}" ]; then
        if [ "${RUN_SAMPLING}" = "1" ] && [ -f "${DF_DIR}/${SAMPLE_SH}" ]; then
            echo "[A] 32 池缺失, 触发采样: ${SAMPLE_SH}"
            ( cd "${DF_DIR}" && bash "${SAMPLE_SH}" )
        else
            echo "[A][ERROR] 32 池不存在且未开启 RUN_SAMPLING=1 (或采样脚本缺失): ${SRC}"
            return 1
        fi
    else
        echo "[A] 32 池已存在, 跳过采样"
    fi

    # ---- Stage B: 执行缓存(缺才触发) ----
    if [ ! -f "${EC}" ]; then
        echo "[B] 构建执行缓存"
        BIRD_FILE="${SRC}" EXEC_CACHE="${EC}" "${PYBIN}" build_exec_cache_generic.py --workers 32
    else
        echo "[B] 执行缓存已存在, 跳过"
    fi

    # ---- Stage C: 各 N best-of-N 锦标赛(复用文本裁决库) ----
    local STORE="cache/verdict_store_${TAG}.json"
    for N in ${NS_DESC}; do
        echo "[C] N=${N} 锦标赛 (TAG=${TAG})"
        BON_NMAX="${N}" BIRD_FILE="${SRC}" EXEC_CACHE="${EC}" \
        MULTIREP_CACHE="cache/multirep_pairs_${TAG}_n${N}.json" \
        VERDICT_STORE="${STORE}" SEED_INDEX_CACHE="${SEED}" \
        SCHEMA_CACHE="${SCHEMA}" MULTIREP_RMAX=3 \
        "${PYBIN}" run_multirep_tournament_bon.py
    done

    # ---- Stage D: 趋势报告(效果 + token/耗时) ----
    echo "[D] 汇总趋势报告"
    BIRD_FILE="${SRC}" EXEC_CACHE="${EC}" \
    "${PYBIN}" eval_bon_trend.py --tag "${TAG}" --src "${SRC}" --greedy "${GREEDY}" \
        --cache-prefix "cache/multirep_pairs_${TAG}_n" --ns "${NS}" \
        --tokenizer "${TOKENIZER}" --out-dir results
}

localize_db

VS_DIR="/path/to/sampling_outputs"
CKPT_ROOT="/path/to/training_outputs/ReAct_agent_bird_redundant"

run_14b() {
    run_profile "step192" \
        "${VS_DIR}/ckpt_20260711_235128_step192_bird_old_dev_maxturns5_32trials_traj_with_sql.json" \
        "${VS_DIR}/ckpt_20260711_235128_step192_bird_old_dev_maxturns5_greedy_traj_with_sql.json" \
        "cache/exec_cache_step192_32.json" \
        "cache/multirep_pairs_step192_v1.json" \
        "${CKPT_ROOT}/20260711_235128/checkpoints/global_step_192/actor/huggingface" \
        "test_bird_redundant_ckpt_20260711_235128_step192_old_dev_32trials_dlc.sh"
}

run_7b() {
    run_profile "bird7b" \
        "${VS_DIR}/ckpt_20260717_124652_step256_7b_bird_old_dev_maxturns5_32trials_traj_with_sql.json" \
        "${VS_DIR}/ckpt_20260717_124652_step256_7b_bird_old_dev_maxturns5_greedy_traj_with_sql.json" \
        "cache/exec_cache_bird7b_32.json" \
        "cache/multirep_pairs_bird7b_v1.json" \
        "${CKPT_ROOT}/20260717_124652/checkpoints/global_step_256/actor/huggingface" \
        "test_bird_redundant_ckpt_20260717_124652_step256_7b_old_dev_32trials_dlc.sh"
}

case "${PROFILE}" in
    14b) run_14b ;;
    7b)  run_7b ;;
    both) run_14b; run_7b ;;
    *) echo "未知 profile: ${PROFILE} (可选 14b|7b|both)"; exit 1 ;;
esac

echo ""
echo "=========================================="
echo " 全部完成! 报告见 bon_structured/results/bon_trend_*.md"
echo "=========================================="
