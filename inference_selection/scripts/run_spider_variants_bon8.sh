#!/bin/bash
# ============================================================
# 对 spider 三变体(realistic/syn/dk)的 8采样文件跑 best-of-8:
#   Step1 建 exec 缓存 -> Step2 基线 -> Step3 多代表锦标赛
#   -> eval(RMAX=3 best-of-8) + eval(RMAX=1 单代表)
# best-of-8 关键: BON_N=8(一致性先验分母)。
# 各数据集 schema/库不同:
#   realistic/syn -> spider dev 库(/tmp/spider_dev_databases) + 各自 schema cache
#   dk            -> DK 专用库(/tmp/dk_databases, 含 new_*) + dk schema cache
# ============================================================
set -u
cd /path/to/LadderSQL/inference_selection
export BON_DATASET=spider
export BON_N=8                       # ★ best-of-8: 一致性先验分母
SP=/path/to/spider_variance_sampling
CA=/path/to/LadderSQL/inference_selection/cache
SCHEMA_DIR=/path/to/LadderSQL/schema_construction/relevant_schema_cache

run_one () {
  NAME=$1; SRC=$2; DBDIR=$3; SCHEMA=$4
  EC=$CA/exec_cache_${NAME}_8.json
  MC=$CA/multirep_pairs_${NAME}_v1.json
  export BON_DB_DIR=$DBDIR

  echo "################################################################"
  echo "############### [$NAME] STEP1 build exec cache ###############"
  BIRD_FILE=$SRC EXEC_CACHE=$EC python3 build_exec_cache_generic.py --workers 16

  echo "############### [$NAME] STEP2 baseline ###############"
  BIRD_FILE=$SRC EXEC_CACHE=$EC python3 report_baseline.py

  echo "############### [$NAME] STEP3 tournament ###############"
  BIRD_FILE=$SRC EXEC_CACHE=$EC MULTIREP_CACHE=$MC MULTIREP_RMAX=3 SCHEMA_CACHE=$SCHEMA \
    python3 run_multirep_tournament.py

  echo "############### [$NAME] STEP4a eval RMAX=3 (best-of-8) ###############"
  BIRD_FILE=$SRC EXEC_CACHE=$EC MULTIREP_CACHE=$MC MULTIREP_RMAX=3 \
    python3 eval_multirep.py
  echo "############### [$NAME] STEP4b eval RMAX=1 (single-rep ref) ###############"
  BIRD_FILE=$SRC EXEC_CACHE=$EC MULTIREP_CACHE=$MC MULTIREP_RMAX=1 \
    python3 eval_multirep.py
  echo "############### [$NAME] DONE ###############"
}

run_one realistic $SP/spider_ckpt_step320_realistic_maxturns5_8trials_traj_with_sql.json \
    /tmp/spider_dev_databases $SCHEMA_DIR/spider_realistic_glm52_schema_cache.json
run_one syn       $SP/spider_ckpt_step320_syn_dev_maxturns5_8trials_traj_with_sql.json \
    /tmp/spider_dev_databases $SCHEMA_DIR/spider_syn_glm52_dev_schema_cache.json
run_one dk        $SP/spider_ckpt_step320_dk_maxturns5_8trials_traj_with_sql.json \
    /tmp/dk_databases $SCHEMA_DIR/spider_dk_glm52_schema_cache.json
echo "################ ALL SPIDER VARIANTS best-of-8 DONE ################"
