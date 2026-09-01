#!/bin/bash
# 全自动: 对 spider dev 的 7b / 14b 两份 32 采样文件依次跑
#   建exec缓存 -> 基线 -> 多代表锦标赛
#   -> eval(RMAX=3 best-of-32) + eval(RMAX=1 单代表对照)
set -u
cd /path/to/LadderSQL/inference_selection
SP=/path/to/spider_variance_sampling
CA=/path/to/LadderSQL/inference_selection/cache
DB=/tmp/spider_dev_databases
SCHEMA=/path/to/LadderSQL/schema_construction/relevant_schema_cache/spider_glm52_dev_schema_cache.json
export BON_DATASET=spider
export BON_DB_DIR=$DB

run_one () {
  NAME=$1; SRC=$2
  EC=$CA/exec_cache_${NAME}_32.json
  MC=$CA/multirep_pairs_${NAME}_v1.json

  echo "################################################################"
  echo "############### [$NAME] STEP1 build exec cache ###############"
  BIRD_FILE=$SRC EXEC_CACHE=$EC python3 build_exec_cache_generic.py --workers 32

  echo "############### [$NAME] STEP2 baseline ###############"
  BIRD_FILE=$SRC EXEC_CACHE=$EC python3 report_baseline.py

  echo "############### [$NAME] STEP3 tournament ###############"
  BIRD_FILE=$SRC EXEC_CACHE=$EC MULTIREP_CACHE=$MC MULTIREP_RMAX=3 SCHEMA_CACHE=$SCHEMA \
    python3 run_multirep_tournament.py

  echo "############### [$NAME] STEP4a eval RMAX=3 (best-of-32) ###############"
  BIRD_FILE=$SRC EXEC_CACHE=$EC MULTIREP_CACHE=$MC MULTIREP_RMAX=3 \
    python3 eval_multirep.py
  echo "############### [$NAME] STEP4b eval RMAX=1 (single-rep ref) ###############"
  BIRD_FILE=$SRC EXEC_CACHE=$EC MULTIREP_CACHE=$MC MULTIREP_RMAX=1 \
    python3 eval_multirep.py
  echo "############### [$NAME] DONE ###############"
}

run_one spider_dev_7b  $SP/spider_7b_ckpt_20260421_0922_step384_spider_dev_maxturns5_32trials_traj_with_sql.json
run_one spider_dev_14b $SP/spider_ckpt_20260629_153045_step320_spider_dev_maxturns5_32trials_traj_with_sql.json
echo "################ ALL SPIDER DEV DONE ################"
