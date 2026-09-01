#!/bin/bash
# 对 3 个采样文件跑 多代表锦标赛(建判别缓存) + 评估扫参(出最优EX)。
cd /path/to/LadderSQL/inference_selection
SP=/path/to/spider_variance_sampling
BR=/path/to/sampling_outputs
CA=/path/to/LadderSQL/inference_selection/cache
SPIDER_SCHEMA=/path/to/LadderSQL/schema_construction/relevant_schema_cache/spider_glm5_test_schema_cache_20260415.json
BIRD_SCHEMA=../apex-sql-schema-link-results/bird_old_dev/schema_cache_old_dev_rendered.json

run_one () {
  NAME=$1; DS=$2; DB=$3; FILE=$4; ECACHE=$5; MCACHE=$6; SCHEMA=$7
  echo "############### TOURNAMENT $NAME ###############"
  BON_DATASET=$DS BON_DB_DIR=$DB BIRD_FILE=$FILE EXEC_CACHE=$ECACHE \
  MULTIREP_CACHE=$MCACHE MULTIREP_RMAX=3 SCHEMA_CACHE=$SCHEMA \
  python3 run_multirep_tournament.py
  echo "############### EVAL $NAME ###############"
  BON_DATASET=$DS BON_DB_DIR=$DB BIRD_FILE=$FILE EXEC_CACHE=$ECACHE \
  MULTIREP_CACHE=$MCACHE MULTIREP_RMAX=3 \
  python3 eval_multirep.py
}

run_one spider7b spider /tmp/spider_test_databases \
  $SP/spider_7b_ckpt_20260421_0922_step384_spider_test_maxturns5_32trials_traj_with_sql.json \
  $CA/exec_cache_spider7b_32.json \
  $CA/multirep_pairs_spider7b_v1.json $SPIDER_SCHEMA

run_one spider14b spider /tmp/spider_test_databases \
  $SP/spider_ckpt_20260629_153045_step320_spider_test_maxturns5_32trials_traj_with_sql.json \
  $CA/exec_cache_spider14b_32.json \
  $CA/multirep_pairs_spider14b_v1.json $SPIDER_SCHEMA

run_one bird7b bird /tmp/bird_dev_databases \
  $BR/ckpt_20260717_124652_step256_7b_bird_old_dev_maxturns5_32trials_traj_with_sql.json \
  $CA/exec_cache_bird7b_32.json \
  $CA/multirep_pairs_bird7b_v1.json $BIRD_SCHEMA

echo "############### TOURNAMENT+EVAL ALL DONE ###############"
