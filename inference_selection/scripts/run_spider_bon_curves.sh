#!/bin/bash
# Spider dev/test × 7B/14B 的 best-of-n 曲线 (n=8/16/24), 候选池口径(与主表89.1/90.2一致)
# n=32 端点用现成缓存, 不重跑。每个 n 用独立 MULTIREP_CACHE(聚簇随n变, 裁决key不可跨n复用)。
set -u
cd /path/to/LadderSQL/inference_selection
SP=/path/to/spider_variance_sampling
CA=/path/to/LadderSQL/inference_selection/cache
DEV_SCHEMA=/path/to/LadderSQL/schema_construction/relevant_schema_cache/spider_glm52_dev_schema_cache.json
TEST_SCHEMA=/path/to/LadderSQL/schema_construction/relevant_schema_cache/spider_glm5_test_schema_cache_20260415.json
export BON_DATASET=spider

run_curve () {
  NAME=$1; DB=$2; REP=$3; EC=$4; SCHEMA=$5
  for N in 8 16 24; do
    MC=$CA/multirep_pairs_${NAME}_n${N}.json
    echo "=============== [$NAME n=$N] tournament ==============="
    BON_DB_DIR=$DB BIRD_FILE=$REP EXEC_CACHE=$EC BON_NMAX=$N \
      MULTIREP_CACHE=$MC MULTIREP_RMAX=3 SCHEMA_CACHE=$SCHEMA \
      python3 run_multirep_tournament.py
    echo "=============== [$NAME n=$N] eval ==============="
    BON_DB_DIR=$DB BIRD_FILE=$REP EXEC_CACHE=$EC BON_NMAX=$N BON_N=$N \
      MULTIREP_CACHE=$MC MULTIREP_RMAX=3 \
      python3 eval_multirep.py
  done
}

# 两联图方案: 只跑 spider test (headline benchmark); dev 曲线暂不需要, 保留命令备用
# run_curve spider_dev_14b /tmp/spider_dev_databases \
#   $SP/spider_ckpt_20260629_153045_step320_spider_dev_maxturns5_32trials_traj_with_sql.json \
#   $CA/exec_cache_spider_dev_14b_32.json $DEV_SCHEMA
# run_curve spider_dev_7b /tmp/spider_dev_databases \
#   $SP/spider_7b_ckpt_20260421_0922_step384_spider_dev_maxturns5_32trials_traj_with_sql.json \
#   $CA/exec_cache_spider_dev_7b_32.json $DEV_SCHEMA

run_curve spider_test_14b /tmp/spider_test_databases \
  $SP/spider_ckpt_20260629_153045_step320_spider_test_maxturns5_32trials_traj_with_sql.json \
  $CA/exec_cache_spider14b_32.json $TEST_SCHEMA

run_curve spider_test_7b /tmp/spider_test_databases \
  $SP/spider_7b_ckpt_20260421_0922_step384_spider_test_maxturns5_32trials_traj_with_sql.json \
  $CA/exec_cache_spider7b_32.json $TEST_SCHEMA

echo "################ ALL SPIDER BEST-OF-N CURVES DONE ################"
