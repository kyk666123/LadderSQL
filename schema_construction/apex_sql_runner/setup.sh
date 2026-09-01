#!/bin/bash
set -e
ROOT=/path/to/APEX-SQL-Project-main/APEX-SQL-Project-main
BIRD=$ROOT/BIRD
mkdir -p $BIRD/data/descriptions
mkdir -p /root/apex_sl/bird_clean_dev
mkdir -p /root/apex_sl_dbs
mkdir -p /path/to/schema_link_results/bird_clean_dev
cp /path/to/LadderSQL/.apex_sl_temp/chat.py $ROOT/chat.py
cp /path/to/LadderSQL/.apex_sl_temp/convert_parquet.py $BIRD/convert_parquet.py
cp /path/to/LadderSQL/.apex_sl_temp/run_apex_sl.py $BIRD/run_apex_sl.py
ls -l $ROOT/chat.py $BIRD/convert_parquet.py $BIRD/run_apex_sl.py
