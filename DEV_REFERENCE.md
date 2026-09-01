# Dev reference — how `predict_dev.json` was produced

`predict_dev.json` in this package is the output of **exactly** the pipeline in `run_test.sh`,
run on the original Bird dev release (1,534 questions, `dev_20240627`).

## Numbers

| Setting | EX (%) | Note |
|---|---|---|
| greedy, 1 sample, T=0 | 68.84 | single trajectory |
| majority voting over 32 candidates | 72.36 | largest execution-equivalence class |
| **LadderSQL, best-of-32 (this package)** | **73.66** | 1130 / 1534 |
| pass@32 ceiling of the same pool | 81.36 | at least one of 32 is correct |

Effect of the repair stage (same pool, before → after): majority 71.58 → 72.36,
pass@32 80.77 → 81.36, final 73.40 → 73.66.

Output quality of the final selection (relevant to the ">5% abnormal outputs" rule):
of the 1,534 selected queries, **3 fail to execute (0.20%) and 5 return an empty result (0.33%)**
— 0.52% in total.

## Commands

Stage 1 produced `schema_cache_old_dev_rendered.json` (1,533 entries, avg 3,067 characters)
following `STAGE1_RUNBOOK.md` with the dev inputs. Then:

```bash
export JUDGE_API_KEY=... JUDGE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
export BON_DATASET=bird BON_DB_DIR=/local/bird_dev_databases

# stage 2: 32 candidates per question (8 GPUs, one vLLM each)
cd generation && python sample_bird_candidates.py \
    --parquet dev.parquet --db-dir /local/bird_dev_databases \
    --output pool.json --n-trials 32 --num-gpus 8 --per-gpu 10 --traj-timeout 120

# stage 2c: execution cache
cd ../inference_selection && BIRD_FILE=pool.json EXEC_CACHE=ec.json \
    python build_exec_cache_generic.py --workers 32

# stage 3: reward-blind repair
cd ../sql_repair && BIRD_FILE=pool.json EXEC_CACHE=ec.json \
    python repair_candidates_generic.py --out-file pool_repaired.json --out-cache ecr.json \
      --budget 6 --chain default --workers 32

# stage 4a: pairwise adjudication
cd ../inference_selection && BIRD_FILE=pool_repaired.json EXEC_CACHE=ecr.json \
    MULTIREP_CACHE=verdicts.json MULTIREP_RMAX=3 SCHEMA_CACHE=<stage1 output> \
    python run_multirep_tournament.py

# stage 4b: selection + export (hyper-parameters below are the ones used for test too)
BIRD_FILE=pool_repaired.json EXEC_CACHE=ecr.json \
    python select_and_export.py --multirep-cache verdicts.json \
      --variant biased --topk 3 --k 1.02 --rmax 3 --n 32 \
      --out-predict predict_dev.json --gold-check
# [gold-check] EX = 1130/1534 = 73.66%
```

On dev we used `run_multirep_tournament.py`, which only adjudicates questions whose 32 candidates
disagree **and** where gold shows the disagreement matters — a cost saving that is only possible
offline. For test, `run_test.sh` uses `tournament_blind.py`, which adjudicates **every** question
with ≥2 execution-equivalence classes. We verified on dev that the two give the *identical* result
(73.66%, same 1130 questions); the blind version simply makes more calls
(14,952 vs 9,950; 659 vs 423 questions adjudicated).

## Fixed hyper-parameters (tuned on dev, unchanged for test)

| Parameter | Value | Meaning |
|---|---|---|
| `n` | 32 | candidates per question |
| temperature | 1.1 | sampling temperature (matches training) |
| max ReAct turns | 5 | each turn executes one SQL and observes the result |
| repair chain | `syntax → time → select → result`, budget 6 | reward-blind checker chain |
| `TOPK` (K) | 3 | clusters entering the tournament |
| `RMAX` (R) | 3 | phrasings per cluster (noise reduction) |
| aggregation variant | `biased` | uses one presentation order |
| `k` | 1.02 | weight of the consistency prior in `score_i = win_rate_i + k·n_i/32` |
| judge | GLM-5.2, temperature 0, thinking disabled, ≤700 output tokens | pairwise verdict `FINAL: A/B` |

Sensitivity of `k` on dev (biased, K=3, R=3): 0.0 → 72.82, 0.6 → 73.27, 0.8 → 73.40,
**1.0 → 73.60**, 1.02 → 73.66, 1.2 → 73.40, 1.5 → 72.88, 2.0 → 72.88. The optimum is a plateau,
not a spike; `k = 1.0` is equally defensible.
