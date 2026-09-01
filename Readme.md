# LadderSQL — BIRD Test Submission Package

**Submission type: 4 — Combined Models** (a locally served open-source model + a closed-source
LLM called through our own API key).

This package contains everything needed to produce `predict_test.json` from
`test.json` + `test_tables.json` + `column_meaning.json` + `test_databases/`.

Contents

**Run these two, in this order:** `STAGE1_RUNBOOK.md` (stage 1, once) → `run_test.sh` (stages 2–4).

| Path | What it is |
|---|---|
| `Readme.md` | this file — run instructions |
| `requirements.txt` | Python dependencies |
| `run_test.sh` | **main entry point** for stages 2–4 (candidate generation → repair → selection → `predict_test.json`) |
| `STAGE1_RUNBOOK.md` | stage 1 step-by-step, including the one external dependency (APEX-SQL). Stage 1 has **no single driver script** — it is four scripts run in order, with paths set as constants at the top of each; every constant that must change is listed in the runbook |
| `MODEL_LINKS.txt` | checkpoint download links (see also §2 *Models*) |
| `predict_dev.json` | our predictions on the **original 1,534-question dev release** (EX 73.66) |
| `DEV_REFERENCE.md` | how the dev numbers were produced, with the exact commands (for verification) |
| `SQL_REPAIR_METHOD.md` | stage 3 design notes (checker chain, budget, scope) |
| `schema_construction/` | stage 1 code (schema linking driver, light schema, value retrieval, prompt rendering) |
| `generation/` + `training/original_agent`, `training/reward_func` | stage 2: multi-turn ReAct agent + candidate sampler |
| `sql_repair/` | stage 3: reward-blind SQL verification / repair |
| `inference_selection/` | stage 4: execution clustering + tournament + final selection/export |
| `tools/check_predict_format.py` | validates the submission file |
| `tools/testjson_to_parquet.py` | `test.json` -> the parquet the sampler reads (row order preserved) |
| `tools/score_predict_dev.py` | EX scorer, for reproducing the dev number when a gold file is available |
| `generation/_agl_stub.py` | 60-line stand-in for the training wrapper's base class, so **no RL training stack is needed** (see §2) |
| `stage1_apex_sql/` | where APEX-SQL goes — the one third-party dependency, used by stage 1 only (see `STAGE1_RUNBOOK.md` §1 and the README in that directory) |
| `make_code_zip.sh` | builds the trimmed `code.zip` we mail out |

Also in the repository but **not needed to reproduce this submission** — research / development
artifacts only. To reproduce `predict_test.json`, follow §3 and **ignore everything below**:

- multi-machine launchers and cost profiling: `run_gen_8gpu.sh`, `run_gen_dlc.sh`,
  `run_profile_140g.sh`, `PROFILE_GUIDE.md`, `sample_selection/scripts/`,
  `tools/profile_report.py`, `tools/merge_llm_usage.py`;
- Spider-benchmark scripts (a different benchmark, unrelated to this BIRD submission):
  `schema_construction/direct_linking/*spider*`, `inference_selection/scripts/run_spider_*.sh`;
- ablation / evaluation-analysis scripts we used during development:
  `inference_selection/{ablation_report,eval_bon_trend,eval_maj_8vs32,eval_multirep,eval_multirep_weighted,report_baseline}.py`,
  `inference_selection/scripts/{run_bon32_pipeline,run_bon_trend,run_tournament_all}.sh`,
  `schema_construction/eval/*`;
- dev/train cache-building and value-retrieval variants (this submission runs on test-set inputs, not on the dev/train caches):
  `schema_construction/cache_build/build_{clean_dev,old_dev,schema_cache_train,rendered_cache_train}*`,
  `schema_construction/value_retrieval/*{train,clean_dev,old_dev,gpu,parallel,rebuild}*`.

The `run_test.sh` driver only ever calls the specific scripts named in §3; none of the files
listed here are on that path, so they cannot affect reproduction.

---

## 1. What the system does

```
test.json + test_databases/
      │
      │  STAGE 1  question-aware schema construction         [GPU (embeddings) + closed LLM]
      ▼           → {"db_id|||question": "<schema text>"}
 schema cache
      │
      │  STAGE 2  32 candidate SQL per question               [GPU ONLY — our 14B via vLLM]
      ▼           multi-turn ReAct, the model executes its own SQL and revises
 candidate pool
      │
      │  STAGE 3  reward-blind SQL verification / repair       [closed LLM]
      ▼           fix candidates that error out / return empty
 repaired pool
      │
      │  STAGE 4  cluster by execution result, adjudicate      [closed LLM]
      ▼           top-3 clusters pairwise, pick with a consistency prior
 predict_test.json
```

**No gold SQL is used anywhere.** Stages 3 and 4 only look at: whether a query executes, the shape
of its result, regex anti-patterns, and pairwise LLM judgements. `test.json` has `"SQL": ""` and the
code never reads that field (stage 2 runs with `--no-gold`, stage 3 with `--no-gold`).

### GPU code vs closed-LLM code (as requested in the guidelines)

| | GPU (our own weights, no external network) | Closed LLM (our API key) |
|---|---|---|
| Stage 1 | sentence-transformers embedding of DB cell values | schema linking + column descriptions |
| Stage 2 | **all of it** — vLLM serving our 14B | none |
| Stage 3 | none | repair suggestions |
| Stage 4 | none | pairwise judge |

### What leaves the machine (data egress disclosure)

Only stages 1, 3, 4 call the external endpoint (`https://dashscope.aliyuncs.com/compatible-mode/v1`,
model `glm-5.2`). Each request contains: the question text, the evidence string, the linked schema
text (table/column names + descriptions), candidate SQL strings, and **at most 5 preview rows plus a
few example rows of the set-difference between two candidate results** (each cell truncated to
≤200 characters). No full table dumps, no file uploads, no other outbound hosts. The only network
client in the code is the `openai` Python SDK pointed at that one endpoint; grep for `base_url` to
verify.

---

## 2. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

- Python 3.11 or 3.12.
- **No RL training stack is required.** The agent used at inference is plain LangGraph + LangChain
  talking to the local vLLM server over the OpenAI API. Our training code shares one module with it
  (`training/original_agent/ReAct_Agent.py` contains both the inference `Agent` and the training-only
  `LitAgent`), so `generation/_agl_stub.py` provides a tiny stand-in for the training wrapper's base
  class. Consequence: **you do not need `agentlightning`, `verl`, `ray`, or `opentelemetry`.**
  The heaviest dependency in this package is vLLM, and only stage 2 uses it.
- **CUDA**: we developed on CUDA 12.8 / torch 2.8. On your CUDA 12.2/12.3 images please install a
  matching `torch` + `vllm` pair (e.g. `torch==2.4.0+cu121` with the corresponding vLLM release);
  everything else in `requirements.txt` is version-agnostic. Only stage 2 needs vLLM.
- No java/jdk or other non-Python runtimes are required.
- Credentials (copy `.env.example`, fill, and `export`):

```bash
export JUDGE_API_KEY=...       # GLM-5.2 key (stages 3, 4)  — supplied separately by e-mail
export JUDGE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
export DASHSCOPE_API_KEY=$JUDGE_API_KEY    # stage 1 uses the same key
export ALICLOUD_API_KEY=$JUDGE_API_KEY
```

### Models

| Checkpoint | Download | Size | Bird-dev EX |
|---|---|---|---|
| `LadderSQL-14B` | <https://www.modelscope.cn/models/kyk666/LadderSQL-14B-BIRD> | 28 GB, bf16, 7 shards | **73.66** |

One checkpoint, used for every question — there is no model ensemble.
The repository is **public — no token or account is needed**. Download it to a **local** disk:

```bash
pip install modelscope
modelscope download --model kyk666/LadderSQL-14B-BIRD \
                    --local_dir /local/models/LadderSQL-14B
```

Then point `MODEL_DIR` at that directory. The repository holds the checkpoint exactly as it was
used for the 73.66 result (`Qwen2ForCausalLM`, bf16, 48 layers, hidden 5120, 40 heads / 8 KV heads).

> ⚠️ **transformers version.** `config.json` declares the dtype as `"dtype": "bfloat16"`, the field
> name used by **transformers ≥ 4.56** (we exported with 4.57.1 / vLLM 0.11.0). Older transformers
> looks for `torch_dtype`, finds nothing, and silently loads in float32 — 28 GB becomes ~56 GB and
> will OOM. **You do not need to do anything about this for the test run:** our launcher passes
> `--dtype bfloat16` explicitly (`sample_selection/scripts/start_vllm_redundant.sh`, overridable
> with `VLLM_DTYPE`), so the served dtype is independent of the transformers version. It only
> matters if you load the checkpoint with `AutoModel...` yourself, in which case either use
> transformers ≥ 4.56 or add `"torch_dtype": "bfloat16"` to `config.json`.

> ⚠️ **SQLite databases and model weights must be on a local disk.** On a network / object-storage
> mount SQLite raises intermittent `disk I/O error`s, which silently corrupts the execution
> signatures that stage 4 clusters on.

---

## 3. Running it

### Stage 1 — schema construction (must run before stage 2)

See **`STAGE1_RUNBOOK.md`**. It has one external dependency (the APEX-SQL project, used as our
schema linker for Bird) and a few paths to set. Output is a single JSON:

```json
{"california_schools|||How many schools ...": "## Table: schools\n- CDSCode: ...", ...}
```

The key format must be exactly `f"{db_id}|||{question}"` with the question verbatim from
`test.json`; a mismatch silently degrades the prompt to raw DDL.

### Stages 2–4 — one command

```bash
export JUDGE_API_KEY=... JUDGE_API_BASE=...
bash run_test.sh
```

**Staging weights/databases to local disk (recommended on a single DLC card).** Running SQLite and
loading weights directly off a slow object-storage / NAS mount is slow and causes intermittent I/O
errors. Set `MODEL_SRC` (the OSS HF-weight dir) and `BIRD_ZIP` (the `dev_databases.zip`) and stage 0
will `cp` the 14B weights and `unzip` the databases to local disk first (idempotent — skipped if
already present), then point `MODEL_DIR` / `TEST_DB_DIR` at the local copies:

```bash
export JUDGE_API_KEY=... JUDGE_API_BASE=...
MODEL_SRC=/oss/.../global_step_192/actor/huggingface \
LOCAL_MODEL_DIR=/root/local_models/LadderSQL-14B \
BIRD_ZIP=/oss/.../bird/dev_20240627/dev_databases.zip \
LOCAL_DB_ROOT=/root/local_bird_db DB_SUBDIR=dev_databases DB_MIN_COUNT=10 \
TEST_JSON=/oss/.../bird/dev_20240627/dev.json \
SCHEMA_CACHE=/root/schema_cache_dev_rendered.json \
NUM_GPUS=1 MAX_NUM_SEQS=48 PER_GPU=48 \
  bash run_test.sh
```

Without `MODEL_SRC`/`BIRD_ZIP`, stage 0 is skipped and the script expects `MODEL_DIR` and
`TEST_DB_DIR` to already be local (this is the mode the BIRD Exp Team uses).

`run_test.sh` reads these variables (all have defaults documented inside):

| Variable | Meaning |
|---|---|
| `TEST_JSON` | BIRD `test.json` (fields used: `db_id`, `question`, `evidence`) |
| `TEST_DB_DIR` | `test_databases/` on local disk, laid out `<db_id>/<db_id>.sqlite` |
| `SCHEMA_CACHE` | stage-1 output |
| `MODEL_DIR` | local directory of the downloaded checkpoint |
| `NUM_GPUS` | number of GPUs (one vLLM instance per GPU) |
| `N_TRIALS` | candidates per question — **must stay 32** to match our reported setting |
| `WORK_DIR` | all intermediate artifacts and logs go here |
| `OUT_PREDICT` | final file, default `predict_test.json` |

Stages can also be run one at a time — `run_test.sh` prints the exact command for each stage before
executing it, so you can copy any single stage out of the log.

### Validate the output

```bash
python tools/check_predict_format.py predict_test.json --expect-n <number of test questions> \
       --db-dir "$TEST_DB_DIR" --questions "$TEST_JSON"
```

---

## 4. Resources and expected runtime

> **Provenance of these numbers.** Stages 3 and 4 were timed end-to-end on dev (real wall clock).
> Stage 2's figure is **derived**: 56.2 single-stream GPU-hours of LLM compute (from the per-trial
> `llm_time` we logged during sampling) divided across 8 GPUs; we did **not** capture a single
> from-scratch wall-clock of the 8-GPU run, so treat 1-2 h as an estimate. Stage 1 is order-of-magnitude.
Measured on the 1,534-question dev release, 8×80GB GPUs, 32 candidates per question:

| Stage | Hardware | Wall clock (1,534 questions) |
|---|---|---|
| 1 schema construction | CPU + 1 GPU for embeddings | hours (dominated by cell-value indexing) |
| 2 candidate generation | 8 GPUs, one vLLM each, `max_num_seqs=16`, 80 concurrent rollouts | **1–2 h** (pure LLM compute: 56.2 single-stream GPU-hours) |
| 3 SQL repair | CPU + external API, 32 concurrent | ~25 min |
| 4 judge + selection | CPU + external API, 24 concurrent | ~35 min |

**Recommended allocation: 8 × A100 80G for stage 2** (only stage 2 needs GPUs; a 14B bf16 model
needs ~28 GB, so one card per instance is comfortable). Fewer cards work with proportionally longer
runtime — a single A100 80G is estimated at 8–16 h for stage 2 (estimated, not measured).
Disk: ~60 GB for weights + ~40 GB for intermediates (candidate pool with trajectories is large;
set `SAVE_TRAJECTORY=0` in `run_test.sh` to cut it by ~10×).

## 5. External-LLM token usage (for your cost estimate)

> Values below are our **best current estimates**, computed on dev either from the API's `usage`
> field (generation) or from prompt-length accounting (repair / judge, chars/3.6). The pipeline now
> also records the **exact** `usage.prompt_tokens / completion_tokens` of every closed-LLM call into
> `${WORK_DIR}/logs/llm_usage.json` (see `inference_selection/llm_usage.py`); after the first real
> run on your side that file holds the authoritative numbers.

Measured / computed on dev (1,534 questions). Per-question figures scale linearly with test size.

| Stage | Calls | Input tokens | Output tokens | Input per question |
|---|---|---|---|---|
| 3 repair | ≈ 5,244 | ≈ 6.2 M | ≤ 3.7 M | ≈ 4.1 k |
| 4 judge | 14,952 | ≈ 38.1 M | ≤ 10.5 M | ≈ 24.8 k |
| **total (closed LLM)** | ≈ 20.2 k | **≈ 44.3 M** | **≤ 14.2 M** | **≈ 28.9 k** |

Stage 1 adds one linking call plus a few description calls per database (small relative to the above).
Stage 2 runs entirely on the local GPU model: 203.5 M input / 15.0 M output tokens on dev, no cost.

Call volume depends on how much the 32 candidates disagree: on dev, **57% of questions had all 32
candidates in a single execution-equivalence class and needed zero judge calls**.

## 6. Logging, error handling and resuming

Every stage writes progress to stdout (redirect into `$WORK_DIR/logs/`) and keeps a resumable
artifact, so **an interrupted or failed run never has to start over** — re-issuing the same command
continues from where it stopped:

| Stage | Resume mechanism |
|---|---|
| 2 generation | output JSON is flushed every 16 questions; finished questions are skipped, questions marked `timed_out` are retried |
| 2b execution cache | keyed by `(db_id, normalised SQL)`; rerun re-executes only missing keys |
| 3 repair | `--ckpt` file flushed every 200 repaired writings (`<out-file>.repair_ckpt.json`) |
| 4 judge | verdict cache flushed every 1,000 calls; keyed per `(question, cluster pair, phrasing, order)` |
| 4 selection | pure cache read, seconds |

Per-item failures are contained: a rollout that times out or throws is dropped and resampled
(hard timeout 120 s per trajectory, 45 s per LLM request); a repair that throws leaves the original
SQL untouched; a judge call that fails is retried with backoff and finally recorded as a tie.
Nothing aborts the whole run.

## 7. `column_meaning.json`

**Yes — please provide it.** `schema_construction/cache_build/build_light_schema.py` reads
`column_meaning.json` (BIRD's summarised column descriptions, TA-SQL format) when building the
per-database light schema, and per-database `database_description/*.csv` files when they are present.
Set `COLUMN_MEANING_PATH` at the top of that file to the file you provide.

## 8. Known limitations / please contact us if these bite

1. **Stage 1 needs APEX-SQL** (<https://github.com/Tencent/APEX-SQL-Project>), our Bird schema
   linker. The mailed package includes the five files stage 1 imports under `stage1_apex_sql/`, so
   nothing needs fetching; see `STAGE1_RUNBOOK.md` §1. If you would rather not use it at all, tell us
   and we will send a self-contained single-call linking variant instead (this changes the schema and
   therefore the accuracy slightly).
2. **Giant test databases** may hit the 15 s SQL execution timeouts used in stages 2–4. If you see
   many timeouts in the logs, raise `EXEC_TIMEOUT` / `SQL_TIMEOUT` (documented in `run_test.sh`)
   to 30–60 s and rerun — thanks to the caches only the affected queries are recomputed.
3. `N_TRIALS` and the selection hyper-parameters (`K=3`, `RMAX=3`, `k=1.02`, `variant=biased`) were
   tuned on dev and are **hard-coded for test**; please do not change them, or the numbers will not
   correspond to our reported dev result.

## 9. Reference dev result

`predict_dev.json` in this package was produced by exactly this pipeline on the original
1,534-question dev release: **EX = 1130/1534 = 73.66%** (majority voting over the same pool: 72.36%;
pass@32 ceiling: 81.36%; greedy single-sample: 68.84%). `DEV_REFERENCE.md` lists the commands.

## 10. Changing the checkpoint or the hyper-parameters

> **For the test run: skip this section.** Everything below is already at the value that produced
> 73.66; §8.3 asks you not to change it. This section exists so that the settings are documented in
> one place for our own follow-up work.

### Swapping in a different checkpoint

Only `MODEL_DIR` changes — nothing else in the pipeline is tied to a particular checkpoint:

```bash
MODEL_DIR=/local/models/<other-checkpoint> bash run_test.sh
```

Keep `SERVED_MODEL_NAME=qwen2.5-coder` as it is: it is the name vLLM advertises and the name the
sampler asks for. If the two stop matching, every request comes back **404** and the run produces
nothing. When the new weights come out of RL training as FSDP shards, merge them to HuggingFace
format first (`python -m verl.model_merger merge --backend fsdp --local_dir <ckpt>/actor
--target_dir <ckpt>/actor/huggingface`) and point `MODEL_DIR` at the `huggingface` directory.

### Every knob, where it lives, and what it costs

Stage 2 — candidate generation (`run_test.sh` §knobs, all overridable by environment variable):

| Variable | Default | Note |
|---|---|---|
| `N_TRIALS` | 32 | candidates per question. Fewer is cheaper and worse; the reported number needs 32 |
| `TEMPERATURE` | 1.1 | tuned for pool diversity, not for single-sample quality |
| `MAX_TURNS_AGENT` | 5 | ReAct turns before the agent must commit |
| `MAX_TOKENS` / `MAX_MODEL_LEN` | 3072 / 24576 | per response / context window |
| `NUM_GPUS` / `MAX_NUM_SEQS` / `PER_GPU` | 1 / 48 / 48 | throughput only, no effect on accuracy. 8 GPUs: `NUM_GPUS=8 MAX_NUM_SEQS=16 PER_GPU=10` |
| `VLLM_DTYPE` | `bfloat16` | weight dtype passed to `vllm serve`. Leave it — `auto` is what causes the fp32/OOM trap described in §2 |
| `SERVED_MODEL_NAME` | `qwen2.5-coder` | must match on both sides (see above) |
| `SAVE_TRAJECTORY` | 0 | 1 keeps full ReAct transcripts, ~10x larger output |
| `LLM_TIMEOUT` / `TRAJ_TIMEOUT` | 45 s / 120 s | per request / per rollout hard kill |

SQL execution, stages 2–4:

| Variable | Default | Note |
|---|---|---|
| `SQL_TIMEOUT` / `EXEC_TIMEOUT` | 15 s / 15 s | raise to 30–60 for very large databases; caches mean only affected queries recompute |

Stage 4 — selection (`run_test.sh` line ~108, plain shell variables, **not** environment-overridable):

| Variable | Default | Meaning |
|---|---|---|
| `VARIANT` | `biased` | how a pairwise verdict becomes a probability. `biased` uses one presentation order; the only other accepted value is `debiased`, which also asks in the swapped order and averages the two (2x the judge calls). We did not measure `debiased` at the final setting, so it has no number here |
| `TOPK` | 3 | how many execution-equivalence clusters enter the tournament |
| `RMAX` | 3 | phrasings per cluster. 1 scores 73.08 on dev instead of 73.66 |
| `KPRIOR` | 1.02 | weight of the cluster-size prior. 0 scores 72.82 on dev — the prior has to stay |
| `MULTIREP_WORKERS` | 32 | concurrency only |

Stage 1 retrieval settings that also affect accuracy are in `STAGE1_RUNBOOK.md` §3 (top-5 cells per
keyword, L2 threshold 0.8, ≤ 8 keywords per question). Dropping value retrieval entirely costs about
3 EX points.

### Running stage 1's schema linking from `run_test.sh`

The linking step of stage 1 can be driven by the same entry point instead of by hand:

```bash
RUN_SCHEMA_LINK=1 APEX_DIR=/path/to/APEX-SQL-Project/BIRD bash run_test.sh
```

The remaining three stage-1 steps (light schema, value retrieval, prompt rendering) are still the
manual sequence in `STAGE1_RUNBOOK.md`.

### Splitting GPU work and API work across machines

`STAGES=gen` runs only what needs a GPU; `STAGES=api` runs only what needs the external endpoint.
Point both at the same `WORK_DIR` (shared storage) and the second picks up where the first left off.

## 11. License

Our code: MIT (`LICENSE`). Third-party components: `NOTICE`.
This inference package bundles no third-party runtime code (the vendored training framework was
removed because inference does not need it); `training/reward_func/spider_eval/` is derived from the
official Spider evaluation scripts and is used only when a gold SQL is available (never on test).
