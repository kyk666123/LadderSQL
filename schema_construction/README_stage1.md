# Stage 1 — Question-aware schema construction

This stage turns each `(question, database)` pair into a compact, value-grounded
schema `S+_q` consisting of (i) the linked subset of tables and columns, (ii)
natural-language descriptions of the retained elements, and (iii) database cell
values that match literals in the question.

Two linking backends are used, depending on the benchmark:

| Benchmark | Linker | Why |
|---|---|---|
| Spider | one LLM call over the full schema (`direct_linking/`) | databases are small and clean |
| Bird | reproduction of APEX-SQL (`apex_sql_runner/`) | databases are large and noisy |

The design target is **high recall with moderate redundancy**: a recall failure
removes a required table or column and is unrecoverable downstream, so the agent
could never earn positive reward; but a schema with perfect recall and no
redundancy trains a model that degrades once it is fed the redundant schemas a
linker actually produces at inference time. Reported quality is in paper Table 6
(Bird-dev: 99.5% table recall / 98.3% column recall, 16.2 columns per question).

## Bird: reproducing APEX-SQL

APEX-SQL is a separate project. Its source is obtained from its own authors and
is **not** redistributed here; this repository only ships the driver we used.

### What the driver expects

- The APEX-SQL project directory, whose `BIRD/schema_linking.py` exposes
  `SchemaLinking.run_schema_linking(question, question_id, evidence,
  sqlite_path, database_root_dir, enable_pruning=True)`.
- `chat.py` placed at the APEX-SQL project root — `schema_linking.py` does
  `from chat import GPTChat`. Use the copy in this directory.
- An empty `BIRD/data/descriptions/` directory must exist, otherwise
  `load_column_meaning` fails on `os.listdir`.
- Bird databases unpacked on a **local** disk, laid out as
  `<db_root>/<db_id>/<db_id>.sqlite`, each keeping its `database_description/`
  subdirectory (that is where column descriptions come from). The exploration
  step executes real SQL, so an object-storage mount is unusably slow.
- `pip install sqlglot openai json_repair func_timeout sqlparse pandas tqdm pyarrow`
  — `sqlglot` is imported by APEX-SQL's `utils.py` and is usually missing.

### Credentials

`chat.py` reads the endpoint from the environment; nothing is hardcoded.

```bash
export OPENAI_API_KEY="<your key>"
export OPENAI_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1/"
export GLM_MODEL="glm-5.2"
```

The model id must be exactly `glm-5.2`. GLM-5.2 is a thinking model and the
thinking channel **must** be disabled (`extra_body={"enable_thinking": False}`,
already set in `chat.py`); otherwise `content` comes back empty because the
budget is consumed by the reasoning trace.

Smoke test:

```bash
python -c "from chat import GPTChat; c=GPTChat(); c.init_messages(); print(repr(c.get_model_response_txt('reply with only: OK')))"
```

### Running

`train.parquet` has no `question_id`, so the converter falls back to the row
index:

```bash
python convert_parquet.py \
  --parquet /path/to/nl2sql_dataset/bird/bird_clean_data/train.parquet \
  --out     <work_dir>/train_data.json        # 6601 records
```

```bash
python run_apex_sl.py \
  --data_json <work_dir>/train_data.json \
  --db_root   <local_db_root>/train_databases \
  --work_dir  <work_dir> \
  --out_dir   <results_dir> \
  --model     glm-5.2 \
  --workers   32
```

Add `--limit 3` for a first validation run; those three results are a subset of
the final output and are skipped on the full run.

The runner is built for a flaky, rate-limited endpoint:

- Every question is written to `<work_dir>/parts/<qid>.json` as soon as it
  finishes, so a crash loses nothing.
- Each round runs the whole pending set with 32 workers; failures roll into the
  next round. Rate-limit failures (tagged `##RATELIMIT##` by `chat.py`) are
  retried indefinitely with backoff, because they always drain eventually.
- It stops only after 5 consecutive rounds with no progress *and* no rate
  limiting — that indicates a hard error (usually a missing or corrupt sqlite
  file), recorded in `failures.json`.
- Rerunning the same command resumes from the local parts plus any previous
  results file.

Output: `results_glm-5.2.json`, keyed by question id, each value holding
`result.refined_schema` and `result.table_judgments`. The run is complete when
the key count equals the input size, or the remainder in `failures.json` are all
confirmed hard errors.

## Value grounding

Question literals often differ in surface form from stored values (`"Fresno"`
vs. `"Fresno County Office of Education"`). All textual cells are indexed with
`all-MiniLM-L6-v2` in ChromaDB; for each keyword extracted from the question the
top-5 most similar cells are retrieved. A retrieved value is attached to its
column only if the linker selected that column — otherwise it is discarded.

```bash
cd ../value_retrieval
python build_chroma_train.py           # build the index
python extract_keywords_train.py       # literals per question
python train_value_retrieval.py        # top-5 cells per keyword
```

`database_cell_process.py` handles cell normalisation; `retrieve.py` is the
retrieval class also imported at agent runtime (`AGL_RETRIEVE_DIR` points the
agent at it).

## Merging and rendering

```bash
cd ../cache_build
python build_column_profiles.py        # types + sample values + descriptions
python build_schema_cache_train.py     # structured candidate schema per question
python build_rendered_cache_train.py   # -> {"db_id|||question": "<schema text>"}
```

`render_schema_prompt.py` produces the DDL-style text that is substituted into
the prompt's `{table_info}` slot: one line per column with type, description and
sample values, with retrieved cell values grouped under the keyword that
triggered them, and an explicit note that the candidate schema is recall-oriented
and contains redundancy the model must resolve itself.

The cache key must be exactly `f"{db_id}|||{question}"`. Training and inference
consume the rendered file via `--schema-truncate cache --schema-cache-path`;
on a key miss the agent silently falls back to the full DDL, which weakens
results without raising an error.

## Evaluating linking quality

```bash
cd ../eval
python eval_srr_v14.py            # SRR / recall / precision / F1, table and column level
python eval_schema_linking_v2.py
```

SRR (strict recall rate) is a per-question 0/1 indicator, 1 only if the selected
schema fully contains every element required by the gold SQL.
