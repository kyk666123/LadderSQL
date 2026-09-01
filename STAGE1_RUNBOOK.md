# Stage 1 Runbook — building the schema cache for a new split (test)

Stage 1 turns `test.json` + `test_databases/` + `column_meaning.json` into one file:

```json
{ "<db_id>|||<question verbatim>": "<schema text fed to the model>", ... }
```

That file is the `SCHEMA_CACHE` input of `run_test.sh`. Everything below runs once, before stage 2.

> **Honest status.** These scripts were written for our `train` / `dev` splits and carry
> **module-level path constants** rather than CLI flags. Running them on `test` means editing the
> constants listed in the tables below — there is no `--split test` switch. We list every constant
> that must change. If you prefer, send us the test inputs' *directory layout* and we will ship a
> patched copy with the paths pre-filled.

## Prerequisites

| Item | Note |
|---|---|
| `test_databases/` on **local** disk, as `<db_id>/<db_id>.sqlite`, keeping each `database_description/` subdir | column descriptions are read from those CSVs |
| `test_tables.json`, `test.json`, `column_meaning.json` | as provided by BIRD |
| Sentence-embedding model `all-MiniLM-L6-v2` | local directory; used for cell-value retrieval (1 GPU, or CPU if slow is acceptable) |
| `JUDGE_API_KEY` / `DASHSCOPE_API_KEY` / `ALICLOUD_API_KEY` exported | stage 1 calls GLM-5.2 |
| **APEX-SQL project source** | our Bird schema linker — see §1 |

## 1. Schema linking (APEX-SQL) — the one external dependency

We reproduce **APEX-SQL** as the Bird linker
(<https://github.com/Tencent/APEX-SQL-Project>, paper <https://arxiv.org/abs/2602.16720>).
Its source belongs to its own authors; we ship the driver, `schema_construction/apex_sql_runner/`,
plus — **in the mailed package only** — the five APEX-SQL files that stage 1 imports, under
`stage1_apex_sql/APEX-SQL-Project/`. See the README in that directory.

If the directory is already populated (it is, in the package we mailed you), go straight to:

```bash
cd schema_construction/apex_sql_runner
python run_apex_sl.py \
    --data_json  /path/to/test.json \
    --db_root    /local/test_databases \
    --work_dir   /work/apex_test
# -> /work/apex_test/results_glm-5.2.json     (one linked-schema record per question)
```

The driver expects, inside the APEX-SQL project: `BIRD/schema_linking.py` exposing
`SchemaLinking.run_schema_linking(...)`, our `chat.py` at the project root, and an existing
(possibly empty) `BIRD/data/descriptions/` directory. It is 32-way concurrent, resumable, and
retries on rate limits. Details: `schema_construction/README_stage1.md`.

**If you would rather not install APEX-SQL, tell us** — we can send a single-LLM-call linking
variant that needs no external project (it changes the schema, so accuracy shifts slightly).

## 2. Per-database light schema (uses `column_meaning.json`)

```bash
cd schema_construction/cache_build
python build_light_schema.py --type dev --workers 24
```

| Constant (top of `build_light_schema.py`) | Set it to |
|---|---|
| `BASE_DATA_PATH` | directory containing `test_tables.json` |
| `DB_FOLDER_OVERRIDE` | `/local/test_databases` |
| `COLUMN_MEANING_PATH` | the `column_meaning.json` you provide |

Output is a `{db_id: "<light schema markdown>"}` JSON; note the exact output path the script prints —
you need it as `LIGHT_PATH` in §4. (`--type dev` is the right branch for a dev-like layout; the
`test` naming is not special-cased, only the output filename differs.)

## 3. Cell-value retrieval

```bash
cd schema_construction/value_retrieval
python build_chroma_parallel.py       # 1) index textual cells of every test DB (ChromaDB)
python extract_keywords_old_dev.py    # 2) extract value literals from each question (GLM-5.2)
python value_retrieval_old_dev.py     # 3) top-5 matching cells per keyword
```

| Script | Constants to set |
|---|---|
| `build_chroma_parallel.py` | database root, output Chroma dir, embedding model path |
| `extract_keywords_old_dev.py` | `APEX_PATH` = §1 output, `OUTPUT_PATH` / `OSS_OUTPUT_PATH` = where to write keywords |
| `value_retrieval_old_dev.py` | `LOCAL_CHROMA`, `EMBEDDING_MODEL_PATH`, `KEYWORDS_PATH` (= previous output), `OUTPUT_PATH` |

Retrieval settings that must stay as-is (they match the reported result): top-`K = 5` per keyword,
L2 distance threshold `0.8`, at most 8 keywords per question. Steps 2 and 3 checkpoint every
100/200 questions and resume.

Skipping value retrieval is possible (`VR_PATH` missing ⇒ empty `linked_values`) but **costs about
3 EX points** on dev, so please don't.

## 4. Merge + render the prompt text

```bash
cd schema_construction/cache_build
python build_old_dev_redundant_cache.py
```

| Constant | Set it to |
|---|---|
| `BASE` | any working directory prefix |
| `APEX_PATH` | §1 output (`results_glm-5.2.json`) |
| `LIGHT_PATH` | §2 output |
| `VR_PATH` | §3 output |
| `OUT_STRUCT` | structured candidate tree (intermediate) |
| `OUT_RENDERED` | **the file to pass as `SCHEMA_CACHE`** |
| `TRAIN_RENDERED`, `OUT_MERGED` | only used by the final "merge with the training cache" step, which is irrelevant here. If `TRAIN_RENDERED` does not exist the script raises at that last step — **that is harmless: `OUT_RENDERED` has already been written** (check the `[2/3] rendered cache -> ...` line in the log). |

## 5. Sanity checks before stage 2

```bash
python - <<'EOF'
import json
c = json.load(open("<OUT_RENDERED>")); t = json.load(open("/path/to/test.json"))
keys = {f'{r["db_id"]}|||{r["question"]}' for r in t}
missing = [k for k in keys if k not in c]
print("cache entries:", len(c), " test questions:", len(t), " missing keys:", len(missing))
print("avg schema chars:", sum(len(v) for v in c.values())//max(len(c),1))
print(missing[:3])
EOF
```

- `missing keys` must be **0**. A mismatch (extra whitespace, unicode normalisation, a trimmed
  question) is silent at run time: the agent falls back to the raw DDL and accuracy drops.
- Average schema size on dev was ~3,100 characters. If test comes out far larger (giant databases),
  expect longer prompts; `MAX_MODEL_LEN=24576` still has headroom, and the agent truncates
  observations at 2,048 characters.
