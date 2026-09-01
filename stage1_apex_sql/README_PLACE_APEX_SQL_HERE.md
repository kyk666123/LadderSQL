# Stage 1's external dependency: APEX-SQL

Stage 1 (schema linking for Bird) reproduces **APEX-SQL**:

- Upstream repository: <https://github.com/Tencent/APEX-SQL-Project>
- Paper: *APEX-SQL: Talking to the data via Agentic Exploration for Text-to-SQL*,
  <https://arxiv.org/abs/2602.16720>

Copyright belongs to its authors, and the copy we obtained carries **no LICENSE file**, so this
public repository does **not** redistribute it — only our driver
(`../schema_construction/apex_sql_runner/`) is here.

> **BIRD Exp Team:** the package mailed to you already contains
> `APEX-SQL-Project/` with the five files stage 1 imports, so you do not have to fetch anything.
> If it is missing, follow the layout below.

## Expected layout

```
stage1_apex_sql/
└── APEX-SQL-Project/
    ├── chat.py                     ← copy of ../schema_construction/apex_sql_runner/chat.py
    └── BIRD/
        ├── schema_linking.py       ← must expose SchemaLinking.run_schema_linking(...)
        ├── preprocess.py           ← provides generate_table_info(...)
        ├── utils.py
        ├── sql.py
        ├── chat.py                 ← same env-driven copy as above
        └── data/descriptions/      ← must exist (may be empty), otherwise load_column_meaning fails
```

Only schema linking is used. APEX-SQL's own SQL generation, reward model and evaluation code
(`agent.py`, `run.py`, `reward.py`, `preprocess_macro_plans.py`, `preprocess_select_tips.py`,
`eval*.py`) plays no part in our pipeline and is not shipped. Its `BIRD/data/*.json` artifacts
(~34 MB of precomputed dev results) are not needed either.

**Replace `chat.py` with ours.** The upstream copy may carry a hardcoded API key; ours reads
`OPENAI_API_KEY` / `OPENAI_API_BASE` / `GLM_MODEL` from the environment and adds the rate-limit
handling the driver expects.

Extra Python packages APEX-SQL needs on top of `requirements.txt`:

```bash
pip install sqlglot json_repair func_timeout sqlparse
```

Then run stage 1 as described in `../STAGE1_RUNBOOK.md` §1.

## If installing APEX-SQL is not acceptable

Contact us. We can supply a single-LLM-call schema-linking variant
(`schema_construction/direct_linking/`) that needs no external project. It produces a different
(slightly less redundant) schema, so the accuracy will not exactly match our reported dev number.
