#!/usr/bin/env python3
"""Generate schema-linking cache for Spider dev.parquet (1034 questions).

For every (db_id, question) in dev.parquet, feed the *complete* light schema of
the database to the LLM (glm-5.2) and let it output the relevant schema.

The light schema already annotates:
  - `### Primary keys`  (PK)
  - `### Foreign keys`  (explicit FK connections, e.g. "a.x = b.y")
so NO extra FK/join injection and NO disambiguation is needed — the schema text
alone is enough.

Output format (flat, matches original_agent/ReAct_Agent.py truncate="cache"):

    { "<db_id>|||<question>": "<raw schema string>" }

Usage:
    export ALICLOUD_API_KEY=...            # required
    export ALICLOUD_BASE_URL=...           # optional, defaults to dashscope
    python generate_spider_dev_schema_cache.py                 # full dev.parquet
    python generate_spider_dev_schema_cache.py --limit 20      # smoke test
    python generate_spider_dev_schema_cache.py --workers 16    # more concurrency
    python generate_spider_dev_schema_cache.py --model glm-5.2 # override model
"""

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from openai import OpenAI

# ── 路径解析（脚本位于 <repo>/schema_preprocessing/relevant_schema_cache/）───────
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]  # /path/to/LadderSQL

LIGHT_SCHEMA_TEST = REPO_ROOT / "schema_preprocessing/db_light_schema/spider_test_light_schema.json"
LIGHT_SCHEMA_TRAIN = REPO_ROOT / "schema_preprocessing/db_light_schema/spider_train_light_schema.json"
# Spider-DK 的 3 个改造库（new_concert_singer / new_orchestra / new_pets_1）
LIGHT_SCHEMA_DK_NEW = REPO_ROOT / "schema_preprocessing/db_light_schema/spider_dk_new_light_schema.json"

DEFAULT_DEV_PARQUET = "/path/to/nl2sql_dataset/spider/dev.parquet"
DEFAULT_OUTPUT = HERE / "spider_glm52_dev_schema_cache.json"


# ── LLM 配置 ─────────────────────────────────────────────────────────────────
MODEL = os.environ.get("MODEL_NAME", "glm-5.2")
client = OpenAI(
    api_key=os.environ.get("ALICLOUD_API_KEY"),
    base_url=os.environ.get("ALICLOUD_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    timeout=120.0,
)

# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a precise schema grounding agent for NL2SQL. Analyze the user's question and database schema, then extract ALL relevant tables, columns and grounded values.

Instructions:
1. Identify entities, attributes, conditions, and aggregations from the question.
2. Map each to tables/columns. A table is relevant if any column is needed for SELECT, FROM/JOIN, WHERE, GROUP BY, ORDER BY, or HAVING.
3. For each relevant table, include ALL necessary columns:
   - Columns for SELECT output and WHERE/HAVING conditions
   - PK/FK columns required for JOIN connectivity
   - Columns for COUNT/SUM/AVG aggregation targets (even FK columns used for counting)
   - Columns whose Description matches question keywords (check descriptions carefully)
4. CRITICAL - Join Paths: Use the `### Foreign keys` section of each table to connect entities. When two entities need connecting, ALWAYS include every intermediate/bridge table on the FK path. Never skip a bridge table.
5. Ground values: Match question keywords to sample values (case-insensitive). Do not invent values.
6. Final check: Ensure all selected tables are connected via FK paths. If isolated, find the bridge table.

Output ONLY the <schema> tag. No other text.

<schema>
Table:
- <table_name>
Columns:
- <column_name> (<type>)
  - Grounded Value: <specific_value_if_found_else_None>
  - Key Info: <PRIMARY KEY | FOREIGN KEY -> ref_table.ref_column | None>

(Repeat for all relevant columns in all relevant tables)
</schema>"""


# ── light schema（懒加载）─────────────────────────────────────────────────────
_light_schema: dict | None = None


def get_light_schema() -> dict:
    """加载 light schema：train 先加载、test 覆盖（test 覆盖 206 库更全）。"""
    global _light_schema
    if _light_schema is None:
        merged: dict = {}
        for path in (LIGHT_SCHEMA_TRAIN, LIGHT_SCHEMA_TEST, LIGHT_SCHEMA_DK_NEW):
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    merged.update(json.load(f))
        _light_schema = merged
    return _light_schema


def generate_schema_for_question(question: str, db_id: str, max_retries: int = 2) -> str:
    """为单条 question 生成 raw schema 字符串（失败返回空串）。"""
    light_schema_text = get_light_schema().get(db_id, "")
    if not light_schema_text:
        return ""

    user_content = f"User Question: {question}\n\nDatabase Schema:\n{light_schema_text}"

    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
                max_tokens=4096,
            )
            content = resp.choices[0].message.content
            match = re.search(r"<schema>(.*?)</schema>", content, re.DOTALL)
            if not match:
                if attempt < max_retries:
                    time.sleep(1)
                    continue
                return ""
            # 扁平格式：直接返回 <schema> 正文（无前缀），与 ReAct_Agent 消费格式一致
            return match.group(1).strip()
        except Exception as e:  # noqa: BLE001
            print(f"  ❌ Error (attempt {attempt + 1}) [{db_id}]: {e}")
            if attempt < max_retries:
                time.sleep(2)
            else:
                return ""
    return ""


def load_input(input_path: str, question_field: str) -> list[tuple[str, str]]:
    """读取输入数据，支持 .json / .parquet，返回 [(db_id, question), ...]。

    - .parquet：需含 `db_id` 与 `question_field` 列
    - .json：Spider 风格 list[dict]，每项含 `db_id` 与 `question_field`
    """
    path = Path(input_path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
        return [(str(r["db_id"]), str(r[question_field])) for _, r in df.iterrows()]
    if path.suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return [(str(item["db_id"]), str(item[question_field])) for item in data]
    raise ValueError(f"Unsupported input format: {path.suffix} (expected .json or .parquet)")


def save_cache(cache: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    tmp.replace(output_path)


def main():
    global MODEL
    parser = argparse.ArgumentParser(description="Generate schema-linking cache for Spider-style dataset (json/parquet)")
    parser.add_argument("--input", "--parquet", dest="input", default=DEFAULT_DEV_PARQUET,
                        help="Path to input .json or .parquet (must contain db_id + question field)")
    parser.add_argument("--question-field", default="question",
                        help="Column/key name holding the question text (e.g. SpiderSynQuestion for Spider-Syn)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output cache json path")
    parser.add_argument("--model", default=MODEL, help="LLM model name (overrides MODEL_NAME env)")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent LLM requests")
    parser.add_argument("--limit", type=int, default=0, help="Only process first N rows (0=all)")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from existing cache")
    args = parser.parse_args()

    MODEL = args.model

    output_path = Path(args.output)
    resume = not args.no_resume

    df_rows = load_input(args.input, args.question_field)
    if args.limit > 0:
        df_rows = df_rows[:args.limit]
    print(f"Loaded {len(df_rows)} samples from {args.input} (question field: {args.question_field})")
    print(f"Model: {MODEL} | workers: {args.workers} | output: {output_path}")

    # 预加载 light schema，并检查覆盖情况
    light_schema = get_light_schema()
    missing_dbs = sorted({db_id for db_id, _ in df_rows if db_id not in light_schema})
    if missing_dbs:
        print(f"⚠️  {len(missing_dbs)} db_id 无 light schema，将被跳过: {missing_dbs}")

    # 载入已有缓存（resume）
    cache: dict[str, str] = {}
    if resume and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            cache = json.load(f)
        print(f"Resuming from existing cache: {len(cache)} entries")

    # 构建待处理任务（跳过已缓存）
    tasks = []
    for db_id, question in df_rows:
        question_key = f"{db_id}|||{question}"
        if question_key in cache:
            continue
        tasks.append((question_key, db_id, question))

    total = len(tasks)
    print(f"To generate: {total} (skipped {len(df_rows) - total} already cached)")
    if total == 0:
        print("Nothing to do. ✅")
        return

    lock = threading.Lock()
    done = 0
    errors = 0
    start = time.time()

    def worker(item):
        question_key, db_id, question = item
        schema = generate_schema_for_question(question, db_id)
        return question_key, schema

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, t): t for t in tasks}
        for fut in as_completed(futures):
            question_key, schema = fut.result()
            with lock:
                done += 1
                if schema:
                    cache[question_key] = schema
                else:
                    errors += 1
                    print(f"❌ Failed: {question_key[:70]}...")

                if done % 10 == 0 or done == total:
                    elapsed = time.time() - start
                    speed = done / elapsed if elapsed > 0 else 0
                    eta = (total - done) / speed if speed > 0 else 0
                    print(
                        f"Progress: {done}/{total} ({done/total*100:.1f}%) | "
                        f"{speed:.2f} q/s | ETA {eta/60:.1f} min | errors {errors}"
                    )

                if done % 50 == 0:
                    save_cache(cache, output_path)
                    print(f"💾 Checkpoint saved: {len(cache)} entries")

    save_cache(cache, output_path)
    print(f"\n✅ Done! Total {len(cache)} entries | errors {errors} | "
          f"{(time.time()-start)/60:.1f} min | Output: {output_path}")


if __name__ == "__main__":
    main()
