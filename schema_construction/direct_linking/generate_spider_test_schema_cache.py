#!/usr/bin/env python3
"""Generate GLM-5 schema cache with descriptions for datasets.

Generates schema caches for:
- test_dev_500.parquet
- test.parquet
- train_with_variance_not_zero_cleaned.parquet

Uses the same prompt as training (REACT_SQL_TOOL_SCHEMA_GROUNDING_PROMPT).

Usage:
    python generate_test_schema_cache.py --dataset test_dev_500
    python generate_test_schema_cache.py --dataset test
    python generate_test_schema_cache.py --dataset train
    python generate_test_schema_cache.py --dataset all  # Generate all
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# Add agent directory to path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from openai import OpenAI
from original_agent.semantic_disambiguation import SemanticDisambiguator


# ── LLM 配置 ──────────────────────────────────────────────────────────
MODEL = os.environ.get("MODEL_NAME", "qwen3-coder-plus")
client = OpenAI(
    api_key=os.environ.get("ALICLOUD_API_KEY"),
    base_url=os.environ.get("ALICLOUD_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    timeout=120.0,
)

# ── System Prompt ──────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a precise schema grounding agent for NL2SQL. Analyze the user's question and database schema, then extract ALL relevant tables, columns and grounded values.

Instructions:
1. Identify entities, attributes, conditions, and aggregations from the question.
2. Map each to tables/columns. A table is relevant if any column is needed for SELECT, FROM/JOIN, WHERE, GROUP BY, ORDER BY, or HAVING.
3. For each relevant table, include ALL necessary columns:
   - Columns for SELECT output and WHERE/HAVING conditions
   - PK/FK columns required for JOIN connectivity
   - Columns for COUNT/SUM/AVG aggregation targets (even FK columns used for counting)
   - Columns whose Description matches question keywords (check descriptions carefully)
4. CRITICAL - Join Paths: Check the [Join Relationships] section. When two entities need connecting, ALWAYS include every intermediate/bridge table on the path. Never skip a bridge table.
5. CRITICAL - Disambiguation: If [Disambiguation Notes] are present, follow the usage_guide to select the correct table based on question intent.
6. Ground values: Match question keywords to sample values (case-insensitive). Do not invent values.
7. Final check: Ensure all selected tables are connected via FK paths. If isolated, find the bridge table.

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


def load_light_schema(db_id: str, spider_dir: str) -> str:
    """Load light schema for a database."""
    # Light schema 在 agent 目录下，不在 spider_dir
    agent_dir = Path(__file__).parent
    
    # Try test light schema first
    test_ls_path = agent_dir / "original_agent/db_schema_preprocess/light_schema/spider_test_light_schema.json"
    if test_ls_path.exists():
        with open(test_ls_path, 'r') as f:
            light_schema = json.load(f)
        if db_id in light_schema:
            return light_schema[db_id]
    
    # Fallback to train light schema
    train_ls_path = agent_dir / "original_agent/db_schema_preprocess/light_schema/spider_train_light_schema.json"
    if train_ls_path.exists():
        with open(train_ls_path, 'r') as f:
            light_schema = json.load(f)
        if db_id in light_schema:
            return light_schema[db_id]
    
    return ""


# 全局加载消歧模块
_disambiguator: SemanticDisambiguator | None = None
_full_schema: dict | None = None


def get_disambiguator() -> SemanticDisambiguator:
    """懒加载消歧模块。"""
    global _disambiguator
    if _disambiguator is None:
        sim_path = Path(__file__).parent / "original_agent/db_schema_cache/semantic_similarity.json"
        _disambiguator = SemanticDisambiguator(sim_path)
    return _disambiguator


def get_full_schema() -> dict:
    """懒加载全量schema（用于提取FK连接关系）。"""
    global _full_schema
    if _full_schema is None:
        full_path = Path(__file__).parent / "original_agent/db_schema_cache/db_full_schema.json"
        with open(full_path, encoding="utf-8") as f:
            _full_schema = json.load(f)
    return _full_schema


def build_join_relationships(db_id: str) -> str:
    """从full_schema中提取FK连接关系，生成连接关系文本。"""
    full_schema = get_full_schema()
    db_info = full_schema.get(db_id)
    if not db_info:
        return ""

    tables = db_info.get("tables", {})
    # 收集所有FK关系: (from_table, from_col) -> (to_table, to_col)
    fk_edges = []
    for tname, tinfo in tables.items():
        for cname, cinfo in tinfo.get("columns", {}).items():
            ki = cinfo.get("key_info", "")
            if "FOREIGN KEY ->" in ki:
                # Parse: "FOREIGN KEY -> Table.Column" or "PRIMARY KEY, FOREIGN KEY -> Table.Column"
                import re
                m = re.search(r"FOREIGN KEY\s*->\s*(\w+)\.(\w+)", ki)
                if m:
                    ref_table = m.group(1)
                    ref_col = m.group(2)
                    fk_edges.append((tname, cname, ref_table, ref_col))

    if not fk_edges:
        return ""

    # 构建连接关系文本
    lines = ["[Join Relationships]"]
    lines.append("Direct FK connections:")
    for from_t, from_c, to_t, to_c in fk_edges:
        lines.append(f"- {from_t}.{from_c} -> {to_t}.{to_c}")

    # 检测桥接表（有2个及以上出方向FK的表）
    from collections import defaultdict
    outgoing_fks = defaultdict(list)
    for from_t, from_c, to_t, to_c in fk_edges:
        outgoing_fks[from_t].append((from_c, to_t, to_c))

    bridge_tables = [(t, fks) for t, fks in outgoing_fks.items() if len(fks) >= 2]
    if bridge_tables:
        lines.append("")
        lines.append("Bridge/Junction tables (connect multiple entities):")
        for bridge_t, fks in bridge_tables:
            connected = [f"{to_t}" for _, to_t, _ in fks]
            lines.append(f"- {bridge_t} connects: {' <-> '.join(connected)}")

    return "\n".join(lines)


def generate_schema_for_question(
    question: str,
    db_id: str,
    light_schema_text: str,
    max_retries: int = 2,
) -> str:
    """Generate schema for a single question using LLM with enhanced context."""
    # 构建增强schema：light_schema + 连接关系 + 消歧注释
    parts = [light_schema_text]

    # 注入连接关系
    join_text = build_join_relationships(db_id)
    if join_text:
        parts.append(join_text)

    # 注入消歧注释
    disambiguator = get_disambiguator()
    hint = disambiguator.get_disambiguation_hint(db_id)
    if hint:
        parts.append(hint)

    enhanced_schema = "\n\n".join(parts)

    user_content = f"User Question: {question}\n\nDatabase Schema:\n{enhanced_schema}"

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

            # Extract schema from <schema> tags
            pattern = r"<schema>(.*?)</schema>"
            match = re.search(pattern, content, re.DOTALL)

            if not match:
                print(f"  ⚠️  No <schema> tag found (attempt {attempt + 1})")
                if attempt < max_retries:
                    time.sleep(1)
                    continue
                return ""

            raw_schema = match.group(1).strip()
            schema = f"Relevant tables, columns retrieved by llm:\n{raw_schema}"
            return schema

        except Exception as e:
            print(f"  ❌ Error (attempt {attempt + 1}): {e}")
            if attempt < max_retries:
                time.sleep(2)
            else:
                return ""

    return ""


def generate_cache_for_dataset(
    dataset_name: str,
    parquet_path: str,
    spider_dir: str,
    output_path: str,
    resume: bool = True,
):
    """Generate schema cache for a dataset."""
    
    print(f"\n{'='*80}")
    print(f"Generating schema cache for: {dataset_name}")
    print(f"Model: {MODEL}")
    print(f"{'='*80}")
    
    # Load dataset
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} samples from {parquet_path}")
    
    # Load existing cache (if resume)
    cache = {}
    if resume and Path(output_path).exists():
        with open(output_path, 'r') as f:
            cache = json.load(f)
        print(f"Resuming from existing cache: {len(cache)} entries")
    
    # Generate schemas
    total = len(df)
    processed = 0
    skipped = 0
    errors = 0
    
    start_time = time.time()
    
    for idx, row in df.iterrows():
        question = row["question"]
        db_id = row["db_id"]
        question_key = f"{db_id}|||{question}"
        rollout_id = row.get("rollout_id", f"ro-{idx}")
        
        # Skip if already in cache (check by question_key in values)
        if any(v.get("question_key") == question_key for v in cache.values() if isinstance(v, dict)):
            skipped += 1
            processed += 1
            continue
        
        # Load light schema
        light_schema_text = load_light_schema(db_id, spider_dir)
        if not light_schema_text:
            print(f"⚠️  Light schema not found for db_id: {db_id}")
            errors += 1
            processed += 1
            continue
        
        # Generate schema
        schema = generate_schema_for_question(question, db_id, light_schema_text)
        
        if schema:
            cache[rollout_id] = {
                "db_id": db_id,
                "question": question,
                "rollout_id": rollout_id,
                "question_key": question_key,
                "schema": schema,
            }
            processed += 1
        else:
            print(f"❌ Failed to generate schema for: {question_key[:60]}...")
            errors += 1
            processed += 1
        
        # Print progress
        if processed % 10 == 0 or processed == total:
            elapsed = time.time() - start_time
            speed = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed) / speed if speed > 0 else 0
            print(
                f"Progress: {processed}/{total} "
                f"({processed/total*100:.1f}%) | "
                f"Speed: {speed:.2f} samples/s | "
                f"ETA: {eta/60:.1f} min | "
                f"Errors: {errors}"
            )
        
        # Save checkpoint every 50 samples
        if processed % 50 == 0:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
            print(f"💾 Saved checkpoint: {len(cache)} entries")
    
    # Final save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"✅ Done!")
    print(f"Total: {len(cache)} entries")
    print(f"Time: {total_time/60:.1f} minutes")
    print(f"Output: {output_path}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate GLM-5 schema cache for datasets")
    parser.add_argument(
        "--dataset",
        choices=["test_dev_500", "test", "train", "all"],
        default="all",
        help="Which dataset to generate cache for"
    )
    parser.add_argument(
        "--spider-dir",
        default="/path/to/dataset/spider",
        help="Path to Spider dataset directory"
    )
    parser.add_argument(
        "--output-dir",
        default="original_agent/db_schema_cache",
        help="Output directory for cache files"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from existing cache"
    )
    
    args = parser.parse_args()
    
    spider_dir = args.spider_dir
    output_dir = args.output_dir
    resume = not args.no_resume
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    datasets = []
    if args.dataset in ["test_dev_500", "all"]:
        datasets.append(("test_dev_500", f"{spider_dir}/test_dev_500.parquet"))
    if args.dataset in ["test", "all"]:
        datasets.append(("test", f"{spider_dir}/test.parquet"))
    if args.dataset in ["train", "all"]:
        datasets.append(("train", f"{spider_dir}/train_with_variance_not_zero_cleaned.parquet"))
    
    for dataset_name, parquet_path in datasets:
        # 输出路径
        if dataset_name == "test_dev_500":
            output_path = f"{output_dir}/glm5_test_dev_500_schema_cache_20260517_disambig.json"
        elif dataset_name == "test":
            output_path = f"{output_dir}/glm5_schema_cache_20260517_disambig.json"
        elif dataset_name == "train":
            output_path = f"{output_dir}/glm5_train_schema_cache_20260517_disambig.json"
        
        if not Path(parquet_path).exists():
            print(f"❌ Parquet file not found: {parquet_path}")
            continue
        
        generate_cache_for_dataset(
            dataset_name=dataset_name,
            parquet_path=parquet_path,
            spider_dir=spider_dir,
            output_path=output_path,
            resume=resume,
        )


if __name__ == "__main__":
    main()
