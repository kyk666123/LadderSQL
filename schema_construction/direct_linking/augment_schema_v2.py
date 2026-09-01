#!/usr/bin/env python3
"""V2: One-step schema augmentation pipeline.
LLM directly outputs train-format schema with value selection + format alignment in one call.

Flow:
1. LLM extracts keywords from question -> vector retrieval
2. LLM receives (question, evidence, schema structure, retrieved values)
   and outputs final schema in bird_train_schema format with precise value selection.

Supports two modes:
- mars: Input is MARS schema cache (dict with db_id/question/schema)
- att:  Input is ATT schema cache (key=db_id|||question, value=train-format schema string)
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import re
import time
import threading
from pathlib import Path
from typing import cast
from concurrent.futures import ThreadPoolExecutor, as_completed

import chromadb

# Monkey-patch: chromadb 0.5.x expects PersistentData object but old DB stores dict in pickle
from chromadb.segment.impl.vector.local_persistent_hnsw import PersistentData
@staticmethod
def _patched_load(filename: str) -> PersistentData:
    with open(filename, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        return PersistentData(
            dimensionality=data.get('dimensionality'),
            total_elements_added=data.get('total_elements_added', 0),
            id_to_label=data.get('id_to_label', {}),
            label_to_id=data.get('label_to_id', {}),
            id_to_seq_id=data.get('id_to_seq_id', {}),
        )
    return cast(PersistentData, data)
PersistentData.load_from_file = _patched_load

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MARS_CACHE_PATH = "/path/to/LadderSQL/schema_construction/relevant_schema_cache/mars_bird_dev_schema_cache.json"
ATT_CACHE_PATH = "/path/to/LadderSQL/schema_construction/relevant_schema_cache/bird_dev_att_schema_cache.json"
MARS_OUTPUT_PATH = "/path/to/LadderSQL/schema_construction/relevant_schema_cache/mars_bird_dev_schema_cache_value_augmented.json"
ATT_OUTPUT_PATH = "/path/to/LadderSQL/schema_construction/relevant_schema_cache/bird_dev_att_schema_cache_v3.json"
BIRD_DEV_PARQUET = "/path/to/nl2sql_dataset/bird/bird_clean_data/dev_20251106.parquet"
CHROMA_PATH = "/root/bird_dev_chroma"
LLM_MODEL = "qwen3-coder-plus"
LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

_thread_local = threading.local()
_chroma_client = None
_chroma_lock = threading.Lock()


def init_chroma():
    """Initialize global ChromaDB client (call once before threading)."""
    global _chroma_client
    _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    logger.info(f"ChromaDB initialized: {len(_chroma_client.list_collections())} collections")


def get_llm_client():
    from openai import OpenAI
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("Set DASHSCOPE_API_KEY")
    return OpenAI(api_key=api_key, base_url=LLM_BASE_URL)


def call_llm(client, messages, temperature=0.0, max_tokens=4096):
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
            )
            content = response.choices[0].message.content or ""
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            return content
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                logger.warning(f"LLM failed after 3 attempts: {e}")
    return ""


# ─── Prompts ───────────────────────────────────────────────────────────────────

KEYWORD_PROMPT = """You are a SQL expert. Given a question about a database, extract **value keywords** that would appear in SQL WHERE/HAVING/JOIN conditions as literal values.
Rules:
- Extract specific names, dates, numbers, categories, status codes mentioned in the question or evidence.
- Do NOT extract generic words (total, average, count, number).
- Do NOT extract column names or table names.
- Output: JSON array of strings. Empty [] if no literal values are needed.

Question: {question}
Evidence: {evidence}
Output:"""


SCHEMA_OUTPUT_PROMPT = """You are a database schema expert. Your task is to SELECT only the relevant tables and columns needed for SQL generation, and produce a minimal pruned schema.

Given:
1. A user question
2. External knowledge/evidence
3. The full database schema structure (tables, columns, types, foreign keys)
4. Retrieved database values from vector search

Your job:
- CAREFULLY analyze which tables and columns are ACTUALLY NEEDED to answer the question.
- REMOVE tables and columns that are NOT relevant to the SQL query.
- Keep only: tables/columns used in SELECT, WHERE, JOIN, GROUP BY, HAVING, ORDER BY.
- Also keep foreign key columns needed for JOIN paths between relevant tables.
- Be aggressive in pruning: fewer columns = better. Aim for 4-6 columns total.

Output the schema in EXACTLY this format (one block per table):
```
Table:
- <table_name>
Columns:
- <column_name> (<TYPE>)
  - Grounded Value: '<value>' or None
  - Key Info: <key info or None>
```

FORMAT RULES (STRICTLY FOLLOW):
1. Column types MUST be UPPERCASE: TEXT, INTEGER, REAL, DATE, NUMERIC, BLOB (never lowercase).
2. ALL Grounded Values (except None) MUST be wrapped in single quotes: `'value'`.
3. Multiple values for same column: separate with ` | ` inside quotes is NOT needed, just: `'val1' | 'val2'`.
4. `Table:` must have NO trailing space before newline.

Rules for Grounded Value:
1. For each column, decide if a specific literal value is needed to write the SQL that answers the question.
2. If YES and only ONE value: `Grounded Value: 'Alameda'`
3. If YES and MULTIPLE values (IN clause, OR conditions): `Grounded Value: 'Colusa' | 'Humboldt'`
4. If NO literal value is needed: `Grounded Value: None`
5. Only use values from the Retrieved Values list or explicitly stated in the question/evidence. NEVER invent values.
6. Integer/numeric values also need quotes: `Grounded Value: '1'` or `Grounded Value: '400'`

Rules for Key Info:
1. If the column is a PRIMARY KEY: `Key Info: PRIMARY KEY`
2. If the column is a FOREIGN KEY: `Key Info: FOREIGN KEY -> <referenced_table>.<referenced_column>`
3. Otherwise: `Key Info: None`

IMPORTANT:
- Output ONLY the schema text, no explanation, no markdown code fences.
- Select ONLY relevant tables and columns. Remove everything not needed for the query.
- The output must be clean and follow the exact format shown above.

---
Question: {question}
Evidence: {evidence}

Full Database Schema:
{schema_structure}

Retrieved Values (table.column: value):
{retrieved_values}

Output:"""


# ─── Helpers ───────────────────────────────────────────────────────────────────

def parse_mars_schema(schema_text):
    """Parse MARS format schema to extract structured info."""
    result = {
        "tables": [],
        "foreign_keys": [],
        "evidence": "",
    }

    # Extract evidence
    if "External Knowledge:" in schema_text:
        ek_part = schema_text.split("External Knowledge:")[1]
        if "Question:" in ek_part:
            ek_part = ek_part.split("Question:")[0]
        result["evidence"] = ek_part.strip()

    # Extract foreign keys
    if "\u3010Foreign keys\u3011" in schema_text:
        fk_part = schema_text.split("\u3010Foreign keys\u3011")[1]
        for marker in ["External Knowledge:", "Question:"]:
            if marker in fk_part:
                fk_part = fk_part.split(marker)[0]
        fk_lines = [l.strip() for l in fk_part.strip().split("\n") if l.strip() and "=" in l]
        result["foreign_keys"] = fk_lines

    # Extract tables and columns from Schema section
    schema_section = schema_text
    if "\u3010Schema\u3011" in schema_text:
        schema_section = schema_text.split("\u3010Schema\u3011")[1]
        if "\u3010Foreign keys\u3011" in schema_section:
            schema_section = schema_section.split("\u3010Foreign keys\u3011")[0]

    # Parse tables: # Table: xxx\n[(...), (...)]
    # Find each table header and its bracket-enclosed column block
    table_headers = [(m.start(), m.group(1)) for m in re.finditer(r'#\s*Table:\s*(\w+)', schema_section)]
    
    for i, (pos, table_name) in enumerate(table_headers):
        # Find the opening '[' after this table header
        bracket_start = schema_section.find('[', pos)
        if bracket_start < 0:
            continue
        # If there's a next table, make sure this bracket belongs to current table
        if i + 1 < len(table_headers) and bracket_start > table_headers[i+1][0]:
            continue
        
        # Find matching closing ']' accounting for nesting
        depth = 0
        bracket_end = -1
        for j in range(bracket_start, len(schema_section)):
            if schema_section[j] == '[':
                depth += 1
            elif schema_section[j] == ']':
                depth -= 1
                if depth == 0:
                    bracket_end = j
                    break
        
        if bracket_end < 0:
            continue
        
        cols_text = schema_section[bracket_start+1:bracket_end]

        columns = []
        # Parse columns by tracking parentheses depth
        col_parts = []
        paren_depth = 0
        current = ""
        for ch in cols_text:
            if ch == '(' and paren_depth == 0:
                current = ""
                paren_depth += 1
            elif ch == '(':
                current += ch
                paren_depth += 1
            elif ch == ')' and paren_depth == 1:
                col_parts.append(current)
                paren_depth = 0
            elif ch == ')':
                current += ch
                paren_depth -= 1
            elif paren_depth > 0:
                current += ch

        for col_text in col_parts:
            # Format: "col_name:TYPE, Examples: [...]" - handle col names with parens
            # Find the LAST colon before a type keyword
            type_match = re.search(r':([A-Z]+(?:\([^)]*\))?)', col_text)
            if type_match:
                col_name = col_text[:type_match.start()].strip()
                col_type = type_match.group(1).strip()
                columns.append({"name": col_name, "type": col_type})
            else:
                # Fallback: split on first colon
                parts = col_text.strip().split(":", 1)
                if len(parts) == 2:
                    col_name = parts[0].strip()
                    col_type = parts[1].strip().split(",")[0].strip()
                    columns.append({"name": col_name, "type": col_type})

        result["tables"].append({"name": table_name, "columns": columns})

    return result


def build_schema_structure_text(parsed):
    """Build a clean text representation of schema structure for the LLM."""
    lines = []
    for table in parsed["tables"]:
        lines.append(f"Table: {table['name']}")
        for col in table["columns"]:
            lines.append(f"  - {col['name']} ({col['type']})")
        lines.append("")

    if parsed["foreign_keys"]:
        lines.append("Foreign Keys:")
        for fk in parsed["foreign_keys"]:
            lines.append(f"  {fk}")

    return "\n".join(lines)


def get_chroma_collection(db_id):
    global _chroma_client
    with _chroma_lock:
        try:
            return _chroma_client.get_collection(db_id)
        except:
            return None


def query_vector_db(db_id, keywords, n_results=10):
    collection = get_chroma_collection(db_id)
    if not collection:
        return []
    all_results = []
    for kw in keywords:
        if not kw.strip():
            continue
        try:
            results = collection.query(query_texts=[kw], n_results=n_results)
            if results and results['documents'] and results['documents'][0]:
                for doc, meta, dist in zip(
                    results['documents'][0], results['metadatas'][0], results['distances'][0]
                ):
                    all_results.append({
                        "value": doc, "table": meta.get("table", ""),
                        "column": meta.get("column", ""), "distance": dist,
                    })
        except:
            pass
    seen = set()
    unique = []
    for r in sorted(all_results, key=lambda x: x["distance"]):
        key = (r["value"], r["table"], r["column"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique[:30]


def format_retrieved_values(retrieved):
    if not retrieved:
        return "None"
    lines = []
    for r in retrieved:
        lines.append(f"{r['table']}.{r['column']}: {r['value']}")
    return "\n".join(lines)


# ─── Main Processing ───────────────────────────────────────────────────────────

def process_one(key, entry, llm_client, mode="mars", evidence_map=None):
    """Process a single entry end-to-end."""
    if mode == "att":
        # ATT: key = "db_id|||question", entry = schema text string
        parts = key.split("|||")
        db_id = parts[0]
        question = parts[1] if len(parts) > 1 else ""
        schema_text = entry  # ATT value is the schema string directly
        # Load evidence from parquet-based map
        evidence = (evidence_map or {}).get(question, "")
        # ATT schema is already in train-format, use it directly as structure
        schema_structure = schema_text
    else:
        # MARS: entry is a dict with db_id, question, schema
        db_id = entry["db_id"]
        question = entry["question"]
        schema_text = entry["schema"]
        parsed = parse_mars_schema(schema_text)
        evidence = parsed["evidence"]
        schema_structure = build_schema_structure_text(parsed)

    # Step 1: Extract keywords
    msg = [{"role": "user", "content": KEYWORD_PROMPT.format(question=question, evidence=evidence)}]
    kw_response = call_llm(llm_client, msg, max_tokens=256)

    keywords = []
    try:
        match = re.search(r'\[.*?\]', kw_response, re.DOTALL)
        if match:
            keywords = json.loads(match.group(0))
            keywords = [str(k) for k in keywords if k]
    except:
        pass

    # Step 2: Vector retrieval
    retrieved = []
    if keywords:
        retrieved = query_vector_db(db_id, keywords, n_results=10)

    # Step 3: LLM one-step output final train-format schema
    retrieved_text = format_retrieved_values(retrieved)

    msg = [{"role": "user", "content": SCHEMA_OUTPUT_PROMPT.format(
        question=question,
        evidence=evidence,
        schema_structure=schema_structure,
        retrieved_values=retrieved_text,
    )}]
    final_schema = call_llm(llm_client, msg, max_tokens=4096)

    # Clean up code fences if present
    final_schema = re.sub(r'^```\w*\n?', '', final_schema)
    final_schema = re.sub(r'\n?```$', '', final_schema)
    final_schema = final_schema.strip()

    if mode == "att":
        # ATT output: keep same format (key -> schema string)
        return key, final_schema
    else:
        return key, {
            "db_id": db_id,
            "question": question,
            "question_id": entry.get("question_id", 0),
            "question_key": entry.get("question_key", ""),
            "schema": final_schema,
            "keywords": keywords,
            "retrieved_count": len(retrieved),
            "retrieved_values_sample": [
                {"value": r["value"], "table": r["table"], "column": r["column"]}
                for r in retrieved[:10]
            ],
        }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mars", "att"], default="mars")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    # Select paths based on mode
    if args.mode == "att":
        input_path = ATT_CACHE_PATH
        output_path = args.output or ATT_OUTPUT_PATH
    else:
        input_path = MARS_CACHE_PATH
        output_path = args.output or MARS_OUTPUT_PATH

    logger.info(f"Mode: {args.mode}")
    logger.info(f"Loading cache from {input_path}...")
    with open(input_path) as f:
        input_cache = json.load(f)
    logger.info(f"Loaded {len(input_cache)} entries")

    # Resume
    output_cache = {}
    if args.resume and Path(output_path).exists():
        try:
            with open(output_path) as f:
                output_cache = json.load(f)
            logger.info(f"Resumed {len(output_cache)} entries")
        except:
            logger.warning("Failed to load existing output, starting fresh")

    to_process = [(k, v) for k, v in input_cache.items() if k not in output_cache]
    if args.limit > 0:
        to_process = to_process[:args.limit]

    logger.info(f"To process: {len(to_process)} (already done: {len(output_cache)})")

    if not to_process:
        logger.info("Nothing to do!")
        return

    # Init ChromaDB globally before threading
    init_chroma()

    llm_client = get_llm_client()
    logger.info(f"Starting with {args.workers} workers...")

    total = len(input_cache)
    total_done = len(output_cache)
    lock = threading.Lock()
    start_time = time.time()
    errors = 0

    # Load evidence map for ATT mode
    evidence_map = {}
    if args.mode == "att":
        import pandas as pd
        logger.info(f"Loading evidence from {BIRD_DEV_PARQUET}...")
        df = pd.read_parquet(BIRD_DEV_PARQUET)
        for _, row in df.iterrows():
            q = row.get("question", "")
            ev = row.get("evidence", "")
            if q and ev:
                evidence_map[q] = ev
        logger.info(f"Evidence map: {len(evidence_map)} entries")

    def worker(item):
        key, entry = item
        return process_one(key, entry, llm_client, mode=args.mode, evidence_map=evidence_map)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(worker, item): item for item in to_process}

        for future in as_completed(futures):
            try:
                key, result = future.result()
                with lock:
                    output_cache[key] = result
                    total_done += 1

                    if total_done % 20 == 0:
                        elapsed = time.time() - start_time
                        logger.info(
                            f"[{total_done}/{total}] "
                            f"elapsed={elapsed:.0f}s | errors={errors}"
                        )

                    if total_done % 50 == 0:
                        with open(output_path, "w") as f:
                            json.dump(output_cache, f, ensure_ascii=False, indent=2)
            except Exception as e:
                with lock:
                    errors += 1
                logger.error(f"Worker error: {e}")

    # Final save
    with open(output_path, "w") as f:
        json.dump(output_cache, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"DONE! Total: {len(output_cache)} | Errors: {errors}")
    logger.info(f"Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    logger.info(f"Output: {output_path}")


if __name__ == "__main__":
    main()
