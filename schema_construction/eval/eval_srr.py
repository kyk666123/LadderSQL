"""Evaluate SRR following RSL-SQL's methodology exactly.

Gold schema extraction (from RSL-SQL):
  1. sqlglot.parse_one(gold_sql) → table set + column set (no table attribution)
  2. Cartesian product: table × column
  3. Validate against real DB schema → gold set of 'table.column'

Predicted schema processing:
  Parse ATT/MARS schema text → extract table.column pairs → validate against DB schema

SRR = 1 if gold_set ⊆ pred_set, else 0

Data sources:
  - ATT v3: bird_clean_data dev json (questions match ATT cache keys)
  - MARS flat: old bird dev_20240627/dev.json (questions match MARS cache keys)
  - Both share the same dev_databases for DB schema validation
"""
import json
import os
import re
import sqlite3
import sqlglot
from typing import Dict, Set, List

# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════

DB_PATH = "/root/bird_eval/dev_databases"
ATT_CACHE = "/root/bird_eval/att_cache.json"
MARS_CACHE = "/root/bird_eval/mars_cache.json"
ATT_DEV = "/root/bird_eval/att_dev.json"
MARS_DEV = "/root/bird_eval/mars_dev.json"


# ═══════════════════════════════════════════════════════════════════════
# Part 1: Load DB Schemas (shared infrastructure)
# ═══════════════════════════════════════════════════════════════════════

def _load_single_db(db_file: str) -> List[str]:
    """Load table.column list from a single sqlite file."""
    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        tables = cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        ).fetchall()
        items = []
        for (table_name,) in tables:
            if table_name == "sqlite_sequence":
                continue
            cols = cursor.execute(
                f"PRAGMA table_info('{table_name}');"
            ).fetchall()
            for col_info in cols:
                col_name = col_info[1]
                items.append(f"{table_name}.{col_name}")
    return items


def get_all_schema(db_base_path: str) -> Dict[str, List[str]]:
    """Load all table.column from each database.
    Returns {db_id: ['table.column', ...]} matching RSL-SQL's get_all_schema().
    Falls back to /tmp copy if OSS mount I/O fails.
    """
    import shutil
    db_schema = {}
    for db_name in os.listdir(db_base_path):
        db_file = os.path.join(db_base_path, db_name, db_name + ".sqlite")
        if not os.path.exists(db_file):
            continue
        try:
            db_schema[db_name] = _load_single_db(db_file)
        except Exception:
            # Fallback: copy to /tmp and retry
            tmp_file = f"/tmp/{db_name}.sqlite"
            try:
                if not os.path.exists(tmp_file):
                    shutil.copy2(db_file, tmp_file)
                db_schema[db_name] = _load_single_db(tmp_file)
                print(f"  [INFO] Loaded {db_name} via /tmp fallback")
            except Exception as e:
                print(f"  [WARN] Failed to load DB {db_name}: {e}")
    return db_schema


# ═══════════════════════════════════════════════════════════════════════
# Part 2: Gold Schema Extraction (RSL-SQL original logic)
# ═══════════════════════════════════════════════════════════════════════

def extract_tables_and_columns(sql_query: str) -> Dict[str, set]:
    """Exact copy of RSL-SQL's util.py extract_tables_and_columns.
    Uses sqlglot to extract table names and column names from SQL.
    """
    parsed_query = sqlglot.parse_one(sql_query, read="sqlite")
    table_names = parsed_query.find_all(sqlglot.exp.Table)
    column_names = parsed_query.find_all(sqlglot.exp.Column)
    return {
        'table': {_table.name for _table in table_names if _table.name},
        'column': {_column.alias_or_name for _column in column_names if _column.alias_or_name}
    }


def build_gold_schema(sql: str, db_id: str, db_schema: Dict[str, List[str]]) -> Set[str]:
    """Build gold schema set following RSL-SQL evaluation_SL.py logic:
    1. Extract tables + columns from SQL
    2. Cartesian product (table × column)
    3. Keep only those existing in DB schema
    """
    try:
        ans = extract_tables_and_columns(sql)
    except Exception:
        # Fallback: try substring match against DB schema
        return _fallback_gold(sql, db_id, db_schema)

    list_db = [item.lower() for item in db_schema.get(db_id, [])]
    gold = set()
    for table in ans['table']:
        for column in ans['column']:
            schema = f"{table}.{column}"
            if schema.lower() in list_db:
                gold.add(schema.lower())
    return gold


def _fallback_gold(sql: str, db_id: str, db_schema: Dict[str, List[str]]) -> Set[str]:
    """Fallback for unparseable SQL: check if column name appears in SQL."""
    sql_lower = sql.lower()
    list_db = db_schema.get(db_id, [])
    gold = set()
    for item in list_db:
        col = item.split(".", 1)[1].lower()
        table = item.split(".", 1)[0].lower()
        if col in sql_lower and table in sql_lower:
            gold.add(item.lower())
    return gold


# ═══════════════════════════════════════════════════════════════════════
# Part 3: Predicted Schema Processing (ATT/MARS → table.column set)
# ═══════════════════════════════════════════════════════════════════════

def parse_pred_schema(schema_text: str, db_id: str, db_schema: Dict[str, List[str]]) -> Set[str]:
    """Parse ATT/MARS schema text into table.column set, validate against DB.

    ATT/MARS format:
        Table:
        - table_name
        Columns:
        - col_name (TYPE)
          - Grounded Value: ...
          - Key Info: ...
        Table:
        - table_name2
        ...
    """
    current_table = None
    section = None
    raw_pairs: Set[str] = set()

    for line in schema_text.split('\n'):
        stripped = line.strip()
        if not stripped:
            continue

        if stripped == 'Table:':
            section = 'table'
            continue
        if stripped == 'Columns:':
            section = 'columns'
            continue

        if not stripped.startswith('- '):
            # Handle "Table: - xxx" inline format
            m = re.match(r'^Table:\s+-\s+(\S+)', stripped)
            if m:
                current_table = m.group(1)
                section = 'columns'
            continue

        # Sub-property lines (indented ≥4 spaces) → skip
        leading_spaces = len(line) - len(line.lstrip())
        if leading_spaces >= 4:
            continue

        if section == 'table':
            match = re.match(r'^-\s+(\S+)', stripped)
            if match:
                current_table = match.group(1)

        elif section == 'columns' and current_table:
            # "- col_name (TYPE)" or "- col_name"
            match = re.match(r'^-\s+(.+?)\s*\(', stripped)
            if match:
                col_name = match.group(1).strip()
                raw_pairs.add(f"{current_table}.{col_name}")
            else:
                match = re.match(r'^-\s+(.+?)$', stripped)
                if match:
                    col_name = match.group(1).strip()
                    # Skip metadata lines
                    if col_name.lower().startswith('grounded') or col_name.lower().startswith('key info'):
                        continue
                    raw_pairs.add(f"{current_table}.{col_name}")

    # DB validation (same as RSL-SQL pred processing)
    list_db = [item.lower() for item in db_schema.get(db_id, [])]
    validated = set()
    for pair in raw_pairs:
        if pair.lower() in list_db:
            validated.add(pair.lower())

    return validated


# ═══════════════════════════════════════════════════════════════════════
# Part 4: SRR + NSR + Avg.T/C Computation
# ═══════════════════════════════════════════════════════════════════════

def evaluate(cache_path: str, dev_path: str, label: str, db_schema: Dict[str, List[str]]):
    """Full evaluation pipeline."""
    with open(cache_path, 'r') as f:
        cache = json.load(f)
    with open(dev_path, 'r') as f:
        dev_data = json.load(f)

    # Build lookup
    dev_lookup = {}
    for entry in dev_data:
        key = f'{entry["db_id"]}|||{entry["question"]}'
        dev_lookup[key] = entry

    total = srr_sum = 0
    nsr_intersect = nsr_gold_total = 0
    total_pred_columns = total_pred_tables = 0
    total_gold_items = 0
    skipped = 0
    failed_cases = []

    for key, pred_text in cache.items():
        if key not in dev_lookup:
            skipped += 1
            continue

        entry = dev_lookup[key]
        db_id = entry["db_id"]
        gold_sql = entry.get("SQL", "")
        if not gold_sql:
            skipped += 1
            continue

        # Gold: RSL-SQL style
        gold_set = build_gold_schema(gold_sql, db_id, db_schema)
        # Pred: parse + DB validation
        pred_set = parse_pred_schema(pred_text, db_id, db_schema)

        # SRR
        srr = 1 if gold_set.issubset(pred_set) else 0
        srr_sum += srr
        total += 1

        # NSR (non-strict recall = |gold ∩ pred| / |gold|)
        nsr_intersect += len(gold_set.intersection(pred_set))
        nsr_gold_total += len(gold_set)

        # Avg.T / Avg.C
        pred_tables = set(item.split('.')[0] for item in pred_set)
        total_pred_tables += len(pred_tables)
        total_pred_columns += len(pred_set)
        total_gold_items += len(gold_set)

        # Collect failed cases
        if srr == 0 and len(failed_cases) < 5:
            missing = gold_set - pred_set
            failed_cases.append({
                "db_id": db_id,
                "question": entry["question"][:80],
                "missing": sorted(list(missing))[:5],
                "gold_size": len(gold_set),
                "pred_size": len(pred_set),
            })

    # Report
    nsr = nsr_intersect / nsr_gold_total if nsr_gold_total > 0 else 0
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"  Gold: sqlglot + Cartesian product + DB validation (RSL-SQL)")
    print(f"{'='*65}")
    print(f"  Evaluated:    {total}")
    print(f"  Skipped:      {skipped}")
    print(f"  {'─'*55}")
    print(f"  SRR:          {srr_sum}/{total} = {srr_sum/total*100:.2f}%")
    print(f"  NSR:          {nsr:.4f} ({nsr*100:.2f}%)")
    print(f"  {'─'*55}")
    print(f"  Avg.T:        {total_pred_tables/total:.2f}")
    print(f"  Avg.C:        {total_pred_columns/total:.2f}")
    print(f"  Avg.Gold:     {total_gold_items/total:.2f}")
    print(f"{'='*65}")

    if failed_cases:
        print(f"\n  Failed case samples:")
        for i, fc in enumerate(failed_cases[:3]):
            print(f"    [{i+1}] {fc['db_id']} | {fc['question']}")
            print(f"        gold={fc['gold_size']}, pred={fc['pred_size']}")
            print(f"        missing: {fc['missing']}")
    print()


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Loading DB schemas...")
    db_schema = get_all_schema(DB_PATH)
    print(f"  Loaded {len(db_schema)} databases\n")

    # ATT v3 evaluation
    evaluate(ATT_CACHE, ATT_DEV, "ATT v3 (bird_dev_att_schema_cache_v3)", db_schema)

    # MARS flat evaluation
    evaluate(MARS_CACHE, MARS_DEV, "MARS flat (bird_dev_schema_cache_flat)", db_schema)
