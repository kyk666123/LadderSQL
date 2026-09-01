"""
Evaluate SRR/Avg.C for Schema Linking v2 cache.
比较 v2 与 MARS 基线。
"""
import json
import os
import re
import sqlite3

import sqlglot

# ── 路径配置 ──
DB_PATH = "/path/to/nl2sql_dataset/bird/dev_20240627/dev_databases"
V2_CACHE = "/path/to/LadderSQL/schema_construction/relevant_schema_cache/bird_dev_schema_linking_v2.json"
MARS_CACHE = "/path/to/LadderSQL/schema_construction/relevant_schema_cache/mars_bird_dev_schema_cache.json"
NO_DESC_CACHE = "/path/to/LadderSQL/schema_construction/relevant_schema_cache/bird_dev_schema_cache_from_rollouts_no_desc.json"
DEV_PATH = "/path/to/nl2sql_dataset/bird/dev_20240627/dev.json"


def load_db_schemas(db_base_path):
    """Load all table.column pairs from SQLite databases."""
    db_schema = {}
    for db_name in os.listdir(db_base_path):
        db_file = os.path.join(db_base_path, db_name, db_name + ".sqlite")
        if os.path.exists(db_file):
            try:
                with sqlite3.connect(db_file) as conn:
                    cursor = conn.cursor()
                    tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
                    items = []
                    for (table_name,) in tables:
                        if table_name == "sqlite_sequence":
                            continue
                        cols = cursor.execute(f"PRAGMA table_info(`{table_name}`)").fetchall()
                        for col_info in cols:
                            items.append(f"{table_name}.{col_info[1]}")
                    db_schema[db_name] = items
            except Exception as e:
                print(f"  [WARN] {db_name}: {e}")
    return db_schema


def build_gold_schema(sql, db_id, db_schema):
    """Extract gold table.column set from SQL using sqlglot."""
    try:
        parsed = sqlglot.parse_one(sql, read="sqlite")
        table_names = {t.name for t in parsed.find_all(sqlglot.exp.Table) if t.name}
        column_names = {c.alias_or_name for c in parsed.find_all(sqlglot.exp.Column) if c.alias_or_name}
    except:
        # Fallback: substring matching
        sql_lower = sql.lower()
        list_db = db_schema.get(db_id, [])
        gold = set()
        for item in list_db:
            col = item.split(".", 1)[1].lower()
            table = item.split(".", 1)[0].lower()
            if col in sql_lower and table in sql_lower:
                gold.add(item.lower())
        return gold

    list_db = [item.lower() for item in db_schema.get(db_id, [])]
    gold = set()
    for table in table_names:
        for column in column_names:
            schema = f"{table}.{column}"
            if schema.lower() in list_db:
                gold.add(schema.lower())
    return gold


def parse_v2_schema(schema_text, db_id, db_schema):
    """Parse v2 schema format (same as v14: Table:/Columns: structure)."""
    current_table = None
    section = None
    raw_pairs = set()
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
            continue
        leading_spaces = len(line) - len(line.lstrip())
        if leading_spaces >= 4:
            continue
        if section == 'table':
            match = re.match(r'^-\s+(.+)$', stripped)
            if match:
                current_table = match.group(1).strip()
        elif section == 'columns' and current_table:
            match = re.match(r'^-\s+(.+?)\s*\(', stripped)
            if match:
                col_name = match.group(1).strip()
                raw_pairs.add(f"{current_table}.{col_name}")
            else:
                match = re.match(r'^-\s+(.+?)$', stripped)
                if match:
                    col_name = match.group(1).strip()
                    if col_name.lower().startswith('grounded') or col_name.lower().startswith('key info'):
                        continue
                    raw_pairs.add(f"{current_table}.{col_name}")

    list_db = [item.lower() for item in db_schema.get(db_id, [])]
    validated = set()
    for pair in raw_pairs:
        if pair.lower() in list_db:
            validated.add(pair.lower())
    return validated


def parse_mars_schema(schema_text, db_id, db_schema):
    """Parse MARS schema format."""
    current_table = None
    raw_pairs = set()
    for line in schema_text.split('\n'):
        stripped = line.strip()
        m = re.match(r'^# Table: (.+)$', stripped)
        if m:
            current_table = m.group(1).strip()
            continue
        if current_table:
            cols = re.findall(r'\(([^)]+?):(?:TEXT|INTEGER|REAL|NUMERIC|DATE|BLOB|FLOAT|VARCHAR|BOOLEAN)', stripped)
            for col_name in cols:
                raw_pairs.add(f"{current_table}.{col_name}")

    list_db = [item.lower() for item in db_schema.get(db_id, [])]
    validated = set()
    for pair in raw_pairs:
        if pair.lower() in list_db:
            validated.add(pair.lower())
    return validated


def evaluate(cache, db_schema, dev_lookup, label, parser_func):
    """Evaluate a schema cache and print metrics."""
    total = srr_sum = 0
    nsr_intersect = nsr_gold_total = 0
    total_pred_cols = total_pred_tables = 0

    for key, pred_text in cache.items():
        if key not in dev_lookup:
            continue
        entry = dev_lookup[key]
        db_id = entry["db_id"]
        gold_sql = entry.get("SQL", "")
        if not gold_sql:
            continue

        gold_set = build_gold_schema(gold_sql, db_id, db_schema)
        pred_set = parser_func(pred_text, db_id, db_schema)

        srr = 1 if gold_set.issubset(pred_set) else 0
        srr_sum += srr
        total += 1
        nsr_intersect += len(gold_set & pred_set)
        nsr_gold_total += len(gold_set)
        pred_tables = set(p.split('.')[0] for p in pred_set)
        total_pred_tables += len(pred_tables)
        total_pred_cols += len(pred_set)

    nsr = nsr_intersect / nsr_gold_total if nsr_gold_total > 0 else 0
    srr_pct = srr_sum / total * 100 if total > 0 else 0
    avg_t = total_pred_tables / total if total > 0 else 0
    avg_c = total_pred_cols / total if total > 0 else 0

    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    print(f"  Evaluated:  {total}")
    print(f"  SRR:        {srr_sum}/{total} = {srr_pct:.2f}%")
    print(f"  NSR:        {nsr*100:.2f}%")
    print(f"  Avg.T:      {avg_t:.2f}")
    print(f"  Avg.C:      {avg_c:.2f}")
    print(f"{'='*65}")
    print()
    return {"srr": srr_pct, "avg_c": avg_c, "total": total}


def main():
    print("Loading DB schemas...")
    db_schema = load_db_schemas(DB_PATH)
    print(f"  Loaded {len(db_schema)} databases")

    print("Loading dev data...")
    with open(DEV_PATH) as f:
        dev_data = json.load(f)
    dev_lookup = {}
    for entry in dev_data:
        key = f'{entry["db_id"]}|||{entry["question"]}'
        dev_lookup[key] = entry
    print(f"  Dev entries: {len(dev_lookup)}")

    results = {}

    # V2 Cache
    if os.path.exists(V2_CACHE):
        with open(V2_CACHE) as f:
            v2_cache = json.load(f)
        print(f"\n  V2 cache entries: {len(v2_cache)}")
        results['v2'] = evaluate(v2_cache, db_schema, dev_lookup,
                                 "Schema Linking v2 (GLM5.2 thinking)", parse_v2_schema)
    else:
        print(f"\n  [SKIP] V2 cache not found: {V2_CACHE}")

    # MARS Cache
    if os.path.exists(MARS_CACHE):
        with open(MARS_CACHE) as f:
            mars_raw = json.load(f)
        mars_cache = {item['question_key']: item['schema'] for item in mars_raw.values()}
        print(f"  MARS cache entries: {len(mars_cache)}")
        results['mars'] = evaluate(mars_cache, db_schema, dev_lookup,
                                   "MARS (baseline)", parse_mars_schema)
    else:
        print(f"  [SKIP] MARS cache not found")

    # No-desc Cache (previous RSL)
    if os.path.exists(NO_DESC_CACHE):
        with open(NO_DESC_CACHE) as f:
            no_desc_cache = json.load(f)
        print(f"  No-desc cache entries: {len(no_desc_cache)}")
        results['no_desc'] = evaluate(no_desc_cache, db_schema, dev_lookup,
                                      "RSL no-desc (previous, EX=60.0%)", parse_v2_schema)

    # Summary comparison
    if results:
        print("\n" + "=" * 65)
        print("  COMPARISON SUMMARY")
        print("=" * 65)
        print(f"  {'Method':<35} {'SRR%':<10} {'Avg.C':<10} {'Entries'}")
        print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*8}")
        for name, r in results.items():
            print(f"  {name:<35} {r['srr']:<10.2f} {r['avg_c']:<10.2f} {r['total']}")


if __name__ == "__main__":
    main()
