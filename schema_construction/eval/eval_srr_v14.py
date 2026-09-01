"""Evaluate SRR for v14, RSL raw, and MARS schema caches."""
import json, os, re, sqlite3, sqlglot
from typing import Dict, Set, List

# ===== Config =====
DB_PATH = "/tmp/bird_dev_databases_local"
V14_CACHE = "/path/to/LadderSQL/schema_construction/relevant_schema_cache/bird_dev_rsl_gpt4o_flat_filtered_v14.json"
V15_CACHE = "/path/to/LadderSQL/schema_construction/relevant_schema_cache/bird_dev_rsl_gpt4o_flat_filtered_v15.json"
RSL_RAW_CACHE = "/tmp/schema_refine_local/bird_dev_rsl_gpt4o_flat.json"
MARS_CACHE = "/path/to/LadderSQL/schema_construction/relevant_schema_cache/mars_bird_dev_schema_cache.json"
DEV_PATH = "/tmp/schema_refine_local/dev.json"

# ===== Load DB Schemas =====
def _load_single_db(db_file):
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
    return items

def get_all_schema(db_base_path):
    db_schema = {}
    for db_name in os.listdir(db_base_path):
        db_file = os.path.join(db_base_path, db_name, db_name + ".sqlite")
        if os.path.exists(db_file):
            try:
                db_schema[db_name] = _load_single_db(db_file)
            except Exception as e:
                print(f"  [WARN] {db_name}: {e}")
    return db_schema

# ===== Gold Schema (RSL-SQL style) =====
def extract_tables_and_columns(sql_query):
    parsed_query = sqlglot.parse_one(sql_query, read="sqlite")
    table_names = parsed_query.find_all(sqlglot.exp.Table)
    column_names = parsed_query.find_all(sqlglot.exp.Column)
    return {
        'table': {t.name for t in table_names if t.name},
        'column': {c.alias_or_name for c in column_names if c.alias_or_name}
    }

def build_gold_schema(sql, db_id, db_schema):
    try:
        ans = extract_tables_and_columns(sql)
    except:
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
    for table in ans['table']:
        for column in ans['column']:
            schema = f"{table}.{column}"
            if schema.lower() in list_db:
                gold.add(schema.lower())
    return gold

# ===== Parse v14 schema =====
def parse_pred_schema(schema_text, db_id, db_schema):
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

# ===== Parse RSL raw schema =====
def parse_rsl_raw_schema(schema_text, db_id, db_schema):
    current_table = None
    raw_pairs = set()
    for line in schema_text.split('\n'):
        stripped = line.strip()
        if stripped.startswith('Table:'):
            rest = stripped[6:].strip()
            if rest:
                current_table = rest
            continue
        if stripped == 'Columns:':
            continue
        if stripped.startswith('- ') and current_table:
            m = re.match(r'^-\s+(.+)\s+\([A-Z]+\)\s*$', stripped)
            if m:
                col_name = m.group(1).strip()
                raw_pairs.add(f"{current_table}.{col_name}")
            else:
                token = stripped[2:].strip()
                if '(' not in token:
                    current_table = token

    list_db = [item.lower() for item in db_schema.get(db_id, [])]
    validated = set()
    for pair in raw_pairs:
        if pair.lower() in list_db:
            validated.add(pair.lower())
    return validated

# ===== Parse MARS schema =====
def parse_mars_schema(schema_text, db_id, db_schema):
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

# ===== Evaluate =====
def evaluate(cache, db_schema, dev_lookup, label, parser_func):
    total = srr_sum = 0
    nsr_intersect = nsr_gold_total = 0
    total_pred_cols = total_pred_tables = 0
    failed_cases = []

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

        if srr == 0 and len(failed_cases) < 5:
            missing = gold_set - pred_set
            failed_cases.append({
                "db_id": db_id,
                "question": entry["question"][:60],
                "missing": sorted(list(missing))[:3],
                "gold": len(gold_set), "pred": len(pred_set)
            })

    nsr = nsr_intersect / nsr_gold_total if nsr_gold_total > 0 else 0
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    print(f"  Evaluated:  {total}")
    print(f"  SRR:        {srr_sum}/{total} = {srr_sum/total*100:.2f}%")
    print(f"  NSR:        {nsr*100:.2f}%")
    print(f"  Avg.T:      {total_pred_tables/total:.2f}")
    print(f"  Avg.C:      {total_pred_cols/total:.2f}")
    print(f"{'='*65}")
    if failed_cases:
        print(f"  Failed samples:")
        for i, fc in enumerate(failed_cases[:3]):
            print(f"    [{i+1}] {fc['db_id']} | {fc['question']}")
            print(f"        gold={fc['gold']}, pred={fc['pred']}, missing: {fc['missing']}")
    print()

# ===== Main =====
if __name__ == "__main__":
    print("Loading DB schemas...")
    db_schema = get_all_schema(DB_PATH)
    print(f"  Loaded {len(db_schema)} databases")

    print("Loading dev data...")
    with open(DEV_PATH) as f:
        dev_data = json.load(f)
    dev_lookup = {}
    for entry in dev_data:
        key = f'{entry["db_id"]}|||{entry["question"]}'
        dev_lookup[key] = entry
    print(f"  Dev entries: {len(dev_lookup)}")

    # Load caches
    print("Loading caches...")
    with open(V14_CACHE) as f:
        v14_cache = json.load(f)
    with open(V15_CACHE) as f:
        v15_cache = json.load(f)
    with open(RSL_RAW_CACHE) as f:
        rsl_raw_cache = json.load(f)
    with open(MARS_CACHE) as f:
        mars_raw = json.load(f)
    mars_cache = {item['question_key']: item['schema'] for item in mars_raw.values()}

    # Evaluate all
    evaluate(v15_cache, db_schema, dev_lookup, "v15 (PK/FK精简 + 仲裁偏保留)", parse_pred_schema)
    evaluate(v14_cache, db_schema, dev_lookup, "v14 (RSL精筛后)", parse_pred_schema)
    evaluate(mars_cache, db_schema, dev_lookup, "MARS", parse_mars_schema)
