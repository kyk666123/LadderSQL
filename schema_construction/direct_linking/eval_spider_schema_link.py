#!/usr/bin/env python3
"""
Evaluate Spider schema-linking quality for the flat-text relevant_schema_cache files.

Steps:
  1. Robustly parse the flat-text schema (handles all observed format variants) into a
     structured form {db_id, question, tables: {table_name: [columns...]}}.
  2. Convert prediction caches AND gold schema files to structured JSON (persisted).
  3. Compute table-level and column-level metrics (SRR / Recall / Precision / F1),
     matching the APEX-SQL evaluate_schema_linking methodology.

Two flat-text variants are handled:
  (A) "Columns:" variant  (both prediction caches, and the dev gold):
        Table:
        - club                     <- table name on its own line
        Columns:
        - Club_ID (number)         <- column (indented sub-lines are details)
          - Grounded Value: None
          - Key Info: PRIMARY KEY
      also inline table-name form:   "Table: club" then "Columns:" ...
  (B) LLM variant           (the test gold, spider_test_schema.json):
        Table: club                <- inline table name, NO "Columns:" line
        - Club_ID (INTEGER, PRIMARY KEY)   <- columns directly follow
        - Name (TEXT)

Unifying rule: after a "Table:" marker, the FIRST top-level "- X" is the table name only
when the name was not already given inline; once a table has a name, every subsequent
top-level "- X" line is a column. "Columns:" lines and indented "  - ..." lines are ignored.
"""

import json
import os
import re
from typing import Dict, List, Set, Tuple

BASE = os.path.dirname(os.path.abspath(__file__))

CACHE_DEV = os.path.join(BASE, "spider_glm52_dev_schema_cache.json")
CACHE_TEST = os.path.join(BASE, "spider_glm5_test_schema_cache_20260415.json")
CACHE_TRAIN = os.path.join(BASE, "spider_glm5_train_schema_cache_with_desc.json")
GOLD_DEV = "/path/to/LadderSQL/schema_construction/gold_schema_rule_based/spider_dev_schema.rule.json"
GOLD_TEST = "/path/to/LadderSQL/schema_construction/gold_schema_rule_based/spider_test_schema.rule.json"
GOLD_TRAIN = "/path/to/LadderSQL/schema_construction/gold_schema_rule_based/spider_train_schema.rule.json"

TABLE_RE = re.compile(r"^Table:\s*(.*)$")


def strip_col_name(raw: str) -> str:
    """Given the text after '- ' on a column line, return just the column name.

    Examples:
      'Club_ID (number)'                       -> 'Club_ID'
      'Club_ID (INTEGER, PRIMARY KEY)'         -> 'Club_ID'
      'Free Meal Count (K-12) (number)'        -> 'Free Meal Count (K-12)'
      'weird_col_without_type'                 -> 'weird_col_without_type'
    """
    s = raw.strip()
    if s.endswith(")") and " (" in s:
        s = s.rsplit(" (", 1)[0]
    return s.strip()


JUNK = "\x00JUNK\x00"


def _valid_table_name(name: str) -> bool:
    """Spider table names are SQL identifiers (no spaces/backticks). Lines with spaces
    or backticks are LLM prose that leaked into a few gold entries -> reject them."""
    return bool(name) and (" " not in name) and ("`" not in name)


def parse_flat_schema(text: str) -> Dict[str, List[str]]:
    """Parse flat-text schema into {table_name: [column_name, ...]}.

    Robust to both format variants and to edge cases (blank lines, inline vs two-line
    table names, missing 'Columns:' header, column names containing parentheses, and
    stray LLM prose lines in a handful of malformed gold entries).
    """
    tables: Dict[str, List[str]] = {}
    current: str = None          # current table name (None => pending, awaiting name line)
    have_pending = False         # a 'Table:' with empty inline name is awaiting its name

    for line in text.split("\n"):
        if line.strip() == "":
            continue

        m = TABLE_RE.match(line)
        # A "Table:" marker must NOT start with "- " (top-level list item)
        if m is not None and not line.startswith("- "):
            name = m.group(1).strip()
            if name:
                if _valid_table_name(name):
                    current = name
                    tables.setdefault(current, [])
                else:
                    current = JUNK
                have_pending = False
            else:
                current = None
                have_pending = True
            continue

        if line.strip() == "Columns:":
            # header only; columns follow as top-level '- ' lines
            continue

        if line.startswith("- "):
            content = line[2:]
            if have_pending or current is None:
                # this top-level item is the table NAME (two-line form)
                name = content.strip()
                if _valid_table_name(name):
                    current = name
                    tables.setdefault(current, [])
                else:
                    current = JUNK
                have_pending = False
            elif current is not JUNK:
                col = strip_col_name(content)
                if col:
                    tables[current].append(col)
            continue

        # indented detail line ("  - Grounded Value: ...", "  - Key Info: ...") -> ignore
        # any other stray line -> ignore (counted by validation separately)

    return tables


def validate_parse_coverage(values, label) -> int:
    """Return count of stray (unclassifiable) non-blank lines; should be 0."""
    stray = 0
    for text in values:
        for line in text.split("\n"):
            if line.strip() == "":
                continue
            if TABLE_RE.match(line) and not line.startswith("- "):
                continue
            if line.strip() == "Columns:":
                continue
            if line.startswith("- "):
                continue
            if line.startswith("  "):  # indented detail
                continue
            stray += 1
    print(f"  [{label}] stray/unclassified non-blank lines: {stray}")
    return stray


# ----------------------------- structured conversion -----------------------------

def build_structured_from_cache(cache: Dict[str, str]) -> Dict[str, Dict]:
    out = {}
    for key, text in cache.items():
        db_id, question = key.split("|||", 1)
        out[key] = {
            "db_id": db_id,
            "question": question,
            "tables": parse_flat_schema(text),
        }
    return out


def build_structured_from_gold(gold: List[Dict]) -> Dict[Tuple[str, str], Dict]:
    out = {}
    for item in gold:
        db_id = item["db_id"]
        question = item["question"]
        out[(db_id, question)] = {
            "db_id": db_id,
            "question": question,
            "tables": parse_flat_schema(item.get("pruned_schema", "")),
            "is_sufficient": item.get("is_pruned_schema_sufficient", True),
        }
    return out


# ----------------------------- metric sets -----------------------------

def norm(s: str) -> str:
    return s.strip().lower()


def table_set(db_id: str, tables: Dict[str, List[str]]) -> Set[str]:
    return {f"{db_id}.{norm(t)}" for t in tables.keys()}


def column_set(db_id: str, tables: Dict[str, List[str]]) -> Set[str]:
    cols = set()
    for t, cs in tables.items():
        for c in cs:
            cols.add(f"{db_id}.{norm(t)}.{norm(c)}")
    return cols


def calculate_metrics(predicted: Set[str], golden: Set[str]) -> Tuple[float, float, float, float]:
    """precision, recall, f1, coverage(SRR) -- identical semantics to APEX-SQL."""
    if len(predicted) == 0 and len(golden) == 0:
        return 1.0, 1.0, 1.0, 1.0
    if len(predicted) == 0:
        return 0.0, 0.0, 0.0, 0.0
    if len(golden) == 0:
        return 0.0, 0.0, 0.0, 0.0
    tp = len(predicted & golden)
    fp = len(predicted - golden)
    fn = len(golden - predicted)
    coverage = float(all(item in predicted for item in golden))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, coverage


def evaluate(pred_struct: Dict[str, Dict], gold_struct: Dict[Tuple[str, str], Dict],
             label: str, only_sufficient: bool = False):
    tp_, tr_, tf_, tc_ = [], [], [], []
    cp_, cr_, cf_, cc_ = [], [], [], []
    pred_tab_n, pred_col_n = [], []
    n = 0
    skipped_insuff = 0

    for key, pred in pred_struct.items():
        db_id = pred["db_id"]
        gkey = (pred["db_id"], pred["question"])
        gold = gold_struct.get(gkey)
        if gold is None:
            continue
        if only_sufficient and not gold.get("is_sufficient", True):
            skipped_insuff += 1
            continue

        p_tabs = table_set(db_id, pred["tables"])
        g_tabs = table_set(db_id, gold["tables"])
        p_cols = column_set(db_id, pred["tables"])
        g_cols = column_set(db_id, gold["tables"])

        a, b, c, d = calculate_metrics(p_tabs, g_tabs)
        tp_.append(a); tr_.append(b); tf_.append(c); tc_.append(d)
        a, b, c, d = calculate_metrics(p_cols, g_cols)
        cp_.append(a); cr_.append(b); cf_.append(c); cc_.append(d)
        pred_tab_n.append(len(p_tabs))
        pred_col_n.append(len(p_cols))
        n += 1

    def avg(x):
        return sum(x) / len(x) if x else 0.0

    perfect_t = sum(1 for f in tf_ if f == 1.0)
    perfect_c = sum(1 for f in cf_ if f == 1.0)

    print("\n" + "=" * 72)
    print(f"{label}  (n={n}" + (f", skipped_insufficient={skipped_insuff}" if only_sufficient else "") + ")")
    print("=" * 72)
    print("Table-Level:")
    print(f"  SRR:       {avg(tc_):.4f}")
    print(f"  Recall:    {avg(tr_):.4f}")
    print(f"  Precision: {avg(tp_):.4f}")
    print(f"  F1:        {avg(tf_):.4f}")
    print(f"  Perfect:   {perfect_t}/{n} ({100*perfect_t/n:.2f}%)" if n else "")
    print(f"  Avg predicted tables: {avg(pred_tab_n):.2f}")
    print("Column-Level:")
    print(f"  SRR:       {avg(cc_):.4f}")
    print(f"  Recall:    {avg(cr_):.4f}")
    print(f"  Precision: {avg(cp_):.4f}")
    print(f"  F1:        {avg(cf_):.4f}")
    print(f"  Perfect:   {perfect_c}/{n} ({100*perfect_c/n:.2f}%)" if n else "")
    print(f"  Avg predicted columns: {avg(pred_col_n):.2f}")

    return {
        "n": n,
        "table": {"srr": avg(tc_), "recall": avg(tr_), "precision": avg(tp_), "f1": avg(tf_),
                   "perfect": perfect_t, "avg_pred": avg(pred_tab_n)},
        "column": {"srr": avg(cc_), "recall": avg(cr_), "precision": avg(cp_), "f1": avg(cf_),
                    "perfect": perfect_c, "avg_pred": avg(pred_col_n)},
    }


def main():
    cache_dev = json.load(open(CACHE_DEV, encoding="utf-8"))
    cache_test = json.load(open(CACHE_TEST, encoding="utf-8"))
    cache_train = json.load(open(CACHE_TRAIN, encoding="utf-8"))
    gold_dev = json.load(open(GOLD_DEV, encoding="utf-8"))
    gold_test = json.load(open(GOLD_TEST, encoding="utf-8"))
    gold_train = json.load(open(GOLD_TRAIN, encoding="utf-8"))

    print("Parser validation (expect 0 stray lines everywhere):")
    validate_parse_coverage(cache_dev.values(), "dev cache")
    validate_parse_coverage(cache_test.values(), "test cache")
    validate_parse_coverage(cache_train.values(), "train cache")
    validate_parse_coverage([g.get("pruned_schema", "") for g in gold_dev], "dev gold")
    validate_parse_coverage([g.get("pruned_schema", "") for g in gold_test], "test gold")
    validate_parse_coverage([g.get("pruned_schema", "") for g in gold_train], "train gold")

    pred_dev = build_structured_from_cache(cache_dev)
    pred_test = build_structured_from_cache(cache_test)
    pred_train = build_structured_from_cache(cache_train)
    gs_dev = build_structured_from_gold(gold_dev)
    gs_test = build_structured_from_gold(gold_test)
    gs_train = build_structured_from_gold(gold_train)

    # Persist structured files
    outdir = os.path.join(BASE, "_structured")
    os.makedirs(outdir, exist_ok=True)
    json.dump(pred_dev, open(os.path.join(outdir, "pred_dev_structured.json"), "w"),
              ensure_ascii=False, indent=2)
    json.dump(pred_test, open(os.path.join(outdir, "pred_test_structured.json"), "w"),
              ensure_ascii=False, indent=2)
    json.dump(pred_train, open(os.path.join(outdir, "pred_train_structured.json"), "w"),
              ensure_ascii=False, indent=2)
    json.dump({f"{k[0]}|||{k[1]}": v for k, v in gs_dev.items()},
              open(os.path.join(outdir, "gold_dev_structured.json"), "w"),
              ensure_ascii=False, indent=2)
    json.dump({f"{k[0]}|||{k[1]}": v for k, v in gs_test.items()},
              open(os.path.join(outdir, "gold_test_structured.json"), "w"),
              ensure_ascii=False, indent=2)
    json.dump({f"{k[0]}|||{k[1]}": v for k, v in gs_train.items()},
              open(os.path.join(outdir, "gold_train_structured.json"), "w"),
              ensure_ascii=False, indent=2)
    print(f"\nStructured files written to {outdir}")

    # diagnostics: empty predictions
    empty_dev = sum(1 for v in pred_dev.values() if not v["tables"])
    empty_test = sum(1 for v in pred_test.values() if not v["tables"])
    empty_train = sum(1 for v in pred_train.values() if not v["tables"])
    print(f"Empty predicted schema: dev={empty_dev}, test={empty_test}, train={empty_train}")

    evaluate(pred_dev, gs_dev, "SPIDER DEV  (glm-5.2)")
    evaluate(pred_test, gs_test, "SPIDER TEST (glm-5)")
    evaluate(pred_train, gs_train, "SPIDER TRAIN (glm-5, with_desc)")


if __name__ == "__main__":
    main()
