#!/usr/bin/env python3
"""Self-check for a BIRD submission file (predict_*.json).

Checks:
  1. top level is a dict whose keys are the consecutive integer strings "0".."N-1"
     (nothing missing, nothing duplicated)
  2. every value contains the separator "\\t----- bird -----\\t" exactly once
  3. the SQL part is non-empty, starts with SELECT/WITH, has no newline, no SQL
     comment, and balanced parentheses
  4. the db_id part is non-empty; with --db-dir, <db_dir>/<db_id>/<db_id>.sqlite
     must exist
  5. with --expect-n, the number of entries must match
  6. with --questions (the question json/parquet), db_id must agree entry by entry

Usage:
  python tools/check_predict_format.py results/predict_test.json --expect-n 1789
  python tools/check_predict_format.py results/predict_test.json \
      --db-dir /root/local_bird_db/test_databases --questions /path/to/test.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

SEP = "\t----- bird -----\t"


def load_questions(path):
    if path.endswith(".parquet"):
        import pandas as pd
        return pd.read_parquet(path).to_dict(orient="records")
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predict")
    ap.add_argument("--expect-n", type=int, default=0)
    ap.add_argument("--db-dir", default="")
    ap.add_argument("--questions", default="", help="question file (json / parquet), to check db_id alignment")
    args = ap.parse_args()

    with open(args.predict) as f:
        d = json.load(f)
    errs, warns = [], []

    if not isinstance(d, dict):
        print("[FAIL] the top level must be a JSON object ({\"0\": \"...\"})")
        sys.exit(1)

    n = len(d)
    print(f"[info] entries = {n}")
    if args.expect_n and n != args.expect_n:
        errs.append(f"entries {n} != expected {args.expect_n}")

    # keys must be consecutive
    try:
        keys = sorted(int(k) for k in d)
    except ValueError:
        errs.append("a non-integer key is present")
        keys = []
    if keys and keys != list(range(n)):
        missing = sorted(set(range(max(keys) + 1)) - set(keys))[:10]
        errs.append(f"keys are not the consecutive integers 0..{n-1}, missing e.g.: {missing}")

    qs = load_questions(args.questions) if args.questions else None
    if qs is not None and len(qs) != n:
        errs.append(f"question file has {len(qs)} entries vs {n} predictions")

    n_empty = n_bad_sep = n_bad_start = n_multiline = n_comment = n_paren = n_nodb = 0
    for k in d:
        v = d[k]
        if not isinstance(v, str):
            errs.append(f"key {k}: value is not a string")
            continue
        if v.count(SEP) != 1:
            n_bad_sep += 1
            continue
        sql, db_id = v.split(SEP)
        s = sql.strip()
        if not s:
            n_empty += 1
        else:
            head = s.lstrip("(").lstrip().upper()
            if not (head.startswith("SELECT") or head.startswith("WITH")):
                n_bad_start += 1
            if "\n" in sql or "\r" in sql:
                n_multiline += 1
            if "--" in sql or "/*" in sql:
                n_comment += 1
            if s.count("(") != s.count(")"):
                n_paren += 1
        if not db_id.strip():
            n_nodb += 1
        elif args.db_dir:
            p = os.path.join(args.db_dir, db_id, f"{db_id}.sqlite")
            if not os.path.exists(p):
                errs.append(f"key {k}: database not found {p}")
        if qs is not None:
            try:
                gold_db = qs[int(k)].get("db_id", "")
            except (ValueError, IndexError):
                gold_db = ""
            if gold_db and gold_db != db_id:
                errs.append(f"key {k}: db_id mismatch (predicted {db_id} vs question {gold_db})")

    for cnt, msg, hard in (
        (n_bad_sep, "separator is not exactly one '\\t----- bird -----\\t'", True),
        (n_empty, "SQL is empty", True),
        (n_nodb, "db_id is empty", True),
        (n_bad_start, "SQL does not start with SELECT/WITH", False),
        (n_paren, "unbalanced parentheses (possibly truncated)", False),
        (n_multiline, "SQL contains a newline", False),
        (n_comment, "SQL contains a comment (-- or /*)", False),
    ):
        if cnt:
            (errs if hard else warns).append(f"{cnt} entries: {msg}")

    for w in warns:
        print(f"[WARN] {w}")
    if errs:
        for e in errs[:30]:
            print(f"[FAIL] {e}")
        if len(errs) > 30:
            print(f"[FAIL] ... and {len(errs)-30} more")
        sys.exit(1)
    print("[OK] format check passed" + (" (with WARNs, worth eyeballing a few)" if warns else ""))


if __name__ == "__main__":
    main()
