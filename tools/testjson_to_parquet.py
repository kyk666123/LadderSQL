#!/usr/bin/env python3
"""Convert BIRD `test.json` (or `dev.json`) into the parquet the candidate sampler reads.

`test.json` records look like:
    {"db_id": "...", "question": "...", "evidence": "...", "SQL": ""}
Only `db_id`, `question`, `evidence` are used downstream; `SQL` is kept as an empty
string so that the no-gold path is taken.

Row order is preserved — the index of each row becomes the key in `predict_test.json`,
so it MUST match the order in the original json.

Usage:
    python tools/testjson_to_parquet.py --input test.json --output work/test.parquet
"""
from __future__ import annotations

import argparse
import json

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):                       # tolerate {"0": {...}} style
        data = [data[k] for k in sorted(data, key=lambda x: int(x))]

    rows = []
    for i, it in enumerate(data):
        rows.append({
            "question_id": it.get("question_id", i),
            "db_id": it["db_id"],
            "question": it["question"],
            "evidence": it.get("evidence", "") or "",
            "query": it.get("SQL", "") or "",        # empty on test → no-gold path
        })
    df = pd.DataFrame(rows)
    df.to_parquet(args.output, index=False)
    n_ev = sum(1 for r in rows if r["evidence"])
    n_gold = sum(1 for r in rows if r["query"])
    print(f"[ok] {len(rows)} questions -> {args.output}   (with evidence: {n_ev}, with gold SQL: {n_gold})")
    print(f"[dbs] {len(set(r['db_id'] for r in rows))} distinct databases")


if __name__ == "__main__":
    main()
