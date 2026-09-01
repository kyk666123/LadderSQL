#!/usr/bin/env python3
"""Independently score a BIRD predict_*.json against gold by real execution.

This is exactly what the BIRD Eval Team does with our dev prediction file, and it is the
cleanest way to CONFIRM the reported dev EX without trusting any cached reward: it re-executes
both the predicted SQL and the gold SQL and compares result sets.

EX (BIRD execution accuracy): a prediction is correct iff its result set equals gold's, as a
multiset of rows ignoring row order (set-compare on stringified rows). Empty/erroring predictions
count as wrong.

Usage:
    python tools/score_predict_dev.py \
        --predict predict_dev.json \
        --gold /path/to/bird/dev_20240627/dev.json \
        --db-dir /local/bird_dev_databases \
        [--workers 16]
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor

SEP = "\t----- bird -----\t"


def run(db_path, sql, timeout=30):
    if not sql or not sql.strip():
        return None, "empty"
    conn = None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=timeout)
        conn.text_factory = lambda b: b.decode(errors="ignore")
        cur = conn.cursor()
        cur.execute(sql)
        return cur.fetchall(), None
    except Exception as e:  # noqa: BLE001
        return None, str(e)[:100]
    finally:
        if conn:
            conn.close()


def sig(rows):
    if rows is None:
        return None
    return frozenset(tuple(str(c) for c in r) for r in rows)


def gold_of(item):
    return item.get("SQL") or item.get("query") or item.get("gold_sql") or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict", required=True)
    ap.add_argument("--gold", required=True, help="dev.json with gold SQL (dev only; test has none)")
    ap.add_argument("--db-dir", required=True, help="<db_id>/<db_id>.sqlite, local disk")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dump-wrong", default="", help="optional: write wrong-question indices here")
    args = ap.parse_args()

    pred = json.load(open(args.predict, encoding="utf-8"))
    gold = json.load(open(args.gold, encoding="utf-8"))
    n = len(gold)
    print(f"[load] predict={len(pred)}  gold={n}")
    if len(pred) != n:
        print(f"[WARN] count mismatch: predict {len(pred)} vs gold {n}")

    def dbp(db_id):
        return os.path.join(args.db_dir, db_id, f"{db_id}.sqlite")

    def score_one(i):
        g = gold[i]
        db_id = g["db_id"]
        gsql = gold_of(g)
        pv = pred.get(str(i), "")
        psql, pdb = (pv.split(SEP) + [""])[:2] if SEP in pv else (pv, db_id)
        pr, perr = run(dbp(db_id), psql)
        gr, gerr = run(dbp(db_id), gsql)
        if gerr:
            return i, None, f"GOLD_ERR:{gerr}"        # gold itself does not run -> question excluded (very rare)
        ok = (perr is None) and (sig(pr) == sig(gr))
        return i, ok, (perr or "")

    hit = tot = gold_bad = 0
    wrong = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, ok, note in ex.map(score_one, range(n)):
            if ok is None:
                gold_bad += 1
                continue
            tot += 1
            if ok:
                hit += 1
            else:
                wrong.append(i)
    print(f"[EX] {hit}/{tot} = {100.0*hit/tot:.2f}%"
          + (f"   (gold un-runnable, skipped: {gold_bad})" if gold_bad else ""))
    if args.dump_wrong:
        json.dump(wrong, open(args.dump_wrong, "w"))
        print(f"[save] wrong indices ({len(wrong)}) -> {args.dump_wrong}")


if __name__ == "__main__":
    main()
