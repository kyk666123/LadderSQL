"""Generic execution-cache builder (works across datasets: spider test / bird 7b / ...).

Reuses build_exec_cache.exec_one (stable md5 signature + hard timeout + profile/preview).
Inputs and outputs come from environment variables, so no reference artifact is touched:
  BIRD_FILE   source sampling file (a list; per question, trials[].pred_sql)
  EXEC_CACHE  output cache path
  BON_DB_DIR  database root directory (read by bird_common)
Usage:
  BON_DB_DIR=... BIRD_FILE=... EXEC_CACHE=... python3 build_exec_cache_generic.py [--workers 32]
"""
from __future__ import annotations
import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import bird_common as B
import build_exec_cache as BEC


def norm(s: str) -> str:
    return " ".join((s or "").split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    src = B.BIRD_FILE
    out = os.environ.get("EXEC_CACHE") or f"{B.CACHE_DIR}/exec_cache_generic.json"
    print(f"[cfg] db_dir={B.DB_DIR}\n      src={src}\n      out={out}", flush=True)

    data = json.load(open(src))
    tasks = {}
    n_cand = 0
    for r in data:
        db_id = r["db_id"]
        dbp = B.db_path_of(db_id)
        for t in r["trials"]:
            s = t.get("pred_sql")
            if not s or not s.strip():
                continue
            n_cand += 1
            key = f"{db_id}|||{norm(s)}"
            if key not in tasks:
                tasks[key] = (dbp, s)
    print(f"[tasks] candidates(non-empty)={n_cand}  unique after dedup={len(tasks)}", flush=True)

    results = {}
    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(BEC.exec_one, dbp, s): k for k, (dbp, s) in tasks.items()}
        for fut in as_completed(futs):
            k = futs[fut]
            try:
                results[k] = fut.result()
            except Exception as e:  # noqa: BLE001
                results[k] = {"ok": False, "error": f"future err: {str(e)[:120]}",
                              "row_count": None, "result_sig": None, "preview": [], "profile": None}
            done += 1
            if done % 2000 == 0:
                print(f"  exec {done}/{len(tasks)} ({time.time()-t0:.0f}s)", flush=True)

    ok = sum(1 for v in results.values() if v["ok"])
    err = len(results) - ok
    empty = sum(1 for v in results.values() if v["ok"] and v.get("row_count") == 0)
    obj = {"meta": {"source_file": src, "db_dir": B.DB_DIR, "n_candidates": n_cand,
                    "n_unique": len(tasks), "n_ok": ok, "n_error": err, "n_empty_result": empty,
                    "key_format": "f\"{db_id}|||{' '.join(pred_sql.split())}\"",
                    "result_sig": "md5(str(sorted(str(r) for r in rows)))"},
           "results": results}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, out)
    print(f"\n[done] {out} ({os.path.getsize(out)/1e6:.1f} MB, {time.time()-t0:.0f}s)")
    print(f"  unique={len(tasks)} ok={ok} errors={err} empty results={empty}")


if __name__ == "__main__":
    main()
