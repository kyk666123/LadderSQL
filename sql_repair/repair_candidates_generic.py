"""Generic repair/verification driver (works across datasets: spider / bird).

Runs the 32 candidates through the DeepEye-style checker chain, producing a **repaired candidate
pool** (a new file) + a new execution cache + a recomputed reward_list.

Differences from repair_candidates.py:
  1. **Scope: only multi-result questions** (reward-blind). Questions are grouped by the execution
     signature cached_sig:
     - single-result (all 32 share one signature) -> skipped (not repaired, and downstream does no
       best-of-32 either, that result is emitted directly);
     - multi-result (>=2 signatures)             -> repaired.
     Repairing only the multi-result questions was verified to give the same final metric as
     repairing everything.
  2. **Reward per dataset**: spider=binary_reward (enumerating column permutations to match);
     bird=bird_reward (set comparison).

Reward-blind throughout: repair decisions use only execution validity / result shape / regex
anti-patterns / self-consistency, never gold.
Reward is recomputed only for the **changed** candidates (matching the final evaluation rule);
unchanged candidates keep their original reward_list entry.
**No-gold setting (e.g. the BIRD test submission)**: pass --no-gold, or simply have no gold_sql /
query field in the file, and relabelling is skipped automatically. Repair itself is unaffected
(it never looks at gold anyway).

Usage:
  BON_DATASET=bird BON_DB_DIR=<local db dir> BIRD_FILE=<src> EXEC_CACHE=<orig_cache> \
  python3 repair_candidates_generic.py --out-file <out.json> --out-cache <cache.json> \
          [--budget 6] [--chain default|full] [--workers 32] [--no-gold]
"""
from __future__ import annotations
import argparse, json, os, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

# same-repo relative paths: this directory (sql_repair) + inference_selection (execution / cache /
# judging) + training (reward_func)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_ROOT, "inference_selection"), os.path.join(_ROOT, "training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bird_common as B
import deepeye_checkers as C
import build_exec_cache as BEC

DATASET = os.environ.get("BON_DATASET", "bird").lower()

if DATASET == "spider":
    from reward_func.reward import binary_reward as _rw
    def reward_label(dbp, pred, gold):
        try:
            return int(_rw(pred, gold, dbp, False))
        except Exception:
            return 0
else:
    import bird_reward_label as RL
    def reward_label(dbp, pred, gold):
        return int(RL.bird_reward(dbp, pred, gold, timeout=30))


def norm(s):
    return " ".join((s or "").split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-file", required=True)
    ap.add_argument("--out-cache", required=True)
    ap.add_argument("--budget", type=int, default=6)
    ap.add_argument("--chain", type=str, default="default", choices=["default", "full"])
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-gold", action="store_true",
                    help="no-gold mode (BIRD test submission): do not recompute reward, only emit the repaired candidate pool and execution cache")
    ap.add_argument("--ckpt", default="",
                    help="checkpoint file for the intermediate repair results (default <out-file>.repair_ckpt.json); "
                         "after a crash or interruption, re-running the same command skips the phrasings already repaired and does not spend tokens twice")
    ap.add_argument("--ckpt-every", type=int, default=200,
                    help="how many unique phrasings to finish before writing a checkpoint")
    args = ap.parse_args()

    chain = C.DEFAULT_CHAIN if args.chain == "default" else C.FULL_CHAIN
    src = B.BIRD_FILE
    print(f"[cfg] dataset={DATASET} db_dir={B.DB_DIR}\n      src={src}\n      orig_cache={os.environ.get('EXEC_CACHE')}\n"
          f"      out_file={args.out_file}\n      out_cache={args.out_cache}\n      chain={chain} budget={args.budget}", flush=True)

    data = json.load(open(src))
    if args.limit:
        data = data[:args.limit]
    print(f"[load] {len(data)} questions", flush=True)

    # 1) group by execution signature, keeping only the **multi-result** questions in the repair
    #    scope (reward-blind)
    n_single = n_multi = 0
    scope_qids = set()
    for qi, it in enumerate(data):
        db = it["db_id"]
        sigs = set()
        for t in it["trials"]:
            s = t.get("pred_sql")
            if not s or not s.strip():
                continue
            sig = B.cached_sig(db, s)   # using the original exec cache
            sigs.add(sig if sig is not None else "NOCACHE")
        if len(sigs) <= 1:
            n_single += 1
        else:
            n_multi += 1
            scope_qids.add(qi)
    print(f"[scope] single-result (skipped)={n_single}  multi-result (repaired)={n_multi}", flush=True)

    # 2) collect unique phrasings from the multi-result questions only
    uniq = {}
    for qi, it in enumerate(data):
        if qi not in scope_qids:
            continue
        db = it["db_id"]; q = it.get("question", ""); ev = it.get("evidence", "") or ""
        for t in it["trials"]:
            s = t.get("pred_sql")
            if not s or not s.strip():
                continue
            k = (db, norm(s))
            if k not in uniq:
                uniq[k] = {"db": db, "sql": s, "q": q, "ev": ev}
    print(f"[uniq] {len(uniq)} unique (db, phrasing) pairs to repair (multi-result questions only)", flush=True)

    # 3) repair in parallel (with checkpointing: phrasings already repaired are not re-run, which
    #    satisfies the organisers' "resume from where it failed" requirement)
    ckpt_path = args.ckpt or (args.out_file + ".repair_ckpt.json")
    repaired = {}
    if os.path.exists(ckpt_path):
        try:
            raw_ck = json.load(open(ckpt_path))
            for k, v in raw_ck.items():
                db, sql = k.split("|||", 1)
                repaired[(db, sql)] = v
            print(f"[ckpt] loaded {len(repaired)} already-repaired entries <- {ckpt_path}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[ckpt] checkpoint file unusable ({e}), starting from scratch", flush=True)
            repaired = {}

    def save_ckpt():
        tmp = ckpt_path + ".tmp"
        json.dump({f"{k[0]}|||{k[1]}": v for k, v in repaired.items()},
                  open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
        os.replace(tmp, ckpt_path)

    items = [kv for kv in uniq.items() if kv[0] not in repaired]
    print(f"[uniq] {len(items)} left to repair ({len(uniq)-len(items)} skipped, already in the checkpoint)", flush=True)
    fired_cnt = 0; t0 = time.time(); done = 0

    def work(kv):
        k, ctx = kv
        try:
            import llm_usage
            llm_usage.set_unit(f"{ctx['db']}|||{k[1]}")   # repair calls are attributed to the unique phrasing (the unit after cross-question dedup)
        except Exception:  # noqa: BLE001
            pass
        rsql, fired = C.revise_chain(ctx["sql"], ctx["db"], ctx["q"], ctx["ev"],
                                     chain=chain, budget=args.budget)
        return k, rsql, fired

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for k, rsql, fired in ex.map(work, items):
            repaired[k] = rsql
            fired_cnt += (1 if fired else 0)
            done += 1
            if done % args.ckpt_every == 0:
                save_ckpt()
            if done % 500 == 0:
                print(f"  repaired {done}/{len(items)} ({time.time()-t0:.0f}s) LLM chains fired={fired_cnt} checkpoint saved", flush=True)
    save_ckpt()
    n_changed_uniq = sum(1 for k, v in repaired.items() if norm(v) != k[1])
    print(f"[repair] done ({time.time()-t0:.0f}s) unique phrasings changed={n_changed_uniq}/{len(uniq)} LLM fired={fired_cnt}", flush=True)

    # 4) write back trials[].pred_sql (still 32 entries) and recompute reward: changed -> recompute;
    #    unchanged -> keep the original label
    n_cand_changed = n_relabel = 0
    keymap = {}                      # repaired phrasing -> pre-repair phrasing (attribution key of the token ledger)
    for it in data:
        db = it["db_id"]; gold = it.get("gold_sql") or it.get("query") or ""
        dbp = B.db_path_of(db)
        rl = it.get("reward_list")
        for i, t in enumerate(it["trials"]):
            s = t.get("pred_sql")
            if not s or not s.strip():
                continue
            k = (db, norm(s))
            rsql = repaired.get(k, s)
            changed = norm(rsql) != norm(s)
            if changed:
                # The token ledger attributes calls to the **pre-repair** phrasing, but the candidate
                # pool is about to be rewritten to the repaired SQL. The cost report joins the ledger
                # against the repaired pool, so without this mapping every rewritten phrasing fails to
                # join -> the per-question repair tokens go missing.
                # A list is used: two different original phrasings can be repaired into the same SQL,
                # and overwriting would lose one of them.
                _rk = f"{db}|||{norm(rsql)}"
                _ov = norm(s)
                _lst = keymap.setdefault(_rk, [])
                if _ov not in _lst:
                    _lst.append(_ov)
            if changed and gold and not args.no_gold:
                n_cand_changed += 1
                new_r = float(reward_label(dbp, rsql, gold)); n_relabel += 1
            else:
                if changed:
                    n_cand_changed += 1
                if rl and i < len(rl) and rl[i] is not None:
                    try:
                        new_r = float(rl[i])
                    except (TypeError, ValueError):
                        new_r = float(t.get("reward") or 0.0)
                else:
                    new_r = float(t.get("reward") or 0.0)
            t["pred_sql"] = rsql
            t["reward"] = new_r
        it["reward_list"] = [float(t.get("reward") or 0.0) for t in it["trials"]]
    print(f"[rewrite] candidates changed={n_cand_changed} rewards recomputed={n_relabel}", flush=True)

    # Save the [repaired phrasing -> pre-repair phrasing] mapping so that tools/profile_report.py can
    # attribute the repair tokens to the right question (otherwise rewritten candidates are missed
    # entirely).
    keymap_path = args.out_file + ".repair_keymap.json"
    try:
        # the value is a list of **pre-repair** phrasings, already prefixed with db, so the report
        # side can use them directly as ledger keys
        json.dump({k: [f"{k.split('|||', 1)[0]}|||{v}" for v in vs] for k, vs in keymap.items()},
                  open(keymap_path, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"[keymap] {len(keymap)} rewrite mappings -> {keymap_path}", flush=True)
    except Exception as e:  # noqa: BLE001  accounting helpers must never affect the main flow
        print(f"[keymap] write failed (does not affect the repair results): {e}", flush=True)

    # 5) build the execution cache for the repaired candidates
    tasks = {}
    for it in data:
        db = it["db_id"]; dbp = B.db_path_of(db)
        for t in it["trials"]:
            s = t.get("pred_sql")
            if not s or not s.strip():
                continue
            key = f"{db}|||{norm(s)}"
            if key not in tasks:
                tasks[key] = (dbp, s)
    print(f"[exec-cache] {len(tasks)} unique statements to execute", flush=True)
    results = {}; t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(BEC.exec_one, dbp, s): key for key, (dbp, s) in tasks.items()}
        for fut in as_completed(futs):
            key = futs[fut]
            try:
                results[key] = fut.result()
            except Exception as e:  # noqa: BLE001
                results[key] = {"ok": False, "error": f"future err: {str(e)[:120]}",
                                "row_count": None, "result_sig": None, "preview": [], "profile": None}
            done += 1
            if done % 2000 == 0:
                print(f"  exec {done}/{len(tasks)} ({time.time()-t0:.0f}s)", flush=True)
    ok = sum(1 for v in results.values() if v["ok"])
    err = len(results) - ok
    empty = sum(1 for v in results.values() if v["ok"] and v.get("row_count") == 0)
    cache_obj = {"meta": {"source_file": args.out_file, "db_dir": B.DB_DIR, "dataset": DATASET,
                          "n_unique": len(tasks), "n_ok": ok, "n_error": err, "n_empty_result": empty,
                          "result_sig": "md5(str(sorted(str(r) for r in rows)))"},
                 "results": results}

    # 6) write out
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_cache), exist_ok=True)
    tmp = args.out_file + ".tmp"
    json.dump(data, open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
    os.replace(tmp, args.out_file)
    print(f"[save] repaired file -> {args.out_file} ({os.path.getsize(args.out_file)/1e6:.1f} MB)", flush=True)
    tmp = args.out_cache + ".tmp"
    json.dump(cache_obj, open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
    os.replace(tmp, args.out_cache)
    print(f"[save] exec cache -> {args.out_cache} (ok={ok} err={err} empty={empty})", flush=True)

    stats = {"dataset": DATASET, "src": src, "out_file": args.out_file, "out_cache": args.out_cache,
             "n_single_cluster": n_single, "n_multi_cluster": n_multi,
             "n_uniq_writings": len(uniq), "n_uniq_changed": n_changed_uniq,
             "n_cand_changed": n_cand_changed, "n_relabel": n_relabel,
             "exec_ok": ok, "exec_err": err, "exec_empty": empty}
    print(f"[stats] {json.dumps(stats, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()
