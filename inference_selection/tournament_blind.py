"""Multi-representative tournament -- the **blind, full-scope** version, for the deployment /
submission path where no gold SQL exists.

The only difference from the paper script run_multirep_tournament.py is the **scope**:

  run_multirep_tournament.py : adjudicates only the contested questions (>=2 clusters where some
                               clusters are right and some wrong -- which requires gold to know).
                               That is a cost saving that only works **offline**: for a
                               non-contested question every choice gives the same outcome
                               (all right / all wrong), so skipping it cannot change EX.
  tournament_blind.py        : adjudicates **every** question with >=2 clusters -- at deployment
                               time gold is invisible, so we cannot know in advance which
                               questions are contested and must judge them all.

Because judging and clustering are reward-blind throughout, the two produce the **identical final
EX** on dev; only the number of LLM calls differs (BIRD old-dev: 423 contested questions vs 621
questions with more than one result).

The verdict-cache key is exactly the same as in the paper script (f"{sid}|{i}|{j}|{ia}|{ib}|{order}"),
so the cache files of the two are interchangeable and can be resumed from either side.

Environment variables (same names and meanings as in the paper script):
  BIRD_FILE        candidate pool JSON (trials[].pred_sql; gold may be absent)
  EXEC_CACHE       execution-result cache for the candidates (output of build_exec_cache_generic.py)
  BON_DB_DIR       sqlite database root (must be a local disk)
  MULTIREP_CACHE   verdict cache output path (resumable)
  MULTIREP_RMAX    representative phrasings per cluster, default 3 (locked value)
  MULTIREP_TOPK    clusters entering the tournament, default 3 (locked value)
  SCHEMA_CACHE     linked-schema cache {"db_id|||question": "<schema text>"} (strongly recommended)
  MULTIREP_MODEL   judge model, default glm-5.2
  MULTIREP_WORKERS concurrent adjudications, default 24
  JUDGE_API_KEY / JUDGE_API_BASE  judge credentials (OpenAI-compatible endpoint)
"""
import functools
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

print = functools.partial(print, flush=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bird_common as B          # noqa: E402
import gated_common as G         # noqa: E402
import run_gated_reasoning_judge as R  # noqa: E402

R.USE_SIMPLICITY = True          # anti-over-interpretation prior (locked on)

TOPK = int(os.environ.get("MULTIREP_TOPK", "3"))
RMAX = int(os.environ.get("MULTIREP_RMAX", "3"))
WORKERS = int(os.environ.get("MULTIREP_WORKERS", "24"))
CACHE = os.environ.get("MULTIREP_CACHE", "cache/multirep_pairs_blind.json")
MODEL = os.environ.get("MULTIREP_MODEL", "glm-5.2")
THINK = bool(os.environ.get("MULTIREP_THINK"))

_SCHEMA_CACHE = os.environ.get("SCHEMA_CACHE", "")
if _SCHEMA_CACHE and os.path.exists(_SCHEMA_CACHE):
    R.SCHEMA_LUT.update(json.load(open(_SCHEMA_CACHE)))
    print(f"[schema] linked schema <- {_SCHEMA_CACHE} ({len(R.SCHEMA_LUT)} entries)")
else:
    print(f"[schema] no linked schema ({_SCHEMA_CACHE or 'unset'}), falling back to DB DDL + column descriptions")
print(f"[model] judge = {MODEL}  thinking={THINK}  TOPK={TOPK} RMAX={RMAX}")

samples = B.load_samples()
clusters = {sid: G.build_clusters(rec) for sid, rec in enumerate(samples)}
scope = [sid for sid, cls in clusters.items() if len(cls) >= 2]
print(f"[scope] questions={len(samples)}  single-result (no adjudication)={len(samples)-len(scope)}  multi-result (to adjudicate)={len(scope)}")


def writings(sid, c):
    """Take at most RMAX most frequent phrasings inside a cluster (exactly the ordering rule used
    by the eval / paper scripts)."""
    cands = samples[sid]["candidates"]
    sqls = [G._norm(cands[i]) for i in c["members"]]
    cnt = Counter(sqls)
    return sorted(cnt, key=lambda s: (-cnt[s], len(s)))[:RMAX]


cache = {}
if os.path.exists(CACHE):
    cache = json.load(open(CACHE))
    print(f"[cache] loaded {len(cache)} verdicts already computed")

exec_cache = {}
prev_cache = {}


def gx(db, sig, sql):
    if sig not in exec_cache:
        exec_cache[sig] = R.exec_full(db, sql)
    return exec_cache[sig]


def gp(db, sql):
    if sql not in prev_cache:
        prev_cache[sql] = B.cached_preview(db, sql)
    return prev_cache[sql]


def save_cache():
    """Atomically persist the verdict cache.

    A plain open(CACHE,"w") truncates before writing -- being killed halfway (or an OOM /
    network restart) leaves half a json behind, json.load then raises on resume, and the few
    thousand verdicts we already paid for are gone.
    In isolate mode we save after every question (to save promptly), so the exposure window
    occurs thousands of times and the write must be atomic.
    """
    tmp = CACHE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, CACHE)


jobs = []
meta = {}
for sid in scope:
    cls = clusters[sid][:TOPK]
    db = samples[sid]["db_id"]
    w = [writings(sid, c) for c in cls]
    meta[sid] = {"db": db, "w": w, "sig": [c["sig"] for c in cls], "m": len(cls)}
    for i in range(len(cls)):
        for j in range(i + 1, len(cls)):
            for ia in range(len(w[i])):
                for ib in range(len(w[j])):
                    for order in (0, 1):
                        k = f"{sid}|{i}|{j}|{ia}|{ib}|{order}"
                        if k not in cache:
                            jobs.append((sid, i, j, ia, ib, order))
print(f"[jobs] {len(jobs)} adjudications to run ({len(cache)} cached), over {len(meta)} questions")


def work(job):
    sid, i, j, ia, ib, order = job
    try:
        import llm_usage
        rec = samples[sid]
        llm_usage.set_unit(f"{rec['db_id']}|||{rec.get('question','')}")  # question key, consistent across stages
    except Exception:  # noqa: BLE001
        pass
    m = meta[sid]
    db = m["db"]
    sa, sb = m["w"][i][ia], m["w"][j][ib]
    exa, exb = gx(db, m["sig"][i], sa), gx(db, m["sig"][j], sb)
    ca = {"rep_sql": sa, "preview": gp(db, sa)}
    cb = {"rep_sql": sb, "preview": gp(db, sb)}
    if order == 0:
        prompt = R.build_prompt(samples[sid], ca, cb, R.struct_facts(exa, exb))
    else:
        prompt = R.build_prompt(samples[sid], cb, ca, R.struct_facts(exb, exa))
    v = R.call_verdict(prompt, MODEL, THINK)
    return f"{sid}|{i}|{j}|{ia}|{ib}|{order}", (v or "T")


if jobs:
    os.makedirs(os.path.dirname(os.path.abspath(CACHE)), exist_ok=True)
    t0 = time.time()
    done = 0
    ISOLATE = os.environ.get("ISOLATE_SAMPLES", "0").lower() in ("1", "true", "yes")
    SAMPLE_CONC = int(os.environ.get("JUDGE_SAMPLE_CONC", "0") or 0)
    # Two measurement regimes; JUDGE_SAMPLE_CONC wins over the global ISOLATE_SAMPLES:
    #   ISOLATE_SAMPLES=1 with JUDGE_SAMPLE_CONC unset -> questions serial, adjudications inside a
    #     question concurrent (same regime as module 2; this is the default for cost measurement --
    #     "all adjudications of one question run at the same time")
    #   JUDGE_SAMPLE_CONC=N -> N questions in flight, adjudications inside a question serial
    #     (total in-flight calls is exactly N)
    # run_test.sh exports ISOLATE_SAMPLES=1 globally for the generation module.
    if ISOLATE and SAMPLE_CONC <= 0:
        # Isolate mode (the regime used for cost measurement): questions are processed in order and
        # the pairwise adjudications of one question run concurrently (<=min(32,WORKERS)), so the
        # per-question judge wall clock is clean. Same regime as module 2 (SQL generation).
        #   Note: the number of adjudications for one question = C(#clusters,2) x #phrasings^2 x 2,
        #   which in practice is between 2 and 54, and most questions have fewer than 32. So the 32
        #   here is a **concurrency cap**, not 32 truly parallel calls; the actual in-flight count
        #   per question is the `calls` field in the wall file. Do not report a blanket "32-way
        #   concurrency".
        from collections import defaultdict as _dd
        jobs_by_sid = _dd(list)
        for jb in jobs:
            jobs_by_sid[jb[0]].append(jb)
        iso_conc = min(32, WORKERS)
        wall_path = os.path.splitext(CACHE)[0] + "_sample_wall.jsonl"
        wf = open(wall_path, "a", encoding="utf-8")
        print(f"[isolate] questions in order, per-question concurrency cap {iso_conc}; per-question wall clock -> {wall_path}")
        if iso_conc < 32:
            print(f"[isolate][WARN] in-question concurrency cap is only {iso_conc} (limited by "
                  f"MULTIREP_WORKERS={WORKERS}), not the intended 32. Please export MULTIREP_WORKERS=32.")
        for sid in [s for s in scope if s in jobs_by_sid]:
            sjobs = jobs_by_sid[sid]
            rec = samples[sid]
            ukey = f"{rec['db_id']}|||{rec.get('question','')}"
            ts = time.time()
            # Multiple threads on one question would hit the same exec/preview cache misses at the
            # same time (a question has only <=TOPK result signatures and <=TOPK*RMAX phrasings) and
            # each would re-execute the same SQL -- wasting CPU and mixing sqlite execution time into
            # this question's judge wall clock. Fill the caches serially first, so the concurrent
            # section is pure API calls.
            # (The warm-up time counts inside this question's wall clock -- it is work this question
            # genuinely needs.)
            _m = meta[sid]
            for _i in range(_m["m"]):
                if not _m["w"][_i]:
                    continue
                gx(_m["db"], _m["sig"][_i], _m["w"][_i][0])   # cache key is the sig, once per cluster is enough
                for _s in _m["w"][_i]:
                    gp(_m["db"], _s)                          # cache key is the sql, per phrasing
            with ThreadPoolExecutor(max_workers=iso_conc) as ex:
                for k, v in ex.map(work, sjobs):
                    cache[k] = v; done += 1
            wf.write(json.dumps({"unit": ukey, "stage": "judge", "wall_sec": round(time.time() - ts, 3),
                                 "calls": len(sjobs)}) + "\n"); wf.flush()
            save_cache()
        wf.close()
        print(f"[done] isolate mode finished in {time.time()-t0:.0f}s -> {CACHE}")
    elif SAMPLE_CONC > 0:
        # Sample-queue mode (recommended for cost measurement): SAMPLE_CONC questions are always in
        # flight and the adjudications inside a question are serial.
        #   Why:
        #     - total in-flight API calls == SAMPLE_CONC, matching the concurrency definition of the
        #       other modules (no 32x32 oversubscription);
        #     - per-question wall clock = the serial time of all adjudications of that question,
        #       a clean definition that excludes cross-question queueing;
        #     - set_unit() inside work() is thread-local and each question gets one thread, so token
        #       attribution is correct.
        #   Difference from ISOLATE: ISOLATE is serial across questions (N times slower); here
        #   questions run concurrently.
        from collections import defaultdict as _dd
        jobs_by_sid = _dd(list)
        for jb in jobs:
            jobs_by_sid[jb[0]].append(jb)
        sids = [s for s in scope if s in jobs_by_sid]
        wall_path = os.path.splitext(CACHE)[0] + "_sample_wall.jsonl"
        wf = open(wall_path, "a", encoding="utf-8")
        print(f"[sampleq] {SAMPLE_CONC} questions in flight (serial within a question), {len(sids)} questions total; "
              f"per-question wall clock -> {wall_path}")

        def do_sid(sid):
            rec = samples[sid]
            ukey = f"{rec['db_id']}|||{rec.get('question','')}"
            ts = time.time()
            out = [work(jb) for jb in jobs_by_sid[sid]]
            return sid, ukey, round(time.time() - ts, 3), out

        # ex.map results are consumed one by one on the main thread -> cache/wf/done are all written
        # single-threaded, so no lock is needed
        with ThreadPoolExecutor(max_workers=SAMPLE_CONC) as ex:
            nq = 0
            for sid, ukey, wall, out in ex.map(do_sid, sids):
                for k, v in out:
                    cache[k] = v
                    done += 1
                wf.write(json.dumps({"unit": ukey, "stage": "judge", "wall_sec": wall,
                                     "calls": len(out)}) + "\n")
                wf.flush()
                nq += 1
                if nq % 50 == 0:
                    save_cache()
                    print(f"  question {nq}/{len(sids)}  adjudications {done}/{len(jobs)} "
                          f"({time.time()-t0:.0f}s) saved")
        wf.close()
        save_cache()
        print(f"[done] sample-queue mode finished in {time.time()-t0:.0f}s -> {CACHE}")
    else:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for k, v in ex.map(work, jobs):
                cache[k] = v
                done += 1
                if done % 1000 == 0:
                    save_cache()
                    print(f"  {done}/{len(jobs)} ({time.time()-t0:.0f}s) saved")
        save_cache()
        print(f"[done] all finished in {time.time()-t0:.0f}s -> {CACHE}")
else:
    print("[done] cache is already complete, you can run select_and_export.py directly")
