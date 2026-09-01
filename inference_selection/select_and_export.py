"""Eq.(11) final selection + export of the BIRD submission file (runs without any gold SQL).

Inputs:
  BIRD_FILE      candidate pool JSON (the repaired one), trials[].pred_sql; gold may be absent
  EXEC_CACHE     execution-result cache for that candidate pool
  MULTIREP_CACHE verdict cache (output of run_multirep_tournament.py or tournament_blind.py)

Per-question decision:
  1. Cluster by execution-result signature (Eq. 9); with a single cluster, emit that cluster's
     phrasing directly (no LLM call).
  2. With >=2 clusters, take the topk largest clusters, aggregate the cached verdicts into pairwise
     win rates p_ij (Eq. 10), compute the round-robin mean win rate js_i, then
     score_i = js_i + k * n_i / N (Eq. 11), take the argmax, breaking ties toward the larger cluster.
  3. A pair with no verdict gets p=0.5 (degenerating to majority voting), and the number of such
     questions is reported in the statistics.

Outputs:
  --out-selected  rich JSON (per question: chosen SQL, cluster structure, decision source, scores)
  --out-predict   official BIRD evaluation format {"0": "SQL\\t----- bird -----\\tdb_id", ...}
  --gold-check    if the candidate pool carries reward_list/trials[].reward, additionally print the
                  EX of this selection (dev self-check only; there is no gold at test submission
                  time, so do not use it there)

The default hyper-parameters are the values locked in by the BIRD-dev sweep:
variant=biased, topk=3, k=1.02, rmax=3, N=32.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bird_common as B          # noqa: E402
import gated_common as G         # noqa: E402


def writings(sample, cluster, rmax):
    """At most rmax most frequent phrasings inside a cluster (exactly the same rule as the
    tournament script)."""
    sqls = [G._norm(sample["candidates"][i]) for i in cluster["members"]]
    cnt = Counter(sqls)
    return sorted(cnt, key=lambda s: (-cnt[s], len(s)))[:rmax]


def pair_p(cache, sid, i, j, nw_i, nw_j, variant):
    """Average the rmax x rmax x two-direction verdicts of one cluster pair into
    p(cluster i beats cluster j). No verdict -> 0.5."""
    o0s, o1s = [], []
    for ia in range(nw_i):
        for ib in range(nw_j):
            v0 = cache.get(f"{sid}|{i}|{j}|{ia}|{ib}|0")
            v1 = cache.get(f"{sid}|{i}|{j}|{ia}|{ib}|1")
            if v0 is not None:
                o0s.append(1.0 if v0 == "A" else (0.0 if v0 == "B" else 0.5))
            if v1 is not None:
                o1s.append(1.0 if v1 == "A" else (0.0 if v1 == "B" else 0.5))
    missing = not o0s and not o1s
    o0 = sum(o0s) / len(o0s) if o0s else 0.5      # P(i wins | i in position A)
    o1 = sum(o1s) / len(o1s) if o1s else 0.5      # P(j wins | j in position A)
    p = o0 if variant == "biased" else 0.5 * (o0 + (1.0 - o1))
    return p, missing


def js_scores(cache, sid, cls, nws, variant):
    """round-robin: js_i = mean win rate of cluster i against the other competing clusters.
    Returns (js, any_missing)."""
    m = len(cls)
    if m <= 1:
        return [1.0], False
    s = [0.0] * m
    any_missing = False
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            p, missing = pair_p(cache, sid, a, b, nws[a], nws[b], variant)
            any_missing = any_missing or missing
            s[i] += p if i < j else (1.0 - p)
    return [x / (m - 1) for x in s], any_missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--multirep-cache", default=os.environ.get("MULTIREP_CACHE", ""),
                    help="verdict cache; without it this degenerates to plain majority voting")
    ap.add_argument("--variant", default="biased", choices=["biased", "debiased"])
    ap.add_argument("--topk", type=int, default=3, help="number of competing clusters K")
    ap.add_argument("--k", type=float, default=1.02, help="weight of the consistency prior k")
    ap.add_argument("--rmax", type=int, default=3, help="representative phrasings per cluster")
    ap.add_argument("--n", type=int, default=32, help="denominator N of the consistency prior = candidate pool size")
    ap.add_argument("--out-selected", default="results/selected.json")
    ap.add_argument("--out-predict", default="results/predict.json")
    ap.add_argument("--gold-check", action="store_true", help="print the EX of this selection when gold is available (dev self-check)")
    args = ap.parse_args()

    cache = {}
    if args.multirep_cache and os.path.exists(args.multirep_cache):
        cache = json.load(open(args.multirep_cache))
    print(f"[cfg] variant={args.variant} topk={args.topk} k={args.k} rmax={args.rmax} N={args.n}")
    print(f"[cache] {len(cache)} verdict <- {args.multirep_cache or '(none, plain majority voting)'}")

    samples = B.load_samples()
    selected, predict = [], {}
    n_single = n_tour = n_overturn = n_missing = n_empty = 0
    ex_hit = ex_total = 0

    for sid, rec in enumerate(samples):
        cls = G.build_clusters(rec)
        if not cls:
            # all 32 candidates empty / unparseable: fall back to any non-empty original, and finally
            # to "SELECT 1" (so that the submission file never misses a key)
            fallback = next((c for c in rec["candidates"] if c and c.strip()), "SELECT 1")
            selected.append({"index": sid, "db_id": rec["db_id"], "question": rec["question"],
                             "sql": G._norm(fallback), "decided_by": "empty_fallback",
                             "n_clusters": 0, "cluster_size": 0})
            predict[str(sid)] = f"{G._norm(fallback)}\t----- bird -----\t{rec['db_id']}"
            n_empty += 1
            continue

        top = cls[:args.topk]
        nws = [len(writings(rec, c, args.rmax)) for c in top]
        if len(top) == 1:
            ci, decided, score_dump, missing = 0, "single_cluster", None, False
            n_single += 1
        else:
            js, missing = js_scores(cache, sid, top, nws, args.variant)
            score = [js[x] + args.k * (top[x]["n"] / args.n) for x in range(len(top))]
            ci = max(range(len(top)), key=lambda x: (score[x], top[x]["n"]))
            decided = "tournament"
            score_dump = [{"cluster": x, "n": top[x]["n"], "js": round(js[x], 4),
                           "score": round(score[x], 4)} for x in range(len(top))]
            n_tour += 1
            if ci != 0:
                n_overturn += 1           # overturned the largest cluster (majority vote)
            if missing:
                n_missing += 1

        chosen_cluster = top[ci]
        sql = writings(rec, chosen_cluster, args.rmax)[0]      # most frequent phrasing in the cluster
        selected.append({"index": sid, "db_id": rec["db_id"], "question": rec["question"],
                         "sql": sql, "decided_by": decided,
                         "n_clusters": len(cls), "cluster_size": chosen_cluster["n"],
                         "chosen_rank": ci, "verdicts_missing": bool(missing),
                         "scores": score_dump})
        predict[str(sid)] = f"{sql}\t----- bird -----\t{rec['db_id']}"

        if args.gold_check:
            ex_total += 1
            ex_hit += 1 if chosen_cluster["correct"] else 0

    for p in (args.out_selected, args.out_predict):
        d = os.path.dirname(os.path.abspath(p))
        os.makedirs(d, exist_ok=True)
    json.dump(selected, open(args.out_selected, "w"), ensure_ascii=False, indent=1)
    json.dump(predict, open(args.out_predict, "w"), ensure_ascii=False, indent=1)

    print(f"[stat] questions={len(samples)}  single-result direct={n_single}  tournament={n_tour}  "
          f"of which overturned the largest cluster={n_overturn}  missing verdicts -> majority vote={n_missing}  empty-candidate fallback={n_empty}")
    print(f"[save] {args.out_selected}")
    print(f"[save] {args.out_predict}   (BIRD submission format)")
    if args.gold_check and ex_total:
        print(f"[gold-check] EX = {ex_hit}/{ex_total} = {100.0*ex_hit/ex_total:.2f}%")


if __name__ == "__main__":
    main()
