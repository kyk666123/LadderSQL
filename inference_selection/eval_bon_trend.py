"""best-of-N 趋势编排 + 报告。

对同一模型的 32 采样池, 画出 N ∈ {1(greedy),8,16,24,32} 的:
  - 效果: best-of-N EX (现有多代表锦标赛+加性选择, 扫 variant/K/k 取峰值)、
           多数投票基线、pass@N 天花板;
  - 成本(每题累计, 均值+中位数): 前 N 条候选的 输入 token / 输出 token / LLM 耗时 / 轮数;
  - 判别侧: 该 N 消耗的成对裁决条数。

效果侧复用 eval_multirep 的 pair_p/js_scores/evaluate 逻辑 (此处内联, 便于按 N 参数化);
成本侧来自 token_cost 的 sidecar (候选池)。

产出: <out_dir>/bon_trend_<tag>.{md,csv,json}

用法(通常由 run_bon_trend.sh 调用):
  BIRD_FILE=<candidates.json> EXEC_CACHE=<exec_cache.json> \
  python eval_bon_trend.py --tag step192 --src <原始32池.json> --greedy <greedy.json> \
      --cache-prefix cache/multirep_pairs_step192_n --ns 8,16,24,32 \
      --tokenizer <hf目录> --out-dir results
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from statistics import median
from typing import Any, Dict, List, Optional

import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import bird_common as B
import gated_common as G
import token_cost as TC

TOPK = 3
RMAX = int(os.environ.get("MULTIREP_RMAX", "3"))


# ============================ 效果侧 (按 N) ============================
def eval_effect_at_n(nval: int, cache_path: str) -> Dict[str, Any]:
    """在 N=nval 口径下: best EX(扫 variant/K/k)、多数投票、pass@N。复用 eval_multirep 逻辑。"""
    os.environ["BON_NMAX"] = str(nval)
    samples = B.load_samples()
    info = G.classify(samples)
    per = info["per_sample"]
    BASE = info["fixed_correct"]
    TOT = info["counts"]["total"]
    contested = info["contested_ids"]

    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

    def writings(sid, c):
        cands = samples[sid]["candidates"]
        sqls = [G._norm(cands[i]) for i in c["members"]]
        cnt = Counter(sqls)
        return sorted(cnt, key=lambda s: (-cnt[s], len(s)))[:RMAX]

    def pair_p(sid, i, j, variant):
        cls = per[sid]["clusters"][:TOPK]
        ni = len(writings(sid, cls[i])); nj = len(writings(sid, cls[j]))
        o0s = []; o1s = []
        for ia in range(ni):
            for ib in range(nj):
                v0 = cache.get(f"{sid}|{i}|{j}|{ia}|{ib}|0")
                v1 = cache.get(f"{sid}|{i}|{j}|{ia}|{ib}|1")
                if v0 is not None:
                    o0s.append(1.0 if v0 == "A" else (0.0 if v0 == "B" else 0.5))
                if v1 is not None:
                    o1s.append(1.0 if v1 == "A" else (0.0 if v1 == "B" else 0.5))
        o0 = sum(o0s) / len(o0s) if o0s else 0.5
        o1 = sum(o1s) / len(o1s) if o1s else 0.5
        return o0 if variant == "biased" else 0.5 * (o0 + (1.0 - o1))

    def js_scores(sid, K, variant):
        cls = per[sid]["clusters"][:K]
        m = len(cls)
        s = [0.0] * m
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                a, b = (i, j) if i < j else (j, i)
                p = pair_p(sid, a, b, variant)
                pij = p if i < j else (1.0 - p)
                s[i] += pij
        return [x / (m - 1) for x in s] if m > 1 else [1.0]

    def evaluate(K, k, variant):
        cc = 0
        for sid in contested:
            cls = per[sid]["clusters"]; top = cls[:K]; maj_ok = per[sid]["maj_correct"]
            if len(top) == 1:
                ok = maj_ok
            else:
                js = js_scores(sid, K, variant)
                sc = [js[x] + k * (top[x]["n"] / nval) for x in range(len(top))]
                ci = max(range(len(top)), key=lambda x: (sc[x], top[x]["n"]))
                ok = bool(top[ci]["correct"])
            cc += 1 if ok else 0
        return 100.0 * (BASE + cc) / TOT

    best = 0.0; best_cfg = None
    for variant in ("biased", "debiased"):
        for K in (2, 3):
            for ki in range(0, 201, 2):
                ex = evaluate(K, ki / 100.0, variant)
                if ex > best:
                    best = ex; best_cfg = (variant, K, round(ki / 100.0, 2))
    maj = 100.0 * (BASE + sum(1 for sid in contested if per[sid]["maj_correct"])) / TOT
    ceil = 100.0 * (BASE + len(contested)) / TOT
    n_verdicts = len(cache)
    return {"best_ex": round(best, 2), "best_cfg": best_cfg,
            "majority": round(maj, 2), "pass_at_n": round(ceil, 2),
            "fixed_correct": BASE, "contested": len(contested), "total": TOT,
            "pair_verdicts": n_verdicts}


def eval_greedy(greedy_file: str) -> Dict[str, Any]:
    with open(greedy_file) as f:
        raw = json.load(f)
    tot = len(raw); ok = 0
    for it in raw:
        rl = it.get("reward_list")
        r = None
        if rl:
            r = rl[0]
        elif it.get("trials"):
            r = it["trials"][0].get("reward")
        try:
            if float(r) == 1.0:
                ok += 1
        except (TypeError, ValueError):
            pass
    return {"best_ex": round(100.0 * ok / tot, 2) if tot else 0.0,
            "majority": round(100.0 * ok / tot, 2) if tot else 0.0,
            "pass_at_n": round(100.0 * ok / tot, 2) if tot else 0.0,
            "total": tot, "contested": 0, "pair_verdicts": 0, "best_cfg": None}


# ============================ 成本侧 (按 N) ============================
def cost_at_n(costs: List[Dict[str, Any]], nval: int) -> Dict[str, Any]:
    """每题取前 nval 条候选, 累加 in/out token、llm_time、turns; 返回均值+中位数+总量。"""
    per_in = []; per_out = []; per_time = []; per_turns = []
    in_src = Counter()
    for item in costs:
        tr = item.get("trials", [])[:nval]
        per_in.append(sum(t["in_tok"] for t in tr))
        per_out.append(sum(t["out_tok"] for t in tr))
        per_time.append(sum(t["llm_time"] for t in tr))
        per_turns.append(sum(t["turns"] for t in tr))
        for t in tr:
            in_src[t.get("in_src", "?")] += 1

    def stat(xs):
        return {"mean": round(sum(xs) / len(xs), 1) if xs else 0.0,
                "median": round(median(xs), 1) if xs else 0.0,
                "total": sum(xs)}
    return {"in_tok": stat(per_in), "out_tok": stat(per_out),
            "llm_time": stat(per_time), "turns": stat(per_turns),
            "in_tok_src": dict(in_src)}


# ============================ 报告 ============================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--src", required=True, help="原始 32 采样池 (成本来源)")
    ap.add_argument("--greedy", default=None, help="greedy(N=1) 采样文件")
    ap.add_argument("--cache-prefix", required=True,
                    help="per-N 下标缓存前缀, 实际路径= <prefix><N>.json")
    ap.add_argument("--ns", default="8,16,24,32")
    ap.add_argument("--tokenizer", default=os.environ.get("TOKENIZER_DIR"))
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    ns = [int(x) for x in args.ns.split(",") if x.strip()]
    costs = TC.compute_or_load(args.src, args.tokenizer)

    rows: List[Dict[str, Any]] = []
    if args.greedy and os.path.exists(args.greedy):
        gcost = TC.compute_or_load(args.greedy, args.tokenizer)
        eff = eval_greedy(args.greedy)
        c = cost_at_n(gcost, 1)
        rows.append({"N": 1, "label": "greedy", **eff, "cost": c})

    for nval in ns:
        cache_path = f"{args.cache_prefix}{nval}.json"
        eff = eval_effect_at_n(nval, cache_path)
        c = cost_at_n(costs, nval)
        rows.append({"N": nval, "label": f"bo{nval}", **eff, "cost": c})

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, f"bon_trend_{args.tag}")
    with open(base + ".json", "w") as f:
        json.dump({"tag": args.tag, "src": args.src, "rows": rows}, f, indent=2, ensure_ascii=False)

    # CSV + Markdown
    hdr = ["N", "label", "best_ex", "majority", "pass_at_n", "best_cfg",
           "in_tok_mean", "in_tok_median", "out_tok_mean", "out_tok_median",
           "llm_time_mean", "turns_mean", "pair_verdicts", "contested"]
    csv_lines = [",".join(hdr)]
    md_lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in rows:
        c = r["cost"]
        vals = [r["N"], r["label"], r.get("best_ex"), r.get("majority"), r.get("pass_at_n"),
                (str(r.get("best_cfg")) if r.get("best_cfg") else "-"),
                c["in_tok"]["mean"], c["in_tok"]["median"],
                c["out_tok"]["mean"], c["out_tok"]["median"],
                c["llm_time"]["mean"], c["turns"]["mean"],
                r.get("pair_verdicts", 0), r.get("contested", 0)]
        csv_lines.append(",".join(str(v) for v in vals))
        md_lines.append("| " + " | ".join(str(v) for v in vals) + " |")
    with open(base + ".csv", "w") as f:
        f.write("\n".join(csv_lines) + "\n")
    with open(base + ".md", "w") as f:
        f.write(f"# BIRD old dev best-of-N 趋势 · {args.tag}\n\n")
        f.write("成本=每题累计(前 N 条候选之和); in_tok 对无 prompt_tokens 的旧池为轨迹估算。\n\n")
        f.write("\n".join(md_lines) + "\n")
    print("\n".join(md_lines))
    print(f"\n[report] -> {base}.md / .csv / .json")


if __name__ == "__main__":
    main()
