"""在 RMAX=3 缓存上试频次加权聚合 (免API): 组合(ia,ib)权重=freq_ia*freq_ib,
典型写法权重大、罕见写法权重小。对照均匀平均 73.47%。"""
import json, sys
from collections import Counter
import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import bird_common as B, gated_common as G

cache = json.load(open("cache/multirep_pairs.json"))
N = 32; TOPK = 3; RMAX = 3
samples = B.load_samples(); info = G.classify(samples); per = info["per_sample"]
BASE = info["fixed_correct"]; TOT = info["counts"]["total"]

def writings_w(sid, c):
    """返回 [(sql, freq)]，前 RMAX 种。"""
    sqls = [G._norm(samples[sid]["candidates"][i]) for i in c["members"]]
    cnt = Counter(sqls)
    uniq = sorted(cnt, key=lambda s: (-cnt[s], len(s)))[:RMAX]
    return [(s, cnt[s]) for s in uniq]

def pair_p(sid, i, j, variant, weighted):
    cls = per[sid]["clusters"][:TOPK]
    wi = writings_w(sid, cls[i]); wj = writings_w(sid, cls[j])
    num0 = den0 = num1 = den1 = 0.0
    for ia, (_, fi) in enumerate(wi):
        for ib, (_, fj) in enumerate(wj):
            w = (fi * fj) if weighted else 1.0
            v0 = cache.get(f"{sid}|{i}|{j}|{ia}|{ib}|0")
            v1 = cache.get(f"{sid}|{i}|{j}|{ia}|{ib}|1")
            if v0 is not None:
                num0 += w * (1.0 if v0=="A" else (0.0 if v0=="B" else 0.5)); den0 += w
            if v1 is not None:
                num1 += w * (1.0 if v1=="A" else (0.0 if v1=="B" else 0.5)); den1 += w
    o0 = num0/den0 if den0 else 0.5
    o1 = num1/den1 if den1 else 0.5
    return o0 if variant == "biased" else 0.5*(o0 + (1.0 - o1))

def js_scores(sid, K, variant, weighted):
    cls = per[sid]["clusters"][:K]; m = len(cls); s = [0.0]*m
    for i in range(m):
        for j in range(m):
            if i == j: continue
            a, b = (i, j) if i < j else (j, i)
            p = pair_p(sid, a, b, variant, weighted)
            s[i] += p if i < j else (1.0 - p)
    return [x/(m-1) for x in s] if m > 1 else [1.0]

def evaluate(K, k, variant, weighted):
    cc = resc = lost = 0
    for sid in info["contested_ids"]:
        top = per[sid]["clusters"][:K]; maj_ok = per[sid]["maj_correct"]
        if len(top) == 1:
            ok = maj_ok
        else:
            js = js_scores(sid, K, variant, weighted)
            sc = [js[x] + k*(top[x]["n"]/N) for x in range(len(top))]
            ci = max(range(len(top)), key=lambda x: (sc[x], top[x]["n"]))
            ok = bool(top[ci]["correct"])
        cc += 1 if ok else 0
        if ok and not maj_ok: resc += 1
        elif maj_ok and not ok: lost += 1
    return 100*(BASE+cc)/TOT, resc, lost

for tag, weighted in [("均匀平均", False), ("频次加权", True)]:
    best = (0,)
    for K in (2, 3):
        for ki in range(0, 201, 2):
            ex, r, l = evaluate(K, ki/100, "biased", weighted)
            if ex > best[0]: best = (ex, K, round(ki/100,2), r, l)
    print(f"{tag}: 最优 EX={best[0]:.2f}%  K={best[1]} k={best[2]} 救回{best[3]}/亏损{best[4]}")
