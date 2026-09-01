"""评估多代表锦标赛: 读 multirep_pairs.json, 把每对簇的 R×R×双向 verdict
聚合成稳健成对胜率 p_ij, 再算 js + 动态推翻, 扫 (K,k,变体) 出最优 EX。
锁定最优: biased K=3 k=1.0 RMAX=3 -> 73.47% (对照单代表 73.08%)。
RMAX 用环境变量 MULTIREP_RMAX 覆盖 (与生成脚本保持一致)。"""
import json, os, sys
from collections import Counter
import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import bird_common as B, gated_common as G

CACHE = os.environ.get("MULTIREP_CACHE", "cache/multirep_pairs.json")
N = int(os.environ.get("BON_N", "32"))   # 一致性先验分母=候选池大小(best-of-8 传 BON_N=8)
TOPK = 3
RMAX = int(os.environ.get("MULTIREP_RMAX", "3"))
cache = json.load(open(CACHE))
samples = B.load_samples(); info = G.classify(samples); per = info["per_sample"]
BASE = info["fixed_correct"]; TOT = info["counts"]["total"]

def writings(sid, c):
    cands = samples[sid]["candidates"]
    sqls = [G._norm(cands[i]) for i in c["members"]]
    cnt = Counter(sqls)
    return sorted(cnt, key=lambda s: (-cnt[s], len(s)))[:RMAX]

def v2s(v, isA):
    """verdict -> A侧胜率分量: A=>1, B=>0, T=>0.5"""
    if v == "A": return 1.0
    if v == "B": return 0.0
    return 0.5

def pair_p(sid, i, j, variant):
    """聚合 R×R×双向: 返回 p(簇i 胜 簇j)。
    order0: A=i,B=j -> o0=P(i胜); order1: A=j,B=i -> verdictA=>j胜 => 对i是(1-那个)"""
    cls = per[sid]["clusters"][:TOPK]
    ni = len(writings(sid, cls[i])); nj = len(writings(sid, cls[j]))
    o0s = []; o1s = []
    for ia in range(ni):
        for ib in range(nj):
            v0 = cache.get(f"{sid}|{i}|{j}|{ia}|{ib}|0")
            v1 = cache.get(f"{sid}|{i}|{j}|{ia}|{ib}|1")
            if v0 is not None: o0s.append(1.0 if v0=="A" else (0.0 if v0=="B" else 0.5))
            if v1 is not None: o1s.append(1.0 if v1=="A" else (0.0 if v1=="B" else 0.5))
    o0 = sum(o0s)/len(o0s) if o0s else 0.5   # P(i胜 | i在A位)
    o1 = sum(o1s)/len(o1s) if o1s else 0.5   # P(j胜 | j在A位)
    if variant == "biased":
        return o0
    return 0.5*(o0 + (1.0 - o1))              # debiased

def js_scores(sid, K, variant):
    cls = per[sid]["clusters"][:K]
    m = len(cls)
    s = [0.0]*m
    for i in range(m):
        for j in range(m):
            if i == j: continue
            a, b = (i, j) if i < j else (j, i)
            p = pair_p(sid, a, b, variant)
            pij = p if i < j else (1.0 - p)
            s[i] += pij
    return [x/(m-1) for x in s] if m > 1 else [1.0]

def evaluate(K, k, variant):
    cc = resc = lost = 0
    for sid in info["contested_ids"]:
        cls = per[sid]["clusters"]; top = cls[:K]; maj_ok = per[sid]["maj_correct"]
        if len(top) == 1:
            ok = maj_ok
        else:
            js = js_scores(sid, K, variant)
            sc = [js[x] + k*(top[x]["n"]/N) for x in range(len(top))]
            ci = max(range(len(top)), key=lambda x: (sc[x], top[x]["n"]))
            ok = bool(top[ci]["correct"])
        cc += 1 if ok else 0
        if ok and not maj_ok: resc += 1
        elif maj_ok and not ok: lost += 1
    return 100*(BASE+cc)/TOT, resc, lost

print(f"[cache] {len(cache)} verdict")
best = (0,)
for variant in ("biased", "debiased"):
    vb = (0,)
    for K in (2, 3):
        for ki in range(0, 201, 2):
            ex, r, l = evaluate(K, ki/100, variant)
            if ex > vb[0]: vb = (ex, K, round(ki/100,2), r, l)
    print(f"多代表 {variant:9s}: 最优 EX={vb[0]:.2f}%  K={vb[1]} k={vb[2]} 救回{vb[3]}/亏损{vb[4]}")
    if vb[0] > best[0]: best = vb + (variant,)
print(f"\n>>> 多代表最优 {best[0]:.2f}%   对照单代表 73.08%")
