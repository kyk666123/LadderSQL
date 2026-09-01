"""消融实验：从完整系统(消除多数+消除位置+子句聚焦, 90.2%)出发，
每次只关掉一个组件，量化其价值，并给出具体翻案/翻错的例子。

三个 LOO 实验（w 固定 0.6，其余组件保持完整系统的设置）：
  Full            : debiased 偏好 + 子句聚焦 + count_prior(w=0.6)
  (1) 不消除多数   : 决策纯按 count(=多数投票 w=1)，LLM 不再撬动
  (2) 不消除位置   : 偏好只用单方向 o0(候选 i 固定在 A)，不做 0.5*(o0+1-o1)
  (3) 不聚焦子句   : 用整条 SQL 比较的缓存(--nofocus)
"""
import json
import os
import sys

CACHE = "/path/to/LadderSQL/inference_selection/cache/candidate_labels.json"
RAW_FOCUS = "/path/to/LadderSQL/inference_selection/cache/bon_pairwise_raw_glm52_think.json"
RAW_NOFOCUS = "/path/to/LadderSQL/inference_selection/cache/bon_pairwise_raw_glm52_think_nofocus.json"

W = 0.6
data = json.load(open(CACHE))
total = len(data)


def sig_clusters(cands):
    g = {}
    for i, c in enumerate(cands):
        if c["sig"] is None:
            continue
        g.setdefault(c["sig"], []).append(i)
    return g


# 单簇固定正确数（所有策略共享）
single_ok = 0
base = 0
for r in data:
    g = sig_clusters(r["candidates"])
    if not g:
        continue
    best = max(g.values(), key=len)
    if r["candidates"][best[0]]["correct"]:
        base += 1
    if len(g) == 1 and r["candidates"][best[0]]["correct"]:
        single_ok += 1


def pair_matrix(rec, pref="debiased"):
    m = len(rec["clusters"])
    P = [[0.5] * m for _ in range(m)]
    for pp in rec["pair_prefs"]:
        i, j = pp["i"], pp["j"]
        if pref == "debiased":
            p = pp["p_ij"]
        elif pref == "single":          # 只用 i 在 A 位置的方向，不去偏
            p = pp["o0"]
        else:
            raise ValueError(pref)
        P[i][j] = p
        P[j][i] = 1.0 - p
    return P


def scores(rec, pref="debiased"):
    m = len(rec["clusters"])
    P = pair_matrix(rec, pref)
    return [sum(P[i][j] for j in range(m) if j != i) / (m - 1) for i in range(m)]


def pick(rec, pref="debiased", w=W, majority_only=False):
    cl = rec["clusters"]
    if majority_only:
        return 0  # clusters 已按 size 降序
    s = scores(rec, pref)
    tot = sum(c["size"] for c in cl)
    final = [(1 - w) * s[i] + w * (cl[i]["size"] / tot) for i in range(len(cl))]
    return max(range(len(cl)), key=lambda i: (final[i], -i))


def evaluate(raw, pref="debiased", w=W, majority_only=False):
    multi_ok = rec_cnt = brk = 0
    flips = []  # (rec, picked_idx)
    for rec in raw:
        cl = rec["clusters"]
        maj_ok = cl[0]["correct"]
        d = pick(rec, pref, w, majority_only)
        sel_ok = cl[d]["correct"]
        multi_ok += sel_ok
        if sel_ok and not maj_ok:
            rec_cnt += 1
        if maj_ok and not sel_ok:
            brk += 1
        flips.append((rec, d))
    acc = (single_ok + multi_ok) / total
    return acc, rec_cnt, brk, flips


raw_focus = json.load(open(RAW_FOCUS))
have_nofocus = os.path.exists(RAW_NOFOCUS)
raw_nofocus = json.load(open(RAW_NOFOCUS)) if have_nofocus else None

# ---- 四个配置 ----
full = evaluate(raw_focus, pref="debiased", w=W)
ab1 = evaluate(raw_focus, majority_only=True)                 # (1) 不消除多数
ab2 = evaluate(raw_focus, pref="single", w=W)                 # (2) 不消除位置
ab3 = evaluate(raw_nofocus, pref="debiased", w=W) if have_nofocus else None  # (3) 不聚焦子句


def line(name, res):
    acc, rc, bk, _ = res
    print(f"  {name:<34} acc={acc:.4f} ({round(acc*total)})  rec+{rc} brk-{bk} net{rc-bk:+d}")


print(f"Total={total}  单簇固定正确={single_ok}  多数投票基线={base} ({base/total:.4f})")
print("\n================ 消融对照 (w=0.6 固定) ================")
line("完整系统(消除多数+位置+聚焦)", full)
line("(1) 不消除多数 [纯多数投票]", ab1)
line("(2) 不消除位置 [单方向o0]", ab2)
if ab3:
    line("(3) 不聚焦子句 [整条SQL]", ab3)
else:
    print("  (3) 不聚焦子句 : nofocus 缓存未就绪，稍后补")

print("\n================ 各组件价值 ================")
f_acc = full[0]
print(f"  (1) 消除多数优势 价值: {f_acc:.4f} - {ab1[0]:.4f} = {(f_acc-ab1[0])*100:+.2f} pp")
print(f"  (2) 消除位置优势 价值: {f_acc:.4f} - {ab2[0]:.4f} = {(f_acc-ab2[0])*100:+.2f} pp")
if ab3:
    print(f"  (3) 子句聚焦     价值: {f_acc:.4f} - {ab3[0]:.4f} = {(f_acc-ab3[0])*100:+.2f} pp")


# ================ 位置偏差诊断 ================
# 对位置无关的判别器: o0 + o1 == 1 (o0=P(i>j)当i在A, o1=P(j>i)当j在A)
# 偏差指标 bias = o0 + o1 - 1: >0 表示“谁在A位置谁占便宜”（A-位置偏好）
RAW_OFF = "/path/to/LadderSQL/inference_selection/cache/bon_pairwise_raw_glm52.json"


def bias_stats(raw):
    signed = absol = n = flip = 0
    for rec in raw:
        for pp in rec["pair_prefs"]:
            o0, o1 = pp["o0"], pp["o1"]
            b = o0 + o1 - 1.0
            signed += b
            absol += abs(b)
            # 翻盘: 仅因位置不同得出相反胜负 (o0>0.5 与 1-o1>0.5 不一致)
            if (o0 > 0.5) != ((1 - o1) > 0.5):
                flip += 1
            n += 1
    return signed / n, absol / n, flip / n, n


print("\n================ 位置偏差诊断 (bias = o0+o1-1) ================")
s_on, a_on, flip_on, n_on = bias_stats(raw_focus)
print(f"  开思考: signed={s_on:+.3f}  |bias|={a_on:.3f}  翻盘率={flip_on:.3f}  (n={n_on})")
if os.path.exists(RAW_OFF):
    raw_off = json.load(open(RAW_OFF))
    s_off, a_off, flip_off, n_off = bias_stats(raw_off)
    print(f"  关思考: signed={s_off:+.3f}  |bias|={a_off:.3f}  翻盘率={flip_off:.3f}  (n={n_off})")
    # 在关思考缓存上跑 (2) 消融: 去偏 vs 单方向，验证去偏在有偏差时的价值
    off_full = evaluate(raw_off, pref="debiased", w=W)
    off_single = evaluate(raw_off, pref="single", w=W)
    print("\n  -- 关思考下 (2) 位置去偏的价值 --")
    print(f"     去偏(debiased) : acc={off_full[0]:.4f} ({round(off_full[0]*total)})")
    print(f"     单方向(o0)     : acc={off_single[0]:.4f} ({round(off_single[0]*total)})")
    print(f"     => 去偏价值(关思考): {(off_full[0]-off_single[0])*100:+.2f} pp")


# ================ 内在指标: 成对方向准确率 ================
# 只看“一簇对一簇错”的可判定对，看 LLM 偏好方向是否指向正确簇（无 count 干扰）
def dir_acc(raw, pref):
    ok = tot = 0
    for rec in raw:
        cl = rec["clusters"]
        P = pair_matrix(rec, pref)
        for pp in rec["pair_prefs"]:
            i, j = pp["i"], pp["j"]
            ci, cj = cl[i]["correct"], cl[j]["correct"]
            if ci == cj:
                continue
            tot += 1
            pred_i = P[i][j] > 0.5
            if (pred_i and ci) or ((not pred_i) and cj):
                ok += 1
    return ok, tot


print("\n================ 内在指标: 成对方向准确率 (可判定对, 无count干扰) ================")
RAW_FOCUSV2 = "/path/to/LadderSQL/inference_selection/cache/bon_pairwise_raw_glm52_think_focusv2.json"
rows = [
    ("整条SQL+去偏       [去(3)]", raw_nofocus, "debiased"),
    ("子句聚焦v1(挖空)+去偏", raw_focus, "debiased"),
    ("子句聚焦v1+单方向 [去(2)]", raw_focus, "single"),
]
if os.path.exists(RAW_FOCUSV2):
    raw_v2 = json.load(open(RAW_FOCUSV2))
    rows.insert(0, ("子句聚焦v2(标注)+去偏 [完整]", raw_v2, "debiased"))
for tag, raw, pref in rows:
    ok, tot = dir_acc(raw, pref)
    print(f"  {tag:<30} {ok}/{tot} = {ok/tot:.4f}")


# ================ 低 w 下 (2)(3) 的价值（LLM 信号主导）================
print("\n================ 扫 w: (2)(3) 在低 w(更依LLM)时的价值 ================")
print(f"  {'w':<6}{'完整':<12}{'(2)单方向':<12}{'(3)整条SQL':<12}")
for w in [0.0, 0.2, 0.3, 0.4, 0.6]:
    a_full = evaluate(raw_focus, "debiased", w)[0]
    a_ab2 = evaluate(raw_focus, "single", w)[0]
    a_ab3 = evaluate(raw_nofocus, "debiased", w)[0]
    print(f"  {w:<6.1f}{round(a_full*total):<12}{round(a_ab2*total):<12}{round(a_ab3*total):<12}")
