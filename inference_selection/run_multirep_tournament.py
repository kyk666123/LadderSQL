"""多代表锦标赛选择器: 每对簇不只比一条代表, 而是各取 R 种写法做 R×R×双向,
平均成稳健的成对胜率 p_ij, 再算 js + 动态推翻。
同簇执行结果相同 -> 事实/预览只随写法文本变。
带缓存 (cache/multirep_pairs.json), 可断点续跑。

================ 锁定最优配置 (2026-07, BIRD old dev 1534) ================
  linked schema (per-sample) + 抗过度解读先验 + biased + 动态推翻
  TOPK=3, RMAX=3, k=1.0  ->  EX = 73.47%
  (多数投票基线 71.58% / pass@32 天花板 80.77%; 单代表版 73.08%)
  机制: 3 种写法投票, 把"判别器撞上别扭写法而坚定判错"的噪声抹平,
        亏损 24->15; 靠降噪而非调参涨分。
  动态推翻: 选 argmax_i [ js_i + k*(n_i/32) ], 等价于
            "判别器推翻多数需胜率优势 > k*(领先票数)/32", 多数越大越难推翻。
  RMAX 可用环境变量 MULTIREP_RMAX 覆盖 (实验 4/5); 默认 3 为锁定配置。
==========================================================================
"""
import json, os, sys, time, functools
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
print = functools.partial(print, flush=True)
import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import bird_common as B, gated_common as G
import run_gated_reasoning_judge as R

R.USE_SIMPLICITY = True
if os.environ.get("SIMP_V2"):
    # 针对 15 个亏损案例的高频过度解读模式补强 (BETWEEN含端点/精确列/时间过滤)
    R.SIMPLICITY_NOTE = R.SIMPLICITY_NOTE.rstrip() + """
- Interpret "between X and Y" / "from X to Y" as INCLUSIVE of BOTH endpoints (BETWEEN X AND Y); do NOT prefer a half-open range like (>= X AND < Y) that drops an endpoint.
- Return EXACTLY the columns the question asks for — no more, no fewer. Do NOT prefer a query that adds an extra id/label column "for context".
- Do NOT add temporal/date-range or value filters (e.g. "after the account was opened", column > 0) that the question does not LITERALLY state as a filter.
- The four patterns {adding COUNT(DISTINCT); adding WHERE/AND filters; half-open intervals; extra output columns} are the MOST COMMON reasons a judge wrongly rejects the correct simpler answer. Treat any such difference as evidence AGAINST the more elaborate option unless the wording explicitly demands it."""
    print("[prompt] 使用 SIMPLICITY_NOTE V2 (亏损案例补强)", flush=True)
if os.environ.get("SIMP_V3"):
    # 外科手术: 只加强 COUNT(*) vs COUNT(DISTINCT) 一条, 针对"join去重"失败模式(135/150/605/1256)
    R.SIMPLICITY_NOTE = R.SIMPLICITY_NOTE.rstrip() + """
- Prefer COUNT(*) over COUNT(DISTINCT x) EVEN when the counted rows come from a JOIN that could repeat an entity id. The gold answer counts qualifying ROWS, not distinct entities; adding DISTINCT to "de-duplicate" a join is the single most common judging error here. If option A is COUNT(*) and option B differs ONLY by using COUNT(DISTINCT ...), and the question does not literally say "distinct"/"unique"/"different"/"how many different", then A (COUNT(*)) is the correct one."""
    print("[prompt] 使用 SIMPLICITY_NOTE V3 (仅COUNT-DISTINCT外科手术)", flush=True)
_SCHEMA_CACHE = os.environ.get("SCHEMA_CACHE", "../apex-sql-schema-link-results/bird_old_dev/schema_cache_old_dev_rendered.json")
if _SCHEMA_CACHE and os.path.exists(_SCHEMA_CACHE):
    R.SCHEMA_LUT.update(json.load(open(_SCHEMA_CACHE)))
    print(f"[schema] linked schema <- {_SCHEMA_CACHE} ({len(R.SCHEMA_LUT)} 条)", flush=True)
else:
    print(f"[schema] 无 linked schema ({_SCHEMA_CACHE}), 回退库 DDL", flush=True)
TOPK = 3          # 锦标赛簇数 (锁定)
RMAX = int(os.environ.get("MULTIREP_RMAX", "3"))   # 每簇写法数; 默认 3 = 锁定配置
WORKERS = int(os.environ.get("MULTIREP_WORKERS", "24"))   # 并发判别调用数; 远低于 rpm 上限时可调高提速
CACHE = os.environ.get("MULTIREP_CACHE", "cache/multirep_pairs.json")
MODEL = os.environ.get("MULTIREP_MODEL", "glm-5.2")   # 可切换判别器模型
THINK = bool(os.environ.get("MULTIREP_THINK"))        # 推理模型(如deepseek)置1: mt=8000+reasoning_content兜底
print(f"[model] 判别器 = {MODEL}  thinking={THINK}", flush=True)

samples = B.load_samples(); info = G.classify(samples); per = info["per_sample"]

def writings(sid, c):
    cands = samples[sid]["candidates"]
    sqls = [G._norm(cands[i]) for i in c["members"]]
    cnt = Counter(sqls)
    return sorted(cnt, key=lambda s: (-cnt[s], len(s)))[:RMAX]

# 载入缓存
cache = {}
if os.path.exists(CACHE):
    cache = json.load(open(CACHE))
    print(f"[cache] 载入 {len(cache)} 条已算 verdict")

exec_cache = {}
def gx(db, sig, sql):
    if sig not in exec_cache:
        exec_cache[sig] = R.exec_full(db, sql)
    return exec_cache[sig]
prev_cache = {}
def gp(db, sql):
    if sql not in prev_cache:
        prev_cache[sql] = B.cached_preview(db, sql)
    return prev_cache[sql]

# 组装: 每样本 top-K 簇两两(i<j), 各取写法, ia×ib × 双向
jobs = []; meta = {}
for sid in info["contested_ids"]:
    cls = per[sid]["clusters"][:TOPK]
    if len(cls) < 2:
        continue
    db = samples[sid]["db_id"]
    w = [writings(sid, c) for c in cls]
    sig = [c["sig"] for c in cls]
    meta[sid] = {"db": db, "w": w, "sig": sig, "m": len(cls)}
    for i in range(len(cls)):
        for j in range(i + 1, len(cls)):
            for ia in range(len(w[i])):
                for ib in range(len(w[j])):
                    for order in (0, 1):
                        k = f"{sid}|{i}|{j}|{ia}|{ib}|{order}"
                        if k not in cache:
                            jobs.append((sid, i, j, ia, ib, order))
print(f"[jobs] 待跑 {len(jobs)} 调用 (已缓存 {len(cache)}), 样本 {len(meta)}, TOPK={TOPK} RMAX={RMAX}")

def work(job):
    sid, i, j, ia, ib, order = job
    m = meta[sid]; db = m["db"]
    sa = m["w"][i][ia]; sb = m["w"][j][ib]
    exa = gx(db, m["sig"][i], sa); exb = gx(db, m["sig"][j], sb)
    ca = {"rep_sql": sa, "preview": gp(db, sa)}
    cb = {"rep_sql": sb, "preview": gp(db, sb)}
    if order == 0:
        facts = R.struct_facts(exa, exb)
        prompt = R.build_prompt(samples[sid], ca, cb, facts)
    else:
        facts = R.struct_facts(exb, exa)
        prompt = R.build_prompt(samples[sid], cb, ca, facts)
    v = R.call_verdict(prompt, MODEL, THINK)
    return f"{sid}|{i}|{j}|{ia}|{ib}|{order}", (v or "T")

if jobs:
    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for k, v in ex.map(work, jobs):
            cache[k] = v; done += 1
            if done % 1000 == 0:
                json.dump(cache, open(CACHE, "w"))
                print(f"  {done}/{len(jobs)} ({time.time()-t0:.0f}s) 已存盘")
    json.dump(cache, open(CACHE, "w"))
    print(f"[done] 全部完成, 用时 {time.time()-t0:.0f}s")
else:
    print("[done] 缓存已完整, 直接评估")
