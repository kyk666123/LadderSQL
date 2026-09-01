"""best-of-N 多代表锦标赛 (子采样版): 与 run_multirep_tournament.py 同口径,
但对战裁决先查【文本键裁决库 verdict_store】, 命中即复用, 未命中才调 glm。

配合 BON_NMAX=N (bird_common.load_samples 子采样前 N 条候选) 使用, 用于在
N=8/16/24/32 上重跑锦标赛而【几乎不新增 glm 调用】:
  - 先用 SEED_INDEX_CACHE 指向已有 32 池锦标赛缓存, 在 N=32 口径下回填文本裁决库;
  - 之后各 N 的对战绝大多数命中该库 (小 N 候选文本是 32 池子集);
  - STRICT_NO_NEW_CALLS=1: 未命中不调 glm, 记平局 T (严格零新增调用)。

产出该 N 的下标键缓存 MULTIREP_CACHE (sid|i|j|ia|ib|order -> verdict), 供 eval_multirep.py 原样评估。

环境变量:
  BON_NMAX              当前 N (子采样候选数); 未设=32
  MULTIREP_CACHE        产出的该 N 下标键缓存路径
  VERDICT_STORE         文本键裁决库路径 (跨 N 共享, 持久化)
  SEED_INDEX_CACHE      (可选) 已有 32 池下标缓存, 首次回填文本库用
  STRICT_NO_NEW_CALLS   1=未命中记 T 不调 glm; 默认 0=调 glm 补齐
  MULTIREP_RMAX/MULTIREP_MODEL/MULTIREP_WORKERS/MULTIREP_THINK/SCHEMA_CACHE  同原脚本
"""
import json, os, sys, time, functools
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
print = functools.partial(print, flush=True)
import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import bird_common as B, gated_common as G
import run_gated_reasoning_judge as R
import verdict_store as VS

R.USE_SIMPLICITY = True   # 与锁定配置一致
_SCHEMA_CACHE = os.environ.get("SCHEMA_CACHE", "../apex-sql-schema-link-results/bird_old_dev/schema_cache_old_dev_rendered.json")
if _SCHEMA_CACHE and os.path.exists(_SCHEMA_CACHE):
    R.SCHEMA_LUT.update(json.load(open(_SCHEMA_CACHE)))
    print(f"[schema] linked schema <- {_SCHEMA_CACHE} ({len(R.SCHEMA_LUT)} 条)")
else:
    print(f"[schema] 无 linked schema ({_SCHEMA_CACHE}), 回退库 DDL")

TOPK = 3
RMAX = int(os.environ.get("MULTIREP_RMAX", "3"))
WORKERS = int(os.environ.get("MULTIREP_WORKERS", "24"))
CACHE = os.environ.get("MULTIREP_CACHE", "cache/multirep_pairs_bon.json")
STORE_PATH = os.environ.get("VERDICT_STORE", "cache/verdict_store_bon.json")
MODEL = os.environ.get("MULTIREP_MODEL", "glm-5.2")
THINK = bool(os.environ.get("MULTIREP_THINK"))
STRICT = os.environ.get("STRICT_NO_NEW_CALLS", "").lower() in ("1", "true", "yes")
NMAX = int(os.environ.get("BON_NMAX", "0") or "0") or 32
print(f"[cfg] N={NMAX} model={MODEL} think={THINK} RMAX={RMAX} strict_no_new_calls={STRICT}")


def writings(candidates, cluster):
    return VS.cluster_writings(candidates, cluster, RMAX)


# ---- 文本裁决库: 载入 + (可选)从已有 32 池下标缓存回填 ----
store = VS.load_store(STORE_PATH)
print(f"[store] 载入文本裁决 {len(store)} 条 <- {STORE_PATH}")
seed_idx = os.environ.get("SEED_INDEX_CACHE")
if seed_idx and os.path.exists(seed_idx):
    _saved = os.environ.get("BON_NMAX")
    os.environ["BON_NMAX"] = "0"                    # 回填必须在 32 全量口径
    s32 = B.load_samples(); i32 = G.classify(s32)
    n_add = VS.seed_from_index_cache(seed_idx, s32, i32, store, TOPK, RMAX)
    if _saved is None:
        os.environ.pop("BON_NMAX", None)
    else:
        os.environ["BON_NMAX"] = _saved
    print(f"[seed] 从 {seed_idx} 回填 {n_add} 条文本裁决 (零 glm)")
    if n_add:
        VS.save_store(STORE_PATH, store)

# ---- 当前 N 口径 classify ----
samples = B.load_samples(); info = G.classify(samples); per = info["per_sample"]
print(f"[classify] N={NMAX} {info['counts']} fixed_correct={info['fixed_correct']}")

# 已产出的该 N 下标缓存 (断点续跑)
idx_cache = {}
if os.path.exists(CACHE):
    idx_cache = json.load(open(CACHE))
    print(f"[cache] 载入该 N 下标缓存 {len(idx_cache)} 条 <- {CACHE}")

# 组装 jobs: 每 contested 样本 top-K 簇两两(i<j), 各取写法, ia×ib × 双向
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

jobs = []; meta = {}
hit = 0
for sid in info["contested_ids"]:
    cls = per[sid]["clusters"][:TOPK]
    if len(cls) < 2:
        continue
    db = samples[sid]["db_id"]
    cands = samples[sid]["candidates"]
    w = [writings(cands, c) for c in cls]
    sig = [c["sig"] for c in cls]
    meta[sid] = {"db": db, "w": w, "sig": sig}
    for i in range(len(cls)):
        for j in range(i + 1, len(cls)):
            for ia in range(len(w[i])):
                for ib in range(len(w[j])):
                    for order in (0, 1):
                        k = f"{sid}|{i}|{j}|{ia}|{ib}|{order}"
                        if k in idx_cache:
                            continue
                        sa, sb = w[i][ia], w[j][ib]
                        posa, posb = (sa, sb) if order == 0 else (sb, sa)
                        v = VS.lookup(store, sid, posa, posb)   # 文本库命中?
                        if v is not None:
                            idx_cache[k] = v; hit += 1
                        elif STRICT:
                            idx_cache[k] = "T"                  # 严格零新增: 记平局
                        else:
                            jobs.append((sid, i, j, ia, ib, order))
print(f"[jobs] 文本库命中 {hit} 条; 待新调 glm {len(jobs)} 条; 样本 {len(meta)}")

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
        posa, posb = sa, sb
    else:
        facts = R.struct_facts(exb, exa)
        prompt = R.build_prompt(samples[sid], cb, ca, facts)
        posa, posb = sb, sa
    v = R.call_verdict(prompt, MODEL, THINK) or "T"
    return f"{sid}|{i}|{j}|{ia}|{ib}|{order}", v, sid, posa, posb

if jobs:
    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for k, v, sid, posa, posb in ex.map(work, jobs):
            idx_cache[k] = v
            store[VS.text_key(sid, posa, posb)] = v      # 顺手写回文本库, 供其它 N 复用
            done += 1
            if done % 500 == 0:
                json.dump(idx_cache, open(CACHE, "w"))
                VS.save_store(STORE_PATH, store)
                print(f"  {done}/{len(jobs)} ({time.time()-t0:.0f}s) 已存盘")
    print(f"[done] 新调 glm {done} 条, 用时 {time.time()-t0:.0f}s")
else:
    print("[done] 无需新调 glm (全部命中/严格模式)")

json.dump(idx_cache, open(CACHE, "w"))
VS.save_store(STORE_PATH, store)
print(f"[save] 该 N 下标缓存 {len(idx_cache)} 条 -> {CACHE}; 文本库 {len(store)} 条 -> {STORE_PATH}")
