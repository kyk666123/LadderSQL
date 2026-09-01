"""文本键裁决库: 把锦标赛成对裁决按 (sid, posA_sql, posB_sql) 文本对存储。

动机: best-of-N 趋势需要在 N=8/16/24 上重跑锦标赛, 但 N<32 时执行聚簇会重排,
现有 multirep_pairs_*.json 用「簇下标」`sid|i|j|ia|ib|order` 存储, 下标失配无法直接复用。
而一次裁决本质只由 (题目, 放在A位的SQL文本, 放在B位的SQL文本) 决定
(schema/evidence/struct_facts/preview 都是这三者的确定函数, order 数字只决定谁在A/B位)。
因此改按【文本对】存储后:
  1) 先把已有 32 池的下标缓存离线回填成文本键 (seed_from_index_cache, 零 glm 调用);
  2) N<32 的对战绝大多数命中 (小 N 候选文本都是 32 池子集), 几乎不新增 glm 调用。

裁决取值 verdict ∈ {"A","B","T"}: 位置A胜 / 位置B胜 / 平局。
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from typing import Any, Dict, List, Optional

import gated_common as G


def _norm(sql: str) -> str:
    return G._norm(sql)


def text_key(sid: int, sql_pos_a: str, sql_pos_b: str) -> str:
    """按 (题目 sid, A位SQL文本, B位SQL文本) 生成稳定 md5 键。

    注意 (a,b) 与 (b,a) 是不同的键: 分别对应 order0/order1 两个摆放方向,
    与 run_multirep_tournament 的双向判别语义一致。"""
    h = hashlib.md5()
    h.update(str(sid).encode())
    h.update(b"\x00")
    h.update(_norm(sql_pos_a).encode("utf-8", "ignore"))
    h.update(b"\x00")
    h.update(_norm(sql_pos_b).encode("utf-8", "ignore"))
    return h.hexdigest()


def load_store(path: Optional[str]) -> Dict[str, str]:
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_store(path: str, store: Dict[str, str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(store, f)
    os.replace(tmp, path)


def cluster_writings(candidates: List[str], cluster: Dict[str, Any], rmax: int) -> List[str]:
    """复刻 run_multirep_tournament/eval_multirep 的 writings(): 簇内按 (频次降, 长度升)
    取前 rmax 种归一化写法。保证下标 ia/ib 与锦标赛/评估侧完全一致。"""
    sqls = [_norm(candidates[i]) for i in cluster["members"]]
    cnt = Counter(sqls)
    return sorted(cnt, key=lambda s: (-cnt[s], len(s)))[:rmax]


def seed_from_index_cache(
    index_cache_path: str,
    samples: List[Dict[str, Any]],
    info: Dict[str, Any],
    store: Dict[str, str],
    topk: int = 3,
    rmax: int = 3,
) -> int:
    """把一个下标键缓存 (multirep_pairs_*.json: sid|i|j|ia|ib|order -> verdict)
    在【当前 classify 口径】下回填成文本键写入 store。返回新增条数。零 glm 调用。

    调用方须保证传入的 samples/info 与该缓存产生时口径一致 (通常是 N=32, 即 BON_NMAX 未设)。
    """
    if not index_cache_path or not os.path.exists(index_cache_path):
        return 0
    with open(index_cache_path) as f:
        idx = json.load(f)
    per = info["per_sample"]
    added = 0
    for key, v in idx.items():
        parts = key.split("|")
        if len(parts) != 6:
            continue
        try:
            sid, i, j, ia, ib, order = (int(x) for x in parts)
        except ValueError:
            continue
        ps = per.get(sid)
        if not ps:
            continue
        cls = ps["clusters"][:topk]
        if i >= len(cls) or j >= len(cls):
            continue
        cands = samples[sid]["candidates"]
        wi = cluster_writings(cands, cls[i], rmax)
        wj = cluster_writings(cands, cls[j], rmax)
        if ia >= len(wi) or ib >= len(wj):
            continue
        sa, sb = wi[ia], wj[ib]
        # order0: A=sa,B=sb ; order1: A=sb,B=sa (与 run_multirep_tournament.work 一致)
        k = text_key(sid, sa, sb) if order == 0 else text_key(sid, sb, sa)
        if k not in store:
            store[k] = v
            added += 1
    return added


def lookup(store: Dict[str, str], sid: int, sql_pos_a: str, sql_pos_b: str) -> Optional[str]:
    return store.get(text_key(sid, sql_pos_a, sql_pos_b))
