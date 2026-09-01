"""置信度门控锦标赛选择器 —— 公共分组/分类组件 (reward-blind)。

判别脚本 (run_gated_tournament_judge.py) 与离线扫描脚本 (sweep_gated_tournament.py)
共用此模块, 保证两侧的簇划分/样本分类完全一致。

样本以 load_samples 的枚举下标 sid 作为稳定 id (两脚本载入同一文件同序)。
分组只用执行结果签名 (cached_sig) + 簇大小; reward 仅打进 cluster["correct"] 供
【最终评估】使用, 判别过程绝不可见 (reward-blind)。
"""
from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import bird_common as B

# 簇代表 SQL 的选取策略: "shortest"(最短) | "mode"(簇内最常见写法) | "repr"(最能代表簇)
# 默认 shortest, 保持既有行为; 实验时由调用方设置。
REP_MODE = "shortest"


def _medoid(sqls):
    """与其他写法最相似的一条 (token-Jaccard 中心点)。平手取最短。"""
    toks = [set(s.lower().split()) for s in sqls]
    best_i, best_sim = 0, -1.0
    for i in range(len(sqls)):
        sim = 0.0
        for j in range(len(sqls)):
            if i == j:
                continue
            a, b = toks[i], toks[j]
            u = a | b
            sim += (len(a & b) / len(u)) if u else 0.0
        if sim > best_sim or (sim == best_sim and len(sqls[i]) < len(sqls[best_i])):
            best_i, best_sim = i, sim
    return sqls[best_i]


def _pick_rep(sqls):
    if REP_MODE == "mode":
        from collections import Counter
        cnt = Counter(sqls)
        top = cnt.most_common(1)[0][1]
        # 最常见; 多个并列时取其中最短, 保证确定性
        cands = [s for s in cnt if cnt[s] == top]
        return min(cands, key=len)
    if REP_MODE == "repr":
        from collections import Counter
        cnt = Counter(sqls)
        top = cnt.most_common(1)[0][1]
        if top >= 2:                       # 有主流写法 -> 众数
            cands = [s for s in cnt if cnt[s] == top]
            return min(cands, key=len)
        return _medoid(sqls)               # 全部独特 -> 中心点
    return min(sqls, key=len)          # shortest


def _norm(sql: str) -> str:
    return " ".join((sql or "").split())


def build_clusters(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """把一个样本的 32 候选按执行结果签名聚簇。

    返回按簇大小降序的 list, 每簇:
      sig, n, rep_sql(最短), members(候选下标), correct(该簇是否正确, 仅评估用),
      preview, profile
    执行失败/无缓存签名 -> 归 'ERR' 簇。
    """
    db_id = rec["db_id"]
    cands = rec["candidates"]
    rewards = rec["rewards"]
    groups: Dict[str, Dict[str, Any]] = {}
    for i, sql in enumerate(cands):
        if not sql or not sql.strip():
            continue
        sig = B.cached_sig(db_id, sql)
        if sig is None:            # 缓存未命中: 现取一次并缓存签名口径
            rows, err, _ = B.execute_sql(B.db_path_of(db_id), sql)
            sig = "ERR" if err else str(B.hash_result(rows))
        g = groups.setdefault(sig, {"sig": sig, "members": [], "sqls": [], "rew": []})
        g["members"].append(i)
        g["sqls"].append(_norm(sql))
        g["rew"].append(bool(rewards[i]))
    out = []
    for sig, g in groups.items():
        rep = _pick_rep(g["sqls"])
        out.append({
            "sig": sig,
            "n": len(g["members"]),
            "rep_sql": rep,
            "members": g["members"],
            "correct": any(g["rew"]),                 # 仅评估用
            "preview": B.cached_preview(db_id, rep),
        })
    out.sort(key=lambda c: (-c["n"], c["sig"]))
    return out


def classify(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """把全集样本分为: 单簇 / 全对多簇 / contested(≥2簇且有对有错)。

    返回 dict:
      per_sample: sid -> {"clusters":..., "kind":..., "maj_correct":bool}
      fixed_correct: 固定正确数 (单簇正确 + 全对多簇)
      contested_ids: contested 样本 sid 列表
      counts: 各类计数
    """
    per_sample: Dict[int, Dict[str, Any]] = {}
    fixed_correct = 0
    contested_ids: List[int] = []
    single_ok = allmulti_ok = single_bad = 0
    for sid, rec in enumerate(samples):
        cls = build_clusters(rec)
        if not cls:
            per_sample[sid] = {"clusters": [], "kind": "empty", "maj_correct": False}
            continue
        maj = cls[0]                                   # 已按大小降序
        maj_correct = bool(maj["correct"])
        has1 = any(c["correct"] for c in cls)
        has0 = any(not c["correct"] for c in cls)
        if len(cls) == 1:
            kind = "single"
            if cls[0]["correct"]:
                single_ok += 1
                fixed_correct += 1
            else:
                single_bad += 1
        elif has1 and not has0:
            kind = "allmulti"                          # 多簇但全部正确
            allmulti_ok += 1
            fixed_correct += 1
        elif has1 and has0:
            kind = "contested"
            contested_ids.append(sid)
        else:
            kind = "allwrong_multi"                    # 多簇全错
        per_sample[sid] = {"clusters": cls, "kind": kind, "maj_correct": maj_correct}
    return {
        "per_sample": per_sample,
        "fixed_correct": fixed_correct,
        "contested_ids": contested_ids,
        "counts": {
            "total": len(samples),
            "single_ok": single_ok,
            "single_bad": single_bad,
            "allmulti_ok": allmulti_ok,
            "contested": len(contested_ids),
        },
    }
