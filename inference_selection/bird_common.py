"""BIRD old dev best-of-N 公共组件.

与 Spider 版 bon_common.py 对齐，但适配 BIRD：
- 候选 SQL 与正误直接来自采样文件（trials[i].pred_sql / reward_list[i]），无需重跑评估。
- schema 从 sqlite 库现取 DDL（避免 cache-key 不匹配）。
- 额外携带 BIRD 的 evidence（外部知识）供判别器使用。
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# ============================================================================
# 路径配置（BIRD old dev = dev_20240627）
# ============================================================================
BIRD_FILE = os.environ.get(
    "BIRD_FILE",
    "/path/to/sampling_outputs/ckpt_20260711_235128_step192_bird_old_dev_maxturns5_32trials_traj_with_sql.json",
)
# 网络/对象存储挂载盘上直接跑 sqlite 会随机 "disk I/O error"，污染聚簇/预览。
# 优先用拷到本地盘(/tmp)的副本；不存在则回退 OSS。
_LOCAL_DB = "/tmp/bird_dev_databases"
_OSS_DB = "/path/to/nl2sql_dataset/bird/dev_20240627/dev_databases"
# 允许用 BON_DB_DIR 覆盖(跨数据集: spider test / bird 7b 等), 未设置时回退 BIRD old-dev 逻辑。
DB_DIR = os.environ.get("BON_DB_DIR") or (_LOCAL_DB if os.path.isdir(_LOCAL_DB) else _OSS_DB)
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

os.makedirs(CACHE_DIR, exist_ok=True)


def _ok(x: Any) -> bool:
    try:
        return float(x) == 1.0
    except (TypeError, ValueError):
        return False


# ============================================================================
# 数据加载
# ============================================================================
def load_samples(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """加载 BIRD 采样文件。每个样本抽出 candidates / rewards / gold / evidence。

    best-of-N 子采样: 设环境变量 BON_NMAX=N 时, 每题只取前 N 条候选(嵌套子采样),
    用于从同一个 32 采样池派生 N=8/16/24 的 best-of-N 曲线。默认(未设/<=0)=全部。
    """
    with open(BIRD_FILE) as f:
        raw = json.load(f)
    if limit is not None:
        raw = raw[:limit]
    try:
        nmax = int(os.environ.get("BON_NMAX", "0") or "0")
    except (TypeError, ValueError):
        nmax = 0
    out: List[Dict[str, Any]] = []
    for item in raw:
        trials = item.get("trials", [])
        cands = [(t.get("pred_sql") or "").strip() for t in trials]
        rewards = [_ok(t.get("reward")) for t in trials]
        # reward_list 与 trials.reward 应一致；以 reward_list 为准（更权威）
        rl = item.get("reward_list")
        if rl and len(rl) == len(cands):
            rewards = [_ok(x) for x in rl]
        # best-of-N 子采样: 只保留前 N 条候选(嵌套, 保证 N 单调可比)
        if nmax and nmax > 0:
            cands = cands[:nmax]
            rewards = rewards[:nmax]
        out.append(
            {
                "db_id": item["db_id"],
                "question": item.get("question", ""),
                "evidence": item.get("evidence", "") or "",
                "gold_sql": item.get("gold_sql") or item.get("query") or "",
                "candidates": cands,
                "rewards": rewards,
            }
        )
    return out


def db_path_of(db_id: str) -> str:
    return f"{DB_DIR}/{db_id}/{db_id}.sqlite"


# ============================================================================
# SQL 执行 / 结果哈希（与 Spider 版一致）
# ============================================================================
def execute_sql(db_path: str, sql: str, timeout: int = 30) -> Tuple[Optional[list], Optional[str], float]:
    if not sql or not sql.strip():
        return None, "empty sql", float("inf")
    conn = None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=timeout)
        conn.text_factory = lambda b: b.decode(errors="ignore")
        start = time.time()
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        return rows, None, time.time() - start
    except Exception as e:
        return None, str(e), float("inf")
    finally:
        if conn:
            conn.close()


def hash_result(rows: Optional[list]) -> Optional[int]:
    if rows is None:
        return None
    return hash(tuple(sorted(str(r) for r in rows)))


# ============================================================================
# 预计算执行缓存（跨进程稳定 result_sig）
# ============================================================================
_EXEC_CACHE: Optional[Dict[str, Any]] = None


def load_exec_cache(path: Optional[str] = None) -> Dict[str, Any]:
    """加载 exec_cache_step192_32.json；若缺失则返回空字典。"""
    global _EXEC_CACHE
    if _EXEC_CACHE is not None:
        return _EXEC_CACHE
    p = path or os.environ.get(
        "EXEC_CACHE", f"{CACHE_DIR}/exec_cache_step192_32.json"
    )
    if os.path.exists(p):
        with open(p) as f:
            d = json.load(f)
        _EXEC_CACHE = d.get("results", {})
    else:
        _EXEC_CACHE = {}
    return _EXEC_CACHE


def _cache_key(db_id: str, sql: str) -> str:
    return f"{db_id}|||{' '.join(sql.split())}"


def cached_result(db_id: str, sql: str) -> Optional[Dict[str, Any]]:
    """查询执行缓存。返回 dict 含 ok/error/row_count/result_sig/preview，未命中返回 None。"""
    cache = load_exec_cache()
    return cache.get(_cache_key(db_id, sql))


def cached_preview(db_id: str, sql: str, k: int = 5, maxw: int = 400) -> str:
    """返回缓存中的执行结果预览字符串。"""
    rec = cached_result(db_id, sql)
    if rec is None:
        return "(not cached)"
    if not rec.get("ok"):
        err = rec.get("error") or "unknown error"
        return f"ERROR: {str(err)[:120]}"
    rows = rec.get("preview", [])
    if not rows:
        return "(empty result)"
    head = rows[:k]
    s = " | ".join(str(r) for r in head)
    rc = rec.get("row_count")
    if rc is not None and rc > k:
        s += f"  ... ({rc} rows total)"
    return s[:maxw]


def cached_sig(db_id: str, sql: str) -> Optional[str]:
    """返回跨进程稳定的 result_sig（md5）；报错统一为 'ERR'）。"""
    rec = cached_result(db_id, sql)
    if rec is None:
        return None
    if not rec.get("ok"):
        return "ERR"
    sig = rec.get("result_sig")
    return sig if sig is not None else "ERR"


# ============================================================================
# Schema DDL（从库现取，模块级缓存）
# ============================================================================
_DDL: Dict[str, str] = {}


def schema_ddl(db_path: str) -> str:
    if db_path in _DDL:
        return _DDL[db_path]
    ddl = ""
    conn = None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cur = conn.cursor()
        cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL")
        ddl = "\n\n".join(r[0] for r in cur.fetchall())
    except Exception as e:
        ddl = f"-- schema unavailable: {e}"
    finally:
        if conn:
            conn.close()
    _DDL[db_path] = ddl
    return ddl



# ============================================================================
# 带列描述的 schema（注入 BIRD 官方 database_description，帮判别器看懂晦涩列名）
# ============================================================================
import csv as _csv
import glob as _glob

_DESC: Dict[str, str] = {}


def _load_col_desc(db_dir_of_db: str) -> Dict[str, list]:
    """读 database_description/*.csv → {table_lower: [(col, desc, valdesc), ...]}"""
    out: Dict[str, list] = {}
    desc_dir = os.path.join(db_dir_of_db, "database_description")
    if not os.path.isdir(desc_dir):
        return out
    for fp in _glob.glob(os.path.join(desc_dir, "*.csv")):
        table = os.path.splitext(os.path.basename(fp))[0]
        rows = []
        try:
            with open(fp, encoding="utf-8-sig", errors="ignore") as f:
                for r in _csv.DictReader(f):
                    col = (r.get("original_column_name") or "").strip()
                    desc = (r.get("column_description") or "").strip()
                    vdesc = (r.get("value_description") or "").strip()
                    if col:
                        rows.append((col, desc, vdesc))
        except Exception:
            continue
        out[table.lower()] = rows
    return out


def schema_with_descriptions(db_path: str, max_val_len: int = 80) -> str:
    """DDL + 每表列的 column_description / value_description（仅保留有信息量的）。"""
    if db_path in _DESC:
        return _DESC[db_path]
    ddl = schema_ddl(db_path)
    db_dir_of_db = os.path.dirname(db_path)
    coldesc = _load_col_desc(db_dir_of_db)
    blocks = [ddl]
    if coldesc:
        blocks.append("\n# Column meanings (BIRD external knowledge):")
        for table, rows in coldesc.items():
            lines = []
            for (col, desc, vdesc) in rows:
                note = ""
                if desc and desc.lower() != col.lower():
                    note = desc
                if vdesc:
                    note = (note + "; values: " + vdesc) if note else ("values: " + vdesc)
                if note:
                    lines.append(f'  - {col}: {note[:max_val_len]}')
            if lines:
                blocks.append(f"[{table}]\n" + "\n".join(lines))
    out = "\n".join(blocks)
    _DESC[db_path] = out
    return out
