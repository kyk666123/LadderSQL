"""BIRD 权威 reward 标签器 (最终评估用, 与 binary_reward_bird 同口径)：
   reward = 1 iff set(pred_rows) == set(gold_rows)   (行序忽略/去重, 列序严格, 无置换)
带硬超时 (Timer+interrupt) 与 gold 结果缓存, 线程安全。
仅在最终评估阶段调用, 不参与聚簇/判别决策。"""
from __future__ import annotations
import sqlite3, threading
from typing import Optional, Dict, Tuple, List

_GOLD_CACHE: Dict[Tuple[str, str], Optional[frozenset]] = {}
_GLOCK = threading.Lock()


def _exec_rows(db_path: str, sql: str, timeout: int = 15):
    conn = None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=timeout)
        conn.text_factory = lambda b: b.decode(errors="ignore")
        timer = threading.Timer(timeout, conn.interrupt)
        timer.start()
        try:
            cur = conn.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
        finally:
            timer.cancel()
        return rows, None
    except Exception as e:  # noqa: BLE001
        return None, str(e)
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


def gold_set(db_path: str, gold_sql: str, timeout: int = 30) -> Optional[frozenset]:
    key = (db_path, " ".join((gold_sql or "").split()))
    with _GLOCK:
        if key in _GOLD_CACHE:
            return _GOLD_CACHE[key]
    rows, err = _exec_rows(db_path, gold_sql, timeout)
    val = None if (err is not None or rows is None) else frozenset(rows)
    with _GLOCK:
        _GOLD_CACHE[key] = val
    return val


def bird_reward(db_path: str, pred_sql: str, gold_sql: str, timeout: int = 15) -> int:
    """1 if set(pred)==set(gold) else 0。pred 执行失败/超时 -> 0。"""
    if not pred_sql or not pred_sql.strip():
        return 0
    gs = gold_set(db_path, gold_sql, timeout=max(timeout, 30))
    if gs is None:
        return 0
    prows, perr = _exec_rows(db_path, pred_sql, timeout)
    if perr is not None or prows is None:
        return 0
    return 1 if frozenset(prows) == gs else 0
