"""Best-of-N 结构化共识选择 - 公共组件.

环境路径已对齐到当前工作区 (/path/to/...)。
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# ============================================================================
# 路径配置（当前环境实际路径）
# ============================================================================
CANDIDATES_FILE = "/path/to/LadderSQL/training/results/20260706/test_rollouts_cache_truncated_20260706_174940.json"
SCHEMA_CACHE_FILE = "/path/to/LadderSQL/schema_construction/relevant_schema_cache/spider_glm5_test_schema_cache_20260415.json"
DB_DIR = "/path/to/nl2sql_dataset/spider/test_database"
REWARD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training")

# 让 reward_func 可导入
if REWARD_DIR not in sys.path:
    sys.path.insert(0, REWARD_DIR)


# ============================================================================
# 候选抽取（与文档 3.2 一致）
# ============================================================================
def extract_candidates(item: Dict[str, Any]) -> List[str]:
    """从一个样本中提取 8 个候选 SQL。"""
    rollouts: List[List[Dict[str, Any]]] = []
    current_rollout: List[Dict[str, Any]] = []
    for t in item["triplets"]:
        prompt_msgs = t["prompt"]["raw_content"]
        # 新 rollout 的标志：prompt 只有 2 条消息（system + user）
        if len(prompt_msgs) == 2 and current_rollout:
            rollouts.append(current_rollout)
            current_rollout = []
        current_rollout.append(t)
    if current_rollout:
        rollouts.append(current_rollout)

    candidates: List[str] = []
    for rollout in rollouts:
        last_sql = None
        for t in rollout:
            resp_content = t["response"]["raw_content"][0]["content"]
            sql_match = re.findall(r"```sql\s*(.*?)\s*```", resp_content, re.S)
            if sql_match:
                last_sql = sql_match[-1].strip()
        candidates.append(last_sql if last_sql else "")
    return candidates


# ============================================================================
# 数据加载
# ============================================================================
def load_samples(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """加载候选文件，抽取每个样本的候选。limit 用于小样本验证。"""
    with open(CANDIDATES_FILE) as f:
        raw = json.load(f)
    samples = [x for x in raw if "db_id" in x]
    if limit is not None:
        samples = samples[:limit]
    for item in samples:
        item["candidates"] = extract_candidates(item)
    return samples


def load_schema_cache() -> Dict[str, str]:
    with open(SCHEMA_CACHE_FILE) as f:
        return json.load(f)


def db_path_of(db_id: str) -> str:
    return f"{DB_DIR}/{db_id}/{db_id}.sqlite"


# ============================================================================
# SQL 执行
# ============================================================================
def execute_sql(db_path: str, sql: str, timeout: int = 30) -> Tuple[Optional[list], Optional[str], float]:
    """执行 SQL，返回 (result_rows, error, execution_time)。"""
    if not sql or not sql.strip():
        return None, "empty sql", float("inf")
    conn = None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=timeout)
        conn.text_factory = lambda b: b.decode(errors="ignore")
        start = time.time()
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        elapsed = time.time() - start
        return rows, None, elapsed
    except Exception as e:
        return None, str(e), float("inf")
    finally:
        if conn:
            conn.close()


def hash_result(rows: Optional[list]) -> Optional[int]:
    """对执行结果做规范化哈希（忽略行顺序），用于一致性分组。"""
    if rows is None:
        return None
    normalized = tuple(sorted(str(r) for r in rows))
    return hash(normalized)


# ============================================================================
# 评估
# ============================================================================
_binary_reward = None


def evaluate(predicted_sql: str, gold_sql: str, db_path: str) -> float:
    """Spider 官方 execution accuracy，返回 1.0 / 0.0。"""
    global _binary_reward
    if _binary_reward is None:
        from reward_func.reward import binary_reward
        _binary_reward = binary_reward
    return _binary_reward(predicted_sql, gold_sql, db_path, False)
