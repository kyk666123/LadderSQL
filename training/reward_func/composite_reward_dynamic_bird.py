"""BIRD 版动态分层复合奖励系统 - 全阶段执行语义驱动.

核心设计：
1. 动态阶段：根据 gold SQL 实际包含的子句决定参与对比的阶段数
2. 动态赋分：最远阶段固定 0.70，每往前一个阶段递减 0.05
3. 全阶段执行对比：利用 SQL 逻辑求值顺序构造中间 SQL 并执行

与 Spider 版的区别：
- FROM/JOIN: 规则对比增加 JOIN 类型语义检查（LEFT/RIGHT/INNER）
- GROUP BY: 从结构化对比改为执行对比（simple_binary_reward）
- SELECT: 从结构化对比改为执行对比（binary_reward，含列排列枚举）
- ORDER BY: 从结构化对比改为执行对比（严格有序比较，验证行序）

奖励示例（6 阶段 SQL）：
  FROM=0.45, WHERE=0.50, GROUP_BY=0.55, HAVING=0.60, SELECT=0.65, ORDER_BY=0.70
奖励示例（3 阶段 SQL, 无 GROUP_BY/HAVING/ORDER_BY）：
  FROM=0.60, WHERE=0.65, SELECT=0.70
Final match = 1.0
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Tuple

from reward_func.composite_reward_fast import (
    SQLClauses,
    SQLParser,
    SQLStage,
    _extract_alias_mapping,
    _extract_join_conditions_from_clauses,
    _extract_tables_from_clauses,
    get_binary_reward_func,
)
from reward_func.reward import (
    simple_binary_reward as _simple_binary_reward_func,
    binary_reward_bird as _binary_reward_bird_func,
)

logger = logging.getLogger(__name__)


# ============================================================================
# 动态奖励参数
# ============================================================================

MAX_STAGE_REWARD = float(os.getenv("REWARD_MAX_STAGE", "0.70"))    # 最远阶段奖励值(环境变量 REWARD_MAX_STAGE 可覆盖)
REWARD_DECREMENT = float(os.getenv("REWARD_DECREMENT", "0.05"))    # 每往前一阶段递减值(环境变量 REWARD_DECREMENT 可覆盖)
FINAL_MATCH_REWARD = 1.0     # 最终完全匹配的奖励

# 实验开关: 聚合模式 gating(最长前缀) | sum(通过比例累加); 分值方案 relative(递减) | absolute(固定映射)
REWARD_AGG_MODE = os.getenv("REWARD_AGG_MODE", "gating").strip().lower()
REWARD_STAGE_SCHEME = os.getenv("REWARD_STAGE_SCHEME", "relative").strip().lower()

# absolute 方案的固定阶段分值(不随阶段数变化, 消除奖励与 SQL 复杂度耦合)
ABSOLUTE_STAGE_REWARD = {
    SQLStage.FROM_JOIN: 0.45,
    SQLStage.WHERE: 0.50,
    SQLStage.GROUP_BY: 0.55,
    SQLStage.HAVING: 0.60,
    SQLStage.SELECT: 0.65,
    SQLStage.ORDER_BY: 0.70,
}


def compute_stage_rewards(active_stages: List[SQLStage]) -> Dict[SQLStage, float]:
    """根据活跃阶段数动态计算各阶段奖励值.

    最远阶段 = MAX_STAGE_REWARD，往前每步递减 REWARD_DECREMENT。

    Examples:
        6 stages: {FROM: 0.45, WHERE: 0.50, GB: 0.55, HAVING: 0.60, SELECT: 0.65, OB: 0.70}
        3 stages: {FROM: 0.60, WHERE: 0.65, SELECT: 0.70}
    """
    n = len(active_stages)
    rewards = {}
    if REWARD_STAGE_SCHEME == "absolute":
        # 固定绝对分值, 不随阶段数变化
        return {stage: ABSOLUTE_STAGE_REWARD[stage] for stage in active_stages}
    for i, stage in enumerate(active_stages):
        # i=0 是第一个阶段，i=n-1 是最后一个阶段
        rewards[stage] = round(MAX_STAGE_REWARD - (n - 1 - i) * REWARD_DECREMENT, 2)
    return rewards


# ============================================================================
# 性能统计
# ============================================================================

@dataclass
class PerformanceStats:
    """性能统计信息."""
    binary_reward_call_count: int = 0
    simple_binary_reward_call_count: int = 0
    ordered_compare_call_count: int = 0
    binary_reward_time: float = 0.0
    simple_binary_reward_time: float = 0.0
    total_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "binary_reward_call_count": self.binary_reward_call_count,
            "simple_binary_reward_call_count": self.simple_binary_reward_call_count,
            "ordered_compare_call_count": self.ordered_compare_call_count,
            "binary_reward_time_sec": round(self.binary_reward_time, 6),
            "simple_binary_reward_time_sec": round(self.simple_binary_reward_time, 6),
            "total_time_sec": round(self.total_time, 6),
        }


# ============================================================================
# JOIN 类型语义检查
# ============================================================================

def _extract_join_types(clauses: SQLClauses) -> Dict[str, str]:
    """提取每个被 JOIN 的表的 JOIN 类型.

    Returns:
        {table_name_lower: join_type} 其中 join_type 为 'INNER', 'LEFT', 'RIGHT', 'FULL', 'CROSS'
    """
    join_types: Dict[str, str] = {}

    for join_str in clauses.joins:
        join_str_upper = join_str.upper().strip()
        if join_str_upper.startswith("LEFT"):
            jtype = "LEFT"
        elif join_str_upper.startswith("RIGHT"):
            jtype = "RIGHT"
        elif join_str_upper.startswith("FULL"):
            jtype = "FULL"
        elif join_str_upper.startswith("CROSS"):
            jtype = "CROSS"
        else:
            jtype = "INNER"

        m = re.search(r'JOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)', join_str, re.I)
        if m:
            table = m.group(1).lower()
            join_types[table] = jtype

    return join_types


def _normalize_join_semantics(join_types: Dict[str, str]) -> Dict[str, str]:
    """规范化 JOIN 语义.

    Returns:
        {table_name: 'PRESERVED' | 'NULLABLE' | 'INNER' | 'FULL'}
        PRESERVED = 该表行全部保留（RIGHT JOIN 的目标表）
        NULLABLE = 该表可能出现 NULL（LEFT JOIN 的目标表）
    """
    semantics: Dict[str, str] = {}
    for table, jtype in join_types.items():
        if jtype == "LEFT":
            semantics[table] = "NULLABLE"
        elif jtype == "RIGHT":
            semantics[table] = "PRESERVED"
        elif jtype == "FULL":
            semantics[table] = "FULL"
        else:
            semantics[table] = "INNER"
    return semantics


def compare_from_join_bird(clauses1: SQLClauses, clauses2: SQLClauses) -> Tuple[bool, Dict[str, Any]]:
    """BIRD 版 FROM/JOIN 比较：表集合 + 连接条件 + JOIN 类型语义.

    A LEFT JOIN B ≠ A INNER JOIN B（行数不同）
    A LEFT JOIN B == B RIGHT JOIN A（语义等价，保留同一侧的行）
    """
    alias1 = _extract_alias_mapping(clauses1)
    alias2 = _extract_alias_mapping(clauses2)

    tables1 = _extract_tables_from_clauses(clauses1)
    tables2 = _extract_tables_from_clauses(clauses2)

    conditions1 = _extract_join_conditions_from_clauses(clauses1, alias1)
    conditions2 = _extract_join_conditions_from_clauses(clauses2, alias2)

    join_types1 = _extract_join_types(clauses1)
    join_types2 = _extract_join_types(clauses2)
    semantics1 = _normalize_join_semantics(join_types1)
    semantics2 = _normalize_join_semantics(join_types2)

    tables_match = (tables1 == tables2)
    conditions_match = (conditions1 == conditions2)
    join_semantics_match = (semantics1 == semantics2)

    is_equal = tables_match and conditions_match and join_semantics_match

    details = {
        "tables1": sorted(tables1),
        "tables2": sorted(tables2),
        "tables_match": tables_match,
        "conditions_match": conditions_match,
        "join_semantics1": semantics1,
        "join_semantics2": semantics2,
        "join_semantics_match": join_semantics_match,
        "is_equal": is_equal,
    }

    return is_equal, details


# ============================================================================
# SQL 执行工具
# ============================================================================

def _execute_sql(database: str, query: str, timeout: int = 15) -> Tuple[bool, List[Tuple]]:
    """执行 SQL 并返回结果（带真正的查询执行超时保护）.

    使用 threading.Timer + conn.interrupt() 确保 SQLite C层长查询可被中断。
    注意: sqlite3.connect(timeout=N) 仅控制锁等待时间，不控制查询执行时间。
    """
    import threading

    try:
        conn = sqlite3.connect(database)
        conn.text_factory = lambda b: b.decode(errors="ignore")
        _timed_out = False

        def _interrupt():
            nonlocal _timed_out
            _timed_out = True
            try:
                conn.interrupt()
            except Exception:
                pass

        timer = threading.Timer(timeout, _interrupt)
        timer.start()
        try:
            cursor = conn.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            timer.cancel()
            cursor.close()
            conn.close()
            return True, results
        except sqlite3.OperationalError as e:
            timer.cancel()
            conn.close()
            if _timed_out or "interrupted" in str(e).lower():
                logger.debug(f"SQL execution timed out ({timeout}s): {query[:120]}")
            else:
                logger.debug(f"SQL execution failed: {e}, query: {query[:120]}")
            return False, []
        except Exception as e:
            timer.cancel()
            conn.close()
            logger.debug(f"SQL execution failed: {e}, query: {query[:120]}")
            return False, []
    except Exception as e:
        logger.debug(f"SQL connection failed: {e}, query: {query[:120]}")
        return False, []


def _ordered_result_eq(result1: List[Tuple], result2: List[Tuple]) -> bool:
    """严格有序比较 - 行顺序和列顺序都必须一致."""
    return result1 == result2


# ============================================================================
# 动态阶段确定
# ============================================================================

# 逻辑求值顺序（固定）
FULL_STAGE_ORDER = [
    SQLStage.FROM_JOIN,
    SQLStage.WHERE,
    SQLStage.GROUP_BY,
    SQLStage.HAVING,
    SQLStage.SELECT,
    SQLStage.ORDER_BY,
]


def determine_active_stages(gold_clauses: SQLClauses) -> List[SQLStage]:
    """根据 gold SQL 的实际子句确定活跃阶段.

    FROM/JOIN 和 SELECT 始终存在，其余根据 gold SQL 是否有对应子句决定。
    """
    active = []
    for stage in FULL_STAGE_ORDER:
        if stage == SQLStage.FROM_JOIN:
            active.append(stage)  # 始终存在
        elif stage == SQLStage.WHERE:
            if gold_clauses.where:
                active.append(stage)
        elif stage == SQLStage.GROUP_BY:
            if gold_clauses.group_by:
                active.append(stage)
        elif stage == SQLStage.HAVING:
            if gold_clauses.having:
                active.append(stage)
        elif stage == SQLStage.SELECT:
            active.append(stage)  # 始终存在
        elif stage == SQLStage.ORDER_BY:
            if gold_clauses.order_by:
                active.append(stage)
    return active


# ============================================================================
# BIRD 动态分层奖励计算器
# ============================================================================

class BirdDynamicRewardCalculator:
    """BIRD 版动态分层奖励计算器.

    全阶段执行语义驱动，奖励值根据活跃阶段数动态计算。
    """

    BINARY_REWARD_TIMEOUT = 20  # binary_reward 超时(秒)

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._binary_reward = get_binary_reward_func()
        # 最终匹配用 BIRD 口径(binary_reward_bird, 列顺序敏感), 对齐评测/采样; SELECT 中间层仍用宽松的 binary_reward
        self._final_binary_reward = _binary_reward_bird_func
        self._simple_binary_reward = _simple_binary_reward_func

    # ---------------------------------------------------------------
    # 计时包装
    # ---------------------------------------------------------------

    def _timed_binary_reward(
        self, pred_sql: str, gold_sql: str, stats: PerformanceStats, context: str = ""
    ) -> float:
        """带超时保护的 binary_reward（用于 SELECT 阶段和最终匹配）."""
        start = time.perf_counter()
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self._binary_reward, pred_sql, gold_sql, self.db_path, False
                )
                result = future.result(timeout=self.BINARY_REWARD_TIMEOUT)
        except FuturesTimeoutError:
            elapsed = time.perf_counter() - start
            logger.warning(
                f"binary_reward timeout ({elapsed:.1f}s > {self.BINARY_REWARD_TIMEOUT}s), "
                f"context={context}"
            )
            stats.binary_reward_call_count += 1
            stats.binary_reward_time += elapsed
            return 0.0
        except Exception:
            elapsed = time.perf_counter() - start
            logger.exception(f"binary_reward error, context={context}")
            stats.binary_reward_call_count += 1
            stats.binary_reward_time += elapsed
            return 0.0

        elapsed = time.perf_counter() - start
        stats.binary_reward_call_count += 1
        stats.binary_reward_time += elapsed
        return result

    def _timed_final_binary_reward(
        self, pred_sql: str, gold_sql: str, stats: PerformanceStats, context: str = ""
    ) -> float:
        """最终匹配奖励(BIRD 口径 binary_reward_bird, 列顺序敏感, 对齐评测), 带超时保护."""
        start = time.perf_counter()
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self._final_binary_reward, pred_sql, gold_sql, self.db_path, False
                )
                result = future.result(timeout=self.BINARY_REWARD_TIMEOUT)
        except FuturesTimeoutError:
            elapsed = time.perf_counter() - start
            logger.warning(
                f"final binary_reward_bird timeout ({elapsed:.1f}s > {self.BINARY_REWARD_TIMEOUT}s), "
                f"context={context}"
            )
            stats.binary_reward_call_count += 1
            stats.binary_reward_time += elapsed
            return 0.0
        except Exception:
            elapsed = time.perf_counter() - start
            logger.exception(f"final binary_reward_bird error, context={context}")
            stats.binary_reward_call_count += 1
            stats.binary_reward_time += elapsed
            return 0.0

        elapsed = time.perf_counter() - start
        stats.binary_reward_call_count += 1
        stats.binary_reward_time += elapsed
        return result

    def _timed_simple_binary_reward(
        self, pred_sql: str, gold_sql: str, stats: PerformanceStats, context: str = ""
    ) -> float:
        """带计时的 simple_binary_reward（忽略行序，保留列序）."""
        start = time.perf_counter()
        try:
            result = self._simple_binary_reward(pred_sql, gold_sql, self.db_path, False)
        except Exception:
            elapsed = time.perf_counter() - start
            logger.exception(f"simple_binary_reward error, context={context}")
            stats.simple_binary_reward_call_count += 1
            stats.simple_binary_reward_time += elapsed
            return 0.0

        elapsed = time.perf_counter() - start
        stats.simple_binary_reward_call_count += 1
        stats.simple_binary_reward_time += elapsed
        return result

    def _timed_ordered_compare(
        self, pred_sql: str, gold_sql: str, stats: PerformanceStats, context: str = ""
    ) -> float:
        """严格有序对比（行序+列序敏感），用于 ORDER BY 阶段."""
        start = time.perf_counter()
        try:
            p_ok, p_res = _execute_sql(self.db_path, pred_sql)
            g_ok, g_res = _execute_sql(self.db_path, gold_sql)
            if not p_ok or not g_ok:
                stats.ordered_compare_call_count += 1
                return 0.0
            match = 1.0 if _ordered_result_eq(p_res, g_res) else 0.0
        except Exception:
            logger.exception(f"ordered_compare error, context={context}")
            stats.ordered_compare_call_count += 1
            return 0.0

        elapsed = time.perf_counter() - start
        stats.ordered_compare_call_count += 1
        return match

    # ---------------------------------------------------------------
    # 别名重映射工具
    # ---------------------------------------------------------------

    @staticmethod
    def _build_from_base(clauses: SQLClauses) -> str:
        """构建 FROM 基础部分（含 JOINs）."""
        if not clauses.from_clause:
            return ""
        base = f"FROM {clauses.from_clause}"
        if clauses.joins:
            base += " " + " ".join(clauses.joins)
        return base

    @staticmethod
    def _remap_clause_aliases(
        clause: str, pred_clauses: SQLClauses, gold_clauses: SQLClauses
    ) -> str:
        """将子句中 pred 的表别名替换为 gold 的别名体系."""
        pred_alias_map = _extract_alias_mapping(pred_clauses)
        gold_alias_map = _extract_alias_mapping(gold_clauses)

        gold_table_to_alias: Dict[str, str] = {}
        for alias, table in gold_alias_map.items():
            if table not in gold_table_to_alias:
                gold_table_to_alias[table] = alias
            else:
                return clause  # 自连接无法自动映射

        alias_remap: Dict[str, str] = {}
        for pred_alias, pred_table in pred_alias_map.items():
            if pred_table in gold_table_to_alias:
                gold_alias = gold_table_to_alias[pred_table]
                if pred_alias != gold_alias:
                    alias_remap[pred_alias] = gold_alias

        if not alias_remap:
            return clause

        result = clause
        for old_alias, new_alias in alias_remap.items():
            result = re.sub(
                rf'\b{re.escape(old_alias)}\.',
                f'{new_alias}.',
                result,
                flags=re.I,
            )
        return result

    # ---------------------------------------------------------------
    # 中间 SQL 构建
    # ---------------------------------------------------------------

    def _build_where_sql(self, pred_clauses: SQLClauses, gold_clauses: SQLClauses) -> Tuple[str, str]:
        """WHERE: SELECT * FROM gold_from WHERE pred/gold_where."""
        gold_from = self._build_from_base(gold_clauses)
        remapped = self._remap_clause_aliases(pred_clauses.where, pred_clauses, gold_clauses)
        return (
            f"SELECT * {gold_from} WHERE {remapped}",
            f"SELECT * {gold_from} WHERE {gold_clauses.where}",
        )

    def _build_group_by_sql(self, pred_clauses: SQLClauses, gold_clauses: SQLClauses) -> Tuple[str, str]:
        """GROUP BY: SELECT group_exprs FROM ... [WHERE ...] GROUP BY group_exprs."""
        gold_from = self._build_from_base(gold_clauses)
        remapped_gb = self._remap_clause_aliases(pred_clauses.group_by, pred_clauses, gold_clauses)

        tail = f" WHERE {gold_clauses.where}" if gold_clauses.where else ""
        return (
            f"SELECT {remapped_gb} {gold_from}{tail} GROUP BY {remapped_gb}",
            f"SELECT {gold_clauses.group_by} {gold_from}{tail} GROUP BY {gold_clauses.group_by}",
        )

    def _build_having_sql(self, pred_clauses: SQLClauses, gold_clauses: SQLClauses) -> Tuple[str, str]:
        """HAVING: SELECT group_exprs FROM ... [WHERE ...] GROUP BY ... HAVING ..."""
        gold_from = self._build_from_base(gold_clauses)
        remapped_having = self._remap_clause_aliases(pred_clauses.having, pred_clauses, gold_clauses)

        tail = f" WHERE {gold_clauses.where}" if gold_clauses.where else ""
        # 用 gold 的 GROUP BY（前面已通过）
        base = f"SELECT {gold_clauses.group_by} {gold_from}{tail} GROUP BY {gold_clauses.group_by}"
        return (
            f"{base} HAVING {remapped_having}",
            f"{base} HAVING {gold_clauses.having}",
        )

    def _build_select_sql(self, pred_clauses: SQLClauses, gold_clauses: SQLClauses) -> Tuple[str, str]:
        """SELECT: 用 binary_reward 对比，处理列顺序差异."""
        gold_from = self._build_from_base(gold_clauses)
        remapped_select = self._remap_clause_aliases(pred_clauses.select, pred_clauses, gold_clauses)

        # 公共尾部（前面阶段已通过，全用 gold 的）
        tail = ""
        if gold_clauses.where:
            tail += f" WHERE {gold_clauses.where}"
        if gold_clauses.group_by:
            tail += f" GROUP BY {gold_clauses.group_by}"
        if gold_clauses.having:
            tail += f" HAVING {gold_clauses.having}"

        dp = "DISTINCT " if pred_clauses.distinct else ""
        dg = "DISTINCT " if gold_clauses.distinct else ""
        return (
            f"SELECT {dp}{remapped_select} {gold_from}{tail}",
            f"SELECT {dg}{gold_clauses.select} {gold_from}{tail}",
        )

    def _build_order_by_sql(self, pred_clauses: SQLClauses, gold_clauses: SQLClauses) -> Tuple[str, str]:
        """ORDER BY: 严格有序对比，验证行序."""
        gold_from = self._build_from_base(gold_clauses)
        remapped_ob = self._remap_clause_aliases(pred_clauses.order_by, pred_clauses, gold_clauses)

        tail = ""
        if gold_clauses.where:
            tail += f" WHERE {gold_clauses.where}"
        if gold_clauses.group_by:
            tail += f" GROUP BY {gold_clauses.group_by}"
        if gold_clauses.having:
            tail += f" HAVING {gold_clauses.having}"

        # 用 gold 的 SELECT（前面 SELECT 已通过）
        dg = "DISTINCT " if gold_clauses.distinct else ""
        base = f"SELECT {dg}{gold_clauses.select} {gold_from}{tail}"

        pred_sql = f"{base} ORDER BY {remapped_ob}"
        gold_sql = f"{base} ORDER BY {gold_clauses.order_by}"
        if pred_clauses.limit:
            pred_sql += f" LIMIT {pred_clauses.limit}"
        if gold_clauses.limit:
            gold_sql += f" LIMIT {gold_clauses.limit}"
        return pred_sql, gold_sql

    # ---------------------------------------------------------------
    # 各阶段比较分发
    # ---------------------------------------------------------------

    def _compare_stage(
        self,
        stage: SQLStage,
        pred_clauses: SQLClauses,
        gold_clauses: SQLClauses,
        stats: PerformanceStats,
    ) -> Dict[str, Any]:
        """根据阶段分发到对应的比较逻辑."""
        if stage == SQLStage.FROM_JOIN:
            is_eq, det = compare_from_join_bird(pred_clauses, gold_clauses)
            return {"match": is_eq, **det}

        if stage == SQLStage.WHERE:
            return self._exec_compare_simple(
                self._build_where_sql, pred_clauses, gold_clauses, stats, "where"
            )

        if stage == SQLStage.GROUP_BY:
            return self._exec_compare_simple(
                self._build_group_by_sql, pred_clauses, gold_clauses, stats, "group_by"
            )

        if stage == SQLStage.HAVING:
            return self._exec_compare_simple(
                self._build_having_sql, pred_clauses, gold_clauses, stats, "having"
            )

        if stage == SQLStage.SELECT:
            return self._exec_compare_binary(pred_clauses, gold_clauses, stats)

        if stage == SQLStage.ORDER_BY:
            return self._exec_compare_ordered(pred_clauses, gold_clauses, stats)

        return {"match": False, "reason": "unknown_stage"}

    def _exec_compare_simple(
        self,
        build_fn,
        pred_clauses: SQLClauses,
        gold_clauses: SQLClauses,
        stats: PerformanceStats,
        context: str,
    ) -> Dict[str, Any]:
        """通用 simple_binary_reward 对比."""
        try:
            pred_sql, gold_sql = build_fn(pred_clauses, gold_clauses)
            match = self._timed_simple_binary_reward(pred_sql, gold_sql, stats, context)
            return {"match": (match == 1.0), "reason": "exec_match" if match == 1.0 else "exec_mismatch"}
        except Exception as e:
            logger.debug(f"{context} comparison failed: {e}")
            return {"match": False, "reason": f"error: {e}"}

    def _exec_compare_binary(
        self,
        pred_clauses: SQLClauses,
        gold_clauses: SQLClauses,
        stats: PerformanceStats,
    ) -> Dict[str, Any]:
        """binary_reward 对比（SELECT 阶段，含列排列枚举）."""
        try:
            pred_sql, gold_sql = self._build_select_sql(pred_clauses, gold_clauses)
            match = self._timed_binary_reward(pred_sql, gold_sql, stats, "select")
            return {"match": (match == 1.0), "reason": "exec_match" if match == 1.0 else "exec_mismatch"}
        except Exception as e:
            logger.debug(f"SELECT comparison failed: {e}")
            return {"match": False, "reason": f"error: {e}"}

    def _exec_compare_ordered(
        self,
        pred_clauses: SQLClauses,
        gold_clauses: SQLClauses,
        stats: PerformanceStats,
    ) -> Dict[str, Any]:
        """严格有序对比（ORDER BY 阶段）."""
        try:
            pred_sql, gold_sql = self._build_order_by_sql(pred_clauses, gold_clauses)
            match = self._timed_ordered_compare(pred_sql, gold_sql, stats, "order_by")
            return {"match": (match == 1.0), "reason": "exec_match" if match == 1.0 else "exec_mismatch"}
        except Exception as e:
            logger.debug(f"ORDER BY comparison failed: {e}")
            return {"match": False, "reason": f"error: {e}"}

    # ---------------------------------------------------------------
    # 主入口
    # ---------------------------------------------------------------

    def calculate_reward(
        self,
        pred_sql: str,
        gold_sql: str,
    ) -> Tuple[float, Dict[str, Any]]:
        """计算 BIRD 动态分层奖励.

        流程：
        1. binary_reward 最终匹配 → 1.0
        2. 复合查询不匹配 → 0.0
        3. 确定活跃阶段 + 动态计算奖励值
        4. 从前往后逐阶段比较，返回最远通过阶段的奖励
        """
        total_start = time.perf_counter()
        stats = PerformanceStats()

        details: Dict[str, Any] = {
            "pred_sql": pred_sql,
            "gold_sql": gold_sql,
            "final_match": False,
            "is_compound": False,
            "active_stages": [],
            "stage_rewards": {},
            "stage_results": {},
            "last_passed_stage": None,
            "reward_type": "",
            "reward": 0.0,
        }

        def finalize(reward: float) -> Tuple[float, Dict[str, Any]]:
            stats.total_time = time.perf_counter() - total_start
            details["performance_stats"] = stats.to_dict()
            details["reward"] = reward
            return reward, details

        # Step 1: 最终结果匹配
        try:
            final_match = self._timed_final_binary_reward(pred_sql, gold_sql, stats, "final_match")
            details["final_match"] = (final_match == 1.0)
            if final_match == 1.0:
                details["reward_type"] = "final_match"
                return finalize(FINAL_MATCH_REWARD)
        except Exception as e:
            logger.warning(f"Final comparison failed: {e}")

        # Step 2: 复合查询检查
        pred_parser = SQLParser(pred_sql)
        gold_parser = SQLParser(gold_sql)

        if pred_parser.is_compound() or gold_parser.is_compound():
            details["is_compound"] = True
            details["reward_type"] = "compound_no_match"
            return finalize(0.0)

        # Step 3: 提取子句
        pred_clauses = pred_parser.extract_clauses()
        gold_clauses = gold_parser.extract_clauses()

        # Step 4: 确定活跃阶段 + 动态奖励
        active_stages = determine_active_stages(gold_clauses)
        stage_rewards = compute_stage_rewards(active_stages)

        details["active_stages"] = [s.value for s in active_stages]
        details["stage_rewards"] = {s.value: r for s, r in stage_rewards.items()}

        # Step 5: 逐阶段比较
        if REWARD_AGG_MODE == "sum":
            # 累加模式: 各阶段独立比较, 按通过比例给分(与顺序无关), 修正 gating 对非前缀式正确的低估
            passed = 0
            for stage in active_stages:
                pred_missing = (
                    (stage == SQLStage.WHERE and not pred_clauses.where)
                    or (stage == SQLStage.GROUP_BY and not pred_clauses.group_by)
                    or (stage == SQLStage.HAVING and not pred_clauses.having)
                    or (stage == SQLStage.ORDER_BY and not pred_clauses.order_by)
                )
                if pred_missing:
                    details["stage_results"][stage.value] = {"match": False, "reason": "pred_missing"}
                    continue
                stage_result = self._compare_stage(stage, pred_clauses, gold_clauses, stats)
                details["stage_results"][stage.value] = stage_result
                if stage_result["match"]:
                    passed += 1
            reward = round(passed / len(active_stages) * MAX_STAGE_REWARD, 4) if active_stages else 0.0
            details["reward_type"] = f"sum_{passed}/{len(active_stages)}"
            return finalize(reward)

        # Step 5(weighted_sum): 各阶段独立比较, 但按 stage_rewards 档位加权(软层级)
        if REWARD_AGG_MODE == "weighted_sum":
            total_w = sum(stage_rewards[s] for s in active_stages)
            passed_w = 0.0
            for stage in active_stages:
                pred_missing = (
                    (stage == SQLStage.WHERE and not pred_clauses.where)
                    or (stage == SQLStage.GROUP_BY and not pred_clauses.group_by)
                    or (stage == SQLStage.HAVING and not pred_clauses.having)
                    or (stage == SQLStage.ORDER_BY and not pred_clauses.order_by)
                )
                if pred_missing:
                    details["stage_results"][stage.value] = {"match": False, "reason": "pred_missing"}
                    continue
                stage_result = self._compare_stage(stage, pred_clauses, gold_clauses, stats)
                details["stage_results"][stage.value] = stage_result
                if stage_result["match"]:
                    passed_w += stage_rewards[stage]
            reward = round(passed_w / total_w * MAX_STAGE_REWARD, 4) if total_w > 0 else 0.0
            details["reward_type"] = f"weighted_sum_{round(passed_w, 3)}/{round(total_w, 3)}"
            return finalize(reward)

        # Step 5(gating 默认): 逐阶段比较（从前往后）
        last_passed_reward = 0.0

        for stage in active_stages:
            # 对于非 FROM 阶段，检查 pred 侧是否缺失该子句
            if stage == SQLStage.WHERE and not pred_clauses.where:
                details["stage_results"][stage.value] = {"match": False, "reason": "pred_missing"}
                details["reward_type"] = f"mismatch_at_{stage.value}"
                return finalize(last_passed_reward)
            if stage == SQLStage.GROUP_BY and not pred_clauses.group_by:
                details["stage_results"][stage.value] = {"match": False, "reason": "pred_missing"}
                details["reward_type"] = f"mismatch_at_{stage.value}"
                return finalize(last_passed_reward)
            if stage == SQLStage.HAVING and not pred_clauses.having:
                details["stage_results"][stage.value] = {"match": False, "reason": "pred_missing"}
                details["reward_type"] = f"mismatch_at_{stage.value}"
                return finalize(last_passed_reward)
            if stage == SQLStage.ORDER_BY and not pred_clauses.order_by:
                details["stage_results"][stage.value] = {"match": False, "reason": "pred_missing"}
                details["reward_type"] = f"mismatch_at_{stage.value}"
                return finalize(last_passed_reward)

            # 执行比较
            stage_result = self._compare_stage(stage, pred_clauses, gold_clauses, stats)
            details["stage_results"][stage.value] = stage_result

            if not stage_result["match"]:
                details["reward_type"] = f"mismatch_at_{stage.value}"
                return finalize(last_passed_reward)

            # 通过，更新奖励
            last_passed_reward = stage_rewards[stage]
            details["last_passed_stage"] = stage.value

        # 所有阶段通过
        details["reward_type"] = "all_stages_passed"
        return finalize(last_passed_reward)


# ============================================================================
# 对外接口
# ============================================================================

def composite_reward_dynamic_bird(
    pred_sql: str,
    gold_sql: str,
    db_path: str,
    raise_on_error: bool = False,
) -> float:
    """计算 BIRD 动态分层奖励（简化接口）.

    Args:
        pred_sql: 预测 SQL
        gold_sql: 标准 SQL
        db_path: 数据库路径
        raise_on_error: 是否抛出异常

    Returns:
        奖励值 (0.0 ~ 1.0)
    """
    try:
        calculator = BirdDynamicRewardCalculator(db_path)
        reward, _ = calculator.calculate_reward(pred_sql, gold_sql)
        return reward
    except Exception as e:
        if raise_on_error:
            raise
        logger.exception(f"BIRD dynamic reward calculation failed: {e}")
        return 0.0


def composite_reward_dynamic_bird_with_details(
    pred_sql: str,
    gold_sql: str,
    db_path: str,
    raise_on_error: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    """计算 BIRD 动态分层奖励并返回详细信息.

    Args:
        pred_sql: 预测 SQL
        gold_sql: 标准 SQL
        db_path: 数据库路径
        raise_on_error: 是否抛出异常

    Returns:
        (reward, details)
    """
    try:
        calculator = BirdDynamicRewardCalculator(db_path)
        return calculator.calculate_reward(pred_sql, gold_sql)
    except Exception as e:
        if raise_on_error:
            raise
        logger.exception(f"BIRD dynamic reward calculation failed: {e}")
        return 0.0, {"error": str(e)}
