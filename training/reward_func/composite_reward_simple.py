"""简化分层复合奖励系统 - 只对关键阶段进行奖励计算.

将7个阶段合并为3个关键阶段：
1. FROM_JOIN: 选表阶段 (0.30)
2. FILTER_GROUP: 过滤+分组阶段 (WHERE/GROUP_BY/HAVING) (0.50)
3. PROJECTION: 投影阶段 (SELECT/ORDER_BY) (0.70)
4. Final: 最终匹配 (1.0)

设计理念：
- 减少中间噪声，只保留关键学习信号
- 每个阶段都有明确的学习目标
- ORDER BY 很少单独出错，与 SELECT 合并处理
"""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Tuple

from reward_func.composite_reward_fast import (
    IntermediateSQLBuilder,
    SQLClauses,
    SQLParser,
    SQLStage,
    get_binary_reward_func,
)
from reward_func.reward import simple_binary_reward as _simple_binary_reward_func

logger = logging.getLogger(__name__)


# ============================================================================
# 关键阶段定义
# ============================================================================

class KeyStage(Enum):
    """关键阶段枚举."""
    FROM_JOIN = "from_join"           # 选表阶段
    FILTER_GROUP = "filter_group"     # 过滤+分组阶段
    PROJECTION = "projection"         # 投影阶段（SELECT/ORDER_BY）


# 关键阶段奖励
KEY_STAGE_REWARDS = {
    KeyStage.FROM_JOIN: 0.30,
    KeyStage.FILTER_GROUP: 0.50,
    KeyStage.PROJECTION: 0.70,
}

# 原始阶段到关键阶段的映射
STAGE_TO_KEY_STAGE = {
    SQLStage.FROM_JOIN: KeyStage.FROM_JOIN,
    SQLStage.WHERE: KeyStage.FILTER_GROUP,
    SQLStage.GROUP_BY: KeyStage.FILTER_GROUP,
    SQLStage.HAVING: KeyStage.FILTER_GROUP,
    SQLStage.SELECT: KeyStage.PROJECTION,
    SQLStage.ORDER_BY: KeyStage.PROJECTION,
}

# 关键阶段的比较顺序
KEY_STAGE_ORDER = [
    KeyStage.FROM_JOIN,
    KeyStage.FILTER_GROUP,
    KeyStage.PROJECTION,
]


# ============================================================================
# 性能统计
# ============================================================================

@dataclass
class PerformanceStats:
    """性能统计信息."""
    binary_reward_call_count: int = 0
    binary_reward_time: float = 0.0
    total_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "binary_reward_call_count": self.binary_reward_call_count,
            "binary_reward_time_sec": round(self.binary_reward_time, 6),
            "total_time_sec": round(self.total_time, 6),
        }


# ============================================================================
# 简化奖励计算器
# ============================================================================

class SimpleLayeredRewardCalculator:
    """简化分层奖励计算器.
    
    只对3个关键阶段进行奖励计算：
    1. FROM_JOIN: 选表正确
    2. FILTER_GROUP: 过滤+分组正确
    3. PROJECTION_ORDER: 投影+排序正确
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._binary_reward = get_binary_reward_func()
        self._simple_binary_reward = _simple_binary_reward_func
    
    # binary_reward 单次调用超时(秒)，防止列排列枚举爆炸
    BINARY_REWARD_TIMEOUT = 20

    def _timed_binary_reward(
        self,
        pred_sql: str,
        gold_sql: str,
        stats: PerformanceStats,
        context: str = "",
    ) -> float:
        """带计时和超时保护的 binary_reward 调用.
        
        使用线程池执行，超时后返回0.0，避免列排列枚举卡死Worker.
        """
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
                f"binary_reward timeout ({elapsed:.1f}s > {self.BINARY_REWARD_TIMEOUT}s) "
                f"context={context}, pred_sql={pred_sql[:80]}..."
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
    
    def calculate_reward(
        self,
        pred_sql: str,
        gold_sql: str,
    ) -> Tuple[float, Dict[str, Any]]:
        """计算简化分层奖励.
        
        Args:
            pred_sql: 预测SQL
            gold_sql: 标准SQL
        
        Returns:
            (reward, details)
        """
        total_start = time.perf_counter()
        stats = PerformanceStats()
        
        details: Dict[str, Any] = {
            "pred_sql": pred_sql,
            "gold_sql": gold_sql,
            "final_match": False,
            "is_compound": False,
            "key_stage_results": {},
            "last_passed_stage": None,
            "reward_type": "",
            "reward": 0.0,
        }
        
        def finalize_and_return(reward: float) -> Tuple[float, Dict[str, Any]]:
            stats.total_time = time.perf_counter() - total_start
            details["performance_stats"] = stats.to_dict()
            details["reward"] = reward
            return reward, details
        
        # Step 1: 检查最终结果是否匹配
        try:
            final_match = self._timed_binary_reward(
                pred_sql, gold_sql, stats, context="final_match_check"
            )
            details["final_match"] = (final_match == 1.0)
            
            if final_match == 1.0:
                details["reward_type"] = "final_match"
                return finalize_and_return(1.0)
                
        except Exception as e:
            logger.warning(f"Final comparison failed: {e}")
            details["error"] = str(e)
        
        # Step 2: 检查是否是复合查询
        pred_parser = SQLParser(pred_sql)
        gold_parser = SQLParser(gold_sql)
        
        if pred_parser.is_compound() or gold_parser.is_compound():
            details["is_compound"] = True
            details["reward_type"] = "compound_no_match"
            return finalize_and_return(0.0)
        
        # Step 3: 提取子句
        pred_clauses = pred_parser.extract_clauses()
        gold_clauses = gold_parser.extract_clauses()
        
        details["pred_clauses"] = {
            "from": pred_clauses.from_clause,
            "joins": pred_clauses.joins,
            "where": pred_clauses.where,
            "group_by": pred_clauses.group_by,
            "having": pred_clauses.having,
            "select": pred_clauses.select,
            "distinct": pred_clauses.distinct,
            "order_by": pred_clauses.order_by,
        }
        details["gold_clauses"] = {
            "from": gold_clauses.from_clause,
            "joins": gold_clauses.joins,
            "where": gold_clauses.where,
            "group_by": gold_clauses.group_by,
            "having": gold_clauses.having,
            "select": gold_clauses.select,
            "distinct": gold_clauses.distinct,
            "order_by": gold_clauses.order_by,
        }
        
        # Step 4: 按关键阶段比较
        last_passed_reward = 0.0
        
        # 关键阶段1: FROM_JOIN（选表）
        from_result = self._compare_from_join(pred_clauses, gold_clauses)
        details["key_stage_results"]["from_join"] = from_result
        
        if not from_result["match"]:
            details["reward_type"] = "from_join_mismatch"
            return finalize_and_return(0.0)
        
        last_passed_reward = KEY_STAGE_REWARDS[KeyStage.FROM_JOIN]
        details["last_passed_stage"] = KeyStage.FROM_JOIN.value
        
        # 关键阶段2: FILTER_GROUP（过滤+分组）
        filter_result = self._compare_filter_group(
            pred_clauses, gold_clauses, pred_sql, gold_sql, stats
        )
        details["key_stage_results"]["filter_group"] = filter_result
        
        if not filter_result["match"]:
            details["reward_type"] = f"filter_group_mismatch_at_{filter_result.get('failed_at', 'unknown')}"
            return finalize_and_return(last_passed_reward)
        
        last_passed_reward = KEY_STAGE_REWARDS[KeyStage.FILTER_GROUP]
        details["last_passed_stage"] = KeyStage.FILTER_GROUP.value
        
        # 关键阶段3: PROJECTION（投影）
        projection_result = self._compare_projection(pred_clauses, gold_clauses)
        details["key_stage_results"]["projection"] = projection_result
        
        if not projection_result["match"]:
            details["reward_type"] = f"projection_mismatch_at_{projection_result.get('failed_at', 'unknown')}"
            return finalize_and_return(last_passed_reward)
        
        # 所有阶段通过
        last_passed_reward = KEY_STAGE_REWARDS[KeyStage.PROJECTION]
        details["last_passed_stage"] = KeyStage.PROJECTION.value
        details["reward_type"] = "all_key_stages_passed"
        return finalize_and_return(last_passed_reward)
    
    def _compare_from_join(
        self,
        pred_clauses: SQLClauses,
        gold_clauses: SQLClauses,
    ) -> Dict[str, Any]:
        """比较FROM_JOIN阶段."""
        from reward_func.composite_reward_fast import (
            _extract_tables_from_clauses,
            _extract_alias_mapping,
            _extract_join_conditions_from_clauses,
            compare_from_join_structure,
        )
        
        is_equal, compare_details = compare_from_join_structure(pred_clauses, gold_clauses)
        
        return {
            "match": is_equal,
            "pred_tables": compare_details.get("tables1", []),
            "gold_tables": compare_details.get("tables2", []),
            "tables_match": compare_details.get("tables_match", False),
            "join_conditions_match": compare_details.get("conditions_match", False),
        }
    
    def _compare_filter_group(
        self,
        pred_clauses: SQLClauses,
        gold_clauses: SQLClauses,
        pred_sql: str,
        gold_sql: str,
        stats: PerformanceStats,
    ) -> Dict[str, Any]:
        """比较FILTER_GROUP阶段（WHERE/GROUP_BY/HAVING）."""
        result = {
            "match": True,
            "failed_at": None,
            "where": None,
            "group_by": None,
            "having": None,
        }
        
        # 检查WHERE
        where_result = self._compare_where(pred_clauses, gold_clauses, pred_sql, gold_sql, stats)
        result["where"] = where_result
        
        if not where_result["match"]:
            result["match"] = False
            result["failed_at"] = "where"
            return result
        
        # 检查GROUP_BY
        group_result = self._compare_group_by(pred_clauses, gold_clauses)
        result["group_by"] = group_result
        
        if not group_result["match"]:
            result["match"] = False
            result["failed_at"] = "group_by"
            return result
        
        # 检查HAVING
        having_result = self._compare_having(pred_clauses, gold_clauses, pred_sql, gold_sql, stats)
        result["having"] = having_result
        
        if not having_result["match"]:
            result["match"] = False
            result["failed_at"] = "having"
            return result
        
        return result
    
    @staticmethod
    def _build_from_base(clauses: SQLClauses) -> str:
        """构建FROM基础部分."""
        if not clauses.from_clause:
            return ""
        base = f"FROM {clauses.from_clause}"
        if clauses.joins:
            base += " " + " ".join(clauses.joins)
        return base

    @staticmethod
    def _remap_clause_aliases(clause: str, pred_clauses: SQLClauses, gold_clauses: SQLClauses) -> str:
        """将子句中 pred 的别名替换为 gold 的别名体系.
        
        例如 pred 用 t1/t2，gold 用 s1/s2，两者指向相同的表，
        则将子句中的 t1 替换为 s1、t2 替换为 s2。
        这样配合 FROM 对齐才能正确执行 SQL。
        """
        from reward_func.composite_reward_fast import _extract_alias_mapping
        
        pred_alias_map = _extract_alias_mapping(pred_clauses)
        gold_alias_map = _extract_alias_mapping(gold_clauses)
        
        # 构建 gold 的反向映射：table -> alias
        gold_table_to_alias: Dict[str, str] = {}
        for alias, table in gold_alias_map.items():
            if table not in gold_table_to_alias:
                gold_table_to_alias[table] = alias
            else:
                # 自连接：同一表多个别名，无法自动映射，留原样
                return clause
        
        # 构建 pred_alias -> gold_alias 的映射
        alias_remap: Dict[str, str] = {}
        for pred_alias, pred_table in pred_alias_map.items():
            if pred_table in gold_table_to_alias:
                gold_alias = gold_table_to_alias[pred_table]
                if pred_alias != gold_alias:
                    alias_remap[pred_alias] = gold_alias
        
        if not alias_remap:
            return clause
        
        # 替换子句中的别名（只替换 alias.column 格式）
        result = clause
        for old_alias, new_alias in alias_remap.items():
            result = re.sub(
                rf'\b{re.escape(old_alias)}\.',
                f'{new_alias}.',
                result,
                flags=re.I
            )
        return result

    def _build_aligned_where_sql(self, pred_clauses: SQLClauses, gold_clauses: SQLClauses) -> Tuple[str, str]:
        """构建FROM对齐的WHERE中间SQL.
        
        pred的FROM部分替换为gold的FROM，确保SELECT *列顺序一致，
        从而可以用simple_binary_reward比较，无需列排列枚举。
        """
        gold_from_base = self._build_from_base(gold_clauses)
        remapped_where = self._remap_clause_aliases(pred_clauses.where, pred_clauses, gold_clauses)
        pred_sql = f"SELECT * {gold_from_base} WHERE {remapped_where}"
        gold_sql = f"SELECT * {gold_from_base} WHERE {gold_clauses.where}"
        return pred_sql, gold_sql

    def _build_aligned_having_sql(self, pred_clauses: SQLClauses, gold_clauses: SQLClauses) -> Tuple[str, str]:
        """构建FROM对齐的HAVING中间SQL.
        
        pred的FROM部分替换为gold的FROM，别名也相应替换。
        """
        gold_from_base = self._build_from_base(gold_clauses)
        
        remapped_where = self._remap_clause_aliases(pred_clauses.where, pred_clauses, gold_clauses) if pred_clauses.where else ""
        remapped_group_by = self._remap_clause_aliases(pred_clauses.group_by, pred_clauses, gold_clauses) if pred_clauses.group_by else ""
        remapped_having = self._remap_clause_aliases(pred_clauses.having, pred_clauses, gold_clauses) if pred_clauses.having else ""
        
        pred_sql = f"SELECT {remapped_group_by} {gold_from_base}"
        if remapped_where:
            pred_sql += f" WHERE {remapped_where}"
        pred_sql += f" GROUP BY {remapped_group_by} HAVING {remapped_having}"
        
        gold_sql = f"SELECT {gold_clauses.group_by} {gold_from_base}"
        if gold_clauses.where:
            gold_sql += f" WHERE {gold_clauses.where}"
        gold_sql += f" GROUP BY {gold_clauses.group_by} HAVING {gold_clauses.having}"
        
        return pred_sql, gold_sql

    def _timed_simple_binary_reward(
        self,
        pred_sql: str,
        gold_sql: str,
        stats: PerformanceStats,
        context: str = "",
    ) -> float:
        """带计时的 simple_binary_reward 调用（不做列排列枚举）.
        
        用于中间阶段（WHERE/HAVING）比较，配合 FROM 对齐使用。
        """
        start = time.perf_counter()
        try:
            result = self._simple_binary_reward(pred_sql, gold_sql, self.db_path, False)
        except Exception:
            elapsed = time.perf_counter() - start
            logger.exception(f"simple_binary_reward error, context={context}")
            stats.binary_reward_call_count += 1
            stats.binary_reward_time += elapsed
            return 0.0
        
        elapsed = time.perf_counter() - start
        stats.binary_reward_call_count += 1
        stats.binary_reward_time += elapsed
        return result

    def _compare_where(
        self,
        pred_clauses: SQLClauses,
        gold_clauses: SQLClauses,
        pred_sql: str,
        gold_sql: str,
        stats: PerformanceStats,
    ) -> Dict[str, Any]:
        """比较WHERE子句.
        
        使用FROM对齐 + simple_binary_reward，避免列排列枚举。
        """
        result = {
            "match": None,
            "reason": "",
            "pred_exists": bool(pred_clauses.where),
            "gold_exists": bool(gold_clauses.where),
        }
        
        # 两边都没有WHERE，跳过
        if not pred_clauses.where and not gold_clauses.where:
            result["match"] = True
            result["reason"] = "两边都没有WHERE，跳过"
            return result
        
        # 只有一边有WHERE，不通过
        if pred_clauses.where and not gold_clauses.where:
            result["match"] = False
            result["reason"] = "pred有WHERE但gold没有"
            return result
        
        if not pred_clauses.where and gold_clauses.where:
            result["match"] = False
            result["reason"] = "gold有WHERE但pred没有"
            return result
        
        # 两边都有WHERE，FROM对齐后用simple_binary_reward比较
        try:
            pred_where_sql, gold_where_sql = self._build_aligned_where_sql(pred_clauses, gold_clauses)
            match = self._timed_simple_binary_reward(
                pred_where_sql,
                gold_where_sql,
                stats,
                context="where_comparison"
            )
            result["match"] = (match == 1.0)
            result["reason"] = "SQL执行比较通过（FROM对齐）" if match == 1.0 else "SQL执行比较不通过"
        except Exception as e:
            logger.debug(f"WHERE comparison failed: {e}")
            result["match"] = False
            result["reason"] = f"比较失败: {str(e)}"
        
        return result
    
    def _compare_group_by(
        self,
        pred_clauses: SQLClauses,
        gold_clauses: SQLClauses,
    ) -> Dict[str, Any]:
        """比较GROUP_BY子句."""
        from reward_func.composite_reward_fast import compare_group_by
        
        is_equal, compare_details = compare_group_by(pred_clauses, gold_clauses)
        
        return {
            "match": is_equal,
            "pred_columns": compare_details.get("columns1", []),
            "gold_columns": compare_details.get("columns2", []),
        }
    
    def _compare_having(
        self,
        pred_clauses: SQLClauses,
        gold_clauses: SQLClauses,
        pred_sql: str,
        gold_sql: str,
        stats: PerformanceStats,
    ) -> Dict[str, Any]:
        """比较HAVING子句.
        
        使用FROM对齐 + simple_binary_reward，避免列排列枚举。
        """
        result = {
            "match": None,
            "reason": "",
            "pred_exists": bool(pred_clauses.having),
            "gold_exists": bool(gold_clauses.having),
        }
        
        # 两边都没有HAVING，跳过
        if not pred_clauses.having and not gold_clauses.having:
            result["match"] = True
            result["reason"] = "两边都没有HAVING，跳过"
            return result
        
        # 只有一边有HAVING，不通过
        if pred_clauses.having and not gold_clauses.having:
            result["match"] = False
            result["reason"] = "pred有HAVING但gold没有"
            return result
        
        if not pred_clauses.having and gold_clauses.having:
            result["match"] = False
            result["reason"] = "gold有HAVING但pred没有"
            return result
        
        # 两边都有HAVING，FROM对齐后用simple_binary_reward比较
        try:
            pred_having_sql, gold_having_sql = self._build_aligned_having_sql(pred_clauses, gold_clauses)
            match = self._timed_simple_binary_reward(
                pred_having_sql,
                gold_having_sql,
                stats,
                context="having_comparison"
            )
            result["match"] = (match == 1.0)
            result["reason"] = "SQL执行比较通过（FROM对齐）" if match == 1.0 else "SQL执行比较不通过"
        except Exception as e:
            logger.debug(f"HAVING comparison failed: {e}")
            result["match"] = False
            result["reason"] = f"比较失败: {str(e)}"
        
        return result
    
    def _compare_projection(
        self,
        pred_clauses: SQLClauses,
        gold_clauses: SQLClauses,
    ) -> Dict[str, Any]:
        """比较PROJECTION阶段（SELECT/ORDER_BY）.
        
        注意：DISTINCT不检查，遵循Spider官方评测标准（keep_distinct=False）。
        """
        from reward_func.composite_reward_fast import (
            compare_select_columns,
            compare_order_by,
        )
        
        result = {
            "match": True,
            "failed_at": None,
            "select": None,
            "order_by": None,
        }
        
        # 检查SELECT
        select_equal, select_details = compare_select_columns(pred_clauses, gold_clauses)
        result["select"] = {
            "match": select_equal,
            "pred_columns": select_details.get("columns1", []),
            "gold_columns": select_details.get("columns2", []),
        }
        
        if not select_equal:
            result["match"] = False
            result["failed_at"] = "select"
            return result
        
        # 检查ORDER_BY
        order_equal, order_details = compare_order_by(pred_clauses, gold_clauses)
        result["order_by"] = {
            "match": order_equal,
            "pred_order": order_details.get("order_by1", ""),
            "gold_order": order_details.get("order_by2", ""),
        }
        
        if not order_equal:
            result["match"] = False
            result["failed_at"] = "order_by"
            return result
        
        return result


# ============================================================================
# 对外接口
# ============================================================================

def composite_reward_simple(
    pred_sql: str,
    gold_sql: str,
    db_path: str,
    raise_on_error: bool = False,
) -> float:
    """计算简化分层奖励（简化接口）.
    
    Args:
        pred_sql: 预测SQL
        gold_sql: 标准SQL
        db_path: 数据库路径
        raise_on_error: 是否抛出异常
    
    Returns:
        奖励值
    """
    try:
        calculator = SimpleLayeredRewardCalculator(db_path)
        reward, _ = calculator.calculate_reward(pred_sql, gold_sql)
        return reward
    except Exception as e:
        if raise_on_error:
            raise
        logger.exception(f"Simple composite reward calculation failed: {e}")
        return 0.0


def composite_reward_simple_with_details(
    pred_sql: str,
    gold_sql: str,
    db_path: str,
    raise_on_error: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    """计算简化分层奖励并返回详细信息.
    
    Args:
        pred_sql: 预测SQL
        gold_sql: 标准SQL
        db_path: 数据库路径
        raise_on_error: 是否抛出异常
    
    Returns:
        (reward, details)
    """
    try:
        calculator = SimpleLayeredRewardCalculator(db_path)
        return calculator.calculate_reward(pred_sql, gold_sql)
    except Exception as e:
        if raise_on_error:
            raise
        logger.exception(f"Simple composite reward calculation failed: {e}")
        return 0.0, {"error": str(e)}


# ============================================================================
# 调试和测试
# ============================================================================

def debug_simple_reward_calculation(pred_sql: str, gold_sql: str, db_path: str) -> None:
    """调试：打印简化奖励计算的详细过程."""
    reward, details = composite_reward_simple_with_details(pred_sql, gold_sql, db_path)
    
    print(f"\n{'='*60}")
    print(f"Simple Reward Calculation Debug")
    print(f"{'='*60}")
    print(f"Pred SQL: {pred_sql}")
    print(f"Gold SQL: {gold_sql}")
    print(f"Final Match: {details.get('final_match', False)}")
    print(f"Is Compound: {details.get('is_compound', False)}")
    print(f"Reward Type: {details.get('reward_type', '')}")
    print(f"Last Passed Stage: {details.get('last_passed_stage', 'None')}")
    print(f"{'='*60}")
    print(f"FINAL REWARD: {reward:.4f}")
    print(f"{'='*60}")
    
    # 打印关键阶段结果
    key_stage_results = details.get("key_stage_results", {})
    if key_stage_results:
        print(f"\nKey Stage Results:")
        for stage_name, result in key_stage_results.items():
            if isinstance(result, dict):
                match = result.get("match")
                match_str = "✓" if match else ("✗" if match is False else "?")
                print(f"  [{stage_name}] {match_str}")
                if "failed_at" in result and result["failed_at"]:
                    print(f"    Failed at: {result['failed_at']}")
    
    # 打印性能统计
    perf_stats = details.get("performance_stats", {})
    if perf_stats:
        print(f"\n{'='*60}")
        print(f"Performance Statistics")
        print(f"{'='*60}")
        print(f"binary_reward Calls: {perf_stats.get('binary_reward_call_count', 0)}")
        print(f"Total Time: {perf_stats.get('total_time_sec', 0)*1000:.2f} ms")
        print(f"{'='*60}")


def demo_simple_rewards():
    """演示简化奖励计算."""
    print("=" * 60)
    print("Simple Reward Distribution Demo")
    print("=" * 60)
    print(f"Key Stage Rewards:")
    for stage, reward in KEY_STAGE_REWARDS.items():
        print(f"  {stage.value}: {reward}")
    print()
    print("Stage Mapping:")
    print("  FROM_JOIN -> KeyStage.FROM_JOIN (0.30)")
    print("  WHERE/GROUP_BY/HAVING -> KeyStage.FILTER_GROUP (0.50)")
    print("  SELECT/ORDER_BY -> KeyStage.PROJECTION (0.70)")
    print("  Final Match -> 1.0")
    print()


if __name__ == "__main__":
    demo_simple_rewards()
