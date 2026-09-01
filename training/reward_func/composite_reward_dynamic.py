"""动态分层复合奖励系统 - 基于实际阶段数动态分配奖励.

相比 composite_reward_fast.py 的主要改进：
- 奖励值根据gold_sql实际存在的阶段数动态计算
- 最后一个阶段总是获得最高奖励（MAX_STAGE_REWARD）
- 往前每个阶段递减固定步长（REWARD_DECREMENT）
- WHERE/HAVING 阶段使用 FROM 对齐 + simple_binary_reward（不做列排列枚举）

示例 (MAX=0.70, DECREMENT=0.10, MIN_FLOOR=0.10):
- gold_sql有4个阶段：奖励分布 0.40 → 0.50 → 0.60 → 0.70
- gold_sql有3个阶段：奖励分布 0.50 → 0.60 → 0.70
- gold_sql有2个阶段：奖励分布 0.60 → 0.70
- gold_sql有7个阶段：奖励分布 0.10 → 0.20 → 0.30 → 0.40 → 0.50 → 0.60 → 0.70
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Tuple

from reward_func.composite_reward_fast import (
    COMPARISON_ORDER,
    EXECUTE_SQL_STAGES,
    STRUCTURAL_COMPARE_STAGES,
    FastLayeredRewardCalculator,
    IntermediateSQLBuilder,
    PerformanceStats,
    SQLClauses,
    SQLParser,
    SQLStage,
    get_binary_reward_func,
)

logger = logging.getLogger(__name__)

# ============================================================================
# 动态奖励配置
# ============================================================================

# 最高阶段奖励（最后通过的阶段）
# 设置为0.7，与最终匹配(1.0)保持0.3差距，防止模型刷分
MAX_STAGE_REWARD = 0.70

# 每个阶段的递减步长
# 增大到0.1，使相邻阶段差距更明显，梯度信号更强
REWARD_DECREMENT = 0.05

# 最低奖励下限（防止负数或过低）
MIN_REWARD_FLOOR = 0.10


def compute_dynamic_rewards(active_stages: List[SQLStage]) -> Dict[SQLStage, float]:
    """根据实际存在的阶段数动态计算奖励.
    
    策略：最后一个阶段获得最高奖励，往前每个阶段递减 REWARD_DECREMENT。
    
    Args:
        active_stages: 按顺序排列的活跃阶段列表
    
    Returns:
        Dict[SQLStage, float]: 每个阶段的奖励映射
    
    示例 (MAX=0.70, DECREMENT=0.10):
        - 4个阶段: [0.40, 0.50, 0.60, 0.70]
        - 3个阶段: [0.50, 0.60, 0.70]
        - 2个阶段: [0.60, 0.70]
        - 1个阶段: [0.70]
        - 7个阶段: [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    """
    n = len(active_stages)
    
    if n == 0:
        return {}
    
    # 从最后一个阶段开始，往前递减
    rewards = {}
    for i, stage in enumerate(active_stages):
        # 距离最后一个阶段的步数
        steps_from_last = n - 1 - i
        reward = MAX_STAGE_REWARD - steps_from_last * REWARD_DECREMENT
        # 确保不低于最低下限
        reward = max(reward, MIN_REWARD_FLOOR)
        rewards[stage] = round(reward, 4)
    
    return rewards


def get_active_stages(clauses: SQLClauses) -> List[SQLStage]:
    """获取SQL中实际存在的阶段列表（按顺序）.
    
    Args:
        clauses: SQL子句结构
    
    Returns:
        按 COMPARISON_ORDER 顺序排列的活跃阶段列表
    """
    active = []
    
    for stage in COMPARISON_ORDER:
        if stage == SQLStage.FROM_JOIN:
            if clauses.from_clause:
                active.append(stage)
        elif stage == SQLStage.WHERE:
            if clauses.where:
                active.append(stage)
        elif stage == SQLStage.GROUP_BY:
            if clauses.group_by:
                active.append(stage)
        elif stage == SQLStage.HAVING:
            if clauses.having:
                active.append(stage)
        elif stage == SQLStage.SELECT:
            # SELECT 总是存在
            active.append(stage)
        elif stage == SQLStage.ORDER_BY:
            if clauses.order_by:
                active.append(stage)
    
    return active


# ============================================================================
# 动态分层奖励计算器
# ============================================================================

class DynamicLayeredRewardCalculator(FastLayeredRewardCalculator):
    """动态分层奖励计算器.
    
    继承自 FastLayeredRewardCalculator，重写奖励计算逻辑：
    - 根据gold_sql实际存在的阶段数动态分配奖励
    - 最后一个阶段总是获得最高奖励
    """
    
    @staticmethod
    def _build_from_base(clauses) -> str:
        """构建FROM基础部分."""
        if not clauses.from_clause:
            return ""
        base = f"FROM {clauses.from_clause}"
        if clauses.joins:
            base += " " + " ".join(clauses.joins)
        return base
    
    @staticmethod
    def _remap_clause_aliases(clause, pred_clauses, gold_clauses):
        """将子句中 pred 的别名替换为 gold 的别名体系."""
        from reward_func.composite_reward_fast import _extract_alias_mapping
        
        pred_alias_map = _extract_alias_mapping(pred_clauses)
        gold_alias_map = _extract_alias_mapping(gold_clauses)
        
        gold_table_to_alias = {}
        for alias, table in gold_alias_map.items():
            if table not in gold_table_to_alias:
                gold_table_to_alias[table] = alias
            else:
                return clause  # 自连接，无法自动映射
        
        alias_remap = {}
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
                flags=re.I
            )
        return result
    
    def _build_aligned_intermediate_sql(
        self,
        stage,
        pred_clauses,
        gold_clauses,
    ):
        """构建FROM对齐的中间SQL.
        
        1. pred的FROM部分替换为gold的FROM
        2. pred子句中的别名替换为gold的别名体系
        
        Returns:
            (pred_sql, gold_sql) 元组，或 None 如果构建失败
        """
        gold_from_base = self._build_from_base(gold_clauses)
        if not gold_from_base:
            return None
        
        if stage == SQLStage.WHERE:
            if not pred_clauses.where or not gold_clauses.where:
                return None
            remapped_where = self._remap_clause_aliases(pred_clauses.where, pred_clauses, gold_clauses)
            pred_sql = f"SELECT * {gold_from_base} WHERE {remapped_where}"
            gold_sql = f"SELECT * {gold_from_base} WHERE {gold_clauses.where}"
            return pred_sql, gold_sql
        
        elif stage == SQLStage.HAVING:
            if not pred_clauses.having or not gold_clauses.having:
                return None
            
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
        
        return None
    
    def calculate_reward(
        self,
        pred_sql: str,
        gold_sql: str,
    ) -> Tuple[float, Dict[str, Any]]:
        """计算动态分层复合奖励.
        
        Args:
            pred_sql: 预测SQL
            gold_sql: 标准SQL
        
        Returns:
            (reward, details)
        """
        import time
        
        total_start = time.perf_counter()
        stats = PerformanceStats()
        
        details: Dict[str, Any] = {
            "pred_sql": pred_sql,
            "gold_sql": gold_sql,
            "final_match": False,
            "is_compound": False,
            "stage_results": {},
            "last_passed_stage": None,
            "reward_type": "",
            "reward": 0.0,
            "dynamic_rewards": {},  # 新增：动态奖励映射
            "active_stages": [],    # 新增：活跃阶段列表
        }
        
        def finalize_and_return(reward: float) -> Tuple[float, Dict[str, Any]]:
            """统一的返回处理."""
            stats.total_time = time.perf_counter() - total_start
            details["performance_stats"] = stats.to_dict()
            details["reward"] = reward
            logger.debug(stats.summary())
            return reward, details
        
        # Step 0: 检查最终结果是否匹配
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
        
        # Step 1: 检查是否是复合查询
        pred_parser = SQLParser(pred_sql)
        gold_parser = SQLParser(gold_sql)
        
        if pred_parser.is_compound() or gold_parser.is_compound():
            details["is_compound"] = True
            details["reward_type"] = "compound_no_match"
            return finalize_and_return(0.0)
        
        # Step 2: 提取子句
        start = time.perf_counter()
        pred_clauses, pred_intermediates = self.builder.build_all_intermediate_sqls(pred_sql)
        gold_clauses, gold_intermediates = self.builder.build_all_intermediate_sqls(gold_sql)
        stats.clause_extraction_count += 2
        stats.clause_extraction_time += time.perf_counter() - start
        
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
        
        # Step 3: 计算动态奖励映射（基于gold_sql的阶段）
        active_stages = get_active_stages(gold_clauses)
        dynamic_rewards = compute_dynamic_rewards(active_stages)
        
        details["active_stages"] = [s.value for s in active_stages]
        details["dynamic_rewards"] = {s.value: r for s, r in dynamic_rewards.items()}
        
        logger.debug(f"Active stages: {[s.value for s in active_stages]}")
        logger.debug(f"Dynamic rewards: {details['dynamic_rewards']}")
        
        # Step 4: 从前往后比较中间状态 (a → g)
        last_passed_reward = 0.0
        
        for stage in COMPARISON_ORDER:
            pred_exists = self._stage_exists(stage, pred_clauses)
            gold_exists = self._stage_exists(stage, gold_clauses)
            
            stage_result = {
                "pred_exists": pred_exists,
                "gold_exists": gold_exists,
                "match": None,
                "reason": "",
                "details": {},
                "stage_reward": dynamic_rewards.get(stage, 0.0),  # 记录该阶段的动态奖励
            }
            
            # 两边都没有这个阶段，跳过
            if not pred_exists and not gold_exists:
                stage_result["reason"] = "两边都没有此阶段，跳过"
                details["stage_results"][stage.value] = stage_result
                continue
            
            # 只有一边有，不通过，返回当前奖励
            if pred_exists != gold_exists:
                stage_result["reason"] = "只有一边有此阶段，不通过"
                stage_result["match"] = False
                details["stage_results"][stage.value] = stage_result
                details["reward_type"] = f"mismatch_at_{stage.value}"
                return finalize_and_return(last_passed_reward)
            
            # 两边都有，进行比较
            if stage in STRUCTURAL_COMPARE_STAGES:
                # 结构化比较
                is_equal, compare_details = self._compare_stage_structural(
                    stage, pred_clauses, gold_clauses, stats
                )
                stage_result["match"] = is_equal
                stage_result["details"] = compare_details
                
                if is_equal:
                    stage_result["reason"] = "结构比较通过"
                    # 使用动态奖励
                    last_passed_reward = dynamic_rewards.get(stage, 0.0)
                    details["last_passed_stage"] = stage.value
                else:
                    stage_result["reason"] = "结构比较不通过"
                    details["stage_results"][stage.value] = stage_result
                    details["reward_type"] = f"structural_mismatch_at_{stage.value}"
                    return finalize_and_return(last_passed_reward)
            
            elif stage in EXECUTE_SQL_STAGES:
                # 执行SQL比较（FROM对齐 + simple_binary_reward，避免列排列枚举）
                aligned_sql = self._build_aligned_intermediate_sql(
                    stage, pred_clauses, gold_clauses
                )
                
                if aligned_sql:
                    pred_aligned, gold_aligned = aligned_sql
                    match = self._timed_simple_binary_reward(
                        pred_aligned,
                        gold_aligned,
                        stats,
                        context=f"stage_{stage.value}"
                    )
                    stage_result["match"] = (match == 1.0)
                    
                    if match == 1.0:
                        stage_result["reason"] = "SQL执行比较通过（FROM对齐）"
                        # 使用动态奖励
                        last_passed_reward = dynamic_rewards.get(stage, 0.0)
                        details["last_passed_stage"] = stage.value
                    else:
                        stage_result["reason"] = "SQL执行比较不通过"
                        details["stage_results"][stage.value] = stage_result
                        details["reward_type"] = f"sql_mismatch_at_{stage.value}"
                        return finalize_and_return(last_passed_reward)
                else:
                    # 中间SQL构建失败
                    stage_result["reason"] = "中间SQL构建失败"
                    stage_result["match"] = False
                    details["stage_results"][stage.value] = stage_result
                    details["reward_type"] = f"build_failed_at_{stage.value}"
                    return finalize_and_return(last_passed_reward)
            
            details["stage_results"][stage.value] = stage_result
        
        # 所有阶段都通过
        details["reward_type"] = "all_stages_passed"
        return finalize_and_return(last_passed_reward)


# ============================================================================
# 对外接口
# ============================================================================

def composite_reward_dynamic(
    pred_sql: str,
    gold_sql: str,
    db_path: str,
    raise_on_error: bool = False,
) -> float:
    """计算动态分层复合奖励（简化接口）.
    
    Args:
        pred_sql: 预测SQL
        gold_sql: 标准SQL
        db_path: 数据库路径
        raise_on_error: 是否抛出异常
    
    Returns:
        奖励值
    """
    try:
        calculator = DynamicLayeredRewardCalculator(db_path)
        reward, _ = calculator.calculate_reward(pred_sql, gold_sql)
        return reward
    except Exception as e:
        if raise_on_error:
            raise
        logger.exception(f"Dynamic composite reward calculation failed: {e}")
        return 0.0


def composite_reward_dynamic_with_details(
    pred_sql: str,
    gold_sql: str,
    db_path: str,
    raise_on_error: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    """计算动态分层复合奖励并返回详细信息.
    
    Args:
        pred_sql: 预测SQL
        gold_sql: 标准SQL  
        db_path: 数据库路径
        raise_on_error: 是否抛出异常
    
    Returns:
        (reward, details)
    """
    try:
        calculator = DynamicLayeredRewardCalculator(db_path)
        return calculator.calculate_reward(pred_sql, gold_sql)
    except Exception as e:
        if raise_on_error:
            raise
        logger.exception(f"Dynamic composite reward calculation failed: {e}")
        return 0.0, {"error": str(e)}


# ============================================================================
# 调试和测试
# ============================================================================

def debug_dynamic_reward_calculation(pred_sql: str, gold_sql: str, db_path: str) -> None:
    """调试：打印动态奖励计算的详细过程."""
    reward, details = composite_reward_dynamic_with_details(pred_sql, gold_sql, db_path)
    
    print(f"\n{'='*60}")
    print(f"Dynamic Reward Calculation Debug")
    print(f"{'='*60}")
    print(f"Pred SQL: {pred_sql}")
    print(f"Gold SQL: {gold_sql}")
    print(f"Final Match: {details.get('final_match', False)}")
    print(f"Is Compound: {details.get('is_compound', False)}")
    
    # 打印动态奖励信息
    active_stages = details.get("active_stages", [])
    dynamic_rewards = details.get("dynamic_rewards", {})
    print(f"\n{'='*60}")
    print(f"Dynamic Reward Configuration")
    print(f"{'='*60}")
    print(f"Active Stages ({len(active_stages)}): {active_stages}")
    print(f"Dynamic Rewards:")
    for stage, reward_val in dynamic_rewards.items():
        print(f"  {stage}: {reward_val:.4f}")
    
    print(f"\n{'='*60}")
    print(f"Reward Type: {details.get('reward_type', '')}")
    print(f"Last Passed Stage: {details.get('last_passed_stage', 'None')}")
    print(f"{'='*60}")
    print(f"FINAL REWARD: {reward:.4f}")
    print(f"{'='*60}")
    
    # 打印阶段结果
    stage_results = details.get("stage_results", {})
    if stage_results:
        print(f"\nStage Results:")
        for stage_name, result in stage_results.items():
            match = result.get("match")
            reason = result.get("reason", "")
            stage_reward = result.get("stage_reward", 0.0)
            match_str = "✓" if match else ("✗" if match is False else "?")
            print(f"  [{stage_name}] {match_str} (reward={stage_reward:.4f}) - {reason}")
    
    # 打印性能统计
    perf_stats = details.get("performance_stats", {})
    if perf_stats:
        print(f"\n{'='*60}")
        print(f"Performance Statistics")
        print(f"{'='*60}")
        print(f"binary_reward Calls: {perf_stats.get('binary_reward_call_count', 0)}")
        print(f"Structural Compares: {perf_stats.get('structural_compare_count', 0)}")
        print(f"Total Time: {perf_stats.get('total_time_sec', 0)*1000:.2f} ms")
        print(f"{'='*60}")


def demo_dynamic_rewards():
    """演示动态奖励计算."""
    print("=" * 60)
    print("Dynamic Reward Distribution Demo")
    print("=" * 60)
    print(f"MAX_STAGE_REWARD: {MAX_STAGE_REWARD}")
    print(f"REWARD_DECREMENT: {REWARD_DECREMENT}")
    print(f"MIN_REWARD_FLOOR: {MIN_REWARD_FLOOR}")
    print()
    
    # 模拟不同阶段数的情况
    test_cases = [
        ("Full SQL (7 stages)", COMPARISON_ORDER),
        ("Simple SELECT (2 stages)", [SQLStage.FROM_JOIN, SQLStage.SELECT]),
        ("With WHERE (3 stages)", [SQLStage.FROM_JOIN, SQLStage.WHERE, SQLStage.SELECT]),
        ("With GROUP BY (4 stages)", [SQLStage.FROM_JOIN, SQLStage.WHERE, SQLStage.GROUP_BY, SQLStage.SELECT]),
        ("With ORDER BY (3 stages)", [SQLStage.FROM_JOIN, SQLStage.SELECT, SQLStage.ORDER_BY]),
        ("Only 1 stage", [SQLStage.SELECT]),
    ]
    
    for name, stages in test_cases:
        rewards = compute_dynamic_rewards(stages)
        print(f"\n{name}:")
        print(f"  Stages: {[s.value for s in stages]}")
        print(f"  Rewards: {[f'{rewards[s]:.2f}' for s in stages]}")


if __name__ == "__main__":
    demo_dynamic_rewards()
