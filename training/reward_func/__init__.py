"""Reward functions for NL2SQL evaluation."""

# reward.py 需要 spider_eval，可能会导入失败
try:
    from reward_func.reward import (
        binary_reward,
        multiple_reward,
        continuous_execution_reward,
    )
except ImportError:
    binary_reward = None
    multiple_reward = None
    continuous_execution_reward = None

# composite_reward 的导入
from reward_func.composite_reward import (
    # 主要接口
    composite_reward,
    composite_reward_with_details,
    # 类
    LayeredRewardCalculator,
    IntermediateSQLBuilder,
    SQLParser,
    SQLClauseExtractor,
    SQLClauses,
    IntermediateSQL,
    # 枚举
    SQLStage,
    # 常量
    STAGE_REWARD_MAP,
    COMPARISON_ORDER,
    # SPEED_BONUS,
    CATEGORY_REWARD_MAP,  # 兼容性别名
    BASE_REWARD,
    EXECUTABLE_BONUS,
    # FASTER_BONUS,
    # 工具函数
    get_binary_reward_func,
    measure_execution_time,
    debug_intermediate_sqls,
    debug_reward_calculation,
)

__all__ = [
    # Basic rewards
    "binary_reward",
    "multiple_reward",
    "continuous_execution_reward",
    # Composite reward
    "composite_reward",
    "composite_reward_with_details",
    "LayeredRewardCalculator",
    "IntermediateSQLBuilder",
    "SQLParser",
    "SQLClauseExtractor",
    "SQLClauses",
    "IntermediateSQL",
    "SQLStage",
    # Constants
    "STAGE_REWARD_MAP",
    "COMPARISON_ORDER",
    "SPEED_BONUS",
    "CATEGORY_REWARD_MAP",
    "BASE_REWARD",
    "EXECUTABLE_BONUS",
    "FASTER_BONUS",
    # Utilities
    "get_binary_reward_func",
    "measure_execution_time",
    # Debug utilities
    "debug_intermediate_sqls",
    "debug_reward_calculation",
]
