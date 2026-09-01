"""分层复合奖励系统 - 基于SQL执行顺序的中间状态对比.

使用 sqlparse 进行专业的 SQL 解析。

阶段划分（按执行顺序）：
a. 选表阶段 (FROM/JOIN): SELECT * FROM ... JOIN ...
b. 行过滤阶段 (WHERE): + WHERE ...
c. 分组阶段 (GROUP BY): + GROUP BY ...
d. 组过滤阶段 (HAVING): + HAVING ...
e. 列投影阶段 (SELECT): 替换 SELECT 列
f. 去重阶段 (DISTINCT): + DISTINCT
g. 排序阶段 (ORDER BY): + ORDER BY ...
h. 限制行数阶段 (LIMIT): + LIMIT ... (等同于最终SQL，不单独判断)

奖励计算策略：
1. 最终结果匹配 → 1.0
2. UNION/INTERSECT/EXCEPT 复合查询不匹配 → 0
3. 中间状态从后往前对比(g→a)，使用AST相似度前置过滤，奖励递减（0.70→0.40）
4. 全部不匹配 → 0

优化策略：
- 去掉速度比较（避免额外SQL执行）
- AST相似度前置过滤（相似度低于阈值时跳过耗时的结果比较）
"""

from __future__ import annotations

import logging
import re
import time
import sqlite3
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Where, Parenthesis, Token, Statement
from sqlparse.tokens import Keyword, DML, Punctuation

# 导入 binary_reward 和 simple_binary_reward
try:
    from reward_func.reward import binary_reward, simple_binary_reward
except ImportError:
    binary_reward = None
    simple_binary_reward = None

logger = logging.getLogger(__name__)


def get_binary_reward_func():
    """获取 binary_reward 函数.
    
    Raises:
        RuntimeError: 如果 binary_reward 不可用
    """
    if binary_reward is None:
        raise RuntimeError(
            "binary_reward function not available. "
            "Please ensure spider_eval is installed and reward_func.reward can be imported."
        )
    return binary_reward


def get_simple_binary_reward_func():
    """获取 simple_binary_reward 函数.
    
    简化版比较不做列排列枚举，速度更快。
    专用于中间阶段比较。
    
    Raises:
        RuntimeError: 如果 simple_binary_reward 不可用
    """
    if simple_binary_reward is None:
        raise RuntimeError(
            "simple_binary_reward function not available. "
            "Please ensure reward_func.reward can be imported."
        )
    return simple_binary_reward


# ============================================================================
# 第一部分：常量和枚举定义
# ============================================================================

class SQLStage(Enum):
    """SQL执行阶段枚举."""
    FROM_JOIN = "from_join"      # a. 选表阶段
    WHERE = "where"              # b. 行过滤阶段
    GROUP_BY = "group_by"        # c. 分组阶段
    HAVING = "having"            # d. 组过滤阶段
    SELECT = "select"            # e. 列投影阶段
    DISTINCT = "distinct"        # f. 去重阶段
    ORDER_BY = "order_by"        # g. 排序阶段
    LIMIT = "limit"              # h. 限制行数阶段（等同于最终SQL）


# 从后往前的比较顺序（不包括 LIMIT，因为它等同于最终SQL）
COMPARISON_ORDER = [
    SQLStage.ORDER_BY,    # g
    SQLStage.DISTINCT,    # f
    SQLStage.SELECT,      # e
    SQLStage.HAVING,      # d
    SQLStage.GROUP_BY,    # c
    SQLStage.WHERE,       # b
    SQLStage.FROM_JOIN,   # a
]

# 阶段奖励映射（从后往前递减，每级降0.05，最高0.7，最低0.4）
STAGE_REWARD_MAP = {
    SQLStage.ORDER_BY: 0.70,    # g 排序正确
    SQLStage.DISTINCT: 0.65,    # f 去重正确
    SQLStage.SELECT: 0.60,      # e 列投影正确
    SQLStage.HAVING: 0.55,      # d 组过滤正确
    SQLStage.GROUP_BY: 0.50,    # c 分组正确
    SQLStage.WHERE: 0.45,       # b 行过滤正确
    SQLStage.FROM_JOIN: 0.40,   # a 选表正确
}

# AST 相似度阈值（低于此阈值时跳过结果比较）
# 设置为 0.7，更激进的过滤，可以跳过更多不太可能匹配的阶段
AST_SIMILARITY_THRESHOLD = 0.7


@dataclass
class PerformanceStats:
    """性能统计信息.
    
    记录奖励计算过程中的各项性能指标，包括：
    - binary_reward 调用次数和时间（内部会执行SQL并比较结果）
    - AST 相似度计算次数和时间
    - 被 AST 过滤跳过的次数
    - 子句提取时间
    - 总耗时
    """
    # 计数器
    binary_reward_call_count: int = 0      # binary_reward 调用次数（每次内部执行2条SQL）
    ast_similarity_count: int = 0          # AST 相似度计算次数
    ast_filtered_count: int = 0            # 被 AST 过滤跳过的次数
    clause_extraction_count: int = 0       # 子句提取次数
    
    # 时间记录（秒）
    clause_extraction_time: float = 0.0    # 子句提取总时间
    ast_similarity_time: float = 0.0       # AST 相似度计算总时间
    binary_reward_time: float = 0.0        # binary_reward 总时间（包含内部SQL执行+结果比较）
    total_time: float = 0.0                # 总时间
    
    # 详细时间列表（用于分析）
    clause_extraction_details: List[Dict[str, Any]] = field(default_factory=list)
    ast_similarity_details: List[Dict[str, Any]] = field(default_factory=list)
    binary_reward_details: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式."""
        # 估算实际 SQL 执行次数：binary_reward 每次调用内部执行 2 条 SQL
        estimated_sql_exec_count = self.binary_reward_call_count * 2
        return {
            "binary_reward_call_count": self.binary_reward_call_count,
            "estimated_sql_exec_count": estimated_sql_exec_count,
            "ast_similarity_count": self.ast_similarity_count,
            "ast_filtered_count": self.ast_filtered_count,
            "clause_extraction_count": self.clause_extraction_count,
            "clause_extraction_time_sec": round(self.clause_extraction_time, 6),
            "ast_similarity_time_sec": round(self.ast_similarity_time, 6),
            "binary_reward_time_sec": round(self.binary_reward_time, 6),
            "total_time_sec": round(self.total_time, 6),
            "clause_extraction_details": self.clause_extraction_details,
            "ast_similarity_details": self.ast_similarity_details,
            "binary_reward_details": self.binary_reward_details,
        }
    
    def summary(self) -> str:
        """生成简洁的摘要字符串."""
        estimated_sql_exec_count = self.binary_reward_call_count * 2
        return (
            f"PerformanceStats: "
            f"binary_reward_calls={self.binary_reward_call_count}, "
            f"estimated_sql_execs={estimated_sql_exec_count}, "
            f"ast_filtered={self.ast_filtered_count}, "
            f"extraction_time={self.clause_extraction_time*1000:.2f}ms, "
            f"ast_sim_time={self.ast_similarity_time*1000:.2f}ms, "
            f"binary_reward_time={self.binary_reward_time*1000:.2f}ms, "
            f"total={self.total_time*1000:.2f}ms"
        )


@dataclass
class SQLClauses:
    """SQL各子句的结构化表示."""
    select: str = "*"
    distinct: bool = False
    from_clause: str = ""
    joins: List[str] = field(default_factory=list)
    where: str = ""
    group_by: str = ""
    having: str = ""
    order_by: str = ""
    limit: str = ""
    
    # 原始SQL
    original_sql: str = ""
    # 是否是复合查询
    is_compound: bool = False


@dataclass
class IntermediateSQL:
    """中间SQL状态."""
    stage: SQLStage
    sql: str
    description: str = ""


# ============================================================================
# 第二部分：SQL解析器（使用sqlparse）
# ============================================================================

class SQLParser:
    """使用 sqlparse 的 SQL 解析器."""
    
    def __init__(self, sql: str):
        self.original_sql = self._normalize(sql)
        self.parsed = None
        if self.original_sql:
            statements = sqlparse.parse(self.original_sql)
            if statements:
                self.parsed = statements[0]
    
    def _normalize(self, sql: str) -> str:
        """标准化SQL（压缩多余空白，去除末尾分号）."""
        if not sql:
            return ""
        # 去除末尾分号
        sql = sql.strip().rstrip(';').strip()
        # 压缩空白但保留单个空格
        return re.sub(r'\s+', ' ', sql)
    
    def is_compound(self) -> bool:
        """检查是否是复合查询（UNION/INTERSECT/EXCEPT）.
        
        注意：需要排除子查询中的这些关键字，只检查顶层。
        使用字符串遍历而不是依赖 sqlparse 的 token 类型（因为 sqlparse 有时会合并 token）。
        """
        if not self.original_sql:
            return False
        
        sql_upper = self.original_sql.upper()
        keywords = ['UNION', 'INTERSECT', 'EXCEPT']
        
        # 遍历 SQL，跟踪括号深度，在顶层查找关键字
        depth = 0
        i = 0
        while i < len(sql_upper):
            char = sql_upper[i]
            
            if char == '(':
                depth += 1
                i += 1
            elif char == ')':
                depth -= 1
                i += 1
            elif depth == 0:
                # 在顶层，检查是否是关键字
                for kw in keywords:
                    if sql_upper[i:].startswith(kw):
                        # 确保是完整的单词（前后不是字母数字）
                        before_ok = (i == 0 or not sql_upper[i-1].isalnum())
                        after_pos = i + len(kw)
                        after_ok = (after_pos >= len(sql_upper) or not sql_upper[after_pos].isalnum())
                        if before_ok and after_ok:
                            return True
                i += 1
            else:
                i += 1
        
        return False
    
    def extract_clauses(self) -> SQLClauses:
        """提取SQL的各个子句."""
        clauses = SQLClauses(original_sql=self.original_sql)
        
        if not self.parsed:
            return clauses
        
        clauses.is_compound = self.is_compound()
        
        # 对于复合查询，不进行详细解析
        if clauses.is_compound:
            return clauses
        
        # 使用正则+sqlparse混合提取（sqlparse对子句边界处理更好）
        sql = self.original_sql
        sql_upper = sql.upper()
        
        # 提取 DISTINCT
        clauses.distinct = bool(re.search(r'\bSELECT\s+DISTINCT\b', sql, re.I))
        
        # 提取 SELECT 列
        clauses.select = self._extract_select(sql)
        
        # 提取 FROM 子句（包含子查询的情况）
        clauses.from_clause = self._extract_from(sql)
        
        # 提取 JOINs
        clauses.joins = self._extract_joins(sql)
        
        # 提取 WHERE（保留子查询）
        clauses.where = self._extract_where(sql)
        
        # 提取 GROUP BY
        clauses.group_by = self._extract_group_by(sql)
        
        # 提取 HAVING
        clauses.having = self._extract_having(sql)
        
        # 提取 ORDER BY
        clauses.order_by = self._extract_order_by(sql)
        
        # 提取 LIMIT
        clauses.limit = self._extract_limit(sql)
        
        return clauses
    
    def _extract_select(self, sql: str) -> str:
        """提取SELECT列（不含DISTINCT关键字）."""
        # 匹配 SELECT [DISTINCT] ... FROM
        m = re.search(r'\bSELECT\s+(?:DISTINCT\s+)?(.*?)\s+FROM\b', sql, re.I | re.S)
        if m:
            return m.group(1).strip()
        return "*"
    
    def _extract_from(self, sql: str) -> str:
        """提取FROM子句（主表，不含JOIN）."""
        # 需要处理子查询的情况
        # 找到 FROM 后面到第一个顶层 JOIN/WHERE/GROUP/ORDER/HAVING/LIMIT 之间的内容
        
        m = re.search(r'\bFROM\s+', sql, re.I)
        if not m:
            return ""
        
        start = m.end()
        # 从start开始，找到顶层的终止关键字
        end = self._find_clause_end(sql, start, 
            ['JOIN', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'FULL JOIN', 'CROSS JOIN',
             'WHERE', 'GROUP BY', 'HAVING', 'ORDER BY', 'LIMIT', 'UNION', 'INTERSECT', 'EXCEPT'])
        
        return sql[start:end].strip()
    
    def _extract_joins(self, sql: str) -> List[str]:
        """提取所有JOIN子句."""
        joins = []
        
        # 匹配各种JOIN
        join_pattern = re.compile(
            r'\b((?:INNER\s+|LEFT\s+(?:OUTER\s+)?|RIGHT\s+(?:OUTER\s+)?|FULL\s+(?:OUTER\s+)?|CROSS\s+)?JOIN)\s+',
            re.I
        )
        
        for m in join_pattern.finditer(sql):
            join_start = m.start()
            join_keyword = m.group(1)
            content_start = m.end()
            
            # 检查这个JOIN是否在顶层（不在子查询内）
            if self._is_inside_subquery(sql, join_start):
                continue
            
            # 找到这个JOIN的结束位置
            end = self._find_clause_end(sql, content_start,
                ['JOIN', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'FULL JOIN', 'CROSS JOIN',
                 'WHERE', 'GROUP BY', 'HAVING', 'ORDER BY', 'LIMIT', 'UNION', 'INTERSECT', 'EXCEPT'])
            
            join_content = sql[content_start:end].strip()
            joins.append(f"{join_keyword.upper()} {join_content}")
        
        return joins
    
    def _extract_where(self, sql: str) -> str:
        """提取WHERE子句（保留子查询）."""
        m = re.search(r'\bWHERE\s+', sql, re.I)
        if not m:
            return ""
        
        start = m.end()
        # 找到顶层的终止关键字（子查询内的不算）
        end = self._find_clause_end(sql, start,
            ['GROUP BY', 'HAVING', 'ORDER BY', 'LIMIT', 'UNION', 'INTERSECT', 'EXCEPT'])
        
        return sql[start:end].strip()
    
    def _extract_group_by(self, sql: str) -> str:
        """提取GROUP BY子句."""
        m = re.search(r'\bGROUP\s+BY\s+', sql, re.I)
        if not m:
            return ""
        
        # 检查是否在子查询内
        if self._is_inside_subquery(sql, m.start()):
            return ""
        
        start = m.end()
        end = self._find_clause_end(sql, start,
            ['HAVING', 'ORDER BY', 'LIMIT', 'UNION', 'INTERSECT', 'EXCEPT'])
        
        return sql[start:end].strip()
    
    def _extract_having(self, sql: str) -> str:
        """提取HAVING子句."""
        m = re.search(r'\bHAVING\s+', sql, re.I)
        if not m:
            return ""
        
        # 检查是否在子查询内
        if self._is_inside_subquery(sql, m.start()):
            return ""
        
        start = m.end()
        end = self._find_clause_end(sql, start,
            ['ORDER BY', 'LIMIT', 'UNION', 'INTERSECT', 'EXCEPT'])
        
        return sql[start:end].strip()
    
    def _extract_order_by(self, sql: str) -> str:
        """提取ORDER BY子句."""
        m = re.search(r'\bORDER\s+BY\s+', sql, re.I)
        if not m:
            return ""
        
        # 检查是否在子查询内
        if self._is_inside_subquery(sql, m.start()):
            return ""
        
        start = m.end()
        end = self._find_clause_end(sql, start,
            ['LIMIT', 'UNION', 'INTERSECT', 'EXCEPT'])
        
        return sql[start:end].strip()
    
    def _extract_limit(self, sql: str) -> str:
        """提取LIMIT子句."""
        m = re.search(r'\bLIMIT\s+(\d+(?:\s*,\s*\d+)?)', sql, re.I)
        if not m:
            return ""
        
        # 检查是否在子查询内
        if self._is_inside_subquery(sql, m.start()):
            return ""
        
        return m.group(1).strip()
    
    def _find_clause_end(self, sql: str, start: int, terminators: List[str]) -> int:
        """找到子句的结束位置（考虑括号嵌套）."""
        depth = 0
        i = start
        sql_upper = sql.upper()
        
        while i < len(sql):
            char = sql[i]
            
            if char == '(':
                depth += 1
                i += 1
            elif char == ')':
                depth -= 1
                i += 1
            elif depth == 0:
                # 只在顶层检查终止符
                for term in terminators:
                    term_upper = term.upper()
                    if sql_upper[i:].startswith(term_upper):
                        # 确保是完整的关键字（前面是空白或开头）
                        if i == 0 or not sql[i-1].isalnum():
                            # 确保后面是空白或结尾
                            end_pos = i + len(term)
                            if end_pos >= len(sql) or not sql[end_pos].isalnum():
                                return i
                i += 1
            else:
                i += 1
        
        return len(sql)
    
    def _is_inside_subquery(self, sql: str, pos: int) -> bool:
        """检查位置是否在子查询内（通过计算括号深度）."""
        depth = 0
        for i in range(pos):
            if sql[i] == '(':
                depth += 1
            elif sql[i] == ')':
                depth -= 1
        return depth > 0


# ============================================================================
# 第三部分：中间SQL构建器
# ============================================================================

class IntermediateSQLBuilder:
    """中间SQL构建器 - 基于解析结果构建各阶段的中间SQL."""
    
    def __init__(self):
        self.parser = None
    
    def build_all_intermediate_sqls(self, sql: str) -> Dict[SQLStage, IntermediateSQL]:
        """构建SQL的所有中间状态.
        
        Returns:
            Dict[SQLStage, IntermediateSQL] - 各阶段的中间SQL
        """
        self.parser = SQLParser(sql)
        clauses = self.parser.extract_clauses()
        results: Dict[SQLStage, IntermediateSQL] = {}
        
        # 复合查询不进行中间状态提取
        if clauses.is_compound:
            logger.debug(f"Compound SQL detected, skipping intermediate extraction")
            return results
        
        # 构建FROM基础部分
        from_base = self._build_from_base(clauses)
        if not from_base:
            return results
        
        # a. 选表阶段 (FROM_JOIN): SELECT * FROM ... JOIN ...
        from_join_sql = f"SELECT * {from_base}"
        results[SQLStage.FROM_JOIN] = IntermediateSQL(
            stage=SQLStage.FROM_JOIN,
            sql=from_join_sql,
            description="a. 选表阶段"
        )
        
        # b. 行过滤阶段 (WHERE)
        if clauses.where:
            where_sql = f"SELECT * {from_base} WHERE {clauses.where}"
            results[SQLStage.WHERE] = IntermediateSQL(
                stage=SQLStage.WHERE,
                sql=where_sql,
                description="b. 行过滤阶段"
            )
        
        # c. 分组阶段 (GROUP_BY) - SELECT 使用 GROUP BY 列
        if clauses.group_by:
            group_sql = f"SELECT {clauses.group_by} {from_base}"
            if clauses.where:
                group_sql += f" WHERE {clauses.where}"
            group_sql += f" GROUP BY {clauses.group_by}"
            results[SQLStage.GROUP_BY] = IntermediateSQL(
                stage=SQLStage.GROUP_BY,
                sql=group_sql,
                description="c. 分组阶段"
            )
        
        # d. 组过滤阶段 (HAVING) - SELECT 使用 GROUP BY 列
        if clauses.having and clauses.group_by:
            having_sql = f"SELECT {clauses.group_by} {from_base}"
            if clauses.where:
                having_sql += f" WHERE {clauses.where}"
            having_sql += f" GROUP BY {clauses.group_by} HAVING {clauses.having}"
            results[SQLStage.HAVING] = IntermediateSQL(
                stage=SQLStage.HAVING,
                sql=having_sql,
                description="d. 组过滤阶段"
            )
        
        # e. 列投影阶段 (SELECT)
        select_sql = f"SELECT {clauses.select} {from_base}"
        if clauses.where:
            select_sql += f" WHERE {clauses.where}"
        if clauses.group_by:
            select_sql += f" GROUP BY {clauses.group_by}"
            if clauses.having:
                select_sql += f" HAVING {clauses.having}"
        results[SQLStage.SELECT] = IntermediateSQL(
            stage=SQLStage.SELECT,
            sql=select_sql,
            description="e. 列投影阶段"
        )
        
        # f. 去重阶段 (DISTINCT)
        if clauses.distinct:
            distinct_sql = f"SELECT DISTINCT {clauses.select} {from_base}"
            if clauses.where:
                distinct_sql += f" WHERE {clauses.where}"
            if clauses.group_by:
                distinct_sql += f" GROUP BY {clauses.group_by}"
                if clauses.having:
                    distinct_sql += f" HAVING {clauses.having}"
            results[SQLStage.DISTINCT] = IntermediateSQL(
                stage=SQLStage.DISTINCT,
                sql=distinct_sql,
                description="f. 去重阶段"
            )
        
        # g. 排序阶段 (ORDER_BY)
        if clauses.order_by:
            order_sql = f"SELECT {'DISTINCT ' if clauses.distinct else ''}{clauses.select} {from_base}"
            if clauses.where:
                order_sql += f" WHERE {clauses.where}"
            if clauses.group_by:
                order_sql += f" GROUP BY {clauses.group_by}"
                if clauses.having:
                    order_sql += f" HAVING {clauses.having}"
            order_sql += f" ORDER BY {clauses.order_by}"
            results[SQLStage.ORDER_BY] = IntermediateSQL(
                stage=SQLStage.ORDER_BY,
                sql=order_sql,
                description="g. 排序阶段"
            )
        
        # h. LIMIT阶段不单独判断（等同于最终SQL）
        
        return results
    
    def _build_from_base(self, clauses: SQLClauses) -> str:
        """构建FROM基础部分（FROM + JOINs）."""
        if not clauses.from_clause:
            return ""
        
        base = f"FROM {clauses.from_clause}"
        
        if clauses.joins:
            base += " " + " ".join(clauses.joins)
        
        return base


# ============================================================================
# 第四部分：执行时间测量
# ============================================================================

def measure_execution_time(sql: str, db_path: str, timeout: float = 30.0) -> Tuple[bool, float]:
    """测量SQL执行时间.
    
    Returns:
        (is_executable, execution_time_seconds)
    """
    try:
        conn = sqlite3.connect(db_path, timeout=timeout)
        cursor = conn.cursor()
        
        start_time = time.perf_counter()
        cursor.execute(sql)
        _ = cursor.fetchall()
        end_time = time.perf_counter()
        
        conn.close()
        return True, end_time - start_time
        
    except Exception as e:
        logger.debug(f"SQL execution failed: {e}")
        return False, float('inf')


# ============================================================================
# 第五部分：AST 相似度计算
# ============================================================================

def tokenize_sql(sql: str) -> List[str]:
    """将SQL解析为token列表.
    
    使用 sqlparse 提取 token，并进行标准化处理。
    """
    if not sql:
        return []
    
    tokens = []
    parsed = sqlparse.parse(sql)
    if not parsed:
        return []
    
    def extract_tokens(token_list):
        for token in token_list:
            if token.is_group:
                extract_tokens(token.tokens)
            elif not token.is_whitespace:
                # 标准化：转大写，去除多余空格
                tval = str(token).strip().upper()
                if tval:
                    tokens.append(tval)
    
    extract_tokens(parsed[0].tokens)
    return tokens


def calculate_ast_similarity(sql1: str, sql2: str) -> float:
    """计算两个SQL的AST相似度（基于Jaccard相似度）.
    
    使用 token 的 Jaccard 相似度，计算速度快，适合作为前置过滤器。
    
    Args:
        sql1: 第一个SQL
        sql2: 第二个SQL
    
    Returns:
        相似度 [0.0, 1.0]，1.0 表示完全相同
    """
    tokens1 = tokenize_sql(sql1)
    tokens2 = tokenize_sql(sql2)
    
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    
    # 使用多重集计算 Jaccard 相似度
    from collections import Counter
    c1 = Counter(tokens1)
    c2 = Counter(tokens2)
    
    # 交集：每个元素取最小计数
    intersection = sum((c1 & c2).values())
    # 并集：每个元素取最大计数
    union = sum((c1 | c2).values())
    
    if union == 0:
        return 1.0
    
    return intersection / union


# ============================================================================
# 第六部分：分层奖励计算器
# ============================================================================

class LayeredRewardCalculator:
    """分层奖励计算器.
    
    奖励计算策略：
    1. 最终结果匹配 → 1.0
    2. UNION/INTERSECT/EXCEPT → 直接判断最终结果，不匹配返回0
    3. 中间状态从后往前对比(g→a)，使用AST相似度前置过滤，奖励递减（最高0.7）
    4. 全部不匹配 → 0
    """
    
    def __init__(self, db_path: str, ast_threshold: float = AST_SIMILARITY_THRESHOLD):
        """初始化.
        
        Args:
            db_path: SQLite数据库路径
            ast_threshold: AST相似度阈值，低于此阈值时跳过结果比较
        """
        self.db_path = db_path
        self.ast_threshold = ast_threshold
        self.builder = IntermediateSQLBuilder()
        self._binary_reward = get_binary_reward_func()
        self._simple_binary_reward = get_simple_binary_reward_func()
    
    def _timed_binary_reward(
        self,
        pred_sql: str,
        gold_sql: str,
        stats: PerformanceStats,
        context: str = "",
    ) -> float:
        """带计时的 binary_reward 调用.
        
        注意：binary_reward 内部会执行 pred_sql 和 gold_sql 两条 SQL，然后比较结果。
        因此这个时间包含了 SQL 执行时间 + 结果比较时间。
        
        Args:
            pred_sql: 预测SQL
            gold_sql: 标准SQL
            stats: 性能统计对象
            context: 调用上下文描述
        
        Returns:
            binary_reward 结果
        """
        start = time.perf_counter()
        result = self._binary_reward(pred_sql, gold_sql, self.db_path, raise_on_error=False)
        elapsed = time.perf_counter() - start
        
        stats.binary_reward_call_count += 1
        stats.binary_reward_time += elapsed
        stats.binary_reward_details.append({
            "context": context,
            "pred_sql": pred_sql[:100] + "..." if len(pred_sql) > 100 else pred_sql,
            "gold_sql": gold_sql[:100] + "..." if len(gold_sql) > 100 else gold_sql,
            "result": result,
            "time_sec": round(elapsed, 6),
        })
        
        return result
    
    def _timed_simple_binary_reward(
        self,
        pred_sql: str,
        gold_sql: str,
        stats: PerformanceStats,
        context: str = "",
    ) -> float:
        """带计时的简化版 binary_reward 调用.
        
        简化版不做列排列枚举，速度更快。
        专用于中间阶段比较，因为中间SQL结构已经通过AST相似度过滤。
        
        Args:
            pred_sql: 预测SQL
            gold_sql: 标准SQL
            stats: 性能统计对象
            context: 调用上下文描述
        
        Returns:
            simple_binary_reward 结果
        """
        start = time.perf_counter()
        result = self._simple_binary_reward(pred_sql, gold_sql, self.db_path, raise_on_error=False)
        elapsed = time.perf_counter() - start
        
        stats.binary_reward_call_count += 1
        stats.binary_reward_time += elapsed
        stats.binary_reward_details.append({
            "context": context + " (simple)",
            "pred_sql": pred_sql[:100] + "..." if len(pred_sql) > 100 else pred_sql,
            "gold_sql": gold_sql[:100] + "..." if len(gold_sql) > 100 else gold_sql,
            "result": result,
            "time_sec": round(elapsed, 6),
        })
        
        return result
    
    def _timed_ast_similarity(
        self,
        sql1: str,
        sql2: str,
        stats: PerformanceStats,
        context: str = "",
    ) -> float:
        """带计时的 AST 相似度计算.
        
        Args:
            sql1: 第一个SQL
            sql2: 第二个SQL
            stats: 性能统计对象
            context: 调用上下文描述
        
        Returns:
            AST 相似度 [0.0, 1.0]
        """
        start = time.perf_counter()
        similarity = calculate_ast_similarity(sql1, sql2)
        elapsed = time.perf_counter() - start
        
        stats.ast_similarity_count += 1
        stats.ast_similarity_time += elapsed
        stats.ast_similarity_details.append({
            "context": context,
            "sql1": sql1[:80] + "..." if len(sql1) > 80 else sql1,
            "sql2": sql2[:80] + "..." if len(sql2) > 80 else sql2,
            "similarity": round(similarity, 4),
            "time_sec": round(elapsed, 6),
        })
        
        return similarity
    
    def _timed_clause_extraction(
        self,
        sql: str,
        stats: PerformanceStats,
        context: str = "",
    ) -> Dict[SQLStage, IntermediateSQL]:
        """带计时的子句提取.
        
        Args:
            sql: 要解析的SQL
            stats: 性能统计对象
            context: 调用上下文描述
        
        Returns:
            中间SQL字典
        """
        start = time.perf_counter()
        result = self.builder.build_all_intermediate_sqls(sql)
        elapsed = time.perf_counter() - start
        
        stats.clause_extraction_count += 1
        stats.clause_extraction_time += elapsed
        stats.clause_extraction_details.append({
            "context": context,
            "sql": sql[:100] + "..." if len(sql) > 100 else sql,
            "stages_extracted": [s.value for s in result.keys()],
            "time_sec": round(elapsed, 6),
        })
        
        return result
    
    def calculate_reward(
        self,
        pred_sql: str,
        gold_sql: str,
    ) -> Tuple[float, Dict[str, Any]]:
        """计算分层复合奖励.
        
        Args:
            pred_sql: 预测SQL
            gold_sql: 标准SQL
        
        Returns:
            (reward, details) - details 中包含 performance_stats 字段，记录详细的性能统计信息
        """
        total_start = time.perf_counter()
        stats = PerformanceStats()
        
        details: Dict[str, Any] = {
            "pred_sql": pred_sql,
            "gold_sql": gold_sql,
            "final_match": False,
            "is_compound": False,
            "stage_results": {},
            "matched_stage": None,
            "reward_type": "",
            "base_reward": 0.0,
            "ast_threshold": self.ast_threshold,
        }
        
        def finalize_and_return(reward: float) -> Tuple[float, Dict[str, Any]]:
            """统一的返回处理，确保性能统计被正确记录."""
            stats.total_time = time.perf_counter() - total_start
            details["performance_stats"] = stats.to_dict()
            logger.debug(stats.summary())
            return reward, details
        
        # Step 0: 检查最终结果是否匹配
        try:
            final_match = self._timed_binary_reward(
                pred_sql, gold_sql, stats, context="final_match_check"
            )
            details["final_match"] = (final_match == 1.0)
            
            if final_match == 1.0:
                # 最终结果匹配，直接返回 1.0
                details["reward_type"] = "final_match"
                details["base_reward"] = 1.0
                return finalize_and_return(1.0)
                
        except Exception as e:
            logger.warning(f"Final comparison failed: {e}")
            details["error"] = str(e)
        
        # Step 1: 检查是否是复合查询
        pred_parser = SQLParser(pred_sql)
        gold_parser = SQLParser(gold_sql)
        
        if pred_parser.is_compound() or gold_parser.is_compound():
            details["is_compound"] = True
            # 复合查询不匹配：直接返回0（不做中间状态分解）
            details["reward_type"] = "compound_no_match"
            details["base_reward"] = 0.0
            return finalize_and_return(0.0)
        
        # Step 2: 提取中间状态
        pred_intermediates = self._timed_clause_extraction(
            pred_sql, stats, context="pred_clause_extraction"
        )
        gold_intermediates = self._timed_clause_extraction(
            gold_sql, stats, context="gold_clause_extraction"
        )
        
        details["pred_stages"] = {k.value: v.sql for k, v in pred_intermediates.items()}
        details["gold_stages"] = {k.value: v.sql for k, v in gold_intermediates.items()}
        
        # Step 3: 从后往前比较中间状态 (g → a)
        for stage in COMPARISON_ORDER:
            pred_inter = pred_intermediates.get(stage)
            gold_inter = gold_intermediates.get(stage)
            
            stage_result = {
                "pred_exists": pred_inter is not None,
                "gold_exists": gold_inter is not None,
                "match": None,
                "reason": "",
            }
            
            # 两边都没有这个阶段，跳到下一个
            if not pred_inter and not gold_inter:
                stage_result["reason"] = "两边都没有此阶段"
                details["stage_results"][stage.value] = stage_result
                continue
            
            # 只有一边有，跳到下一个阶段
            if not pred_inter or not gold_inter:
                stage_result["reason"] = "只有一边有此阶段，跳过"
                details["stage_results"][stage.value] = stage_result
                continue
            
            # 两边都有，先计算 AST 相似度进行前置过滤
            ast_sim = self._timed_ast_similarity(
                pred_inter.sql,
                gold_inter.sql,
                stats,
                context=f"stage_{stage.value}_ast_filter"
            )
            stage_result["ast_similarity"] = round(ast_sim, 4)
            
            # 如果 AST 相似度低于阈值，跳过耗时的结果比较
            if ast_sim < self.ast_threshold:
                stats.ast_filtered_count += 1
                stage_result["reason"] = f"AST相似度{ast_sim:.2f}低于阈值{self.ast_threshold}，跳过"
                stage_result["match"] = False
                details["stage_results"][stage.value] = stage_result
                continue
            
            # AST 相似度达标，进行结果比较
            # 使用简化版比较，不做列排列枚举，速度更快
            try:
                match = self._timed_simple_binary_reward(
                    pred_inter.sql,
                    gold_inter.sql,
                    stats,
                    context=f"stage_{stage.value}_comparison"
                )
                stage_result["match"] = (match == 1.0)
                
                if match == 1.0:
                    # 找到匹配的阶段
                    stage_result["reason"] = "结果匹配"
                    details["stage_results"][stage.value] = stage_result
                    details["matched_stage"] = stage.value
                    
                    reward = STAGE_REWARD_MAP[stage]
                    details["base_reward"] = reward
                    details["reward_type"] = f"stage_match_{stage.value}"
                    
                    return finalize_and_return(reward)
                else:
                    stage_result["reason"] = "结果不匹配"
                    
            except Exception as e:
                logger.debug(f"Stage {stage.value} comparison failed: {e}")
                stage_result["match"] = None
                stage_result["reason"] = f"比较失败: {str(e)}"
            
            details["stage_results"][stage.value] = stage_result
        
        # Step 4: 所有阶段都不匹配，返回0
        details["reward_type"] = "no_stage_match"
        details["base_reward"] = 0.0
        
        return finalize_and_return(0.0)


# ============================================================================
# 第七部分：对外接口
# ============================================================================

def composite_reward(
    pred_sql: str,
    gold_sql: str,
    db_path: str,
    raise_on_error: bool = False,
) -> float:
    """计算分层复合奖励（简化接口）.
    
    Args:
        pred_sql: 预测SQL
        gold_sql: 标准SQL
        db_path: 数据库路径
        raise_on_error: 是否抛出异常
    
    Returns:
        奖励值
    """
    try:
        calculator = LayeredRewardCalculator(db_path)
        reward, _ = calculator.calculate_reward(pred_sql, gold_sql)
        return reward
    except Exception as e:
        if raise_on_error:
            raise
        logger.exception(f"Composite reward calculation failed: {e}")
        return 0.0


def composite_reward_with_details(
    pred_sql: str,
    gold_sql: str,
    db_path: str,
    raise_on_error: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    """计算分层复合奖励并返回详细信息.
    
    Args:
        pred_sql: 预测SQL
        gold_sql: 标准SQL  
        db_path: 数据库路径
        raise_on_error: 是否抛出异常
    
    Returns:
        (reward, details)
    """
    try:
        calculator = LayeredRewardCalculator(db_path)
        return calculator.calculate_reward(pred_sql, gold_sql)
    except Exception as e:
        if raise_on_error:
            raise
        logger.exception(f"Composite reward calculation failed: {e}")
        return 0.0, {"error": str(e)}


# ============================================================================
# 第八部分：导出和测试辅助
# ============================================================================

# 为了兼容性，保留一些旧的导出名称
CATEGORY_REWARD_MAP = STAGE_REWARD_MAP
BASE_REWARD = 0.05
EXECUTABLE_BONUS = 0.0  # 不再单独给可执行奖励
# FASTER_BONUS = SPEED_BONUS


class SQLClauseExtractor:
    """SQL子句提取器（兼容性包装）."""
    
    def __init__(self):
        self.parser = None
    
    def extract_all_clauses(self, sql: str) -> Dict[str, Any]:
        """提取SQL的所有子句."""
        self.parser = SQLParser(sql)
        clauses = self.parser.extract_clauses()
        
        return {
            "select": clauses.select,
            "distinct": clauses.distinct,
            "from": clauses.from_clause,
            "joins": clauses.joins,
            "where": clauses.where,
            "group_by": clauses.group_by,
            "having": clauses.having,
            "order_by": clauses.order_by,
            "limit": clauses.limit,
        }


def debug_intermediate_sqls(sql: str) -> None:
    """调试：打印SQL的所有中间状态."""
    builder = IntermediateSQLBuilder()
    intermediates = builder.build_all_intermediate_sqls(sql)
    
    print(f"\n{'='*60}")
    print(f"Original SQL: {sql}")
    print(f"{'='*60}")
    
    for stage in [SQLStage.FROM_JOIN, SQLStage.WHERE, SQLStage.GROUP_BY, 
                  SQLStage.HAVING, SQLStage.SELECT, SQLStage.DISTINCT, SQLStage.ORDER_BY]:
        inter = intermediates.get(stage)
        if inter:
            print(f"\n[{stage.value}] {inter.description}")
            print(f"  {inter.sql}")
        else:
            print(f"\n[{stage.value}]")
            print(f"  (not applicable)")


def debug_reward_calculation(pred_sql: str, gold_sql: str, db_path: str) -> None:
    """调试：打印奖励计算的详细过程."""
    reward, details = composite_reward_with_details(pred_sql, gold_sql, db_path)
    
    print(f"\n{'='*60}")
    print(f"Reward Calculation Debug")
    print(f"{'='*60}")
    print(f"Pred SQL: {pred_sql}")
    print(f"Gold SQL: {gold_sql}")
    print(f"Final Match: {details.get('final_match', False)}")
    print(f"Is Compound: {details.get('is_compound', False)}")
    print(f"Reward Type: {details.get('reward_type', '')}")
    print(f"Matched Stage: {details.get('matched_stage', 'None')}")
    print(f"AST Threshold: {details.get('ast_threshold', 0.5)}")
    print(f"{'='*60}")
    print(f"FINAL REWARD: {reward:.4f}")
    print(f"{'='*60}")
    
    # 打印性能统计
    perf_stats = details.get("performance_stats", {})
    if perf_stats:
        print(f"\n{'='*60}")
        print(f"Performance Statistics")
        print(f"{'='*60}")
        print(f"binary_reward Calls: {perf_stats.get('binary_reward_call_count', 0)}")
        print(f"Estimated SQL Executions: {perf_stats.get('estimated_sql_exec_count', 0)}")
        print(f"AST Similarity Checks: {perf_stats.get('ast_similarity_count', 0)}")
        print(f"AST Filtered (skipped): {perf_stats.get('ast_filtered_count', 0)}")
        print(f"Clause Extraction Count: {perf_stats.get('clause_extraction_count', 0)}")
        print(f"{'='*60}")
        print(f"Clause Extraction Time: {perf_stats.get('clause_extraction_time_sec', 0)*1000:.2f} ms")
        print(f"AST Similarity Time: {perf_stats.get('ast_similarity_time_sec', 0)*1000:.2f} ms")
        print(f"binary_reward Time: {perf_stats.get('binary_reward_time_sec', 0)*1000:.2f} ms")
        print(f"Total Time: {perf_stats.get('total_time_sec', 0)*1000:.2f} ms")
        print(f"{'='*60}")


if __name__ == "__main__":
    # 测试示例
    test_sql = """
    SELECT T1.name, COUNT(T2.id) 
    FROM students T1 
    JOIN enrollments T2 ON T1.id = T2.student_id 
    WHERE T1.age > 18 
    GROUP BY T1.name 
    HAVING COUNT(T2.id) > 2
    ORDER BY COUNT(T2.id) DESC
    """
    
    debug_intermediate_sqls(test_sql)
