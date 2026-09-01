"""快速分层复合奖励系统 - 基于结构化比较优化.

主要优化：
1. FROM/JOIN 阶段：结构比较（表集合 + 连接关系），不执行SQL
2. GROUP BY 阶段：列集合比较，不执行SQL
3. SELECT 阶段：列集合比较（Spider不关心顺序），不执行SQL
4. DISTINCT 阶段：关键字检查，不执行SQL
5. ORDER BY 阶段：字符串标准化比较，不执行SQL
6. WHERE/HAVING 阶段：使用 simple_binary_reward 执行SQL比较

奖励计算策略：
1. 最终结果匹配 → 1.0
2. UNION/INTERSECT/EXCEPT 复合查询不匹配 → 0
3. 从前往后顺序对比(a→g)，返回最后通过阶段的奖励（0.40→0.70）
4. 全部不匹配 → 0
"""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import sqlparse

# 导入奖励函数
try:
    from reward_func.reward import binary_reward, simple_binary_reward
except ImportError:
    binary_reward = None
    simple_binary_reward = None

logger = logging.getLogger(__name__)


def get_binary_reward_func():
    """获取 binary_reward 函数（用于最终结果检查）."""
    if binary_reward is None:
        raise RuntimeError(
            "binary_reward function not available. "
            "Please ensure reward_func.reward can be imported."
        )
    return binary_reward


def get_simple_binary_reward_func():
    """获取 simple_binary_reward 函数（用于 WHERE/HAVING 阶段检查）."""
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
    """SQL执行阶段枚举.
    
    注意：DISTINCT不作为独立阶段，遵循Spider官方评测标准（keep_distinct=False）。
    """
    FROM_JOIN = "from_join"      # a. 选表阶段
    WHERE = "where"              # b. 行过滤阶段
    GROUP_BY = "group_by"        # c. 分组阶段
    HAVING = "having"            # d. 组过滤阶段
    SELECT = "select"            # e. 列投影阶段
    ORDER_BY = "order_by"        # f. 排序阶段


# 从前往后的比较顺序（不含DISTINCT，遵循Spider官方标准）
COMPARISON_ORDER = [
    SQLStage.FROM_JOIN,   # a (0.40)
    SQLStage.WHERE,       # b (0.45)
    SQLStage.GROUP_BY,    # c (0.50)
    SQLStage.HAVING,      # d (0.55)
    SQLStage.SELECT,      # e (0.60)
    SQLStage.ORDER_BY,    # f (0.70)
]

# 阶段奖励映射
STAGE_REWARD_MAP = {
    SQLStage.FROM_JOIN: 0.40,   # a 选表正确
    SQLStage.WHERE: 0.45,       # b 行过滤正确
    SQLStage.GROUP_BY: 0.50,    # c 分组正确
    SQLStage.HAVING: 0.55,      # d 组过滤正确
    SQLStage.SELECT: 0.60,      # e 列投影正确
    SQLStage.ORDER_BY: 0.70,    # f 排序正确
}

# 比较方式：结构比较 vs 执行SQL
STRUCTURAL_COMPARE_STAGES = {
    SQLStage.FROM_JOIN,
    SQLStage.GROUP_BY,
    SQLStage.SELECT,
    SQLStage.ORDER_BY,
}

EXECUTE_SQL_STAGES = {
    SQLStage.WHERE,
    SQLStage.HAVING,
}


# ============================================================================
# 第二部分：数据结构定义
# ============================================================================

@dataclass
class PerformanceStats:
    """性能统计信息."""
    binary_reward_call_count: int = 0
    structural_compare_count: int = 0
    clause_extraction_count: int = 0
    
    clause_extraction_time: float = 0.0
    structural_compare_time: float = 0.0
    binary_reward_time: float = 0.0
    total_time: float = 0.0
    
    details: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式."""
        return {
            "binary_reward_call_count": self.binary_reward_call_count,
            "structural_compare_count": self.structural_compare_count,
            "clause_extraction_count": self.clause_extraction_count,
            "clause_extraction_time_sec": round(self.clause_extraction_time, 6),
            "structural_compare_time_sec": round(self.structural_compare_time, 6),
            "binary_reward_time_sec": round(self.binary_reward_time, 6),
            "total_time_sec": round(self.total_time, 6),
            "details": self.details,
        }
    
    def summary(self) -> str:
        """生成简洁的摘要字符串."""
        return (
            f"PerformanceStats: "
            f"binary_reward_calls={self.binary_reward_call_count}, "
            f"structural_compares={self.structural_compare_count}, "
            f"extraction_time={self.clause_extraction_time*1000:.2f}ms, "
            f"structural_time={self.structural_compare_time*1000:.2f}ms, "
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
    
    original_sql: str = ""
    is_compound: bool = False


@dataclass
class FromJoinStructure:
    """FROM/JOIN 结构化表示."""
    tables: FrozenSet[str] = field(default_factory=frozenset)  # 表名集合（小写）
    join_conditions: FrozenSet[FrozenSet[str]] = field(default_factory=frozenset)  # 连接条件集合


@dataclass
class IntermediateSQL:
    """中间SQL状态."""
    stage: SQLStage
    sql: str
    clauses: Optional[SQLClauses] = None
    description: str = ""


# ============================================================================
# 第三部分：SQL解析器
# ============================================================================

class SQLParser:
    """SQL 解析器."""
    
    def __init__(self, sql: str):
        self.original_sql = self._normalize(sql)
        self.parsed = None
        if self.original_sql:
            statements = sqlparse.parse(self.original_sql)
            if statements:
                self.parsed = statements[0]
    
    def _normalize(self, sql: str) -> str:
        """标准化SQL."""
        if not sql:
            return ""
        sql = sql.strip().rstrip(';').strip()
        return re.sub(r'\s+', ' ', sql)
    
    def is_compound(self) -> bool:
        """检查是否是复合查询."""
        if not self.original_sql:
            return False
        
        sql_upper = self.original_sql.upper()
        keywords = ['UNION', 'INTERSECT', 'EXCEPT']
        
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
                for kw in keywords:
                    if sql_upper[i:].startswith(kw):
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
        
        if clauses.is_compound:
            return clauses
        
        sql = self.original_sql
        
        clauses.distinct = bool(re.search(r'\bSELECT\s+DISTINCT\b', sql, re.I))
        clauses.select = self._extract_select(sql)
        clauses.from_clause = self._extract_from(sql)
        clauses.joins = self._extract_joins(sql)
        clauses.where = self._extract_where(sql)
        clauses.group_by = self._extract_group_by(sql)
        clauses.having = self._extract_having(sql)
        clauses.order_by = self._extract_order_by(sql)
        clauses.limit = self._extract_limit(sql)
        
        return clauses
    
    def _extract_select(self, sql: str) -> str:
        """提取SELECT列."""
        m = re.search(r'\bSELECT\s+(?:DISTINCT\s+)?(.*?)\s+FROM\b', sql, re.I | re.S)
        if m:
            return m.group(1).strip()
        return "*"
    
    def _extract_from(self, sql: str) -> str:
        """提取FROM子句（主表）."""
        m = re.search(r'\bFROM\s+', sql, re.I)
        if not m:
            return ""
        
        start = m.end()
        end = self._find_clause_end(sql, start, 
            ['JOIN', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'FULL JOIN', 'CROSS JOIN',
             'WHERE', 'GROUP BY', 'HAVING', 'ORDER BY', 'LIMIT', 'UNION', 'INTERSECT', 'EXCEPT'])
        
        return sql[start:end].strip()
    
    def _extract_joins(self, sql: str) -> List[str]:
        """提取所有JOIN子句."""
        joins = []
        
        join_pattern = re.compile(
            r'\b((?:INNER\s+|LEFT\s+(?:OUTER\s+)?|RIGHT\s+(?:OUTER\s+)?|FULL\s+(?:OUTER\s+)?|CROSS\s+)?JOIN)\s+',
            re.I
        )
        
        for m in join_pattern.finditer(sql):
            join_start = m.start()
            join_keyword = m.group(1)
            content_start = m.end()
            
            if self._is_inside_subquery(sql, join_start):
                continue
            
            end = self._find_clause_end(sql, content_start,
                ['JOIN', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'FULL JOIN', 'CROSS JOIN',
                 'WHERE', 'GROUP BY', 'HAVING', 'ORDER BY', 'LIMIT', 'UNION', 'INTERSECT', 'EXCEPT'])
            
            join_content = sql[content_start:end].strip()
            joins.append(f"{join_keyword.upper()} {join_content}")
        
        return joins
    
    def _extract_where(self, sql: str) -> str:
        """提取WHERE子句."""
        m = re.search(r'\bWHERE\s+', sql, re.I)
        if not m:
            return ""
        
        start = m.end()
        end = self._find_clause_end(sql, start,
            ['GROUP BY', 'HAVING', 'ORDER BY', 'LIMIT', 'UNION', 'INTERSECT', 'EXCEPT'])
        
        return sql[start:end].strip()
    
    def _extract_group_by(self, sql: str) -> str:
        """提取GROUP BY子句."""
        m = re.search(r'\bGROUP\s+BY\s+', sql, re.I)
        if not m:
            return ""
        
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
        
        if self._is_inside_subquery(sql, m.start()):
            return ""
        
        return m.group(1).strip()
    
    def _find_clause_end(self, sql: str, start: int, terminators: List[str]) -> int:
        """找到子句的结束位置."""
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
                for term in terminators:
                    term_upper = term.upper()
                    if sql_upper[i:].startswith(term_upper):
                        if i == 0 or not sql[i-1].isalnum():
                            end_pos = i + len(term)
                            if end_pos >= len(sql) or not sql[end_pos].isalnum():
                                return i
                i += 1
            else:
                i += 1
        
        return len(sql)
    
    def _is_inside_subquery(self, sql: str, pos: int) -> bool:
        """检查位置是否在子查询内."""
        depth = 0
        for i in range(pos):
            if sql[i] == '(':
                depth += 1
            elif sql[i] == ')':
                depth -= 1
        return depth > 0


# ============================================================================
# 第四部分：结构化比较函数
# ============================================================================

def _extract_alias_mapping(clauses: SQLClauses) -> Dict[str, str]:
    """从SQLClauses中提取别名到表名的映射.
    
    支持的格式：
    - table_name AS alias
    - table_name alias
    - table_name（无别名，表名即为自己的键）
    - 逗号分隔的多表：table1 AS t1, table2 AS t2
    """
    alias_to_table: Dict[str, str] = {}
    
    # 从 from_clause 提取（可能是逗号分隔的多表）
    if clauses.from_clause:
        for part in clauses.from_clause.split(','):
            part = part.strip()
            if not part or part.startswith('('):
                continue
            # 匹配 table [AS] alias
            m = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)\s+(?:AS\s+)?([a-zA-Z_][a-zA-Z0-9_]*)', part, re.I)
            if m:
                table = m.group(1).lower()
                alias = m.group(2).lower()
                # 跳过SQL关键字被误识为别名的情况
                if alias not in ('on', 'where', 'group', 'having', 'order', 'limit', 'join',
                                 'inner', 'left', 'right', 'full', 'cross', 'natural'):
                    alias_to_table[alias] = table
            else:
                # 无别名，表名即为自己的键
                m2 = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)', part)
                if m2:
                    table = m2.group(1).lower()
                    alias_to_table[table] = table
    
    # 从 joins 提取
    for join in clauses.joins:
        # 匹配 JOIN table [AS] alias ON ...
        m = re.search(
            r'^(?:INNER\s+|LEFT\s+(?:OUTER\s+)?|RIGHT\s+(?:OUTER\s+)?|FULL\s+(?:OUTER\s+)?|CROSS\s+)?'
            r'JOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+(?:AS\s+)?([a-zA-Z_][a-zA-Z0-9_]*)',
            join, re.I
        )
        if m:
            table = m.group(1).lower()
            alias = m.group(2).lower()
            if alias not in ('on', 'where', 'group', 'having', 'order', 'limit'):
                alias_to_table[alias] = table
        else:
            # JOIN table ON ... （无别名）
            m2 = re.search(
                r'^(?:INNER\s+|LEFT\s+(?:OUTER\s+)?|RIGHT\s+(?:OUTER\s+)?|FULL\s+(?:OUTER\s+)?|CROSS\s+)?'
                r'JOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+ON\b',
                join, re.I
            )
            if m2:
                table = m2.group(1).lower()
                alias_to_table[table] = table
    
    return alias_to_table


def _extract_tables_from_clauses(clauses: SQLClauses) -> FrozenSet[str]:
    """从SQLClauses中提取表集合.
    
    支持逗号分隔的多表FROM，如 FROM students, courses。
    """
    tables: Set[str] = set()
    
    # 从 from_clause 提取（支持逗号分隔的多表）
    if clauses.from_clause:
        for part in clauses.from_clause.split(','):
            part = part.strip()
            if not part or part.startswith('('):
                continue
            # 取第一个标识符作为表名（忽略别名）
            m = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)', part)
            if m:
                tables.add(m.group(1).lower())
    
    # 从 joins 提取
    for join in clauses.joins:
        # JOIN table AS alias ON ...
        m = re.search(r'^(?:INNER\s+|LEFT\s+(?:OUTER\s+)?|RIGHT\s+(?:OUTER\s+)?|FULL\s+(?:OUTER\s+)?|CROSS\s+)?JOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)', join, re.I)
        if m:
            tables.add(m.group(1).lower())
    
    return frozenset(tables)


def _extract_join_conditions_from_clauses(clauses: SQLClauses, alias_to_table: Dict[str, str]) -> FrozenSet[FrozenSet[str]]:
    """从SQLClauses中提取连接条件集合.
    
    返回连接条件集合，每个条件是一个frozenset，包含两个列引用（格式：table.column）。
    使用frozenset表示无序对，这样 a.x=b.y 和 b.y=a.x 会被视为相同。
    """
    conditions: Set[FrozenSet[str]] = set()
    
    def resolve_column_ref(ref: str) -> str:
        """解析列引用，将别名转换为表名."""
        ref = ref.strip().lower()
        if '.' in ref:
            parts = ref.split('.')
            if len(parts) == 2:
                alias_or_table = parts[0].strip()
                column = parts[1].strip()
                # 尝试将别名转换为表名
                table = alias_to_table.get(alias_or_table, alias_or_table)
                return f"{table}.{column}"
        return ref
    
    # 从 joins 中提取 ON 条件
    for join in clauses.joins:
        # 查找 ON 子句
        on_match = re.search(r'\bON\s+(.+)$', join, re.I)
        if on_match:
            on_clause = on_match.group(1)
            
            # 提取所有等值条件 col1 = col2
            eq_pattern = re.compile(r'([a-zA-Z_][a-zA-Z0-9_.]*)\s*=\s*([a-zA-Z_][a-zA-Z0-9_.]*)', re.I)
            for eq_match in eq_pattern.finditer(on_clause):
                left = resolve_column_ref(eq_match.group(1))
                right = resolve_column_ref(eq_match.group(2))
                # 使用 frozenset 表示无序对
                conditions.add(frozenset([left, right]))
    
    return frozenset(conditions)


def compare_from_join_structure(clauses1: SQLClauses, clauses2: SQLClauses) -> Tuple[bool, Dict[str, Any]]:
    """比较两个SQL的FROM/JOIN结构是否等价.
    
    比较规则：
    1. 表集合相等
    2. 连接条件集合相等（忽略顺序和方向）
    
    Returns:
        (is_equal, details)
    """
    # 提取别名映射
    alias1 = _extract_alias_mapping(clauses1)
    alias2 = _extract_alias_mapping(clauses2)
    
    # 提取表集合
    tables1 = _extract_tables_from_clauses(clauses1)
    tables2 = _extract_tables_from_clauses(clauses2)
    
    # 提取连接条件
    conditions1 = _extract_join_conditions_from_clauses(clauses1, alias1)
    conditions2 = _extract_join_conditions_from_clauses(clauses2, alias2)
    
    tables_match = (tables1 == tables2)
    conditions_match = (conditions1 == conditions2)
    is_equal = tables_match and conditions_match
    
    details = {
        "tables1": sorted(tables1),
        "tables2": sorted(tables2),
        "tables_match": tables_match,
        "conditions1": [sorted(list(c)) for c in sorted(conditions1, key=lambda x: sorted(x))],
        "conditions2": [sorted(list(c)) for c in sorted(conditions2, key=lambda x: sorted(x))],
        "conditions_match": conditions_match,
        "is_equal": is_equal,
    }
    
    return is_equal, details


def _normalize_column_expr(expr: str, alias_map: Dict[str, str] | None = None) -> str:
    """标准化列表达式.
    
    - 转小写
    - 移除多余空格
    - 移除列别名（AS xxx）
    - 将表别名替换为实际表名（如 t1.name → students.name）
    """
    expr = expr.strip().lower()
    expr = re.sub(r'\s+', ' ', expr)
    # 移除列别名
    expr = re.sub(r'\s+as\s+[a-zA-Z_][a-zA-Z0-9_]*$', '', expr, flags=re.I)
    # 将表别名替换为实际表名
    if alias_map:
        def _replace_alias(m: re.Match) -> str:
            alias = m.group(1)
            col = m.group(2)
            real_table = alias_map.get(alias, alias)
            return f"{real_table}.{col}"
        expr = re.sub(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_*]*)\b', _replace_alias, expr)
    return expr


def _extract_column_set(select_clause: str, alias_map: Dict[str, str] | None = None) -> FrozenSet[str]:
    """从子句提取列集合.
    
    处理逗号分隔的列列表，注意处理括号内的逗号（如函数参数）。
    支持将表别名替换为实际表名。
    """
    if not select_clause or select_clause.strip() == '*':
        return frozenset(['*'])
    
    columns: Set[str] = set()
    current = ""
    depth = 0
    
    for char in select_clause:
        if char == '(':
            depth += 1
            current += char
        elif char == ')':
            depth -= 1
            current += char
        elif char == ',' and depth == 0:
            col = _normalize_column_expr(current, alias_map)
            if col:
                columns.add(col)
            current = ""
        else:
            current += char
    
    # 最后一列
    col = _normalize_column_expr(current, alias_map)
    if col:
        columns.add(col)
    
    return frozenset(columns)


def compare_select_columns(clauses1: SQLClauses, clauses2: SQLClauses) -> Tuple[bool, Dict[str, Any]]:
    """比较SELECT列集合是否相等.
    
    将表别名替换为实际表名后再比较，避免 t1.name vs t2.name 误判。
    """
    alias_map1 = _extract_alias_mapping(clauses1)
    alias_map2 = _extract_alias_mapping(clauses2)
    cols1 = _extract_column_set(clauses1.select, alias_map1)
    cols2 = _extract_column_set(clauses2.select, alias_map2)
    
    is_equal = (cols1 == cols2)
    
    details = {
        "columns1": sorted(cols1),
        "columns2": sorted(cols2),
        "is_equal": is_equal,
    }
    
    return is_equal, details


def compare_group_by(clauses1: SQLClauses, clauses2: SQLClauses) -> Tuple[bool, Dict[str, Any]]:
    """比较GROUP BY列集合是否相等.
    
    将表别名替换为实际表名后再比较。
    """
    alias_map1 = _extract_alias_mapping(clauses1)
    alias_map2 = _extract_alias_mapping(clauses2)
    cols1 = _extract_column_set(clauses1.group_by, alias_map1) if clauses1.group_by else frozenset()
    cols2 = _extract_column_set(clauses2.group_by, alias_map2) if clauses2.group_by else frozenset()
    
    is_equal = (cols1 == cols2)
    
    details = {
        "columns1": sorted(cols1),
        "columns2": sorted(cols2),
        "is_equal": is_equal,
    }
    
    return is_equal, details


def compare_distinct(clauses1: SQLClauses, clauses2: SQLClauses) -> Tuple[bool, Dict[str, Any]]:
    """比较DISTINCT是否一致."""
    is_equal = (clauses1.distinct == clauses2.distinct)
    
    details = {
        "distinct1": clauses1.distinct,
        "distinct2": clauses2.distinct,
        "is_equal": is_equal,
    }
    
    return is_equal, details


def _normalize_order_by(order_by: str, alias_map: Dict[str, str] | None = None) -> str:
    """标准化ORDER BY子句.
    
    - 转小写
    - 移除多余空格
    - 将表别名替换为实际表名
    """
    if not order_by:
        return ""
    
    order_by = order_by.strip().lower()
    order_by = re.sub(r'\s+', ' ', order_by)
    # 将表别名替换为实际表名
    if alias_map:
        def _replace_alias(m: re.Match) -> str:
            alias = m.group(1)
            col = m.group(2)
            real_table = alias_map.get(alias, alias)
            return f"{real_table}.{col}"
        order_by = re.sub(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_*]*)\b', _replace_alias, order_by)
    return order_by


def compare_order_by(clauses1: SQLClauses, clauses2: SQLClauses) -> Tuple[bool, Dict[str, Any]]:
    """比较ORDER BY是否相等.
    
    将表别名替换为实际表名后再比较。
    """
    alias_map1 = _extract_alias_mapping(clauses1)
    alias_map2 = _extract_alias_mapping(clauses2)
    order1 = _normalize_order_by(clauses1.order_by, alias_map1)
    order2 = _normalize_order_by(clauses2.order_by, alias_map2)
    
    is_equal = (order1 == order2)
    
    details = {
        "order_by1": order1,
        "order_by2": order2,
        "is_equal": is_equal,
    }
    
    return is_equal, details


# ============================================================================
# 第五部分：中间SQL构建器
# ============================================================================

class IntermediateSQLBuilder:
    """中间SQL构建器."""
    
    def __init__(self):
        self.parser = None
    
    def build_all_intermediate_sqls(self, sql: str) -> Tuple[SQLClauses, Dict[SQLStage, IntermediateSQL]]:
        """构建SQL的所有中间状态.
        
        Returns:
            (clauses, Dict[SQLStage, IntermediateSQL])
        """
        self.parser = SQLParser(sql)
        clauses = self.parser.extract_clauses()
        results: Dict[SQLStage, IntermediateSQL] = {}
        
        if clauses.is_compound:
            logger.debug(f"Compound SQL detected, skipping intermediate extraction")
            return clauses, results
        
        from_base = self._build_from_base(clauses)
        if not from_base:
            return clauses, results
        
        # a. FROM_JOIN - 总是存在
        from_join_sql = f"SELECT * {from_base}"
        results[SQLStage.FROM_JOIN] = IntermediateSQL(
            stage=SQLStage.FROM_JOIN,
            sql=from_join_sql,
            clauses=clauses,
            description="a. 选表阶段"
        )
        
        # b. WHERE
        if clauses.where:
            where_sql = f"SELECT * {from_base} WHERE {clauses.where}"
            results[SQLStage.WHERE] = IntermediateSQL(
                stage=SQLStage.WHERE,
                sql=where_sql,
                clauses=clauses,
                description="b. 行过滤阶段"
            )
        
        # c. GROUP_BY
        if clauses.group_by:
            group_sql = f"SELECT {clauses.group_by} {from_base}"
            if clauses.where:
                group_sql += f" WHERE {clauses.where}"
            group_sql += f" GROUP BY {clauses.group_by}"
            results[SQLStage.GROUP_BY] = IntermediateSQL(
                stage=SQLStage.GROUP_BY,
                sql=group_sql,
                clauses=clauses,
                description="c. 分组阶段"
            )
        
        # d. HAVING
        if clauses.having and clauses.group_by:
            having_sql = f"SELECT {clauses.group_by} {from_base}"
            if clauses.where:
                having_sql += f" WHERE {clauses.where}"
            having_sql += f" GROUP BY {clauses.group_by} HAVING {clauses.having}"
            results[SQLStage.HAVING] = IntermediateSQL(
                stage=SQLStage.HAVING,
                sql=having_sql,
                clauses=clauses,
                description="d. 组过滤阶段"
            )
        
        # e. SELECT - 总是存在
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
            clauses=clauses,
            description="e. 列投影阶段"
        )
        
        # f. ORDER_BY
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
                clauses=clauses,
                description="f. 排序阶段"
            )
        
        return clauses, results
    
    def _build_from_base(self, clauses: SQLClauses) -> str:
        """构建FROM基础部分."""
        if not clauses.from_clause:
            return ""
        
        base = f"FROM {clauses.from_clause}"
        
        if clauses.joins:
            base += " " + " ".join(clauses.joins)
        
        return base


# ============================================================================
# 第六部分：快速分层奖励计算器
# ============================================================================

class FastLayeredRewardCalculator:
    """快速分层奖励计算器.
    
    奖励计算策略：
    1. 最终结果匹配 → 1.0
    2. UNION/INTERSECT/EXCEPT → 直接判断最终结果，不匹配返回0
    3. 从前往后对比(a→g)，返回最后通过阶段的奖励（0.40→0.70）
    4. 全部不匹配 → 0
    """
    
    def __init__(self, db_path: str):
        """初始化.
        
        Args:
            db_path: SQLite数据库路径
        """
        self.db_path = db_path
        self.builder = IntermediateSQLBuilder()
        self._binary_reward = get_binary_reward_func()
        self._simple_binary_reward = get_simple_binary_reward_func()
    
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
            stats.details.append({
                "type": "binary_reward",
                "context": context,
                "result": 0.0,
                "time_sec": round(elapsed, 6),
                "timeout": True,
            })
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
        stats.details.append({
            "type": "binary_reward",
            "context": context,
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
        stats.details.append({
            "type": "simple_binary_reward",
            "context": context,
            "result": result,
            "time_sec": round(elapsed, 6),
        })
        
        return result

    @staticmethod
    def _build_from_base_static(clauses: SQLClauses) -> str:
        """构建FROM基础部分."""
        if not clauses.from_clause:
            return ""
        base = f"FROM {clauses.from_clause}"
        if clauses.joins:
            base += " " + " ".join(clauses.joins)
        return base

    @staticmethod
    def _remap_clause_aliases(clause: str, pred_clauses: SQLClauses, gold_clauses: SQLClauses) -> str:
        """将子句中 pred 的别名替换为 gold 的别名体系."""
        pred_alias_map = _extract_alias_mapping(pred_clauses)
        gold_alias_map = _extract_alias_mapping(gold_clauses)
        
        gold_table_to_alias: Dict[str, str] = {}
        for alias, table in gold_alias_map.items():
            if table not in gold_table_to_alias:
                gold_table_to_alias[table] = alias
            else:
                return clause  # 自连接，无法自动映射
        
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
                flags=re.I
            )
        return result

    def _build_aligned_intermediate_sql(
        self,
        stage: SQLStage,
        pred_clauses: SQLClauses,
        gold_clauses: SQLClauses,
    ) -> Optional[Tuple[str, str]]:
        """构建FROM对齐的中间SQL（WHERE/HAVING阶段）."""
        gold_from_base = self._build_from_base_static(gold_clauses)
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

    def _compare_stage_structural(
        self,
        stage: SQLStage,
        pred_clauses: SQLClauses,
        gold_clauses: SQLClauses,
        stats: PerformanceStats,
    ) -> Tuple[bool, Dict[str, Any]]:
        """结构化比较某个阶段."""
        start = time.perf_counter()
        
        if stage == SQLStage.FROM_JOIN:
            is_equal, details = compare_from_join_structure(pred_clauses, gold_clauses)
        elif stage == SQLStage.GROUP_BY:
            is_equal, details = compare_group_by(pred_clauses, gold_clauses)
        elif stage == SQLStage.SELECT:
            is_equal, details = compare_select_columns(pred_clauses, gold_clauses)
        elif stage == SQLStage.ORDER_BY:
            is_equal, details = compare_order_by(pred_clauses, gold_clauses)
        else:
            raise ValueError(f"Unknown structural compare stage: {stage}")
        
        elapsed = time.perf_counter() - start
        
        stats.structural_compare_count += 1
        stats.structural_compare_time += elapsed
        stats.details.append({
            "type": "structural_compare",
            "stage": stage.value,
            "is_equal": is_equal,
            "time_sec": round(elapsed, 6),
        })
        
        return is_equal, details
    
    def _stage_exists(self, stage: SQLStage, clauses: SQLClauses) -> bool:
        """检查某个阶段是否存在."""
        if stage == SQLStage.FROM_JOIN:
            return bool(clauses.from_clause)
        elif stage == SQLStage.WHERE:
            return bool(clauses.where)
        elif stage == SQLStage.GROUP_BY:
            return bool(clauses.group_by)
        elif stage == SQLStage.HAVING:
            return bool(clauses.having)
        elif stage == SQLStage.SELECT:
            # SELECT 总是存在
            return True
        elif stage == SQLStage.ORDER_BY:
            return bool(clauses.order_by)
        return False
    
    def calculate_reward(
        self,
        pred_sql: str,
        gold_sql: str,
    ) -> Tuple[float, Dict[str, Any]]:
        """计算快速分层复合奖励.
        
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
            "stage_results": {},
            "last_passed_stage": None,
            "reward_type": "",
            "reward": 0.0,
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
        
        # Step 3: 从前往后比较中间状态 (a → g)
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
                    last_passed_reward = STAGE_REWARD_MAP[stage]
                    details["last_passed_stage"] = stage.value
                else:
                    stage_result["reason"] = "结构比较不通过"
                    details["stage_results"][stage.value] = stage_result
                    details["reward_type"] = f"structural_mismatch_at_{stage.value}"
                    return finalize_and_return(last_passed_reward)
            
            elif stage in EXECUTE_SQL_STAGES:
                # 执行SQL比较（FROM对齐 + simple_binary_reward）
                aligned_sql = self._build_aligned_intermediate_sql(
                    stage, pred_clauses, gold_clauses
                )
                
                if aligned_sql:
                    pred_aligned, gold_aligned = aligned_sql
                    try:
                        match = self._timed_simple_binary_reward(
                            pred_aligned,
                            gold_aligned,
                            stats,
                            context=f"stage_{stage.value}"
                        )
                        stage_result["match"] = (match == 1.0)
                        
                        if match == 1.0:
                            stage_result["reason"] = "SQL执行比较通过（FROM对齐）"
                            last_passed_reward = STAGE_REWARD_MAP[stage]
                            details["last_passed_stage"] = stage.value
                        else:
                            stage_result["reason"] = "SQL执行比较不通过"
                            details["stage_results"][stage.value] = stage_result
                            details["reward_type"] = f"sql_mismatch_at_{stage.value}"
                            return finalize_and_return(last_passed_reward)
                            
                    except Exception as e:
                        logger.debug(f"Stage {stage.value} comparison failed: {e}")
                        stage_result["match"] = False
                        stage_result["reason"] = f"比较失败: {str(e)}"
                        details["stage_results"][stage.value] = stage_result
                        details["reward_type"] = f"error_at_{stage.value}"
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
# 第七部分：对外接口
# ============================================================================

def composite_reward_fast(
    pred_sql: str,
    gold_sql: str,
    db_path: str,
    raise_on_error: bool = False,
) -> float:
    """计算快速分层复合奖励（简化接口）.
    
    Args:
        pred_sql: 预测SQL
        gold_sql: 标准SQL
        db_path: 数据库路径
        raise_on_error: 是否抛出异常
    
    Returns:
        奖励值
    """
    try:
        calculator = FastLayeredRewardCalculator(db_path)
        reward, _ = calculator.calculate_reward(pred_sql, gold_sql)
        return reward
    except Exception as e:
        if raise_on_error:
            raise
        logger.exception(f"Fast composite reward calculation failed: {e}")
        return 0.0


def composite_reward_fast_with_details(
    pred_sql: str,
    gold_sql: str,
    db_path: str,
    raise_on_error: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    """计算快速分层复合奖励并返回详细信息.
    
    Args:
        pred_sql: 预测SQL
        gold_sql: 标准SQL  
        db_path: 数据库路径
        raise_on_error: 是否抛出异常
    
    Returns:
        (reward, details)
    """
    try:
        calculator = FastLayeredRewardCalculator(db_path)
        return calculator.calculate_reward(pred_sql, gold_sql)
    except Exception as e:
        if raise_on_error:
            raise
        logger.exception(f"Fast composite reward calculation failed: {e}")
        return 0.0, {"error": str(e)}


# ============================================================================
# 第八部分：调试和测试
# ============================================================================

def debug_fast_reward_calculation(pred_sql: str, gold_sql: str, db_path: str) -> None:
    """调试：打印快速奖励计算的详细过程."""
    reward, details = composite_reward_fast_with_details(pred_sql, gold_sql, db_path)
    
    print(f"\n{'='*60}")
    print(f"Fast Reward Calculation Debug")
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
    
    # 打印阶段结果
    stage_results = details.get("stage_results", {})
    if stage_results:
        print(f"\nStage Results:")
        for stage_name, result in stage_results.items():
            match = result.get("match")
            reason = result.get("reason", "")
            match_str = "✓" if match else ("✗" if match is False else "?")
            print(f"  [{stage_name}] {match_str} - {reason}")
    
    # 打印性能统计
    perf_stats = details.get("performance_stats", {})
    if perf_stats:
        print(f"\n{'='*60}")
        print(f"Performance Statistics")
        print(f"{'='*60}")
        print(f"binary_reward Calls: {perf_stats.get('binary_reward_call_count', 0)}")
        print(f"Structural Compares: {perf_stats.get('structural_compare_count', 0)}")
        print(f"Clause Extraction Count: {perf_stats.get('clause_extraction_count', 0)}")
        print(f"{'='*60}")
        print(f"Clause Extraction Time: {perf_stats.get('clause_extraction_time_sec', 0)*1000:.2f} ms")
        print(f"Structural Compare Time: {perf_stats.get('structural_compare_time_sec', 0)*1000:.2f} ms")
        print(f"binary_reward Time: {perf_stats.get('binary_reward_time_sec', 0)*1000:.2f} ms")
        print(f"Total Time: {perf_stats.get('total_time_sec', 0)*1000:.2f} ms")
        print(f"{'='*60}")


if __name__ == "__main__":
    # 测试示例
    test_pred = """
    SELECT T1.name, COUNT(T2.id) 
    FROM students T1 
    JOIN enrollments T2 ON T1.id = T2.student_id 
    WHERE T1.age > 18 
    GROUP BY T1.name 
    HAVING COUNT(T2.id) > 2
    ORDER BY COUNT(T2.id) DESC
    """
    
    test_gold = """
    SELECT T1.name, COUNT(T2.id) 
    FROM students AS T1 
    INNER JOIN enrollments AS T2 ON T2.student_id = T1.id 
    WHERE T1.age > 18 
    GROUP BY T1.name 
    HAVING COUNT(T2.id) > 2
    ORDER BY COUNT(T2.id) DESC
    """
    
    print("Testing FROM/JOIN structure comparison:")
    parser1 = SQLParser(test_pred)
    parser2 = SQLParser(test_gold)
    clauses1 = parser1.extract_clauses()
    clauses2 = parser2.extract_clauses()
    
    is_equal, details = compare_from_join_structure(clauses1, clauses2)
    print(f"FROM/JOIN equal: {is_equal}")
    print(f"Tables1: {details['tables1']}")
    print(f"Tables2: {details['tables2']}")
    print(f"Conditions1: {details['conditions1']}")
    print(f"Conditions2: {details['conditions2']}")
