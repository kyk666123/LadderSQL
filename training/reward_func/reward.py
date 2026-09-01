import os
import logging
import sqlite3
from collections import Counter
from typing import List, Tuple, Any
from reward_func.spider_eval.exec_eval import eval_exec_match, my_eval_exec_match, eval_exec_match_continuous


logger = logging.getLogger(__name__)


def binary_reward_bird(query: str, ground_truth: str, database: str, raise_on_error: bool = True) -> float:
    """BIRD官方EX评估逻辑的二元奖励函数.
    
    与Spider的binary_reward区别：
    - 列顺序：不忽略（直接比较tuple，列顺序必须严格一致）
    - 行顺序：无条件忽略（使用set比较，无论是否有ORDER BY）
    
    参考: /root/DAMO-ConvAI/bird/llm/src/evaluation.py
    """
    try:
        if not query or not query.strip():
            logger.warning("Empty query provided, returning 0.0")
            return 0.0

        database = os.path.abspath(database)
        if not os.path.exists(database):
            raise FileNotFoundError(f"Database file {database} does not exist.")

        conn = sqlite3.connect(database)
        conn.text_factory = lambda b: b.decode(errors="ignore")
        cursor = conn.cursor()

        cursor.execute(query)
        predicted_res = cursor.fetchall()

        cursor.execute(ground_truth)
        ground_truth_res = cursor.fetchall()

        cursor.close()
        conn.close()

        # BIRD官方评估：set比较，忽略行顺序，不忽略列顺序
        if set(predicted_res) == set(ground_truth_res):
            return 1.0
        else:
            return 0.0

    except Exception as e:
        if raise_on_error:
            raise
        else:
            logger.exception(f"Error evaluating query (bird): {e}")
            return 0.0


def binary_reward(query: str, ground_truth: str, database: str, raise_on_error: bool = True) -> float:
    try:
        if not query or not query.strip():
            logger.warning("Empty query provided, returning 0.0")
            return 0.0

        database = os.path.abspath(database)
        if not os.path.exists(database):
            raise FileNotFoundError(f"Database file {database} does not exist.")

        # Parameters following the default setting
        exec_score = eval_exec_match(
            db=database,
            p_str=query,
            g_str=ground_truth,
            plug_value=False,
            keep_distinct=False,
            progress_bar_for_each_datapoint=False,
        )
        if exec_score == 1:
            return 1.0
        else:
            return 0.0
    except Exception as e:
        if raise_on_error:
            raise
        else:
            logger.exception(f"Error evaluating query: {e}")
            return 0.0


def _execute_sql(database: str, query: str, timeout: int = 15) -> Tuple[bool, List[Tuple]]:
    """执行SQL并返回结果（带查询执行超时保护）.

    使用 threading.Timer + conn.interrupt() 确保长查询可被中断。
    
    Args:
        database: 数据库路径
        query: SQL查询
        timeout: 超时时间(秒)
    
    Returns:
        (success, results) 元组
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
                logger.debug(f"SQL execution failed: {e}")
            return False, []
        except Exception as e:
            timer.cancel()
            conn.close()
            logger.debug(f"SQL execution failed: {e}")
            return False, []
    except Exception as e:
        logger.debug(f"SQL connection failed: {e}")
        return False, []


def _simple_result_eq(result1: List[Tuple], result2: List[Tuple]) -> bool:
    """简化版结果比较 - 不做列排列枚举.
    
    直接用 multiset 比较，忽略行顺序但保留列顺序。
    这比完整版 result_eq 快很多，因为不需要枚举 n! 种列排列。
    """
    if len(result1) == 0 and len(result2) == 0:
        return True
    
    if len(result1) != len(result2):
        return False
    
    if len(result1) > 0 and len(result2) > 0:
        if len(result1[0]) != len(result2[0]):
            return False
    
    # 使用 Counter 做 multiset 比较（忽略行顺序）
    return Counter(result1) == Counter(result2)


def simple_binary_reward(query: str, ground_truth: str, database: str, raise_on_error: bool = False) -> float:
    """简化版二元奖励 - 不做列排列枚举，速度更快.
    
    专用于中间阶段比较，不需要考虑列顺序不同的情况。
    因为中间SQL的结构已经通过AST相似度过滤。
    
    Args:
        query: 预测SQL
        ground_truth: 标准SQL
        database: 数据库路径
        raise_on_error: 是否抛出异常
    
    Returns:
        1.0 如果结果匹配，否则 0.0
    """
    try:
        database = os.path.abspath(database)
        if not os.path.exists(database):
            if raise_on_error:
                raise FileNotFoundError(f"Database file {database} does not exist.")
            return 0.0
        
        # 执行两个 SQL
        g_success, g_results = _execute_sql(database, ground_truth)
        if not g_success:
            logger.debug(f"Gold SQL execution failed: {ground_truth}")
            return 0.0
        
        p_success, p_results = _execute_sql(database, query)
        if not p_success:
            logger.debug(f"Pred SQL execution failed: {query}")
            return 0.0
        
        # 简化版比较
        if _simple_result_eq(g_results, p_results):
            return 1.0
        else:
            return 0.0
            
    except Exception as e:
        if raise_on_error:
            raise
        else:
            logger.debug(f"Error in simple_binary_reward: {e}")
            return 0.0


def multiple_reward(query: str, ground_truth: str, database: str, raise_on_error: bool = True) -> float:
    try:
        database = os.path.abspath(database)
        if not os.path.exists(database):
            raise FileNotFoundError(f"Database file {database} does not exist.")

        # Parameters following the default setting
        exec_score = my_eval_exec_match(
            db=database,
            p_str=query,
            g_str=ground_truth,
            plug_value=False,
            keep_distinct=False,
            progress_bar_for_each_datapoint=False,
        )
        return float(exec_score)
    except Exception as e:
        if raise_on_error:
            raise
        else:
            logger.exception(f"Error evaluating query: {e}")
            return 0.0
        
        

SQL_TIMEOUT = 120
DF_CUTOFF = 5       # 允许的最大行数差异
PRE_FLOAT = 0.005   # 浮点数比较容差

def continuous_execution_reward(query: str, ground_truth: str, database: str, raise_on_error: bool = True) -> float:
    """
    修改版奖励函数：
    不再返回 0 或 1，而是返回基于执行结果相似度的连续奖励值 (0.0 ~ 1.0+)。
    如果完全匹配返回 1.0，部分匹配返回比例值，执行成功但结果不匹配返回 0.5 (安慰奖)。
    """
    try:
        database = os.path.abspath(database)
        if not os.path.exists(database):
            raise FileNotFoundError(f"Database file {database} does not exist.")

        # 调用修改后的评估函数，返回连续得分
        exec_score = eval_exec_match_continuous(
            db=database,
            p_str=query,
            g_str=ground_truth,
            plug_value=False,
            keep_distinct=False,
            progress_bar_for_each_datapoint=False,
        )
        
        # 这里的 exec_score 已经是 0.0 到 1.0 之间的浮点数 (或者略高于1如果逻辑调整过，但通常归一化)
        # 如果需要严格限制在 0-1，可以加 max(0.0, min(1.0, exec_score))
        return exec_score

    except Exception as e:
        if raise_on_error:
            raise
        else:
            logger.exception(f"Error evaluating query: {e}")
            return 0.0



if __name__ == "__main__":
    database_file = "/path/to/dataset/spider/test_database/e_commerce/e_commerce.sqlite"

    query = "SELECT T2.invoice_status_code FROM Orders AS T1 INNER JOIN Invoices AS T2 ON T1.order_id = T2.invoice_number WHERE T1.order_status_code != 'Shipped'"
    ground_truth = "SELECT invoice_status_code FROM Invoices WHERE invoice_number NOT IN ( SELECT invoice_number FROM Shipments )"

    print(binary_reward(query, ground_truth, database_file))