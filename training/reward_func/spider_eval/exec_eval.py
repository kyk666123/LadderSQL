# Copyright (c) Microsoft. All rights reserved.

# type: ignore
# The evaluation code is from https://github.com/taoyds/test-suite-sql-eval

import asyncio
import pandas as pd
import os
import pickle as pkl
import random
import re
import sqlite3
import subprocess
import threading
import time
from collections import defaultdict
from itertools import chain, product
from typing import Any, List, Set, Tuple
from collections import Counter

import tqdm

from .async_utils import run_sync_ephemeral
from .parse import get_all_preds_for_execution, remove_distinct

threadLock = threading.Lock()
TIMEOUT = 60
EXEC_TMP_DIR = "/tmp/"


def permute_tuple(element: Tuple, perm: Tuple) -> Tuple:
    assert len(element) == len(perm)
    return tuple([element[i] for i in perm])


def unorder_row(row: Tuple) -> Tuple:
    return tuple(sorted(row, key=lambda x: str(x) + str(type(x))))


# unorder each row in the table
# [result_1 and result_2 has the same bag of unordered row]
# is a necessary condition of
# [result_1 and result_2 are equivalent in denotation]
def quick_rej(result1: List[Tuple], result2: List[Tuple], order_matters: bool) -> bool:
    s1 = [unorder_row(row) for row in result1]
    s2 = [unorder_row(row) for row in result2]
    if order_matters:
        return s1 == s2
    else:
        return set(s1) == set(s2)


# return whether two bag of relations are equivalent
def multiset_eq(l1: List, l2: List) -> bool:
    if len(l1) != len(l2):
        return False
    d = defaultdict(int)
    for e in l1:
        d[e] = d[e] + 1
    for e in l2:
        d[e] = d[e] - 1
        if d[e] < 0:
            return False
    return True


def get_constraint_permutation(tab1_sets_by_columns: List[Set], result2: List[Tuple]):
    num_cols = len(result2[0])
    perm_constraints = [{i for i in range(num_cols)} for _ in range(num_cols)]
    if num_cols <= 3:
        return product(*perm_constraints)

    # we sample 20 rows and constrain the space of permutations
    for _ in range(20):
        random_tab2_row = random.choice(result2)

        for tab1_col in range(num_cols):
            for tab2_col in set(perm_constraints[tab1_col]):
                if random_tab2_row[tab2_col] not in tab1_sets_by_columns[tab1_col]:
                    perm_constraints[tab1_col].remove(tab2_col)
    return product(*perm_constraints)


# check whether two denotations are correct
def result_eq(result1: List[Tuple], result2: List[Tuple], order_matters: bool) -> bool:
    if len(result1) == 0 and len(result2) == 0:
        return True

    # if length is not the same, then they are definitely different bag of rows
    if len(result1) != len(result2):
        return False

    num_cols = len(result1[0])

    # if the results do not have the same number of columns, they are different
    if len(result2[0]) != num_cols:
        return False

    # unorder each row and compare whether the denotation is the same
    # this can already find most pair of denotations that are different
    if not quick_rej(result1, result2, order_matters):
        return False

    # the rest of the problem is in fact more complicated than one might think
    # we want to find a permutation of column order and a permutation of row order,
    # s.t. result_1 is the same as result_2
    # we return true if we can find such column & row permutations
    # and false if we cannot
    tab1_sets_by_columns = [{row[i] for row in result1} for i in range(num_cols)]

    # on a high level, we enumerate all possible column permutations that might make result_1 == result_2
    # we decrease the size of the column permutation space by the function get_constraint_permutation
    # if one of the permutation make result_1, result_2 equivalent, then they are equivalent
    for perm in get_constraint_permutation(tab1_sets_by_columns, result2):
        if len(perm) != len(set(perm)):
            continue
        if num_cols == 1:
            result2_perm = result2
        else:
            result2_perm = [permute_tuple(element, perm) for element in result2]
        if order_matters:
            if result1 == result2_perm:
                return True
        else:
            # in fact the first condition must hold if the second condition holds
            # but the first is way more efficient implementation-wise
            # and we use it to quickly reject impossible candidates
            if set(result1) == set(result2_perm) and multiset_eq(result1, result2_perm):
                return True
    return False


def replace_cur_year(query: str) -> str:
    return re.sub("YEAR\s*\(\s*CURDATE\s*\(\s*\)\s*\)\s*", "2020", query, flags=re.IGNORECASE)


# get the database cursor for a sqlite database path
def get_cursor_from_path(sqlite_path: str):
    try:
        if not os.path.exists(sqlite_path):
            print("Opening a new connection %s" % sqlite_path)
        connection = sqlite3.connect(sqlite_path)
    except Exception as e:
        print(sqlite_path)
        raise e
    connection.text_factory = lambda b: b.decode(errors="ignore")
    cursor = connection.cursor()
    return cursor


async def exec_on_db_(sqlite_path: str, query: str) -> Tuple[str, Any]:
    query = replace_cur_year(query)
    cursor = get_cursor_from_path(sqlite_path)
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        cursor.close()
        cursor.connection.close()
        return "result", result
    except Exception as e:
        cursor.close()
        cursor.connection.close()
        return "exception", e


async def exec_on_db(sqlite_path: str, query: str, process_id: str = "", timeout: int = TIMEOUT) -> Tuple[str, Any]:
    try:
        return await asyncio.wait_for(exec_on_db_(sqlite_path, query), timeout)
    except asyncio.TimeoutError:
        return ("exception", TimeoutError)
    except Exception as e:
        return ("exception", e)


# postprocess the model predictions to avoid execution errors
# e.g. removing spaces between ">" and "="
def postprocess(query: str) -> str:
    query = query.replace("> =", ">=").replace("< =", "<=").replace("! =", "!=")
    return query


# approximate whether p_str and g_str are semantically equivalent
# db is the database path
# we are going to evaluate whether they are equivalent in all the databases
# that are in the same directory as db
# 0 if denotationally equivalent
# 1 otherwise
# the meaning of each auxillary argument can be seen in the parser definition in evaluation.py
def eval_exec_match(
    db: str, p_str: str, g_str: str, plug_value: bool, keep_distinct: bool, progress_bar_for_each_datapoint: bool
) -> int:
    # post-process the prediction.
    # e.g. removing spaces between ">" and "="
    p_str, g_str = postprocess(p_str), postprocess(g_str)
    if not keep_distinct:
        p_str = remove_distinct(p_str)
        g_str = remove_distinct(g_str)

    # we decide whether two denotations are equivalent based on "bag semantics"
    # https://courses.cs.washington.edu/courses/cse444/10sp/lectures/lecture16.pdf
    # if there is order by in query, then we assume order of the rows matter
    # order by might also be used to find the max/min instead of sorting,
    # but in that case the result mostly only contains one row and hence order_matters does not make a difference
    order_matters = "order by" in g_str.lower()

    # find all databases in the same directory
    db_dir = os.path.dirname(db)
    db_paths = [os.path.join(db_dir, basename) for basename in os.listdir(db_dir) if basename.endswith(".sqlite")]

    preds = [p_str]
    # if plug in value (i.e. we do not consider value prediction correctness)
    # enumerate all ways to plug in values in the gold query to the model predictions
    # otherwise, we only evaluate the predicted query with its own value prediction
    if plug_value:
        _, preds = get_all_preds_for_execution(g_str, p_str)
        # we did not add this line in our EMNLP work
        # this reduces "false negatives" when value is substituted
        preds = chain([p_str], preds)

    for pred in preds:

        pred_passes = 1
        # compare the gold and predicted denotations on each database in the directory
        # wrap with progress bar if required
        if progress_bar_for_each_datapoint:
            ranger = tqdm.tqdm(db_paths)
        else:
            ranger = db_paths

        for db_path in ranger:
            g_flag, g_denotation = run_sync_ephemeral(exec_on_db(db_path, g_str))
            p_flag, p_denotation = run_sync_ephemeral(exec_on_db(db_path, pred))

            # we should expect the gold to be succesfully executed on the database
            assert g_flag != "exception", "gold query %s has error on database file %s" % (g_str, db_path)

            # wrong if execution fails
            if p_flag == "exception":
                pred_passes = 0

            # if denotations are not equivalent, the prediction must be wrong
            elif not result_eq(g_denotation, p_denotation, order_matters=order_matters):
                pred_passes = 0
            if pred_passes == 0:
                break

        # the model prediction has the same denotation as the gold for all databases
        if pred_passes == 1:
            return 1

    # none of the predictions passed
    return 0



def eval_exec_match(
    db: str, p_str: str, g_str: str, plug_value: bool, keep_distinct: bool, progress_bar_for_each_datapoint: bool
) -> int:
    # post-process the prediction.
    # e.g. removing spaces between ">" and "="
    p_str, g_str = postprocess(p_str), postprocess(g_str)
    if not keep_distinct:
        p_str = remove_distinct(p_str)
        g_str = remove_distinct(g_str)

    # we decide whether two denotations are equivalent based on "bag semantics"
    # https://courses.cs.washington.edu/courses/cse444/10sp/lectures/lecture16.pdf
    # if there is order by in query, then we assume order of the rows matter
    # order by might also be used to find the max/min instead of sorting,
    # but in that case the result mostly only contains one row and hence order_matters does not make a difference
    order_matters = "order by" in g_str.lower()

    # find all databases in the same directory
    db_dir = os.path.dirname(db)
    db_paths = [os.path.join(db_dir, basename) for basename in os.listdir(db_dir) if basename.endswith(".sqlite")]

    preds = [p_str]
    # if plug in value (i.e. we do not consider value prediction correctness)
    # enumerate all ways to plug in values in the gold query to the model predictions
    # otherwise, we only evaluate the predicted query with its own value prediction
    if plug_value:
        _, preds = get_all_preds_for_execution(g_str, p_str)
        # we did not add this line in our EMNLP work
        # this reduces "false negatives" when value is substituted
        preds = chain([p_str], preds)

    for pred in preds:

        pred_passes = 1
        # compare the gold and predicted denotations on each database in the directory
        # wrap with progress bar if required
        if progress_bar_for_each_datapoint:
            ranger = tqdm.tqdm(db_paths)
        else:
            ranger = db_paths

        for db_path in ranger:
            g_flag, g_denotation = run_sync_ephemeral(exec_on_db(db_path, g_str))
            p_flag, p_denotation = run_sync_ephemeral(exec_on_db(db_path, pred))

            # we should expect the gold to be succesfully executed on the database
            assert g_flag != "exception", "gold query %s has error on database file %s" % (g_str, db_path)

            # wrong if execution fails
            if p_flag == "exception":
                pred_passes = 0

            # if denotations are not equivalent, the prediction must be wrong
            elif not result_eq(g_denotation, p_denotation, order_matters=order_matters):
                pred_passes = 0
            if pred_passes == 0:
                break

        # the model prediction has the same denotation as the gold for all databases
        if pred_passes == 1:
            return 1

    # none of the predictions passed
    return 0

# 用JACCARD相似度来判断
def my_eval_exec_match(
    db: str, p_str: str, g_str: str, plug_value: bool, keep_distinct: bool, progress_bar_for_each_datapoint: bool
) -> float:
    # post-process the prediction.
    # e.g. removing spaces between ">" and "="
    p_str, g_str = postprocess(p_str), postprocess(g_str)
    if not keep_distinct:
        p_str = remove_distinct(p_str)
        g_str = remove_distinct(g_str)

    # we decide whether two denotations are equivalent based on "bag semantics"
    # https://courses.cs.washington.edu/courses/cse444/10sp/lectures/lecture16.pdf
    # if there is order by in query, then we assume order of the rows matter
    # order by might also be used to find the max/min instead of sorting,
    # but in that case the result mostly only contains one row and hence order_matters does not make a difference
    order_matters = "order by" in g_str.lower()

    # find all databases in the same directory
    db_dir = os.path.dirname(db)
    db_paths = [os.path.join(db_dir, basename) for basename in os.listdir(db_dir) if basename.endswith(".sqlite")]

    preds = [p_str]
    # if plug in value (i.e. we do not consider value prediction correctness)
    # enumerate all ways to plug in values in the gold query to the model predictions
    # otherwise, we only evaluate the predicted query with its own value prediction
    if plug_value:
        _, preds = get_all_preds_for_execution(g_str, p_str)
        # we did not add this line in our EMNLP work
        # this reduces "false negatives" when value is substituted
        preds = chain([p_str], preds)

    pred_reward = 0.0

    for pred in preds:

        # compare the gold and predicted denotations on each database in the directory
        # wrap with progress bar if required
        if progress_bar_for_each_datapoint:
            ranger = tqdm.tqdm(db_paths)
        else:
            ranger = db_paths

        for db_path in ranger:
            g_flag, g_denotation = run_sync_ephemeral(exec_on_db(db_path, g_str))
            p_flag, p_denotation = run_sync_ephemeral(exec_on_db(db_path, pred))

            # we should expect the gold to be succesfully executed on the database
            assert g_flag != "exception", "gold query %s has error on database file %s" % (g_str, db_path)

            # wrong if execution fails
            if p_flag == "exception":
                break

            # if denotations are not equivalent, the prediction must be wrong
            pred_reward = my_result_eq(g_denotation, p_denotation, order_matters=order_matters)
 
    return pred_reward

def multiset_jaccard_rows(r1: List[Tuple], r2: List[Tuple]) -> float:
    """
    多重集 Jaccard：把“行 tuple”当元素，考虑重复次数。
    返回值范围 [0, 1]，两边都空定义为 1。
    """
    c1, c2 = Counter(r1), Counter(r2)

    # Counter 支持 &（min）和 |（max）
    inter = sum((c1 & c2).values())   # Σ min
    uni = sum((c1 | c2).values())     # Σ max
    return inter / uni

# def position_match_ratio(r1: List[Tuple], r2: List[Tuple]) -> float:
#     """
#     order_matters=True 且已保证 len(r1)==len(r2) 的情况下：
#     返回逐位置完全相等的行比例。
#     """
#     n = len(r1)
#     same = sum(1 for i in range(n) if r1[i] == r2[i])
#     return same / n

def position_match_ratio(r1: List[Tuple], r2: List[Tuple]) -> float:
    min_len = min(len(r1), len(r2))
    hits = sum(1 for i in range(min_len) if r1[i] == r2[i])

    precision = hits / len(r2)
    recall = hits / len(r1)

    return (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

def my_result_eq(
    result1: List[Tuple],
    result2: List[Tuple],
    order_matters: bool,
) -> float:
    """
    返回值范围 [0, 1]：
    - 只要存在某个 perm 完全匹配 Spider 等价定义 => 1.0
    - 否则返回 best-perm 的相似度（<1）
    """
    if len(result1) == 0 and len(result2) == 0:
        return 1.0

    # if len(result1) != len(result2):  # 放开行数
    #     return 0.0

    if not result1 or not result2:
        return 0.0

    num_cols = len(result1[0])

    if len(result2[0]) != num_cols:
        return 0.0

    # if not quick_rej(result1, result2, order_matters):
    #     return 0.0

    tab1_sets_by_columns = [{row[i] for row in result1} for i in range(num_cols)]

    best_sim = 0.0

    for perm in get_constraint_permutation(tab1_sets_by_columns, result2):
        if len(perm) != len(set(perm)):
            continue

        if num_cols == 1:
            result2_perm = result2
        else:
            result2_perm = [permute_tuple(element, perm) for element in result2]

        if order_matters:
            if result1 == result2_perm:
                return 1.0
            sim = position_match_ratio(result1, result2_perm)
        else:
            if set(result1) == set(result2_perm) and multiset_eq(result1, result2_perm):
                return 1.0
            sim = multiset_jaccard_rows(result1, result2_perm)

        if sim > best_sim:
            best_sim = sim

    return best_sim


# 失败的尝试，用列数和行数来匹配
# def my_eval_exec_match(
#     db: str,
#     p_str: str,
#     g_str: str,
#     plug_value: bool,
#     keep_distinct: bool,
# ) -> float:
#     # 1. 预处理 SQL
#     p_str, g_str = postprocess(p_str), postprocess(g_str)
#     if not keep_distinct:
#         p_str = remove_distinct(p_str)
#         g_str = remove_distinct(g_str)

#     # 2. 看 gold 里有没有 ORDER BY，用于 exact match 的顺序判断
#     order_matters = "order by" in g_str.lower()

#     # 3. 找到同目录下所有 sqlite 库
#     db_dir = os.path.dirname(db)
#     db_paths = [
#         os.path.join(db_dir, basename)
#         for basename in os.listdir(db_dir)
#         if ".sqlite" in basename
#     ]
#     total_db_count = len(db_paths)
#     if total_db_count == 0:
#         return 0.0

#     preds = [p_str]
#     if plug_value:
#         _, preds = get_all_preds_for_execution(g_str, p_str)
#         preds = chain([p_str], preds)

#     best_reward = 0.0

#     # 一般你 plug_value=False，只会跑一次，但这里写成对所有 preds 取最大 reward
#     for pred in preds:
#         exact_match_count = 0
#         ignore_order_match_count = 0
#         struct_sims = []

#         for db_path in db_paths:
#             g_flag, g_denotation = run_sync_ephemeral(exec_on_db(db_path, g_str))
#             p_flag, p_denotation = run_sync_ephemeral(exec_on_db(db_path, pred))

#             if g_flag == "exception":
#                 # gold 出错说明环境有问题，直接抛异常
#                 raise Exception(f"Gold query {g_str} has error on database file {db_path}")

#             if p_flag == "exception":
#                 # 预测执行失败：这个库上既不是 exact、也不是 ignore_order 匹配
#                 # 结构相似度也算 0（因为根本没结果）
#                 struct_sims.append(0.0)
#                 continue

#             # ------ 层 1：完全匹配 ------
#             if result_eq(g_denotation, p_denotation, order_matters=True):
#                 exact_match_count += 1
#                 # 对于这个库，结构肯定也是完全一致，所以结构相似度记 1
#                 struct_sims.append(1.0)
#                 continue

#             # ------ 层 2：忽略顺序匹配 ------
#             if result_eq(g_denotation, p_denotation, order_matters=False):
#                 ignore_order_match_count += 1
#                 # 行列数一定一样，但我们还是按结构算一遍（结构相似度会接近 1）
#                 struct_sim = _compute_struct_similarity(g_denotation, p_denotation)
#                 struct_sims.append(struct_sim)
#                 continue

#             # ------ 层 3：只比行列结构 ------
#             struct_sim = _compute_struct_similarity(g_denotation, p_denotation)
#             struct_sims.append(struct_sim)

#         # 计算这一条 pred 的 reward
#         if exact_match_count == total_db_count:
#             reward = 1.0
#         elif ignore_order_match_count > 0:
#             # 按忽略顺序匹配的比例给 0.7 封顶
#             ratio = ignore_order_match_count / total_db_count
#             reward = 0.7 * ratio
#         else:
#             # 完全没有内容匹配，只看结构
#             if struct_sims:
#                 avg_struct_sim = sum(struct_sims) / len(struct_sims)  # ∈ [0,1]
#                 reward = 0.4 * avg_struct_sim  # ∈ [0,0.4]
#             else:
#                 reward = 0.0

#         if reward > best_reward:
#             best_reward = reward

#     return best_reward


# def _compute_struct_similarity(
#     g_denotation: List[Tuple],
#     p_denotation: List[Tuple],
# ) -> float:
#     """
#     只根据行数和列数算一个结构相似度 ∈ [0,1]
#     0：行列完全不对劲
#     1：行列数量完全一致
#     """
#     # 行数相似度
#     g_rows = len(g_denotation)
#     p_rows = len(p_denotation)
#     max_rows = max(g_rows, p_rows)
#     if max_rows == 0:
#         # 两边都没有行的话，结构上算 1（但一般这种会在前面就被 result_eq 判等了）
#         row_sim = 1.0
#     else:
#         row_diff = abs(g_rows - p_rows)
#         row_sim = max(0.0, 1.0 - row_diff / max_rows)

#     # 列数相似度
#     def _num_cols(denotation: List[Tuple]) -> int:
#         if len(denotation) == 0:
#             return 0
#         return len(denotation[0])

#     g_cols = _num_cols(g_denotation)
#     p_cols = _num_cols(p_denotation)
#     max_cols = max(g_cols, p_cols)
#     if max_cols == 0:
#         col_sim = 1.0
#     else:
#         col_diff = abs(g_cols - p_cols)
#         col_sim = max(0.0, 1.0 - col_diff / max_cols)

#     return (row_sim + col_sim) / 2.0



def eval_exec_match_continuous(
    db: str, p_str: str, g_str: str, plug_value: bool, keep_distinct: bool, progress_bar_for_each_datapoint: bool
) -> float:
    """
    连续版本的执行匹配评估。
    返回一个浮点数奖励，而不是二元的 0/1。
    """
    # 1. 预处理 (保持原有逻辑)
    p_str, g_str = postprocess(p_str), postprocess(g_str)
    if not keep_distinct:
        p_str = remove_distinct(p_str)
        g_str = remove_distinct(g_str)

    order_matters = "order by" in g_str.lower()

    # 2. 准备数据库路径
    db_dir = os.path.dirname(db)
    # 注意：这里假设所有相关数据库都在同一目录，且后缀包含 .sqlite
    db_paths = [os.path.join(db_dir, basename) for basename in os.listdir(db_dir) if basename.endswith(".sqlite")]
    if not db_paths:
        # 如果没找到，尝试使用传入的 db 本身
        db_paths = [db]

    preds = [p_str]
    if plug_value:
        _, preds = get_all_preds_for_execution(g_str, p_str)
        preds = chain([p_str], preds)

    max_reward = 0.0

    # 3. 遍历所有预测和数据库
    for pred in preds:
        current_reward = 0.0
        pred_executed_successfully = True
        
        # 用于累积所有数据库上的平均匹配度 (如果有多库测试)
        total_matched_ratio = 0.0
        db_count = 0

        ranger = tqdm.tqdm(db_paths) if progress_bar_for_each_datapoint else db_paths

        all_pass = True # 标记是否所有库都完全通过 (用于传统EM判断，这里主要用作参考)

        for db_path in ranger:
            # 执行 Gold SQL
            g_flag, g_denotation = run_sync_ephemeral(exec_on_db(db_path, g_str))
            # 执行 Predicted SQL
            p_flag, p_denotation = run_sync_ephemeral(exec_on_db(db_path, pred))

            # Gold 必须成功
            if g_flag == "exception":
                logger.warning(f"Gold query has error on {db_path}: {g_str}")
                continue # 或者 raise

            # 如果预测执行失败
            if p_flag == "exception":
                all_pass = False
                pred_executed_successfully = False
                # 执行失败奖励为 0，直接跳出当前 pred 的该库循环，但为了计算平均，我们记为0
                current_db_reward = 0.0
            else:
                # 核心修改：使用连续相似度比较代替 result_eq
                matched_rows, matched_cols, total_rows, total_cols = compare_results_continuous(
                    g_denotation, p_denotation, order_matters=order_matters
                )
                
                if total_rows == 0 and total_cols == 0:
                    # 空结果集特殊情况
                    if matched_rows == 0: # 都是空
                        current_db_reward = 1.0
                    else:
                        current_db_reward = 0.0
                else:
                    # 计算相似度比例: (匹配行 * 匹配列) / (总行 * 总列)
                    # 防止除以零
                    if total_rows > 0 and total_cols > 0:
                        acc = (matched_rows * matched_cols) / (total_rows * total_cols)
                    else:
                        acc = 0.0
                    
                    # # 应用奖励策略
                    # if acc == 0.0:
                    #     # 执行成功但结果完全不匹配 -> 安慰奖 0.5 (归一化后)
                    #     # 原代码是 0.5 (绝对值)，这里我们假设满分是 1.0，所以给 0.5
                    #     current_db_reward = 0.5
                    # else:
                    # 匹配度越高奖励越高，满分 1.0
                    current_db_reward = acc
                
                if acc < 1.0:
                    all_pass = False

            total_matched_ratio += current_db_reward
            db_count += 1

        if db_count > 0:
            current_reward = total_matched_ratio / db_count
        
        # 取所有预测中的最大奖励 (对应 plug_value 时的多条候选)
        if current_reward > max_reward:
            max_reward = current_reward
            
        # 如果已经拿到满分，可以提前结束
        if max_reward >= 1.0:
            return 1.0

    return max_reward

def compare_results_continuous(
    g_denotation: List[Tuple[Any]], 
    p_denotation: List[Tuple[Any]], 
    order_matters: bool
) -> Tuple[int, int, int, int]:
    """
    核心对比函数：将 denotation (List[Tuple]) 转为 DataFrame 并进行细粒度匹配。
    返回: (matched_rows, matched_cols, total_gold_rows, total_gold_cols)
    """
    import pandas as pd
    
    # 辅助函数：将 denotation 转为 DataFrame
    def denotation_to_df(denotation, columns_prefix="col"):
        if not denotation:
            return pd.DataFrame()
        # 生成默认列名 col_0, col_1...
        num_cols = len(denotation[0])
        col_names = [f"{columns_prefix}_{i}" for i in range(num_cols)]
        return pd.DataFrame(denotation, columns=col_names)

    df_gold = denotation_to_df(g_denotation, "g")
    df_pred = denotation_to_df(p_denotation, "p")

    # 如果 order_matters (有 Order By)，则行顺序必须一致，且不能进行灵活的行匹配
    # 但即使有 Order By，列的顺序可能还是不一致，且浮点数需要容错
    # 这里我们简化处理：如果 order_matters，我们强制要求行顺序一致，只进行列匹配和值容错
    if order_matters:
        # 简单模式：形状必须相同，然后逐列比较
        if df_gold.shape != df_pred.shape:
            return 0, 0, df_gold.shape[0], df_gold.shape[1]
        
        # 即使顺序固定，也要处理列名不对应的问题 (虽然通常 Order By 会固定列顺序，但以防万一)
        # 这里复用 check_dataframe 逻辑，但限制行重排
        # 为了简化，如果 order_matters 且形状相同，我们直接逐元素比较 (带容错)
        # 但为了利用论文的列匹配能力，我们还是调用 _compare_results_outcomes 的核心逻辑
        # 只是传入时标记需要严格行序？ 
        # 原论文代码 _compare_results_outcomes 主要通过行数裁剪来处理，这里我们直接调用它
        # 注意：原代码逻辑中，如果行数差异大直接返回0，如果行数相同则进入 check_dataframe
        pass 

    # 调用核心对比逻辑 (复用你提供的论文代码逻辑)
    # 我们需要适配一下输入，因为原代码期望 DataFrame
    res_shape_info = _compare_results_outcomes(df_pred, df_gold)
    
    # 解析返回值: [pred_shape, gold_shape, (matched_rows, matched_cols)]
    # 注意：原代码返回的 matched_rows 其实是基于 gold 的行数逻辑计算的
    # res_shape_info[2][0] -> 匹配的有效行数 (通常是 gold 的行数，如果在 cutoff 内)
    # res_shape_info[2][1] -> 匹配的列数
    
    matched_rows_stat = res_shape_info[2][0]
    matched_cols_stat = res_shape_info[2][1]
    
    total_gold_rows = res_shape_info[1][0]
    total_gold_cols = res_shape_info[1][1]

    # 特殊处理：如果原逻辑因为行数差异过大返回了 (total, 0)，这里 matched_cols 为 0
    # 如果完全匹配，matched_rows 应该等于 total_gold_rows, matched_cols 等于 total_gold_cols
    
    return matched_rows_stat, matched_cols_stat, total_gold_rows, total_gold_cols

# -------------------------------------------------------------------------
# 下面是从你提供的论文代码中提取的核心对比逻辑 (_compare_results_outcomes 及其依赖)
# 做了少量适配以独立运行
# -------------------------------------------------------------------------

def _compare_results_outcomes(temp_predicted_res: pd.DataFrame, ground_truth_res: pd.DataFrame):
    """
    直接复用论文代码逻辑，输入两个 DataFrame，返回统计信息。
    返回: [predicted_shape, ground_truth_shape, (matched_rows, matched_cols)]
    """
    
    def df_normalization(df1, df2, num, type='ground'):
        # 复用原逻辑
        cols1 = list(df1.columns)
        cols2 = list(df2.columns)
        
        target_col = set()
        # 寻找一个基准列 (优先 object, 然后 float, 然后 int)
        found = False
        for col1 in cols1:
            df1_type = str(df1[col1].dtype)
            if df1_type == 'object':
                target_col = set(deepcopy(df1[col1].dropna()))
                found = True
                break
        if not found:
            for col1 in cols1:
                df1_type = str(df1[col1].dtype)
                if 'float' in df1_type:
                    target_col = set(deepcopy(df1[col1].dropna()))
                    found = True
                    break
        if not found:
            for col1 in cols1:
                df1_type = str(df1[col1].dtype)
                if 'int' in df1_type:
                    target_col = set(deepcopy(df1[col1].dropna()))
                    found = True
                    break

        if len(target_col) == 0:
            return pd.DataFrame([])
        
        new_target_col = [] 
        match_col_idx = -1
        
        for idx, col2 in enumerate(cols2):
            if df1[col1].dtype != df2[col2].dtype:
                # 类型不同跳过，或者尝试转换？原代码是直接 continue
                # 为了鲁棒性，这里严格一点
                continue
            
            if 'float' in str(df1[col1].dtype):
                count = 0
                temp_matches = []
                for e in df2[col2].dropna():
                    for f in target_col:
                        if abs(e - f) < PRE_FLOAT:
                            count += 1
                            temp_matches.append(e)
                            break # 找到一个匹配就break内层循环? 原逻辑是遍历所有f
                # 原逻辑有点奇怪，它是统计有多少个e能在target_col里找到近似值
                # 如果 count >= len(target_col)，认为这一列包含了目标列的所有值
                if count >= len(target_col):
                    new_target_col = list(set(temp_matches))
                    match_col_idx = idx
                    break
            else:
                count = 0
                for e in df2[col2].dropna():
                    if e in target_col:
                        count += 1
                if count >= len(target_col):
                    new_target_col = list(target_col)
                    match_col_idx = idx
                    break
        
        if match_col_idx == -1 or len(new_target_col) == 0:
            return pd.DataFrame([])

        col2_name = cols2[match_col_idx]
        
        # 原代码这里的逻辑比较复杂，涉及 Counter 和去重
        # 简化理解：如果找到了对应的列，返回过滤后的 df2 或者 df1
        if num == 1 and len(target_col) == len(set(target_col)):
             # 这里的具体逻辑取决于是否需要保持多重集语义
             # 为了简化，我们直接返回基于该列过滤的 df2
             return df2[df2[col2_name].isin(new_target_col)]
        else:
            return df2[df2[col2_name].isin(new_target_col)]

    def check_dataframe(df1, df2):
        """
        检查 df1 的列是否能在 df2 中找到对应列 (内容相同)
        返回: (df1的行数, 匹配的列数)
        """
        cols1 = df1.columns
        cols2 = df2.columns
        used_cols2 = set()
        count = 0
        
        for col1 in cols1:
            df1_type = str(df1[col1].dtype)
            # 数据清洗与排序
            s1_raw = df1[col1].dropna()
            if 'float' in df1_type:
                s1 = sorted(s1_raw.round(3))
            elif df1_type == 'object':
                try:
                    s1 = sorted(s1_raw.astype('int64'))
                except:
                    s1 = sorted(s1_raw)
            else:
                s1 = sorted(s1_raw)

            found_match = False
            for col2 in cols2:
                if col2 in used_cols2:
                    continue

                df2_type = str(df2[col2].dtype)
                s2_raw = df2[col2].dropna()
                if 'float' in df2_type:
                    s2 = sorted(s2_raw.round(3))
                elif df2_type == 'object':
                    try:
                        s2 = sorted(s2_raw.astype('int64'))
                    except:
                        s2 = sorted(s2_raw)
                else:
                    s2 = sorted(s2_raw)

                # 长度不同直接跳过
                if len(s1) != len(s2):
                    continue

                is_equal = False
                if 'float' in df1_type and 'float' in df2_type:
                    is_equal = True
                    for i in range(len(s1)):
                        if abs(s1[i] - s2[i]) > PRE_FLOAT:
                            is_equal = False
                            break
                else:
                    if s1 == s2:
                        is_equal = True
                
                if is_equal:
                    used_cols2.add(col2)
                    found_match = True
                    break
            
            if found_match:
                count += 1
        
        return (df1.shape[0], count)

    # 主逻辑开始
    try:
        # 1. 填充 NaN (复用原逻辑)
        # ... (此处省略详细的 fillna 代码，直接调用原逻辑块，实际使用时建议完整复制原 fillna 部分)
        # 为简洁，这里假设输入数据比较干净，或者依赖 pandas 默认行为
        # 如果需要严格复现，请将原代码中的 fillna 块复制到这里
        
        # 2. 行数检查与裁剪
        pred_rows = temp_predicted_res.shape[0]
        gold_rows = ground_truth_res.shape[0]
        
        res = (0, 0) # 默认 (matched_rows, matched_cols)

        if pred_rows < gold_rows:
            if gold_rows - pred_rows > DF_CUTOFF:
                res = (gold_rows, 0) # 差异太大，匹配列数为0
            else:
                # 尝试从 gold 中裁剪出与 pred 行数相近的部分？
                # 原逻辑是：df_normalization(temp_predicted_res, ground_truth_res, ...)
                # 意思是看 ground_truth 中是否有子集能匹配 temp_predicted_res
                partial_ground = df_normalization(temp_predicted_res, ground_truth_res, ground_truth_res.shape[1], 'predicted')
                if partial_ground.shape[0] != temp_predicted_res.shape[0]:
                    res = (partial_ground.shape[0], 0)
                else:
                    res = check_dataframe(partial_ground, temp_predicted_res)

        elif pred_rows > gold_rows:
            if pred_rows - gold_rows > DF_CUTOFF:
                res = (gold_rows, 0)
            else:
                predicted_subset = df_normalization(ground_truth_res, temp_predicted_res, ground_truth_res.shape[1])
                if predicted_subset.shape[0] != ground_truth_res.shape[0]:
                    res = (ground_truth_res.shape[0], 0)
                else:
                    res = check_dataframe(ground_truth_res, predicted_subset)
        
        else: # 行数相等
            # 直接比较
            res = check_dataframe(ground_truth_res, temp_predicted_res)
            
        return [temp_predicted_res.shape, ground_truth_res.shape, res]

    except Exception as e:
        # 发生任何错误，返回 0 匹配
        return [temp_predicted_res.shape, ground_truth_res.shape, (0, 0)]




