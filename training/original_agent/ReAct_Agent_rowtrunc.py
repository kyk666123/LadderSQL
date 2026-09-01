"""行数版执行结果截断 Agent (RowTruncAgent)。

在不改动基础 `Agent` 的前提下, 以子类方式覆盖"SQL 执行结果 -> 观测字符串"的格式化:
- 按【整行】展示, 绝不切断某一行的中间;
- 单元格值要么【完整展示】、要么【整体隐藏】为占位符, 绝不做值内部字符截断
  (避免把 "New York" 截成 "New Yor…" 误导模型);
- 从首行首列起累加完整单元格, 累计不超过 execution_truncate(字符预算);
- 预算用尽时当前行剩余列以 "..." 占位, 并停止展示后续行;
- 至少保留 1 行结构; 若首个单元格本身就超预算, 用 "<N chars omitted>" 占位;
- 行数上限 execution_max_rows (env REACT_EXEC_MAX_ROWS, 默认 50)。

采样与训练均实例化本类即可获得一致的行数版观测截断; 基础 Agent 行为不受影响。
"""
from __future__ import annotations

import os
import time
from typing import Optional

from original_agent.ReAct_Agent import Agent, State, logger


class RowTruncAgent(Agent):
    def __init__(self, *args, execution_max_rows: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        # 行数上限: 未显式传入则读 env(便于采样/训练统一开启), 默认 50 行
        self.execution_max_rows = (
            execution_max_rows if execution_max_rows is not None
            else int(os.getenv("REACT_EXEC_MAX_ROWS", "50"))
        )

    def truncate_execution(self, execution: str) -> str:
        # 行模式: 结果已在 execute_query 按整行+预算裁好, 不再按字符切(避免切断整行/尾注);
        # 仅对异常超长(如未预期的错误串)做 2x 预算硬兜底。
        hard = max(self.execution_truncate * 2, 512)
        if len(execution) > hard:
            return execution[:hard] + "\n... (truncated)"
        return execution

    def _format_rows_by_line(self, rows: list) -> str:
        """按整行 + 整单元格展示执行结果(行数版截断)。

        规则见模块 docstring。绝不做值内部字符截断。
        """
        total = len(rows)
        if total == 0:
            return "[]"
        B = self.execution_truncate
        R = self.execution_max_rows

        out_parts: list[str] = []
        used = 2  # 当前 body 字符数(含首尾 "[" "]")
        values_hidden = False
        rows_shown = 0

        for ri, row in enumerate(rows):
            cells: list[str] = []
            budget_hit = False
            first_cell_omitted = False
            cur = used + (2 if out_parts else 0) + 2  # ", "(行间) + "(" ")"
            for ci, v in enumerate(row):
                cell = repr(v)  # 与 str(tuple) 元素一致: 字符串带引号/数字原样/bytes 为 b'...'
                add = (2 if cells else 0) + len(cell)  # ", "(列间) + 值
                if cur + add <= B:
                    cells.append(cell)
                    cur += add
                else:
                    budget_hit = True
                    values_hidden = True
                    if ri == 0 and ci == 0:
                        # 首行首格就超预算: 占位(告知长度), 保证 >=1 行有结构
                        cells.append(f"<{len(cell)} chars omitted>")
                        first_cell_omitted = True
                    break
            if budget_hit and not cells:
                break  # 本行(非首行)一格都放不下 -> 不展示空行
            if budget_hit and not first_cell_omitted:
                cells.append("...")  # 剩余列值隐藏
            row_str = "(" + ", ".join(cells) + ")"
            out_parts.append(row_str)
            used = len("[" + ", ".join(out_parts) + "]")  # 精确重算已用字符数
            rows_shown += 1
            if budget_hit:
                break
            if R and rows_shown >= R:
                break

        body = "[" + ", ".join(out_parts) + "]"
        notes = []
        if rows_shown < total:
            notes.append(f"showing {rows_shown} of {total} rows")
        if values_hidden:
            notes.append("some values hidden (exceeded budget)")
        if notes:
            body += "\n... (" + "; ".join(notes) + ")"
        return body

    def execute_query(self, state: State, _sql_timeout: int = 15) -> State:
        """镜像基础 Agent.execute_query, 仅把结果格式化(str(rows))换成按整行截断。

        其余(超时中断/错误处理/计数)与基础实现保持一致, 便于两条路径行为对齐。
        """
        import sqlite3
        import threading

        t_start = time.time()
        query = state["query"]

        try:
            conn = sqlite3.connect(self._db_file_path)
            _timed_out = False

            def _interrupt():
                nonlocal _timed_out
                _timed_out = True
                try:
                    conn.interrupt()
                except Exception:
                    pass

            timer = threading.Timer(_sql_timeout, _interrupt)
            timer.start()
            try:
                cursor = conn.execute(query)
                rows = cursor.fetchall()
                timer.cancel()
                # 行数版: 按整行 + 整单元格截断(唯一与基础实现不同之处)
                execution_result = self._format_rows_by_line(rows)
            except sqlite3.OperationalError as e:
                timer.cancel()
                if _timed_out or "interrupted" in str(e).lower():
                    execution_result = f"Error: Query execution timed out after {_sql_timeout}s. The query is too slow, please optimize it or try a different approach."
                    logger.warning(f"[SQL TIMEOUT] Query exceeded {_sql_timeout}s: {query[:100]}...")
                else:
                    execution_result = f"Error: {e}"
            except Exception as e:
                timer.cancel()
                execution_result = f"Error: {e}"
            finally:
                conn.close()
        except Exception as e:
            execution_result = f"Error: Failed to connect to database: {e}"

        if not isinstance(execution_result, str):
            execution_result = str(execution_result)

        # Detect SQL execution errors in the result
        _result_lower = execution_result.lower()
        is_error = any(
            kw in _result_lower
            for kw in ("error", "operationalerror", "no such", "syntax error", "ambiguous", "timed out")
        )

        logger.info(f"[NODE TIMING] execute_query: {time.time()-t_start:.3f}s")
        return {
            **state,
            "execution": execution_result,
            "sql_exec_count": state.get("sql_exec_count", 0) + 1,
            "sql_error_count": state.get("sql_error_count", 0) + (1 if is_error else 0),
        }
