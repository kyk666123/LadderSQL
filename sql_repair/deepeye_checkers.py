"""DeepEye-SQL repair/verification checker chain (ported into this project, reward-blind).

Strictly serial chain: every candidate SQL passes through the checkers in order, the output of one
feeding the next.
- Execution-driven (Syntax/Result): run the SQL, look at result_type, and only call the LLM when a
  checker fires; sample N repaired versions -> execute all of them -> keep only the ones that run
  -> pick one by **majority vote over execution signatures** (reward-blind self-consistency).
- Rule-driven (Join/OrderByLimit/Time/Select/MaxMin/OrderByNull): only acts when a regex matches an
  anti-pattern; Time and Select-concatenation are purely deterministic rewrites (zero LLM), the rest
  use the COMMON prompt as a tightly constrained rewrite (n=1).

Throughout, only **execution validity / result shape / regex anti-patterns / self-consistency** are
used; gold and reward are never looked at.
The LLM is fixed to glm-5.2 with thinking off (reusing run_gated_reasoning_judge.client, with
credentials from JUDGE_API_KEY / JUDGE_API_BASE; REPAIR_MODEL overrides the model).
Reference implementation: DeepEye-SQL (SIGMOD 2026) app/pipeline/sql_revision/ -- this file is an
independent reimplementation of its checkers, keeping only the reward-blind parts.
"""
from __future__ import annotations

import os
import re
import sqlite3
import sys
import threading
from collections import Counter
from typing import Dict, List, Optional, Tuple

# reuse the execution / schema / judge client from inference_selection (same-repo relative path, no install needed)
_HERE = os.path.dirname(os.path.abspath(__file__))
_SELECTION_DIR = os.path.join(os.path.dirname(_HERE), "inference_selection")
for _p in (_HERE, _SELECTION_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bird_common as B
import run_gated_reasoning_judge as R

# LLM used for repair: defaults to glm-5.2 (backward compatible); REPAIR_MODEL or MULTIREP_MODEL can
# override it so that generation / repair / judging all use the same model (best-of-32 same-model setup).
MODEL = os.environ.get("REPAIR_MODEL") or os.environ.get("MULTIREP_MODEL", "glm-5.2")
EXEC_TIMEOUT = int(os.environ.get("EXEC_TIMEOUT", "15"))   # giant DB 可调到 30~60
IDENT = r'(?:`[^`]+`|\[[^\]]+\]|"[^"]+"|[\w\.]+)'

# ----------------------------------------------------------------------------
# execution + five-way result_type classification (aligned with DeepEye app/db_utils/execution.py)
# ----------------------------------------------------------------------------
def run_sql(db_id: str, sql: str, timeout: int = EXEC_TIMEOUT):
    """Returns (result_type, cols, rows, err).
    result_type ∈ {success, empty_result, all_null_result, execution_error, timeout}"""
    if not sql or not sql.strip():
        return "execution_error", [], [], "empty sql"
    path = B.db_path_of(db_id)
    conn = None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout)
        conn.text_factory = lambda b: b.decode(errors="ignore")
        timed = {"v": False}

        def _intr():
            timed["v"] = True
            try:
                conn.interrupt()
            except Exception:
                pass

        timer = threading.Timer(timeout, _intr)
        timer.start()
        try:
            cur = conn.cursor()
            cur.execute(sql)
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchall()
        finally:
            timer.cancel()
        if len(rows) == 0:
            return "empty_result", cols, rows, None
        if not any(any(v is not None for v in row) for row in rows):
            return "all_null_result", cols, rows, None
        return "success", cols, rows, None
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if timed.get("v") or "interrupt" in msg.lower():
            return "timeout", [], [], f"timeout after {timeout}s"
        return "execution_error", [], [], msg
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


def _result_str(rtype, cols, rows, err, k=5):
    if err:
        return f"[{rtype}] {err[:200]}"
    if not rows:
        return "[empty_result] The SQL query returned an empty result table."
    head = rows[:k]
    lines = [" | ".join(str(c) for c in cols)]
    for r in head:
        lines.append(" | ".join(str(x)[:100] for x in r))
    extra = f"\n... ({len(rows)} rows total)" if len(rows) > k else ""
    return "\n".join(lines) + extra


def _sig(rows) -> str:
    import hashlib
    return hashlib.md5(str(sorted(str(r) for r in rows)).encode("utf-8", "ignore")).hexdigest()


# ----------------------------------------------------------------------------
# LLM call (glm-5.2 no-thinking) + <result> parsing
# ----------------------------------------------------------------------------
_RESULT_RE = re.compile(r"<result>(.*?)</result>", re.DOTALL | re.IGNORECASE)
_SQLFENCE_RE = re.compile(r"```sql\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _parse_sql(text: str) -> Optional[str]:
    if not text:
        return None
    m = _RESULT_RE.search(text)
    body = m.group(1).strip() if m else text.strip()
    fm = _SQLFENCE_RE.search(body)
    if fm:
        body = fm.group(1).strip()
    body = body.strip().strip("`").strip()
    if body.lower().startswith("sql"):
        body = body[3:].strip()
    return body or None


def _call_llm(prompt: str, temperature: float, retries: int = 5) -> Optional[str]:
    import time, random
    last = None
    for att in range(retries):
        try:
            _t0 = time.time()
            resp = R.client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=1200,
                extra_body={"enable_thinking": False},
            )
            try:
                import llm_usage
                llm_usage.record("repair", getattr(resp, "usage", None), time.time() - _t0, model=MODEL)
            except Exception:  # noqa: BLE001  accounting must never affect the main flow
                pass
            txt = resp.choices[0].message.content or ""
            sql = _parse_sql(txt)
            if sql:
                return sql
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(3 * (2 ** att), 30) + random.uniform(0, 2))
    if last:
        print(f"[llm-err] {str(last)[:120]}", flush=True)
    return None


def _schema(db_id: str) -> str:
    return B.schema_with_descriptions(B.db_path_of(db_id))


# ----------------------------------------------------------------------------
# execution-driven: sampled repair + self-consistency majority vote (reward-blind)
# ----------------------------------------------------------------------------
def _self_consistency_pick(db_id: str, cands: List[str], accept_types) -> Optional[str]:
    """Execute every sampled repair, keep only those whose result_type is in accept_types, and pick
    one by majority vote over execution signatures."""
    valid = []  # (sql, sig)
    for c in cands:
        if not c:
            continue
        rtype, cols, rows, err = run_sql(db_id, c)
        if rtype in accept_types:
            valid.append((c, _sig(rows)))
    if not valid:
        return None
    cnt = Counter(sig for _, sig in valid)
    # majority signature; ties go to the first seen
    best_sig = max(cnt, key=lambda s: cnt[s])
    for c, sig in valid:
        if sig == best_sig:
            return c
    return None


def _exec_repair(sql, db_id, question, evidence, rtype, cols, rows, err,
                 budget: int, accept_types, temperature: float) -> str:
    """EXECUTION_CHECKER_PROMPT: use the error / result to have the LLM sample `budget` repaired
    versions, then pick one by self-consistency."""
    result_str = _result_str(rtype, cols, rows, err)
    prompt = EXECUTION_CHECKER_PROMPT.format(
        DATABASE_SCHEMA=_schema(db_id), QUESTION=question,
        HINT=evidence or "(none)", QUERY=sql, RESULT=result_str,
    )
    samples = []
    for _ in range(budget):
        s = _call_llm(prompt, temperature=temperature)
        if s:
            samples.append(s)
    picked = _self_consistency_pick(db_id, samples, accept_types)
    return picked if picked else sql


def _common_repair(sql, db_id, question, evidence, suggestion) -> str:
    """COMMON_CHECKER_PROMPT: tightly constrained "only the suggested point may change", n=1 temperature=0."""
    prompt = COMMON_CHECKER_PROMPT.format(
        DATABASE_SCHEMA=_schema(db_id), QUESTION=question,
        HINT=evidence or "(none)", QUERY=sql, SUGGESTIONS=suggestion,
    )
    s = _call_llm(prompt, temperature=0.0)
    return s if s else sql


# ----------------------------------------------------------------------------
# applying one checker (returns the repaired SQL) + the serial chain
# ----------------------------------------------------------------------------
DEFAULT_CHAIN = ["syntax", "time", "select", "result"]
FULL_CHAIN = ["syntax", "join", "order_by_limit", "time", "select", "max_min", "order_by_null", "result"]


def _apply_checker(name, sql, db_id, question, evidence, budget, temperature):
    """Apply a single checker, returning (revised_sql, n_llm_calls_flag)."""
    if name == "syntax":
        rtype, cols, rows, err = run_sql(db_id, sql)
        if rtype in ("success", "empty_result", "all_null_result"):
            return sql, False
        # error / timeout -> sampled repair, accept anything that runs (including empty/all_null)
        return _exec_repair(sql, db_id, question, evidence, rtype, cols, rows, err,
                            budget, ("success", "empty_result", "all_null_result"), 0.8), True
    if name == "result":
        rtype, cols, rows, err = run_sql(db_id, sql)
        if rtype == "success":
            return sql, False
        # not success (error / empty / all null) -> sampled repair, accept only success
        return _exec_repair(sql, db_id, question, evidence, rtype, cols, rows, err,
                            budget, ("success",), 0.8), True
    if name == "time":
        f = fix_time(sql)
        return (f, False) if f else (sql, False)
    if name == "select":
        # (a) concatenation -> comma (deterministic)  (b) ambiguous T.* -> COMMON
        f = fix_select_concat(sql)
        if f:
            sql = f
        sug = detect_select_amb(sql)
        if sug:
            return _common_repair(sql, db_id, question, evidence, sug), True
        return sql, False
    if name == "join":
        sug = detect_join(sql)
        return (_common_repair(sql, db_id, question, evidence, sug), True) if sug else (sql, False)
    if name == "order_by_limit":
        sug = detect_order_by_limit(sql)
        return (_common_repair(sql, db_id, question, evidence, sug), True) if sug else (sql, False)
    if name == "max_min":
        sug = detect_max_min(sql)
        return (_common_repair(sql, db_id, question, evidence, sug), True) if sug else (sql, False)
    if name == "order_by_null":
        sug = detect_order_by_null(sql)
        return (_common_repair(sql, db_id, question, evidence, sug), True) if sug else (sql, False)
    return sql, False


def revise_chain(sql, db_id, question, evidence, chain=None, budget=8, temperature=0.8):
    """One candidate passes through the checkers of `chain` in order; returns
    (final_sql, n_llm_checkers_fired)."""
    chain = chain or DEFAULT_CHAIN
    cur = (sql or "").strip()
    fired = 0
    for name in chain:
        try:
            cur, used_llm = _apply_checker(name, cur, db_id, question, evidence, budget, temperature)
            if used_llm:
                fired += 1
        except Exception as e:  # noqa: BLE001
            print(f"[chain-err] {name}: {str(e)[:120]}", flush=True)
    return cur, fired


# ----------------------------------------------------------------------------
# rule-based anti-pattern detection (DeepEye's regexes, copied verbatim)
# ----------------------------------------------------------------------------
def detect_join(sql):
    pat = re.compile(
        rf"JOIN\s+{IDENT}(\s+AS\s+{IDENT}){{0,1}}\s+ON(\s+{IDENT}\.{IDENT}\s*(=\s*{IDENT}\.{IDENT}(?:\s+OR\s+{IDENT}\.{IDENT}\s*=\s*{IDENT}\.{IDENT})+|IN\s+\(.*?\)))",
        re.IGNORECASE | re.DOTALL)
    if pat.findall(sql):
        return ("The SQL uses the JOIN function incorrectly, due to using "
                "`JOIN table AS T ON Ta.column1 = Tb.column2 OR Ta.column1 = Tb.column3` "
                "or `JOIN table AS T ON Ta.column1 IN`, please only keep the highest "
                "priority group of `Ta.column = Tb.column` in `OR`.")
    return None


def detect_order_by_limit(sql):
    pat = re.compile(rf"ORDER BY ((MIN|MAX)\(\s*({IDENT})\s*\)).*? LIMIT \d+", re.IGNORECASE | re.DOTALL)
    res = pat.search(sql)
    if res:
        return (f"The SQL uses the ORDER BY function incorrectly, using MIN/MAX in ORDER BY "
                f"caluse is incrorrect (`{res.group()}`), please correct the SQL. If the SQL "
                f"contains GROUP BY, please judge whether the content of `{res.groups()[0]}` "
                f"needs to use `SUM({res.groups()[2]})`.")
    return None


def detect_max_min(sql):
    max_min_pat = re.compile(rf"=\s*\(\s*SELECT\s*(MAX|MIN)\s*\(\s*({IDENT})\s*\)\s*FROM\s*({IDENT})", re.IGNORECASE | re.DOTALL)
    fun_amb = max_min_pat.findall(sql)
    order_amb = set(re.findall(r"= (\(SELECT .* LIMIT \d\))", sql, re.IGNORECASE | re.DOTALL))
    sel_pat = re.compile(rf"^SELECT[^\(\)]*? ((MIN|MAX)\(\s*{IDENT}\s*\)).*?LIMIT 1", re.IGNORECASE | re.DOTALL | re.MULTILINE)
    select_amb = set(sel_pat.findall(sql))
    sug = []
    for fuc, col, table in fun_amb:
        order = "DESC" if fuc == "MAX" else "ASC"
        sug.append(f"WHERE {col} = (SELECT {fuc}({col}) FROM {table}): Please use ORDER BY {table}.{col} {order} LIMIT 1 instead of nested SQL")
    for fun in order_amb:
        sug.append(f"{fun}: Please use JOIN instead of nested SQL")
    for fun in select_amb:
        sug.append(f"{fun[0]}: {fun[1]} function is redundant due to LIMIT clause, please use ORDER BY + LIMIT instead")
    if sug:
        return "\n".join(f"{i+1}. {s}" for i, s in enumerate(sug))
    return None


def detect_order_by_null(sql):
    inn = re.findall(r"ORDER BY .*?(?<!DESC )LIMIT +\d+;{0,1}", sql)
    if not inn:
        return None
    for x in inn:
        if re.findall(r"SUM\(|COUNT\(", x):
            return None
    return "".join(f"Please add `IS NOT NULL` condition **in the WHERE clause** for the ORDER BY column: {x}\n" for x in inn)


def detect_select_amb(sql):
    amb = re.findall(rf"^SELECT.*? ({IDENT}\.\*).*?FROM", sql, re.IGNORECASE | re.DOTALL | re.MULTILINE)
    if amb:
        return "".join(
            f"{i+1}. We have specified that the ambiguous query is the corresponding id column, "
            f"please replace {x} with the corresponding id column in the above SQL\n"
            for i, x in enumerate(amb))
    return None


# purely deterministic rewrites (zero LLM)
def fix_time(sql):
    res = re.sub(r"(strftime *\([^\(]*?\) *[>=<]+ *)(\d{4,})", r"\1'\2'", sql)
    return res if res != sql else None


def fix_select_concat(sql):
    if re.findall(r"^SELECT.*?\|\| ' ' \|\| .*?FROM", sql, re.IGNORECASE | re.DOTALL | re.MULTILINE):
        return sql.replace("|| ' ' ||", ", ").replace("|| ', ' ||", ", ")
    return None


# ----------------------------------------------------------------------------
# Prompts (taken from DeepEye's originals)
# ----------------------------------------------------------------------------
EXECUTION_CHECKER_PROMPT = """# Task:
You are an SQL database expert tasked with correcting a SQL query. A previous attempt to run a query did not yield the correct results, either due to errors in execution or because the result returned was empty or unexpected. Your role is to analyze the error based on the provided database schema and the details of the failed execution, and then provide a corrected version of the SQL query.

# Instructions:
1. Review Database Schema to understand the database structure.
2. Analyze Query Requirements: the original question, the hint, the executed SQL, and its execution result (why it failed: syntax error, wrong column, empty/unexpected result, etc.).
3. Correct the Query so it correctly fetches the requested data according to the schema and requirements.

# Output Format (respond with XML):
<reasoning>
Your detailed reasoning: analyze the previous query, the schema, and fix the failed query.
</reasoning>
<result>
The final revised SQL query, executable by SQLite directly. No SQLite comment, no explanation text, no XML-special characters.
</result>

# Input:
## Database Schema:
{DATABASE_SCHEMA}

## Question:
{QUESTION}

## Hint:
{HINT}

## Previous SQL:
{QUERY}

## Execution Result:
{RESULT}

Based on the question, schema, previous query, and the execution result, fix the query, and only output the XML (<reasoning>...</reasoning> and <result>...</result>).
"""

COMMON_CHECKER_PROMPT = """# Task:
You are an SQL database expert tasked with correcting a SQL query. An external SQL checker tool has checked the query and provided suggestions. Analyze the suggestions and, based on the provided database schema, provide a corrected version of the SQL query.

# Instructions:
1. Review Database Schema.
2. Analyze: the original question, the hint, the SQL query, and the modification suggestions.
3. Correct the query based ONLY on the given suggestions.

[IMPORTANT]
You are NOT ALLOWED to do any other modifications which are not listed in the given suggestions.

# Output Format (respond with XML):
<reasoning>
Your reasoning: understand the suggestions, analyze the query and schema, then fix the query.
</reasoning>
<result>
The final revised SQL query, executable by SQLite directly. No SQLite comment, no explanation text, no XML-special characters.
</result>

# Input:
## Database Schema:
{DATABASE_SCHEMA}

## Question:
{QUESTION}

## Hint:
{HINT}

## Previous SQL:
{QUERY}

## Modification Suggestions:
{SUGGESTIONS}

Based on the question, schema, previous SQL and suggestions, fix the query, and only output the XML (<reasoning>...</reasoning> and <result>...</result>).
"""
