"""推理式 + 规则化 grounding 的成对判别 (no-thinking, 但 prompt 让模型输出推理过程)。

关键设计 (回应用户约束):
- enable_thinking=False, temperature=0 不变; 但 prompt 要求模型【先写推理再给结论】,
  末行输出 "FINAL: A" / "FINAL: B", 由代码解析裁决 (不再读首token logprobs)。
- 模型看到的是扁平 token, 无法自行判断"行数/列数是否一致"。因此结构差异【全部用代码
  (规则) 预先算好】, 以自然语言事实喂给模型:
    * 每个结果的 行数 / 列数 / 列名表头 (列序对 BIRD 至关重要)
    * 列数差异 / 列序差异 / 行数差异
    * 结果集【对称差】: A 有而 B 无的示例行, 反之亦然
  模型只需在这些既定事实 + 问题/evidence 上推理, 不必自己去数 token。
- 默认聚焦 top-2 (最大簇 vs 次大簇), 每对正反两向 (debias)。

输出 schema 与 run_gated_tournament_judge.py 对齐, 可直接喂 sweep_gated_tournament.py:
  {model, topk, samples:{sid:{db_id, topk_sigs, pairs:[{a,b,o0,o1}]}}}
  o0=P(i胜|i在A) (推理裁决, 取 0/1), o1=P(j胜|j在A)。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import bird_common as B
import gated_common as G
from openai import OpenAI

# base_url / api_key 可用 env 覆盖 (跨判别器供应商: DashScope / routify 等), 默认 DashScope 向后兼容。
API_BASE = os.environ.get("JUDGE_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
API_KEY = os.environ.get("JUDGE_API_KEY", "")
MODEL = "glm-5.2"
USE_SIMPLICITY = False  # --simplicity 开启: 注入抗过度解读先验

# 可选 DNS 覆盖: 本机 VPC resolver 解析不了某内网公网域名时, 用 "host=ip" 直接映射
# (保留 hostname 做 TLS SNI, 只把连接 IP 换掉)。未设置则不影响原有行为。
_DNS_OVR = os.environ.get("JUDGE_DNS_OVERRIDE")
if _DNS_OVR and "=" in _DNS_OVR:
    _ovh, _ovip = _DNS_OVR.split("=", 1)
    _orig_gai = socket.getaddrinfo
    def _patched_gai(host, *a, **k):
        return _orig_gai(_ovip if host == _ovh else host, *a, **k)
    socket.getaddrinfo = _patched_gai


def out_path(model: str, thinking: bool = False, rep: str = "shortest") -> str:
    tag = model.replace(".", "").replace("-", "")
    suf = "_think" if thinking else ""
    rsuf = "" if rep == "shortest" else f"_{rep}"
    bsuf = "_simp" if USE_SIMPLICITY else ""
    return f"{B.CACHE_DIR}/gated_pairwise_{tag}_reason{suf}{rsuf}{bsuf}_top2.json"


OUT = out_path(MODEL)

client = OpenAI(api_key=API_KEY, base_url=API_BASE)

PROMPT = """You are a meticulous SQL judge. Two candidate SQL queries (A and B) answer the SAME question over the SAME database. Exactly one of them better matches what the question asks. Decide which one.

# Database Schema:
{schema}

# Question:
{question}

# External knowledge (evidence):
{evidence}

# Option A SQL:
{sql_a}

# Option B SQL:
{sql_b}

# Pre-computed comparison of the two execution results (produced by an exact tool — TRUST these facts, you do not need to recount anything):
{facts}

# A few sample rows (for reference):
A: {prev_a}
B: {prev_b}

# How to decide — reason through these checks IN ORDER:
1. Requested columns & COLUMN ORDER. The result must contain exactly the columns the question asks for, in the SAME order (BIRD compares columns by position). Use the column headers listed above to check order (e.g. a question asking "Street, City, Zip, State" requires that exact header order).
2. Column count. An extra or missing column (see the facts above) makes an option wrong.
3. Filtering precision. WHERE must match the question and evidence exactly. Use the ROW-COUNT difference above as a cue: a larger count often means a filter that is too broad (extra OR / wrong column); a smaller count means too strict.
4. Aggregation & grouping intent. Decide from the question whether it wants individual rows or a single aggregate value. Do not assume GROUP BY/AVG/COUNT just because words like "average"/"how many" appear. A wrong GROUP BY usually shows up as a different row count above.
5. Malformed results. If the sample rows echo a column name as a literal string, contain errors, or are nonsensical, that option is broken.
6. Join path & evidence formulas. Joins must connect the intended entities; if evidence gives a formula (rate = X/Y) or mapping, the query must implement it.
{bias}
Write a BRIEF reasoning (2-5 sentences) grounded in the facts above. Then, on the LAST line, output the verdict in EXACTLY this format:
FINAL: A
or
FINAL: B"""


# 通用检查清单 (V5): V1 已验证规则集 + 加强 COUNT-join 一条; 例子均抽象化, 无 dev 具体样本。
# 机制: prompt 是给"系统性偏复杂"的判别器加反向配重。实测: 无关的"精确"细化(区间含端点/
# 输出列精确/时间过滤)会重新激活 see-saw 净亏, 故不纳入。仅在 --simplicity 时注入。
SIMPLICITY_NOTE = """
# CRITICAL PRIOR — do NOT over-interpret (this is by far the most common judging mistake):
Reference gold answers in this benchmark almost always take the SIMPLEST, most LITERAL reading of the question. A query is NOT better for looking more thorough, more rigorous, or more "semantically complete". When two candidate queries differ only in that ONE adds extra machinery, STRONGLY PREFER the simpler one UNLESS the question or evidence EXPLICITLY demands that extra machinery. Concretely:
- Prefer COUNT(*) over COUNT(DISTINCT x), and keep this preference EVEN when the counted rows come from a JOIN that could repeat an entity id: count the qualifying ROWS, not distinct entities. Adding DISTINCT merely to de-duplicate a join is a frequent error. Keep DISTINCT ONLY when the question literally says "distinct"/"unique"/"different"/"how many different".
- Do NOT prefer a query that adds GROUP BY / AVG / aggregation unless the question explicitly asks for a per-group or averaged value. "highest/lowest/most/least X" is usually a plain ORDER BY ... LIMIT 1 over rows, NOT a group average.
- Do NOT prefer extra IS NOT NULL filters on columns the question did not single out.
- Do NOT prefer broadening with OR across similar columns (e.g. several comparable name or role columns) unless evidence explicitly says they are equivalent and must all be used.
- Prefer the SHORTER join path. Do NOT prefer a longer, "more correct-looking" join through extra tables when a direct join already answers the question.
- Interpret filters LITERALLY: a number or id named in the question is an equality on that id column, not a positional "the next/previous one"; an attribute of an entity means that entity's own column, not one reached via extra joins.
- Prefer GROUP BY / ORDER BY on the human-readable column the question names (e.g. a NAME) rather than a "more precise" surrogate id.
Only choose the more complex option when you can quote EXPLICIT wording in the question or evidence that requires it."""


# --------------------------------------------------------------------------
# 规则化结构提取 (代码算好, 喂事实给模型)
# --------------------------------------------------------------------------
def exec_full(db_id, sql, cap=5000):
    """执行 SQL, 返回 (col_names, rows[<=cap], truncated, err)。"""
    path = B.db_path_of(db_id)
    conn = None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
        conn.text_factory = lambda b: b.decode(errors="ignore")
        cur = conn.cursor()
        cur.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchmany(cap)
        trunc = len(rows) >= cap
        return cols, rows, trunc, None
    except Exception as e:  # noqa: BLE001
        return [], [], False, str(e)
    finally:
        if conn:
            conn.close()


def _rc(n, trunc):
    return f"{n}+" if trunc else str(n)


def struct_facts(ca, cb):
    """ca/cb: (cols, rows, trunc, err)。产出自然语言事实。"""
    cols_a, rows_a, ta, err_a = ca
    cols_b, rows_b, tb, err_b = cb
    L = []
    if err_a:
        L.append(f"- Option A FAILED to execute: {err_a[:100]}")
    if err_b:
        L.append(f"- Option B FAILED to execute: {err_b[:100]}")
    na, nb = _rc(len(rows_a), ta), _rc(len(rows_b), tb)
    L.append(f"- A returns {na} rows x {len(cols_a)} columns. Column headers: {cols_a}")
    L.append(f"- B returns {nb} rows x {len(cols_b)} columns. Column headers: {cols_b}")
    if len(cols_a) != len(cols_b):
        L.append(f"- COLUMN COUNT DIFFERS: A has {len(cols_a)}, B has {len(cols_b)}.")
    elif cols_a != cols_b:
        L.append(f"- Same column count but header labels/ORDER differ: A={cols_a} vs B={cols_b}.")
    if len(rows_a) != len(rows_b):
        L.append(f"- ROW COUNT DIFFERS: A={na}, B={nb}.")
    # 对称差 (忽略行序; 以 str 元组比较)
    sa = {tuple(str(x) for x in r) for r in rows_a}
    sb = {tuple(str(x) for x in r) for r in rows_b}
    only_a = list(sa - sb)[:5]
    only_b = list(sb - sa)[:5]
    if not only_a and not only_b:
        L.append("- The two result sets are IDENTICAL as sets (same rows, ignoring order).")
    else:
        if only_a:
            L.append(f"- Example rows in A but NOT in B ({len(sa - sb)} such rows): {only_a}")
        if only_b:
            L.append(f"- Example rows in B but NOT in A ({len(sb - sa)} such rows): {only_b}")
    return "\n".join(L)


VERD = re.compile(r"FINAL\s*[:：]?\s*\*{0,2}\s*([AB])", re.I)
TOK = re.compile(r"\b([AB])\b")


def parse_verdict(text):
    if not text:
        return None
    m = VERD.findall(text)
    if m:
        return m[-1].upper()
    m2 = TOK.findall(text.strip())
    return m2[-1].upper() if m2 else None


def call_verdict(prompt, model, thinking=False, retries=6):
    """返回 'A'/'B'/None。允许输出推理文本。
    glm 模型显式控制 enable_thinking; 其它模型不发该参数。
    思考模式: 提高 max_tokens, 且 content 无结果时回退到 reasoning_content 解析。"""
    import random
    kw = {}
    # 有些供商的模型名带组织前缀(如 SiliconFlow 的 "zai-org/GLM-5.2"), 前缀会让
    # startswith 判断全部落空 -> 不发 enable_thinking -> 思考模式默认开启, 输出
    # token 暴涨约百倍且 max_tokens=700 会截断结论, 裁决直接解析失败。故先剥前缀。
    ml = model.lower().rsplit("/", 1)[-1]
    if ml.startswith("glm") or "deepseek" in ml or ml.startswith("qwen3") or "kimi" in ml or "moonshot" in ml:
        kw["extra_body"] = {"enable_thinking": bool(thinking)}
        kw["temperature"] = 0.0
    elif ml.startswith("claude"):
        # claude 默认开 extended thinking (慢 ~6x: 50s->8s), 显式关闭提速; 且拒绝 temperature 参数。
        kw["extra_body"] = {"thinking": {"type": "disabled"}}
    elif ml.startswith("gpt"):
        # gpt-5 系列默认推理, 降到 low 提速 (8s->4s); temperature=0 可正常发。
        kw["reasoning_effort"] = "low"
        kw["temperature"] = 0.0
    else:
        kw["temperature"] = 0.0
    # gpt-5 / claude 可能消耗隐藏推理 token, 放宽 max_tokens 防止 FINAL 被截断。
    if thinking:
        mt = 8000
    elif ml.startswith("gpt") or ml.startswith("claude"):
        mt = 1500
    else:
        mt = 700
    last = None
    for att in range(retries):
        try:
            _t0 = time.time()
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=mt,
                **kw,
            )
            try:
                import llm_usage
                llm_usage.record("judge", getattr(resp, "usage", None), time.time() - _t0, model=model)
            except Exception:  # noqa: BLE001  统计绝不影响主流程
                pass
            msg = resp.choices[0].message
            v = parse_verdict(msg.content or "")
            if v is None:
                v = parse_verdict(getattr(msg, "reasoning_content", "") or "")
            if v is not None:
                return v
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(3 * (2 ** att), 40) + random.uniform(0, 2))
    if last:
        try:
            import llm_usage
            llm_usage.record("judge", None, 0.0, error=True, model=model)
        except Exception:  # noqa: BLE001
            pass
        print(f"[err] {str(last)[:120]}", flush=True)
    return None


SCHEMA_LUT = {}  # "db|||question" -> per-sample linked schema (rollout 生成时实际所见)


def build_prompt(rec, ca_cluster, cb_cluster, facts):
    key = f"{rec['db_id']}|||{rec.get('question','')}"
    schema = SCHEMA_LUT.get(key)
    if schema is None:
        schema = B.schema_with_descriptions(B.db_path_of(rec["db_id"]))
    return PROMPT.format(
        schema=schema, question=rec.get("question", ""),
        evidence=rec.get("evidence", "") or "(none)",
        sql_a=ca_cluster["rep_sql"], sql_b=cb_cluster["rep_sql"],
        facts=facts,
        prev_a=ca_cluster["preview"], prev_b=cb_cluster["preview"],
        bias=(SIMPLICITY_NOTE if USE_SIMPLICITY else ""),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", type=str, default=MODEL)
    ap.add_argument("--thinking", action="store_true", help="glm 开启思考模式")
    ap.add_argument("--simplicity", action="store_true", help="注入抗过度解读先验")
    ap.add_argument("--rep", type=str, default="shortest", choices=["shortest", "mode"],
                    help="簇代表SQL选取: shortest(最短) | mode(最常见写法)")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--schema_cache", type=str, default=None,
                    help="per-sample linked schema json (db|||question -> schema). 不给则用全量 schema")
    args = ap.parse_args()
    global USE_SIMPLICITY
    USE_SIMPLICITY = args.simplicity
    if args.schema_cache:
        sc = json.load(open(args.schema_cache))
        SCHEMA_LUT.update(sc)
        print(f"[schema] per-sample linked schema 载入 {len(SCHEMA_LUT)} 条 <- {args.schema_cache}", flush=True)
    G.REP_MODE = args.rep
    model = args.model
    out = args.out or out_path(model, args.thinking, args.rep)
    print(f"[model] {model}  thinking={args.thinking}  rep={args.rep}  -> {out}", flush=True)

    samples = B.load_samples()
    info = G.classify(samples)
    print(f"[classify] {info['counts']}  fixed_correct={info['fixed_correct']}", flush=True)
    contested = info["contested_ids"]
    if args.limit:
        contested = contested[:args.limit]
    per = info["per_sample"]

    raw = {}
    if os.path.exists(out):
        raw = {int(k): v for k, v in json.load(open(out))["samples"].items()}
        print(f"[resume] 已完成 {len(raw)} 个样本", flush=True)

    # 预算每个样本 top-K 簇的执行结构 (进程内缓存 sig->exec)
    exec_cache = {}

    def get_exec(db_id, cluster):
        key = (db_id, cluster["sig"])
        if key not in exec_cache:
            exec_cache[key] = exec_full(db_id, cluster["rep_sql"])
        return exec_cache[key]

    jobs = []
    ctx = {}
    for sid in contested:
        if sid in raw:
            continue
        cls = per[sid]["clusters"][:args.topk]
        ctx[sid] = {"rec": samples[sid], "cls": cls}
        m = len(cls)
        for i in range(m):
            for j in range(i + 1, m):
                jobs.append((sid, i, j, 0))
                jobs.append((sid, i, j, 1))
    print(f"[jobs] 待跑样本={len(ctx)} 成对方向调用={len(jobs)} topk={args.topk}", flush=True)
    if not jobs:
        print("[done] 无待跑任务"); return

    pref = {}
    t0 = time.time()
    done = 0

    def work(job):
        sid, i, j, order = job
        cls = ctx[sid]["cls"]; rec = ctx[sid]["rec"]
        db = rec["db_id"]
        ci, cj = cls[i], cls[j]
        # 事实按 (A,B) 的呈现顺序算
        if order == 0:
            a, b = ci, cj
        else:
            a, b = cj, ci
        facts = struct_facts(get_exec(db, a), get_exec(db, b))
        v = call_verdict(build_prompt(rec, a, b, facts), model, args.thinking)
        return sid, i, j, order, v

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, jb) for jb in jobs]
        for fut in as_completed(futs):
            sid, i, j, order, v = fut.result()
            d = pref.setdefault((sid, i, j), {})
            # order0: A=i,B=j -> verdict A => i胜 => o0=1
            # order1: A=j,B=i -> verdict A => j胜 => o1=1
            win_first = 1.0 if v == "A" else (0.0 if v == "B" else 0.5)
            if order == 0:
                d["o0"] = win_first
            else:
                d["o1"] = win_first
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(jobs)}  ({time.time()-t0:.0f}s)", flush=True)

    # 组装输出
    out_samples = dict(raw)
    for sid in ctx:
        cls = ctx[sid]["cls"]
        sigs = [c["sig"] for c in cls]
        pairs = []
        m = len(cls)
        for i in range(m):
            for j in range(i + 1, m):
                d = pref.get((sid, i, j), {})
                pairs.append({"a": cls[i]["sig"], "b": cls[j]["sig"],
                              "o0": d.get("o0", 0.5), "o1": d.get("o1", 0.5)})
        out_samples[sid] = {"db_id": ctx[sid]["rec"]["db_id"], "topk_sigs": sigs, "pairs": pairs}

    json.dump({"model": model, "topk": args.topk,
               "samples": {str(k): v for k, v in out_samples.items()}},
              open(out, "w"), ensure_ascii=False)
    print(f"[save] {out}  样本={len(out_samples)}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
