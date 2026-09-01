#!/usr/bin/env python3
"""Per-question / per-module cost report for a LadderSQL dev (or test) run.

Produces exactly the four breakdowns requested:
  (1) end-to-end per question:            avg / min / max        [time, in-tok, out-tok]
  (2) per module, per question:           avg / min / max        [time]
  (3) end-to-end per question:            avg / min / max        [in-tok, out-tok]   (== part of 1)
  (4) per module, per question:           avg / min / max        [in-tok, out-tok]

Modules
  - generation : local 14B (vLLM). Per-trial in/out/time come from the candidate pool
                 (`trials[].in_tok/out_tok/llm_time` if present, else recomputed with
                 `token_cost`-style trajectory accounting). Per question = sum over its 32 trials.
  - repair     : closed LLM (GLM-5.2). From llm_usage detail JSONL, stage="repair".
                 Repair dedups writings ACROSS questions, so its natural unit is a writing,
                 not a question; we report it BOTH ways (per-writing, and amortised per-question
                 by mapping each writing back to the questions that contain it).
  - judge      : closed LLM. From llm_usage detail JSONL, stage="judge", unit = question index.

Inputs
  --pool         candidate pool JSON (the *_repaired.json produced by stage 3; has trials[])
  --detail       llm_usage detail JSONL (default: <work>/logs/llm_usage_detail.jsonl)
  --out          write the full report as JSON here (optional)

Time note
  For the local generator, 32 trials of one question run concurrently, so a per-question WALL
  time is not well defined; we report per-question COMPUTE-seconds (sum of the 32 trials'
  llm_time) and, separately, single-trial time. Closed-LLM per-call seconds ARE wall time.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict


def _stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0, "avg": 0, "min": 0, "max": 0, "sum": 0}
    return {"n": len(xs), "avg": round(sum(xs) / len(xs), 2),
            "min": round(min(xs), 3), "max": round(max(xs), 3), "sum": round(sum(xs), 2)}


def _norm_sql(s):
    return " ".join((s or "").split())


def _sum_by_unit(rows_by_unit):
    """{unit: [rows]} -> {unit: {in, out, sec}} 每个 unit 的 token/秒求和。"""
    out = {}
    for unit, rows in rows_by_unit.items():
        out[unit] = {
            "in": sum(r.get("prompt_tokens") or 0 for r in rows),
            "out": sum(r.get("completion_tokens") or 0 for r in rows),
            "sec": sum(r.get("seconds") or 0 for r in rows),
        }
    return out


def _wall_by_unit(path):
    """sample_wall.jsonl -> {unit: wall_sec}。"""
    d = {}
    if path and os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("unit") is not None and r.get("wall_sec") is not None:
                    d[str(r["unit"])] = r["wall_sec"]
            except json.JSONDecodeError:
                pass
    return d


def end2end_join(pool_path, by_stage, schema_wall_path, judge_wall_path, repair_stage_wall, n,
                 repair_keymap=None):
    """按题 (key=db|||question) 把四模块合并 -> 端到端每题 token/时间, 再求 avg/min/max。

    token: 四模块可加(repair 通过候选池写法归属)。
    time:  schema/gen/judge 有每题干净墙钟; repair 是批处理阶段, 无每题墙钟, 以
           (repair 阶段总墙钟 / N) 作为每题摄派常量加入(对 avg 准确, 对 min/max 不引入方差)。
    """
    data = json.load(open(pool_path, encoding="utf-8"))
    sl = _sum_by_unit(by_stage.get("schema_link", {}))
    jd = _sum_by_unit(by_stage.get("judge", {}))
    rp_w = _sum_by_unit(by_stage.get("repair", {}))          # 按写法
    sl_wall = _wall_by_unit(schema_wall_path)
    jd_wall = _wall_by_unit(judge_wall_path)
    repair_const = (float(repair_stage_wall) / max(n, 1)) if repair_stage_wall else 0.0

    e2e_in, e2e_out, e2e_time = [], [], []
    rows = []                     # 逐样本明细(供 --out-per-sample 导出)
    cov = {"schema": 0, "judge": 0, "repair": 0}
    for it in data:
        key = f"{it['db_id']}|||{it.get('question','')}"
        trials = [t for t in it.get("trials", []) if isinstance(t, dict)]
        # generation 每题
        g_in = sum((t.get("prompt_tokens") or t.get("in_tok") or 0) for t in trials)
        g_out = sum((t.get("gen_tokens") or t.get("out_tok") or 0) for t in trials)
        # 每题生成耗时取哪个字段, 取决于池子是不是【隔离模式】跑出来的:
        #   隔离(样本间串行, 成本测量用的口径): wall_sec 不含跨样本排队, 且比
        #     wall_exec_max 多算了题内调度开销 -> 它才是这题真实的端到端墙钟。
        #     这里若仍取 wall_exec_max, 会系统性【低估】模块 2 与端到端总时间 ——
        #     而这正是要上报给官方的核心数字, 宁可保守也不能虚低。
        #   非隔离(样本间并发): wall_sec 含等槽位排队会虚高 -> 退回 wall_exec_max。
        #   老池子无 isolate_samples 标记 -> 按非隔离处理, 与历史行为一致。
        if it.get("isolate_samples"):
            g_time = it.get("wall_sec") or it.get("wall_exec_max") or 0
        else:
            g_time = it.get("wall_exec_max") or it.get("wall_sec") or 0
        # schema link
        s = sl.get(key, {}); s_time = sl_wall.get(key, 0)
        if key in sl or key in sl_wall:
            cov["schema"] += 1
        # judge
        j = jd.get(key, {}); j_time = jd_wall.get(key, 0)
        if key in jd or key in jd_wall:
            cov["judge"] += 1
        # repair: 本题唯一写法的修复开销之和。
        # 注: 池里的 pred_sql 已经是【修复后】的 SQL, 而账本按【修复前】的写法归属,
        # 所以被改写过的候选要先用 keymap 反查回原写法, 否则整体漏计。
        writings = {f"{it['db_id']}|||{_norm_sql(t.get('pred_sql'))}" for t in trials if t.get("pred_sql")}
        r_in = r_out = 0; hit = False
        seen = set()
        for wkey in writings:
            # keymap 一个修复结果可能对应多个原写法; 没命中则按原 key 查(未被改写)
            for k2 in (repair_keymap or {}).get(wkey, [wkey]):
                if k2 in seen:
                    continue
                seen.add(k2)
                if k2 in rp_w:
                    r_in += rp_w[k2]["in"]; r_out += rp_w[k2]["out"]; hit = True
        if hit:
            cov["repair"] += 1
        e2e_in.append(g_in + s.get("in", 0) + j.get("in", 0) + r_in)
        e2e_out.append(g_out + s.get("out", 0) + j.get("out", 0) + r_out)
        e2e_time.append(g_time + s_time + j_time + repair_const)
        rows.append({
            "task_idx": it.get("task_idx"),
            "db_id": it.get("db_id", ""),
            "question": (it.get("question") or "").replace("\n", " "),
            "schema_link_time": round(s_time, 3),
            "schema_link_in": s.get("in", 0),
            "schema_link_out": s.get("out", 0),
            "generation_time": round(g_time, 3),
            "generation_in": g_in,
            "generation_out": g_out,
            # repair 是跨题去重后按【SQL 写法】计费的批处理阶段, 没有每题墙钟 ->
            # 时间用 (阶段总墙钟 / 题数) 摊派, token 则是本题各写法的真实之和。
            "repair_time_amortized": round(repair_const, 3),
            "repair_in": r_in,
            "repair_out": r_out,
            "judge_time": round(j_time, 3),
            "judge_in": j.get("in", 0),
            "judge_out": j.get("out", 0),
            "e2e_time": round(e2e_time[-1], 3),
            "e2e_in": e2e_in[-1],
            "e2e_out": e2e_out[-1],
        })
    return {"n": len(data), "coverage": cov, "_rows": rows,
            "input_tok": _stats(e2e_in), "output_tok": _stats(e2e_out), "time_sec": _stats(e2e_time)}


def load_pool_generation(pool_path, tokcost_path=""):
    """Return per-question and per-trial generation stats from the candidate pool.

    Trial token field names differ by sampler version:
      input  = trials[].in_tok  OR  prompt_tokens   (the submission sampler writes prompt_tokens)
      output = trials[].out_tok  OR  gen_tokens
    If the pool has no input-token field and a `--tokcost` sidecar is given, input is taken there.
    """
    data = json.load(open(pool_path, encoding="utf-8"))

    def _in(t):
        for k in ("in_tok", "prompt_tokens"):
            if t.get(k) is not None:
                return t[k]
        return None

    def _out(t):
        for k in ("out_tok", "gen_tokens"):
            if t.get(k) is not None:
                return t[k]
        return None

    # optional sidecar keyed by task_idx -> list of {in_tok,out_tok,llm_time,turns}
    side = {}
    if tokcost_path and os.path.exists(tokcost_path):
        for it in json.load(open(tokcost_path, encoding="utf-8")):
            side[it.get("task_idx", it.get("db_id"))] = it.get("trials", [])

    q_in, q_out, q_sec, q_turns = [], [], [], []
    q_wall, q_wall_clean, q_wall_timeout = [], [], []
    q_execmax, q_execmax_clean = [], []
    n_timeout = 0
    t_in, t_out, t_sec, t_turns, t_wall = [], [], [], [], []
    have_in = True
    have_wall = True
    have_execmax = True
    for qi, it in enumerate(data):
        trials = [t for t in it.get("trials", []) if isinstance(t, dict)]
        side_tr = side.get(it.get("task_idx", qi), side.get(it.get("db_id"), []))
        w = it.get("wall_sec")
        if w is None:
            have_wall = False
        else:
            q_wall.append(w)
            if it.get("timed_out"):
                q_wall_timeout.append(w); n_timeout += 1
            else:
                q_wall_clean.append(w)
        em = it.get("wall_exec_max")
        if em is None:
            have_execmax = False
        else:
            q_execmax.append(em)
            if not it.get("timed_out"):
                q_execmax_clean.append(em)
        ins, outs, secs, turns = [], [], [], []
        for k, t in enumerate(trials):
            iv = _in(t)
            if iv is None and k < len(side_tr) and isinstance(side_tr[k], dict):
                iv = side_tr[k].get("in_tok")
            if iv is None:
                have_in = False
            ins.append(iv)
            outs.append(_out(t))
            secs.append(t.get("llm_time"))
            turns.append(t.get("turns"))
            if t.get("trial_wall") is not None:
                t_wall.append(t["trial_wall"])
        q_in.append(sum(v or 0 for v in ins))
        q_out.append(sum(v or 0 for v in outs))
        q_sec.append(sum(v or 0 for v in secs))
        q_turns.append(sum(v or 0 for v in turns))
        for a, b, c, d in zip(ins, outs, secs, turns):
            t_in.append(a or 0); t_out.append(b or 0); t_sec.append(c or 0); t_turns.append(d or 0)
    return {
        "n_questions": len(data),
        "have_input_token": have_in,
        "have_wall": have_wall,
        "have_execmax": have_execmax,
        "n_timeout_questions": n_timeout,
        "per_question": {"input_tok": _stats(q_in), "output_tok": _stats(q_out),
                         "compute_sec": _stats(q_sec),
                         "wall_exec_max": _stats(q_execmax),
                         "wall_exec_max_clean": _stats(q_execmax_clean),
                         "wall_sec": _stats(q_wall),
                         "wall_sec_clean": _stats(q_wall_clean),
                         "wall_sec_timeout": _stats(q_wall_timeout),
                         "turns": _stats(q_turns)},
        "per_trial": {"input_tok": _stats(t_in), "output_tok": _stats(t_out),
                      "sec": _stats(t_sec), "exec_wall": _stats(t_wall), "turns": _stats(t_turns)},
    }


def load_detail(detail_path):
    """Group the per-call JSONL by stage, then by unit. Returns {stage: {unit: [rows]}}."""
    by_stage = defaultdict(lambda: defaultdict(list))
    if not detail_path or not os.path.exists(detail_path):
        return by_stage
    with open(detail_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            by_stage[r.get("stage", "?")][r.get("unit")].append(r)
    return by_stage


def module_from_detail(rows_by_unit, n_questions):
    """avg/min/max over units for a closed-LLM stage.

    Reports two granularities:
      - per_call : one API call
      - per_unit : one question (judge) or one writing (repair) = sum of its calls
    Plus an amortised per-question average = totals / n_questions.
    """
    call_in, call_out, call_sec = [], [], []
    unit_in, unit_out, unit_sec, unit_calls = [], [], [], []
    tot_in = tot_out = tot_sec = tot_calls = 0
    for unit, rows in rows_by_unit.items():
        ui = sum(r.get("prompt_tokens") or 0 for r in rows)
        uo = sum(r.get("completion_tokens") or 0 for r in rows)
        us = sum(r.get("seconds") or 0 for r in rows)
        unit_in.append(ui); unit_out.append(uo); unit_sec.append(us); unit_calls.append(len(rows))
        tot_in += ui; tot_out += uo; tot_sec += us; tot_calls += len(rows)
        for r in rows:
            call_in.append(r.get("prompt_tokens")); call_out.append(r.get("completion_tokens"))
            call_sec.append(r.get("seconds"))
    amort = {}
    if n_questions:
        amort = {"input_tok_per_question": round(tot_in / n_questions, 1),
                 "output_tok_per_question": round(tot_out / n_questions, 1),
                 "compute_sec_per_question": round(tot_sec / n_questions, 2),  # 并发调用相加=计算量，非墙钟
                 "calls_per_question": round(tot_calls / n_questions, 2)}
    return {
        "totals": {"calls": tot_calls, "input_tok": tot_in, "output_tok": tot_out,
                   "compute_sec": round(tot_sec, 1), "units": len(rows_by_unit)},
        "per_call": {"input_tok": _stats(call_in), "output_tok": _stats(call_out), "sec": _stats(call_sec)},
        "per_unit": {"input_tok": _stats(unit_in), "output_tok": _stats(unit_out),
                     "compute_sec": _stats(unit_sec), "calls": _stats(unit_calls)},
        "amortised_per_question": amort,
    }


def _fmt(s):
    return f"avg {s['avg']:>10} | min {s['min']:>8} | max {s['max']:>10}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, help="candidate pool JSON (repaired), has trials[]")
    ap.add_argument("--tokcost", default="", help="optional <pool>.tokcost.json sidecar (fallback for input tokens)")
    ap.add_argument("--detail", default="", help="llm_usage_detail.jsonl (repair+judge per-call)")
    ap.add_argument("--stage-wall", default="", help="stage_wall.json {generation,repair,judge} 阶段真实墙钟(秒)")
    ap.add_argument("--judge-wall", default="", help="tournament 隔离模式产物 <cache>_sample_wall.jsonl (每题判别墙钟)")
    ap.add_argument("--schema-wall", default="", help="APEX 隔离模式产物 schema_link_sample_wall.jsonl (每题 schema-link 墙钟)")
    ap.add_argument("--repair-keymap", default="",
                    help="<pool>.repair_keymap.json: 修复后写法->修复前写法, 缺失时被改写的候选 repair token 会漏计")
    ap.add_argument("--out", default="")
    ap.add_argument("--out-per-sample", default="",
                    help="导出逐样本明细 CSV: 每题的四模块时间/token + 端到端时间/token")
    args = ap.parse_args()

    gen = load_pool_generation(args.pool, args.tokcost)
    N = gen["n_questions"]
    by_stage = load_detail(args.detail)
    repair = module_from_detail(by_stage.get("repair", {}), N)
    judge = module_from_detail(by_stage.get("judge", {}), N)
    schema_link = module_from_detail(by_stage.get("schema_link", {}), N)

    stage_wall = {}
    if args.stage_wall and os.path.exists(args.stage_wall):
        try:
            stage_wall = json.load(open(args.stage_wall, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            stage_wall = {}

    report = {"n_questions": N, "stage_wall_sec": stage_wall,
              "generation": gen, "schema_link": schema_link, "repair": repair, "judge": judge}

    # end-to-end per question = generation(per-question) + repair(amortised) + judge(per-question sum)
    def amort_get(m, k):
        return m["amortised_per_question"].get(k, 0) if m["amortised_per_question"] else 0
    e2e = {
        "input_tok_per_question_avg": round(
            gen["per_question"]["input_tok"]["avg"]
            + amort_get(repair, "input_tok_per_question")
            + amort_get(judge, "input_tok_per_question"), 1),
        "output_tok_per_question_avg": round(
            gen["per_question"]["output_tok"]["avg"]
            + amort_get(repair, "output_tok_per_question")
            + amort_get(judge, "output_tok_per_question"), 1),
        "closed_llm_compute_sec_per_question_avg": round(
            amort_get(repair, "compute_sec_per_question") + amort_get(judge, "compute_sec_per_question"), 2),
        "generation_time_sec_per_question": (gen["per_question"]["wall_exec_max_clean"]["avg"] if gen.get("have_execmax") else None),
        "generation_wall_sec_per_question": (gen["per_question"]["wall_sec"]["avg"] if gen["have_wall"] else None),
        "generation_compute_sec_per_question_avg": gen["per_question"]["compute_sec"]["avg"],
        "stage_wall_sec": stage_wall,
        "total_wall_sec": round(sum(float(v or 0) for v in stage_wall.values()), 1) if stage_wall else None,
    }
    report["end_to_end_per_question"] = e2e

    def _load_wall(path):
        xs = []
        if path and os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if r.get("wall_sec") is not None:
                        xs.append(r["wall_sec"])
                except json.JSONDecodeError:
                    pass
        return xs

    report["judge"]["per_sample_wall"] = _stats(_load_wall(args.judge_wall))
    report["schema_link"]["per_sample_wall"] = _stats(_load_wall(args.schema_wall))

    # 端到端每题(按题 join, 真实 avg/min/max) —— max/min 无法由模块相加得到, 必须 join
    # 默认去旁边找 <pool>.repair_keymap.json; 缺失时被修复改写过的候选会漏计 repair token。
    km_path = args.repair_keymap or (args.pool + ".repair_keymap.json")
    repair_keymap = {}
    if os.path.exists(km_path):
        try:
            repair_keymap = json.load(open(km_path, encoding="utf-8"))
            print(f"[keymap] 载入 {len(repair_keymap)} 条修复改写映射 <- {km_path}")
        except Exception:  # noqa: BLE001
            repair_keymap = {}
    else:
        print(f"[keymap] 未找到 {km_path} —— 被修复改写的候选其 repair token 会漏计(coverage.repair 会偏小)")
    e2e_joined = end2end_join(args.pool, by_stage, args.schema_wall, args.judge_wall,
                              stage_wall.get("repair"), N, repair_keymap)
    report["end_to_end_joined"] = e2e_joined

    # ---- pretty print ----
    print("=" * 78)
    print(f"LadderSQL cost report   N={N} questions")
    print("=" * 78)
    print("\n[generation]  local 14B (vLLM). per-question input-token present:", gen["have_input_token"])
    pq = gen["per_question"]
    print("  per-question  input tok :", _fmt(pq["input_tok"]))
    print("  per-question  output tok:", _fmt(pq["output_tok"]))
    if gen.get("have_execmax"):
        print("  per-question  TIME (exec):", _fmt(pq["wall_exec_max_clean"]),
              " ★ per-sample time = max of its 32 trials' exec wall (no slot-queue, NOT confounded by other samples)")
    else:
        print("  per-question  TIME (exec): (not in pool; re-run generation with the current sampler)")
    if gen["have_wall"]:
        print("  per-question  wall(+queue):", _fmt(pq["wall_sec_clean"]),
              " (submit→done, INCLUDES waiting for a free slot; higher under load)")
        nto = gen.get("n_timeout_questions", 0)
        if nto:
            print(f"    ( {nto} questions had a timed-out trial, capped by TRAJ_TIMEOUT; excluded from the 'clean' rows above )")
    print("  per-question  compute-s :", _fmt(pq["compute_sec"]), " (SUM of 32 trials' llm_time = total compute, NOT wall)")
    print("  per-question  turns     :", _fmt(pq["turns"]))
    pt = gen["per_trial"]
    print("  per-trial     input tok :", _fmt(pt["input_tok"]))
    print("  per-trial     output tok:", _fmt(pt["output_tok"]))
    print("  per-trial     exec wall :", _fmt(pt["exec_wall"]), " (one trial's own execution)")
    print("  per-trial     llm-sec   :", _fmt(pt["sec"]), " (sum of its LLM calls)")

    for name, mod in (("schema_link (closed LLM, APEX; unit=question)", schema_link),
                      ("repair (closed LLM, unit=writing)", repair),
                      ("judge (closed LLM, unit=question)", judge)):
        t = mod["totals"]
        print(f"\n[{name}]")
        if t["calls"] == 0:
            print("  (no calls recorded — pass --detail from a real run)")
            if mod.get("per_sample_wall", {}).get("n"):
                print("  per-question WALL   :", _fmt(mod["per_sample_wall"]), " ★ clean per-question time (isolate mode)")
            continue
        print(f"  totals: calls={t['calls']} units={t['units']} "
              f"input={t['input_tok']} output={t['output_tok']} compute_sec={t['compute_sec']}")
        print("  per-call   input tok :", _fmt(mod["per_call"]["input_tok"]))
        print("  per-call   output tok:", _fmt(mod["per_call"]["output_tok"]))
        print("  per-call   latency-s :", _fmt(mod["per_call"]["sec"]), " (one API round-trip; real)")
        print("  per-unit   input tok :", _fmt(mod["per_unit"]["input_tok"]))
        print("  per-unit   output tok:", _fmt(mod["per_unit"]["output_tok"]))
        print("  per-unit   calls     :", _fmt(mod["per_unit"]["calls"]))
        a = mod["amortised_per_question"]
        if a:
            print(f"  amortised/question: in={a['input_tok_per_question']} out={a['output_tok_per_question']} "
                  f"calls={a['calls_per_question']} compute_sec={a['compute_sec_per_question']} (compute, NOT wall)")
        if report["judge"].get("per_sample_wall", {}).get("n") and name.startswith("judge"):
            print("  per-question WALL   :", _fmt(report["judge"]["per_sample_wall"]),
                  " ★ clean per-question judge time (isolate mode)")
        if mod.get("per_sample_wall", {}).get("n") and name.startswith("schema_link"):
            print("  per-question WALL   :", _fmt(mod["per_sample_wall"]),
                  " ★ clean per-question schema-link time (isolate mode)")

    print("\n[end-to-end per question, averages]")
    print(f"  input  tok/question : {e2e['input_tok_per_question_avg']}")
    print(f"  output tok/question : {e2e['output_tok_per_question_avg']}")
    gt = e2e['generation_time_sec_per_question']
    if gt is not None:
        print(f"  generation per-sample time : {gt} (max of 32 trials' exec wall; conflict-free)")
    gw = e2e['generation_wall_sec_per_question']
    if gw is not None:
        print(f"  generation submit→done wall : {gw} (includes slot-queue; load-dependent)")
    print(f"  generation compute-sec/question : {e2e['generation_compute_sec_per_question_avg']} (total compute; NOT wall)")
    print(f"  closed-LLM compute-sec/question : {e2e['closed_llm_compute_sec_per_question_avg']} (sum of concurrent calls; NOT wall)")

    if stage_wall:
        print("\n[stage wall clock, REAL — the only correct per-stage time]")
        for k in ("schema_link", "generation", "repair", "judge"):
            if k in stage_wall:
                sec = float(stage_wall[k] or 0)
                print(f"  {k:11s}: {sec:8.0f} s total   ({sec/max(N,1):.2f} s/question amortised)")
        print(f"  TOTAL      : {e2e['total_wall_sec']:8.0f} s   ({e2e['total_wall_sec']/max(N,1):.2f} s/question)")
    else:
        print("\n[stage wall clock] (pass --stage-wall stage_wall.json from run_test.sh for real per-stage wall)")

    # ===== 四模块干净均值表 (你要的: 每模块 平均耗时/输入token/输出token) =====
    # 均为【总量 ÷ 题数】或【每题干净墙钟】, 对平均而言与去重无关, 完全干净。
    def _mod_row(label, avg_t, avg_in, avg_out, note=""):
        print(f"  {label:16s} time/q={avg_t:>8}  in_tok/q={avg_in:>9}  out_tok/q={avg_out:>8}  {note}")

    def _amt(mod, k):
        return mod["amortised_per_question"].get(k, 0) if mod.get("amortised_per_question") else 0

    def _sw(k):
        return round(float(stage_wall[k]) / max(N, 1), 2) if stage_wall.get(k) else None

    print("\n[per-module averages per question — clean, module-level]")
    print("  (time = stage wall / N; tokens = module total / N; 均为平均值, 与去重无关, 干净)")
    # schema link
    _mod_row("schema_link", _sw("schema_link") if _sw("schema_link") is not None else "n/a",
             _amt(schema_link, "input_tok_per_question"), _amt(schema_link, "output_tok_per_question"),
             "(APEX glm)")
    # generation
    _mod_row("sql_generation",
             (_sw("generation") if _sw("generation") is not None else e2e.get("generation_time_sec_per_question") or "n/a"),
             gen["per_question"]["input_tok"]["avg"], gen["per_question"]["output_tok"]["avg"],
             "(local 14B)")
    # repair
    _mod_row("sql_repair", _sw("repair") if _sw("repair") is not None else "n/a",
             _amt(repair, "input_tok_per_question"), _amt(repair, "output_tok_per_question"),
             "(glm)")
    # selection = tournament judge (+ final pick, no LLM)
    _mod_row("sql_selection", _sw("judge") if _sw("judge") is not None else "n/a",
             _amt(judge, "input_tok_per_question"), _amt(judge, "output_tok_per_question"),
             "(tournament judge, glm)")
    print("  → max/min 看各模块上面的 per-call / per-question 行; 平均可相加, max/min 不可相加。")

    # ===== 端到端每题(按题 join) —— 真实 avg/min/max, 不是模块相加 =====
    ej = e2e_joined
    cov = ej["coverage"]
    print("\n[end-to-end per question, JOINED — real avg/min/max (不是模块 max 相加)]")
    print(f"  参与题数={ej['n']}  覆盖: schema={cov['schema']} judge={cov['judge']} repair={cov['repair']}")
    print("  input  tok/q :", _fmt(ej["input_tok"]))
    print("  output tok/q :", _fmt(ej["output_tok"]))
    print("  time    sec/q:", _fmt(ej["time_sec"]), " (schema+gen+judge 每题干净墙钟 + repair 摄派常量)")
    print("  注: token 四模块按题相加(repair 按写法归属); time 的 repair 项为批阶段摄派(不影响 max/min 排序)。")

    # ===== 逐样本明细 CSV —— “每个样本的时间和 token”的交付物 =====
    rows = ej.pop("_rows", [])          # 不让 1534 行明细灌进汇总 JSON
    if args.out_per_sample and rows:
        import csv
        d = os.path.dirname(os.path.abspath(args.out_per_sample))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(args.out_per_sample, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[save] 逐样本明细 {len(rows)} 行 -> {args.out_per_sample}")

    if args.out:
        json.dump(report, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"\n[save] {args.out}")


if __name__ == "__main__":
    main()
