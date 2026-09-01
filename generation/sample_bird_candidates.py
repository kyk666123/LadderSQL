#!/usr/bin/env python3
"""BIRD inference-time candidate generator (dev / test): for each question, sample N
multi-turn ReAct trajectories with the trained checkpoint and persist the final SQL of
each (+ an optional 0/1 reward and the full intermediate trajectory).

This file = the inference version of sample_selection/sample_bird_redundant_with_sql.py
(the pre-training disagreement sampler). The logic is identical except for the two points
below (prompt / schema / ReAct loop / truncation / streaming resume / hard timeout are all
unchanged, so training and inference stay consistent):

  1. [--no-gold] no-gold mode: when the question file has no gold SQL (BIRD test is exactly
     this case), reward computation is skipped and reward is written as 0.0 (the downstream
     repair / selection stages are reward-blind throughout and do not need it).
  2. Gold-field tolerance: query / SQL / missing are all handled.

Output format (the input contract of downstream sql_repair / inference_selection):
  [{db_id, question, evidence, gold_sql?, query?, reward_list, trials:[{pred_sql, reward, ...}]}]

Environment variables (same names and meanings as in training-time sampling):
  SERVED_MODEL_NAME / TEMPERATURE / MAX_TOKENS / MAX_MODEL_LEN / LLM_TIMEOUT /
  SCHEMA_CACHE / PER_GPU / TRAJ_TIMEOUT / MAX_TURNS_AGENT / SAVE_TRAJECTORY / NO_GOLD

Usage:
    python sample_bird_candidates.py --parquet <questions.parquet> --db-dir <local db dir> \
        --output <candidates.json> --n-trials 32 --num-gpus 8 [--no-gold]
Resuming: re-running skips the already-finished samples and automatically re-runs the ones
marked timed_out.
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
import logging
import signal
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from typing import Any, Dict, List, Optional

import pandas as pd

# ---- make original_agent / reward_func importable (point at the training/ directory) ----
_AGENT_DIR = Path(__file__).resolve().parent.parent / "training"  # provides original_agent + reward_func
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

# ---- inference needs no training framework: if agentlightning cannot be imported, install a tiny stub ----
#      (only the training-side LitAgent in ReAct_Agent.py uses it; the inference Agent is plain LangGraph)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _agl_stub  # noqa: E402
_AGL_MODE = _agl_stub.install()

# Row-level observation truncation: with REACT_EXEC_MAX_ROWS>0 use RowTruncAgent (truncate whole
# rows), otherwise use the base Agent.
if int(os.environ.get("REACT_EXEC_MAX_ROWS", "0")) > 0:
    from original_agent.ReAct_Agent_rowtrunc import RowTruncAgent as Agent  # type: ignore
else:
    from original_agent.ReAct_Agent import Agent  # type: ignore
from original_agent.prompt_redundant_schema import REACT_SQL_BIRD_REDUNDANT_PROMPT  # type: ignore
from reward_func.reward import binary_reward_bird  # type: ignore

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("sample_bird_redundant_with_sql")
logger.setLevel(logging.INFO)

# ============================ default configuration ============================
# Inference defaults to dev; for the test submission, override with
# --parquet/--db-dir/--output/SCHEMA_CACHE to point at the test set.
DEFAULT_PARQUET = "/path/to/nl2sql_dataset/bird/dev_20240627/dev.parquet"
DEFAULT_DB_DIR = "/root/local_bird_db/dev_databases"   # {db_id}/{db_id}.sqlite (must be a local disk)
DEFAULT_OUTPUT = "/path/to/candidate_pools/bird_dev_32trials_candidates.json"
DEFAULT_SCHEMA_CACHE = (
    "/path/to/schema_link_results/bird_old_dev/"
    "schema_cache_old_dev_rendered.json"
)

N_TRIALS = 32               # candidates per question n (main result uses 32)
NUM_GPUS = 8                # number of vLLM instances (one per GPU)
BASE_PORT = 9000            # vLLM base port -> 9000..9007
MAX_TURNS = int(os.environ.get("MAX_TURNS_AGENT", "5"))  # ReAct turns at inference (main result 5; training 8)
EXECUTION_TRUNCATE = int(os.environ.get("EXECUTION_TRUNCATE", "2048"))
RECURSION_LIMIT = 100
SAVE_EVERY = 16

# ---- runtime configuration controlled by environment variables (set at the script level, to
#      distinguish the three model kinds) ----
SERVED_MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", "qwen2.5-coder")
TEMPERATURE = float(os.environ.get("TEMPERATURE", "1.1"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "2048"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "16384"))
# ReAct early-stop prompt-token budget = context window - single-generation cap.
# Overrides ReAct_Agent's default 12288 (old training config), aligning it with the
# max_model_len / max_tokens of this sampling run.
os.environ.setdefault("REACT_PROMPT_TOKEN_BUDGET", str(max(1024, MAX_MODEL_LEN - MAX_TOKENS)))
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "45"))
TRAJ_TIMEOUT = int(os.environ.get("TRAJ_TIMEOUT", "120"))
PARALLEL_PER_GPU = int(os.environ.get("PER_GPU", "10"))
_ENABLE_THINKING_ENV = os.environ.get("ENABLE_THINKING")  # "true"/"false"/None
_THINKING_BUDGET_ENV = os.environ.get("THINKING_BUDGET")  # "2048"/None (only effective when thinking=true)
# Whether to additionally save the full ReAct intermediate process (message sequence) of every
# trajectory. Off by default, turned on with SAVE_TRAJECTORY=1.
_SAVE_TRAJ = os.environ.get("SAVE_TRAJECTORY", "0").lower() in ("1", "true", "yes")

# No-gold mode: BIRD test has no gold SQL, so reward computation is skipped (downstream is
# reward-blind throughout). Enable with NO_GOLD=1 or --no-gold; it also degrades automatically
# when the question file has no gold to begin with.
NO_GOLD = os.environ.get("NO_GOLD", "0").lower() in ("1", "true", "yes")

# Isolate mode: samples run sequentially (one question finishes before the next starts), while the
# 32 trials within a sample still run concurrently. Used to obtain a completely clean per-sample
# wall clock with no cross-sample queueing (cost: lower card utilisation, longer total time).
ISOLATE_SAMPLES = os.environ.get("ISOLATE_SAMPLES", "0").lower() in ("1", "true", "yes")

# ---- inject into the Agent's internal LLM calls: per-request timeout + max_tokens override + thinking switch ----
import original_agent.ReAct_Agent as _react_mod  # type: ignore  # noqa: E402
_ORIG_INIT_CHAT_MODEL = _react_mod.init_chat_model


def _init_chat_model_patched(*a, **k):
    k.setdefault("timeout", LLM_TIMEOUT)
    k["max_retries"] = 1  # was 3; when a connection hangs it multiplies the wait by 3
    k["max_tokens"] = MAX_TOKENS
    if _ENABLE_THINKING_ENV is not None:
        thinking_on = _ENABLE_THINKING_ENV.lower() == "true"
        extra_body = {"chat_template_kwargs": {"enable_thinking": thinking_on}}
        if thinking_on and _THINKING_BUDGET_ENV:
            extra_body["thinking_token_budget"] = int(_THINKING_BUDGET_ENV)
        k["extra_body"] = extra_body
    return _ORIG_INIT_CHAT_MODEL(*a, **k)


_react_mod.init_chat_model = _init_chat_model_patched


# ============================ redundant schema cache ============================
def _make_key(db_id: str, question: str) -> str:
    # exactly the same as the cache branch of ReAct_Agent.get_table_info: raw question, three pipes
    return f"{db_id}|||{question}"


def _load_schema_cache(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(
            f"redundant schema cache does not exist: {path} "
            f"(should be the output of build_rendered_cache_train.py, key=db_id|||question)"
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"[schema] loaded {len(data)} redundant schema cache entries: {path}")
    return data


_SCHEMA_CACHE_PATH = Path(os.environ.get("SCHEMA_CACHE", DEFAULT_SCHEMA_CACHE))
# module-level load (before fork), inherited by child processes
_SCHEMA_MAP: Dict[str, str] = _load_schema_cache(_SCHEMA_CACHE_PATH)
# whitespace-normalised fallback index (to absorb the rare whitespace differences between the
# parquet question and the APEX question)
_SCHEMA_MAP_NORM: Dict[str, str] = {
    f"{k.split('|||', 1)[0]}|||{' '.join(k.split('|||', 1)[1].split())}": v
    for k, v in _SCHEMA_MAP.items() if "|||" in k
}


def build_endpoints(num_gpus: int) -> List[str]:
    return [f"http://localhost:{BASE_PORT + i}/v1" for i in range(num_gpus)]


def get_db_path(db_dir: str, db_id: str) -> str:
    return os.path.join(db_dir, db_id, f"{db_id}.sqlite")


def get_redundant_schema(db_id: str, question: str) -> str:
    """Fetch the pre-rendered redundant candidate schema text (by db_id+question), untruncated."""
    s = _SCHEMA_MAP.get(_make_key(db_id, question))
    if s is None:
        s = _SCHEMA_MAP_NORM.get(f"{db_id}|||{' '.join(question.strip().split())}")
    if s is None:
        logger.warning(f"[schema] redundant schema miss: {db_id} | {question[:80]}")
        return "No candidate schema available."
    return s


def _serialize_trajectory(messages) -> List[Dict[str, Any]]:
    """Serialise the ReAct message sequence (HumanMessage/AIMessage/...) into [{role, content}] for review."""
    out: List[Dict[str, Any]] = []
    for m in messages or []:
        role = getattr(m, "type", None) or m.__class__.__name__
        content = getattr(m, "content", m)
        if not isinstance(content, str):
            content = str(content)
        out.append({"role": role, "content": content})
    return out


def run_one_trial(task: Dict[str, Any], endpoint: str, db_dir: str) -> Optional[Dict[str, Any]]:
    """Single rollout: run the redundant-schema agent to get a SQL, then compute the 0/1 reward.
    In no-gold mode (NO_GOLD, or no gold in task) the reward is fixed at 0.0. Returns None on failure."""
    db_id = task["db_id"]
    question = task["question"]
    ground_truth = task.get("query") or task.get("SQL") or ""
    evidence = task.get("evidence", "") or ""

    db_file = get_db_path(db_dir, db_id)
    if not os.path.exists(db_file):
        logger.error(f"[trial] database does not exist: {db_file}")
        return None

    db_uri = f"sqlite:///{db_file}"
    db_schema = get_redundant_schema(db_id, question)   # redundant candidate schema

    _t_trial = time.time()   # when this trial actually starts executing in the worker (excludes slot-wait queueing)
    try:
        agent_obj = Agent(
            db_path=db_uri,
            db_schema=db_schema,
            max_turns=MAX_TURNS,
            endpoint=endpoint,
            verl_replacement={"model": SERVED_MODEL_NAME, "temperature": TEMPERATURE},
            execution_truncate=EXECUTION_TRUNCATE,
            prompt_template=REACT_SQL_BIRD_REDUNDANT_PROMPT,
            evidence=evidence,
        )
        result = agent_obj.graph().invoke({"question": question}, {"recursion_limit": RECURSION_LIMIT})
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[trial] db_id={db_id} agent execution failed: {e}")
        return None

    pred_sql = result.get("query", "") if isinstance(result, dict) else ""
    if NO_GOLD or not ground_truth:
        reward = 0.0                      # no gold: no scoring, only produce candidates
    else:
        try:
            reward = binary_reward_bird(pred_sql, ground_truth, db_file, raise_on_error=False)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[trial] db_id={db_id} reward computation failed: {e}")
            return None

    # aggregate this trajectory's thinking time / length (from the Agent's per-turn _turn_history;
    # for non-thinking models reasoning=0)
    hist = getattr(agent_obj, "_turn_history", []) or []
    llm_time = sum(float(h.get("llm_time") or 0.0) for h in hist)
    gen_tokens = sum(int(h.get("gen_tokens") or 0) for h in hist)
    prompt_tokens = sum(int(h.get("prompt_tokens") or 0) for h in hist)
    total_tokens = sum(int(h.get("total_tokens") or 0) for h in hist)
    rsn_chars = sum(int(h.get("reasoning_chars") or 0) for h in hist)
    _rtoks = [h.get("reasoning_tokens") for h in hist if h.get("reasoning_tokens") is not None]
    rsn_tokens = int(sum(_rtoks)) if _rtoks else None
    trajectory = (_serialize_trajectory(result.get("messages", []))
                  if (_SAVE_TRAJ and isinstance(result, dict)) else [])
    return {
        "pred_sql": pred_sql,          # core of this script: save the SQL the model actually generated
        "trajectory": trajectory,      # full ReAct intermediate trajectory for that SQL (non-empty when SAVE_TRAJECTORY=1)
        "reward": float(reward),       # the 0/1 reward of that SQL
        "turns": len(hist),
        "llm_time": round(llm_time, 3),
        "trial_wall": round(time.time() - _t_trial, 3),  # this trial's execution wall clock (excludes slot-wait); per question take the max
        "gen_tokens": gen_tokens,
        "prompt_tokens": prompt_tokens,
        "total_tokens": total_tokens,
        "reasoning_tokens": rsn_tokens,
        "reasoning_chars": rsn_chars,
    }


def _run_trial_task(packed):
    """Run a single rollout in a child process: impose a hard timeout via SIGALRM (the process can be
    killed) so it can never hang."""
    task, endpoint, db_dir, traj_timeout = packed

    def _on_alarm(signum, frame):
        raise TimeoutError(f"trajectory exceeded {traj_timeout}s")

    old = signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(int(traj_timeout))
    try:
        return run_one_trial(task, endpoint, db_dir)
    except BaseException as e:  # noqa: BLE001 (includes TimeoutError)
        logger.warning(f"[trial] db={task.get('db_id')} timeout/exception, recorded as None: {e}")
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# ============================ persisting results ============================
def load_existing(output_file: str) -> List[Dict]:
    if not os.path.exists(output_file):
        return []
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"result file is corrupt ({e}), starting from scratch.")
        return []
    if data and any("task_idx" not in r for r in data):
        logger.warning("detected old-format results (no task_idx), starting from scratch (please back up the old file).")
        return []
    print(f"[INFO] loaded {len(data)} historical results: {output_file}")
    return data


def save_results(output_file: str, results: List[Dict]) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    tmp = output_file + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    os.replace(tmp, output_file)


def summarize(results: List[Dict]) -> None:
    all_one = all_zero = mixed = timed = invalid = 0
    for s in results:
        if s.get("timed_out"):
            timed += 1
            continue
        rl = s.get("reward_list", [])
        if not rl or any(r is None for r in rl):
            invalid += 1
            continue
        if all(r == 1.0 for r in rl):
            all_one += 1
        elif all(r == 0.0 for r in rl):
            all_zero += 1
        else:
            mixed += 1
    total = len(results)
    print("\n" + "=" * 70)
    print(f"sampling stats (total={total})")
    print(f"  mixed (variance>0): {mixed}")
    print(f"  all_1 (all correct): {all_one}")
    print(f"  all_0 (all wrong)  : {all_zero}")
    print(f"  timed_out          : {timed}")
    print(f"  invalid (has None) : {invalid}")
    print("=" * 70)


# ============================ main flow ============================
def process(args) -> None:
    import multiprocessing as mp
    from collections import deque, defaultdict

    tasks = pd.read_parquet(args.parquet).to_dict(orient="records")
    # tolerate different gold-SQL column names across parquets: old dev uses "query", clean dev
    # (dev_20251106) uses "SQL"; BIRD test has no gold at all -> fill an empty string and skip
    # scoring via NO_GOLD.
    for _t in tasks:
        if "query" not in _t:
            _t["query"] = _t.get("SQL", "") or ""
    total = len(tasks)
    endpoints = build_endpoints(args.num_gpus)
    max_inflight = max(1, args.num_gpus * args.per_gpu)
    print(f"[INFO] model: served={SERVED_MODEL_NAME} thinking={_ENABLE_THINKING_ENV} "
          f"think_budget={_THINKING_BUDGET_ENV} max_tokens={MAX_TOKENS} temp={TEMPERATURE} "
          f"llm_timeout={LLM_TIMEOUT}s")
    print(f"[INFO] max_model_len={MAX_MODEL_LEN} early-stop prompt budget (REACT_PROMPT_TOKEN_BUDGET)="
          f"{os.environ.get('REACT_PROMPT_TOKEN_BUDGET')}")
    print(f"[INFO] schema cache: {_SCHEMA_CACHE_PATH} ({len(_SCHEMA_MAP)} entries)")
    print(f"[INFO] total samples: {total}; {args.n_trials} trials per sample -> {len(endpoints)} endpoints; "
          f"concurrent rollouts: {max_inflight}")
    print(f"[INFO] max_turns={MAX_TURNS} per-trajectory hard timeout={args.traj_timeout}s")
    print(f"[INFO] saving the actually generated SQL + each 0/1 reward -> {args.output}")

    results = load_existing(args.output)
    done_idx = {r["task_idx"] for r in results if "task_idx" in r}
    pending = [i for i in range(total) if i not in done_idx]
    if not pending:
        print(f"[INFO] all {total} samples already finished.")
        summarize(results)
        return
    print(f"[INFO] to sample {len(pending)} / {total} (finished {len(done_idx)})")

    ctx = mp.get_context("fork")
    executor = ProcessPoolExecutor(max_workers=max_inflight, mp_context=ctx)

    task_queue = deque((si, ti) for si in pending for ti in range(args.n_trials))
    futures = {}                       # fut -> (si, ti)
    per_sample = defaultdict(lambda: [None] * args.n_trials)
    remaining = {si: args.n_trials for si in pending}
    submit_ts = {}                     # si -> when the first trial was submitted (for the real per-question wall clock)
    counter = 0

    def submit_next():
        nonlocal counter
        si, ti = task_queue.popleft()
        if si not in submit_ts:
            submit_ts[si] = time.time()   # first trial of this question is enqueued
        ep = endpoints[counter % len(endpoints)]
        counter += 1
        fut = executor.submit(_run_trial_task, (tasks[si], ep, args.db_dir, args.traj_timeout))
        futures[fut] = (si, ti)

    def finalize(si):
        rl = per_sample.pop(si)   # list[dict|None]
        task = tasks[si]
        # the question's real wall clock: first trial enqueued -> last trial returned (end-to-end latency under concurrency)
        wall_sec = round(time.time() - submit_ts.pop(si, time.time()), 3)
        reward_list = [(d["reward"] if isinstance(d, dict) else None) for d in rl]
        timed = any(d is None for d in rl)
        # per-trajectory save: pred_sql + 0/1 reward (+ light stats)
        trials = []
        for ti, d in enumerate(rl):
            if isinstance(d, dict):
                trials.append({
                    "trial": ti,
                    "pred_sql": d.get("pred_sql", ""),
                    "reward": d["reward"],
                    "turns": d["turns"],
                    "llm_time": d["llm_time"],
                    "trial_wall": d.get("trial_wall"),
                    "gen_tokens": d["gen_tokens"],
                    "prompt_tokens": d.get("prompt_tokens"),
                    "total_tokens": d.get("total_tokens"),
                    "trajectory": d.get("trajectory", []),   # intermediate trajectory
                })
            else:
                trials.append({
                    "trial": ti,
                    "pred_sql": None,      # None = this trial timed out / errored, no generated SQL
                    "reward": None,
                    "turns": None,
                    "llm_time": None,
                    "gen_tokens": None,
                    "prompt_tokens": None,
                    "total_tokens": None,
                    "trajectory": None,
                })
        valid_d = [d for d in rl if isinstance(d, dict)]
        think_stats = {
            "turns": [d["turns"] for d in valid_d],
            "llm_time": [d["llm_time"] for d in valid_d],
            "gen_tokens": [d["gen_tokens"] for d in valid_d],
            "prompt_tokens": [d.get("prompt_tokens") for d in valid_d],
            "total_tokens": [d.get("total_tokens") for d in valid_d],
            "reasoning_tokens": [d["reasoning_tokens"] for d in valid_d],
            "reasoning_chars": [d["reasoning_chars"] for d in valid_d],
        }
        # two per-question timing measures:
        #   wall_sec      = first trial enqueued -> last returned (includes slot-wait, affected by cross-sample scheduling)
        #   wall_exec_max = the max of the 32 trials' execution wall clocks (excludes slot-wait; only spans this
        #                   question's own trials)
        #                   -> this is the per-sample time that "does not conflict with other samples"
        _tw = [d["trial_wall"] for d in valid_d if d.get("trial_wall") is not None]
        wall_exec_max = round(max(_tw), 3) if _tw else None
        results.append({
            "task_idx": si,
            "db_id": task["db_id"],
            "question": task["question"],
            "gold_sql": task.get("query", ""),   # gold SQL ("" when there is no gold)
            "query": task.get("query", ""),      # compatible with filter_variance_zero.py / downstream eval
            "evidence": task.get("evidence", ""),
            "reward_list": reward_list,     # compatible with the original filtering / statistics
            "trials": trials,               # each: pred_sql + 0/1 reward
            "timed_out": timed,
            "wall_sec": wall_sec,           # per-question enqueue->done wall clock (includes slot-wait, affected by concurrency)
            "wall_exec_max": wall_exec_max, # per-question max over the 32 trials' execution wall clocks (excludes slot-wait, no cross-sample conflict)
            # the cost report uses this flag to decide which of the two wall clocks above to use (see tools/profile_report.py):
            #   in isolate mode samples do not overlap -> wall_sec is already clean, and it includes the
            #   intra-question scheduling overhead that wall_exec_max omits, so it is the true end-to-end
            #   time of this question. Missing this flag systematically underestimates module 2.
            "isolate_samples": ISOLATE_SAMPLES,
            "think_stats": think_stats,
        })
        valid = [r for r in reward_list if r is not None]
        avg = (sum(valid) / len(valid)) if valid else 0.0
        return timed, avg

    # ============ isolate mode: samples processed sequentially, 32 concurrent within a sample -> per-sample time is fully clean ============
    if ISOLATE_SAMPLES:
        iso_conc = min(args.n_trials, max_inflight)
        print(f"[INFO] ISOLATE_SAMPLES=1: samples run sequentially, {iso_conc} concurrent per sample (32 SQL concurrent within a sample),"
              f" samples do not overlap -> wall_sec excludes cross-sample queueing", flush=True)
        completed = 0; since_save = 0; t_start = time.time()
        for si in pending:
            submit_ts[si] = time.time()          # sample start (no other sample on the GPU at this moment)
            tq = deque(range(args.n_trials)); inflight = {}
            def _sub_iso():
                ti = tq.popleft(); ep = endpoints[ti % len(endpoints)]
                f = executor.submit(_run_trial_task, (tasks[si], ep, args.db_dir, args.traj_timeout))
                inflight[f] = ti
            while tq and len(inflight) < iso_conc:
                _sub_iso()
            while inflight:
                done, _ = wait(list(inflight.keys()), timeout=30, return_when=FIRST_COMPLETED)
                if not done:
                    if time.time() - submit_ts[si] > args.traj_timeout + 120:
                        logger.error(f"[stuck] idx={si} unresponsive for a long time, mark the rest as timed out")
                        for f in list(inflight.keys()):
                            per_sample[si][inflight.pop(f)] = None
                        break
                    continue
                for f in done:
                    ti = inflight.pop(f)
                    try:
                        per_sample[si][ti] = f.result()
                    except Exception as e:  # noqa: BLE001
                        logger.warning(f"[trial] future exception: {e}"); per_sample[si][ti] = None
                    if tq:
                        _sub_iso()
            timed, avg = finalize(si); completed += 1; since_save += 1
            rate = completed / max(1e-9, time.time() - t_start) * 60
            print(f"[{len(results)}/{total}] idx={si} db={tasks[si]['db_id']} "
                  f"avg={avg:.3f} timed_out={timed} ~{rate:.1f}/min", flush=True)
            if since_save >= args.save_every:
                results.sort(key=lambda r: r.get("task_idx", 0)); save_results(args.output, results); since_save = 0
        results.sort(key=lambda r: r.get("task_idx", 0)); save_results(args.output, results)
        executor.shutdown(wait=False, cancel_futures=True)
        summarize(results)
        print(f"\n[OK] isolate-mode sampling done (per-sample time is clean): {args.output}", flush=True)
        return

    while task_queue and len(futures) < max_inflight:
        submit_next()

    completed = 0
    since_save = 0
    t_start = time.time()
    last_progress = time.time()
    try:
        while futures:
            done, _ = wait(list(futures.keys()), timeout=30, return_when=FIRST_COMPLETED)
            if not done:
                if time.time() - last_progress > args.traj_timeout + 120:
                    logger.error(f"[stuck] {len(futures)} workers unresponsive for a long time, mark timed out and rebuild the process pool.")
                    for f, (si, ti) in list(futures.items()):
                        per_sample[si][ti] = None
                        remaining[si] -= 1
                        if remaining[si] == 0:
                            finalize(si); completed += 1; since_save += 1
                    executor.shutdown(wait=False, cancel_futures=True)
                    futures.clear()
                    executor = ProcessPoolExecutor(max_workers=max_inflight, mp_context=ctx)
                    while task_queue and len(futures) < max_inflight:
                        submit_next()
                    last_progress = time.time()
                continue
            last_progress = time.time()
            for fut in done:
                si, ti = futures.pop(fut)
                try:
                    per_sample[si][ti] = fut.result()
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"[trial] future exception: {e}")
                    per_sample[si][ti] = None
                remaining[si] -= 1
                if task_queue:
                    submit_next()
                if remaining[si] == 0:
                    timed, avg = finalize(si)
                    completed += 1
                    since_save += 1
                    rate = completed / max(1e-9, time.time() - t_start) * 60
                    print(f"[{len(results)}/{total}] idx={si} db={tasks[si]['db_id']} "
                          f"avg={avg:.3f} timed_out={timed} ~{rate:.1f}/min")
            if since_save >= args.save_every:
                results.sort(key=lambda r: r.get("task_idx", 0))
                save_results(args.output, results)
                since_save = 0
    finally:
        results.sort(key=lambda r: r.get("task_idx", 0))
        save_results(args.output, results)
        executor.shutdown(wait=False, cancel_futures=True)

    summarize(results)
    print(f"\n[OK] sampling done, the actually generated SQL + 0/1 reward of every trajectory saved: {args.output}")


def main() -> None:
    p = argparse.ArgumentParser(description="BIRD inference-time candidate generation (save the actual SQL + optional 0/1 reward)")
    p.add_argument("--parquet", default=DEFAULT_PARQUET)
    p.add_argument("--db-dir", default=DEFAULT_DB_DIR)
    p.add_argument("--output", default=os.environ.get("OUTPUT", DEFAULT_OUTPUT))
    p.add_argument("--n-trials", type=int, default=int(os.environ.get("N_TRIALS", N_TRIALS)))
    p.add_argument("--num-gpus", type=int, default=NUM_GPUS)
    p.add_argument("--no-gold", action="store_true",
                   help="no-gold mode (BIRD test): do not compute reward, only produce candidate SQL")
    p.add_argument("--per-gpu", type=int, default=PARALLEL_PER_GPU,
                   help="concurrent rollouts per vLLM endpoint, total concurrency = num_gpus*per_gpu")
    p.add_argument("--traj-timeout", type=int, default=TRAJ_TIMEOUT,
                   help="hard timeout in seconds for a single rollout (ReAct trajectory)")
    p.add_argument("--save-every", type=int, default=SAVE_EVERY,
                   help="persist to disk every N finished samples (streaming)")
    args = p.parse_args()
    if args.no_gold:
        global NO_GOLD
        NO_GOLD = True                 # forked child processes inherit this global value
        print("[INFO] no-gold mode: skipping reward computation")
    process(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[WARN] interrupted; results were saved safely and the run can be resumed.")
