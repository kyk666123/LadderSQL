"""8采样 vs 32采样 多数投票对比 (同一套现场执行聚类方法, 公平对比)。

多数投票: 把 N 个候选 SQL 按【执行结果签名】聚类, 取最大簇;
该簇是否正确 = 簇内成员 reward (同结果同 reward)。
- 执行报错 -> 签名 'ERR'
- 成功但空结果集 -> 有效签名 (空也是合法答案)
tie-break: 簇大小相同时取【最先出现】的簇 (与 gated_common 一致)。
"""
from __future__ import annotations

import ast
import json
import os
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import bird_common as B

F8 = "/path/to/sampling_outputs/ckpt_20260711_235128_step192_bird_old_dev_maxturns5_8trials_traj_with_sql.json"
F32 = "/path/to/sampling_outputs/ckpt_20260711_235128_step192_bird_old_dev_maxturns5_32trials_traj_with_sql.json"
SIG_CACHE_PATH = f"{B.CACHE_DIR}/exec_sig_cache_8vs32.json"

_SIG: dict = {}


def load_sig_cache():
    global _SIG
    if os.path.exists(SIG_CACHE_PATH):
        with open(SIG_CACHE_PATH) as f:
            _SIG = json.load(f)


def save_sig_cache():
    tmp = SIG_CACHE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_SIG, f)
    os.replace(tmp, SIG_CACHE_PATH)


def norm(sql: str) -> str:
    return " ".join((sql or "").split())


def _exec_sig_direct(db_id: str, sql: str, max_steps: int = 40_000_000) -> str:
    """直接执行单条 SQL 返回签名; 带步数上限, 防病态查询挂死。"""
    conn = None
    try:
        conn = sqlite3.connect(f"file:{B.db_path_of(db_id)}?mode=ro", uri=True, timeout=30)
        conn.text_factory = lambda b: b.decode(errors="ignore")
        steps = {"n": 0}

        def _handler():
            steps["n"] += 1
            return 1 if steps["n"] > (max_steps // 1000) else 0
        conn.set_progress_handler(_handler, 1000)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        return str(B.hash_result(rows))
    except Exception:
        return "ERR"
    finally:
        if conn:
            conn.close()


_LOCK = threading.Lock()


def precompute_sigs(pairs, workers=32):
    """pairs: set of (db_id, norm_sql). 并发填充 _SIG。"""
    todo = [(db, s) for (db, s) in pairs if f"{db}|||{s}" not in _SIG]
    print(f"[precompute] 总唯一={len(pairs)} 待算={len(todo)} workers={workers}", flush=True)
    done = {"n": 0}
    t0 = time.time()

    def work(item):
        db, s = item
        sig = _exec_sig_direct(db, s)
        with _LOCK:
            _SIG[f"{db}|||{s}"] = sig
            done["n"] += 1
            if done["n"] % 2000 == 0:
                print(f"  {done['n']}/{len(todo)}  ({time.time()-t0:.0f}s)", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, todo))
    print(f"[precompute] 完成 ({time.time()-t0:.0f}s)", flush=True)


def sig_of(db_id: str, sql: str) -> str:
    if not sql or not sql.strip():
        return "EMPTY_SQL"
    return _SIG.get(f"{db_id}|||{norm(sql)}", "ERR")


def _r1(x):
    try:
        return 1 if float(x) == 1.0 else 0
    except (TypeError, ValueError):
        return 0


def parse_rewards(item, n_expected):
    rl = item.get("reward_list")
    if isinstance(rl, str):
        try:
            rl = ast.literal_eval(rl)
        except Exception:
            rl = None
    if rl and len(rl) >= 1:
        return [_r1(x) for x in rl]
    # 回退到 trials.reward
    tr = item.get("trials")
    if isinstance(tr, str):
        tr = ast.literal_eval(tr)
    return [_r1(t.get("reward")) for t in tr]


def parse_trials(item):
    tr = item.get("trials")
    if isinstance(tr, str):
        tr = ast.literal_eval(tr)
    return [(t.get("pred_sql") or "").strip() for t in tr]


def eval_file(path, ntrials):
    with open(path) as f:
        data = json.load(f)
    n_total = len(data)
    maj_correct = 0
    passk = 0
    reward_sum = 0
    reward_cnt = 0
    for item in data:
        db = item["db_id"]
        sqls = parse_trials(item)[:ntrials]
        rews = parse_rewards(item, ntrials)[:ntrials]
        if len(rews) < len(sqls):
            rews += [0] * (len(sqls) - len(rews))
        # pass@k / mean
        if any(rews):
            passk += 1
        reward_sum += sum(rews)
        reward_cnt += len(rews)
        # 聚类 (保持首次出现顺序)
        groups = {}
        order = []
        for sql, rw in zip(sqls, rews):
            s = sig_of(db, sql)
            if s not in groups:
                groups[s] = {"n": 0, "reward": rw, "first": len(order)}
                order.append(s)
            groups[s]["n"] += 1
        if not groups:
            continue
        # 最大簇, tie-break 首次出现
        best = min(groups.values(), key=lambda g: (-g["n"], g["first"]))
        if best["reward"] == 1:
            maj_correct += 1
    return {
        "n": n_total,
        "maj": maj_correct,
        "maj_pct": 100 * maj_correct / n_total,
        "passk": passk,
        "passk_pct": 100 * passk / n_total,
        "mean": 100 * reward_sum / reward_cnt,
    }


def gather_pairs(path, ntrials):
    with open(path) as f:
        data = json.load(f)
    pairs = set()
    for item in data:
        db = item["db_id"]
        for sql in parse_trials(item)[:ntrials]:
            if sql and sql.strip():
                pairs.add((db, norm(sql)))
    return pairs


def main():
    load_sig_cache()
    # 1) 汇总两个文件的所有唯一 (db,sql), 并发预计算签名
    pairs = gather_pairs(F8, 8) | gather_pairs(F32, 32)
    precompute_sigs(pairs, workers=32)
    save_sig_cache()
    # 2) 统计 (纯内存, 快)
    r8 = eval_file(F8, 8)
    r32 = eval_file(F32, 32)
    print("\n================= 结果 =================")
    for tag, r in [("8采样 ", r8), ("32采样", r32)]:
        print(f"{tag}: 多数投票 {r['maj']}/{r['n']} = {r['maj_pct']:.2f}%   "
              f"pass@N={r['passk_pct']:.2f}%   mean@N={r['mean']:.2f}%")
    print(f"\n差值 (8 - 32) 多数投票: {r8['maj_pct'] - r32['maj_pct']:+.2f} pp")


if __name__ == "__main__":
    main()
