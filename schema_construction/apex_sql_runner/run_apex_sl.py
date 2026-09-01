import os, sys, json, time, argparse, threading
import concurrent.futures
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                     # BIRD/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))    # 项目根（chat.py 所在）
# inference_selection/ 里的 llm_usage (供 chat.py 记 token/耗时); 从 submission 包根向下找
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # LadderSQL_BIRD_submission/
sys.path.insert(0, os.path.join(_ROOT, "inference_selection"))
try:
    import llm_usage
except Exception:  # noqa: BLE001
    llm_usage = None

# ★ 这两行的顺序不能换 —— 必须先 import chat, 再 import schema_linking。
#   APEX 的 schema_linking.py 顶部有一句 sys.path.insert(0, <它的上一级目录>), 而那个
#   目录里还放着另一份 chat.py, 把一个【已失效的 API key 和 dashscope 的 base 硬编成
#   模块常量】, 完全不读 OPENAI_API_KEY/OPENAI_API_BASE。若让 schema_linking 先 import,
#   它会把那份劫走 -> 模块 1 每题都 401, 且日志里看到的 env 变量明明是对的, 极难排查。
#   先在这里把正确的 chat(本目录下这份, 读环境变量) 装进 sys.modules, 之后 schema_linking
#   里的 `from chat import GPTChat` 就直接复用它, 不会再去搜 sys.path。
import chat as chat_mod
from schema_linking import SchemaLinking

# 隔离模式: 样本顺序(workers=1), 每题内 APEX 自带 2 采样并发 -> 每题 schema-link 耗时干净
ISOLATE_SAMPLES = os.environ.get("ISOLATE_SAMPLES", "0").lower() in ("1", "true", "yes")

# 进程级样本队列(推荐用于成本测量): N 个子进程, 每进程一次只处理一题。
#   为什么不能用线程池: APEX 内部有嵌套线程池(PLAN/PRUNE 采样), 子线程读不到
#   threading.local 的 unit, 只能回退到模块全局 _GLOBAL_UNIT; 线程池里 N 道题共享
#   同一个模块全局变量 -> token 会归到错误的题上。进程隔离后每进程同一时刻只有一题,
#   _GLOBAL_UNIT 无歧义, 既拿到 N 路并发, 又保证 per-question token 干净。
#   优先级: APEX_PROC_WORKERS>0 胜过全局 ISOLATE_SAMPLES —— 因为成本测量里只有 GPU 生成
#   需要样本串行, schema link 是样本间并发。run_test.sh 会全局 export ISOLATE_SAMPLES=1
#   给生成模块用, 这里不能被它误伤。
APEX_PROC_WORKERS = int(os.environ.get("APEX_PROC_WORKERS", "0") or 0)

# fork 前由父进程填好, 子进程继承(只读)。ProcessPool 无法 pickle 闭包, 故走模块全局。
_G: dict = {}


def load_json(p, default):
    if os.path.exists(p):
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:
            return default
    return default


def _proc_init():
    """子进程初始化: 把 llm_usage 账本切到本进程独占的分片文件(见 llm_usage.set_paths)。"""
    if llm_usage is not None:
        pid = os.getpid()
        d = _G["detail_dir"]
        llm_usage.set_paths(log_path=os.path.join(d, f"llm_usage_{pid}.json"),
                            detail_path=os.path.join(d, f"llm_usage_detail_{pid}.jsonl"))


def _proc_worker(case):
    """进程池 worker: 一次只处理一题 -> set_global_unit 无歧义, 嵌套线程 token 归属正确。

    返回 (状态, qid, 错误, pid, 本进程累计限流次数)。结果只经 parts/<qid>.json 落盘,
    不回写父进程的 results(fork 后内存不共享), 由父进程读分片。
    """
    qid = str(case["question_id"]); db_id = case["db_id"]
    pid = os.getpid()
    sqlite_path = os.path.join(_G["db_root"], db_id, f"{db_id}.sqlite")
    if not os.path.exists(sqlite_path):
        return ("err", qid, f"db not found: {sqlite_path}", pid, 0)
    try:
        unit_key = f"{db_id}|||{case['question']}"   # 跨阶段一致的题 key
        if llm_usage is not None:
            llm_usage.set_global_unit(unit_key)
        _t0 = time.time()
        linker = SchemaLinking(model=_G["model"], cache_dir=_G["cache_dir"])
        res = linker.run_schema_linking(
            case["question"], qid, evidence=case.get("evidence", ""),
            sqlite_path=sqlite_path, database_root_dir=_G["db_root"], enable_pruning=True)
        _wall = round(time.time() - _t0, 3)
        out = {"question_id": qid, "question": case["question"], "db_id": db_id,
               "evidence": case.get("evidence", ""), "result": res}
        json.dump(out, open(os.path.join(_G["parts_dir"], f"{qid}.json"), "w", encoding="utf-8"),
                  ensure_ascii=False)
        # 每题墙钟写本进程独占分片(父进程事后合并), 避免多进程抢同一文件
        with open(os.path.join(_G["wall_dir"], f"wall_{pid}.jsonl"), "a", encoding="utf-8") as wf:
            wf.write(json.dumps({"unit": unit_key, "stage": "schema_link", "wall_sec": _wall}) + "\n")
        return ("ok", qid, None, pid, chat_mod.RATE_LIMIT_COUNTER)
    except Exception as e:
        return ("err", qid, f"{type(e).__name__}: {e}", pid, chat_mod.RATE_LIMIT_COUNTER)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_json", required=True)
    ap.add_argument("--db_root", required=True)      # 本地库根目录
    ap.add_argument("--work_dir", required=True)     # 本地工作目录（cache + 分片 + 结果）
    ap.add_argument("--out_dir", required=True)      # persistent results directory
    ap.add_argument("--model", default="glm-5.2")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max_rounds", type=int, default=500)   # 安全阀；限流会持续重跑到消化，通常用不到此上限
    ap.add_argument("--limit", type=int, default=0)   # >0 时只处理前 N 条（验证用）；0=全量
    a = ap.parse_args()

    # 模型名可能带组织前缀(如 SiliconFlow 的 zai-org/GLM-5.2), 其中的 '/' 不能直接进
    # 文件名/目录名, 否则会当成子目录导致写盘失败。传给 API 的仍是 a.model 原名。
    model_slug = a.model.replace("/", "_").replace("\\", "_")

    cache_dir = os.path.join(a.work_dir, "cache", model_slug)
    parts_dir = os.path.join(a.work_dir, "parts")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(parts_dir, exist_ok=True)
    os.makedirs(a.out_dir, exist_ok=True)
    proc_mode = False
    if APEX_PROC_WORKERS > 0:
        proc_mode = True
        a.workers = APEX_PROC_WORKERS
        print(f"[procq] APEX_PROC_WORKERS={a.workers}: 进程级样本队列(始终 {a.workers} 道题在飞, "
              f"每进程一次一题) -> 每题 token/耗时干净")
    elif ISOLATE_SAMPLES:
        a.workers = 1     # 隔离: 一次只跑一题(题内仍保留 APEX 的 2/4 采样并发)
        print("[isolate] ISOLATE_SAMPLES=1: workers=1, 每题 schema-link 耗时干净")
    wall_path = os.path.join(a.out_dir, "schema_link_sample_wall.jsonl")
    wall_dir = os.path.join(a.work_dir, "wall_parts")
    detail_dir = os.path.join(a.work_dir, "usage_parts")
    os.makedirs(wall_dir, exist_ok=True)
    os.makedirs(detail_dir, exist_ok=True)
    _G.update(db_root=a.db_root, model=a.model, cache_dir=cache_dir, parts_dir=parts_dir,
              wall_dir=wall_dir, detail_dir=detail_dir)

    results_local = os.path.join(a.work_dir, f"results_{model_slug}.json")
    results_out   = os.path.join(a.out_dir,  f"results_{model_slug}.json")
    fail_out      = os.path.join(a.out_dir,  "failures.json")

    data = json.load(open(a.data_json, encoding="utf-8"))
    if a.limit and a.limit > 0:
        data = data[:a.limit]
        print(f"[limit] 验证模式：仅处理前 {len(data)} 条样本")
    lock = threading.Lock()

    # resume: local results first, then previous shared results, then per-question shards
    results = load_json(results_local, {})
    for k, v in load_json(results_out, {}).items():
        results.setdefault(k, v)
    for fn in os.listdir(parts_dir):
        if fn.endswith(".json"):
            results.setdefault(fn[:-5], load_json(os.path.join(parts_dir, fn), None))
    results = {k: v for k, v in results.items() if v}

    def flush(to_shared=True):
        with lock:
            snapshot = dict(results)
        json.dump(snapshot, open(results_local, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        if to_shared:
            json.dump(snapshot, open(results_out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    def worker(case):
        qid = str(case["question_id"]); db_id = case["db_id"]
        sqlite_path = os.path.join(a.db_root, db_id, f"{db_id}.sqlite")
        if not os.path.exists(sqlite_path):
            return ("err", qid, f"db not found: {sqlite_path}")
        try:
            unit_key = f"{db_id}|||{case['question']}"   # 跨阶段一致的题 key(与候选池/其它模块对齐)
            if llm_usage is not None:
                llm_usage.set_global_unit(unit_key)   # 隔离模式下一次只一题, 全局单元无歧义
            _t0 = time.time()
            linker = SchemaLinking(model=a.model, cache_dir=cache_dir)
            res = linker.run_schema_linking(
                case["question"], qid, evidence=case.get("evidence", ""),
                sqlite_path=sqlite_path, database_root_dir=a.db_root, enable_pruning=True)
            _wall = round(time.time() - _t0, 3)
            out = {"question_id": qid, "question": case["question"], "db_id": db_id,
                   "evidence": case.get("evidence", ""), "result": res}
            json.dump(out, open(os.path.join(parts_dir, f"{qid}.json"), "w", encoding="utf-8"),
                      ensure_ascii=False)                       # 分片落盘（崩溃安全，无需大锁）
            with lock:
                results[qid] = out
                with open(wall_path, "a", encoding="utf-8") as wf:
                    wf.write(json.dumps({"unit": unit_key, "stage": "schema_link", "wall_sec": _wall}) + "\n")
            return ("ok", qid, None)
        except Exception as e:
            return ("err", qid, f"{type(e).__name__}: {e}")

    pending = [c for c in data if str(c["question_id"]) not in results]
    print(f"[init] total={len(data)} done={len(results)} pending={len(pending)}")

    all_fail = {}
    HARD_FAIL_ROUNDS = 5      # 连续「无进展且无限流」达到此轮数 → 判定为硬错误，停止
    no_progress = 0
    rnd = 0
    rl_by_pid = {}            # 进程模式下子进程的限流计数不会回写父进程, 靠返回值汇总
    while pending:
        rnd += 1
        print(f"\n===== ROUND {rnd} : {len(pending)} cases, workers={a.workers}"
              f"{' (进程)' if proc_mode else ''} =====")
        rl_before = chat_mod.RATE_LIMIT_COUNTER + sum(rl_by_pid.values())
        n_before = len(pending)
        failed = []
        if proc_mode:
            import multiprocessing as _mp
            by_qid = {str(c["question_id"]): c for c in pending}
            ctx = _mp.get_context("fork")
            cnt = 0
            with ctx.Pool(processes=a.workers, initializer=_proc_init) as pool:
                for st, qid, err, pid, rl in tqdm(
                        pool.imap_unordered(_proc_worker, pending), total=len(pending)):
                    rl_by_pid[pid] = rl
                    if st == "err":
                        failed.append(by_qid[qid]); all_fail[qid] = err
                    else:
                        all_fail.pop(qid, None)
                        # 子进程只写了分片, 父进程回读才能计入 results / 断点统计
                        got = load_json(os.path.join(parts_dir, f"{qid}.json"), None)
                        if got:
                            results[qid] = got
                    cnt += 1
                    if cnt % 50 == 0:
                        flush()
                        print(f"[progress] round={rnd} done={cnt}/{len(pending)} "
                              f"ok_total={len(results)} "
                              f"rl_hits_total={sum(rl_by_pid.values())}", flush=True)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
                futs = {ex.submit(worker, c): c for c in pending}
                cnt = 0
                for f in tqdm(concurrent.futures.as_completed(futs), total=len(futs)):
                    st, qid, err = f.result()
                    if st == "err":
                        failed.append(futs[f]); all_fail[qid] = err
                    else:
                        all_fail.pop(qid, None)
                    cnt += 1
                    if cnt % 50 == 0:
                        flush()
                        print(f"[progress] round={rnd} done={cnt}/{len(futs)} "
                              f"ok_total={len(results)} rl_hits_total={chat_mod.RATE_LIMIT_COUNTER}", flush=True)
        flush()
        rl_total = chat_mod.RATE_LIMIT_COUNTER + sum(rl_by_pid.values())
        rl = [q for q, e in all_fail.items()
              if "##RATELIMIT##" in e or "429" in e or "rate" in e.lower() or "throttl" in e.lower() or "限流" in e]
        json.dump({"round": rnd, "failed": all_fail, "rate_limit_ids": rl,
                   "rate_limit_hits_total": rl_total,
                   "rate_limit_hits_this_round": rl_total - rl_before},
                  open(fail_out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"[round {rnd}] ok_total={len(results)} failed={len(failed)} rate_limit={len(rl)} "
              f"rl_hits_total={rl_total}")
        pending = failed
        if not pending:
            break                                     # 全部成功，结束
        made_progress = len(pending) < n_before
        if len(rl) > 0 or made_progress:
            # 仍有限流 或 本轮有进展 → 继续重跑（限流一定会被消化）；限流时多退避
            no_progress = 0
            time.sleep(30 if len(rl) > 0 else 5)
        else:
            # 无进展且无任何限流 → 疑似硬错误（重跑也不会好）
            no_progress += 1
            print(f"[warn] round {rnd}: no progress & no rate-limit "
                  f"({no_progress}/{HARD_FAIL_ROUNDS}); {len(pending)} non-ratelimit failures.")
            if no_progress >= HARD_FAIL_ROUNDS:
                print(f"[stop] {len(pending)} hard-error cases can't be fixed by retrying; see failures.json.")
                break
            time.sleep(5)
        if rnd >= a.max_rounds:                        # 安全阀，防止极端死循环
            print(f"[stop] reached max_rounds={a.max_rounds}; {len(pending)} still pending. "
                  f"Rerun the SAME command to resume (断点续跑).")
            break

    flush()
    # 进程模式: 合并各子进程的每题墙钟分片 -> 单一 wall 文件(供 profile_report 读)。
    # 合并后改名为 .done, 断点续跑时不会重复计入。
    if proc_mode:
        merged = 0
        with open(wall_path, "a", encoding="utf-8") as wf:
            for fn in sorted(os.listdir(wall_dir)):
                if not fn.endswith(".jsonl"):
                    continue
                src = os.path.join(wall_dir, fn)
                with open(src, encoding="utf-8") as rf:
                    for line in rf:
                        if line.strip():
                            wf.write(line if line.endswith("\n") else line + "\n")
                            merged += 1
                os.replace(src, src + ".done")
        print(f"[procq] 合并每题墙钟 {merged} 行 -> {wall_path}")
        print(f"[procq] token 明细分片在 {detail_dir}/ , 用 tools/merge_llm_usage.py 合并")
    print(f"\n[done] results={len(results)}/{len(data)} -> {results_out}")
    print(f"[rate-limit total hits] {chat_mod.RATE_LIMIT_COUNTER + sum(rl_by_pid.values())}")


if __name__ == "__main__":
    main()
