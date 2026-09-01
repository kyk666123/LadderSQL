"""构建 step192·32 候选的【执行结果缓存】，供后续 best-of-N 复用（AI 不再需碰数据库）。

- 去重：按 (db_id, 规范化SQL) 只执行一次（49086 → ~13626）。
- 稳定签名：用 md5(sorted rows)，**跨进程稳定**（不能用 Python 内置 hash，它有随机盐）。
- 硬超时：线程 Timer + conn.interrupt() 中止长查询，绝不挂死。
- 输出：cache/exec_cache_step192_32.json  = {meta, results: {key -> {ok,error,row_count,result_sig,preview}}}
  key = f"{db_id}|||{' '.join(pred_sql.split())}"  （AI 用同样方式重建 key 查表）
"""
import hashlib
import json
import os
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import bird_common as B

FP = "/path/to/sampling_outputs/ckpt_20260711_235128_step192_bird_old_dev_maxturns5_32trials_traj_with_sql.json"
OUT = f"{B.CACHE_DIR}/exec_cache_step192_32.json"
TIMEOUT = int(os.environ.get("EXEC_TIMEOUT", "15"))   # giant DB 可调到 30~60
MAX_ROWS = 30       # preview 最多保留行数
CELL_CAP = 200      # 单元格字符上限
WORKERS = 32


def norm(sql: str) -> str:
    return " ".join((sql or "").split())


def _sortedness(vals):
    try:
        if vals == sorted(vals):
            return "asc"
        if vals == sorted(vals, reverse=True):
            return "desc"
    except TypeError:
        return "na"
    return "no"


def _profile(rows):
    """在完整结果上算结构画像: 行/列数 + 每列(类型/null/distinct/数值统计/有序性)。"""
    if rows is None:
        return None
    rc = len(rows)
    if rc == 0:
        return {"row_count": 0, "col_count": 0, "cols": []}
    first = rows[0]
    cc = len(first) if isinstance(first, (tuple, list)) else 1
    cols = []
    for j in range(cc):
        vals = [(r[j] if isinstance(r, (tuple, list)) else r) for r in rows]
        nonnull = [v for v in vals if v is not None and v != ""]
        col = {"n_null": rc - len(nonnull), "n_distinct": len(set(vals))}
        nums = []
        is_num = bool(nonnull)
        for v in nonnull:
            try:
                nums.append(float(v))
            except (TypeError, ValueError):
                is_num = False
                break
        if is_num and nums:
            col["type"] = "num"
            col["min"] = round(min(nums), 4)
            col["max"] = round(max(nums), 4)
            col["sum"] = round(sum(nums), 4)
        else:
            col["type"] = "text"
            col["sample"] = [str(v)[:30] for v in nonnull[:2]]
        col["sorted"] = _sortedness(vals)
        cols.append(col)
    return {"row_count": rc, "col_count": cc, "cols": cols}


def exec_one(db_path: str, sql: str):
    if not sql or not sql.strip():
        return {"ok": False, "error": "empty sql", "row_count": None, "result_sig": None, "preview": []}
    if not os.path.exists(db_path):
        return {"ok": False, "error": f"db not found: {db_path}", "row_count": None, "result_sig": None, "preview": []}
    conn = None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=TIMEOUT)
        conn.text_factory = lambda b: b.decode(errors="ignore")
        timer = threading.Timer(TIMEOUT, conn.interrupt)  # 跨线程 interrupt 是官方允许的
        timer.start()
        try:
            cur = conn.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
        finally:
            timer.cancel()
        # 稳定、顺序无关的结果签名（与 hash_result 的 sorted-str 语义一致）
        sig = hashlib.md5(str(sorted(str(r) for r in rows)).encode("utf-8", "ignore")).hexdigest()
        preview = []
        for r in rows[:MAX_ROWS]:
            cells = r if isinstance(r, (tuple, list)) else (r,)
            preview.append([str(c)[:CELL_CAP] for c in cells])
        return {"ok": True, "error": None, "row_count": len(rows), "result_sig": sig,
                "preview": preview, "profile": _profile(rows)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)[:200], "row_count": None, "result_sig": None,
                "preview": [], "profile": None}
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


def main():
    data = json.load(open(FP))
    # 去重收集任务
    tasks = {}  # key -> (db_path, 原始sql)
    n_cand = 0
    for r in data:
        db_id = r["db_id"]
        dbp = B.db_path_of(db_id)
        for t in r["trials"]:
            s = t.get("pred_sql")
            if not s:
                continue
            n_cand += 1
            key = f"{db_id}|||{norm(s)}"
            if key not in tasks:
                tasks[key] = (dbp, s)
    print(f"候选(非None)={n_cand}  去重唯一={len(tasks)}", flush=True)

    results = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(exec_one, dbp, s): k for k, (dbp, s) in tasks.items()}
        done = 0
        for fut in as_completed(futs):
            k = futs[fut]
            try:
                results[k] = fut.result()
            except Exception as e:  # noqa: BLE001
                results[k] = {"ok": False, "error": f"future err: {str(e)[:120]}",
                              "row_count": None, "result_sig": None, "preview": [], "profile": None}
            done += 1
            if done % 1000 == 0:
                print(f"  {done}/{len(tasks)}  ({time.time()-t0:.0f}s)", flush=True)

    ok = sum(1 for v in results.values() if v["ok"])
    err = len(results) - ok
    empty = sum(1 for v in results.values() if v["ok"] and v["row_count"] == 0)
    out = {
        "meta": {
            "source_file": FP,
            "db_dir": B.DB_DIR,
            "n_candidates": n_cand,
            "n_unique": len(tasks),
            "n_ok": ok,
            "n_error": err,
            "n_empty_result": empty,
            "timeout_s": TIMEOUT,
            "max_preview_rows": MAX_ROWS,
            "key_format": "f\"{db_id}|||{' '.join(pred_sql.split())}\"",
            "result_sig": "md5(str(sorted(str(r) for r in rows)))  # 跨进程稳定, 聚簇用",
        },
        "results": results,
    }
    os.makedirs(B.CACHE_DIR, exist_ok=True)
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    os.replace(tmp, OUT)
    sz = os.path.getsize(OUT) / 1e6
    print(f"\n完成: {OUT} ({sz:.1f} MB, {time.time()-t0:.0f}s)")
    print(f"  唯一={len(tasks)}  成功={ok}  报错={err}  空结果={empty}")


if __name__ == "__main__":
    main()
