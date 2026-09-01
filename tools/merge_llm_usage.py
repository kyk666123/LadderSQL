#!/usr/bin/env python3
"""合并多进程 llm_usage 分片账本 -> 单一 detail JSONL + 汇总 json。

背景
----
APEX schema linking 用【进程级样本队列】跑(APEX_PROC_WORKERS=32): 每个子进程把
token 明细写到自己的分片文件(见 inference_selection/llm_usage.py::set_paths),
原因是多进程 append 同一文件在 ossfs 上不保证原子, 且 llm_usage.json 的
read-merge-replace 会互相覆盖。本脚本把分片汇总回主账本。

用法
----
    python tools/merge_llm_usage.py \
        --parts-dir  WORK_DIR/apex/usage_parts \
        --out-detail WORK_DIR/logs/llm_usage_detail.jsonl \
        --out-json   WORK_DIR/logs/llm_usage.json

汇总的真相源是【明细行】而不是子进程的 llm_usage.json 分片。原因: 子进程由
multiprocessing.Pool 管理, 退出走 os._exit(), **不执行 atexit**, 所以
llm_usage.dump() 从来不会被触发, json 分片根本不存在。而明细 JSONL 是每次
record() 实时 append + flush 的, 必然完整。因此本脚本从明细重建汇总, 并把
偶然存在的 json 分片直接归档(不累加), 避免同一批调用被计两次。

幂等: 合并过的分片会改名为 *.merged, 重复执行不会重复计入。
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def _blank() -> dict:
    return {"calls": 0, "errors": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "no_usage_field": 0, "seconds": 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts-dir", required=True, help="子进程分片目录(usage_parts)")
    ap.add_argument("--out-detail", required=True, help="合并后的明细 JSONL(追加写)")
    ap.add_argument("--out-json", default="", help="合并后的汇总 json(累加)")
    a = ap.parse_args()

    if not os.path.isdir(a.parts_dir):
        print(f"[merge] 分片目录不存在, 跳过: {a.parts_dir}")
        return

    # ---- 1) 明细: 逐行追加到主 detail, 同时就地聚合出本批的 stage 汇总 ----
    os.makedirs(os.path.dirname(os.path.abspath(a.out_detail)), exist_ok=True)
    n_lines = n_files = 0
    batch: dict = {}                              # stage -> 计数, 由明细行重建
    with open(a.out_detail, "a", encoding="utf-8") as out:
        for src in sorted(glob.glob(os.path.join(a.parts_dir, "llm_usage_detail_*.jsonl"))):
            with open(src, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    out.write(line if line.endswith("\n") else line + "\n")
                    n_lines += 1
                    try:
                        r = json.loads(line)
                    except Exception:  # noqa: BLE001  明细写坏一行不该让合并失败
                        continue
                    s = batch.setdefault(str(r.get("stage") or "unknown"), _blank())
                    s["calls"] += 1
                    s["seconds"] += float(r.get("seconds") or 0.0)
                    if r.get("error"):
                        s["errors"] += 1
                    pt, ct = r.get("prompt_tokens"), r.get("completion_tokens")
                    if pt is None and ct is None:
                        s["no_usage_field"] += 1   # 端点没返回 usage, 只能计 calls
                    else:
                        s["prompt_tokens"] += int(pt or 0)
                        s["completion_tokens"] += int(ct or 0)
            os.replace(src, src + ".merged")     # 幂等: 已合并的不再参与
            n_files += 1
    print(f"[merge] 明细: {n_files} 个分片 / {n_lines} 行 -> {a.out_detail}")

    # ---- 2) 汇总 json: 把本批(由明细重建的)计数累加进主账本 ----
    if not a.out_json:
        return
    cur: dict = {}
    if os.path.exists(a.out_json):
        try:
            cur = json.load(open(a.out_json, encoding="utf-8"))
            cur.pop("_total", None)
        except Exception as e:  # noqa: BLE001
            print(f"[merge] 已有汇总读取失败({e}), 从空开始")
            cur = {}
    for k, v in batch.items():
        tgt = cur.setdefault(k, _blank())
        for kk in _blank():
            tgt[kk] = tgt.get(kk, 0) + v.get(kk, 0)

    # 子进程的 json 分片正常不存在(Pool worker 走 os._exit, 不跑 atexit)。若确实
    # 存在, 说明该进程是正常退出的 —— 它的调用已经在明细里了, 这里只归档不累加。
    n_json = 0
    for src in sorted(glob.glob(os.path.join(a.parts_dir, "llm_usage_*.json"))):
        if src.endswith(".merged"):
            continue
        os.replace(src, src + ".merged")
        n_json += 1
    if n_json:
        print(f"[merge] 归档(不累加, 避免与明细重复计入) {n_json} 个 json 分片")

    total = _blank()
    for v in cur.values():
        if isinstance(v, dict):
            for kk in total:
                total[kk] += v.get(kk, 0)
    cur["_total"] = total
    tmp = a.out_json + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(a.out_json)), exist_ok=True)
    json.dump(cur, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    os.replace(tmp, a.out_json)
    print(f"[merge] 汇总 -> {a.out_json}  "
          f"calls={total['calls']} in_tok={total['prompt_tokens']} "
          f"out_tok={total['completion_tokens']}")


if __name__ == "__main__":
    main()
