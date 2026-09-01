"""闭源 LLM 用量统计（线程安全）—— 满足官方 Required: 上报 dev 上的 prompt token 数。

为什么需要
----------
BIRD 官方要求 Type 3/4 提交方"提前告知 dev 环境下的 prompt token 数"，用于估算 test 成本。
之前我们只能用「字符数 / 3.6」粗估判别与修复的 prompt 长度；本模块直接采集
API 返回的 `usage.prompt_tokens / completion_tokens`，跑完自动出账。

用法（调用方只需一行）
---------------------
    import llm_usage
    ...
    t0 = time.time()
    resp = client.chat.completions.create(...)
    llm_usage.record("judge", getattr(resp, "usage", None), time.time() - t0)

进程退出时自动落盘 + 打印汇总：
    LLM_USAGE_LOG=/work/logs/llm_usage.json   # 不设则写到当前目录 llm_usage.json
若文件已存在则**累加**（跨阶段、跨断点续跑都能汇总到一份账上）。
"""
from __future__ import annotations

import atexit
import json
import os
import threading

_LOCK = threading.Lock()
_STATS: dict = {}
_PATH = os.environ.get("LLM_USAGE_LOG", "") or os.path.join(os.getcwd(), "llm_usage.json")
# 逐调用明细(JSONL): 一行一次调用 {stage, unit, prompt_tokens, completion_tokens, seconds}。
# 用于事后算【每题 / 每模块 avg/max/min】(tools/profile_report.py)。
_DETAIL = os.environ.get("LLM_USAGE_DETAIL", "") or (
    os.path.splitext(_PATH)[0] + "_detail.jsonl" if _PATH else ""
)
_DETAIL_FH = None
_DIRTY = False

# 当前线程的【题号 / 单元】上下文: 深层调用(如修复 checker 链)不方便透传 unit 时,
# 由驱动层 set_unit() 在本线程设好, record() 自动读取。ThreadPoolExecutor 的 worker
# 自己 set/读, 同线程内一致, 安全。
_CTX = threading.local()
# 模块级全局单元: 用于嵌套线程池(如 APEX 内部的 PLAN/PRUNE 采样池)——子线程拿不到
# threading.local, 但可读模块全局。仅在【隔离模式: 一次只跑一题】时无歧义。
_GLOBAL_UNIT = None


def set_unit(unit) -> None:
    _CTX.unit = unit


def clear_unit() -> None:
    _CTX.unit = None


def set_global_unit(unit) -> None:
    """隔离模式专用: 设当前正在处理的题号(全局)。供嵌套线程里的 record 回退使用。"""
    global _GLOBAL_UNIT
    _GLOBAL_UNIT = unit


def set_paths(log_path: str = "", detail_path: str = "") -> None:
    """多进程隔离专用: 把【本进程】的账本切到独立文件, 并清空本进程已有统计。

    为什么需要: 多进程样本队列(每进程一次只跑一题)下, 若所有子进程共用同一个
    账本, 会有两个问题 ——
      1) 明细 JSONL 多写者 append: 本地 ext4 上小行原子, 但 WORK_DIR 常在 ossfs 上,
         多进程 append 不保证原子, 会串行;
      2) llm_usage.json 是 read-merge-replace, 多个子进程 atexit 同时 dump 会
         互相覆盖(lost update), 总量偏小。
    因此每个子进程在 fork 后、首次 record() 前调用本函数指向自己的分片文件,
    父进程跑完后用 tools/merge_llm_usage.py 合并。
    """
    global _PATH, _DETAIL, _DETAIL_FH, _DIRTY
    with _LOCK:
        if _DETAIL_FH is not None:
            try:
                _DETAIL_FH.close()
            except Exception:  # noqa: BLE001
                pass
            _DETAIL_FH = None
        if log_path:
            _PATH = log_path
        if detail_path:
            _DETAIL = detail_path
        # fork 继承了父进程的计数; 子进程只该记自己那部分, 否则合并时重复计入。
        _STATS.clear()
        _DIRTY = False


def _blank() -> dict:
    return {"calls": 0, "errors": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "no_usage_field": 0, "seconds": 0.0}


def record(tag: str, usage=None, seconds: float = 0.0, error: bool = False,
           model: str = "", unit=None) -> None:
    """记一次调用。usage 可以是 OpenAI SDK 的 usage 对象、dict、或 None。
    unit 不传时回退到本线程 set_unit() 的值(用于每题归属)。"""
    global _DIRTY
    if unit is None:
        unit = getattr(_CTX, "unit", None)
    if unit is None:
        unit = _GLOBAL_UNIT
    key = f"{tag}:{model}" if model else tag
    pt = ct = None
    if usage is not None:
        if isinstance(usage, dict):
            pt, ct = usage.get("prompt_tokens"), usage.get("completion_tokens")
        else:
            pt, ct = getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None)
    with _LOCK:
        s = _STATS.setdefault(key, _blank())
        s["calls"] += 1
        s["seconds"] += float(seconds or 0.0)
        if error:
            s["errors"] += 1
        if pt is None and ct is None:
            s["no_usage_field"] += 1        # 端点没返回 usage, 只能靠 calls 计数
        else:
            s["prompt_tokens"] += int(pt or 0)
            s["completion_tokens"] += int(ct or 0)
        _DIRTY = True
        _write_detail(tag, unit, pt, ct, seconds, error)


def _write_detail(tag, unit, pt, ct, seconds, error) -> None:
    """追写一行明细(已持锁)。失败不影响主流程。"""
    global _DETAIL_FH
    if not _DETAIL:
        return
    try:
        if _DETAIL_FH is None:
            d = os.path.dirname(os.path.abspath(_DETAIL))
            if d:
                os.makedirs(d, exist_ok=True)
            _DETAIL_FH = open(_DETAIL, "a", encoding="utf-8")
        rec = {"stage": tag, "unit": unit, "prompt_tokens": pt, "completion_tokens": ct,
               "seconds": round(float(seconds or 0.0), 3), "error": bool(error)}
        _DETAIL_FH.write(json.dumps(rec, ensure_ascii=False) + "\n")
        _DETAIL_FH.flush()
    except Exception as e:  # noqa: BLE001
        print(f"[llm_usage] detail write failed: {e}", flush=True)


def snapshot() -> dict:
    with _LOCK:
        out = {k: dict(v) for k, v in _STATS.items()}
    total = _blank()
    for v in out.values():
        for k in total:
            total[k] += v[k]
    out["_total"] = total
    return out


def dump(path: str | None = None) -> str | None:
    """把统计累加进 json 文件（已存在则合并），返回路径。"""
    global _DIRTY
    if not _DIRTY:
        return None
    p = path or _PATH
    cur = snapshot()
    cur.pop("_total", None)
    try:
        if os.path.exists(p):
            old = json.load(open(p, encoding="utf-8"))
            old.pop("_total", None)
            for k, v in old.items():
                if not isinstance(v, dict):
                    continue
                tgt = cur.setdefault(k, _blank())
                for kk in _blank():
                    tgt[kk] = tgt.get(kk, 0) + v.get(kk, 0)
        total = _blank()
        for v in cur.values():
            if isinstance(v, dict):
                for kk in total:
                    total[kk] += v.get(kk, 0)
        cur["_total"] = total
        d = os.path.dirname(os.path.abspath(p))
        if d:
            os.makedirs(d, exist_ok=True)
        tmp = p + ".tmp"
        json.dump(cur, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        os.replace(tmp, p)
        _DIRTY = False
        return p
    except Exception as e:  # noqa: BLE001  统计失败绝不能影响主流程
        print(f"[llm_usage] dump failed: {e}", flush=True)
        return None


def _at_exit():
    p = dump()
    if not p:
        return
    t = json.load(open(p, encoding="utf-8")).get("_total", {})
    print(f"[llm_usage] calls={t.get('calls')} prompt_tok={t.get('prompt_tokens')} "
          f"completion_tok={t.get('completion_tokens')} errors={t.get('errors')} "
          f"no_usage={t.get('no_usage_field')} -> {p}", flush=True)


atexit.register(_at_exit)
