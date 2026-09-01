"""离线 token / 耗时核算 (不重采样、不调 glm)。

对某个 32 采样池 JSON, 逐 trial 产出 (输入 token, 输出 token, LLM 耗时, 轮数),
缓存到 sidecar `<src>.tokcost.json`, 供 eval_bon_trend 汇总 best-of-N 每题累计成本。

口径:
  - 输出 token out_tok = trials[i].gen_tokens (精确)。
  - 输入 token in_tok:
      * 若 trials[i].prompt_tokens 存在(新采样精确落盘) -> 直接用;
      * 否则用 tokenizer 对 trials[i].trajectory 做【逐轮前缀累计分词】估算 cumulative prompt tokens
        (ReAct 每轮把增长的上下文重新发给模型, 故累计输入=Σ_轮 tokens(该轮之前的全部消息));
      * 无 tokenizer 时回退 字符数/4 粗估。
  - 耗时 llm_time = trials[i].llm_time; 轮数 turns = trials[i].turns。

用法:
  python token_cost.py --src <32池.json> [--tokenizer <hf目录>] [--out <sidecar.json>]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

_AI_ROLES = {"ai", "assistant", "AIMessage"}


def get_tokenizer(tok_dir: Optional[str]):
    if not tok_dir or not os.path.isdir(tok_dir):
        return None
    try:
        from transformers import AutoTokenizer  # type: ignore
        return AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
    except Exception as e:  # noqa: BLE001
        print(f"[tokcost] tokenizer 载入失败({e}), 回退 字符数/4 估算", flush=True)
        return None


def _count(tok, text: str) -> int:
    if not text:
        return 0
    if tok is None:
        return max(1, len(text) // 4)
    try:
        return len(tok(text, add_special_tokens=False)["input_ids"])
    except Exception:
        return max(1, len(text) // 4)


def _cumulative_input_from_traj(tok, trajectory: List[Dict[str, Any]]) -> int:
    """逐轮前缀累计: 对每条 AI 消息, 把它之前的所有消息拼接分词后累加。"""
    if not trajectory:
        return 0
    total = 0
    for p, m in enumerate(trajectory):
        role = (m.get("role") or "").strip()
        if role in _AI_ROLES:
            ctx = "\n".join(str(x.get("content") or "") for x in trajectory[:p])
            total += _count(tok, ctx)
    return total


def trial_cost(trial: Dict[str, Any], tok) -> Dict[str, Any]:
    out_tok = int(trial.get("gen_tokens") or 0)
    pt = trial.get("prompt_tokens")
    if pt is not None:
        in_tok = int(pt or 0)
        in_src = "exact"
    else:
        in_tok = _cumulative_input_from_traj(tok, trial.get("trajectory") or [])
        in_src = "traj_est" if trial.get("trajectory") else "none"
    return {
        "in_tok": in_tok,
        "out_tok": out_tok,
        "llm_time": float(trial.get("llm_time") or 0.0),
        "turns": int(trial.get("turns") or 0),
        "in_src": in_src,
    }


def compute_or_load(src_path: str, tokenizer_dir: Optional[str] = None,
                    out_path: Optional[str] = None, force: bool = False) -> List[Dict[str, Any]]:
    """返回按采样文件原序对齐的每题成本: [{task_idx, trials:[trial_cost,...]}]。带 sidecar 缓存。"""
    out_path = out_path or (src_path + ".tokcost.json")
    if os.path.exists(out_path) and not force:
        with open(out_path) as f:
            return json.load(f)
    with open(src_path) as f:
        raw = json.load(f)
    need_tok = any(
        any(t.get("prompt_tokens") is None for t in item.get("trials", []))
        for item in raw
    )
    tok = get_tokenizer(tokenizer_dir) if need_tok else None
    if need_tok and tok is None:
        print("[tokcost] 无 prompt_tokens 且无 tokenizer, 输入 token 用 字符数/4 估算", flush=True)
    result: List[Dict[str, Any]] = []
    for item in raw:
        costs = [trial_cost(t, tok) for t in item.get("trials", [])]
        result.append({"task_idx": item.get("task_idx"), "db_id": item.get("db_id"), "trials": costs})
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result, f)
    os.replace(tmp, out_path)
    print(f"[tokcost] 写出 {len(result)} 题成本 -> {out_path}", flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="32 采样池 JSON")
    ap.add_argument("--tokenizer", default=os.environ.get("TOKENIZER_DIR"),
                    help="模型 huggingface 目录(用于离线估算输入 token); 无则回退 字符数/4")
    ap.add_argument("--out", default=None, help="sidecar 输出路径; 默认 <src>.tokcost.json")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    compute_or_load(args.src, args.tokenizer, args.out, args.force)


if __name__ == "__main__":
    main()
