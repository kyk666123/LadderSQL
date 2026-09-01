"""BIRD **old dev** 关键词提取（仅 LLM, glm-5.2, 不依赖向量库, 可与建库并行）。

= extract_keywords_train.py 的 old dev 版：逻辑/prompt/参数完全一致，只改数据源。
qid 直接取自 APEX 结果的 key(=question_id)，保证与 APEX / 值检索 / build cache 的 key 严格对齐。

输入:  apex-sql-schema-link-results/bird_old_dev/results_glm-5.2.json (1534, key=question_id)
输出:  {qid: {db_id, question, evidence, keywords}}
       - 本地: /root/bird_old_dev_keywords.json
       - OSS : apex-sql-schema-link-results/bird_old_dev/keywords_old_dev.json
断点续跑。并发 12 + 6 次指数退避（dashscope 429 加固）。
"""
import os
import re
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

APEX_PATH = "/path/to/schema_link_results/bird_old_dev/results_glm-5.2.json"
OUTPUT_PATH = "/root/bird_old_dev_keywords.json"
OSS_OUTPUT_PATH = "/path/to/schema_link_results/bird_old_dev/keywords_old_dev.json"

API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
KW_MODEL = "glm-5.2"

MAX_KEYWORDS = 8
WORKERS = 12
LIMIT = int(os.environ.get("LIMIT", "0"))
SAVE_EVERY = 100

# 与 extract_keywords_train.py 完全一致的 prompt
KEYWORD_PROMPT = """You are a SQL expert. Given a question about a database, extract **value keywords** that would appear in SQL WHERE/HAVING/JOIN conditions as literal values.
Rules:
- Extract specific names, dates, numbers, categories, status codes mentioned in the question or evidence.
- Do NOT extract generic words (total, average, count, number).
- Do NOT extract column names or table names.
- Output: JSON array of strings. Empty [] if no literal values are needed.

Question: {question}
Evidence: {evidence}
Output:"""


def _save():
    with open(OUTPUT_PATH, "w") as f:
        json.dump(_RESULTS, f, ensure_ascii=False, indent=2)


def _persist():
    _save()
    try:
        os.makedirs(os.path.dirname(OSS_OUTPUT_PATH), exist_ok=True)
        with open(OSS_OUTPUT_PATH, "w") as f:
            json.dump(_RESULTS, f, ensure_ascii=False, indent=2)
        print(f"  saved -> {OUTPUT_PATH} & {OSS_OUTPUT_PATH}")
    except Exception as e:
        print(f"  [OSS persist warn] {e}; local saved -> {OUTPUT_PATH}")


_RESULTS = {}


def main():
    global _RESULTS
    t0 = time.time()
    apex = json.load(open(APEX_PATH))
    items = list(apex.items())
    if LIMIT > 0:
        items = items[:LIMIT]
    print(f"old dev questions: {len(items)}")

    client = OpenAI(api_key=API_KEY, base_url=API_BASE)

    if os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH) as f:
                _RESULTS = json.load(f)
            print(f"resume: {len(_RESULTS)} done")
        except Exception:
            _RESULTS = {}

    tasks = []
    for qid, rec in items:
        qid = str(qid)
        if qid in _RESULTS:
            continue
        tasks.append({
            "qid": qid,
            "db_id": rec["db_id"],
            "question": rec["question"],
            "evidence": rec.get("evidence", "") or "",
        })
    print(f"tasks to run: {len(tasks)}")
    if not tasks:
        print("Nothing to do.")
        _persist()
        return

    lock = threading.Lock()
    counters = {"done": 0, "kw_hit": 0}

    def call_llm(messages, max_tokens=256):
        for attempt in range(6):
            try:
                resp = client.chat.completions.create(
                    model=KW_MODEL, messages=messages,
                    temperature=0.01, max_tokens=max_tokens,
                    extra_body={"enable_thinking": False},
                )
                content = resp.choices[0].message.content or ""
                return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            except Exception as e:
                if attempt < 5:
                    time.sleep(min(2 ** attempt, 30))
                else:
                    print(f"    [LLM ERROR] {e}")
                    return ""
        return ""

    def extract(task):
        msg = [{"role": "user", "content": KEYWORD_PROMPT.format(
            question=task["question"], evidence=task["evidence"])}]
        resp = call_llm(msg)
        kws = []
        try:
            m = re.search(r"\[.*?\]", resp, re.DOTALL)
            if m:
                arr = json.loads(m.group(0))
                kws = [str(k).strip() for k in arr if k and str(k).strip()][:MAX_KEYWORDS]
        except Exception:
            pass
        return task["qid"], {
            "db_id": task["db_id"], "question": task["question"],
            "evidence": task["evidence"], "keywords": kws,
        }

    print(f"Start: {len(tasks)} tasks, {WORKERS} workers, model={KW_MODEL}")
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(extract, t): t for t in tasks}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                qid, rec = fut.result()
            except Exception as e:
                qid, rec = t["qid"], {
                    "db_id": t["db_id"], "question": t["question"],
                    "evidence": t["evidence"], "keywords": [], "error": str(e),
                }
            with lock:
                _RESULTS[qid] = rec
                counters["done"] += 1
                if rec.get("keywords"):
                    counters["kw_hit"] += 1
                if counters["done"] % SAVE_EVERY == 0:
                    _save()
                    print(f"  [{counters['done']}/{len(tasks)}] kw={counters['kw_hit']}, "
                          f"{time.time()-t0:.0f}s")

    _persist()
    print(f"\nDONE! total={len(_RESULTS)}, kw_hit={counters['kw_hit']}, "
          f"time={time.time()-t0:.0f}s -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
