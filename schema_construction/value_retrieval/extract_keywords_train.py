"""BIRD train 关键词提取（仅 LLM，不依赖向量库，可与建库并行）。

用 glm-5.2 根据 question + evidence 提取“值关键词”，输出：
  /root/bird_train_keywords.json = {qid: {db_id, question, evidence, keywords}}

qid 用 train.parquet 行号。并发 12 + 6 次指数退避（dashscope 429 加固）。断点续跑。
"""
import os
import re
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI

TRAIN_PARQUET = "/path/to/nl2sql_dataset/bird/bird_clean_data/train.parquet"
OUTPUT_PATH = "/root/bird_train_keywords.json"

API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
KW_MODEL = "glm-5.2"

MAX_KEYWORDS = 8
WORKERS = 12
LIMIT = int(os.environ.get("LIMIT", "0"))
SAVE_EVERY = 100

KEYWORD_PROMPT = """You are a SQL expert. Given a question about a database, extract **value keywords** that would appear in SQL WHERE/HAVING/JOIN conditions as literal values.
Rules:
- Extract specific names, dates, numbers, categories, status codes mentioned in the question or evidence.
- Do NOT extract generic words (total, average, count, number).
- Do NOT extract column names or table names.
- Output: JSON array of strings. Empty [] if no literal values are needed.

Question: {question}
Evidence: {evidence}
Output:"""


def main():
    t0 = time.time()
    df = pd.read_parquet(TRAIN_PARQUET)
    if LIMIT > 0:
        df = df.head(LIMIT)
    print(f"train questions: {len(df)}")

    client = OpenAI(api_key=API_KEY, base_url=API_BASE)

    results = {}
    if os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH) as f:
                results = json.load(f)
            print(f"resume: {len(results)} done")
        except Exception:
            results = {}

    tasks = []
    for idx, row in df.iterrows():
        qid = str(idx)
        if qid in results:
            continue
        tasks.append({
            "qid": qid,
            "db_id": row["db_id"],
            "question": row["question"],
            "evidence": row.get("evidence", "") or "",
        })
    print(f"tasks to run: {len(tasks)}")
    if not tasks:
        print("Nothing to do.")
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

    def save():
        with open(OUTPUT_PATH, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

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
                results[qid] = rec
                counters["done"] += 1
                if rec.get("keywords"):
                    counters["kw_hit"] += 1
                if counters["done"] % SAVE_EVERY == 0:
                    save()
                    print(f"  [{counters['done']}/{len(tasks)}] kw={counters['kw_hit']}, "
                          f"{time.time()-t0:.0f}s")

    save()
    print(f"\nDONE! total={len(results)}, kw_hit={counters['kw_hit']}, "
          f"time={time.time()-t0:.0f}s -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
