"""clean dev 关键词: 复用 old dev(question&evidence 均未变的 qid) + glm 重算变化的。

= extract_keywords_old_dev.py 的逻辑/prompt/参数完全一致, 仅:
  - 数据源换成 clean dev(results_glm-5.2.json + dev_20251106.parquet)
  - 预填 old dev 关键词到输出, 只对 (question∪evidence) 变化的 qid 调 LLM 重算
输出: /root/bird_clean_dev_keywords.json  &  bird_clean_dev/keywords_clean_dev.json
"""
import os
import re
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI

BASE = "/path/to/LadderSQL"
CLEAN_APEX = f"{BASE}/apex-sql-schema-link-results/bird_clean_dev/results_glm-5.2.json"
OLD_KW = f"{BASE}/apex-sql-schema-link-results/bird_old_dev/keywords_old_dev.json"
OLD_PARQUET = "/path/to/nl2sql_dataset/bird/dev_20240627/dev.parquet"
CLEAN_PARQUET = "/path/to/nl2sql_dataset/bird/bird_clean_data/dev_20251106.parquet"
OUTPUT_PATH = "/root/bird_clean_dev_keywords.json"
OSS_OUTPUT_PATH = f"{BASE}/apex-sql-schema-link-results/bird_clean_dev/keywords_clean_dev.json"

API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
KW_MODEL = "glm-5.2"
MAX_KEYWORDS = 8
WORKERS = 12

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
    apex = json.load(open(CLEAN_APEX))
    old_kw = json.load(open(OLD_KW))
    old = pd.read_parquet(OLD_PARQUET)
    new = pd.read_parquet(CLEAN_PARQUET)
    om = {str(r["question_id"]): (str(r["question"]), str(r["evidence"])) for _, r in old.iterrows()}
    nm = {str(r["question_id"]): (str(r["question"]), str(r["evidence"])) for _, r in new.iterrows()}

    results = {}
    reused = 0
    tasks = []
    for qid, rec in apex.items():
        qid = str(qid)
        q = rec["question"]
        ev = rec.get("evidence", "") or ""
        unchanged = (qid in om and qid in nm and om[qid] == nm[qid] and qid in old_kw)
        if unchanged:
            # 复用 old dev 关键词, 但记录 clean 的 question/evidence(二者相同)
            results[qid] = {"db_id": rec["db_id"], "question": q, "evidence": ev,
                            "keywords": old_kw[qid].get("keywords", [])}
            reused += 1
        else:
            tasks.append({"qid": qid, "db_id": rec["db_id"], "question": q, "evidence": ev})
    print(f"复用 old dev 关键词={reused}  需重算(q∪e变化)={len(tasks)}")

    client = OpenAI(api_key=API_KEY, base_url=API_BASE)
    lock = threading.Lock()
    counters = {"done": 0, "kw_hit": 0}

    def call_llm(messages, max_tokens=256):
        for attempt in range(6):
            try:
                resp = client.chat.completions.create(
                    model=KW_MODEL, messages=messages, temperature=0.01,
                    max_tokens=max_tokens, extra_body={"enable_thinking": False})
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
        return task["qid"], {"db_id": task["db_id"], "question": task["question"],
                             "evidence": task["evidence"], "keywords": kws}

    if tasks:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futures = {ex.submit(extract, t): t for t in tasks}
            for fut in as_completed(futures):
                t = futures[fut]
                try:
                    qid, rec = fut.result()
                except Exception as e:
                    qid, rec = t["qid"], {"db_id": t["db_id"], "question": t["question"],
                                          "evidence": t["evidence"], "keywords": [], "error": str(e)}
                with lock:
                    results[qid] = rec
                    counters["done"] += 1
                    if rec.get("keywords"):
                        counters["kw_hit"] += 1
                    if counters["done"] % 100 == 0:
                        print(f"  [{counters['done']}/{len(tasks)}] {time.time()-t0:.0f}s")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    os.makedirs(os.path.dirname(OSS_OUTPUT_PATH), exist_ok=True)
    with open(OSS_OUTPUT_PATH, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"DONE! total={len(results)} (复用{reused}+重算{len(tasks)})  重算命中关键词={counters['kw_hit']}")
    print(f"  -> {OUTPUT_PATH} & {OSS_OUTPUT_PATH}  time={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
