"""BIRD **old dev** 值向量检索（消费预提取关键词，不再调用 LLM）。

= train_value_retrieval.py 的 old dev 版：检索逻辑/参数(K=5, l2<0.8, 跳纯数字, 去重)完全一致，
只改数据源为 old dev, 并把 chroma 向量库从 OSS 复制到本地再打开(OSS FUSE 直接打开会 disk I/O error)。

依赖:
- 关键词:  /root/bird_old_dev_keywords.json (extract_keywords_old_dev.py 产出; 缺失则回退 OSS 版)
- 向量库:  OSS chroma/bird_dev -> 本地 /root/bird_dev_chroma (每库一个 collection, name=db_id)

产物:
- 本地: /root/bird_old_dev_value_retrieval_kw.json
- OSS : apex-sql-schema-link-results/bird_old_dev/value_retrieval_old_dev_kw.json
qid 与 APEX/keywords 严格对齐(=question_id)。断点续跑 + 段错误自动跳过。
"""
import os
import json
import time
import shutil

import chromadb
from sentence_transformers import SentenceTransformer
import torch

# ===================== CONFIG =====================
# OSS 上的 chroma/bird_dev 由更新版 chromadb 构建, 本环境 chromadb 1.1.1 打开会 panic;
# 故改用 rebuild_dev_chroma_v111.py 用 1.1.1 重建的本地库(内容/抽取规则一致)。
LOCAL_CHROMA = "/root/bird_dev_chroma_v111"   # rebuild_dev_chroma_v111.py 产出
EMBEDDING_MODEL_PATH = "/path/to/embedding_models/all-MiniLM-L6-v2"

KEYWORDS_PATH = "/root/bird_clean_dev_keywords.json"
KEYWORDS_OSS_FALLBACK = "/path/to/schema_link_results/bird_clean_dev/keywords_clean_dev.json"

OUTPUT_PATH = "/root/bird_clean_dev_value_retrieval_kw.json"
OSS_OUTPUT_PATH = "/path/to/schema_link_results/bird_clean_dev/value_retrieval_clean_dev_kw.json"

# 与 train_value_retrieval.py / Agentar-Scale-SQL 严格对齐
K = 5                 # n_results：每个关键词取 top-5
THRESHOLD = 0.8       # 仅保留 l2 距离 < 0.8 的命中
ENCODE_BATCH = 2048

CHECKPOINT_EVERY = 200
CUR_FILE = OUTPUT_PATH + ".current"
SKIP_DB_FILE = OUTPUT_PATH + ".skip_dbs"


def _ensure_local_chroma():
    if os.path.exists(os.path.join(LOCAL_CHROMA, "chroma.sqlite3")):
        print(f"  本地 chroma 已存在: {LOCAL_CHROMA}")
        return
    raise SystemExit(
        f"[ERR] 未找到本地向量库 {LOCAL_CHROMA}\n"
        f"      请先运行: python rebuild_dev_chroma_v111.py 重建与 chromadb 1.1.1 兼容的 dev 向量库"
    )


def main():
    t0 = time.time()

    kw_path = KEYWORDS_PATH if os.path.exists(KEYWORDS_PATH) else KEYWORDS_OSS_FALLBACK
    print(f"Loading keywords: {kw_path}")
    with open(kw_path) as f:
        kw_data = json.load(f)
    print(f"  questions with keyword records: {len(kw_data)}")

    print("Preparing local chroma...")
    _ensure_local_chroma()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SentenceTransformer on {device}...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL_PATH, device=device)

    print("Loading ChromaDB (本地)...")
    cli = chromadb.PersistentClient(path=LOCAL_CHROMA)
    collections = {c.name: c for c in cli.list_collections()}
    print(f"  总 collections: {len(collections)} -> {sorted(collections)}")

    # 1) 汇总唯一关键词, 一次性 GPU 批量编码
    all_kws = set()
    for rec in kw_data.values():
        for k in rec.get("keywords", []):
            if isinstance(k, str) and k.strip() and not k.isdigit():
                all_kws.add(k)
    all_kws = sorted(all_kws)
    print(f"  unique keywords to encode: {len(all_kws)}")

    kw2emb = {}
    if all_kws:
        embs = embed_model.encode(
            all_kws, batch_size=ENCODE_BATCH, show_progress_bar=False, device=device
        ).tolist()
        kw2emb = dict(zip(all_kws, embs))

    # 2) 逐条检索 (断点续跑 + 段错误跳库)
    results = {}
    if os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH) as f:
                results = json.load(f)
            print(f"  resume: 已完成 {len(results)} 条")
        except Exception:
            results = {}

    skip_dbs = set()
    if os.path.exists(SKIP_DB_FILE):
        with open(SKIP_DB_FILE) as f:
            skip_dbs = {line.strip() for line in f if line.strip()}

    if os.path.exists(CUR_FILE):
        try:
            with open(CUR_FILE) as f:
                last = f.read().strip()
        except Exception:
            last = ""
        if last:
            parts = last.split("\t")
            last_qid = parts[0]
            last_db = parts[1] if len(parts) > 1 else ""
            if last_qid and last_qid not in results and last_db:
                skip_dbs.add(last_db)
                with open(SKIP_DB_FILE, "w") as f:
                    f.write("\n".join(sorted(skip_dbs)) + "\n")
                print(f"  [段错误检测] 上次崩溃于 qid={last_qid} db={last_db}, 已跳过该库")
    if skip_dbs:
        print(f"  跳过库(段错误): {sorted(skip_dbs)}")

    counters = {"done": len(results), "kw_hit": 0, "retr_hit": 0, "no_coll": 0, "skipped": 0}

    for qid, rec in kw_data.items():
        if qid in results:
            continue
        db_id = rec["db_id"]
        keywords = [k for k in rec.get("keywords", []) if isinstance(k, str) and k.strip()]
        retrieved = []

        with open(CUR_FILE, "w") as cf:
            cf.write(f"{qid}\t{db_id}")
            cf.flush()
            os.fsync(cf.fileno())

        if keywords and db_id in skip_dbs:
            counters["skipped"] += 1
        elif keywords and db_id in collections:
            collection = collections[db_id]
            seen = set()
            for kw in keywords:
                if kw.isdigit():
                    continue
                emb = kw2emb.get(kw)
                if emb is None:
                    continue
                try:
                    r = collection.query(query_embeddings=[emb], n_results=K)
                except Exception:
                    continue
                if not (r and r["documents"] and r["documents"][0]):
                    continue
                for doc, meta, dist in zip(
                    r["documents"][0], r["metadatas"][0], r["distances"][0]
                ):
                    if float(dist) >= THRESHOLD:
                        continue
                    tbl = meta.get("table", "")
                    col = meta.get("column", "")
                    key = "{}_{}_{}".format(tbl, col, doc)
                    if key in seen:
                        continue
                    seen.add(key)
                    retrieved.append({
                        "table": tbl, "column": col, "content": doc, "keyword": kw,
                    })
        elif keywords and db_id not in collections:
            counters["no_coll"] += 1

        results[qid] = {
            "db_id": db_id,
            "question": rec["question"],
            "evidence": rec.get("evidence", ""),
            "keywords": keywords,
            "retrieved": retrieved,
        }
        counters["done"] += 1
        if keywords:
            counters["kw_hit"] += 1
        if retrieved:
            counters["retr_hit"] += 1
        if counters["done"] % CHECKPOINT_EVERY == 0:
            _save_local(results)
            print(f"  [{counters['done']}/{len(kw_data)}] "
                  f"kw={counters['kw_hit']}, retr={counters['retr_hit']}, "
                  f"skip={counters['skipped']}, {time.time()-t0:.0f}s")

    _persist(results)
    if os.path.exists(CUR_FILE):
        os.remove(CUR_FILE)
    print(f"\n{'='*60}\nDONE! total={len(results)}")
    print(f"  kw_hit={counters['kw_hit']}, retr_hit={counters['retr_hit']}, "
          f"skipped={counters['skipped']}, missing_collection={counters['no_coll']}")
    print(f"  time={time.time()-t0:.0f}s")


def _save_local(results):
    tmp = OUTPUT_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    os.replace(tmp, OUTPUT_PATH)


def _persist(results):
    _save_local(results)
    try:
        os.makedirs(os.path.dirname(OSS_OUTPUT_PATH), exist_ok=True)
        with open(OSS_OUTPUT_PATH, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"  saved -> {OUTPUT_PATH} & {OSS_OUTPUT_PATH}")
    except Exception as e:
        print(f"  [OSS persist warn] {e}; local saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
