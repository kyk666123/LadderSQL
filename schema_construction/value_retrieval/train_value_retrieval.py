"""BIRD train 值向量检索（消费预提取关键词，不再调用 LLM）。

依赖：
- /root/bird_train_keywords.json （extract_keywords_train.py 产出：{qid:{db_id,question,evidence,keywords}}）
- /root/bird_train_chroma        （build_chroma_train.py 产出：每库一个 collection）

流程：读关键词 -> all-MiniLM-L6-v2 GPU 编码 -> chroma query_embeddings 检索对应库
-> 严格对齐 Agentar-Scale-SQL 值检索(DatabaseCellRetrieval.retrieve)：
   k=5、l2 距离<0.8 才保留、跳过纯数字关键词、table_column_content 去重、无数量上限，
   结果结构 {table, column, content}。

产物：
- /root/bird_train_value_retrieval.json
- /path/to/schema_link_results/bird_train/value_retrieval_train.json （OSS 持久化）
"""
import os
import json
import time

import chromadb
from sentence_transformers import SentenceTransformer

# ===================== CONFIG =====================
# 建库产物分布在多个目录：主目录(12 库) + 5 个 worker 目录(并行建的其余库)。
# worker 目录优先：同名库(如 codebase_comments)以 worker 目录的完整版为准，
# 主目录里的残缺版被覆盖忽略。
CHROMA_PATHS = (
    ["/root/bird_train_chroma_authors"]                    # authors 修复版(plan B)最优先，覆盖主目录段错误版
    + [f"/root/bird_train_chroma_w{k}" for k in range(5)]  # worker 目录优先
    + ["/root/bird_train_chroma"]                        # 主目录补剩下的
)
EMBEDDING_MODEL_PATH = "/path/to/embedding_models/all-MiniLM-L6-v2"
KEYWORDS_PATH = "/root/bird_train_keywords.json"

# 带关键词溯源的新版输出（旧的 value_retrieval_train.json 保留作备份）：
# retrieved 每一项额外记录触发该命中的关键词 keyword。
OUTPUT_PATH = "/root/bird_train_value_retrieval_kw.json"
OSS_OUTPUT_PATH = "/path/to/schema_link_results/bird_train/value_retrieval_train_kw.json"

# 与 Agentar-Scale-SQL 值检索严格对齐（DatabaseCellRetrieval.retrieve 默认参数）：
K = 5                 # n_results：每个关键词取 top-5
THRESHOLD = 0.8       # 仅保留 l2 距离 < 0.8 的命中
ENCODE_BATCH = 2048

# 断点续跑 / 段错误自动跳过：
CHECKPOINT_EVERY = 200                      # 每处理 N 条落盘一次（仅本地）
CUR_FILE = OUTPUT_PATH + ".current"          # 记录当前处理条目 (qid\tdb_id)，用于段错误定位
SKIP_DB_FILE = OUTPUT_PATH + ".skip_dbs"     # 触发段错误的库集合


def main():
    t0 = time.time()

    print("Loading keywords...")
    with open(KEYWORDS_PATH) as f:
        kw_data = json.load(f)
    print(f"  questions with keyword records: {len(kw_data)}")

    print("Loading SentenceTransformer on GPU...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL_PATH, device="cuda")

    print("Loading ChromaDB (聚合多目录, worker 优先)...")
    clients = []  # 保持引用避免被 GC
    collections = {}
    for path in CHROMA_PATHS:
        if not os.path.exists(path):
            continue
        try:
            cli = chromadb.PersistentClient(path=path)
        except Exception as e:
            print(f"  [warn] open {path} failed: {e}")
            continue
        clients.append(cli)
        n_new = 0
        for c in cli.list_collections():
            if c.name not in collections:   # 前面(worker)已有则不覆盖 -> worker 优先
                collections[c.name] = c
                n_new += 1
        print(f"  {path}: +{n_new} collections")
    print(f"  总 collections: {len(collections)}")

    # 1) 汇总所有唯一关键词，一次性 GPU 批量编码（避免逐条编码）
    all_kws = set()
    for rec in kw_data.values():
        for k in rec.get("keywords", []):
            # 对齐 Agentar：跳过非字符串 / 纯数字关键词
            if isinstance(k, str) and k.strip() and not k.isdigit():
                all_kws.add(k)
    all_kws = sorted(all_kws)
    print(f"  unique keywords to encode: {len(all_kws)}")

    kw2emb = {}
    if all_kws:
        embs = embed_model.encode(
            all_kws, batch_size=ENCODE_BATCH, show_progress_bar=False, device="cuda"
        ).tolist()
        kw2emb = dict(zip(all_kws, embs))

    # 2) 逐条问题检索（带断点续跑 + 段错误自动跳过）
    # chromadb/hnswlib 查询个别库时可能触发 C++ 段错误（无法被 Python 捕获，
    # 直接杀死进程）。因此：处理每条前把 (qid, db_id) 写入 CUR_FILE 并 fsync；
    # 定期将已完成结果落盘。重启时读已完成 qid 跳过；若 CUR_FILE 记录的 qid
    # 未完成，则判定其所属 db 触发了段错误，加入 SKIP_DB 集合，后续跳过该库全部问题。
    results = {}
    if os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH) as f:
                results = json.load(f)
            print(f"  resume: 已完成 {len(results)} 条，继续未完成部分")
        except Exception:
            results = {}

    skip_dbs = set()
    if os.path.exists(SKIP_DB_FILE):
        with open(SKIP_DB_FILE) as f:
            skip_dbs = {line.strip() for line in f if line.strip()}

    # 崩溃检测：上次正在处理但未完成的 qid -> 其 db 触发段错误
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
                print(f"  [段错误检测] 上次崩溃于 qid={last_qid} db={last_db}，已将该库加入跳过集")
    if skip_dbs:
        print(f"  跳过库(段错误): {sorted(skip_dbs)}")

    counters = {"done": len(results), "kw_hit": 0, "retr_hit": 0, "no_coll": 0, "skipped": 0}

    for qid, rec in kw_data.items():
        if qid in results:
            continue
        db_id = rec["db_id"]
        keywords = [k for k in rec.get("keywords", []) if isinstance(k, str) and k.strip()]
        retrieved = []

        # 记录当前正在处理的条目（用于段错误定位/跳过）
        with open(CUR_FILE, "w") as cf:
            cf.write(f"{qid}\t{db_id}")
            cf.flush()
            os.fsync(cf.fileno())

        if keywords and db_id in skip_dbs:
            counters["skipped"] += 1
        elif keywords and db_id in collections:
            collection = collections[db_id]
            seen = set()
            # 对齐 Agentar DatabaseCellRetrieval.retrieve：逐关键词遍历、按返回顺序去重累加
            for kw in keywords:
                if kw.isdigit():                      # 跳过纯数字关键词
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
                    if float(dist) >= THRESHOLD:      # 仅保留距离 < 0.8
                        continue
                    tbl = meta.get("table", "")
                    col = meta.get("column", "")
                    key = "{}_{}_{}".format(tbl, col, doc)   # 对齐 Agentar 去重 key
                    if key in seen:
                        continue
                    seen.add(key)
                    retrieved.append({
                        "table": tbl,
                        "column": col,
                        "content": doc,
                        "keyword": kw,      # 触发该命中的关键词（去重后保留首次命中的关键词）
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
    if os.path.exists(CUR_FILE):    # 正常完成，清理游标文件
        os.remove(CUR_FILE)
    elapsed = time.time() - t0
    print(f"\n{'='*60}\nDONE! total={len(results)}")
    print(f"  kw_hit={counters['kw_hit']}, retr_hit={counters['retr_hit']}, "
          f"skipped={counters['skipped']}, missing_collection={counters['no_coll']}")
    print(f"  time={elapsed:.0f}s")


def _save_local(results):
    tmp = OUTPUT_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    os.replace(tmp, OUTPUT_PATH)   # 原子替换，避免崩溃时写出残缺 JSON


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
