"""并行构建 BIRD train 值向量库 (方案 A+B)。

背景/原理:
- chromadb 单 collection 写入是「单线程 Python 逐条记账 + 默认每 1000 条全量 pickle 落盘」,
  默认 batch_size=100 / sync_threshold=1000 导致大库 O(N^2) 落盘，吃不满多核。
- 方案 A: 跨库多进程并行，每个 worker 写各自独立的 chroma 目录(规避 SQLite 单写锁)。
- 方案 B: 调大 hnsw:batch_size / hnsw:sync_threshold，大幅减少落盘次数，消除 O(N^2)。

产物: /root/bird_train_chroma_w{0..N-1}，与已建好的 /root/bird_train_chroma(12 库) 并存。
检索时由 retrieval 脚本扫描所有目录聚合 (db_id -> collection 唯一)。
"""
import os
import sys
import time
import sqlite3
import logging
import multiprocessing as mp
from uuid import uuid4

# ===================== CONFIG =====================
DATABASE_FOLDER = "/root/local_bird_db/train_databases"
MAIN_CHROMA = "/root/bird_train_chroma"                 # 已建好的 12 库(保留)
WORKER_CHROMA_TPL = "/root/bird_train_chroma_w{k}"      # 每 worker 独立目录
EMBEDDING_MODEL_PATH = "/path/to/embedding_models/all-MiniLM-L6-v2"
MAX_STR_LEN = 256
SKIP_KEYWORDS = ["_id", " id", "url", "email", "web", "time", "date", "address"]

N_WORKERS = 5
ENCODE_BATCH = 1024          # GPU 编码批大小(小一点省显存, 5 workers 共享 ~8G)
CHROMA_ADD_BATCH = 5000      # chromadb 单次 add 上限 5461
# 方案 B: 大幅提高落盘阈值，消除 O(N^2)；num_threads 每 worker 限 6 避免超订 32 核
HNSW_META = {
    "hnsw:batch_size": 10000,
    "hnsw:sync_threshold": 100000,
    "hnsw:num_threads": 6,
    "hnsw:space": "l2",
}


def get_db_values(db_path):
    """提取所有文本列去重值 (与 build_chroma_train.py 完全一致的过滤规则)。"""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    conn.text_factory = lambda b: b.decode(errors="ignore")
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    all_values = []
    for (table_name,) in tables:
        cursor.execute(f"PRAGMA table_info(`{table_name}`);")
        for col in cursor.fetchall():
            col_name = col[1]
            col_type_lower = (col[2] if col[2] else "UNKNOWN").lower()
            if not any(k in col_type_lower for k in ["text", "char", "clob"]):
                continue
            if col[5] > 0 or col_name.lower().endswith("id"):
                continue
            if any(k in col_name.lower() for k in SKIP_KEYWORDS):
                continue
            try:
                cursor.execute(
                    f"SELECT DISTINCT `{col_name}` FROM `{table_name}` WHERE `{col_name}` IS NOT NULL"
                )
                for r in cursor.fetchall():
                    if isinstance(r[0], str) and 0 < len(r[0]) <= MAX_STR_LEN:
                        all_values.append((r[0], table_name, col_name))
            except Exception:
                pass
    conn.close()
    return all_values


def count_db_values(db_path):
    """快速 SQL COUNT，仅用于按值数量做负载均衡分配。"""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [t[0] for t in cur.fetchall()]
        total = 0
        for t in tables:
            cur.execute(f"PRAGMA table_info(`{t}`);")
            for col in cur.fetchall():
                name, ctype, pk = col[1], (col[2] or "").lower(), col[5]
                if not any(k in ctype for k in ["text", "char", "clob"]):
                    continue
                if pk > 0 or name.lower().endswith("id"):
                    continue
                if any(k in name.lower() for k in SKIP_KEYWORDS):
                    continue
                try:
                    cur.execute(
                        f"SELECT COUNT(*) FROM (SELECT DISTINCT `{name}` FROM `{t}` "
                        f"WHERE `{name}` IS NOT NULL AND typeof(`{name}`)='text' "
                        f"AND length(`{name}`) BETWEEN 1 AND {MAX_STR_LEN})"
                    )
                    total += cur.fetchone()[0]
                except Exception:
                    pass
        conn.close()
        return total
    except Exception:
        return 0


def build_worker(worker_id, db_list):
    """单个 worker 进程: 独立 chroma 目录，顺序建自己分到的库。"""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    import chromadb
    from sentence_transformers import SentenceTransformer

    log_path = f"/root/build_parallel_w{worker_id}.log"
    logger = logging.getLogger(f"w{worker_id}")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(fh)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[worker {worker_id}] start, {len(db_list)} dbs, device={device}, dbs={db_list}")
    model = SentenceTransformer(EMBEDDING_MODEL_PATH, device=device)

    chroma_path = WORKER_CHROMA_TPL.format(k=worker_id)
    os.makedirs(chroma_path, exist_ok=True)
    client = chromadb.PersistentClient(path=chroma_path)
    existing = {c.name for c in client.list_collections()}

    def encode(docs):
        try:
            return model.encode(docs, batch_size=ENCODE_BATCH, show_progress_bar=False,
                                 normalize_embeddings=False, device=device).tolist()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                logger.info("  CUDA OOM, fallback to CPU for this chunk")
                return model.encode(docs, batch_size=256, show_progress_bar=False,
                                    normalize_embeddings=False, device="cpu").tolist()
            raise

    for i, db_name in enumerate(db_list, 1):
        db_path = os.path.join(DATABASE_FOLDER, db_name, f"{db_name}.sqlite")
        if not os.path.exists(db_path):
            logger.info(f"[{i}/{len(db_list)}] {db_name}: sqlite missing, skip")
            continue
        # 断点续跑
        if db_name in existing:
            try:
                if client.get_collection(db_name).count() > 0:
                    logger.info(f"[{i}/{len(db_list)}] {db_name}: exists non-empty, skip")
                    continue
            except Exception:
                pass
        t1 = time.time()
        logger.info(f"[{i}/{len(db_list)}] {db_name}: extracting...")
        values = get_db_values(db_path)
        try:
            client.delete_collection(db_name)
        except Exception:
            pass
        if not values:
            client.create_collection(name=db_name, metadata=dict(HNSW_META))
            logger.info(f"  {db_name}: empty collection created")
            continue
        documents = [v[0] for v in values]
        metadatas = [{"table": v[1], "column": v[2]} for v in values]
        n = len(documents)
        logger.info(f"  {db_name}: {n} values, encoding+inserting...")
        collection = client.create_collection(name=db_name, metadata=dict(HNSW_META))
        for j in range(0, n, CHROMA_ADD_BATCH):
            end = min(j + CHROMA_ADD_BATCH, n)
            emb = encode(documents[j:end])
            collection.add(documents=documents[j:end], embeddings=emb,
                           metadatas=metadatas[j:end],
                           ids=[str(uuid4()) for _ in range(end - j)])
        logger.info(f"  {db_name}: DONE. {n} vectors, {time.time()-t1:.1f}s")
    logger.info(f"[worker {worker_id}] ALL DONE")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("launcher")

    # 1. 找出主目录已建好的库(count>0)，这些跳过
    import chromadb
    # 主目录默认视为已完成的库(不重复统计 count 以免大库卡顿)
    FORCE_REBUILD = {"codebase_comments"}  # 主目录里是残缺(默认参数)版，强制用 plan B 重建
    done = set()
    try:
        mc = chromadb.PersistentClient(path=MAIN_CHROMA)
        for c in mc.list_collections():
            if c.name in FORCE_REBUILD:
                continue
            done.add(c.name)
    except Exception as e:
        log.warning(f"read main chroma failed: {e}")
    log.info(f"主目录视为已完成 {len(done)} 库(跳过): {sorted(done)}")
    log.info(f"强制重建(plan B): {sorted(FORCE_REBUILD)}")

    all_dbs = sorted(d for d in os.listdir(DATABASE_FOLDER)
                     if os.path.isdir(os.path.join(DATABASE_FOLDER, d)) and not d.startswith("__"))
    todo = [d for d in all_dbs if d not in done]
    log.info(f"待建 {len(todo)} 库")

    # 2. 计算值数量并按降序 round-robin 分配到 N 个 worker(大库分散)
    log.info("统计各库值数量用于均衡分配...")
    counts = []
    for d in todo:
        p = os.path.join(DATABASE_FOLDER, d, f"{d}.sqlite")
        counts.append((d, count_db_values(p)))
    counts.sort(key=lambda x: x[1], reverse=True)
    log.info("值数量 Top10: " + ", ".join(f"{d}={n}" for d, n in counts[:10]))

    buckets = [[] for _ in range(N_WORKERS)]
    for idx, (d, n) in enumerate(counts):
        buckets[idx % N_WORKERS].append(d)
    for k, b in enumerate(buckets):
        log.info(f"worker{k}: {len(b)} 库 -> {b}")

    # 3. spawn workers (CUDA 必须 spawn)
    procs = []
    for k in range(N_WORKERS):
        if not buckets[k]:
            continue
        pr = mp.Process(target=build_worker, args=(k, buckets[k]))
        pr.start()
        procs.append(pr)
        log.info(f"启动 worker{k} PID={pr.pid}")
    for pr in procs:
        pr.join()
    log.info("=" * 50)
    log.info("并行建库全部完成 (ALL PARALLEL DONE)")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
