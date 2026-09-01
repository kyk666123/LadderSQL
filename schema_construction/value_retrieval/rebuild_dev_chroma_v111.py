"""重建 BIRD dev 值向量库, 与安装的 chromadb 1.1.1 兼容。

背景: OSS 上现有 chroma/bird_dev 由更新版 chromadb(含 migration 00010) 构建,
      当前环境 chromadb 1.1.1(migration 仅到 00009) 打开会 rust panic。
      本环境无 pip(无法升级 chromadb), 故用仓库自带的 dev 建库逻辑
      (build_chroma_gpu.py 的抽取规则完全一致) 用 1.1.1 重建, 保证可读。

抽取规则与 build_chroma_gpu.py 完全一致:
  文本列(text/char/clob)、跳过PK、跳过以id结尾列、跳过 SKIP_KEYWORDS、DISTINCT、len<=256。
唯一差异: CHROMA_BATCH=5000 (chromadb 1.1.1 单次 add 上限 5461)。

源:  tasql_bird/dev_databases (11 标准 dev 库)
出:  /root/bird_dev_chroma_v111  (每库一个 collection, name=db_id)
"""
import os
import time
import sqlite3
import logging
from uuid import uuid4

import torch
import chromadb
from sentence_transformers import SentenceTransformer

DATABASE_FOLDER = "/root/dev_databases_local"   # 先从 OSS 复制到本地(OSS 直读 sqlite 会 disk I/O error)
CHROMA_PATH = "/root/bird_dev_chroma_v111"
EMBEDDING_MODEL_PATH = "/path/to/embedding_models/all-MiniLM-L6-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4096
MAX_STR_LEN = 256
SKIP_KEYWORDS = ["_id", " id", "url", "email", "web", "time", "date", "address"]
CHROMA_BATCH = 5000   # chromadb 1.1.1 单次 add 上限 5461

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def get_db_values(db_path: str) -> list:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    conn.text_factory = lambda b: b.decode(errors="ignore")
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    all_values = []
    for (table_name,) in tables:
        cursor.execute(f"PRAGMA table_info(`{table_name}`);")
        columns = cursor.fetchall()
        for col in columns:
            col_name = col[1]
            col_type_lower = (col[2] if col[2] else "UNKNOWN").lower()
            if not any(k in col_type_lower for k in ["text", "char", "clob"]):
                continue
            if col[5] > 0:
                continue
            if col_name.lower().endswith("id"):
                continue
            if any(k in col_name.lower() for k in SKIP_KEYWORDS):
                continue
            try:
                cursor.execute(
                    f"SELECT DISTINCT `{col_name}` FROM `{table_name}` WHERE `{col_name}` IS NOT NULL"
                )
                rows = cursor.fetchall()
                all_values.extend([
                    (r[0], table_name, col_name)
                    for r in rows
                    if isinstance(r[0], str) and 0 < len(r[0]) <= MAX_STR_LEN
                ])
            except Exception as e:
                logger.warning(f"  Error querying {table_name}.{col_name}: {e}")
    conn.close()
    return all_values


def main():
    t0 = time.time()
    logger.info(f"Loading SentenceTransformer on {DEVICE}...")
    model = SentenceTransformer(EMBEDDING_MODEL_PATH, device=DEVICE)
    logger.info(f"Model loaded. dim={model.get_sentence_embedding_dimension()}")

    os.makedirs(CHROMA_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    logger.info(f"ChromaDB at {CHROMA_PATH}")

    db_names = sorted([
        d for d in os.listdir(DATABASE_FOLDER)
        if os.path.isdir(os.path.join(DATABASE_FOLDER, d))
    ])
    logger.info(f"Found {len(db_names)} databases: {db_names}")

    total_vectors = 0
    for idx, db_name in enumerate(db_names, 1):
        db_path = os.path.join(DATABASE_FOLDER, db_name, f"{db_name}.sqlite")
        if not os.path.exists(db_path):
            logger.warning(f"[{idx}/{len(db_names)}] {db_name}: sqlite not found, skip")
            continue
        logger.info(f"[{idx}/{len(db_names)}] {db_name}: extracting values...")
        t1 = time.time()
        values_data = get_db_values(db_path)
        if not values_data:
            logger.warning(f"  {db_name}: no valid values found")
            continue
        documents = [v[0] for v in values_data]
        metadatas = [{"table": v[1], "column": v[2]} for v in values_data]
        ids = [str(uuid4()) for _ in values_data]
        logger.info(f"  {db_name}: {len(documents)} values, encoding on GPU...")
        embeddings = model.encode(
            documents, batch_size=BATCH_SIZE, show_progress_bar=False,
            normalize_embeddings=False, device=DEVICE,
        ).tolist()
        t2 = time.time()
        logger.info(f"  {db_name}: encode={t2-t1:.1f}s, inserting...")
        try:
            client.delete_collection(db_name)
        except Exception:
            pass
        collection = client.create_collection(name=db_name)
        for i in range(0, len(documents), CHROMA_BATCH):
            end = min(i + CHROMA_BATCH, len(documents))
            collection.add(
                documents=documents[i:end], embeddings=embeddings[i:end],
                metadatas=metadatas[i:end], ids=ids[i:end],
            )
        t3 = time.time()
        total_vectors += len(documents)
        logger.info(f"  {db_name}: DONE {len(documents)} vecs, "
                    f"encode={t2-t1:.1f}s insert={t3-t2:.1f}s")

    logger.info("=" * 60)
    logger.info(f"ALL DONE! {len(db_names)} dbs, {total_vectors} vectors, "
                f"{(time.time()-t0)/60:.1f}min -> {CHROMA_PATH}")

    logger.info("Verifying query capability...")
    for db_name in db_names:
        try:
            col = client.get_collection(db_name)
            test_emb = model.encode(["test"], device=DEVICE).tolist()
            r = col.query(query_embeddings=test_emb, n_results=3)
            n = len(r['documents'][0]) if r['documents'] else 0
            logger.info(f"  {db_name}: {col.count()} docs, query OK ({n})")
        except Exception as e:
            logger.error(f"  {db_name}: QUERY FAILED - {e}")


if __name__ == "__main__":
    main()
