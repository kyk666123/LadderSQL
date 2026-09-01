"""Build BIRD *train* value ChromaDB with GPU-accelerated embeddings.

与 build_chroma_gpu.py 逻辑一致，仅将数据源换成 BIRD train 的 69 个库，
产物落到本地磁盘 /root/bird_train_chroma（快，避免 OSS 读写瓶颈）。

每个数据库 -> 一个 collection（名为 db_id）。
提取文本类型列去重值，跳过 PK / id 结尾列 / 关键词列，
用 all-MiniLM-L6-v2 在 GPU 上批量编码后直接写入 chroma（不设 embedding_function，
检索时由调用方用同一模型算 query_embeddings 传入）。
"""
import os
import time
import sqlite3
import logging
from uuid import uuid4

import torch
import chromadb
from sentence_transformers import SentenceTransformer

# ===================== CONFIG =====================
DATABASE_FOLDER = "/root/local_bird_db/train_databases"   # 69 个 train 库（已解压）
CHROMA_PATH = "/root/bird_train_chroma"                    # 本地磁盘，快
EMBEDDING_MODEL_PATH = "/path/to/embedding_models/all-MiniLM-L6-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4096
MAX_STR_LEN = 256
SKIP_KEYWORDS = ["_id", " id", "url", "email", "web", "time", "date", "address"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def get_db_values(db_path: str) -> list:
    """Extract all text column distinct values from a SQLite database."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    conn.text_factory = lambda b: b.decode(errors="ignore")
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    all_values = []  # list of (value, table_name, col_name)

    for (table_name,) in tables:
        cursor.execute(f"PRAGMA table_info(`{table_name}`);")
        columns = cursor.fetchall()
        # col: (cid, name, type, notnull, dflt_value, pk)

        for col in columns:
            col_name = col[1]
            col_type_raw = col[2] if col[2] else "UNKNOWN"
            col_type_lower = col_type_raw.lower()

            is_text_type = any(k in col_type_lower for k in ["text", "char", "clob"])
            if not is_text_type:
                continue
            if col[5] > 0:  # skip PK
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
                valid_values = [
                    (r[0], table_name, col_name)
                    for r in rows
                    if isinstance(r[0], str) and 0 < len(r[0]) <= MAX_STR_LEN
                ]
                all_values.extend(valid_values)
            except Exception as e:
                logger.warning(f"  Error querying {table_name}.{col_name}: {e}")

    conn.close()
    return all_values


def main():
    t0 = time.time()

    logger.info(f"Loading SentenceTransformer on {DEVICE}...")
    model = SentenceTransformer(EMBEDDING_MODEL_PATH, device=DEVICE)
    logger.info(f"Model loaded. Embedding dim: {model.get_sentence_embedding_dimension()}")

    os.makedirs(CHROMA_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    logger.info(f"ChromaDB initialized at {CHROMA_PATH}")

    db_names = sorted([
        d for d in os.listdir(DATABASE_FOLDER)
        if os.path.isdir(os.path.join(DATABASE_FOLDER, d)) and not d.startswith("__")
    ])
    logger.info(f"Found {len(db_names)} databases")

    existing = set(c.name for c in client.list_collections())
    total_vectors = 0

    for idx, db_name in enumerate(db_names, 1):
        db_path = os.path.join(DATABASE_FOLDER, db_name, f"{db_name}.sqlite")
        if not os.path.exists(db_path):
            logger.warning(f"[{idx}/{len(db_names)}] {db_name}: sqlite not found, skip")
            continue

        # 断点续跑：已建好且非空的 collection 跳过
        if db_name in existing:
            try:
                cnt = client.get_collection(db_name).count()
                if cnt > 0:
                    logger.info(f"[{idx}/{len(db_names)}] {db_name}: exists ({cnt} docs), skip")
                    total_vectors += cnt
                    continue
            except Exception:
                pass

        logger.info(f"[{idx}/{len(db_names)}] {db_name}: extracting values...")
        t1 = time.time()

        values_data = get_db_values(db_path)
        if not values_data:
            logger.warning(f"  {db_name}: no valid values found, creating empty collection")
            try:
                client.delete_collection(db_name)
            except Exception:
                pass
            client.create_collection(name=db_name)
            continue

        documents = [v[0] for v in values_data]
        metadatas = [{"table": v[1], "column": v[2]} for v in values_data]
        n = len(documents)

        logger.info(f"  {db_name}: {n} values, streaming encode+insert...")

        try:
            client.delete_collection(db_name)
        except Exception:
            pass
        collection = client.create_collection(name=db_name)

        # 分块「编码即写入」，把峰值内存限制在单块，避免超大库(百万级)OOM
        CHROMA_BATCH = 5000  # chromadb 1.1.1 单次 add 上限 5461
        for i in range(0, n, CHROMA_BATCH):
            end = min(i + CHROMA_BATCH, n)
            chunk_docs = documents[i:end]
            chunk_emb = model.encode(
                chunk_docs,
                batch_size=BATCH_SIZE,
                show_progress_bar=False,
                normalize_embeddings=False,
                device=DEVICE,
            ).tolist()
            collection.add(
                documents=chunk_docs,
                embeddings=chunk_emb,
                metadatas=metadatas[i:end],
                ids=[str(uuid4()) for _ in range(end - i)],
            )

        t3 = time.time()
        total_vectors += n
        logger.info(f"  {db_name}: DONE. {n} vectors, total={t3-t1:.1f}s")

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info(f"ALL DONE! {len(db_names)} databases, {total_vectors} total vectors")
    logger.info(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    logger.info(f"Output: {CHROMA_PATH}")

    # 校验：抽查每个 collection 能否用 query_embeddings 检索
    logger.info("Verifying query capability...")
    test_emb = model.encode(["test"], device=DEVICE).tolist()
    ok, fail = 0, 0
    for db_name in db_names:
        try:
            col = client.get_collection(db_name)
            col.query(query_embeddings=test_emb, n_results=3)
            ok += 1
        except Exception as e:
            fail += 1
            logger.error(f"  {db_name}: QUERY FAILED - {e}")
    logger.info(f"Verify done: {ok} OK, {fail} FAIL")


if __name__ == "__main__":
    main()
