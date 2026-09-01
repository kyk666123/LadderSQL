"""Build Bird dev ChromaDB with GPU-accelerated embeddings.

Uses SentenceTransformer on CUDA for batch encoding, then inserts
pre-computed embeddings into ChromaDB (no monkey-patch needed).
"""
import os
import sys
import time
import sqlite3
import logging
from uuid import uuid4

import torch
import chromadb
from sentence_transformers import SentenceTransformer

# ===================== CONFIG =====================
DATABASE_FOLDER = "/root/bird_eval/dev_databases"
CHROMA_PATH = "/root/bird_dev_chroma"
EMBEDDING_MODEL_PATH = "/path/to/embedding_models/all-MiniLM-L6-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4096  # GPU can handle large batches
MAX_STR_LEN = 256
SKIP_KEYWORDS = ["_id", " id", "url", "email", "web", "time", "date", "address"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def get_db_values(db_path: str) -> list:
    """Extract all text column values from a SQLite database."""
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

            # Only text-type columns
            is_text_type = any(k in col_type_lower for k in ["text", "char", "clob"])
            if not is_text_type:
                continue

            # Skip PK
            if col[5] > 0:
                continue

            # Skip columns ending with 'id'
            if col_name.lower().endswith("id"):
                continue

            # Skip by keywords
            if any(k in col_name.lower() for k in SKIP_KEYWORDS):
                continue

            # Extract distinct values
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

    # Init model on GPU
    logger.info(f"Loading SentenceTransformer on {DEVICE}...")
    model = SentenceTransformer(EMBEDDING_MODEL_PATH, device=DEVICE)
    logger.info(f"Model loaded. Embedding dim: {model.get_sentence_embedding_dimension()}")

    # Init ChromaDB (fresh)
    os.makedirs(CHROMA_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    logger.info(f"ChromaDB initialized at {CHROMA_PATH}")

    # Get all database folders
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

        # Extract values
        values_data = get_db_values(db_path)
        if not values_data:
            logger.warning(f"  {db_name}: no valid values found")
            continue

        documents = [v[0] for v in values_data]
        metadatas = [{"table": v[1], "column": v[2]} for v in values_data]
        ids = [str(uuid4()) for _ in values_data]

        logger.info(f"  {db_name}: {len(documents)} values extracted, encoding on GPU...")

        # GPU batch encode
        embeddings = model.encode(
            documents,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=False,
            device=DEVICE,
        ).tolist()

        t2 = time.time()
        logger.info(f"  {db_name}: encoding done in {t2-t1:.1f}s, inserting to ChromaDB...")

        # Create collection and insert (no embedding function - we provide embeddings directly)
        try:
            client.delete_collection(db_name)
        except:
            pass
        collection = client.create_collection(name=db_name)

        # Insert in batches (ChromaDB has a limit per call)
        CHROMA_BATCH = 40000
        for i in range(0, len(documents), CHROMA_BATCH):
            end = min(i + CHROMA_BATCH, len(documents))
            collection.add(
                documents=documents[i:end],
                embeddings=embeddings[i:end],
                metadatas=metadatas[i:end],
                ids=ids[i:end],
            )

        t3 = time.time()
        total_vectors += len(documents)
        logger.info(
            f"  {db_name}: DONE. {len(documents)} vectors, "
            f"encode={t2-t1:.1f}s, insert={t3-t2:.1f}s, total={t3-t1:.1f}s"
        )

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info(f"ALL DONE! {len(db_names)} databases, {total_vectors} total vectors")
    logger.info(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    logger.info(f"Output: {CHROMA_PATH}")

    # Verify all collections can be queried
    logger.info("Verifying query capability...")
    for db_name in db_names:
        try:
            col = client.get_collection(db_name)
            # Test query with pre-computed embedding
            test_emb = model.encode(["test"], device=DEVICE).tolist()
            results = col.query(query_embeddings=test_emb, n_results=3)
            n = len(results['documents'][0]) if results['documents'] else 0
            logger.info(f"  {db_name}: {col.count()} docs, test query OK ({n} results)")
        except Exception as e:
            logger.error(f"  {db_name}: QUERY FAILED - {e}")


if __name__ == "__main__":
    main()
