"""
Schema Linking v2 Pipeline
对 BIRD dev 1534 题批量执行新的 schema linking (GLM5.2 + thinking mode)。

使用方式:
    python schema_linking_v2.py

输出:
    relevant_schema_cache/bird_dev_schema_linking_v2.json
"""
import asyncio
import json
import os
import re
import time
from pathlib import Path

# Ensure env vars are set
if not os.environ.get('ALICLOUD_API_KEY'):
    raise RuntimeError("ALICLOUD_API_KEY is not set; export it before running")

from openai import AsyncOpenAI

from schema_linking_v2_prompt import (
    SYSTEM_PROMPT,
    build_db_schema_context,
    build_user_prompt,
)

# ── 配置 ──
MODEL = "glm-5.2"
MAX_CONCURRENT = 30
TEMPERATURE = 0.0
MAX_TOKENS = 8192
MAX_RETRIES = 3

# ── 路径 ──
BASE_DIR = Path(__file__).parent
LIGHT_SCHEMA_FILE = BASE_DIR / "db_light_schema" / "bird_dev_light_schema.json"
COLUMN_PROFILES_FILE = BASE_DIR / "column_profiles_bird_dev.json"
JOIN_GRAPH_FILE = BASE_DIR / "join_graph_bird_dev.json"
DEV_FILE = Path("/path/to/nl2sql_dataset/bird/dev_20240627/dev.json")
OUTPUT_FILE = BASE_DIR / "relevant_schema_cache" / "bird_dev_schema_linking_v2.json"

# ── 客户端 ──
client = AsyncOpenAI(
    api_key=os.environ.get("ALICLOUD_API_KEY"),
    base_url=os.environ.get("ALICLOUD_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    timeout=60.0,
)


def parse_schema_output(content: str) -> str | None:
    """Parse <schema>...</schema> from model output (explicit thinking mode)."""
    if not content:
        return None
    
    # Extract <schema> block
    match = re.search(r'<schema>(.*?)</schema>', content, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Fallback: content starts with or contains "Table:"
    idx = content.find("Table:\n- ")
    if idx >= 0:
        tail = content[idx:]
        if "Columns:" in tail:
            return tail.strip()
    
    return None


async def process_single(
    semaphore: asyncio.Semaphore,
    key: str,
    db_id: str,
    question: str,
    evidence: str,
    db_schema_text: str,
) -> tuple[str, str | None]:
    """Process a single question through schema linking."""
    async with semaphore:
        user_prompt = build_user_prompt(question, evidence, db_schema_text)
        
        for attempt in range(MAX_RETRIES):
            try:
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=MODEL,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                    ),
                    timeout=90.0,
                )
                content = response.choices[0].message.content or ""
                schema_text = parse_schema_output(content)
                
                if schema_text:
                    return key, schema_text
                else:
                    print(f"  [WARN] No schema parsed for {key[:60]}... (attempt {attempt+1})", flush=True)
                    
            except asyncio.TimeoutError:
                print(f"  [TIMEOUT] {key[:50]}... attempt {attempt+1}", flush=True)
            except Exception as e:
                wait = 2 * (2 ** attempt)  # 2s, 4s, 8s
                print(f"  [ERR] {key[:50]}... attempt {attempt+1}: {e}", flush=True)
                await asyncio.sleep(wait)
        
        return key, None


async def main():
    print("=" * 60)
    print("Schema Linking v2 Pipeline")
    print("=" * 60)
    
    # Load resources
    print("\n[1/5] Loading light schema...")
    with open(LIGHT_SCHEMA_FILE, 'r', encoding='utf-8') as f:
        light_schema = json.load(f)
    print(f"  Loaded {len(light_schema)} databases")
    
    print("[2/5] Loading column profiles...")
    with open(COLUMN_PROFILES_FILE, 'r', encoding='utf-8') as f:
        column_profiles = json.load(f)
    print(f"  Loaded profiles for {len(column_profiles)} databases")
    
    print("[3/5] Loading join graph...")
    with open(JOIN_GRAPH_FILE, 'r', encoding='utf-8') as f:
        join_graph = json.load(f)
    print(f"  Loaded join graph for {len(join_graph)} databases")
    
    print("[4/5] Loading dev questions...")
    with open(DEV_FILE, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    print(f"  Loaded {len(dev_data)} questions")
    
    # Load existing cache for resume
    cache = {}
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        print(f"  Resuming: {len(cache)} already completed")
    else:
        # Check if there was a partial file from interrupted run
        print("  Starting fresh")
    
    # Prepare tasks
    print("[5/5] Preparing tasks...")
    tasks_to_run = []
    
    for item in dev_data:
        db_id = item['db_id']
        question = item['question']
        evidence = item.get('evidence', '')
        key = f"{db_id}|||{question}"
        
        # Skip if already cached
        if key in cache:
            continue
        
        # Build schema context
        if db_id not in light_schema:
            print(f"  [SKIP] No light schema for db: {db_id}")
            continue
        
        db_schema_text = build_db_schema_context(
            db_id=db_id,
            light_schema_text=light_schema[db_id],
            column_profiles=column_profiles,
            join_graph=join_graph,
        )
        
        tasks_to_run.append((key, db_id, question, evidence, db_schema_text))
    
    total = len(tasks_to_run)
    print(f"\n  Tasks to process: {total}")
    print(f"  Concurrency: {MAX_CONCURRENT}")
    print(f"  Model: {MODEL} (explicit-think, temp={TEMPERATURE})")
    
    if total == 0:
        print("\n✓ All questions already processed!")
        return
    
    # Process with semaphore
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    start_time = time.time()
    completed = 0
    failed = 0
    
    # Process in batches for periodic saving
    BATCH_SIZE = 100
    
    for batch_start in range(0, total, BATCH_SIZE):
        batch = tasks_to_run[batch_start:batch_start + BATCH_SIZE]
        
        coros = [
            process_single(semaphore, key, db_id, question, evidence, db_schema_text)
            for key, db_id, question, evidence, db_schema_text in batch
        ]
        
        results = await asyncio.gather(*coros)
        
        for key, schema_text in results:
            if schema_text:
                cache[key] = schema_text
                completed += 1
            else:
                failed += 1
        
        # Save checkpoint
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        
        elapsed = time.time() - start_time
        rate = (completed + failed) / elapsed if elapsed > 0 else 0
        print(
            f"  Progress: {batch_start + len(batch)}/{total} "
            f"(ok={completed}, fail={failed}, "
            f"{rate:.1f} q/s, elapsed={elapsed:.0f}s)",
            flush=True,
        )
    
    # Final stats
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"COMPLETE")
    print(f"  Total processed: {completed + failed}")
    print(f"  Success: {completed}")
    print(f"  Failed: {failed}")
    print(f"  Cache size: {len(cache)}")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Output: {OUTPUT_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
