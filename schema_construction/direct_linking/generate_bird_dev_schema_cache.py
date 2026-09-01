#!/usr/bin/env python
"""Generate GLM-5 schema cache for BIRD dev dataset.

This script reads the BIRD parquet file and generates schema linking results
using GLM-5 model, using REACT_SQL_SCHEMA_GROUNDING_PROMPT from prompt.py.

Usage (one-click, all defaults pre-wired):
    python generate_bird_dev_schema_cache.py [--resume]

Or override any path:
    python generate_bird_dev_schema_cache.py \\
        --val-data /path/to/nl2sql_dataset/bird/bird_clean_data/dev_20251106.parquet \\
        --light-schema /path/to/LadderSQL/schema_construction/db_light_schema/bird_dev_light_schema.json \\
        --output /path/to/LadderSQL/schema_construction/relevant_schema_cache/bird_glm5_dev_schema_cache.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from langchain.chat_models import init_chat_model

# Import prompt from prompt.py (located at nl2sql_rl/agent/original_agent/prompt.py)
import sys
_REPO_AGENT_DIR = Path(__file__).resolve().parents[2] / "agent" / "original_agent"
sys.path.insert(0, str(_REPO_AGENT_DIR))
from prompt import REACT_SQL_SCHEMA_GROUNDING_PROMPT

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ================== Default Paths ==================

DEFAULT_VAL_DATA = "/path/to/nl2sql_dataset/bird/bird_clean_data/dev_20251106.parquet"
DEFAULT_LIGHT_SCHEMA = (
    "/path/to/LadderSQL/schema_construction/"
    "db_light_schema/bird_dev_light_schema.json"
)
DEFAULT_OUTPUT = "/path/to/bird_glm5_dev_schema_cache.json"


def extract_schema_from_response(response: str) -> str:
    """Extract schema content from GLM-5 response."""
    pattern = r"<schema>(.*?)</schema>"
    match = re.search(pattern, response, re.DOTALL)
    
    if match:
        raw_schema = match.group(1).strip()
        prefix = "Relevant tables, columns retrieved by llm:"
        return f"{prefix}\n{raw_schema}"
    else:
        logger.warning("No <schema> tag found in response")
        return ""


def generate_schema_for_question(
    llm: Any,
    question: str,
    light_schema: str,
) -> str:
    """Generate schema linking for a single question using REACT_SQL_SCHEMA_GROUNDING_PROMPT."""
    prompt = REACT_SQL_SCHEMA_GROUNDING_PROMPT.invoke({
        "question": question,
        "schema": light_schema,
    })
    
    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        return extract_schema_from_response(content)
    except Exception as e:
        logger.error(f"Failed to generate schema for question: {e}")
        return ""


def load_light_schema(light_schema_path: str, db_id: str) -> str:
    """Load light schema for a database from the given JSON file."""
    path = Path(light_schema_path)
    
    if path.exists():
        with open(path, "r") as f:
            light_schemas = json.load(f)
        if db_id in light_schemas:
            return light_schemas[db_id]
    
    logger.warning(f"Light schema not found for db_id: {db_id}")
    return ""


def main():
    parser = argparse.ArgumentParser(description="Generate GLM-5 schema cache for BIRD dev dataset")
    parser.add_argument("--val-data", type=str, default=DEFAULT_VAL_DATA,
                        help=f"Path to BIRD dev parquet file (default: {DEFAULT_VAL_DATA})")
    parser.add_argument("--light-schema", type=str, default=DEFAULT_LIGHT_SCHEMA,
                        help=f"Path to bird_dev_light_schema.json (default: {DEFAULT_LIGHT_SCHEMA})")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help=f"Output cache file path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--resume", action="store_true", help="Resume from existing cache")
    args = parser.parse_args()
    
    # Initialize GLM-5
    logger.info("Initializing GLM-5 model...")
    llm = init_chat_model(
        "glm-5",
        model_provider="openai",
        openai_api_base=os.environ["ALICLOUD_BASE_URL"],
        openai_api_key=os.environ["ALICLOUD_API_KEY"],
        temperature=0.0,
        max_retries=3,
        max_tokens=4096,
        timeout=120,
    )
    
    # Load light schema
    logger.info(f"Loading light schema from {args.light_schema}")
    with open(args.light_schema, "r") as f:
        light_schemas = json.load(f)
    logger.info(f"Loaded {len(light_schemas)} light schemas")
    
    # Load validation data
    logger.info(f"Loading validation data from {args.val_data}")
    val_data = pd.read_parquet(args.val_data)
    logger.info(f"Loaded {len(val_data)} validation samples")
    
    # Load existing cache if resuming
    cache: Dict[str, Dict[str, Any]] = {}
    if args.resume and Path(args.output).exists():
        with open(args.output, "r") as f:
            cache = json.load(f)
        logger.info(f"Resumed from existing cache with {len(cache)} entries")
    
    # Generate schema cache
    total = len(val_data)
    processed = 0
    skipped = 0
    
    for idx, row in val_data.iterrows():
        question = row["question"]
        db_id = row["db_id"]
        question_key = f"{db_id}|||{question}"
        
        # Check if already in cache (by question_key)
        already_cached = False
        for entry in cache.values():
            if entry.get("question_key") == question_key:
                already_cached = True
                skipped += 1
                break
        
        if already_cached:
            continue
        
        # Load light schema for this db_id
        light_schema = light_schemas.get(db_id, "")
        if not light_schema:
            logger.warning(f"Skipping {question_key}: no light schema available")
            skipped += 1
            continue
        
        # Generate schema
        logger.info(f"[{idx+1}/{total}] Processing: {db_id} - {question[:50]}...")
        schema = generate_schema_for_question(llm, question, light_schema)
        
        if schema:
            # Create cache entry with same format as schema_cache_20260415.json
            cache_entry = {
                "db_id": db_id,
                "question": question,
                "rollout_id": f"bird_{idx}",
                "question_key": question_key,
                "schema": schema,
            }
            cache[f"bird_{idx}"] = cache_entry
            processed += 1
            
            # Save periodically
            if (idx + 1) % 50 == 0:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                with open(args.output, "w") as f:
                    json.dump(cache, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved intermediate cache: {len(cache)} entries")
        else:
            logger.warning(f"Failed to generate schema for: {question_key}")
    
    # Final save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Schema cache generation complete!")
    logger.info(f"Total samples: {total}")
    logger.info(f"Newly processed: {processed}")
    logger.info(f"Skipped (already cached): {skipped}")
    logger.info(f"Final cache size: {len(cache)}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
