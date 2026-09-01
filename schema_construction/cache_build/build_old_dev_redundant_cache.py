"""Build the redundant (high-recall) schema cache for BIRD **old dev**, consumed at training
time by the validation split via truncate="cache", and merged with the existing train rendered
cache into a single file (the agent only ever loads one schema_linking_cache).

This reuses exactly the same pipeline as train (build_schema_cache_train.build_one / parse_light +
render_schema_prompt.render_table_info); only the inputs change to old dev:
  - APEX:  apex-sql-schema-link-results/bird_old_dev/results_glm-5.2.json (1534)
  - light: schema_preprocessing/db_light_schema/bird_dev_light_schema_glm52.json
  - VR:    apex-sql-schema-link-results/bird_old_dev/value_retrieval_old_dev_kw.json
           (produced by value_retrieval_old_dev.py; if missing, linked_values stays empty)

Outputs:
  1. bird_old_dev/schema_cache_old_dev.json            (structured candidate tree)
  2. bird_old_dev/schema_cache_old_dev_rendered.json   (rendered text, key=db_id|||question)
  3. bird_train/schema_cache_train_plus_olddev_rendered.json
        = train rendered cache U old dev rendered cache (what --schema-cache-path points at)
"""
import os
import sys
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from build_schema_cache_train import build_one, parse_light  # exactly the same merge logic as train
from render_schema_prompt import render_table_info            # exactly the same rendering as train

BASE = "/path/to/LadderSQL"
APEX_PATH   = f"{BASE}/apex-sql-schema-link-results/bird_old_dev/results_glm-5.2.json"
LIGHT_PATH  = f"{BASE}/schema_preprocessing/db_light_schema/bird_dev_light_schema_glm52.json"
VR_PATH     = f"{BASE}/apex-sql-schema-link-results/bird_old_dev/value_retrieval_old_dev_kw.json"
TRAIN_RENDERED = f"{BASE}/apex-sql-schema-link-results/bird_train/schema_cache_train_rendered.json"

OUT_STRUCT   = f"{BASE}/apex-sql-schema-link-results/bird_old_dev/schema_cache_old_dev.json"
OUT_RENDERED = f"{BASE}/apex-sql-schema-link-results/bird_old_dev/schema_cache_old_dev_rendered.json"
OUT_MERGED   = f"{BASE}/apex-sql-schema-link-results/bird_train/schema_cache_train_plus_olddev_rendered.json"


def _save(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main():
    print("loading inputs...")
    apex = json.load(open(APEX_PATH))
    light_raw = json.load(open(LIGHT_PATH))
    light = {db: parse_light(md) for db, md in light_raw.items()}
    vr = json.load(open(VR_PATH)) if os.path.exists(VR_PATH) else {}
    print(f"  apex={len(apex)}  light dbs={len(light)}  vr={len(vr)}"
          + ("" if vr else "  (no VR, linked_values left empty)"))

    # 1) structured candidate tree (including VR linked_values)
    struct = {}
    for qid, rec in apex.items():
        struct[qid] = build_one(qid, rec, vr.get(qid), light.get(rec["db_id"], {}))
    _save(OUT_STRUCT, struct)
    print(f"  [1/3] structured candidate tree -> {OUT_STRUCT} ({len(struct)})")

    # 2) render (key = db_id|||question, strictly aligned with the get_table_info cache branch)
    rendered = {}
    dup = 0
    for qid, s in struct.items():
        key = f"{s['db_id']}|||{s['question']}"
        if key in rendered:
            dup += 1
        rendered[key] = render_table_info(s)
    _save(OUT_RENDERED, rendered)
    avg = sum(len(v) for v in rendered.values()) / max(len(rendered), 1)
    print(f"  [2/3] rendered cache -> {OUT_RENDERED} ({len(rendered)} keys, dup={dup}, avg {avg:.0f} chars)")

    # 3) merge train U old dev (train wins; keys never actually collide, the questions differ)
    train_rendered = json.load(open(TRAIN_RENDERED))
    merged = dict(train_rendered)
    collide = sum(1 for k in rendered if k in merged)
    merged.update(rendered)
    _save(OUT_MERGED, merged)
    print(f"  [3/3] merged cache -> {OUT_MERGED}")
    print(f"        train={len(train_rendered)}  old_dev={len(rendered)}  merged={len(merged)}  key collisions={collide}")


if __name__ == "__main__":
    main()
