"""为 BIRD **old dev** 生成冗余(高召回) schema 缓存, 供训练时 val 用 truncate="cache" 消费,
并与已有的 train 渲染缓存合并成一个文件(agent 只加载单一 schema_linking_cache)。

复用 train 的完全相同流程 (build_schema_cache_train.build_one / parse_light +
render_schema_prompt.render_table_info), 只把输入换成 old dev:
  - APEX:  apex-sql-schema-link-results/bird_old_dev/results_glm-5.2.json (1534)
  - light: schema_preprocessing/db_light_schema/bird_dev_light_schema_glm52.json
  - VR:    apex-sql-schema-link-results/bird_old_dev/value_retrieval_old_dev_kw.json
           (value_retrieval_old_dev.py 产出; 缺失则 linked_values 为空)

产物:
  1. bird_old_dev/schema_cache_old_dev.json            (结构化候选树)
  2. bird_old_dev/schema_cache_old_dev_rendered.json   (渲染文本, key=db_id|||question)
  3. bird_train/schema_cache_train_plus_olddev_rendered.json
        = train 渲染缓存 ∪ old dev 渲染缓存 (训练脚本 --schema-cache-path 指向它)
"""
import os
import sys
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from build_schema_cache_train import build_one, parse_light  # 与 train 完全一致的合并逻辑
from render_schema_prompt import render_table_info            # 与 train 完全一致的渲染

BASE = "/path/to/LadderSQL"
APEX_PATH   = f"{BASE}/apex-sql-schema-link-results/bird_clean_dev/results_glm-5.2.json"
LIGHT_PATH  = f"{BASE}/schema_preprocessing/db_light_schema/bird_dev_light_schema_glm52.json"
VR_PATH     = f"{BASE}/apex-sql-schema-link-results/bird_clean_dev/value_retrieval_clean_dev_kw.json"

OUT_STRUCT   = f"{BASE}/apex-sql-schema-link-results/bird_clean_dev/schema_cache_clean_dev.json"
OUT_RENDERED = f"{BASE}/apex-sql-schema-link-results/bird_clean_dev/schema_cache_clean_dev_rendered.json"


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
          + ("" if vr else "  (无VR, linked_values置空)"))

    # 1) 结构化候选树 (含 VR linked_values)
    struct = {}
    for qid, rec in apex.items():
        struct[qid] = build_one(qid, rec, vr.get(qid), light.get(rec["db_id"], {}))
    _save(OUT_STRUCT, struct)
    print(f"  [1/3] 结构化候选树 -> {OUT_STRUCT} ({len(struct)})")

    # 2) 渲染 (key = db_id|||question, 严格对齐 get_table_info cache 分支)
    rendered = {}
    dup = 0
    for qid, s in struct.items():
        key = f"{s['db_id']}|||{s['question']}"
        if key in rendered:
            dup += 1
        rendered[key] = render_table_info(s)
    _save(OUT_RENDERED, rendered)
    avg = sum(len(v) for v in rendered.values()) / max(len(rendered), 1)
    print(f"  [2/2] 渲染缓存 -> {OUT_RENDERED} ({len(rendered)} keys, dup={dup}, avg {avg:.0f} chars)")


if __name__ == "__main__":
    main()
