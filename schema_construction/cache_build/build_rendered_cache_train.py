"""把 schema_cache_train.json 预渲染成 agent 的 "cache" 模式可直接消费的字符串缓存。

背景：original_agent.ReAct_Agent.LitAgent.get_table_info 的 truncate=="cache" 分支：
    question_key = f"{rollout_db_id}|||{rollout_question}"
    return self.schema_linking_cache[question_key]
即缓存格式为 {"db_id|||question": "<schema 文本>"}。

本脚本把结构化的 schema_cache_train.json（候选树）用 render_table_info() 渲染成
喂给 prompt {table_info} 的文本，并以 db_id|||question 为 key 落盘。

产物：apex-sql-schema-link-results/bird_train/schema_cache_train_rendered.json
训练/评测脚本用 truncate="cache" + schema_cache_path=<该文件> 即可，无需改 agent 代码。
"""
import os
import json
import sys

sys.path.insert(0, os.path.dirname(__file__))
from render_schema_prompt import render_table_info

BASE = "/path/to/LadderSQL"
CACHE_PATH = f"{BASE}/apex-sql-schema-link-results/bird_train/schema_cache_train.json"
OUT_PATH = f"{BASE}/apex-sql-schema-link-results/bird_train/schema_cache_train_rendered.json"


def main():
    d = json.load(open(CACHE_PATH))
    rendered = {}
    dup = 0
    for qid, s in d.items():
        key = f"{s['db_id']}|||{s['question']}"     # 严格对齐 get_table_info cache 分支
        if key in rendered:
            dup += 1
        rendered[key] = render_table_info(s)

    tmp = OUT_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rendered, f, ensure_ascii=False, indent=2)
    os.replace(tmp, OUT_PATH)

    avg_chars = sum(len(v) for v in rendered.values()) / max(len(rendered), 1)
    print(f"DONE! {len(d)} samples -> {len(rendered)} keys (dup collapsed: {dup})")
    print(f"  avg table_info chars: {avg_chars:.0f}")
    print(f"  -> {OUT_PATH}")


if __name__ == "__main__":
    main()
