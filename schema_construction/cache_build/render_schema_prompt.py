"""把 schema_cache 的候选树渲染成嵌入 prompt 的紧凑文本形式。

设计：DDL 风格的分层文本（比 JSON 省 token、对 LLM 更友好）：
- 顶部给出 db / question / evidence；
- 候选 schema 明确标注「schema-linking 召回、含冗余、需自行取舍」；
- 每列一行：列名(类型): 描述；带 sample_values；
- 命中的向量值按「触发关键词」分组挂在列下，提示这是与问题强相关的实际单元格值。
"""
import json

CACHE_PATH = "/path/to/schema_link_results/bird_train/schema_cache_train.json"


def render_table_info(sample, max_samples=3):
    """只渲染候选 schema 主体（喂给 prompt 的 {table_info}）。

    不含 db/question/evidence 头部——因为 REACT_SQL_BIRD_REDUNDANT_PROMPT 模板
    自身已有 Question({input}) 与 Evidence({evidence}) 两块，避免重复。
    """
    lines = []
    for t in sample["candidate_schema"]:
        lines.append(f"# Table: {t['table']}")
        if t.get("table_description"):
            lines.append(f"  # {t['table_description']}")
        if t.get("primary_keys"):
            lines.append(f"  primary key: {', '.join(t['primary_keys'])}")
        for c in t["columns"]:
            samp = c.get("sample_values", [])
            if isinstance(samp, list):
                samp = samp[:max_samples]
            samp_str = ", ".join(json.dumps(s, ensure_ascii=False) for s in samp) if samp else ""
            lines.append(f"  - {c['column']} ({c.get('type','')}): {c.get('description','')}")
            if samp_str:
                lines.append(f"      examples: {samp_str}")
            lv = c.get("linked_values", [])
            if lv:
                by_kw = {}
                for it in lv:
                    by_kw.setdefault(it["keyword"], []).append(it["value"])
                for kw, vals in by_kw.items():
                    vals_str = ", ".join(json.dumps(v, ensure_ascii=False) for v in vals)
                    lines.append(f"      matched values (keyword=\"{kw}\"): {vals_str}")
        lines.append("")
    fks = sample.get("foreign_keys", [])
    if fks:
        lines.append("# Foreign keys (use these as the join conditions):")
        for fk in fks:
            lines.append(f"  {fk['from_table']}.{fk['from_column']} = {fk['to_table']}.{fk['to_column']}")
    return "\n".join(lines).rstrip()


def render_sample(sample, max_samples=3):
    lines = []
    lines.append(f"【Database】{sample['db_id']}")
    lines.append(f"【Question】{sample['question']}")
    if sample.get("evidence"):
        lines.append(f"【Evidence】{sample['evidence']}")
    lines.append("")
    lines.append("【Candidate Schema】(schema-linking 召回，可能包含冗余；请依据描述与命中值，"
                 "只选出回答问题真正需要的表和列)")
    for t in sample["candidate_schema"]:
        lines.append("")
        lines.append(f"# Table: {t['table']}")
        if t.get("table_description"):
            lines.append(f"  # {t['table_description']}")
        for c in t["columns"]:
            samp = c.get("sample_values", [])
            if isinstance(samp, list):
                samp = samp[:max_samples]
            samp_str = ", ".join(json.dumps(s, ensure_ascii=False) for s in samp) if samp else ""
            line = f"  - {c['column']} ({c.get('type','')}): {c.get('description','')}"
            lines.append(line)
            if samp_str:
                lines.append(f"      examples: {samp_str}")
            lv = c.get("linked_values", [])
            if lv:
                # 按触发关键词分组
                by_kw = {}
                for it in lv:
                    by_kw.setdefault(it["keyword"], []).append(it["value"])
                for kw, vals in by_kw.items():
                    vals_str = ", ".join(json.dumps(v, ensure_ascii=False) for v in vals)
                    lines.append(f"      matched values (keyword=\"{kw}\"): {vals_str}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    d = json.load(open(CACHE_PATH))
    qid = sys.argv[1] if len(sys.argv) > 1 else "6"
    print(render_sample(d[qid]))
