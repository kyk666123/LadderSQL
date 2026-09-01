"""生成 bird train 每个样本的 schema cache（合并候选树）。

融合三个来源：
1. APEX-SQL 召回的表/列  (apex-sql-schema-link-results/bird_train/results_glm-5.2.json)
   —— 高召回、高冗余，作为候选树骨架。
2. light schema 的表/列描述 + 类型 + 样例值
   (schema_preprocessing/db_light_schema/bird_train_light_schema_glm52.json)
   —— 供模型据描述剪枝。
3. 向量检索命中的值 + 触发的关键词
   (apex-sql-schema-link-results/bird_train/value_retrieval_train_kw.json)
   —— 作为 linked_values 挂到对应列节点。

合并/去重规则：
- 候选集 = APEX 召回的表列 ∪ 向量检索命中的表列（不含全库无关表）。
- 向量检索命中的列若 APEX 已召回 -> 值追加进该列 linked_values（不新建节点）。
- 向量检索命中、APEX 漏召的列 -> 新建列节点补召回，描述同样取自 light schema。

产物：apex-sql-schema-link-results/bird_train/schema_cache_train.json
每列字段：column / type / description / sample_values / linked_values[{value, keyword}]
（按用户要求：不含 relevance_reason、不含 source）
"""
import os
import re
import json

BASE = "/path/to/LadderSQL"
APEX_PATH  = f"{BASE}/apex-sql-schema-link-results/bird_train/results_glm-5.2.json"
VR_PATH    = f"{BASE}/apex-sql-schema-link-results/bird_train/value_retrieval_train_kw.json"
LIGHT_PATH = f"{BASE}/schema_preprocessing/db_light_schema/bird_train_light_schema_glm52.json"
OUT_PATH   = f"{BASE}/apex-sql-schema-link-results/bird_train/schema_cache_train.json"


def parse_light(md):
    """light schema markdown -> {table: {desc, columns:{col:{type,description,samples}}}}"""
    tables = {}
    fks = []  # (t1, c1, t2, c2)
    cur = None
    cur_col = None
    sec = None
    for line in md.splitlines():
        m = re.match(r'^## Table:\s*(.+)$', line)
        if m:
            cur = m.group(1).strip()
            tables[cur] = {"desc": "", "columns": {}, "pk": []}
            cur_col = None
            sec = None
            continue
        if cur is None:
            continue
        if line.startswith('### Table description'):
            sec = 'tdesc'; continue
        if line.startswith('### Column information'):
            sec = 'cols'; continue
        if line.startswith('### Primary keys'):
            sec = 'pk'; continue
        if line.startswith('### Foreign keys'):
            sec = 'fk'; continue
        if sec == 'tdesc' and line.strip():
            tables[cur]["desc"] += (" " if tables[cur]["desc"] else "") + line.strip()
        elif sec == 'cols':
            m = re.match(r'^-\s*Column:\s*(.+)$', line)
            if m:
                cur_col = m.group(1).strip()
                tables[cur]["columns"][cur_col] = {"type": "", "description": "", "samples": []}
                continue
            if cur_col:
                mt = re.match(r'^\s*-\s*Type:\s*(.+)$', line)
                md_ = re.match(r'^\s*-\s*Description:\s*(.+)$', line)
                ms = re.match(r'^\s*-\s*Samples:\s*(.+)$', line)
                if mt:
                    tables[cur]["columns"][cur_col]["type"] = mt.group(1).strip()
                elif md_:
                    tables[cur]["columns"][cur_col]["description"] = md_.group(1).strip()
                elif ms:
                    try:
                        tables[cur]["columns"][cur_col]["samples"] = json.loads(ms.group(1).strip())
                    except Exception:
                        tables[cur]["columns"][cur_col]["samples"] = ms.group(1).strip()
        elif sec == 'pk' and line.strip() and not line.startswith('#'):
            for part in line.split(','):
                p = part.strip()
                if p:
                    tables[cur]["pk"].append(p)
        elif sec == 'fk' and line.strip() and not line.startswith('#'):
            mf = re.match(r'^\s*(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)\s*$', line)
            if mf:
                fks.append((mf.group(1), mf.group(2), mf.group(3), mf.group(4)))
    return {"tables": tables, "fks": fks}


def build_one(qid, apex_rec, vr_rec, light):
    db = apex_rec["db_id"]
    light_tables = light.get("tables", {})
    light_fks = light.get("fks", [])
    cand = {}  # table -> {table, table_description, columns: {col: node}}

    def ensure_table(t):
        if t not in cand:
            lt = light_tables.get(t, {"desc": "", "columns": {}, "pk": []})
            cand[t] = {"table": t, "table_description": lt["desc"], "columns": {}}
        return cand[t]

    def new_col_node(t, col):
        lc = light_tables.get(t, {"columns": {}})["columns"].get(col, {})
        return {
            "column": col,
            "type": lc.get("type", ""),
            "description": lc.get("description", ""),
            "sample_values": lc.get("samples", []),
            "linked_values": [],
        }

    def ensure_col(t, col):
        node = ensure_table(t)
        if col not in node["columns"]:
            node["columns"][col] = new_col_node(t, col)
        return node["columns"][col]

    # 1) APEX 召回的表列
    for tfull, info in apex_rec["result"]["refined_schema"].items():
        t = tfull.split(".", 1)[1] if "." in tfull else tfull
        ensure_table(t)
        for c in info.get("relevant_columns", []):
            ensure_col(t, c["column_name"])

    # 2) 向量检索命中值：只挂到 APEX 已召回的列上（不新增列/表），去重
    # 理由：向量检索能识别值的精确格式（修正查空）；但命中在 APEX 未召回列上的
    # 大多是巧合噪声（APEX 语义 linker 已判定该列无关），故丢弃。
    for r in (vr_rec.get("retrieved", []) if vr_rec else []):
        t, col, val, kw = r["table"], r["column"], r["content"], r.get("keyword", "")
        if t not in cand or col not in cand[t]["columns"]:
            continue
        cn = cand[t]["columns"][col]
        if not any(lv["value"] == val for lv in cn["linked_values"]):
            cn["linked_values"].append({"value": val, "keyword": kw})

    # 3) 主键：纯标注，不补列。只在「已被召回的列」上标注主键
    for t in list(cand.keys()):
        pk = light_tables.get(t, {}).get("pk", [])
        present = cand[t]["columns"]
        cand[t]["primary_keys"] = [c for c in pk if c in present]

    # 4) 外键 / JOIN 关系：仅当两端 join 列都已被召回（两端表自然都存在）才保留，去重；不补列
    foreign_keys = []
    seen = set()
    for (t1, c1, t2, c2) in light_fks:
        if (t1 in cand and c1 in cand[t1]["columns"]
                and t2 in cand and c2 in cand[t2]["columns"]):
            key = tuple(sorted([f"{t1}.{c1}", f"{t2}.{c2}"]))
            if key in seen:
                continue
            seen.add(key)
            foreign_keys.append({"from_table": t1, "from_column": c1,
                                 "to_table": t2, "to_column": c2})

    candidate_schema = [
        {"table": n["table"], "table_description": n["table_description"],
         "primary_keys": n.get("primary_keys", []),
         "columns": list(n["columns"].values())}
        for n in cand.values()
    ]
    return {
        "question_id": apex_rec["question_id"],
        "db_id": db,
        "question": apex_rec["question"],
        "evidence": apex_rec.get("evidence", ""),
        "candidate_schema": candidate_schema,
        "foreign_keys": foreign_keys,
    }


def main():
    print("loading inputs...")
    apex = json.load(open(APEX_PATH))
    vr = json.load(open(VR_PATH))
    light_raw = json.load(open(LIGHT_PATH))
    light = {db: parse_light(md) for db, md in light_raw.items()}
    print(f"  apex={len(apex)}, vr={len(vr)}, light dbs={len(light)}")

    out = {}
    n_tables = n_cols = n_vals = n_fks = 0
    for qid, rec in apex.items():
        db = rec["db_id"]
        item = build_one(qid, rec, vr.get(qid), light.get(db, {}))
        out[qid] = item
        n_tables += len(item["candidate_schema"])
        n_fks += len(item["foreign_keys"])
        for t in item["candidate_schema"]:
            n_cols += len(t["columns"])
            n_vals += sum(len(c["linked_values"]) for c in t["columns"])

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    tmp = OUT_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    os.replace(tmp, OUT_PATH)
    print(f"DONE! {len(out)} samples -> {OUT_PATH}")
    print(f"  avg candidate tables/sample: {n_tables/len(out):.2f}")
    print(f"  avg candidate columns/sample: {n_cols/len(out):.2f}")
    print(f"  total linked_values: {n_vals}")
    print(f"  avg foreign_keys/sample: {n_fks/len(out):.2f}")


if __name__ == "__main__":
    main()
