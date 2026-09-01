"""
Step 0b: 整合 ATT short_profile + light_schema 表描述
为精筛环节提供统一的元信息查询接口

输出格式:
{
    "financial": {
        "table_descriptions": {
            "account": "This table stores information about bank accounts...",
            ...
        },
        "column_profiles": {
            "account.account_id": "Account ID is a unique integer primary key ranging from 1 to 11382...",
            "account.district_id": "Integer identifier for bank branch locations, ranging from 1-77...",
            ...
        }
    }
}

数据源:
- 表描述: bird_dev_light_schema.json (每个表的 Table description 段)
- 列描述: ATT short_profiles (extracted_final 字段, 更丰富)
- 兜底: light_schema 的 Column Description (若ATT没有)
"""

import json
import os
import re

LIGHT_SCHEMA_PATH = '/path/to/LadderSQL/schema_construction/db_light_schema/bird_dev_light_schema.json'
PROFILES_DIR = '/path/to/LadderSQL/schema_construction/bird_dev_profiles'
OUTPUT_PATH = '/path/to/LadderSQL/schema_construction/column_profiles_bird_dev.json'


def parse_light_schema(text: str):
    """从 light_schema text 提取表描述和列描述"""
    table_descs = {}
    column_descs = {}

    current_table = None
    current_col = None

    for line in text.split('\n'):
        # 表名
        m = re.match(r'^## Table: (.+)', line)
        if m:
            current_table = m.group(1).strip()
            current_col = None
            continue

        # 表描述
        if line.startswith('### Table description'):
            continue

        # 如果在表描述段（紧跟### Table description之后，到下一个###之前）
        if current_table and current_table not in table_descs and not line.startswith('#') and not line.startswith('- Column:'):
            if line.strip() and not line.startswith('### '):
                table_descs[current_table] = line.strip()
                continue

        # 列名
        m = re.match(r'^- Column: (.+)', line)
        if m and current_table:
            current_col = m.group(1).strip()
            continue

        # 列描述
        if current_col and current_table:
            m = re.match(r'^\s+- Description: (.+)', line)
            if m:
                desc = m.group(1).strip()
                column_descs[f"{current_table}.{current_col}"] = desc
                continue

            # Value Description (补充信息)
            m = re.match(r'^\s+- Value Description: (.+)', line)
            if m:
                val_desc = m.group(1).strip()
                key = f"{current_table}.{current_col}"
                if key in column_descs:
                    column_descs[key] += f" ({val_desc})"
                continue

    return table_descs, column_descs


def load_att_profiles(db_id: str):
    """加载 ATT short_profiles"""
    profiles = {}
    profile_file = os.path.join(PROFILES_DIR, db_id, f'{db_id}.short_profiles.jsonl')

    if not os.path.exists(profile_file):
        return profiles

    with open(profile_file) as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            table = item.get('table', '')
            column = item.get('column', '')
            # 优先使用 extracted_final，其次 short_profile_en
            desc = item.get('extracted_final', '') or item.get('short_profile_en', '')
            if table and column and desc:
                profiles[f"{table}.{column}"] = desc

    return profiles


def build_unified_profiles(db_id: str, light_schema: dict):
    """构建统一的表描述 + 列描述"""
    light_text = light_schema.get(db_id, '')

    # 从 light_schema 提取
    table_descs, light_col_descs = parse_light_schema(light_text)

    # 从 ATT 加载更丰富的列描述
    att_profiles = load_att_profiles(db_id)

    # 合并：ATT 优先，light_schema 兜底
    merged_col_profiles = {}
    all_cols = set(list(att_profiles.keys()) + list(light_col_descs.keys()))

    for col_key in all_cols:
        if col_key in att_profiles:
            merged_col_profiles[col_key] = att_profiles[col_key]
        elif col_key in light_col_descs:
            merged_col_profiles[col_key] = light_col_descs[col_key]

    return {
        'table_descriptions': table_descs,
        'column_profiles': merged_col_profiles
    }


def main():
    # 加载 light schema
    with open(LIGHT_SCHEMA_PATH) as f:
        light_schema = json.load(f)

    results = {}
    for db_id in sorted(light_schema.keys()):
        print(f"Building profiles for: {db_id}")
        profiles = build_unified_profiles(db_id, light_schema)
        results[db_id] = profiles

        # 验证输出
        n_tables = len(profiles['table_descriptions'])
        n_cols = len(profiles['column_profiles'])
        print(f"  Tables: {n_tables}, Columns: {n_cols}")

    # 保存
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved unified profiles for {len(results)} databases to {OUTPUT_PATH}")

    # 验证样例
    fin = results.get('financial', {})
    print(f"\n=== financial 表描述 ===")
    for t, d in list(fin['table_descriptions'].items())[:3]:
        print(f"  {t}: {d[:80]}")

    print(f"\n=== financial 列描述 (ATT优先) ===")
    for c, d in list(fin['column_profiles'].items())[:5]:
        print(f"  {c}: {d[:80]}")


if __name__ == '__main__':
    main()
