"""
Step 0a: 预计算每个 DB 的 JOIN Graph（含业务语义）

输出格式 (per db_id):
{
    "db_id": "financial",
    "tables": {
        "account": "银行账户信息（ID、开户地区、使用频率、创建日期）",
        ...
    },
    "edges": [
        {
            "from_table": "trans",
            "from_col": "account_id",
            "to_table": "account",
            "to_col": "account_id",
            "semantic": "交易所属的银行账户"
        },
        ...
    ],
    "adjacency": {
        "account": ["district", "trans", "order", "loan", "disp"],
        ...
    }
}

数据源:
- PK/FK: PRAGMA table_info + PRAGMA foreign_key_list
- 表描述: bird_dev_light_schema.json
- 列描述: 用于生成edge的业务语义标注（轻量LLM）
"""

import json
import sqlite3
import os
import re

DB_PATH = '/tmp/bird_dev_databases'
LIGHT_SCHEMA_PATH = '/path/to/LadderSQL/schema_construction/db_light_schema/bird_dev_light_schema.json'
OUTPUT_PATH = '/path/to/LadderSQL/schema_construction/join_graph_bird_dev.json'


def get_db_metadata(db_id: str):
    """从 PRAGMA 获取完整的 PK/FK/类型信息"""
    db_file = os.path.join(DB_PATH, db_id, f'{db_id}.sqlite')
    tables = {}
    foreign_keys = []

    with sqlite3.connect(f'file:{db_file}?mode=ro', uri=True) as conn:
        cursor = conn.cursor()

        # 获取所有表
        all_tables = cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence'"
        ).fetchall()

        for (table_name,) in all_tables:
            # 列信息 + PK
            cols = cursor.execute(f'PRAGMA table_info(`{table_name}`)').fetchall()
            columns = {}
            pk_cols = []
            for col in cols:
                # col: (cid, name, type, notnull, default, pk)
                col_name = col[1]
                col_type = col[2].upper() if col[2] else 'TEXT'
                is_pk = col[5] > 0
                columns[col_name] = {
                    'type': col_type,
                    'is_pk': is_pk
                }
                if is_pk:
                    pk_cols.append(col_name)

            tables[table_name] = {
                'columns': columns,
                'pk_cols': pk_cols
            }

            # FK信息
            fks = cursor.execute(f'PRAGMA foreign_key_list(`{table_name}`)').fetchall()
            for fk in fks:
                # fk: (id, seq, table, from, to, on_update, on_delete, match)
                foreign_keys.append({
                    'from_table': table_name,
                    'from_col': fk[3],
                    'to_table': fk[2],
                    'to_col': fk[4]
                })

    return tables, foreign_keys


def parse_table_descriptions(light_schema_text: str):
    """从 light_schema 提取每个表的描述"""
    table_descs = {}
    # 匹配 ## Table: xxx 后面的 ### Table description 段落
    pattern = r'## Table: (\w+)\n### Table description\n(.+?)(?=\n### |\n## )'
    matches = re.findall(pattern, light_schema_text, re.DOTALL)
    for table_name, desc in matches:
        # 取第一句话作为精简描述
        desc_clean = desc.strip().split('\n')[0].strip()
        table_descs[table_name] = desc_clean
    return table_descs


def build_join_graph(db_id: str, light_schema: dict):
    """构建单个DB的JOIN Graph"""
    tables_meta, foreign_keys = get_db_metadata(db_id)

    # 提取表描述
    light_text = light_schema.get(db_id, '')
    table_descs = parse_table_descriptions(light_text)

    # 构建edges（带语义：from_col 的描述作为连接语义）
    edges = []
    for fk in foreign_keys:
        edge = {
            'from_table': fk['from_table'],
            'from_col': fk['from_col'],
            'to_table': fk['to_table'],
            'to_col': fk['to_col'],
            'semantic': f"{fk['from_table']}.{fk['from_col']} references {fk['to_table']}.{fk['to_col']}"
        }
        edges.append(edge)

    # 构建邻接表
    adjacency = {}
    for fk in foreign_keys:
        adjacency.setdefault(fk['from_table'], set()).add(fk['to_table'])
        adjacency.setdefault(fk['to_table'], set()).add(fk['from_table'])
    adjacency = {k: sorted(list(v)) for k, v in adjacency.items()}

    # 构建表信息（含PK）
    tables_info = {}
    for table_name, meta in tables_meta.items():
        desc = table_descs.get(table_name, '')
        tables_info[table_name] = {
            'description': desc,
            'pk_cols': meta['pk_cols'],
            'columns': {col_name: col_info['type'] for col_name, col_info in meta['columns'].items()}
        }

    return {
        'db_id': db_id,
        'tables': tables_info,
        'edges': edges,
        'adjacency': adjacency
    }


def format_join_graph_text(graph: dict) -> str:
    """将 JOIN Graph 格式化为 LLM 可读的文本"""
    lines = []
    lines.append("## Database JOIN Graph\n")

    # 表描述
    lines.append("### Tables:")
    for table_name, info in sorted(graph['tables'].items()):
        desc = info.get('description', '')
        pk = ', '.join(info.get('pk_cols', []))
        desc_str = f" - {desc}" if desc else ""
        pk_str = f" [PK: {pk}]" if pk else ""
        lines.append(f"  - {table_name}{desc_str}{pk_str}")

    lines.append("")

    # FK连接关系
    lines.append("### Foreign Key Connections:")
    for edge in graph['edges']:
        lines.append(
            f"  - {edge['from_table']}.{edge['from_col']} → {edge['to_table']}.{edge['to_col']}"
        )

    lines.append("")

    # 邻接表（可达性）
    lines.append("### Reachability:")
    for table, neighbors in sorted(graph['adjacency'].items()):
        lines.append(f"  - {table} ↔ [{', '.join(neighbors)}]")

    return '\n'.join(lines)


def main():
    # 加载 light schema
    with open(LIGHT_SCHEMA_PATH) as f:
        light_schema = json.load(f)

    # 获取所有 DB
    dbs = sorted([
        d for d in os.listdir(DB_PATH)
        if os.path.isdir(os.path.join(DB_PATH, d))
    ])

    results = {}
    for db_id in dbs:
        print(f"Building JOIN graph for: {db_id}")
        graph = build_join_graph(db_id, light_schema)
        results[db_id] = graph

        # 打印可读文本（验证）
        if db_id == 'financial':
            print(format_join_graph_text(graph))
            print()

    # 保存
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved JOIN graphs for {len(results)} databases to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
