import os
import sys
import json
import csv
import logging
import sqlite3
import time
from typing import Any, Dict, List, Optional
from functools import lru_cache
import concurrent.futures
from dotenv import load_dotenv

# LangChain 相关
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

# =================配置区域 (Hardcoded)=================
# train_tables.json 所在目录
BASE_DATA_PATH = "/path/to/nl2sql_dataset/bird/train"   # 存放 *_tables.json 的目录
# 已解压且带 database_description CSV 的库目录（Value Description 依赖 CSV，不能丢）
DB_FOLDER_OVERRIDE = "/root/local_bird_db"
COLUMN_MEANING_PATH = "/path/to/column_meaning.json"   # BIRD 官方 column_meaning.json（test 提交时用他们给的那份）
MODEL_NAME = os.environ.get("GLM_MODEL", "glm-5.2")  # provider-specific (SiliconFlow: zai-org/GLM-5.2); keep the hyphen or it 404s
MODEL_PROVIDER = "openai"
ENV_API_KEY = "ALICLOUD_API_KEY"
ENV_BASE_URL = "ALICLOUD_BASE_URL"
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_API_KEY = ""   # 由 ALICLOUD_API_KEY / DASHSCOPE_API_KEY 环境变量提供
# =====================================================

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("bird_schema_generation.log", mode='a', encoding='utf-8')
    ]
)

# --- 工具函数 ---
def read_json(filename: str):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"文件 '{filename}' 不存在")
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data: Any, filename: str):
    dir_path = os.path.dirname(filename)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logging.info(f"✅ 文件已保存: {filename}")

def read_csv_description(csv_path: str) -> Dict[str, Dict]:
    """
    读取BIRD的database_description CSV文件
    返回: {column_name: {"description": ..., "value_description": ..., "data_format": ...}}
    """
    column_metadata = {}
    
    if not os.path.exists(csv_path):
        logging.warning(f"CSV描述文件不存在: {csv_path}")
        return column_metadata
    
    # 尝试多种编码
    encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']  # utf-8-sig处理BOM
    for encoding in encodings:
        try:
            with open(csv_path, 'r', encoding=encoding) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    col_name = row.get('original_column_name', '').strip()
                    if col_name:
                        column_metadata[col_name] = {
                            'column_name': row.get('column_name', '').strip(),
                            'description': row.get('column_description', '').strip(),
                            'value_description': row.get('value_description', '').strip(),
                            'data_format': row.get('data_format', '').strip()
                        }
            logging.info(f"   📖 从CSV读取了 {len(column_metadata)} 个列的元数据 (encoding: {encoding})")
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logging.warning(f"   ⚠️ 读取CSV失败 ({csv_path}): {e}")
            break
    
    return column_metadata


def load_column_meaning_json(json_path: str) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    读取train_column_meaning.json
    返回: {db_id: {table_name: {column_name: description}}}
    """
    result = {}
    if not os.path.exists(json_path):
        logging.warning(f"train_column_meaning.json 不存在: {json_path}")
        return result
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for key, desc in data.items():
            parts = key.split('|')
            if len(parts) >= 3:
                db_id = parts[0]
                table_name = parts[1]
                column_name = '|'.join(parts[2:])  # column name might contain |
                
                if db_id not in result:
                    result[db_id] = {}
                if table_name not in result[db_id]:
                    result[db_id][table_name] = {}
                result[db_id][table_name][column_name] = desc
        
        total_cols = sum(len(cols) for tables in result.values() for cols in tables.values())
        logging.info(f"📖 从 train_column_meaning.json 加载了 {len(result)} 个数据库, {total_cols} 个列描述")
    except Exception as e:
        logging.error(f"❌ 读取 train_column_meaning.json 失败: {e}")
    
    return result

@lru_cache(maxsize=16)
def get_cursor_from_path(sqlite_path: str):
    try:
        if not os.path.exists(sqlite_path):
            raise FileNotFoundError(f"SQLite database not found: {sqlite_path}")
        disk_conn = sqlite3.connect(f'file:{sqlite_path}?mode=ro', uri=True, check_same_thread=False)
        disk_conn.text_factory = lambda b: b.decode(errors="ignore")
        memory_conn = sqlite3.connect(':memory:', check_same_thread=False)
        memory_conn.text_factory = lambda b: b.decode(errors="ignore")
        disk_conn.backup(memory_conn)
        disk_conn.close()
        return memory_conn.cursor()
    except Exception as e:
        logging.error(f"Failed to load database {sqlite_path}: {e}")
        raise e

# --- Light Schema 模板逻辑 ---

# BIRD扩展模板：在Spider基础上增加 value_description 字段
LIGHT_SCHEMA_TEMPLATE = """## Table: {table}
### Table description
{table_description}
### Column information
{column_information}"""

PRIMARY_KEY_TEMPLATE = """### Primary keys
{primary_key}"""

FOREIGN_KEY_TEMPLATE = """### Foreign keys
{foreign_key}"""


class LightSchema(object):
    @staticmethod
    def create_schema(
        table: str,
        table_description: str,
        columns: List[Dict],
        primary_key: Optional[List[str]] = None,
        foreign_key: Optional[List[str]] = None,
    ) -> str:
        """
        生成BIRD Light Schema（对齐Spider格式，保留BIRD特有字段）
        """
        # 1. 生成列信息块（包含value_description）
        column_information = LightSchema._create_column_information(columns)
        
        # 2. 填充主模板
        final_desc = table_description.strip() if table_description else ""
        if not final_desc:
            final_desc = "No description available."
            
        schema_content = LIGHT_SCHEMA_TEMPLATE.format(
            table=table,
            table_description=final_desc,
            column_information=column_information
        )
        
        schema_parts = [schema_content]
        
        # 3. 追加主键
        if primary_key:
            pk_str = ", ".join(primary_key)
            schema_parts.append(PRIMARY_KEY_TEMPLATE.format(primary_key=pk_str))
        
        # 4. 追加外键
        if foreign_key:
            fk_str = "\n".join(foreign_key)
            schema_parts.append(FOREIGN_KEY_TEMPLATE.format(foreign_key=fk_str))
            
        return "\n\n".join(schema_parts)

    @staticmethod
    def _create_column_information(columns: List[Dict]) -> str:
        """
        将列信息转换为适合LLM阅读的缩进文本块。
        包含BIRD特有字段：value_description
        """
        lines = []
        for col in columns:
            name = col["name"]
            col_type = col["type"]
            desc = col.get("description", "No description available.")
            value_desc = col.get("value_description", "")  # BIRD特有
            samples = col.get("samples", [])

            # 构建单个列的块
            col_lines = [f"- Column: {name}"]
            col_lines.append(f"  - Type: {col_type}")
            col_lines.append(f"  - Description: {desc}")
            
            # BIRD特有：value_description（如果有则保留）
            if value_desc:
                # 清理换行符，保持单行
                value_desc_clean = value_desc.replace('\n', '; ').replace('\r', '')
                col_lines.append(f"  - Value Description: {value_desc_clean}")
            
            # 添加样本数据
            if samples:
                sample_str = json.dumps(samples, ensure_ascii=False)
                if len(sample_str) > 150:
                    sample_str = sample_str[:147] + "..."
                col_lines.append(f"  - Samples: {sample_str}")
            
            lines.append("\n".join(col_lines))
        
        return "\n\n".join(lines)


# --- Pydantic 模型（用于LLM输出解析）---

class TableDescriptionModel(BaseModel):
    table_description: str = Field(description="A brief, clear description of what the table represents.")


# --- 核心生成类 ---

class BIRDSchemaGenerator:
    def __init__(self):
        self.llm = self._init_llm()

    def _init_llm(self):
        api_key = os.environ.get(ENV_API_KEY) or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("JUDGE_API_KEY") or DEFAULT_API_KEY
        base_url = os.environ.get(ENV_BASE_URL) or os.environ.get("JUDGE_API_BASE") or os.environ.get("OPENAI_API_BASE") or DEFAULT_BASE_URL

        if not api_key:
            logging.error(f"❌ 未找到 API Key。请设置环境变量 {ENV_API_KEY} 或 DASHSCOPE_API_KEY")
            raise ValueError("API Key missing")

        logging.info(f"🚀 初始化模型: {MODEL_NAME} @ {base_url}")
        try:
            llm = init_chat_model(
                MODEL_NAME,
                model_provider=MODEL_PROVIDER,
                openai_api_base=base_url,
                openai_api_key=api_key,
                temperature=0,
                max_retries=2,
                max_tokens=1024,
                extra_body={"enable_thinking": False},  # GLM-5.2 必须关思考，否则 content 空
            )
            logging.info("✅ 模型初始化成功")
            return llm
        except Exception as e:
            logging.error(f"❌ 模型初始化失败: {e}")
            raise e

    def _generate_table_description(
        self, 
        table_name: str, 
        columns: List[str],
        column_descriptions: Dict[str, str]
    ) -> str:
        """
        调用LLM生成表的宏观描述
        利用BIRD提供的列描述作为context，生成更准确的表描述
        """
        if not self.llm:
            return ""
        
        # 构建上下文：列名 + 列描述
        col_context = []
        for col_name in columns:
            desc = column_descriptions.get(col_name, "")
            if desc:
                col_context.append(f"- {col_name}: {desc}")
            else:
                col_context.append(f"- {col_name}")
        
        context_str = "\n".join(col_context)
        
        system_prompt = (
            "You are a data analyst. Generate a brief, clear description of the table based on its name and columns.\n"
            "Output ONLY a valid JSON object with key 'table_description'.\n"
            "Do not output any markdown code blocks or extra text."
        )
        user_prompt = (
            f"Table Name: {table_name}\n"
            f"Columns with descriptions:\n{context_str}\n\n"
            "Describe what this table stores:"
        )

        response = None
        for attempt in range(6):
            try:
                response = self.llm.invoke([("system", system_prompt), ("human", user_prompt)])
                break
            except Exception as e:
                msg = str(e)
                if ("429" in msg or "limit_requests" in msg) and attempt < 5:
                    time.sleep(min(2 ** attempt, 30))  # 429 限流：指数退避重试
                    continue
                logging.warning(f"⚠️ 表描述生成失败 ({table_name}): {e}. 返回空字符串。")
                return ""

        try:
            content = response.content.strip()
            
            # 清理 Markdown 标记
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            parsed = json.loads(content)
            desc = parsed.get("table_description", "")
            
            if desc:
                logging.debug(f"   -> 成功生成表 '{table_name}' 的描述: {desc[:50]}...")
            else:
                logging.warning(f"   -> 表 '{table_name}' 的描述生成结果为空 JSON 值。")
            
            return desc
        except Exception as e:
            logging.warning(f"⚠️ 表描述生成失败 ({table_name}): {e}. 返回空字符串。")
            return ""

    def generate(self, eval_type: str = "dev", max_workers: int = 24):
        """
        生成BIRD数据集的Light Schema
        eval_type: "dev" 或 "train"
        """
        if eval_type == "train":
            tables_file = "/mnt/data/nl2sql_dataset/bird/train/train_tables.json"
            db_folder = "/root/local_bird_db/train_databases"
            column_meaning_path = "/mnt/data/nl2sql_dataset/bird/bird_clean_data/train_column_meaning.json"
            output_file = "/mnt/data/agl_20260626/schema_preprocessing/db_light_schema/bird_train_light_schema_glm52.json"
        else:
            # dev：old dev 与 clean dev 共用同一批 11 个 dev 库（库结构/CSV 完全相同），
            # light schema 库级内容一致，仅按 old_dev / clean_dev 区分输出文件名
            tables_file = "/mnt/data/nl2sql_dataset/bird/dev_20240627/dev_tables.json"
            db_folder = "/root/apex_sl_dbs/dev_databases"  # 本地磁盘，11 库均带 database_description CSV
            column_meaning_path = ""  # dev 无专用 column_meaning，列描述走 CSV database_description
            _dev_out = {
                "old_dev": "bird_old_dev_light_schema_glm52.json",
                "clean_dev": "bird_clean_dev_light_schema_glm52.json",
                "dev": "bird_dev_light_schema_glm52.json",
            }
            out_name = _dev_out.get(eval_type, "bird_dev_light_schema_glm52.json")
            output_file = f"/mnt/data/agl_20260626/schema_preprocessing/db_light_schema/{out_name}"

        logging.info(f"📂 读取元数据: {tables_file}")
        metadata_list = read_json(tables_file)
        
        # 加载 column_meaning（dev 为空则全走 CSV fallback）
        column_meaning = load_column_meaning_json(column_meaning_path)
        
        final_schemas = {}

        # 每个数据库作为一个并发任务（库内串行，库间并发）
        def _process_db(idx, db_info):
            db_id = db_info["db_id"]
            logging.info(f"[{idx+1}/{len(metadata_list)}] 处理数据库: {db_id}")
            
            db_path = os.path.join(db_folder, db_id, f"{db_id}.sqlite")
            csv_desc_dir = os.path.join(db_folder, db_id, "database_description")
            
            # 解析元数据
            table_names = db_info["table_names_original"]
            col_names_orig = db_info["column_names_original"] 
            col_types = db_info["column_types"]
            pks = db_info["primary_keys"]
            fks = db_info["foreign_keys"]

            # 扁平化 PK 列表
            pks_flat = [i for sub in pks for i in (sub if isinstance(sub, list) else [sub])]
            pk_col_indices = set(pks_flat)
            
            # 构建外键映射
            fk_map = {}
            for [id1, id2] in fks:
                if id1 < len(col_names_orig) and id2 < len(col_names_orig):
                    t1, c1 = col_names_orig[id1]
                    t2, c2 = col_names_orig[id2]
                    fk_map[(t1, c1)] = f"{table_names[t2]}.{c2}"
                    fk_map[(t2, c2)] = f"{table_names[t1]}.{c1}"

            # 读取BIRD的CSV描述文件（作为fallback）
            csv_descriptions = {}
            for t_name in table_names:
                csv_file = os.path.join(csv_desc_dir, f"{t_name}.csv")
                if os.path.exists(csv_file):
                    csv_descriptions[t_name] = read_csv_description(csv_file)
                    if len(csv_descriptions[t_name]) == 0:
                        logging.warning(f"   ⚠️ CSV文件存在但读取到0个列: {csv_file}")
                else:
                    logging.warning(f"   ⚠️ 未找到CSV描述文件: {csv_file}")
                    csv_descriptions[t_name] = {}

            # 获取当前数据库的 column_meaning 描述
            db_column_meaning = column_meaning.get(db_id, {})

            # 按表分组列
            table_cols_map = {}
            for col_idx, ([t_id, c_name], c_type) in enumerate(zip(col_names_orig, col_types)):
                if t_id < 0: continue
                t_name = table_names[t_id]
                if t_name not in table_cols_map:
                    table_cols_map[t_name] = []
                
                is_pk = col_idx in pk_col_indices
                is_fk = (t_id, c_name) in fk_map
                fk_ref = fk_map.get((t_id, c_name), "")

                table_cols_map[t_name].append({
                    "name": c_name, 
                    "type": c_type, 
                    "orig_idx": col_idx,
                    "is_pk": is_pk,
                    "is_fk": is_fk,
                    "fk_ref": fk_ref
                })

            db_schema_text = ""
            
            # 遍历每张表生成schema
            for t_name, cols in table_cols_map.items():
                col_names_only = [c["name"] for c in cols]
                
                # A. 优先从 train_column_meaning.json 获取列描述，CSV作为fallback补充value_description和data_format
                col_metadata = csv_descriptions.get(t_name, {})
                column_descriptions = {}
                column_value_descriptions = {}
                
                for c_info in cols:
                    c_name = c_info["name"]
                    # 优先用 train_column_meaning.json 的描述
                    json_desc = db_column_meaning.get(t_name, {}).get(c_name, "")
                    csv_meta = col_metadata.get(c_name, {})
                    
                    if json_desc:
                        column_descriptions[c_name] = json_desc
                    else:
                        column_descriptions[c_name] = csv_meta.get('description', '')
                    
                    column_value_descriptions[c_name] = csv_meta.get('value_description', '')
                
                # B. 生成表描述（利用CSV的列描述作为context）
                logging.info(f"   -> 正在生成表 '{t_name}' 的描述...")
                table_desc = self._generate_table_description(
                    t_name, 
                    col_names_only, 
                    column_descriptions
                )
                
                # C. 查询样本数据 & 组装最终列对象
                final_cols = []
                for c_info in cols:
                    c_name = c_info["name"]
                    c_type = c_info["type"]
                    
                    # 优先使用CSV的data_format
                    csv_meta = col_metadata.get(c_name, {})
                    if csv_meta.get('data_format'):
                        c_type = csv_meta['data_format']
                    
                    samples = []
                    try:
                        cursor = get_cursor_from_path(db_path)
                        query = f'SELECT DISTINCT "{c_name}" FROM "{t_name}" WHERE "{c_name}" IS NOT NULL ORDER BY RANDOM() LIMIT 3;'
                        cursor.execute(query)
                        rows = cursor.fetchall()
                        samples = [r[0] for r in rows]
                    except Exception as e:
                        logging.warning(f"   采样失败 ({t_name}.{c_name}): {e}")

                    clean_samples = []
                    for s in samples:
                        if isinstance(s, str) and len(s) > 32:
                            clean_samples.append(s[:32] + "...")
                        else:
                            clean_samples.append(s)

                    final_cols.append({
                        "name": c_name,
                        "type": c_type,
                        "description": column_descriptions.get(c_name, ""),
                        "value_description": column_value_descriptions.get(c_name, ""),  # BIRD特有
                        "samples": clean_samples,
                        "is_pk": c_info["is_pk"],
                        "is_fk": c_info["is_fk"],
                        "fk_ref": c_info["fk_ref"]
                    })

                # D. 提取该表的主键列表
                pk_list = [c["name"] for c in final_cols if c["is_pk"]]

                # E. 提取该表的外键列表
                fk_list = []
                for c in final_cols:
                    if c["is_fk"] and c["fk_ref"]:
                        fk_str = f"{t_name}.{c['name']} = {c['fk_ref']}"
                        fk_list.append(fk_str)

                # F. 生成该表的完整Schema文本
                table_md = LightSchema.create_schema(
                    table=t_name,
                    table_description=table_desc,
                    columns=final_cols,
                    primary_key=pk_list if pk_list else None,
                    foreign_key=fk_list if fk_list else None
                )
                db_schema_text += table_md + "\n\n"

            return db_id, db_schema_text.strip()

        # 库级并发调度（LLM 表描述请求并行发出，采样各库用各自 sqlite 连接，线程安全）
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_process_db, idx, db_info) for idx, db_info in enumerate(metadata_list)]
            done = 0
            for fut in concurrent.futures.as_completed(futures):
                try:
                    r_db_id, text = fut.result()
                    final_schemas[r_db_id] = text
                    done += 1
                    logging.info(f"✅ 完成 {done}/{len(metadata_list)}: {r_db_id}")
                except Exception as e:
                    logging.error(f"❌ 数据库处理失败: {e}")

        save_json(final_schemas, output_file)
        logging.info("🎉 全部完成！")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["dev", "train", "old_dev", "clean_dev"], default="dev", help="数据集类型：train / dev(通用) / old_dev / clean_dev（dev 家族共用同批库，仅输出文件名区分）")
    parser.add_argument("--workers", type=int, default=24, help="库级并发数")
    args = parser.parse_args()

    load_dotenv()

    generator = BIRDSchemaGenerator()
    generator.generate(eval_type=args.type, max_workers=args.workers)
