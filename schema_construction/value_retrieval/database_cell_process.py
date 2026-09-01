import logging
import sys
import os
import platform
import sqlite3
from functools import lru_cache
from uuid import uuid4
import torch
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import SentenceTransformer
import traceback

# ================= 配置区域 =================
# 【重要】直接在这里定义绝对的日志文件路径，避免任何相对路径解析问题
# 请确保 /path/to/LadderSQL/examples/my_spider/agent/ 目录存在且可写
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE_PATH = os.path.join(SCRIPT_DIR, "detailed_process_log.log")

# 全局统计
GLOBAL_STATS = {
    "total_dbs": 0,
    "empty_dbs": [],
    "total_vectors": 0
}

def setup_logging(log_level=logging.INFO):
    """
    高可靠性日志配置
    1. 强制使用绝对路径
    2. 预先检查目录写权限
    3. 清除旧 handler
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG) # 内部设为 DEBUG，通过 handler 控制输出级别

    # 清除现有 handler
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    log_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 1. 控制台 Handler (INFO 级别)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    console_handler.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)

    # 2. 文件 Handler (DEBUG 级别 - 记录所有细节)
    # 再次确认路径是绝对的
    abs_log_path = os.path.abspath(LOG_FILE_PATH)
    log_dir = os.path.dirname(abs_log_path)

    # 【关键检查】确保目录存在且可写
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir, exist_ok=True)
            print(f"已创建日志目录: {log_dir}")
        except Exception as e:
            print(f"CRITICAL ERROR: 无法创建日志目录 {log_dir}. 错误: {e}")
            raise PermissionError(f"无法创建日志目录: {log_dir}")
    
    if not os.access(log_dir, os.W_OK):
        raise PermissionError(f"当前用户没有权限写入日志目录: {log_dir}")

    try:
        file_handler = logging.FileHandler(abs_log_path, mode='a', encoding='utf-8')
        file_handler.setFormatter(log_format)
        file_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        
        # 测试写入
        logging.info("-" * 50)
        logging.info(f"日志系统初始化成功。文件路径: {abs_log_path}")
        logging.info("-" * 50)
        
        # 立即刷新以确保文件句柄已打开
        file_handler.flush()
        
    except Exception as e:
        print(f"CRITICAL ERROR: 无法初始化日志文件 {abs_log_path}. 错误: {e}")
        print("尝试回退到临时目录...")
        # fallback 到 /tmp
        fallback_path = "/tmp/fallback_process.log"
        try:
            fh = logging.FileHandler(fallback_path, mode='a', encoding='utf-8')
            fh.setFormatter(log_format)
            root_logger.addHandler(fh)
            logging.warning(f"已回退日志到: {fallback_path}")
        except:
            logging.error("完全无法写入日志文件，程序将继续运行但无文件日志。")

def get_default_device():
    if torch.cuda.is_available():
        return "cuda"
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

@lru_cache(maxsize=16)
def get_cursor_from_path(sqlite_path):
    try:
        if not os.path.exists(sqlite_path):
            raise FileNotFoundError(f"SQLite database not found: {sqlite_path}")
        
        # 只读模式加载到内存，避免锁竞争
        disk_conn = sqlite3.connect(f'file:{sqlite_path}?mode=ro', uri=True, check_same_thread=False)
        disk_conn.text_factory = lambda b: b.decode(errors="ignore")
        
        memory_conn = sqlite3.connect(':memory:', check_same_thread=False)
        memory_conn.text_factory = lambda b: b.decode(errors="ignore")
        
        disk_conn.backup(memory_conn)
        disk_conn.close()
        
        return memory_conn.cursor()
    except Exception as e:
        logging.error(f"加载数据库失败 {sqlite_path}: {e}")
        raise e

class ChromaWriter:
    def __init__(self, database_folder, dataset_cell_chroma_path, embedding_model_path, device="mps", batch_size=1000, max_str_len=256, embedding_model="all-MiniLM-L6-v2"):
        self.database_folder = database_folder
        self.dataset_cell_chroma_path = dataset_cell_chroma_path
        self.embedding_model_path = embedding_model_path
        self.device = device
        self.batch_size = batch_size
        self.max_str_len = max_str_len
        self.skip_keywords = ["_id", " id", "url", "email", "web", "time", "date", "address"]

        if not os.path.exists(embedding_model_path):
            logging.info(f"模型不存在，正在下载: {embedding_model}")
            model = SentenceTransformer(embedding_model)
            saved = model.save(embedding_model_path)
            self.embedding_model_path = saved if saved else embedding_model_path

    def _write_direct_backup(self, message):
        """双重保障：直接写入文件，绕过 logging 缓冲"""
        try:
            with open(LOG_FILE_PATH, 'a', encoding='utf-8') as f:
                f.write(f"[DIRECT_BACKUP] {message}\n")
                f.flush()
        except:
            pass

    def process_single_db(self, collection_name):
        client = chromadb.PersistentClient(path=self.dataset_cell_chroma_path)
        embedding_function = SentenceTransformerEmbeddingFunction(model_name=self.embedding_model_path, device=self.device)
        
        try:
            collection = client.get_collection(name=collection_name, embedding_function=embedding_function)
        except Exception:
            logging.warning(f"集合 {collection_name} 不存在，正在创建...")
            collection = client.create_collection(name=collection_name, embedding_function=embedding_function)

        db_path = os.path.join(self.database_folder, collection_name, f"{collection_name}.sqlite")
        
        logging.info("=" * 100)
        logging.info(f"🚀 开始处理数据库: [{collection_name}]")
        logging.info(f"📂 源文件: {db_path}")
        logging.info("=" * 100)
        
        # 双重备份记录
        self._write_direct_backup(f"START_DB: {collection_name}")

        cursor = get_cursor_from_path(db_path)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        db_vector_count = 0

        if not tables:
            logging.warning(f"⚠️ 数据库 {collection_name} 中没有发现任何表！")
            GLOBAL_STATS["empty_dbs"].append(collection_name)
            self._write_direct_backup(f"END_DB: {collection_name} (NO TABLES)")
            return

        # 遍历所有表
        for table_idx, table in enumerate(tables):
            table_name = table[0]
            logging.info(f"\n📋 表 [{table_idx+1}/{len(tables)}]: {table_name}")
            logging.info("-" * 80)
            
            # 获取列详情
            cursor.execute(f"PRAGMA table_info(`{table_name}`);")
            columns = cursor.fetchall()
            # col: (cid, name, type, notnull, dflt_value, pk)
            
            primary_keys = [col[1].lower() for col in columns if col[5] > 0]
            
            table_summary = {
                "processed": [],
                "skipped": [],
                "total_columns": len(columns)
            }

            logging.info(f"   该表共有 {len(columns)} 列。详细分析如下:")
            
            for col in columns:
                col_name = col[1]
                col_type_raw = col[2] if col[2] else "UNKNOWN"
                col_type_lower = col_type_raw.lower()
                is_pk = col[5] > 0
                
                # 1. 判断是否为文本类型 (扩展支持 VARCHAR, CHAR, TEXT, CLOB)
                is_text_type = any(k in col_type_lower for k in ["text", "char", "clob"])
                
                skip_reason = None
                
                if not is_text_type:
                    skip_reason = f"非文本类型 ({col_type_raw})"
                elif is_pk:
                    skip_reason = "主键 (Primary Key)"
                elif col_name.lower().endswith("id"):
                    skip_reason = "命名规则 (以 'id' 结尾)"
                else:
                    # 检查关键词
                    match_kw = next((k for k in self.skip_keywords if k in col_name.lower()), None)
                    if match_kw:
                        skip_reason = f"包含关键词 '{match_kw}'"
                
                # 记录详细信息
                col_info = {
                    "name": col_name,
                    "type": col_type_raw,
                    "is_pk": is_pk,
                    "reason": skip_reason
                }

                if skip_reason:
                    table_summary["skipped"].append(col_info)
                    logging.info(f"      ❌ [跳过] 列名: '{col_name:<20}' | 类型: {col_type_raw:<15} | 原因: {skip_reason}")
                else:
                    table_summary["processed"].append(col_info)
                    logging.info(f"      ✅ [处理] 列名: '{col_name:<20}' | 类型: {col_type_raw:<15} | 状态: 提取数据中...")
                    
                    # 执行数据提取
                    try:
                        query = f"SELECT DISTINCT `{col_name}` FROM `{table_name}` WHERE `{col_name}` IS NOT NULL"
                        cursor.execute(query)
                        rows = cursor.fetchall()
                        
                        # 过滤
                        valid_values = [
                            r[0] for r in rows 
                            if isinstance(r[0], str) and len(r[0]) <= self.max_str_len
                        ]
                        
                        if valid_values:
                            # 准备批量数据
                            metas = [{"table": table_name, "column": col_name} for _ in valid_values]
                            ids = [str(uuid4()) for _ in valid_values]
                            
                            # 分批写入 Chroma
                            count = 0
                            for i in range(0, len(valid_values), self.batch_size):
                                end = min(i + self.batch_size, len(valid_values))
                                collection.add(
                                    documents=valid_values[i:end],
                                    metadatas=metas[i:end],
                                    ids=ids[i:end]
                                )
                                count += (end - i)
                            
                            db_vector_count += count
                            logging.info(f"          📥 成功写入 {count} 条向量 (原始行数: {len(rows)}, 过滤后: {len(valid_values)})")
                        else:
                            logging.info(f"          ⚪ 无有效数据 (原始 {len(rows)} 行，经长度/类型过滤后为 0)")
                            
                    except Exception as e:
                        logging.error(f"          💥 查询出错: {e}")
                        logging.debug(traceback.format_exc())

            # 表级总结
            if not table_summary["processed"]:
                logging.info(f"   >> 表 '{table_name}' 所有列均被跳过，未产生向量。")
            else:
                logging.info(f"   >> 表 '{table_name}' 处理完毕：处理 {len(table_summary['processed'])} 列，跳过 {len(table_summary['skipped'])} 列。")

        # 数据库级总结
        GLOBAL_STATS["total_vectors"] += db_vector_count
        if db_vector_count == 0:
            GLOBAL_STATS["empty_dbs"].append(collection_name)
            msg = f"⚠️ 数据库 [{collection_name}] 处理完成，但未写入任何向量！"
            logging.warning(msg)
            self._write_direct_backup(f"END_DB: {collection_name} (EMPTY)")
        else:
            msg = f"✅ 数据库 [{collection_name}] 处理完成，共写入 {db_vector_count} 条向量。"
            logging.info(msg)
            self._write_direct_backup(f"END_DB: {collection_name} (COUNT: {db_vector_count})")

        # 强制刷新所有 handler
        for h in logging.getLogger().handlers:
            h.flush()

    def process_db(self, collections):
        client = chromadb.PersistentClient(path=self.dataset_cell_chroma_path)
        exist_collections = [c.name for c in client.list_collections()]

        logging.info("🔄 正在重置 ChromaDB 集合...")
        for name in collections:
            if name in exist_collections:
                try:
                    client.delete_collection(name)
                except: pass
            try:
                ef = SentenceTransformerEmbeddingFunction(model_name=self.embedding_model_path, device=self.device)
                client.create_collection(name=name, embedding_function=ef)
            except Exception as e:
                logging.error(f"创建集合 {name} 失败: {e}")

        logging.info("▶️ 开始顺序处理数据库...\n")
        for name in collections:
            try:
                self.process_single_db(name)
            except Exception as e:
                logging.critical(f"💀 数据库 {name} 处理崩溃: {e}", exc_info=True)
                GLOBAL_STATS["empty_dbs"].append(name)
                self._write_direct_backup(f"CRASH_DB: {name} - {str(e)}")

    def print_final_summary(self):
        logging.info("\n" + "=" * 100)
        logging.info(" " * 35 + "🏁 最终执行汇总报告 🏁")
        logging.info("=" * 100)
        logging.info(f"📊 总处理数据库数: {GLOBAL_STATS['total_dbs']}")
        logging.info(f"💾 总写入向量数:   {GLOBAL_STATS['total_vectors']}")
        
        if GLOBAL_STATS["empty_dbs"]:
            logging.warning(f"\n⚠️ 以下 {len(GLOBAL_STATS['empty_dbs'])} 个数据库未写入任何向量:")
            for db in GLOBAL_STATS["empty_dbs"]:
                logging.warning(f"   - {db}")
            self._write_direct_backup(f"SUMMARY_EMPTY_DBS: {', '.join(GLOBAL_STATS['empty_dbs'])}")
        else:
            logging.info("\n🎉 所有数据库均成功写入向量！")
            
        logging.info("=" * 100)
        # 最后一次强制刷新
        for h in logging.getLogger().handlers:
            h.flush()
        
        logging.info(f"💡 提示：详细日志已保存至绝对路径 -> {LOG_FILE_PATH}")

def ChromaWriteMain(database_folder, dataset_cell_chroma_path, embedding_model_path):
    if not os.path.exists(database_folder):
        logging.error(f"数据库文件夹不存在: {database_folder}")
        return

    collections = [f for f in os.listdir(database_folder) if os.path.isdir(os.path.join(database_folder, f))]
    
    if not collections:
        logging.warning("未找到任何数据库子文件夹。")
        return

    GLOBAL_STATS["total_dbs"] = len(collections)
    logging.info(f"🔍 发现 {len(collections)} 个待处理数据库: {collections}")
    
    writer = ChromaWriter(
        database_folder=database_folder,
        dataset_cell_chroma_path=dataset_cell_chroma_path,
        embedding_model_path=embedding_model_path,
        device=get_default_device()
    )
    
    writer.process_db(collections)
    writer.print_final_summary()

if __name__ == "__main__":
    # 初始化日志 (现在会强制写入绝对路径)
    setup_logging()
    
    database_folder = "/path/to/dataset/spider/test_database"
    dataset_cell_chroma_path = os.path.join(os.path.dirname(__file__), "..", "chroma", "spider_test")
    embedding_model_path = "/path/to/embedding_models/all-MiniLM-L6-v2"

    ChromaWriteMain(database_folder, dataset_cell_chroma_path, embedding_model_path)