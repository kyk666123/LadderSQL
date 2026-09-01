import logging
import sys
import chromadb
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field



def setup_logging(log_level=logging.INFO, log_file="app.log"):
    """一个更专业的日志配置函数"""

    # 获取根 logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # 创建一个 Formatter
    log_format = logging.Formatter(
        '%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s'
    )

    # 1. 配置控制台输出 (StreamHandler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    # 可以为不同的 handler 设置不同的日志级别
    console_handler.setLevel(logging.INFO)

    # 2. 配置文件输出 (FileHandler)
    # 'a' 表示追加模式
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(log_format)
    # 文件中可以记录更详细的 DEBUG 信息
    file_handler.setLevel(logging.DEBUG)

    # 将 handlers 添加到根 logger
    # 防止重复添加 handler
    if not root_logger.handlers:
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
        
        
setup_logging()


class Condition(BaseModel):
    field: str
    operator: Literal["=", "!=", "in", "not in"] = Field(default="in")
    value: List[Any]


class RetrieveRequest(BaseModel):
    search_query: str
    mode: Literal["text", "vector", "hybrid"]
    index_name: str
    filter_conditions: Optional[List[Condition]] = Field(default_factory=list)
    threshold: Optional[float] = Field(default=0.0)
    k: Optional[int] = Field(default=10)
    only_retrieve: Optional[bool] = Field(default=None)
    rerank_mode: Optional[Literal["bge", "gte", "rrf"]] = Field(default=None)
    trace_id: Optional[str] = Field(default=None)
    extra_args: Optional[Dict[str, Any]] = Field(default_factory=dict)

    def to_api_payload(self) -> Dict[str, Any]:
        """Convert request to API payload format"""
        return {
            "query": self.search_query,
            "mode": self.mode,
            "collectionName": self.index_name,
            "filterConditions": [
                condition.model_dump() for condition in self.filter_conditions
            ],
            "threshold": self.threshold,
            "limit": self.k,
            "onlyRetrieve": self.only_retrieve,
            "rerankMode": self.rerank_mode,
            "traceId": self.trace_id,
        }


class RetrieveDoc(BaseModel):
    content: str
    biz_data: Optional[Dict[str, Any]] = Field(default_factory=dict)
    score: Optional[float] = Field(default=None)


class RetrieveResponse(BaseModel):
    docs: Optional[List[RetrieveDoc]] = Field(default_factory=list)
    error_message: Optional[str] = Field(default=None)


class BaseVectorStore(ABC):
    """Base class for vector store."""

    def __init__(self, index_name: str = None):
        self.index_name = index_name
        self.client = None

    @abstractmethod
    def add_documents(self, documents: List[Dict]) -> List[str]:
        """Add documents to the vector store.

        Args:
            documents (List[Dict]): List of documents to be added.

        Returns:
            list[str]: List of IDs of the added documents.
        """
        pass

    @abstractmethod
    def search(self, retrieve_request: RetrieveRequest) -> RetrieveResponse:
        """Search for documents in the vector store.

        Args:
            retrieve_request (RetrieveRequest): Retrieve request.

        Returns:
            RetrieveResponse: Retrieve response.
        """
        pass
    

class ChromaVectorStore(BaseVectorStore):
    """
    基于 ChromaDB 实现的 VectorStore。
    """

    def __init__(self, client_path: str, index_name: str):
        """
        初始化 ChromaVectorStore。
        """
        super().__init__(index_name)
        self.collection = None
        self.client = chromadb.PersistentClient(path=client_path)

    def connect(self):
        """
        建立与 Chroma 数据库的连接，并获取或创建集合。
        """
        if self.collection is None:
            try:
                self.collection = self.client.get_or_create_collection(
                    name=self.index_name
                )
            except Exception as e:
                raise Exception(f"Error connecting to chromadb collection: {e}") from e

    def disconnect(self):
        """
        对于 ChromaDB，通常不需要显式地“断开”本地客户端。
        对于 HTTP 客户端，连接由底层 HTTP 库管理。
        """
        pass

    def add_documents(self, documents: List[RetrieveDoc]) -> List[str]:
        if self.collection is None:
            raise Exception(
                "No active chromadb collection connection. Call connect() first."
            )

        try:
            existing_count = self.collection.count()
            contents = [document.content for document in documents]
            metadatas = [document.biz_data for document in documents]
            ids = [f"doc_{i + existing_count}" for i in range(len(contents))]

            self.collection.add(documents=contents, metadatas=metadatas, ids=ids)

            print("Data successfully add to the collection!")

        except Exception as e:
            raise Exception(f"Error adding documents to chromadb collection: {e}")

        return ids

    def _query_collection(self, query_text: str, n_results: int, where):
        """先尝试 collection 内置 embedding; 若 collection 配置硬编码 cuda 但本机无 cuda /
        kernel 不兼容时, 降级用本地 CPU sentence_transformer 算 query embedding 并
        传 query_embeddings 给 chroma.
        """
        try:
            return self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where,
            )
        except Exception as e:
            err_msg = str(e)
            if not (
                "CUDA" in err_msg
                or "cuda" in err_msg
                or "No CUDA GPUs" in err_msg
                or "kernel image" in err_msg
                or "sentence_transformer" in err_msg.lower()
                or "embedding function" in err_msg.lower()
            ):
                raise
            if not hasattr(ChromaVectorStore, "_cpu_st_model"):
                import os
                from sentence_transformers import SentenceTransformer
                model_path = os.environ.get(
                    "AGL_ST_MODEL_PATH",
                    "/path/to/embedding_models/all-MiniLM-L6-v2",
                )
                ChromaVectorStore._cpu_st_model = SentenceTransformer(model_path, device="cpu")
            emb = ChromaVectorStore._cpu_st_model.encode(
                [query_text], normalize_embeddings=False
            ).tolist()
            return self.collection.query(
                query_embeddings=emb,
                n_results=n_results,
                where=where,
            )

    def search(self, retrieve_request: RetrieveRequest) -> RetrieveResponse:
        try:
            where_clause = None
            if retrieve_request.filter_conditions:
                and_conditions = []
                for cond in retrieve_request.filter_conditions:
                    if cond.operator.lower() == "in":
                        op = "$in"
                    elif cond.operator.lower() == "nin":
                        op = "$nin"
                    else:
                        continue

                    and_conditions.append({cond.field: {op: cond.value}})

                if and_conditions:
                    where_clause = {"$and": and_conditions}

            results = self._query_collection(
                retrieve_request.search_query,
                retrieve_request.k,
                where_clause,
            )

            docs = []
            if results and results.get("documents"):
                result_documents = results["documents"][0]
                result_metadatas = results["metadatas"][0]
                result_distances = results["distances"][0]

                for document, metadata, distance in zip(
                        result_documents, result_metadatas, result_distances
                ):
                    # 只保留那些满足质量阈值（距离足够近）的结果
                    if distance < retrieve_request.threshold:
                        docs.append(
                            RetrieveDoc(content=document, biz_data=metadata)
                        )

            return RetrieveResponse(docs=docs)

        except Exception as e:
            error = str(e)
            return RetrieveResponse(error_message=error)


class DatabaseCellRetrieval:
    def __init__(self, database_literals, search_client, collection_name):
        self.database_literals = database_literals
        self.search_client = ChromaVectorStore(
            client_path=search_client, index_name=collection_name
        )
        self.search_client.connect()
        self.retrieval_results = []

    def retrieve(self, threshold=0.8, k=5):
        results_set = set()
        for value in self.database_literals:
            # 跳过非字符串 / 空串。
            # 注: 不能跳过纯数字字符串 —— TEXT/CHAR/CLOB 列里存的数字串 (如 product_code='12345',
            # year='1990') 是合法检索 key, 之前用 .isdigit() 一刷子跳过会导致检索返回空集、
            # GLM-5 误判“值不存在”并改写为通配符。INTEGER 列本身在入库阶段已被跳过,
            # 不会出现在 chroma 里。
            if not isinstance(value, str) or not value.strip():
                continue
            request = RetrieveRequest(
                search_query=value, mode="text", threshold=threshold, k=k, index_name=""
            )
            responses = self.search_client.search(request)
            contents = [doc.content for doc in responses.docs if doc.content]
            metadatas = [doc.biz_data for doc in responses.docs if doc.biz_data]
            content_meta_pairs = [
                (content, metadata) for content, metadata in zip(contents, metadatas)
            ]

            for content, metadata in content_meta_pairs:
                set_key = "{}_{}_{}".format(metadata["table"], metadata["column"], content)
                if set_key in results_set:
                    continue
                results_set.add(set_key)
                self.retrieval_results.append(
                    {
                        "table": metadata["table"],
                        "column": metadata["column"],
                        "content": content,
                    }
                )
        return self.retrieval_results
    
    
if __name__ == "__main__":
    retriever = DatabaseCellRetrieval(database_literals=["cookies", "food"], search_client="/path/to/LadderSQL/examples/my_spider/agent/chroma/spider_test/spider_test", collection_name="bakery_1")
    results = retriever.retrieve()
    print(results, type(results))
    


