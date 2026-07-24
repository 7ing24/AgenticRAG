"""Milvus 知识库管理器

继承 MilvusStore 基类，提供知识库特有的 Schema、混合检索、文档 CRUD。
"""

import os
import logging
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from core.milvus_store import MilvusStore

load_dotenv()

logger = logging.getLogger(__name__)


class KnowledgeBaseMilvusManager(MilvusStore):
    """Milvus 知识库管理器 — 支持 Dense + BM25 混合检索"""

    def __init__(self, uri: str, token: str, embeddings,
                 collection_name: str = "ai_knowledge_collection",
                 dense_dim: int = 1536):
        super().__init__(uri, token, embeddings, collection_name, dense_dim)

    # =========================================================================
    # Collection 初始化
    # =========================================================================

    def init_collection(self) -> bool:
        if self._loaded:
            return True
        try:
            if self.has_collection():
                logger.info(f"知识库集合 {self.collection_name} 已存在")
                self.load_collection()
                self._loaded = True
                return True

            logger.info(f"开始创建知识库集合 {self.collection_name}...")
            self._create_collection()
            self._loaded = True
            return True
        except Exception as e:
            logger.error(f"初始化知识库集合失败: {e}")
            raise

    def _create_collection(self):
        from pymilvus import DataType, Function, FunctionType

        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=True)

        schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=50,
                         is_primary=True, auto_id=True)
        schema.add_field(field_name="doc_id", datatype=DataType.INT64)
        schema.add_field(field_name="chunk_index", datatype=DataType.INT32)
        schema.add_field(field_name="knowledge_base_id", datatype=DataType.VARCHAR,
                         max_length=128, is_partition_key=True)
        schema.add_field(field_name="user_id", datatype=DataType.INT64)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR,
                         max_length=65535, enable_analyzer=True,
                         analyzer_params={"type": "chinese"})
        schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="page_number", datatype=DataType.INT32, nullable=True)
        schema.add_field(field_name="file_name", datatype=DataType.VARCHAR,
                         max_length=255, nullable=True)
        schema.add_field(field_name="file_hash", datatype=DataType.VARCHAR,
                         max_length=64, nullable=True)
        schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR,
                         max_length=100, nullable=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR,
                         dim=self.dense_dim)
        schema.add_field(field_name="sparse_vector",
                         datatype=DataType.SPARSE_FLOAT_VECTOR)

        bm25_function = Function(
            name="bm25_func", function_type=FunctionType.BM25,
            input_field_names=["content"], output_field_names=["sparse_vector"],
        )
        schema.add_function(bm25_function)

        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="vector", index_name="vector_index",
                               index_type="HNSW", metric_type="COSINE",
                               params={"M": 16, "efConstruction": 200})
        index_params.add_index(field_name="sparse_vector", index_name="sparse_index",
                               index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
        for field in ["doc_id", "user_id"]:
            index_params.add_index(field_name=field, index_type="STL_SORT")

        logger.info(f"正在创建集合 {self.collection_name}（含 BM25 Function）...")
        self.client.create_collection(
            collection_name=self.collection_name, schema=schema,
            index_params=index_params, properties={"partitionkey.isolation": True},
        )
        logger.info(f"集合 {self.collection_name} 创建完成，正在加载...")
        self.load_collection()
        logger.info(f"知识库集合 {self.collection_name} 创建成功")

    # =========================================================================
    # 文档写入
    # =========================================================================

    def add_chunks(self, documents: List[Document], knowledge_base_id: str,
                   user_id: int) -> int:
        if not documents:
            return 0

        texts, records = [], []
        for i, doc in enumerate(documents):
            content = doc.page_content.strip()
            if not content:
                continue
            texts.append(content)
            records.append({
                "doc_id": doc.metadata.get("doc_id", 0),
                "chunk_index": doc.metadata.get("chunk_index", i),
                "knowledge_base_id": knowledge_base_id,
                "user_id": user_id,
                "content": content,
                "source": doc.metadata.get("source", ""),
                "page_number": doc.metadata.get("page", doc.metadata.get("page_number", 1)),
                "file_name": doc.metadata.get("file_name", ""),
                "file_hash": doc.metadata.get("file_hash", ""),
                "parent_id": doc.metadata.get("parent_id", ""),
            })

        if not texts:
            return 0

        vectors = self._embed_documents(texts)
        for record, vector in zip(records, vectors):
            record["vector"] = vector

        self._insert(records)
        logger.info(f"知识库写入完成 — {len(records)} 条, KB: {knowledge_base_id}, 用户: {user_id}")
        return len(records)

    # =========================================================================
    # 混合检索
    # =========================================================================

    def hybrid_search(self, query: str, top_k: int = 5,
                      knowledge_base_id: str = "默认知识库",
                      user_id: int = 0) -> List[Document]:
        dense_vector = self._embed_query(query)
        if not dense_vector:
            return []

        filter_parts = [f"knowledge_base_id == '{knowledge_base_id}'"]
        if user_id > 0:
            filter_parts.append(f"user_id == {user_id}")
        filter_expr = " and ".join(filter_parts)

        try:
            return self._native_hybrid_search(dense_vector, query, filter_expr, top_k)
        except Exception as e:
            logger.error(f"混合检索失败: {e}，降级为稠密检索")
            try:
                return self._dense_search_only(dense_vector, filter_expr, top_k)
            except Exception as fallback_e:
                logger.error(f"降级检索也失败: {fallback_e}")
                return []

    def _native_hybrid_search(self, dense_vector, raw_query, filter_expr, top_k):
        from pymilvus import AnnSearchRequest, RRFRanker

        dense_req = AnnSearchRequest(
            data=[dense_vector], anns_field="vector",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k * 6, expr=filter_expr,
        )
        sparse_req = AnnSearchRequest(
            data=[raw_query], anns_field="sparse_vector",
            param={"metric_type": "BM25"}, limit=top_k * 4, expr=filter_expr,
        )
        results = self._hybrid_search(
            reqs=[dense_req, sparse_req], ranker=RRFRanker(), limit=top_k * 3,
            output_fields=["id", "doc_id", "chunk_index", "content", "source",
                           "page_number", "file_name", "knowledge_base_id", "parent_id"],
        )

        docs, seen = [], set()
        if results and len(results) > 0:
            for hit in results[0]:
                entity = hit["entity"]
                content = entity.get("content", "")
                if content in seen:
                    continue
                seen.add(content)
                docs.append(Document(page_content=content, metadata={
                    "doc_id": entity.get("doc_id"),
                    "chunk_index": entity.get("chunk_index"),
                    "source": entity.get("source", ""),
                    "page": entity.get("page_number", ""),
                    "file_name": entity.get("file_name", ""),
                    "knowledge_base_id": entity.get("knowledge_base_id", ""),
                    "parent_id": entity.get("parent_id", ""),
                    "score": hit["distance"], "source_type": "hybrid",
                }))
        return docs[:top_k]

    def _dense_search_only(self, dense_vector, filter_expr, top_k):
        results = self._search(
            data=[dense_vector], anns_field="vector",
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k * 3, filter_expr=filter_expr,
            output_fields=["id", "doc_id", "chunk_index", "content", "source",
                           "page_number", "file_name", "knowledge_base_id", "parent_id"],
        )
        docs = []
        if results and len(results) > 0:
            for hit in results[0]:
                entity = hit["entity"]
                docs.append(Document(page_content=entity.get("content", ""), metadata={
                    "doc_id": entity.get("doc_id"),
                    "chunk_index": entity.get("chunk_index"),
                    "source": entity.get("source", ""),
                    "page": entity.get("page_number", ""),
                    "file_name": entity.get("file_name", ""),
                    "knowledge_base_id": entity.get("knowledge_base_id", ""),
                    "parent_id": entity.get("parent_id", ""),
                    "score": hit["distance"], "source_type": "dense",
                }))
        return docs[:top_k]

    # =========================================================================
    # 删除
    # =========================================================================

    def delete_by_doc_id(self, doc_id: int, knowledge_base_id: str,
                         user_id: int) -> int:
        filter_expr = (
            f"doc_id == {doc_id} and "
            f"knowledge_base_id == '{knowledge_base_id}' and "
            f"user_id == {user_id}"
        )
        result = self._delete(filter_expr)
        count = result["delete_count"] if isinstance(result, dict) else getattr(result, "delete_count", 0)
        logger.info(f"删除 doc_id={doc_id}, KB={knowledge_base_id}, {count} 条")
        return count

    def delete_by_knowledge_base(self, knowledge_base_id: str, user_id: int) -> int:
        filter_expr = (
            f"knowledge_base_id == '{knowledge_base_id}' and "
            f"user_id == {user_id}"
        )
        result = self._delete(filter_expr)
        count = result["delete_count"] if isinstance(result, dict) else getattr(result, "delete_count", 0)
        logger.info(f"删除 KB {knowledge_base_id}, 用户={user_id}, {count} 条")
        return count

    # =========================================================================
    # 统计
    # =========================================================================

    def get_chunk_count(self, knowledge_base_id: str = None, user_id: int = 0) -> int:
        parts = []
        if knowledge_base_id:
            parts.append(f"knowledge_base_id == '{knowledge_base_id}'")
        if user_id > 0:
            parts.append(f"user_id == {user_id}")
        results = self._query(
            filter_expr=" and ".join(parts) if parts else "",
            output_fields=["id"],
        )
        return len(results)


# =============================================================================
# 全局单例
# =============================================================================

_global_kb_milvus_manager: Optional[KnowledgeBaseMilvusManager] = None


def get_kb_milvus_manager(uri: str = None, token: str = None, embeddings=None,
                          collection_name: str = None,
                          dense_dim: int = None) -> KnowledgeBaseMilvusManager:
    global _global_kb_milvus_manager
    if _global_kb_milvus_manager is None:
        uri = uri or os.getenv("MILVUS_URI", "")
        if not uri:
            uri = f"http://{os.getenv('MILVUS_HOST', 'localhost')}:{os.getenv('MILVUS_PORT', '19530')}"
        token = token or os.getenv("MILVUS_TOKEN", "")

        _global_kb_milvus_manager = KnowledgeBaseMilvusManager(
            uri=uri, token=token, embeddings=embeddings,
            collection_name=collection_name or os.getenv(
                "VECTOR_STORE_COLLECTION_NAME", "ai_knowledge_collection"),
            dense_dim=dense_dim or int(os.getenv("MEMORY_EMBEDDING_DIM", "1536")),
        )
        _global_kb_milvus_manager.init_collection()
    return _global_kb_milvus_manager
