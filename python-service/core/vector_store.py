import os
from typing import List, Optional, Dict, Any, Tuple
from langchain_community.embeddings import DashScopeEmbeddings, HuggingFaceEmbeddings
from langchain_core.documents import Document
from core.config import config


class VectorStoreManager:
    def __init__(self, embeddings=None):
        """
        初始化向量存储管理器（仅 Milvus）
        """
        self.collection_name = config.VECTOR_STORE_COLLECTION_NAME
        self.metric_type = config.MILVUS_METRIC_TYPE
        self.vector_store = None  # 兼容旧代码的属性，初始化后设为 milvus_manager

        config.logger.info(f"Vector store metric type: {self.metric_type}")

        # 初始化 Reranker
        self._init_reranker()

        # 初始化 Embedding 模型
        if embeddings:
            self.embeddings = embeddings
        else:
            self._init_embeddings()

        # 初始化 Milvus
        self._init_milvus()

    def _init_embeddings(self):
        """根据配置选择 Embedding 模型"""
        embedding_model = config.EMBEDDING_MODEL.lower()
        if embedding_model == "local":
            local_model = config.LOCAL_EMBEDDING_MODEL
            config.logger.info(f"Using local HuggingFace Embeddings ({local_model})")
            self.embeddings = HuggingFaceEmbeddings(model_name=local_model)
        else:
            api_key = config.DASHSCOPE_API_KEY
            if api_key:
                config.logger.info("Using DashScope Embeddings (text-embedding-v1)")
                self.embeddings = DashScopeEmbeddings(
                    model="text-embedding-v1",
                    dashscope_api_key=api_key
                )
            else:
                config.logger.warning("DASHSCOPE_API_KEY not found. Falling back to local HuggingFace Embeddings.")
                self.embeddings = HuggingFaceEmbeddings(model_name=config.LOCAL_EMBEDDING_MODEL)

    def _init_reranker(self):
        """初始化 Reranker"""
        reranker_type = config.RERANKER_TYPE
        if reranker_type == "none":
            self.reranker = None
            config.logger.info("Reranker is disabled")
            return
        try:
            from service.reranker import create_reranker
            self.reranker = create_reranker(reranker_type)
            config.logger.info(f"Reranker initialized: {reranker_type}")
        except Exception as e:
            config.logger.error(f"Failed to initialize reranker: {e}")
            self.reranker = None

    @property
    def milvus_manager(self):
        """延迟加载 KnowledgeBaseMilvusManager"""
        if not hasattr(self, '_milvus_manager'):
            from core.milvus_kb import KnowledgeBaseMilvusManager
            milvus_uri = os.getenv("MILVUS_URI", "")
            if not milvus_uri:
                milvus_uri = f"http://{config.MILVUS_HOST}:{config.MILVUS_PORT}"
            milvus_token = os.getenv("MILVUS_TOKEN", "")

            # 根据当前 embedding 模型确定维度
            model_name = config.EMBEDDING_MODEL.lower()
            if model_name == "local":
                local_model = config.LOCAL_EMBEDDING_MODEL
                if "large" in local_model or "m3" in local_model:
                    dense_dim = 1024
                elif "small" in local_model:
                    dense_dim = 512
                else:
                    dense_dim = 768
            else:
                dense_dim = 1536  # dashscope text-embedding-v1

            self._milvus_manager = KnowledgeBaseMilvusManager(
                uri=milvus_uri,
                token=milvus_token,
                embeddings=self.embeddings,
                collection_name=self.collection_name,
                dense_dim=dense_dim,
            )
            self._milvus_manager.init_collection()
        return self._milvus_manager

    def _init_milvus(self):
        """初始化 Milvus"""
        try:
            config.logger.info(
                f"Connecting to Milvus at {config.MILVUS_HOST}:{config.MILVUS_PORT}"
            )
            _ = self.milvus_manager
            self.metric_type = "COSINE"
            self.vector_store = self.milvus_manager
            config.logger.info(
                f"Milvus collection '{self.collection_name}' ready (native BM25)"
            )
            # 预热 MemoryManager
            try:
                from memory.memory_manager import get_memory_manager
                get_memory_manager(embeddings=self.embeddings)
            except Exception as e:
                config.logger.warning(f"MemoryManager 预热失败: {e}")
        except Exception as e:
            config.logger.error(f"Failed to initialize Milvus: {e}")
            raise

    def add_documents(self, documents: List[Document],
                      knowledge_base_id: str = "默认知识库", user_id: int = 0,
                      parent_documents: List[Document] = None):
        """添加文档到 Milvus"""
        if not documents:
            return

        # 父子块模式
        if parent_documents:
            from core.mysql_client import mysql_client
            doc_id = documents[0].metadata.get("doc_id", 0)
            parent_data = [
                {"parent_id": d.metadata.get("parent_id", ""),
                 "chunk_text": d.page_content,
                 "chunk_index": d.metadata.get("parent_chunk_index", i),
                 "page_number": d.metadata.get("page", 1),
                 "source": d.metadata.get("source", "")}
                for i, d in enumerate(parent_documents)
            ]
            mysql_client.insert_chunks(doc_id, parent_data, parent_mode=True)
            count = self.milvus_manager.add_chunks(documents, knowledge_base_id, user_id)
            config.logger.info(
                f"Added {len(parent_data)} parents + {count} children to Milvus (KB: {knowledge_base_id})"
            )
            return

        # 普通模式
        try:
            count = self.milvus_manager.add_chunks(documents, knowledge_base_id, user_id)
            config.logger.info(f"Added {count} chunks to Milvus (KB: {knowledge_base_id})")
        except Exception as e:
            config.logger.error(f"Milvus add_chunks failed: {e}")
            raise

    def _to_similarity(self, raw_score: float) -> float:
        """将原始分数转换为统一相似度 [0, 1]"""
        if self.metric_type == "L2":
            return float(1.0 / (1.0 + raw_score))
        elif self.metric_type == "IP":
            return float(max(0.0, min(1.0, (raw_score + 1.0) / 2.0)))
        else:
            return float(raw_score)

    def search(self, query: str, k: int = 3, filter_dict: Optional[Dict[str, Any]] = None,
               similarity_threshold: float = 0.75, use_rerank: bool = True, hybrid: bool = True,
               knowledge_base_id: str = None, user_id: int = 0,
               skip_parent_fetch: bool = False) -> List[Document]:
        """Milvus 混合检索（Dense + BM25, RRF 融合）"""
        import time
        start_time = time.time()

        if self.vector_store is None:
            config.logger.info(f"Search completed in {time.time() - start_time:.4f}s, no vector store available")
            return []

        kb_id = knowledge_base_id or "默认知识库"
        try:
            docs = self.milvus_manager.hybrid_search(
                query=query, top_k=k,
                knowledge_base_id=kb_id, user_id=user_id,
            )
        except Exception as e:
            config.logger.error(f"Milvus hybrid search failed: {e}")
            docs = []

        # 父子块：子块检索结果 → 查 MySQL 父块
        if not skip_parent_fetch:
            docs = self._fetch_parent_chunks(docs)

        # Rerank
        if docs and use_rerank and self.reranker:
            try:
                rerank_results = self.reranker.rerank(query, docs, top_k=k)
                for r in rerank_results:
                    r.document.metadata["score"] = r.score
                docs = [r.document for r in rerank_results]
            except Exception as e:
                config.logger.warning(f"Rerank failed: {e}")

        config.logger.info(
            f"Milvus search completed in {time.time() - start_time:.4f}s, "
            f"returning {len(docs)} documents"
        )
        return docs

    def _fetch_parent_chunks(self, child_docs: List[Document]) -> List[Document]:
        """从子块提取 parent_id → 查 MySQL → 返回父块（去重+分数继承）"""
        if not child_docs:
            return []
        parent_ids = list(set(
            d.metadata.get("parent_id", "") for d in child_docs
            if d.metadata.get("parent_id", "")
        ))
        if not parent_ids:
            return child_docs

        from core.mysql_client import mysql_client
        rows = mysql_client.get_parent_chunks_by_ids(parent_ids)
        parent_map = {r["parent_id"]: r for r in rows}

        parent_scores = {}
        for doc in child_docs:
            pid = doc.metadata.get("parent_id", "")
            if not pid:
                continue
            s = doc.metadata.get("score", 0)
            if pid not in parent_scores or s > parent_scores[pid]:
                parent_scores[pid] = s

        parent_docs = []
        for pid, score in sorted(parent_scores.items(), key=lambda x: x[1], reverse=True):
            row = parent_map.get(pid)
            if not row:
                continue
            parent_docs.append(Document(page_content=row["chunk_text"], metadata={
                "doc_id": row["doc_id"], "parent_id": pid,
                "source": row.get("source", ""), "page": row.get("page_number", ""),
                "score": score, "source_type": "parent_child",
            }))
        return parent_docs[:config.PARENT_CHILD_MAX_PARENTS] if parent_docs else child_docs

    def delete_document(self, doc_id: int, knowledge_base_id: str = "默认知识库",
                         user_id: int = 0):
        """根据 doc_id 删除文档向量"""
        try:
            config.logger.info(f"Deleting doc_id={doc_id} from Milvus (KB: {knowledge_base_id})")
            count = self.milvus_manager.delete_by_doc_id(doc_id, knowledge_base_id, user_id)
            config.logger.info(f"Deleted {count} chunks for doc_id={doc_id} from Milvus")
        except Exception as e:
            config.logger.error(f"Failed to delete from Milvus: {e}")

    def delete_collection(self):
        """删除整个向量库 (慎用)"""
        try:
            self.milvus_manager.client.drop_collection(self.collection_name)
            config.logger.info(f"Deleted Milvus collection '{self.collection_name}'")
        except Exception as e:
            config.logger.error(f"Failed to delete Milvus collection: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """获取向量库统计信息"""
        stats = {
            "using_milvus": True,
            "collection_name": self.collection_name,
        }
        if self.vector_store:
            try:
                row_count = self.milvus_manager.get_chunk_count()
                stats["row_count"] = row_count
            except Exception as e:
                stats["error"] = str(e)
        return stats


# 创建单例实例
vector_store_manager = VectorStoreManager()

# 导出（保持兼容性）
vector_store = vector_store_manager
