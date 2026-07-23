import os
import shutil
from typing import List, Optional, Dict, Any, Tuple
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings, HuggingFaceEmbeddings
from langchain_core.documents import Document
from core.config import config

# pymilvus 和 Milvus 是可选依赖，仅在使用 Milvus 时需要
try:
    from pymilvus import connections, utility
    from langchain_community.vectorstores import Milvus
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False


class VectorStoreManager:
    def __init__(self, persist_directory=None, use_milvus=None):
        """
        初始化向量存储管理器

        Args:
            persist_directory: FAISS持久化目录（仅当use_milvus=False时使用）
            use_milvus: 是否使用Milvus（默认从环境变量读取，默认值为True）
        """
        # 从环境变量读取持久化目录，默认值为./faiss_index
        self.persist_directory = persist_directory or config.VECTOR_STORE_PERSIST_DIR
        # 从环境变量读取use_milvus配置，如果未设置则默认为True
        if use_milvus is None:
            use_milvus = config.USE_MILVUS
        self.use_milvus = use_milvus
        # 从环境变量读取集合名称，默认值为ai_knowledge_collection
        self.collection_name = config.VECTOR_STORE_COLLECTION_NAME
        self.vector_store = None

        # 确定度量类型：FAISS 固定 L2，Milvus 从配置读取
        if self.use_milvus:
            self.metric_type = config.MILVUS_METRIC_TYPE
        else:
            self.metric_type = "L2"
        config.logger.info(f"Vector store metric type: {self.metric_type}")

        # 初始化Reranker
        self._init_reranker()

        # 初始化 BM25 索引
        self._init_bm25()

        # 根据 EMBEDDING_MODEL 配置选择 Embedding 模型
        embedding_model = config.EMBEDDING_MODEL.lower()

        if embedding_model == "local":
            # 使用本地中文 Embedding 模型（如 BAAI/bge-small-zh-v1.5）
            local_model = config.LOCAL_EMBEDDING_MODEL
            config.logger.info(f"Using local HuggingFace Embeddings ({local_model})")
            self.embeddings = HuggingFaceEmbeddings(model_name=local_model)
        else:
            # 默认使用阿里云 DashScope Embeddings (text-embedding-v1)
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

        # 如果 pymilvus 未安装，强制使用 FAISS
        if self.use_milvus and not MILVUS_AVAILABLE:
            config.logger.warning("pymilvus not installed, falling back to FAISS")
            self.use_milvus = False

        # 根据配置选择向量数据库
        if self.use_milvus:
            self._init_milvus()
        else:
            self._init_faiss()

    def _init_reranker(self):
        """初始化Reranker"""
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

    def _init_bm25(self):
        """初始化 BM25 索引"""
        from service.bm25 import BM25Index  # 懒加载，避免循环导入
        # 使用绝对路径，避免不同运行目录导致找不到索引
        base_dir = os.path.dirname(os.path.abspath(self.persist_directory))
        if not base_dir:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        bm25_path = os.path.join(base_dir, "bm25_index.pkl")
        self.bm25_index = BM25Index(persist_path=bm25_path)
        config.logger.info(f"BM25 index initialized ({self.bm25_index.document_count} docs)")

    def _init_milvus(self):
        """初始化Milvus连接和集合"""
        try:
            # Milvus连接配置
            milvus_host = config.MILVUS_HOST
            milvus_port = config.MILVUS_PORT

            config.logger.info(f"Connecting to Milvus at {milvus_host}:{milvus_port}")
            connections.connect(alias="default", host=milvus_host, port=milvus_port)

            # 检查连接
            if not connections.has_connection("default"):
                raise ConnectionError("Failed to connect to Milvus")

            config.logger.info("Successfully connected to Milvus")

            # 初始化Milvus向量存储
            self.vector_store = Milvus(
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
                connection_args={
                    "host": milvus_host,
                    "port": milvus_port,
                    "alias": "default"
                },
                # 自动创建集合（如果不存在）
                auto_id=True
            )

            config.logger.info(f"Milvus collection '{self.collection_name}' ready")

        except Exception as e:
            config.logger.error(f"Failed to initialize Milvus: {e}")
            config.logger.info("Falling back to FAISS...")
            self.use_milvus = False
            self._init_faiss()

    def _init_faiss(self):
        """
        初始化 FAISS 向量存储（作为 Milvus 的 fallback）
        """
        config.logger.info("Using FAISS vector store (fallback mode)")
        if os.path.exists(self.persist_directory):
            try:
                self.vector_store = FAISS.load_local(self.persist_directory, self.embeddings, allow_dangerous_deserialization=True)
                config.logger.info(f"Loaded existing FAISS index from {self.persist_directory}")
            except Exception as e:
                config.logger.error(f"Error loading existing FAISS index: {e}")
                config.logger.info("This may be caused by embedding model change (dimension mismatch). Re-initializing empty FAISS vector store...")
                # Backup old index just in case
                if os.path.exists(self.persist_directory + "_backup"):
                    shutil.rmtree(self.persist_directory + "_backup")
                shutil.move(self.persist_directory, self.persist_directory + "_backup")
                self.vector_store = None
        else:
            self.vector_store = None
            config.logger.info("No existing FAISS index found, will create new one when needed")

    def add_documents(self, documents: List[Document]):
        """
        添加文档到向量数据库
        """
        if not documents:
            return

        # 同步写入 BM25 索引
        try:
            self.bm25_index.add_documents(documents)
        except Exception as e:
            config.logger.warning(f"BM25 add_documents failed: {e}")

        if self.vector_store is None:
            if self.use_milvus:
                # Milvus会自动创建集合
                milvus_host = config.MILVUS_HOST
                milvus_port = config.MILVUS_PORT
                self.vector_store = Milvus.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    collection_name=self.collection_name,
                    connection_args={
                        "host": milvus_host,
                        "port": milvus_port,
                        "alias": "default"
                    }
                )
                config.logger.info(f"Created Milvus collection '{self.collection_name}' with {len(documents)} documents")
            else:
                # FAISS
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
                self.vector_store.save_local(self.persist_directory)
                config.logger.info(f"Created FAISS index with {len(documents)} documents")
        else:
            # 添加文档到现有存储
            if self.use_milvus:
                # Milvus添加文档
                self.vector_store.add_documents(documents)
                config.logger.info(f"Added {len(documents)} documents to Milvus")
            else:
                # FAISS添加文档
                self.vector_store.add_documents(documents)
                self.vector_store.save_local(self.persist_directory)
                config.logger.info(f"Added {len(documents)} documents to FAISS")

    def _to_similarity(self, raw_score: float) -> float:
        """
        将向量库返回的原始分数转换为统一相似度 [0, 1]，越大越相似。

        FAISS 默认 IndexFlatL2，返回 L2 欧氏距离（越小越相似）。
        Milvus 根据配置可能返回 L2 距离 / 内积(IP) / 余弦相似度(COSINE)。

        注意：FAISS/Milvus 返回的 raw_score 可能是 numpy.float32，
        float() 转换确保 FastAPI JSON 序列化不报错。
        """
        if self.metric_type == "L2":
            # L2 距离 → 相似度：1/(1+d)，范围 (0, 1]
            return float(1.0 / (1.0 + raw_score))
        elif self.metric_type == "IP":
            # 内积：直接归一化到 [0, 1]
            return float(max(0.0, min(1.0, (raw_score + 1.0) / 2.0)))
        else:
            # COSINE 或其他：假设已经是相似度
            return float(raw_score)

    def search(self, query: str, k: int = 3, filter_dict: Optional[Dict[str, Any]] = None, similarity_threshold: float = 0.75, use_rerank: bool = True, hybrid: bool = True) -> List[Document]:
        """
        相似度搜索

        Args:
            query: 查询文本
            k: 返回结果数量
            filter_dict: 过滤条件（仅Milvus支持）
            similarity_threshold: 相似度阈值
            use_rerank: 是否使用Rerank进行结果重排序
            hybrid: 是否开启向量+BM25混合检索（RRF融合）
        """
        import time
        start_time = time.time()

        if self.vector_store is None:
            config.logger.info(f"Search completed in {time.time() - start_time:.4f}s, no vector store available")
            return []

        # 混合检索：向量 + BM25 → RRF 融合
        if hybrid and self.bm25_index.document_count > 0:
            return self._hybrid_search(query, k, similarity_threshold, use_rerank, start_time)

        try:
            # 初始检索数量应该比最终返回的多，以便Rerank有足够的候选
            initial_k = k * 3 if use_rerank and self.reranker else k

            search_start = time.time()
            if self.use_milvus and filter_dict:
                # Milvus支持过滤查询
                docs_with_scores = self.vector_store.similarity_search_with_score(query, k=initial_k, filter=filter_dict)
            else:
                # FAISS或无条件查询
                docs_with_scores = self.vector_store.similarity_search_with_score(query, k=initial_k)
            search_time = time.time() - search_start
            config.logger.info(f"Vector search completed in {search_time:.4f}s, found {len(docs_with_scores) if isinstance(docs_with_scores, list) else 0} documents")

            # 提取文档
            if isinstance(docs_with_scores, list):
                if len(docs_with_scores) > 0 and isinstance(docs_with_scores[0], tuple):
                    # (doc, score) 格式
                    docs = [doc for doc, score in docs_with_scores]
                else:
                    docs = docs_with_scores
            else:
                docs = docs_with_scores

            # 如果没有启用Rerank或没有Reranker，直接返回初步检索结果
            if not use_rerank or not self.reranker:
                filtered_docs = []
                for i, (doc, score) in enumerate(docs_with_scores):
                    normalized = self._to_similarity(score)
                    # L2 距离在高维空间中数值较大（10-30），1/(1+d) 转换后
                    # 相似度可能很低（0.03-0.09），因此对 L2 只取 top-k 不做绝对阈值过滤
                    if self.metric_type == "L2" or normalized >= similarity_threshold:
                        doc.metadata['score'] = normalized
                        filtered_docs.append(doc)
                        config.logger.debug(f"Doc accepted: similarity={normalized:.4f}, raw_score={score:.4f}")
                    else:
                        config.logger.debug(f"Doc filtered: similarity={normalized:.4f} < threshold={similarity_threshold}, raw_score={score:.4f}")
                # 对 L2 限制返回数量为请求的 k，避免返回过多低质量结果
                if self.metric_type == "L2":
                    filtered_docs = filtered_docs[:k]
                config.logger.info(f"Search completed in {time.time() - start_time:.4f}s, returning {len(filtered_docs)} documents after threshold filtering")
                return filtered_docs

            # 使用Rerank进行重排序
            try:
                rerank_start = time.time()
                rerank_results = self.reranker.rerank(query, docs, top_k=k)
                rerank_time = time.time() - rerank_start
                config.logger.info(f"Rerank completed in {rerank_time:.4f}s, top {len(rerank_results)} results")

                # 提取重排序后的文档，回写 rerank 分数到 metadata
                for r in rerank_results:
                    r.document.metadata["score"] = r.score
                reranked_docs = [r.document for r in rerank_results]

                # Rerank已经按相关性排序，这里不再应用similarity_threshold
                # 但如果需要可以在这里添加额外的过滤逻辑
                config.logger.info(f"Search completed in {time.time() - start_time:.4f}s, returning {len(reranked_docs)} documents")

                return reranked_docs

            except Exception as rerank_error:
                config.logger.error(f"Rerank failed: {rerank_error}, falling back to vector search")
                # Rerank失败时，回退到原始的向量搜索结果
                filtered_docs = []
                for doc, score in docs_with_scores:
                    normalized = self._to_similarity(score)
                    if normalized >= similarity_threshold:
                        doc.metadata['score'] = normalized
                        filtered_docs.append(doc)
                config.logger.info(f"Search completed in {time.time() - start_time:.4f}s (Rerank failed, fallback to vector search), returning {len(filtered_docs)} documents")
                return filtered_docs

        except Exception as e:
            config.logger.error(f"Search error: {e}")
            # 如果 similarity_search_with_score 失败，回退到普通搜索
            try:
                fallback_start = time.time()
                if self.use_milvus and filter_dict:
                    result = self.vector_store.similarity_search(query, k=k, filter=filter_dict)
                else:
                    result = self.vector_store.similarity_search(query, k=k)
                fallback_time = time.time() - fallback_start
                config.logger.info(f"Fallback search completed in {fallback_time:.4f}s, returning {len(result)} documents")
                return result
            except Exception as e2:
                config.logger.error(f"Fallback search also failed: {e2}")
                config.logger.info(f"Search completed in {time.time() - start_time:.4f}s (all searches failed), returning empty results")
                return []

    def _hybrid_search(self, query: str, k: int, similarity_threshold: float,
                       use_rerank: bool, start_time: float) -> List[Document]:
        """向量 + BM25 混合检索，RRF 融合"""
        import time
        candidate_k = k * 2

        # 1. 向量检索
        vector_results = self._vector_search_raw(query, candidate_k)
        # 2. BM25 检索
        bm25_results = self.bm25_index.search(query, top_k=candidate_k)

        # 3. RRF 融合
        rrf_scores: Dict[str, float] = {}  # chunk_id → RRF score
        doc_by_id: Dict[str, Any] = {}

        rrf_k = 60
        for rank, (doc, _score) in enumerate(vector_results):
            chunk_id = f"doc_{doc.metadata.get('doc_id', '')}_chunk_{doc.metadata.get('chunk_index', 0)}"
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (rrf_k + rank + 1)
            doc_by_id[chunk_id] = doc

        for rank, (doc_dict, _score) in enumerate(bm25_results):
            chunk_id = f"doc_{doc_dict['metadata'].get('doc_id', '')}_chunk_{doc_dict['metadata'].get('chunk_index', 0)}"
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (rrf_k + rank + 1)
            if chunk_id not in doc_by_id:
                doc_by_id[chunk_id] = {
                    "content": doc_dict["content"],
                    "metadata": doc_dict["metadata"],
                    "is_from_bm25": True,
                }

        # 按 RRF 分排序
        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
        merged = []
        for doc_id in sorted_ids[:k]:
            entry = doc_by_id[doc_id]
            if isinstance(entry, dict) and entry.get("is_from_bm25"):
                from langchain_core.documents import Document
                doc = Document(page_content=entry["content"], metadata=entry["metadata"])
                doc.metadata["score"] = rrf_scores[doc_id]
                doc.metadata["source_type"] = "bm25"
            else:
                doc = entry
                doc.metadata["score"] = rrf_scores[doc_id]
                doc.metadata["source_type"] = "vector"
            merged.append(doc)

        config.logger.info(
            f"[Hybrid] vector={len(vector_results)}, bm25={len(bm25_results)}, "
            f"merged={len(merged)} in {time.time() - start_time:.4f}s"
        )

        # 4. 可选 reranker
        if use_rerank and self.reranker and len(merged) > 1:
            try:
                reranked = self.reranker.rerank(query, merged, top_k=k)
                for r in reranked:
                    r.document.metadata["score"] = r.score
                merged = [r.document for r in reranked]
            except Exception as e:
                config.logger.warning(f"[Hybrid] rerank failed: {e}")
                merged = merged[:k]

        return merged

    def _vector_search_raw(self, query: str, k: int) -> List[Tuple[Any, float]]:
        """纯向量检索，返回 [(doc, score), ...]"""
        try:
            docs_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
            if isinstance(docs_with_scores, list) and len(docs_with_scores) > 0:
                if isinstance(docs_with_scores[0], tuple):
                    return [(doc, self._to_similarity(score)) for doc, score in docs_with_scores]
                return [(doc, 0.5) for doc in docs_with_scores]
        except Exception as e:
            config.logger.warning(f"[Hybrid] vector search failed: {type(e).__name__}: {e}")
        return []

    def delete_document(self, doc_id: int):
        """
        根据 doc_id 删除文档向量

        Milvus: 支持高效删除
        FAISS: 标记删除（实际需要重建索引）
        """
        if self.vector_store is None:
            config.logger.warning(f"No vector store available, cannot delete doc_id: {doc_id}")
            # 仍然尝试从 BM25 删除
            try:
                self.bm25_index.remove_by_doc_id(doc_id)
            except Exception as e:
                config.logger.warning(f"BM25 remove_by_doc_id failed: {e}")
            return

        # 同步从 BM25 删除
        try:
            self.bm25_index.remove_by_doc_id(doc_id)
        except Exception as e:
            config.logger.warning(f"BM25 remove_by_doc_id failed: {e}")

        if self.use_milvus:
            # Milvus删除逻辑
            try:
                config.logger.info(f"Deleting document with doc_id: {doc_id} from Milvus")

                # 构建删除表达式（使用类型转换确保安全性）
                if not isinstance(doc_id, int):
                    raise ValueError(f"doc_id must be an integer, got {type(doc_id)}")
                delete_expr = f'doc_id in [{doc_id}]'

                # 执行删除
                result = self.vector_store.delete(expr=delete_expr)
                config.logger.info(f"Milvus delete result: {result}")

                # 可选：压缩集合以释放空间
                # utility.compact(collection_name=self.collection_name)

                config.logger.info(f"Successfully deleted document {doc_id} from Milvus")

            except Exception as e:
                config.logger.error(f"Failed to delete document from Milvus: {e}")
                # 尝试其他删除方法
                self._delete_document_fallback(doc_id)

        else:
            # FAISS删除逻辑（效率较低）
            config.logger.info(f"Deleting document with doc_id: {doc_id} from FAISS")
            self._delete_document_faiss(doc_id)

    def _delete_document_fallback(self, doc_id: int):
        """备用删除方法：通过查询找到ID然后删除"""
        try:
            # 先搜索包含该doc_id的文档
            filter_dict = {"doc_id": doc_id}
            docs_to_delete = self.search("", k=1000, filter_dict=filter_dict)

            if not docs_to_delete:
                config.logger.info(f"No documents found with doc_id: {doc_id}")
                return

            # 提取文档ID（假设metadata中有唯一ID）
            ids_to_delete = []
            for doc in docs_to_delete:
                if 'chunk_id' in doc.metadata:
                    ids_to_delete.append(doc.metadata['chunk_id'])

            if ids_to_delete:
                # 执行删除
                self.vector_store.delete(ids=ids_to_delete)
                config.logger.info(f"Deleted {len(ids_to_delete)} chunks for doc_id {doc_id}")
            else:
                config.logger.info(f"No deletable chunks found for doc_id {doc_id}")

        except Exception as e:
            config.logger.error(f"Fallback delete failed: {e}")

    def _delete_document_faiss(self, doc_id: int):
        """FAISS删除实现（需要重建索引）"""
        try:
            # 找到所有 metadata['doc_id'] == doc_id 的 ID
            ids_to_delete = []
            for doc_uuid, doc in self.vector_store.docstore._dict.items():
                if doc.metadata.get('doc_id') == doc_id:
                    ids_to_delete.append(doc_uuid)

            if ids_to_delete:
                # FAISS的delete方法可能不彻底，这里尝试删除
                self.vector_store.delete(ids_to_delete)
                self.vector_store.save_local(self.persist_directory)
                config.logger.info(f"Deleted {len(ids_to_delete)} chunks for doc_id {doc_id}")

                # 建议：定期重建FAISS索引以提高效率
                if len(ids_to_delete) > 100:
                    config.logger.warning("Large deletion in FAISS. Consider rebuilding index for better performance.")
            else:
                config.logger.info(f"No chunks found for doc_id {doc_id}")

        except Exception as e:
            config.logger.error(f"Failed to delete document from FAISS: {e}")

    def delete_collection(self):
        """
        删除整个向量库 (慎用)
        """
        if self.use_milvus:
            try:
                # 删除Milvus集合
                utility.drop_collection(self.collection_name)
                config.logger.info(f"Successfully deleted Milvus collection '{self.collection_name}'")
            except Exception as e:
                config.logger.error(f"Failed to delete Milvus collection: {e}")
        else:
            # 删除FAISS目录
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
            self.vector_store = None
            config.logger.info("Successfully deleted FAISS collection")

    def get_stats(self) -> Dict[str, Any]:
        """获取向量库统计信息"""
        stats = {
            "using_milvus": self.use_milvus,
            "collection_name": self.collection_name if self.use_milvus else None,
            "persist_directory": self.persist_directory if not self.use_milvus else None,
        }

        if self.use_milvus and self.vector_store:
            try:
                # 获取Milvus集合信息
                collection_stats = utility.get_collection_stats(self.collection_name)
                stats.update({
                    "row_count": collection_stats.get("row_count", 0),
                    "partitions": collection_stats.get("partitions", []),
                })
            except Exception as e:
                stats["error"] = f"Failed to get Milvus stats: {e}"
        elif not self.use_milvus and self.vector_store:
            # FAISS统计
            stats["doc_count"] = len(self.vector_store.docstore._dict) if hasattr(self.vector_store, 'docstore') else 0

        return stats

    def migrate_faiss_to_milvus(self):
        """将FAISS数据迁移到Milvus"""
        if not self.use_milvus or self.vector_store is None:
            config.logger.warning("Cannot migrate: not using Milvus or no vector store")
            return False

        try:
            config.logger.info("Starting migration from FAISS to Milvus...")

            # 1. 加载FAISS数据
            if os.path.exists(self.persist_directory):
                faiss_store = FAISS.load_local(self.persist_directory, self.embeddings, allow_dangerous_deserialization=True)

                # 2. 提取所有文档
                all_docs = []
                for doc_uuid, doc in faiss_store.docstore._dict.items():
                    all_docs.append(doc)

                # 3. 添加到Milvus
                if all_docs:
                    self.add_documents(all_docs)
                    config.logger.info(f"Migrated {len(all_docs)} documents from FAISS to Milvus")

                    # 4. 备份原FAISS数据
                    backup_dir = self.persist_directory + "_migrated_backup"
                    if os.path.exists(backup_dir):
                        shutil.rmtree(backup_dir)
                    shutil.move(self.persist_directory, backup_dir)
                    config.logger.info(f"Backed up FAISS data to {backup_dir}")

                    return True

            return False

        except Exception as e:
            config.logger.error(f"Migration failed: {e}")
            return False


# 创建单例实例
vector_store_manager = VectorStoreManager()

# 导出（保持兼容性，同时保留完整管理器）
vector_store = vector_store_manager
