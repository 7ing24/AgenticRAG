import os
import shutil
from typing import List, Optional, Dict, Any, Tuple
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings, HuggingFaceEmbeddings
from langchain_core.documents import Document
from core.config import config

# pymilvus 是可选依赖
try:
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

        # BM25 索引（仅 FAISS 路径需要，Milvus 原生内置）
        if not self.use_milvus:
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

    @property
    def milvus_manager(self):
        """延迟加载 KnowledgeBaseMilvusManager（原生 BM25 混合检索）"""
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
        """初始化 Milvus — 使用原生 KnowledgeBaseMilvusManager"""
        try:
            config.logger.info(
                f"Connecting to Milvus at {config.MILVUS_HOST}:{config.MILVUS_PORT}"
            )
            _ = self.milvus_manager  # 触发知识库集合初始化
            self.metric_type = "COSINE"
            self.vector_store = self.milvus_manager  # 标记已初始化
            config.logger.info(
                f"Milvus collection '{self.collection_name}' ready (native BM25)"
            )

            # 预热 MemoryManager，共享 embeddings 避免重复加载模型
            try:
                from memory.memory_manager import get_memory_manager
                get_memory_manager(embeddings=self.embeddings)
            except Exception as e:
                config.logger.warning(f"MemoryManager 预热失败: {e}")
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

    def add_documents(self, documents: List[Document],
                      knowledge_base_id: str = "默认知识库", user_id: int = 0,
                      parent_documents: List[Document] = None):
        """
        添加文档到向量数据库。parent_documents 不为空时走父子块模式。
        """
        if not documents:
            return

        # ── 父子块模式 ──
        if parent_documents and self.use_milvus:
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

            count = self.milvus_manager.add_chunks(
                documents, knowledge_base_id, user_id
            )
            config.logger.info(
                f"Added {len(parent_data)} parents + {count} children to Milvus (KB: {knowledge_base_id})"
            )
            return

        # ── Milvus 原生写入 ──
        if self.use_milvus:
            try:
                count = self.milvus_manager.add_chunks(
                    documents, knowledge_base_id, user_id
                )
                config.logger.info(
                    f"Added {count} chunks to Milvus (KB: {knowledge_base_id})"
                )
            except Exception as e:
                config.logger.error(f"Milvus add_chunks failed: {e}")
                raise
            return

        # ── FAISS 路径 ──
        try:
            self.bm25_index.add_documents(documents)
        except Exception as e:
            config.logger.warning(f"BM25 add_documents failed: {e}")

        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            self.vector_store.save_local(self.persist_directory)
            config.logger.info(f"Created FAISS index with {len(documents)} documents")
        else:
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

    def search(self, query: str, k: int = 3, filter_dict: Optional[Dict[str, Any]] = None,
               similarity_threshold: float = 0.75, use_rerank: bool = True, hybrid: bool = True,
               knowledge_base_id: str = None, user_id: int = 0,
               skip_parent_fetch: bool = False) -> List[Document]:
        """
        相似度搜索

        Args:
            query: 查询文本
            k: 返回结果数量
            filter_dict: 过滤条件（仅 Milvus 旧 API 使用）
            similarity_threshold: 相似度阈值
            use_rerank: 是否使用 Rerank 重排序
            hybrid: 是否使用 Dense + BM25 混合检索
            knowledge_base_id: 知识库 ID（Milvus 原生检索时用于过滤）
            user_id: 用户 ID（Milvus 原生检索时用于过滤）
        """
        import time
        start_time = time.time()

        if self.vector_store is None:
            config.logger.info(f"Search completed in {time.time() - start_time:.4f}s, no vector store available")
            return []

        # ── Milvus 原生混合检索（Dense + BM25, RRF 融合）──
        if self.use_milvus and hybrid:
            kb_id = knowledge_base_id or "默认知识库"
            try:
                docs = self.milvus_manager.hybrid_search(
                    query=query, top_k=k,
                    knowledge_base_id=kb_id, user_id=user_id,
                )
            except Exception as e:
                config.logger.error(f"Milvus native hybrid search failed: {e}")
                docs = []

            # 父子块：子块检索结果 → 查 MySQL 父块（baseline评测时跳过）
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
                f"Milvus native hybrid search completed in {time.time() - start_time:.4f}s, "
                f"returning {len(docs)} documents"
            )
            return docs

        # ── Milvus 纯 Dense 检索（hybrid=False 时）──
        if self.use_milvus and not hybrid:
            try:
                docs = self.milvus_manager.hybrid_search(
                    query=query, top_k=k,
                    knowledge_base_id=knowledge_base_id or "默认知识库",
                    user_id=user_id,
                )
                if not skip_parent_fetch:
                    docs = self._fetch_parent_chunks(docs)
                if use_rerank and self.reranker and docs:
                    reranked = self.reranker.rerank(query, docs, top_k=k)
                    docs = [r.document for r in reranked]
                return docs
            except Exception as e:
                config.logger.error(f"Milvus dense search failed: {e}")
                return []

        # ── FAISS 路径（含自定义 BM25 混合检索）──
        if hybrid and self.bm25_index.document_count > 0:
            return self._hybrid_search(query, k, similarity_threshold, use_rerank, start_time)

        try:
            initial_k = k * 3 if use_rerank and self.reranker else k
            search_start = time.time()
            docs_with_scores = self.vector_store.similarity_search_with_score(query, k=initial_k)
            search_time = time.time() - search_start
            config.logger.info(
                f"FAISS search completed in {search_time:.4f}s, "
                f"found {len(docs_with_scores) if isinstance(docs_with_scores, list) else 0} documents"
            )

            if isinstance(docs_with_scores, list) and len(docs_with_scores) > 0 and isinstance(docs_with_scores[0], tuple):
                docs = [doc for doc, _ in docs_with_scores]
            else:
                docs = list(docs_with_scores) if docs_with_scores else []

            if not use_rerank or not self.reranker:
                filtered = []
                for item in docs_with_scores:
                    if isinstance(item, tuple):
                        doc, score = item
                    else:
                        doc, score = item, 0.5
                    normalized = self._to_similarity(score)
                    if self.metric_type == "L2" or normalized >= similarity_threshold:
                        doc.metadata['score'] = normalized
                        filtered.append(doc)
                if self.metric_type == "L2":
                    filtered = filtered[:k]
                config.logger.info(f"Search completed in {time.time() - start_time:.4f}s, returning {len(filtered)} documents")
                return filtered

            try:
                rerank_results = self.reranker.rerank(query, docs, top_k=k)
                for r in rerank_results:
                    r.document.metadata["score"] = r.score
                docs = [r.document for r in rerank_results]
                config.logger.info(f"Search completed in {time.time() - start_time:.4f}s, returning {len(docs)} documents")
                return docs
            except Exception as rerank_error:
                config.logger.error(f"Rerank failed: {rerank_error}, falling back to FAISS raw results")
                filtered = []
                for item in docs_with_scores:
                    if isinstance(item, tuple):
                        doc, score = item
                    else:
                        doc, score = item, 0.5
                    normalized = self._to_similarity(score)
                    if normalized >= similarity_threshold:
                        doc.metadata['score'] = normalized
                        filtered.append(doc)
                return filtered

        except Exception as e:
            config.logger.error(f"Search error: {e}")
            try:
                return self.vector_store.similarity_search(query, k=k)
            except Exception as e2:
                config.logger.error(f"Fallback search also failed: {e2}")
                return []

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

    def delete_document(self, doc_id: int, knowledge_base_id: str = "默认知识库",
                         user_id: int = 0):
        """
        根据 doc_id 删除文档向量
        """
        if self.use_milvus:
            try:
                config.logger.info(
                    f"Deleting doc_id={doc_id} from Milvus "
                    f"(KB: {knowledge_base_id})"
                )
                count = self.milvus_manager.delete_by_doc_id(
                    doc_id, knowledge_base_id, user_id
                )
                config.logger.info(
                    f"Deleted {count} chunks for doc_id={doc_id} from Milvus"
                )
            except Exception as e:
                config.logger.error(f"Failed to delete from Milvus: {e}")
        else:
            config.logger.info(f"Deleting doc_id={doc_id} from FAISS")
            self._delete_document_faiss(doc_id)

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
        """删除整个向量库 (慎用)"""
        if self.use_milvus:
            try:
                self.milvus_manager.client.drop_collection(self.collection_name)
                config.logger.info(f"Deleted Milvus collection '{self.collection_name}'")
            except Exception as e:
                config.logger.error(f"Failed to delete Milvus collection: {e}")
        else:
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
            self.vector_store = None
            config.logger.info("Deleted FAISS collection")

    def get_stats(self) -> Dict[str, Any]:
        """获取向量库统计信息"""
        stats = {
            "using_milvus": self.use_milvus,
            "collection_name": self.collection_name,
        }

        if self.use_milvus and self.vector_store:
            try:
                row_count = self.milvus_manager.get_chunk_count()
                stats["row_count"] = row_count
            except Exception as e:
                stats["error"] = str(e)
        elif not self.use_milvus and self.vector_store:
            stats["doc_count"] = (
                len(self.vector_store.docstore._dict)
                if hasattr(self.vector_store, 'docstore') else 0
            )
            stats["persist_directory"] = self.persist_directory

        return stats


# 创建单例实例
vector_store_manager = VectorStoreManager()

# 导出（保持兼容性，同时保留完整管理器）
vector_store = vector_store_manager
