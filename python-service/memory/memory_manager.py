"""L1 长期分层记忆管理器

继承 MilvusStore 基类，负责 Milvus 集合的初始化、混合检索（Dense + BM25）、
三维评分排序、冲突去重、批量写入。

三种记忆类型：
- semantic   (语义记忆): 稳定的事实、知识、偏好
- episodic   (情景记忆): 具体的历史互动事件
- procedural (程序记忆): 可复用的方法、工作流
"""

import os
import time
import logging
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from core.milvus_store import MilvusStore

load_dotenv()

logger = logging.getLogger(__name__)


class MemoryManager(MilvusStore):
    """长期分层记忆管理器（三层记忆中的 L1 层）"""

    # 检索评分参数
    DEFAULT_ALPHA = 0.45
    DEFAULT_BETA = 0.25
    DEFAULT_GAMMA = 0.30
    DECAY_RATE = 0.995

    DEFAULT_TYPE_WEIGHTS = {"semantic": 1.3, "episodic": 1.0, "procedural": 1.2}
    CONFLICT_THRESHOLD = 0.9

    def __init__(self, uri: str, token: str, embeddings,
                 collection_name: str = "long_term_memory", dense_dim: int = 1536,
                 alpha: float = None, beta: float = None, gamma: float = None,
                 type_weights: Dict[str, float] = None,
                 conflict_threshold: float = None):
        super().__init__(uri, token, embeddings, collection_name, dense_dim)
        self.alpha = alpha or self.DEFAULT_ALPHA
        self.beta = beta or self.DEFAULT_BETA
        self.gamma = gamma or self.DEFAULT_GAMMA
        self.type_weights = type_weights or self.DEFAULT_TYPE_WEIGHTS
        self.conflict_threshold = conflict_threshold or self.CONFLICT_THRESHOLD

    # =========================================================================
    # Collection 初始化
    # =========================================================================

    def init_collection(self) -> bool:
        if self._loaded:
            return True
        try:
            if self.has_collection():
                logger.info(f"记忆集合 {self.collection_name} 已存在")
                self.load_collection()
                self._loaded = True
                return True

            self._create_collection()
            self._loaded = True
            return True
        except Exception as e:
            logger.error(f"初始化记忆集合失败: {e}")
            raise

    def _create_collection(self):
        from pymilvus import DataType, Function, FunctionType

        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=True)

        schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=50,
                         is_primary=True, auto_id=True)
        schema.add_field(field_name="user_id", datatype=DataType.INT64,
                         is_partition_key=True)
        schema.add_field(field_name="conversation_id", datatype=DataType.VARCHAR,
                         max_length=100)
        schema.add_field(field_name="memory_type", datatype=DataType.VARCHAR,
                         max_length=20)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR,
                         max_length=65535, enable_analyzer=True,
                         analyzer_params={"type": "chinese"})
        schema.add_field(field_name="importance", datatype=DataType.FLOAT)
        schema.add_field(field_name="created_at", datatype=DataType.INT64)
        schema.add_field(field_name="last_access_at", datatype=DataType.INT64,
                         default_value=0)
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
        for field in ["user_id", "importance", "last_access_at", "created_at"]:
            index_params.add_index(field_name=field, index_type="STL_SORT")

        logger.info(f"正在创建记忆集合 {self.collection_name}（含 BM25 Function）...")
        self.client.create_collection(
            collection_name=self.collection_name, schema=schema,
            index_params=index_params, properties={"partitionkey.isolation": True},
        )
        logger.info(f"记忆集合 {self.collection_name} 创建完成，正在加载...")
        self.load_collection()
        logger.info(f"记忆集合 {self.collection_name} 创建成功")

    # =========================================================================
    # 混合检索
    # =========================================================================

    def hybrid_retrieval_memories(
        self, query: str, user_id: int,
        semantic_k: int = 3, episodic_k: int = 2, procedural_k: int = 2,
    ) -> Dict[str, List[Dict]]:
        memory_configs = {
            "semantic": {"k": semantic_k,
                         "filter": f"user_id == {user_id} and memory_type == 'semantic'"},
            "episodic": {"k": episodic_k,
                         "filter": f"user_id == {user_id} and memory_type == 'episodic'"},
            "procedural": {"k": procedural_k,
                           "filter": f"user_id == {user_id} and memory_type == 'procedural'"},
        }
        memory_configs = {k: v for k, v in memory_configs.items() if v["k"] > 0}
        if not memory_configs:
            return {"semantic": [], "episodic": [], "procedural": []}

        dense_vector = self._embed_query(query)
        if not dense_vector:
            return {"semantic": [], "episodic": [], "procedural": []}

        final_results = {"semantic": [], "episodic": [], "procedural": []}
        for mem_type, config in memory_configs.items():
            try:
                final_results[mem_type] = self._hybrid_search_single_type(
                    dense_vector, query, config["filter"], config["k"])
            except Exception as e:
                logger.error(f"记忆类型 {mem_type} 混合检索失败: {e}")
                try:
                    final_results[mem_type] = self._dense_search_single_type(
                        dense_vector, config["filter"], config["k"])
                except Exception as fallback_e:
                    logger.error(f"记忆类型 {mem_type} 降级检索也失败: {fallback_e}")
                    final_results[mem_type] = []

        return self._get_top_k_memories(final_results, memory_configs)

    def _hybrid_search_single_type(self, dense_vector, raw_query, filter_expr, k):
        from pymilvus import AnnSearchRequest, RRFRanker

        dense_req = AnnSearchRequest(
            data=[dense_vector], anns_field="vector",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=k * 6, expr=filter_expr,
        )
        sparse_req = AnnSearchRequest(
            data=[raw_query], anns_field="sparse_vector",
            param={"metric_type": "BM25"}, limit=k * 4, expr=filter_expr,
        )
        results = self._hybrid_search(
            reqs=[dense_req, sparse_req], ranker=RRFRanker(), limit=k * 3,
            output_fields=["id", "user_id", "conversation_id", "memory_type",
                           "content", "importance", "last_access_at"],
        )

        memories, seen_ids = [], set()
        if results and len(results) > 0:
            for hit in results[0]:
                entity = hit["entity"]
                mid = entity.get("id")
                if mid and mid not in seen_ids:
                    seen_ids.add(mid)
                    memories.append({
                        "id": mid, "user_id": entity.get("user_id"),
                        "conversation_id": entity.get("conversation_id"),
                        "memory_type": entity.get("memory_type"),
                        "content": entity.get("content"),
                        "importance": entity.get("importance", 0.5),
                        "last_access_at": entity.get("last_access_at", 0),
                        "score": hit["distance"],
                    })
        return memories[: k * 2]

    def _dense_search_single_type(self, dense_vector, filter_expr, k):
        results = self._search(
            data=[dense_vector], anns_field="vector",
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=k * 6, filter_expr=filter_expr,
            output_fields=["id", "user_id", "conversation_id", "memory_type",
                           "content", "importance", "last_access_at"],
        )
        memories = []
        if results and len(results) > 0:
            for hit in results[0]:
                entity = hit["entity"]
                memories.append({
                    "id": entity.get("id"), "user_id": entity.get("user_id"),
                    "conversation_id": entity.get("conversation_id"),
                    "memory_type": entity.get("memory_type"),
                    "content": entity.get("content"),
                    "importance": entity.get("importance", 0.5),
                    "last_access_at": entity.get("last_access_at", 0),
                    "score": hit["distance"],
                })
        return memories[: k * 2]

    # =========================================================================
    # 三维评分排序
    # =========================================================================

    def _get_top_k_memories(self, memory_dict, memory_configs):
        current_time = time.time()
        result = {}
        for mem_type, memories in memory_dict.items():
            if not memories or mem_type not in memory_configs:
                result[mem_type] = []
                continue

            type_weight = self.type_weights.get(mem_type, 1.0)
            scored = []
            for mem in memories:
                semantic_score = mem.get("score", 0.5)
                hours_passed = (current_time - mem.get("last_access_at", current_time)) / 3600
                recency_score = self.DECAY_RATE ** max(hours_passed, 0)
                final_score = (
                    self.alpha * semantic_score
                    + self.beta * recency_score
                    + self.gamma * mem.get("importance", 0.5)
                ) * type_weight
                scored.append((mem, final_score))

            scored.sort(key=lambda x: x[1], reverse=True)
            top_k = memory_configs[mem_type]["k"]
            result[mem_type] = [
                {"id": m["id"], "user_id": m.get("user_id"),
                 "conversation_id": m.get("conversation_id"),
                 "memory_type": m.get("memory_type"),
                 "content": m.get("content"),
                 "last_access_at": m.get("last_access_at")}
                for m, _ in scored[:top_k]
            ]
        return result

    # =========================================================================
    # 冲突去重
    # =========================================================================

    def resolve_conflicts(self, filtered_memory: Dict[str, Any], user_id: int
                           ) -> Dict[str, Any]:
        type_map = {"semantic_memory": "semantic", "episodic_memory": "episodic",
                    "procedural_memory": "procedural"}
        items = []
        for key, mem_type in type_map.items():
            for idx, mem in enumerate(filtered_memory.get(key, [])):
                if isinstance(mem, dict) and mem.get("content", "").strip():
                    items.append((key, idx, mem["content"], mem_type))
        if not items:
            return filtered_memory

        contents = [item[2] for item in items]
        vectors = self._embed_documents(contents)
        if not vectors or len(vectors) != len(items):
            return filtered_memory

        from collections import defaultdict
        type_to_indices = defaultdict(list)
        for i, (_, _, _, mem_type) in enumerate(items):
            type_to_indices[mem_type].append(i)

        is_similar = [False] * len(items)
        for mem_type, indices in type_to_indices.items():
            type_vectors = [vectors[i] for i in indices]
            try:
                search_results = self._search(
                    data=type_vectors, anns_field="vector",
                    search_params={"metric_type": "COSINE", "params": {"ef": 64}},
                    limit=1,
                    filter_expr=f"user_id == {user_id} and memory_type == '{mem_type}'",
                    output_fields=["distance"],
                )
                for j, sr in enumerate(search_results):
                    if sr and sr[0]["distance"] >= self.conflict_threshold:
                        is_similar[indices[j]] = True
            except Exception as e:
                logger.error(f"冲突检测搜索失败 (type={mem_type}): {e}")

        to_remove = {k: [] for k in type_map}
        for i, (key, idx, _, _) in enumerate(items):
            if is_similar[i] and idx is not None:
                to_remove[key].append(idx)

        for key, indices in to_remove.items():
            for idx in sorted(indices, reverse=True):
                if idx < len(filtered_memory[key]):
                    logger.info(f"去重: 丢弃 {key}[{idx}]")
                    del filtered_memory[key][idx]
        return filtered_memory

    # =========================================================================
    # 批量写入
    # =========================================================================

    def add_memories_batch(self, user_id: int, conversation_id: str,
                           memory_dict: Dict[str, Any]) -> bool:
        created_at = int(time.time())
        type_mapping = {"semantic_memory": "semantic", "episodic_memory": "episodic",
                        "procedural_memory": "procedural"}

        texts, records = [], []
        for key, memory_type in type_mapping.items():
            for item in memory_dict.get(key, []) if isinstance(memory_dict.get(key), list) else []:
                if not isinstance(item, dict):
                    continue
                content = item.get("content", "").strip()
                if not content:
                    continue
                texts.append(content)
                records.append({
                    "user_id": user_id, "conversation_id": conversation_id,
                    "memory_type": memory_type, "content": content,
                    "importance": float(item.get("importance_score", 0.5)),
                    "created_at": created_at, "last_access_at": created_at,
                })

        if not texts:
            return False

        vectors = self._embed_documents(texts)
        if not vectors or len(vectors) != len(records):
            return False
        for r, v in zip(records, vectors):
            r["vector"] = v

        self._insert(records)
        logger.info(f"批量写入记忆完成 — {len(records)} 条, 用户={user_id}, 会话={conversation_id}")
        return True

    # =========================================================================
    # 统计 / 删除
    # =========================================================================

    def get_memory_count(self, user_id: int, memory_type: str = None) -> int:
        filter_expr = f"user_id == {user_id}"
        if memory_type:
            filter_expr += f" and memory_type == '{memory_type}'"
        return len(self._query(filter_expr, ["id"]))

    def delete_user_memories(self, user_id: int) -> bool:
        self._delete(f"user_id == {user_id}")
        logger.info(f"已删除用户 {user_id} 的所有记忆")
        return True


# =============================================================================
# 工厂函数
# =============================================================================

_global_memory_manager: Optional[MemoryManager] = None


def get_memory_manager(embeddings=None) -> MemoryManager:
    global _global_memory_manager
    if _global_memory_manager is None:
        from langchain_community.embeddings import HuggingFaceEmbeddings, DashScopeEmbeddings
        from core.config import config as app_config

        uri = os.getenv("MILVUS_URI", "")
        if not uri:
            uri = f"http://{os.getenv('MILVUS_HOST', 'localhost')}:{os.getenv('MILVUS_PORT', '19530')}"
        token = os.getenv("MILVUS_TOKEN", "")
        collection = os.getenv("MEMORY_COLLECTION", "long_term_memory")

        if embeddings is not None:
            logger.info("MemoryManager: using shared Embeddings")
            try:
                dense_dim = len(embeddings.embed_query("test"))
            except Exception:
                dense_dim = int(os.getenv("MEMORY_EMBEDDING_DIM", "512"))
        else:
            if app_config.EMBEDDING_MODEL.lower() == "local":
                local_model = app_config.LOCAL_EMBEDDING_MODEL
                logger.info(f"MemoryManager: using local Embeddings ({local_model})")
                embeddings = HuggingFaceEmbeddings(model_name=local_model)
                dense_dim = 512 if "small" in local_model else (1024 if "large" in local_model or "m3" in local_model else 768)
            else:
                model_name = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-v1")
                logger.info(f"MemoryManager: using DashScope Embeddings ({model_name})")
                embeddings = DashScopeEmbeddings(model=model_name,
                                                  dashscope_api_key=os.getenv("DASHSCOPE_API_KEY", ""))
                dense_dim = int(os.getenv("MEMORY_EMBEDDING_DIM", "1536"))

        _global_memory_manager = MemoryManager(
            uri=uri, token=token, embeddings=embeddings,
            collection_name=collection, dense_dim=dense_dim,
        )
        _global_memory_manager.init_collection()
    return _global_memory_manager
