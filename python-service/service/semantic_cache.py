"""语义缓存服务

在 Milvus 中维护一个轻量集合，存储已缓存问题的 embedding，
用于精确缓存未命中时的相似问题查找。

当 Milvus 不可用时（如本地开发环境），语义缓存自动降级为不可用状态，
不影响精确缓存匹配等核心功能。
"""

import logging
import time
from typing import Optional, Tuple

from core.config import config

logger = logging.getLogger(__name__)

COLLECTION_NAME = "ai_question_cache"
SIMILARITY_THRESHOLD = 0.90  # 余弦相似度阈值


def _resolve_dense_dim() -> int:
    """根据当前 embedding 模型推导向量维度，与 vector_store.py 的 dense_dim 逻辑保持一致"""
    embedding_model = config.EMBEDDING_MODEL.lower()
    if embedding_model == "local":
        local_model = config.LOCAL_EMBEDDING_MODEL
        if "large" in local_model or "m3" in local_model:
            return 1024
        if "small" in local_model:
            return 512
        return 768
    return 1024  # dashscope text-embedding-v3


DENSE_DIM = _resolve_dense_dim()

# pymilvus 可选依赖
try:
    from pymilvus import MilvusClient, DataType
    _PYMILVUS_AVAILABLE = True
except ImportError:
    _PYMILVUS_AVAILABLE = False


class SemanticCacheStore:
    """基于 Milvus 的语义缓存存储"""

    def __init__(self):
        self._client = None
        self._embeddings = None
        self._available: Optional[bool] = None  # None 表示未初始化

    @property
    def available(self) -> bool:
        """语义缓存是否可用（Milvus 连接正常）"""
        if self._available is None:
            self._try_init()
        return self._available

    def _try_init(self):
        """尝试初始化 Milvus 连接，失败则标记不可用"""
        if not _PYMILVUS_AVAILABLE:
            logger.warning("pymilvus not installed, semantic cache disabled")
            self._available = False
            return

        try:
            uri = config.MILVUS_URI or f"http://{config.MILVUS_HOST}:{config.MILVUS_PORT}"
            client = MilvusClient(
                uri=uri,
                token=config.MILVUS_TOKEN or "",
                timeout=10,  # 短超时，快速判定
            )
            # 立即验证连接
            client.has_collection(COLLECTION_NAME)
            self._client = client
            self._available = True
            logger.info(f"Semantic cache enabled, Milvus: {uri}")
        except Exception as e:
            logger.warning(f"Semantic cache disabled (Milvus unavailable): {e}")
            self._available = False

    @property
    def client(self):
        if not self.available:
            return None
        return self._client

    @property
    def embeddings(self):
        if self._embeddings is None:
            from core.vector_store import vector_store_manager
            self._embeddings = vector_store_manager.embeddings
        return self._embeddings

    def ensure_collection(self):
        """确保集合存在，不存在则创建"""
        if not self.available:
            return
        if self._client.has_collection(COLLECTION_NAME):
            return

        schema = self._client.create_schema(
            auto_id=True,
            enable_dynamic_field=False,
        )
        schema.add_field(
            field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True
        )
        schema.add_field(
            field_name="question", datatype=DataType.VARCHAR, max_length=1024
        )
        schema.add_field(
            field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=DENSE_DIM
        )

        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )

        self._client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params,
        )
        self._client.load_collection(COLLECTION_NAME)
        logger.info(f"Created semantic cache collection: {COLLECTION_NAME} (dim={DENSE_DIM})")

    def add_question(self, question: str) -> bool:
        """将问题及其 embedding 存入 Milvus"""
        if not self.available:
            return False
        try:
            self.ensure_collection()
            embedding = self._embed_text(question)
            if embedding is None:
                return False
            data = [{"question": question, "vector": embedding}]
            self._client.insert(collection_name=COLLECTION_NAME, data=data)
            logger.debug(f"Semantic cache: added question (len={len(question)})")
            return True
        except Exception as e:
            logger.warning(f"Semantic cache add failed: {e}")
            return False

    def lookup(self, question: str) -> Tuple[bool, Optional[str], float]:
        """查找语义相似的问题

        Returns:
            (found, cache_key, similarity)
        """
        if not self.available:
            return False, None, 0.0
        try:
            self.ensure_collection()
            embedding = self._embed_text(question)
            if embedding is None:
                return False, None, 0.0

            results = self._client.search(
                collection_name=COLLECTION_NAME,
                data=[embedding],
                anns_field="vector",
                search_params={"metric_type": "COSINE", "params": {"ef": 64}},
                limit=1,
                output_fields=["question"],
            )

            if not results or not results[0]:
                return False, None, 0.0

            top = results[0][0]
            similarity = 1.0 - top["distance"]

            if similarity >= SIMILARITY_THRESHOLD:
                cache_key = top["entity"]["question"]
                logger.info(
                    f"Semantic cache HIT: query='{question[:50]}...' "
                    f"→ key='{cache_key[:50]}...' (similarity={similarity:.3f})"
                )
                return True, cache_key, similarity

            logger.debug(
                f"Semantic cache MISS: query='{question[:50]}...' "
                f"best similarity={similarity:.3f} < threshold={SIMILARITY_THRESHOLD}"
            )
            return False, None, similarity

        except Exception as e:
            logger.warning(f"Semantic cache lookup failed (will fall back to exact match): {e}")
            return False, None, 0.0

    def _embed_text(self, text: str) -> Optional[list]:
        """计算文本 embedding，带重试"""
        for attempt in range(3):
            try:
                return self.embeddings.embed_query(text)
            except Exception as e:
                logger.error(f"embed_query failed (attempt {attempt + 1}/3): {e}")
                if attempt == 2:
                    return None
                time.sleep(1)
        return None


# 模块级单例
semantic_cache_store = SemanticCacheStore()
