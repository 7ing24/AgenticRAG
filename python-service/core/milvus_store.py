"""Milvus 存储基类

封装 MilvusClient 初始化、集合生命周期、Embedding、CRUD 等通用逻辑。
MemoryManager 和 KnowledgeBaseMilvusManager 继承此类。
"""

import os
import time
import logging
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class MilvusStore:
    """Milvus 存储基类"""

    def __init__(self, uri: str, token: str, embeddings,
                 collection_name: str, dense_dim: int):
        self.uri = uri
        self.token = token
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.dense_dim = dense_dim
        self._client = None
        self._loaded = False

    # ── MilvusClient ──────────────────────────────────────────────

    @property
    def client(self):
        if self._client is None:
            from pymilvus import MilvusClient
            self._client = MilvusClient(uri=self.uri, token=self.token, timeout=300)
        return self._client

    # ── Collection 生命周期 ───────────────────────────────────────

    def has_collection(self) -> bool:
        return self.client.has_collection(self.collection_name)

    def load_collection(self):
        self.client.load_collection(self.collection_name)

    def drop_collection(self):
        self.client.drop_collection(self.collection_name)

    # ── Embedding ─────────────────────────────────────────────────

    def _embed_query(self, text: str) -> Optional[List[float]]:
        for attempt in range(3):
            try:
                return self.embeddings.embed_query(text)
            except Exception as e:
                logger.error(f"embed_query 失败 (尝试 {attempt + 1}/3): {e}")
                if attempt == 2:
                    return None
                time.sleep(1)
        return None

    def _embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self.embeddings.embed_documents(texts)

    # ── CRUD 快捷方法 ─────────────────────────────────────────────

    def _insert(self, data: list):
        self.client.insert(collection_name=self.collection_name, data=data)

    def _search(self, data: list, anns_field: str, search_params: dict,
                limit: int, filter_expr: str, output_fields: List[str]):
        return self.client.search(
            collection_name=self.collection_name,
            data=data, anns_field=anns_field,
            search_params=search_params, limit=limit,
            filter=filter_expr, output_fields=output_fields,
        )

    def _hybrid_search(self, reqs: list, ranker, limit: int,
                       output_fields: List[str]):
        return self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=reqs, ranker=ranker, limit=limit,
            output_fields=output_fields,
        )

    def _query(self, filter_expr: str, output_fields: List[str],
               limit: int = 10000):
        return self.client.query(
            collection_name=self.collection_name,
            filter=filter_expr, output_fields=output_fields, limit=limit,
        )

    def _delete(self, filter_expr: str):
        return self.client.delete(
            collection_name=self.collection_name, filter=filter_expr,
        )
