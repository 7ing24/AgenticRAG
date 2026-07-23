"""BM25 关键词检索索引 — 与向量检索互补，支持混合检索"""

import os
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple

import jieba
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class BM25Index:
    """BM25 关键词索引，维护文档列表和分词后的语料"""

    def __init__(self, persist_path: Optional[str] = None):
        self._documents: List[Dict[str, Any]] = []   # [{content, metadata}, ...]
        self._tokenized_corpus: List[List[str]] = []  # tokenized content
        self._bm25: Optional[BM25Okapi] = None
        self.persist_path = persist_path

        if persist_path and os.path.exists(persist_path):
            self._load()

    # ── Public API ──────────────────────────────────────────────

    def add_documents(self, documents: List[Any]):
        """添加文档，文档可以是 langchain Document 或 dict"""
        for doc in documents:
            content = self._extract_content(doc)
            metadata = self._extract_metadata(doc)
            tokens = self._tokenize(content)
            self._documents.append({"content": content, "metadata": metadata})
            self._tokenized_corpus.append(tokens)

        self._rebuild()
        logger.info(f"[BM25] Added {len(documents)} docs, total: {len(self._documents)}")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        """返回 [(doc_dict, normalized_score), ...]，按 score 降序"""
        if not self._bm25 or not self._documents:
            return []

        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # 归一化到 [0, 1]
        max_score = max(scores) if len(scores) > 0 else 1.0
        indexed = [
            (self._documents[i], scores[i] / max_score if max_score > 0 else 0.0)
            for i in range(len(scores))
        ]
        indexed.sort(key=lambda x: x[1], reverse=True)
        return indexed[:top_k]

    def remove_by_doc_id(self, doc_id: int):
        """按 doc_id 删除文档"""
        before = len(self._documents)
        self._documents = [
            d for d in self._documents
            if d["metadata"].get("doc_id") != doc_id
        ]
        removed = before - len(self._documents)
        if removed > 0:
            self._tokenized_corpus = [self._tokenize(d["content"]) for d in self._documents]
            self._rebuild()
            logger.info(f"[BM25] Removed {removed} docs for doc_id={doc_id}, remaining: {len(self._documents)}")

    def save(self):
        """持久化到磁盘"""
        if not self.persist_path:
            return
        data = {
            "documents": self._documents,
            "tokenized_corpus": self._tokenized_corpus,
        }
        with open(self.persist_path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"[BM25] Saved {len(self._documents)} docs to {self.persist_path}")

    # ── Private ─────────────────────────────────────────────────

    def _tokenize(self, text: str) -> List[str]:
        return list(jieba.cut(text))

    def _rebuild(self):
        self._bm25 = BM25Okapi(self._tokenized_corpus) if self._tokenized_corpus else None
        if self.persist_path:
            self.save()

    @staticmethod
    def _extract_content(doc) -> str:
        if hasattr(doc, "page_content"):
            return doc.page_content
        if isinstance(doc, dict):
            return doc.get("content", doc.get("page_content", ""))
        return str(doc)

    @staticmethod
    def _extract_metadata(doc) -> Dict[str, Any]:
        if hasattr(doc, "metadata"):
            return doc.metadata
        if isinstance(doc, dict):
            return doc.get("metadata", {})
        return {}

    def _load(self):
        try:
            with open(self.persist_path, "rb") as f:
                data = pickle.load(f)
            self._documents = data.get("documents", [])
            self._tokenized_corpus = data.get("tokenized_corpus", [])
            self._rebuild()
            logger.info(f"[BM25] Loaded {len(self._documents)} docs from {self.persist_path}")
        except Exception as e:
            logger.warning(f"[BM25] Failed to load from {self.persist_path}: {e}")

    @property
    def document_count(self) -> int:
        return len(self._documents)
