"""真正的 embedding-based 语义切分器

包装 LangChain 的 SemanticChunker，通过计算相邻句子的 embedding 距离
来检测语义边界（话题转换点），在此处切分。
"""

import logging
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import TextSplitter

logger = logging.getLogger(__name__)


class EmbeddingSemanticChunker(TextSplitter):
    """基于 embedding 的语义切分器

    工作流程：
    1. 将文本拆分为句子
    2. 计算每个句子的 embedding 向量
    3. 计算相邻句子之间的余弦距离
    4. 在距离超过断点阈值的位置切分
    """

    def __init__(
        self,
        embeddings: Embeddings,
        chunk_size: int = 500,
        chunk_overlap: int = 0,
        min_chunk_size: int = 100,
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: Optional[float] = None,
        buffer_size: int = 1,
    ):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.min_chunk_size = min_chunk_size
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount
        self.buffer_size = buffer_size

        self._semantic_chunker = SemanticChunker(
            embeddings=embeddings,
            buffer_size=buffer_size,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
            min_chunk_size=min_chunk_size,
        )

        logger.info(
            f"EmbeddingSemanticChunker initialized: "
            f"breakpoint_type={breakpoint_threshold_type}, "
            f"breakpoint_amount={breakpoint_threshold_amount}, "
            f"min_chunk_size={min_chunk_size}"
        )

    def split_text(self, text: str) -> List[str]:
        """使用 embedding 距离检测语义边界并切分"""
        if not text or not text.strip():
            return []
        return self._semantic_chunker.split_text(text)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档列表，保留原始 metadata 并使用全局索引"""
        result = []
        global_index = 0

        # 先全部切分，缓存结果，避免重复调用 embedding API
        doc_chunks: List[tuple] = []
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            doc_chunks.append((doc, chunks))

        total_chunks_all = sum(len(chunks) for _, chunks in doc_chunks)

        for doc, chunks in doc_chunks:
            for chunk in chunks:
                result.append(Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_index": global_index,
                        "total_chunks": total_chunks_all,
                    }
                ))
                global_index += 1

        logger.info(f"EmbeddingSemanticChunker produced {len(result)} chunks from {len(documents)} documents")
        return result
