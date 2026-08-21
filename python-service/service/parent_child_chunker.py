"""父子块双切分器

父块：语义切分（按话题边界自然分组，块内上下文连贯）
子块：递归切分（小粒度保证检索精准）
子块 embed 进 Milvus 做检索，父块存 MySQL 做 LLM 上下文。
"""

from typing import List, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)


class ParentChildChunker:
    """父子块切分器

    父块: 语义切分 → 存入 MySQL，检索时作为 LLM 上下文
    子块: 递归切分 → embed 存入 Milvus，做精准语义检索
    """

    def __init__(self, embeddings, child_chunk_size: int = 300,
                 child_chunk_overlap: int = 50,
                 semantic_buffer_size: int = 1,
                 semantic_breakpoint_type: str = "percentile",
                 semantic_breakpoint_amount: float = None):
        """embeddings: 用于语义切分父块（复用全局实例）"""
        from langchain_experimental.text_splitter import SemanticChunker

        self.semantic_splitter = SemanticChunker(
            embeddings,
            buffer_size=semantic_buffer_size,
            breakpoint_threshold_type=semantic_breakpoint_type,
            breakpoint_threshold_amount=semantic_breakpoint_amount,
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size, chunk_overlap=child_chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        )

    def split_documents(self, documents: List[Document]
                         ) -> Tuple[List[Document], List[Document]]:
        """切分文档为父块和子块

        Returns:
            (parent_docs, child_docs)
        """
        all_parents = []
        all_children = []
        doc_id = documents[0].metadata.get("doc_id", "unknown") if documents else "unknown"
        global_parent_idx = 0

        for doc in documents:
            # 语义切分父块
            try:
                parent_docs_raw = self.semantic_splitter.split_documents([doc])
            except Exception:
                # 兜底：机械切分
                parent_texts = self.child_splitter.split_text(doc.page_content)
                parent_docs_raw = [Document(page_content=t, metadata=doc.metadata)
                                   for t in parent_texts]

            for parent_doc in parent_docs_raw:
                parent_text = parent_doc.page_content.strip()
                if not parent_text:
                    continue

                parent_id = f"doc_{doc_id}_parent_{global_parent_idx}"
                global_parent_idx += 1

                parent_doc.metadata.update({
                    "parent_id": parent_id,
                    "parent_chunk_index": global_parent_idx - 1,
                })
                all_parents.append(parent_doc)

                # 子切分：机械细切后把过小的尾巴合并到前一块
                child_texts = self.child_splitter.split_text(parent_text)
                # 合并过小的尾部（< 100 chars 的最后一个子块合并到前一个）
                if len(child_texts) >= 2 and len(child_texts[-1]) < 100:
                    child_texts[-2] = child_texts[-2] + child_texts[-1]
                    child_texts.pop()
                for child_text in child_texts:
                    if not child_text.strip():
                        continue
                    child_doc = Document(
                        page_content=child_text,
                        metadata={
                            **doc.metadata,
                            "parent_id": parent_id,
                            "chunk_index": len(all_children),
                            "total_chunks": 0,
                        },
                    )
                    all_children.append(child_doc)

        total = len(all_children)
        for child in all_children:
            child.metadata["total_chunks"] = total

        logger.info(
            f"ParentChildChunker: {len(documents)} docs → "
            f"{len(all_parents)} parents + {len(all_children)} children"
        )
        return all_parents, all_children
