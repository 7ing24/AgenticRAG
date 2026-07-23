from typing import Dict, Any, Optional, Generator, List
from tools.question_rewrite import QuestionRewriteTool
from tools.knowledge_search import KnowledgeSearchTool
from tools.citation import CitationIntegrator, CitationTracker
from core.vector_store import vector_store
from core.config import config
import logging
import json

logger = logging.getLogger(__name__)


class RetrievalResult:
    """检索结果封装"""

    def __init__(
        self,
        original_query: str,
        rewritten_query: Optional[str] = None,
        retrieved_documents: Optional[List] = None,
        reranked_documents: Optional[List] = None,
        scores: Optional[List[float]] = None,
        citations: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error: Optional[str] = None
    ):
        self.original_query = original_query
        self.rewritten_query = rewritten_query or original_query
        self.retrieved_documents = retrieved_documents or []
        self.reranked_documents = reranked_documents or []
        self.scores = scores or []
        self.citations = citations or {}
        self.success = success
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "rewritten_query": self.rewritten_query,
            "retrieved_documents": self.retrieved_documents,
            "reranked_documents": self.reranked_documents,
            "scores": self.scores,
            "citations": self.citations,
            "success": self.success,
            "error": self.error
        }


class RetrievalAgent:
    """Retrieval Agent - 负责检索全流程：query rewrite -> recall -> rerank -> citation integration"""

    def __init__(self):
        self.question_rewrite_tool = QuestionRewriteTool()
        self.knowledge_search_tool = KnowledgeSearchTool()
        self.citation_integrator = CitationIntegrator()
        self.citation_tracker = CitationTracker()
        self.vector_store = vector_store

        self.config = {
            "top_k": 5,
            "use_rerank": False,   # 默认关闭重排序，减少开销
            "use_rewrite": False,  # 默认关闭问题重写，减少LLM调用
            "use_hybrid": True,    # 默认开启向量+BM25混合检索
            "max_citations": 5
        }

    def retrieve(
        self,
        query: str,
        conversation_context: str = "",
        use_rewrite: bool = False,
        use_rerank: bool = False,
        use_hybrid: bool = True,
        top_k: int = 5,
        **kwargs
    ) -> RetrievalResult:
        """
        执行完整检索流程

        Args:
            query: 用户问题
            conversation_context: 对话上下文
            use_rewrite: 是否使用问题改写
            use_rerank: 是否使用重排序
            use_hybrid: 是否使用向量+BM25混合检索
            top_k: 返回结果数量

        Returns:
            RetrievalResult: 检索结果
        """
        logger.info(f"[RetrievalAgent] Starting retrieval for: {query[:50]}...")

        original_query = query
        rewritten_query = query

        try:
            if use_rewrite:
                rewrite_result = self.question_rewrite_tool.execute({
                    "question": query,
                    "conversation_context": conversation_context
                })
                rewritten_query = rewrite_result.get("rewritten_question", query)
                logger.info(f"[RetrievalAgent] Query rewritten: '{query}' -> '{rewritten_query}'")

            retrieved_docs = self._retrieve_documents(
                rewritten_query,
                k=top_k,
                use_hybrid=use_hybrid,
                use_rerank=use_rerank,
            )

            reranked_docs = retrieved_docs
            scores = [
                doc.get("metadata", {}).get("score", 0.5) if isinstance(doc, dict)
                else getattr(doc, "metadata", {}).get("score", 0.5)
                for doc in retrieved_docs
            ]

            citation_result = self.citation_integrator.integrate(
                answer="",
                documents=reranked_docs,
                scores=scores,
                query=query
            )

            result = RetrievalResult(
                original_query=original_query,
                rewritten_query=rewritten_query,
                retrieved_documents=retrieved_docs,
                reranked_documents=reranked_docs,
                scores=scores,
                citations=citation_result,
                success=True
            )

            logger.info(
                f"[RetrievalAgent] Retrieval completed: "
                f"original='{original_query[:30]}...', "
                f"rewritten='{rewritten_query[:30]}...', "
                f"retrieved={len(retrieved_docs)}, "
                f"reranked={len(reranked_docs)}"
            )

            return result

        except Exception as e:
            logger.error(f"[RetrievalAgent] Retrieval failed: {e}")
            return RetrievalResult(
                original_query=original_query,
                success=False,
                error=str(e)
            )

    def retrieve_stream(
        self,
        query: str,
        conversation_context: str = "",
        use_rewrite: bool = False,
        use_rerank: bool = False,
        use_hybrid: bool = True,
        top_k: int = 5,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        流式执行检索流程

        Yields:
            JSON格式的事件流
        """
        logger.info(f"[RetrievalAgent] Stream retrieval for: {query[:50]}...")

        original_query = query
        rewritten_query = query

        try:
            yield json.dumps({
                "type": "retrieval_started",
                "query": original_query
            })

            if use_rewrite:
                yield json.dumps({
                    "type": "step",
                    "step_name": "question_rewrite",
                    "status": "started"
                })

                rewrite_result = self.question_rewrite_tool.execute({
                    "question": query,
                    "conversation_context": conversation_context
                })
                rewritten_query = rewrite_result.get("rewritten_question", query)

                yield json.dumps({
                    "type": "step",
                    "step_name": "question_rewrite",
                    "status": "completed",
                    "output": {
                        "original_query": original_query,
                        "rewritten_query": rewritten_query
                    }
                })

            yield json.dumps({
                "type": "step",
                "step_name": "knowledge_search",
                "status": "started"
            })

            retrieved_docs = self._retrieve_documents(
                rewritten_query,
                k=top_k,
                use_hybrid=use_hybrid,
                use_rerank=use_rerank,
            )

            reranked_docs = retrieved_docs
            scores = [
                doc.get("metadata", {}).get("score", 0.5) if isinstance(doc, dict)
                else getattr(doc, "metadata", {}).get("score", 0.5)
                for doc in retrieved_docs
            ]

            yield json.dumps({
                "type": "step",
                "step_name": "knowledge_search",
                "status": "completed",
                "output": {
                    "retrieved_count": len(retrieved_docs),
                    "top_scores": scores[:3] if scores else [],
                }
            })

            yield json.dumps({
                "type": "retrieval_completed",
                "output": {
                    "original_query": original_query,
                    "rewritten_query": rewritten_query,
                    "retrieved_count": len(retrieved_docs),
                    "final_count": len(reranked_docs),
                    "documents": reranked_docs,
                    "scores": scores
                }
            })

        except Exception as e:
            logger.error(f"[RetrievalAgent] Stream retrieval failed: {e}")
            yield json.dumps({
                "type": "error",
                "error": str(e)
            })

    def _retrieve_documents(
        self,
        query: str,
        k: int = 10,
        use_hybrid: bool = True,
        use_rerank: bool = False,
    ) -> List:
        """执行文档检索"""
        try:
            if self.knowledge_search_tool.vector_store:
                result = self.knowledge_search_tool.execute({
                    "query": query,
                    "top_k": k,
                    "use_rerank": use_rerank,
                    "use_hybrid": use_hybrid
                })
                return result.get("documents", [])
        except Exception as e:
            logger.warning(f"[RetrievalAgent] knowledge_search tool failed: {e}")

        docs = self.vector_store.search(
            query=query,
            k=k,
            use_rerank=use_rerank,
            hybrid=use_hybrid
        )

        return [
            {
                "content": getattr(doc, "page_content", str(doc)),
                "metadata": getattr(doc, "metadata", {}),
                "score": getattr(doc, "score", 0.5)
            }
            for doc in docs
        ]


    def integrate_citations(
        self,
        answer: str,
        documents: List[Dict[str, Any]],
        scores: Optional[List[float]] = None,
        query: str = ""
    ) -> Dict[str, Any]:
        """整合引用到答案"""
        return self.citation_integrator.integrate(
            answer=answer,
            documents=documents,
            scores=scores,
            query=query
        )

    def track_retrieval(
        self,
        query: str,
        rewritten_query: str,
        retrieved_docs: List[Dict[str, Any]],
        reranked_docs: List[Dict[str, Any]],
        final_answer: str
    ) -> Dict[str, Any]:
        """追踪检索链路"""
        return self.citation_tracker.track(
            query=query,
            rewritten_query=rewritten_query,
            retrieved_docs=retrieved_docs,
            reranked_docs=reranked_docs,
            final_answer=final_answer
        )

    def get_stats(self) -> Dict[str, Any]:
        """获取检索统计信息"""
        return {
            "config": self.config,
            "citation_stats": self.citation_tracker.get_citation_stats()
        }


retrieval_agent = RetrievalAgent()