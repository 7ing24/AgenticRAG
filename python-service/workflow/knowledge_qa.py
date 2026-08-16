from typing import Dict, Any, Optional, Generator, List
from types import SimpleNamespace
from engine.orchestrator import Orchestrator
from engine.planner import Planner
from agent.memory_agent import MemoryAgent
from engine.state import AgentState
from engine.events import EventBus, event_bus
from agent.react_agent import ReActAgent, react_agent
from service.retrieval import RetrievalAgent
from core.vector_store import vector_store
from core.llm import LLMService
from tools.registry import tool_registry
from tools.question_rewrite import QuestionRewriteTool
from engine.trace_collector import TraceCollector
import logging
import json

logger = logging.getLogger(__name__)


class KnowledgeQAAgent:
    """知识问答Agent - 三级链路：L1简化/L2标准/L3推理"""

    def __init__(self):
        self.orchestrator = Orchestrator()
        self.event_bus = event_bus
        self.retrieval_agent = RetrievalAgent()
        self.vector_store = vector_store
        self.llm_service = LLMService()
        self.planner = Planner()
        self.question_rewrite_tool = QuestionRewriteTool()
        self.memory_agent = MemoryAgent()
        self.max_retrieval_rounds = 3
        self._router = None
        self._reasoning_agent = None
        self._react_agent = None
        self._multi_agent_orchestrator = None
        self._agents_registered = False

    @property
    def router(self):
        """延迟加载 RouterAgent（避免循环导入）"""
        if self._router is None:
            from agent.router_agent import RouterAgent
            self._router = RouterAgent()
        return self._router

    @property
    def reasoning_agent(self):
        """延迟加载 ReasoningAgent"""
        if self._reasoning_agent is None:
            from workflow.reasoning import ReasoningAgent
            self._reasoning_agent = ReasoningAgent()
        return self._reasoning_agent

    @property
    def react_agent(self):
        """延迟加载 ReActAgent"""
        if self._react_agent is None:
            self._react_agent = ReActAgent(max_iterations=10)
        return self._react_agent

    @property
    def multi_agent_orchestrator(self):
        """延迟加载 MultiAgentOrchestrator"""
        if self._multi_agent_orchestrator is None:
            from engine.agent_orchestrator import MultiAgentOrchestrator
            from engine.agent_registry import agent_registry
            self._multi_agent_orchestrator = MultiAgentOrchestrator()
            # 首次加载时注册 worker agents
            if not self._agents_registered:
                agent_registry.register(
                    "ReActAgent", self.react_agent,
                    capabilities=["knowledge_search", "reasoning", "multi_step"],
                    description="ReAct循环Agent，支持动态工具调用的多步推理"
                )
                agent_registry.register(
                    "RetrievalAgent", self.retrieval_agent,
                    capabilities=["knowledge_search", "retrieval"],
                    description="检索Agent，向量+BM25混合检索+重排序"
                )
                self._agents_registered = True
        return self._multi_agent_orchestrator

    def ask(self, question: str, conversation_id: Optional[str] = None,
            user_id: Optional[str] = None,
            trace_id: str = "",
            **kwargs) -> Dict[str, Any]:
        """
        处理知识问答 - 两条链路

        simple：快速链路（检索+充分性自省+生成）
        complex：ReAct 链路（LLM 动态工具调用）
        """
        logger.info(f"[KnowledgeQAAgent] Processing question: {question[:50]}...")

        try:
            # 1. 统一通过 MemoryAgent 加载记忆（对话历史 + 用户画像）
            memory_state = SimpleNamespace(
                conversation_id=conversation_id, user_id=user_id,
                original_input=question, run_id="memory_load"
            )
            memory_result = self.memory_agent.load_memory(memory_state)
            conversation_history = memory_result.get("text", "")
            context_token_count = memory_result.get("token_count", 0)

            # 2. 判断复杂度，选择链路（LLM 优先，关键词 fallback）
            complexity = self._classify_complexity_with_llm(question)
            if complexity is None:
                # fallback：用 classifier 的三分类，react → complex，其余 → simple
                raw = self.router.classifier.classify_complexity(question)
                complexity = "complex" if raw == "react" else "simple"
                logger.info(f"[KnowledgeQAAgent] Complexity (keyword): {complexity}")
            else:
                logger.info(f"[KnowledgeQAAgent] Complexity (LLM): {complexity}")

            if complexity == "complex":
                return self._ask_with_multi_agent(question, conversation_id, user_id, conversation_history,
                                                  trace_id=trace_id, **kwargs)
            else:
                return self._ask_fast(question, conversation_id, user_id, conversation_history,
                                     trace_id=trace_id, **kwargs)

        except Exception as e:
            logger.error(f"[KnowledgeQAAgent] QA failed: {e}", exc_info=True)
            return {
                "answer": "抱歉，处理您的问题时遇到了错误，请稍后再试。",
                "sources": [],
                "has_sources": False,
                "task_type": "knowledge_qa",
                "error": True,
                "trace_id": trace_id,
                "runs": [],
            }

    def _ask_fast(self, question: str, conversation_id: Optional[str],
                  user_id: Optional[str], conversation_history: str,
                  trace_id: str = "", **kwargs) -> Dict[str, Any]:
        """快速链路：改写 + 检索 + rerank + 自省循环（合并原 L1/L2）"""
        import uuid
        run_id = str(uuid.uuid4())
        best_docs = []
        best_scores = []
        steps = []
        collector = kwargs.get("trace_collector")  # type: Optional[TraceCollector]

        # 首轮改写（指代消解 + 关键词提取）
        if collector:
            collector.start_timer("rewrite")
        try:
            rewritten = self.question_rewrite_tool.execute({
                "question": question,
                "conversation_context": conversation_history
            })
            current_query = rewritten.get("rewritten_question", question)
            logger.info(f"[KnowledgeQAAgent] Fast rewritten: '{question[:30]}...' -> '{current_query[:50]}...'")
        except Exception:
            current_query = question
        if collector:
            rewrite_latency, rewrite_start = collector.stop_timer("rewrite")
            collector.record_event(
                event_type="QUESTION_REWRITTEN",
                phase="REWRITE",
                input_data={"original": question},
                output_data={"rewritten": current_query},
                agent_name="QuestionRewriteTool",
                latency_ms=rewrite_latency,
                event_time=rewrite_start,
            )

        for round_num in range(self.max_retrieval_rounds):
            if collector:
                collector.start_timer(f"retrieval_r{round_num}")
            retrieval_result = self.retrieval_agent.retrieve(
                query=current_query,
                conversation_context=conversation_history,
                use_rewrite=False,
                use_rerank=True,
                top_k=5,
            )

            docs = retrieval_result.reranked_documents
            scores = retrieval_result.scores
            ret_result = collector.stop_timer(f"retrieval_r{round_num}") if collector else (0, "")
            ret_latency, ret_start = ret_result if isinstance(ret_result, tuple) else (ret_result, "")
            logger.info(f"[KnowledgeQAAgent] Fast round {round_num + 1}: "
                        f"retrieved {len(docs)} docs")

            if collector:
                collector.record_event(
                    event_type="RETRIEVAL_EXECUTED",
                    phase="RETRIEVAL",
                    input_data={"query": current_query, "round": round_num + 1},
                    output_data={
                        "chunks": [
                            {
                                "doc_id": self._extract_metadata(doc).get("doc_id"),
                                "parent_id": self._extract_metadata(doc).get("parent_id",
                                    self._extract_metadata(doc).get("chunk_index") or self._extract_metadata(doc).get("parent_chunk_index", "-")),
                                "page": self._extract_metadata(doc).get("page"),
                                "score": round(score, 3),
                            }
                            for doc, score in zip(docs, scores)
                        ] if docs else [],
                    },
                    agent_name="RetrievalAgent",
                    latency_ms=ret_latency,
                    event_time=ret_start,
                    metadata={"mode": "hybrid", "vector_store": "FAISS" if not self.vector_store.use_milvus else "Milvus",
                              "use_rerank": True, "top_k": 5},
                )

            sufficient = docs and self.planner.evaluate_retrieval_sufficiency(
                docs, question, scores, score_threshold=0.2
            ).is_sufficient
            steps.append({
                "step_name": f"knowledge_search_r{round_num + 1}",
                "step_type": "knowledge_search",
                "status": "completed",
                "is_sufficient": sufficient,
                "chunks": [
                    {
                        "doc_id": self._extract_metadata(doc).get("doc_id"),
                        "parent_id": self._extract_metadata(doc).get("parent_id",
                            self._extract_metadata(doc).get("chunk_index") or self._extract_metadata(doc).get("parent_chunk_index", "-")),
                        "page": self._extract_metadata(doc).get("page"),
                        "score": round(score, 3),
                    }
                    for doc, score in zip(docs, scores)
                ] if docs else [],
            })

            if sufficient:
                logger.info(f"[KnowledgeQAAgent] Fast sufficient at round {round_num + 1}")
                best_docs = docs
                best_scores = scores
                break

            if len(docs) > len(best_docs):
                best_docs = docs
                best_scores = scores

            if round_num < self.max_retrieval_rounds - 1:
                current_query = self._rewrite_query_for_retry(
                    current_query, conversation_history, round_num + 1
                )
                logger.info(f"[KnowledgeQAAgent] Fast retry with: {current_query[:50]}...")
        else:
            logger.info(f"[KnowledgeQAAgent] Fast max rounds reached, using best ({len(best_docs)} docs)")

        if not best_docs:
            if conversation_history:
                answer = self.llm_service.get_answer(question, [], conversation_history)
            else:
                answer = "抱歉，知识库中没有找到与您问题相关的内容。"
            steps.append({"step_name": "answer_generation", "step_type": "answer_generation",
                          "status": "completed"})
            if collector:
                token_usage = self.llm_service.get_last_token_usage()
                collector.record_event(
                    event_type="ANSWER_GENERATED",
                    phase="GENERATION",
                    input_data={"question": question},
                    output_data={"answer": answer, "chunk_count": 0},
                    agent_name="KnowledgeQAAgent",
                    model_name="qwen-plus",
                    input_tokens=token_usage.get("input_tokens") if token_usage else None,
                    output_tokens=token_usage.get("output_tokens") if token_usage else None,
                    total_tokens=token_usage.get("total_tokens") if token_usage else None,
                )
            self._save_to_memory(conversation_id, question, answer, user_id)
            return {"answer": answer, "sources": [], "has_sources": False,
                    "task_type": "knowledge_qa", "steps": steps,
                    "trace_id": trace_id, "run_id": run_id,
                    "runs": [{"run_id": run_id, "parent_run_id": None,
                              "agent_type": "knowledge_qa", "steps": steps}]}

        if collector:
            collector.start_timer("generation")
        llm_docs = self._docs_to_llm_format(best_docs)
        answer = self.llm_service.get_answer(question, llm_docs, conversation_history)
        gen_latency, gen_start = collector.stop_timer("generation") if collector else (0, "")
        steps.append({"step_name": "answer_generation", "step_type": "answer_generation",
                      "status": "completed"})
        if collector:
            token_usage = self.llm_service.get_last_token_usage()
            collector.record_event(
                event_type="ANSWER_GENERATED",
                phase="GENERATION",
                input_data={"question": question, "chunk_count": len(best_docs)},
                output_data={"answer": answer},
                agent_name="KnowledgeQAAgent",
                model_name="qwen-plus",
                latency_ms=gen_latency,
                event_time=gen_start,
                input_tokens=token_usage.get("input_tokens") if token_usage else None,
                output_tokens=token_usage.get("output_tokens") if token_usage else None,
                total_tokens=token_usage.get("total_tokens") if token_usage else None,
            )
        self._save_to_memory(conversation_id, question, answer, user_id)
        sources = self._build_sources(best_docs)

        return {
            "answer": answer, "sources": sources,
            "has_sources": len(sources) > 0, "task_type": "knowledge_qa",
            "steps": steps,
            "trace_id": trace_id, "run_id": run_id,
            "runs": [{"run_id": run_id, "parent_run_id": None,
                      "agent_type": "knowledge_qa", "steps": steps}]
        }

    def _ask_with_react(self, question: str, conversation_id: Optional[str] = None,
                        user_id: Optional[str] = None, conversation_history: str = "",
                        context_token_count: int = 0,
                        **kwargs) -> Dict[str, Any]:
        """ReAct 链路（降级回退）：LLM 动态决定工具调用，自主判断何时输出答案"""
        logger.info(f"[KnowledgeQAAgent] Using ReAct agent (fallback)...")
        collector = kwargs.get("trace_collector")
        if collector:
            collector.start_timer("react")
        result = self.react_agent.run(
            question=question,
            conversation_id=conversation_id,
            user_id=user_id,
            conversation_history=conversation_history,
            context_token_count=context_token_count,
            **kwargs
        )
        if collector:
            latency, start_time = collector.stop_timer("react")
            collector.record_event(
                event_type="REACT_EXECUTED",
                phase="REACT",
                input_data={"question": question, "context_tokens": context_token_count},
                output_data={"answer": result.get("answer", "")},
                agent_name="ReActAgent",
                model_name="qwen-plus",
                latency_ms=latency,
                event_time=start_time,
            )
        return result

    def _ask_with_multi_agent(self, question: str, conversation_id: Optional[str] = None,
                              user_id: Optional[str] = None, conversation_history: str = "",
                              trace_id: str = "",
                              **kwargs) -> Dict[str, Any]:
        """Multi-Agent 链路：Orchestrator 拆解 → 多 worker 并行 → 汇总"""
        logger.info(f"[KnowledgeQAAgent] Using Multi-Agent orchestrator...")
        # 内部细节由 MultiAgentOrchestrator 自己记录（PLAN / WORKERS / SYNTHESIS / REACT_ITERATION）
        return self.multi_agent_orchestrator.run(
            question=question,
            conversation_id=conversation_id,
            user_id=user_id,
            conversation_history=conversation_history,
            trace_id=trace_id,
            **kwargs
        )

    # ── 以下为旧链路方法，保留供未来使用 ──

    def _ask_with_reasoning(self, question: str, conversation_id: Optional[str] = None,
                            user_id: Optional[str] = None, context: str = "",
                            **kwargs) -> Dict[str, Any]:
        """Reasoning 链路（预留，当前路由不调用）"""
        logger.info(f"[KnowledgeQAAgent] Using Reasoning agent...")
        return self.reasoning_agent.reason(
            question=question,
            context=context,
            conversation_id=conversation_id,
            user_id=user_id
        )

    # ── 以下为旧链路方法，保留供未来使用，当前路由不再指向它们 ──

    def _ask_l1(self, question: str, conversation_id: Optional[str],
                conversation_history: str) -> Dict[str, Any]:
        """L1 简化链路：检索 + 自省循环（不充分时改写重试）"""
        best_docs = []
        best_scores = []
        current_query = question

        for round_num in range(self.max_retrieval_rounds):
            docs = self.vector_store.search(
                query=current_query, k=5, similarity_threshold=0.5, use_rerank=False
            )
            scores = [doc.metadata.get('score', 0.5) if hasattr(doc, 'metadata') else 0.5 for doc in docs] if docs else []
            logger.info(f"[KnowledgeQAAgent] L1 round {round_num + 1}: "
                        f"retrieved {len(docs)} docs")

            if docs and self.planner.evaluate_retrieval_sufficiency(
                docs, question, scores, score_threshold=0.5
            ).is_sufficient:
                logger.info(f"[KnowledgeQAAgent] L1 sufficient at round {round_num + 1}")
                best_docs = docs
                best_scores = scores
                break

            if len(docs) > len(best_docs):
                best_docs = docs
                best_scores = scores

            if round_num < self.max_retrieval_rounds - 1:
                current_query = self._rewrite_query_for_retry(
                    current_query, conversation_history, round_num + 1
                )
                logger.info(f"[KnowledgeQAAgent] L1 retry with: {current_query[:50]}...")
        else:
            logger.info(f"[KnowledgeQAAgent] L1 max rounds reached, using best ({len(best_docs)} docs)")

        if not best_docs:
            if conversation_history:
                answer = self.llm_service.get_answer(question, [], conversation_history)
            else:
                answer = "抱歉，知识库中没有找到与您问题相关的内容。"
            self._save_to_memory(conversation_id, question, answer)
            return {"answer": answer, "sources": [], "has_sources": False, "task_type": "knowledge_qa"}

        answer = self.llm_service.get_answer(question, best_docs, conversation_history)
        self._save_to_memory(conversation_id, question, answer)
        sources = self._build_sources(best_docs)

        return {
            "answer": answer, "sources": sources,
            "has_sources": len(sources) > 0, "task_type": "knowledge_qa"
        }

    def _ask_l2(self, question: str, conversation_id: Optional[str],
                conversation_history: str) -> Dict[str, Any]:
        """L2 标准链路：问题改写/指代消解 + 检索 + 重排序 + 生成 + 自省循环"""
        best_docs = []
        best_scores = []
        best_citations = {}

        # L2 每次都先用重写工具改写，有对话历史时自然消解指代
        rewritten = self.question_rewrite_tool.execute({
            "question": question,
            "conversation_context": conversation_history
        })
        current_query = rewritten.get("rewritten_question", question)
        logger.info(f"[KnowledgeQAAgent] L2 rewritten: '{question[:30]}...' -> '{current_query[:50]}...'")

        for round_num in range(self.max_retrieval_rounds):
            retrieval_result = self.retrieval_agent.retrieve(
                query=current_query,
                conversation_context=conversation_history,
                use_rewrite=False,    # 改写由外层自省循环统一控制
                use_rerank=True,
                top_k=5,
                similarity_threshold=0.5
            )

            docs = retrieval_result.reranked_documents
            scores = retrieval_result.scores
            logger.info(f"[KnowledgeQAAgent] L2 round {round_num + 1}: "
                        f"retrieved {len(docs)} docs")

            if docs and self.planner.evaluate_retrieval_sufficiency(
                docs, question, scores, score_threshold=0.3
            ).is_sufficient:
                logger.info(f"[KnowledgeQAAgent] L2 sufficient at round {round_num + 1}")
                best_docs = docs
                best_scores = scores
                best_citations = retrieval_result.citations
                break

            if len(docs) > len(best_docs):
                best_docs = docs
                best_scores = scores
                best_citations = retrieval_result.citations

            if round_num < self.max_retrieval_rounds - 1:
                current_query = self._rewrite_query_for_retry(
                    current_query, conversation_history, round_num + 1
                )
                logger.info(f"[KnowledgeQAAgent] L2 retry with: {current_query[:50]}...")
        else:
            logger.info(f"[KnowledgeQAAgent] L2 max rounds reached, using best ({len(best_docs)} docs)")

        if not best_docs:
            answer = "抱歉，知识库中没有找到与您问题相关的内容。"
            self._save_to_memory(conversation_id, question, answer)
            return {"answer": answer, "sources": [], "has_sources": False, "task_type": "knowledge_qa"}

        llm_docs = self._docs_to_llm_format(best_docs)
        answer = self.llm_service.get_answer(question, llm_docs, conversation_history)
        self._save_to_memory(conversation_id, question, answer)

        citation_sources = best_citations.get("sources", []) if best_citations else []
        sources = citation_sources if citation_sources else self._build_sources(best_docs)

        return {
            "answer": answer, "sources": sources,
            "has_sources": len(sources) > 0, "task_type": "knowledge_qa"
        }

    def _classify_complexity_with_llm(self, question: str) -> Optional[str]:
        """LLM 判断问题复杂度，决定走 ReAct 还是快速链路

        Returns:
            "complex" / "simple" / None（解析失败时返回 None，由关键词兜底）
        """
        try:
            prompt = (
                "判断以下问题是否为复杂问题。\n\n"
                f"问题：{question}\n\n"
                "【复杂问题 complex】满足以下任一条件：\n"
                "- 涉及多个事物的对比/比较/区别/异同（如A和B的区别）\n"
                "- 要求分析原因、评估优劣、归纳总结\n"
                "- 需要分别介绍多个独立概念后再汇总\n"
                "- 问题包含多个子问题（如分别说明X、Y、Z）\n\n"
                "【简单问题 simple】必须同时满足：\n"
                "- 只涉及单一概念或事物的定义/解释\n"
                "- 不需要对比、不需要分析、不需要归纳\n"
                "- 一次关键词搜索就能直接找到答案\n\n"
                "只回复 complex 或 simple，不要解释。"
            )
            result = self.llm_service.generate(prompt)
            if result:
                result = result.strip().lower()
                if "complex" in result:
                    return "complex"
                if "simple" in result:
                    return "simple"
        except Exception as e:
            logger.warning(f"[KnowledgeQAAgent] LLM complexity classification failed: {e}")
        return None

    def _rewrite_query_for_retry(self, query: str, context: str, retry_round: int) -> str:
        """统一走 QuestionRewriteTool，通过 retry_round 切换策略"""
        try:
            result = self.question_rewrite_tool.execute({
                "question": query,
                "conversation_context": context,
                "retry_round": retry_round,
            })
            return result.get("rewritten_question", query)
        except Exception as e:
            logger.warning(f"[KnowledgeQAAgent] Query rewrite failed: {e}")
            return query

    def _docs_to_llm_format(self, docs: list) -> list:
        """将检索结果转为 LLM 可用的文档格式"""
        llm_docs = []
        for doc in docs:
            if hasattr(doc, 'page_content'):
                llm_docs.append(doc)
            elif isinstance(doc, dict):
                from langchain_core.documents import Document
                llm_docs.append(Document(
                    page_content=doc.get('content', ''),
                    metadata=doc.get('metadata', {})
                ))
        return llm_docs

    @staticmethod
    def _extract_metadata(doc) -> dict:
        """从 Document 或 dict 中提取 metadata"""
        if hasattr(doc, 'metadata'):
            return getattr(doc, 'metadata', {}) or {}
        if isinstance(doc, dict):
            return doc.get('metadata', {}) or {}
        return {}

    def _build_sources(self, docs: list) -> list:
        """构建引用来源（按 doc_id 去重）"""
        seen_doc_ids = set()
        sources = []
        for doc in docs:
            metadata = self._extract_metadata(doc)
            doc_id = metadata.get("doc_id")
            if doc_id and doc_id in seen_doc_ids:
                continue
            if doc_id:
                seen_doc_ids.add(doc_id)
            sources.append({
                "doc_id": doc_id,
                "doc": metadata.get("source", "未知文档"),
                "page": metadata.get("page"),
                "chunk_index": metadata.get("chunk_index"),
                "score": metadata.get("score", 0)
            })
        return sources

    def _save_to_memory(self, conversation_id: str, question: str, answer: str,
                        user_id: Optional[str] = None):
        """保存对话记忆（统一委托给 MemoryAgent）"""
        if not conversation_id:
            return
        memory_state = SimpleNamespace(
            conversation_id=conversation_id, user_id=user_id, run_id="memory_save"
        )
        self.memory_agent.save_memory(memory_state, question, answer)

    def ask_stream(self, question: str, conversation_id: Optional[str] = None,
                   user_id: Optional[str] = None, context: str = "",
                   **kwargs) -> Generator[str, None, None]:
        """
        流式处理知识问答 — 与非流式完全相同的逻辑链路。

        simple：改写 → 检索+Rerank+自省 → **流式 LLM 生成**（真 token 级流式）
        complex：_ask_with_multi_agent() → 逐字 SSE 输出
        """
        collector = kwargs.get("trace_collector")
        trace_id = kwargs.get("trace_id", "")
        logger.info(f"[KnowledgeQAAgent] Stream processing question: {question[:50]}...")

        try:
            # 1. 通过 MemoryAgent 加载记忆（与非流式完全相同）
            memory_state = SimpleNamespace(
                conversation_id=conversation_id, user_id=user_id,
                original_input=question, run_id="memory_load"
            )
            memory_result = self.memory_agent.load_memory(memory_state)
            conversation_history = memory_result.get("text", "")

            # 2. 判断复杂度，选择链路（与非流式完全相同）
            complexity = self._classify_complexity_with_llm(question)
            if complexity is None:
                raw = self.router.classifier.classify_complexity(question)
                complexity = "complex" if raw == "react" else "simple"
                logger.info(f"[KnowledgeQAAgent] Stream complexity (keyword): {complexity}")
            else:
                logger.info(f"[KnowledgeQAAgent] Stream complexity (LLM): {complexity}")

            # 3. 复杂问题：MultiAgentOrchestrator 流式（PLAN→WORKERS静默，SYNTHESIS真流式）
            if complexity == "complex":
                logger.info(f"[KnowledgeQAAgent] Stream routing to MultiAgentOrchestrator.run_stream")
                for event in self.multi_agent_orchestrator.run_stream(
                    question, conversation_id=conversation_id,
                    user_id=user_id, conversation_history=conversation_history,
                    **kwargs
                ):
                    yield event
                return

            # 4. 简单问题：完全复用 _ask_fast 的检索逻辑 → 真流式 LLM 生成
            logger.info(f"[KnowledgeQAAgent] Stream routing to _ask_fast pipeline + streaming generation")
            import uuid as _uuid
            run_id = str(_uuid.uuid4())
            best_docs = []
            best_scores = []

            # ── 4a. 改写 ──
            if collector:
                collector.start_timer("rewrite")
            try:
                rewritten = self.question_rewrite_tool.execute({
                    "question": question,
                    "conversation_context": conversation_history
                })
                current_query = rewritten.get("rewritten_question", question)
                logger.info(f"[KnowledgeQAAgent] Stream rewritten: '{question[:30]}...' -> '{current_query[:50]}...'")
            except Exception:
                current_query = question
            if collector:
                rewrite_latency, rewrite_start = collector.stop_timer("rewrite")
                collector.record_event(
                    event_type="QUESTION_REWRITTEN",
                    phase="REWRITE",
                    input_data={"original": question},
                    output_data={"rewritten": current_query},
                    agent_name="QuestionRewriteTool",
                    latency_ms=rewrite_latency,
                    event_time=rewrite_start,
                )

            # ── 4b. 检索 + Rerank + 自省循环（与 _ask_fast 完全一致）──
            for round_num in range(self.max_retrieval_rounds):
                if collector:
                    collector.start_timer(f"retrieval_r{round_num}")
                retrieval_result = self.retrieval_agent.retrieve(
                    query=current_query,
                    conversation_context=conversation_history,
                    use_rewrite=False,
                    use_rerank=True,
                    top_k=5,
                )
                docs = retrieval_result.reranked_documents
                scores = retrieval_result.scores
                ret_result = collector.stop_timer(f"retrieval_r{round_num}") if collector else (0, "")
                ret_latency, ret_start = ret_result if isinstance(ret_result, tuple) else (ret_result, "")
                logger.info(f"[KnowledgeQAAgent] Stream round {round_num + 1}: "
                            f"retrieved {len(docs)} docs")

                if collector:
                    collector.record_event(
                        event_type="RETRIEVAL_EXECUTED",
                        phase="RETRIEVAL",
                        input_data={"query": current_query, "round": round_num + 1},
                        output_data={
                            "chunks": [
                                {
                                    "doc_id": self._extract_metadata(doc).get("doc_id"),
                                    "parent_id": self._extract_metadata(doc).get("parent_id",
                                        self._extract_metadata(doc).get("chunk_index") or self._extract_metadata(doc).get("parent_chunk_index", "-")),
                                    "page": self._extract_metadata(doc).get("page"),
                                    "score": round(score, 3),
                                }
                                for doc, score in zip(docs, scores)
                            ] if docs else [],
                        },
                        agent_name="RetrievalAgent",
                        latency_ms=ret_latency,
                        event_time=ret_start,
                        metadata={"mode": "hybrid",
                                  "vector_store": "FAISS" if not self.vector_store.use_milvus else "Milvus",
                                  "use_rerank": True, "top_k": 5},
                    )

                sufficient = docs and self.planner.evaluate_retrieval_sufficiency(
                    docs, question, scores, score_threshold=0.2
                ).is_sufficient

                if sufficient:
                    logger.info(f"[KnowledgeQAAgent] Stream sufficient at round {round_num + 1}")
                    best_docs = docs
                    best_scores = scores
                    break

                if len(docs) > len(best_docs):
                    best_docs = docs
                    best_scores = scores

                if round_num < self.max_retrieval_rounds - 1:
                    current_query = self._rewrite_query_for_retry(
                        current_query, conversation_history, round_num + 1
                    )
                    logger.info(f"[KnowledgeQAAgent] Stream retry with: {current_query[:50]}...")
            else:
                logger.info(f"[KnowledgeQAAgent] Stream max rounds reached, using best ({len(best_docs)} docs)")

            sources = self._build_sources(best_docs) if best_docs else []

            # ── 4c. 无文档时：直接返回 ──
            if not best_docs:
                if conversation_history:
                    answer = self.llm_service.get_answer(question, [], conversation_history)
                else:
                    answer = "抱歉，知识库中没有找到与您问题相关的内容。"
                if collector:
                    collector.record_event(
                        event_type="ANSWER_GENERATED",
                        phase="GENERATION",
                        input_data={"question": question},
                        output_data={"answer": answer, "chunk_count": 0},
                        agent_name="KnowledgeQAAgent",
                        model_name="qwen-plus",
                    )
                self._save_to_memory(conversation_id, question, answer, user_id)
                yield json.dumps({"type": "start"})
                for char in answer:
                    yield json.dumps({"type": "token", "content": char})
                yield json.dumps({"type": "end", "content": answer, "task_type": "knowledge_qa"})
                if sources:
                    yield json.dumps({"type": "sources", "content": sources, "task_type": "knowledge_qa"})
                return

            # ── 4d. 真流式 LLM 生成（唯一与非流式不同的地方）──
            if collector:
                collector.start_timer("generation")
            llm_docs = self._docs_to_llm_format(best_docs)
            full_answer = ""
            for chunk in self.llm_service.get_answer_stream(
                question=question,
                context_docs=llm_docs,
                conversation_context=conversation_history
            ):
                try:
                    event = json.loads(chunk)
                    if event.get("type") == "token":
                        full_answer += event.get("content", "")
                    elif event.get("type") == "end":
                        full_answer = event.get("content", full_answer)
                except Exception:
                    pass
                yield chunk

            if collector:
                gen_latency, gen_start = collector.stop_timer("generation")
                token_usage = self.llm_service.get_last_token_usage()
                collector.record_event(
                    event_type="ANSWER_GENERATED",
                    phase="GENERATION",
                    input_data={"question": question, "chunk_count": len(best_docs)},
                    output_data={"answer": full_answer},
                    agent_name="KnowledgeQAAgent",
                    model_name="qwen-plus",
                    latency_ms=gen_latency,
                    event_time=gen_start,
                    input_tokens=token_usage.get("input_tokens") if token_usage else None,
                    output_tokens=token_usage.get("output_tokens") if token_usage else None,
                    total_tokens=token_usage.get("total_tokens") if token_usage else None,
                )

            self._save_to_memory(conversation_id, question, full_answer, user_id)

            # 发送来源信息
            if sources:
                yield json.dumps({"type": "sources", "content": sources, "task_type": "knowledge_qa"})

        except Exception as e:
            logger.error(f"[KnowledgeQAAgent] Stream QA failed: {e}", exc_info=True)
            yield json.dumps({
                "type": "error",
                "content": "处理问题时遇到错误，请稍后再试。"
            })

    def _ask_with_orchestrator(self, question: str, conversation_id: Optional[str] = None,
                               user_id: Optional[str] = None, context: str = "",
                               **kwargs) -> Dict[str, Any]:
        """
        使用原有的Orchestrator方式进行知识问答（回退方案）
        """
        logger.info(f"[KnowledgeQAAgent] Using orchestrator fallback...")
        result = self.orchestrator.run(
            input_text=question,
            conversation_id=conversation_id,
            user_id=user_id,
            context=context,
            goal=f"回答知识问题: {question[:50]}...",
            **kwargs
        )
        return result

    def _ask_stream_with_orchestrator(self, question: str, conversation_id: Optional[str] = None,
                                      user_id: Optional[str] = None, context: str = "",
                                      **kwargs) -> Generator[str, None, None]:
        """
        使用原有的Orchestrator方式进行流式知识问答（回退方案）
        """
        logger.info(f"[KnowledgeQAAgent] Using orchestrator stream fallback...")
        for event in self.orchestrator.run_stream(
            input_text=question,
            conversation_id=conversation_id,
            user_id=user_id,
            context=context,
            goal=f"流式回答知识问题: {question[:50]}...",
            **kwargs
        ):
            yield event

    def register_callback(self, event_type: str, callback):
        """注册事件回调"""
        self.event_bus.subscribe(event_type, callback)

    def unregister_callback(self, event_type: str, callback):
        """取消事件回调"""
        self.event_bus.unsubscribe(event_type, callback)
