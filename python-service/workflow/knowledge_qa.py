from typing import Dict, Any, Optional, Generator
from types import SimpleNamespace
from agent.memory_agent import MemoryAgent
from engine.events import event_bus
from agent.react_agent import ReActAgent
from service.retrieval import RetrievalAgent
import logging
import json

logger = logging.getLogger(__name__)


class KnowledgeQAAgent:
    """知识问答Agent - 统一走 AgentOrchestrator"""

    def __init__(self):
        self.event_bus = event_bus
        self.retrieval_agent = RetrievalAgent()
        self.memory_agent = MemoryAgent()
        self._react_agent = None
        self._multi_agent_orchestrator = None
        self._agents_registered = False

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
            from engine.agent_orchestrator import AgentOrchestrator
            from engine.agent_registry import agent_registry
            self._multi_agent_orchestrator = AgentOrchestrator()
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
        """处理知识问答 - 统一走 AgentOrchestrator"""
        logger.info(f"[KnowledgeQAAgent] Processing question: {question[:50]}...")

        try:
            memory_state = SimpleNamespace(
                conversation_id=conversation_id, user_id=user_id,
                original_input=question, run_id="memory_load"
            )
            memory_result = self.memory_agent.load_memory(memory_state)
            conversation_history = memory_result.get("text", "")

            return self._ask_with_multi_agent(question, conversation_id, user_id, conversation_history,
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
        """流式处理知识问答 - 统一走 AgentOrchestrator"""
        logger.info(f"[KnowledgeQAAgent] Stream processing question: {question[:50]}...")

        try:
            memory_state = SimpleNamespace(
                conversation_id=conversation_id, user_id=user_id,
                original_input=question, run_id="memory_load"
            )
            memory_result = self.memory_agent.load_memory(memory_state)
            conversation_history = memory_result.get("text", "")

            for event in self.multi_agent_orchestrator.run_stream(
                question, conversation_id=conversation_id,
                user_id=user_id, conversation_history=conversation_history,
                **kwargs
            ):
                yield event

        except Exception as e:
            logger.error(f"[KnowledgeQAAgent] Stream QA failed: {e}", exc_info=True)
            yield json.dumps({
                "type": "error",
                "content": "处理问题时遇到错误，请稍后再试。"
            })

    def register_callback(self, event_type: str, callback):
        """注册事件回调"""
        self.event_bus.subscribe(event_type, callback)

    def unregister_callback(self, event_type: str, callback):
        """取消事件回调"""
        self.event_bus.unsubscribe(event_type, callback)
