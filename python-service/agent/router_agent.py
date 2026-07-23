from typing import Dict, Any, Optional, Generator
from enum import Enum
from workflow.knowledge_qa import KnowledgeQAAgent
from workflow.chitchat import ChitChatAgent
from workflow.admin_copilot import AdminCopilotAgent
from service.retrieval import RetrievalAgent
from intent.classifier import IntentClassifier, IntentType
from engine.trace_collector import TraceCollector
import logging
import json

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """任务类型枚举"""
    CHITCHAT = "chitchat"
    KNOWLEDGE_QA = "knowledge_qa"
    ADMIN_COPILOT = "admin_copilot"
    UNKNOWN = "unknown"


class RouterAgent:
    """路由Agent - 负责将用户请求路由到合适的工作流"""

    def __init__(self):
        self.knowledge_qa_agent = KnowledgeQAAgent()
        self.chitchat_agent = ChitChatAgent()
        self.admin_copilot_agent = AdminCopilotAgent()
        self.retrieval_agent = RetrievalAgent()
        self.classifier = IntentClassifier()

    def route(self, input_text: str, conversation_id: Optional[str] = None,
              user_id: Optional[str] = None,
              is_admin: bool = False, **kwargs) -> Dict[str, Any]:
        """
        路由并执行任务

        Args:
            input_text: 用户输入
            conversation_id: 会话ID
            user_id: 用户ID
            is_admin: 是否为管理员
            **kwargs: 其他参数（含 trace_id）

        Returns:
            执行结果（含 traces 字段）
        """
        trace_id = kwargs.get("trace_id", "")
        collector = TraceCollector(
            trace_id=trace_id,
            session_id=str(conversation_id) if conversation_id else "",
            user_id=str(user_id) if user_id else "",
        )

        # ── 1. 意图分类 ────────────────────────────
        collector.start_timer("intent")
        result = self.classifier.classify(input_text, is_admin)
        intent_latency, intent_start = collector.stop_timer("intent")
        collector.record_event(
            event_type="INTENT_CLASSIFIED",
            phase="INTENT",
            input_data={"question": input_text[:200], "is_admin": is_admin},
            output_data={"intent": result.intent.value if hasattr(result.intent, 'value') else str(result.intent),
                         "confidence": getattr(result, 'confidence', None)},
            agent_name="IntentClassifier",
            model_name="qwen-plus",
            latency_ms=intent_latency,
            metadata={"fallback": getattr(result, 'is_fallback', False)},
            event_time=intent_start,
        )

        task_type = self.classify_task(input_text, is_admin)
        logger.info(f"[RouterAgent] Routing to: {task_type.value} for input: {input_text[:50]}...")

        # ── 2. 路由分发 ────────────────────────────
        collector.record_event(
            event_type="ROUTE_SELECTED",
            phase="ROUTE",
            input_data={"task_type": task_type.value, "intent": result.intent.value if hasattr(result.intent, 'value') else str(result.intent)},
            output_data={"agent": task_type.value},
        )

        # 将 collector 传给下游 agent
        kwargs["trace_collector"] = collector

        try:
            if task_type == TaskType.CHITCHAT:
                response = self.chitchat_agent.chat(
                    input_text, conversation_id, user_id, **kwargs
                )

            elif task_type == TaskType.KNOWLEDGE_QA:
                response = self.knowledge_qa_agent.ask(
                    input_text, conversation_id, user_id, **kwargs
                )

            elif task_type == TaskType.ADMIN_COPILOT:
                response = self.admin_copilot_agent.handle(
                    input_text, conversation_id, user_id, **kwargs
                )

            else:
                response = self.knowledge_qa_agent.ask(
                    input_text, conversation_id, user_id, **kwargs
                )

            # ── 3. 将 traces 写入响应 ──────────────
            response["traces"] = collector.to_list()
            return response

        except Exception as e:
            logger.error(f"[RouterAgent] Route error: {str(e)}")
            collector.record_event(
                event_type="ROUTE_FAILED",
                phase="ROUTE",
                output_data={"error": str(e)[:200]},
                metadata={"error_type": type(e).__name__},
            )
            return {
                "answer": "抱歉，服务暂时不可用，请稍后再试。",
                "sources": [],
                "has_sources": False,
                "task_type": task_type.value,
                "error": True,
                "error_message": str(e),
                "traces": collector.to_list(),
            }

    def route_stream(self, input_text: str, conversation_id: Optional[str] = None,
                     user_id: Optional[str] = None,
                     is_admin: bool = False, **kwargs) -> Generator[str, None, None]:
        """
        流式路由并执行任务

        Args:
            input_text: 用户输入
            conversation_id: 会话ID
            user_id: 用户ID
            is_admin: 是否为管理员
            **kwargs: 其他参数

        Yields:
            JSON格式的事件流
        """
        task_type = self.classify_task(input_text, is_admin)
        logger.info(f"[RouterAgent] Streaming route to: {task_type.value}")

        try:
            yield json.dumps({
                "type": "routed",
                "task_type": task_type.value
            })

            if task_type == TaskType.CHITCHAT:
                for event in self.chitchat_agent.chat_stream(
                    input_text, conversation_id, user_id, **kwargs
                ):
                    yield event

            elif task_type == TaskType.KNOWLEDGE_QA:
                for event in self.knowledge_qa_agent.ask_stream(
                    input_text, conversation_id, user_id, **kwargs
                ):
                    yield event

            elif task_type == TaskType.ADMIN_COPILOT:
                for event in self.admin_copilot_agent.handle_stream(
                    input_text, conversation_id, user_id, **kwargs
                ):
                    yield event

            else:
                for event in self.knowledge_qa_agent.ask_stream(
                    input_text, conversation_id, user_id, context, **kwargs
                ):
                    yield event

        except Exception as e:
            logger.error(f"[RouterAgent] Stream route error: {str(e)}")
            yield json.dumps({
                "type": "error",
                "content": str(e)
            })

    def classify_task(self, input_text: str, is_admin: bool = False) -> TaskType:
        """
        分类任务类型 - 委托给统一分类器

        Args:
            input_text: 用户输入
            is_admin: 是否为管理员

        Returns:
            任务类型
        """
        result = self.classifier.classify(input_text, is_admin)

        # 将IntentType映射到TaskType
        intent_to_task = {
            IntentType.CHITCHAT: TaskType.CHITCHAT,
            IntentType.KNOWLEDGE_QA: TaskType.KNOWLEDGE_QA,
            IntentType.ADMIN_OPERATION: TaskType.ADMIN_COPILOT,
            IntentType.IDENTITY_QUERY: TaskType.CHITCHAT,
            IntentType.UNKNOWN: TaskType.KNOWLEDGE_QA,
        }

        task_type = intent_to_task.get(result.intent, TaskType.KNOWLEDGE_QA)

        # 非管理员不能访问管理功能（含知识巡检），强制降级为知识问答
        if not is_admin and task_type == TaskType.ADMIN_COPILOT:
            logger.info(f"[RouterAgent] Non-admin user attempted admin operation, "
                        f"downgrading to KNOWLEDGE_QA (intent: {result.intent})")
            task_type = TaskType.KNOWLEDGE_QA

        return task_type

    def get_agent(self, task_type: TaskType):
        """获取对应的Agent"""
        agent_map = {
            TaskType.CHITCHAT: self.chitchat_agent,
            TaskType.KNOWLEDGE_QA: self.knowledge_qa_agent,
            TaskType.ADMIN_COPILOT: self.admin_copilot_agent,
        }
        return agent_map.get(task_type, self.knowledge_qa_agent)

    def get_task_stats(self) -> Dict[str, int]:
        """获取各类关键词数量统计（用于调试和分析）"""
        return self.classifier.get_keyword_stats()
