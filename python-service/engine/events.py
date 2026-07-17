from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class EventType:
    """事件类型枚举"""
    RUN_STARTED = "run_started"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
    RUN_INTERRUPTED = "run_interrupted"
    RUN_TIMEOUT = "run_timeout"
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    STEP_FAILED = "step_failed"
    STEP_SKIPPED = "step_skipped"
    TOOL_CALL_STARTED = "tool_call_started"
    TOOL_CALL_COMPLETED = "tool_call_completed"
    TOOL_CALL_FAILED = "tool_call_failed"
    INTENT_RECOGNIZED = "intent_recognized"
    QUESTION_CLASSIFIED = "question_classified"
    CLARIFICATION_NEEDED = "clarification_needed"
    QUESTION_REWRITTEN = "question_rewritten"
    RETRIEVAL_COMPLETED = "retrieval_completed"
    SUFFICIENCY_EVALUATED = "sufficiency_evaluated"
    ANSWER_GENERATED = "answer_generated"
    MEMORY_WRITTEN = "memory_written"
    STATE_UPDATED = "state_updated"
    # 多 Agent 协作事件
    SUB_TASK_STARTED = "sub_task_started"
    SUB_TASK_COMPLETED = "sub_task_completed"
    SUB_TASK_FAILED = "sub_task_failed"
    FINDING_PUBLISHED = "finding_published"
    SYNTHESIS_COMPLETED = "synthesis_completed"


@dataclass
class Event:
    """事件基类"""
    event_type: str
    run_id: str
    trace_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "run_id": self.run_id,
            "trace_id": self.trace_id,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data
        }


@dataclass
class RunStartedEvent(Event):
    """运行开始事件"""
    def __init__(self, run_id: str, goal: str, input_data: str, trace_id: str = "", **kwargs):
        super().__init__(
            event_type=EventType.RUN_STARTED,
            run_id=run_id,
            trace_id=trace_id,
            data={"goal": goal, "input": input_data}
        )


@dataclass
class RunCompletedEvent(Event):
    """运行完成事件"""
    def __init__(self, run_id: str, output: Dict[str, Any], trace_id: str = "", **kwargs):
        super().__init__(
            event_type=EventType.RUN_COMPLETED,
            run_id=run_id,
            trace_id=trace_id,
            data={"output": output}
        )


@dataclass
class RunFailedEvent(Event):
    """运行失败事件"""
    def __init__(self, run_id: str, error: str, error_code: Optional[str] = None, trace_id: str = "", **kwargs):
        super().__init__(
            event_type=EventType.RUN_FAILED,
            run_id=run_id,
            trace_id=trace_id,
            data={"error": error, "error_code": error_code}
        )


@dataclass
class StepEvent(Event):
    """步骤事件基类"""
    step_id: str = ""
    step_name: str = ""
    step_type: str = ""

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["data"]["step_id"] = self.step_id
        result["data"]["step_name"] = self.step_name
        result["data"]["step_type"] = self.step_type
        return result


@dataclass
class StepStartedEvent(StepEvent):
    """步骤开始事件"""
    def __init__(self, run_id: str, step_id: str, step_name: str, step_type: str, trace_id: str = "", **kwargs):
        super().__init__(
            event_type=EventType.STEP_STARTED,
            run_id=run_id,
            trace_id=trace_id,
            step_id=step_id,
            step_name=step_name,
            step_type=step_type
        )


@dataclass
class StepCompletedEvent(StepEvent):
    """步骤完成事件"""
    def __init__(self, run_id: str, step_id: str, step_name: str, step_type: str,
                 output: Dict[str, Any], duration_ms: float, trace_id: str = "", **kwargs):
        super().__init__(
            event_type=EventType.STEP_COMPLETED,
            run_id=run_id,
            trace_id=trace_id,
            step_id=step_id,
            step_name=step_name,
            step_type=step_type,
            data={"output": output, "duration_ms": duration_ms}
        )


@dataclass
class StepFailedEvent(StepEvent):
    """步骤失败事件"""
    def __init__(self, run_id: str, step_id: str, step_name: str, step_type: str,
                 error: str, trace_id: str = "", **kwargs):
        super().__init__(
            event_type=EventType.STEP_FAILED,
            run_id=run_id,
            trace_id=trace_id,
            step_id=step_id,
            step_name=step_name,
            step_type=step_type,
            data={"error": error}
        )


@dataclass
class ToolCallEvent(Event):
    """工具调用事件"""
    tool_call_id: str = ""
    tool_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["data"]["tool_call_id"] = self.tool_call_id
        result["data"]["tool_name"] = self.tool_name
        return result


@dataclass
class ToolCallCompletedEvent(ToolCallEvent):
    """工具调用完成事件"""
    def __init__(self, run_id: str, tool_call_id: str, tool_name: str,
                 output: Dict[str, Any], duration_ms: float, trace_id: str = "", **kwargs):
        super().__init__(
            event_type=EventType.TOOL_CALL_COMPLETED,
            run_id=run_id,
            trace_id=trace_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            data={"output": output, "duration_ms": duration_ms}
        )


@dataclass
class ToolCallFailedEvent(ToolCallEvent):
    """工具调用失败事件"""
    def __init__(self, run_id: str, tool_call_id: str, tool_name: str,
                 error: str, trace_id: str = "", **kwargs):
        super().__init__(
            event_type=EventType.TOOL_CALL_FAILED,
            run_id=run_id,
            trace_id=trace_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            data={"error": error}
        )


@dataclass
class AnswerGeneratedEvent(Event):
    """答案生成事件"""
    def __init__(self, run_id: str, answer: str, sources: List[Dict[str, Any]], trace_id: str = "", **kwargs):
        super().__init__(
            event_type=EventType.ANSWER_GENERATED,
            run_id=run_id,
            trace_id=trace_id,
            data={"answer": answer, "sources": sources}
        )


# ── 多 Agent 协作事件 ──────────────────────────────────────────


@dataclass
class FindingPublishedEvent(Event):
    """Worker 向黑板发布发现的事件

    通过 event_bus 共享中间结果，其他 worker 可以通过 get_latest(key) 读取。
    """
    def __init__(self, run_id: str, key: str, value: Any,
                 publisher: str = "", trace_id: str = "", **kwargs):
        super().__init__(
            event_type=EventType.FINDING_PUBLISHED,
            run_id=run_id,
            trace_id=trace_id,
            data={
                "key": key,
                "value": value,
                "publisher": publisher,
            }
        )

    @property
    def key(self) -> str:
        return self.data.get("key", "")

    @property
    def value(self) -> Any:
        return self.data.get("value")

    @property
    def publisher(self) -> str:
        return self.data.get("publisher", "")


@dataclass
class SubTaskEvent(Event):
    """子任务事件基类"""
    sub_task_id: str = ""
    sub_task_name: str = ""
    worker: str = ""

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["data"]["sub_task_id"] = self.sub_task_id
        result["data"]["sub_task_name"] = self.sub_task_name
        result["data"]["worker"] = self.worker
        return result


@dataclass
class SubTaskStartedEvent(SubTaskEvent):
    """子任务开始事件"""
    def __init__(self, run_id: str, sub_task_id: str, sub_task_name: str,
                 worker: str = "", trace_id: str = "", **kwargs):
        super().__init__(
            event_type=EventType.SUB_TASK_STARTED,
            run_id=run_id,
            trace_id=trace_id,
            sub_task_id=sub_task_id,
            sub_task_name=sub_task_name,
            worker=worker
        )


@dataclass
class SubTaskCompletedEvent(SubTaskEvent):
    """子任务完成事件"""
    def __init__(self, run_id: str, sub_task_id: str, sub_task_name: str,
                 worker: str = "", result: Any = None, duration_ms: float = 0, trace_id: str = "", **kwargs):
        super().__init__(
            event_type=EventType.SUB_TASK_COMPLETED,
            run_id=run_id,
            trace_id=trace_id,
            sub_task_id=sub_task_id,
            sub_task_name=sub_task_name,
            worker=worker,
            data={"result": result, "duration_ms": duration_ms}
        )


@dataclass
class SynthesisCompletedEvent(Event):
    """汇总完成事件"""
    def __init__(self, run_id: str, answer: str, sub_task_count: int = 0, trace_id: str = "", **kwargs):
        super().__init__(
            event_type=EventType.SYNTHESIS_COMPLETED,
            run_id=run_id,
            trace_id=trace_id,
            data={"answer": answer, "sub_task_count": sub_task_count}
        )


class EventBus:
    """事件总线 - 负责事件的发布和订阅，兼作多 Agent 共享黑板"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._handlers = {}
            cls._instance._global_handlers = []
            cls._instance._latest = {}  # event_type → latest event (黑板缓存)
            cls._instance._findings = {}  # finding_key → FindingPublishedEvent (按 key 索引)
        return cls._instance

    def subscribe(self, event_type: str, handler: Callable[[Event], None]):
        """订阅事件"""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        logger.debug(f"Subscribed handler for event type: {event_type}")

    def subscribe_global(self, handler: Callable[[Event], None]):
        """订阅所有事件"""
        self._global_handlers.append(handler)
        logger.debug("Subscribed global event handler")

    def unsubscribe(self, event_type: str, handler: Callable[[Event], None]):
        """取消订阅"""
        if event_type in self._handlers:
            self._handlers[event_type].remove(handler)

    def publish(self, event: Event):
        """发布事件，同时缓存到黑板"""
        # 缓存最新事件（黑板功能）
        self._latest[event.event_type] = event
        if isinstance(event, FindingPublishedEvent) and event.key:
            self._findings[event.key] = event

        logger.debug(f"Publishing event: {event.event_type} for run: {event.run_id}")

        if event.event_type in self._handlers:
            for handler in self._handlers[event.event_type]:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"Error handling event {event.event_type}: {e}")

        for handler in self._global_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in global event handler: {e}")

    # ── 黑板读取方法 ──────────────────────────────────────────

    def get_latest(self, event_type: str) -> Optional[Event]:
        """按事件类型读取最新事件"""
        return self._latest.get(event_type)

    def get_finding(self, key: str) -> Optional[FindingPublishedEvent]:
        """按 key 读取共享发现"""
        return self._findings.get(key)

    def get_all_findings(self) -> Dict[str, FindingPublishedEvent]:
        """读取所有共享发现"""
        return dict(self._findings)

    def get_all_latest(self) -> Dict[str, Event]:
        """读取所有最新事件"""
        return dict(self._latest)

    def clear_run(self, run_id: str):
        """清理指定 run 的缓存数据"""
        keys_to_remove = []
        for key, finding in self._findings.items():
            if finding.run_id == run_id:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del self._findings[key]

    def clear(self):
        """清空所有订阅和缓存"""
        self._handlers.clear()
        self._global_handlers.clear()
        self._latest.clear()
        self._findings.clear()


event_bus = EventBus()
