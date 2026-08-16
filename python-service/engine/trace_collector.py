"""
请求级 trace 收集器。

每个 HTTP 请求创建一个 TraceCollector 实例，在请求处理过程中收集各阶段的 trace 事件。
请求结束时将事件列表塞入响应 JSON，由 Java 侧的 AgentTraceService 合并后统一落库。

使用方式:
    collector = TraceCollector(trace_id="abc123", session_id="123", user_id="456")

    collector.start_timer("intent")
    intent = classifier.classify(text)
    collector.record_event(
        event_type="INTENT_CLASSIFIED",
        phase="INTENT",
        input_data={"question": text},
        output_data={"intent": intent},
        latency_ms=collector.stop_timer("intent"),
    )

    return {"answer": "...", "traces": collector.to_list()}
"""

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class TraceEvent:
    """单条 trace 事件，字段名与 Java 侧 RequestTrace 实体保持一致（下划线风格）"""
    trace_id: str = ""
    session_id: str = ""
    user_id: str = ""
    step_order: int = 0
    event_type: str = ""
    phase: str = ""
    source: str = "python"
    agent_name: str = ""
    model_name: str = ""
    input_data: Any = None
    output_data: Any = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    latency_ms: float = 0
    metadata: Optional[Dict[str, Any]] = None
    event_time: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # 过滤掉 None 值，减小 JSON 体积
        return {k: v for k, v in d.items() if v is not None}


class TraceCollector:
    """请求级的 trace 事件收集器，非线程安全（每个请求创建一个新实例）"""

    MAX_SNAPSHOT_LENGTH = 100000  # 不截断

    def __init__(self, trace_id: str, session_id: str = "", user_id: str = ""):
        self.trace_id = trace_id
        self.session_id = session_id
        self.user_id = user_id
        self._events: List[TraceEvent] = []
        self._step_counter = 0
        self._timers: Dict[str, float] = {}

    # ── 计时器 ──────────────────────────────────────

    def start_timer(self, key: str):
        """开始计时，同时记录开始时间（用于设置事件时间）"""
        now = time.time()
        self._timers[key] = now
        self._timer_start_times = getattr(self, '_timer_start_times', {})
        self._timer_start_times[key] = self._format_time(now)

    def stop_timer(self, key: str):
        """结束计时，返回 (耗时ms, 开始时间字符串)"""
        start = self._timers.pop(key, time.time())
        latency = (time.time() - start) * 1000
        start_time = self._timer_start_times.pop(key, self._format_time(start))
        return (latency, start_time)

    # ── 事件记录 ────────────────────────────────────

    def record_event(
        self,
        event_type: str,
        phase: str,
        input_data: Any = None,
        output_data: Any = None,
        agent_name: str = "",
        model_name: str = "",
        latency_ms: float = 0,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        event_time: str = "",
    ):
        """追加一条 trace 事件，event_time 为空时使用当前时间"""
        self._step_counter += 1
        event = TraceEvent(
            trace_id=self.trace_id,
            session_id=self.session_id,
            user_id=self.user_id,
            step_order=self._step_counter,
            event_type=event_type,
            phase=phase,
            source="python",
            agent_name=agent_name,
            model_name=model_name,
            input_data=self._truncate(input_data),
            output_data=self._truncate(output_data),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            latency_ms=round(latency_ms, 2),
            metadata=metadata,
            event_time=event_time if event_time else self._format_time(time.time()),
        )
        self._events.append(event)

    def _format_time(self, t: float) -> str:
        """秒级时间戳 → ISO 格式字符串"""
        import datetime
        dt = datetime.datetime.fromtimestamp(t)
        return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{int(dt.microsecond / 1000):03d}"

    # ── 导出 ────────────────────────────────────────

    def to_list(self) -> List[Dict[str, Any]]:
        """导出为 dict 列表，可直接序列化为 JSON"""
        return [e.to_dict() for e in self._events]

    def to_dict(self) -> Dict[str, Any]:
        """导出完整结构"""
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "event_count": len(self._events),
            "events": self.to_list(),
        }

    # ── 内部工具 ────────────────────────────────────

    def _truncate(self, value: Any) -> Any:
        """保留方法签名但不再截断，直接返回原值"""
        return value
