from typing import Dict, Any
from tools.base import Tool, ToolSchema, SchemaProperty, ToolMetadata
from core.config import config
from core.redis_client import redis_client
import uuid


class WorkingMemoryWriteTool(Tool):
    """对话记忆写入工具（L0 工作记忆 + 触发 L0→L1 提取检查）"""

    def __init__(self):
        input_schema = ToolSchema(
            properties={
                "conversation_id": SchemaProperty(
                    type="string",
                    description="对话ID",
                    required=True
                ),
                "role": SchemaProperty(
                    type="string",
                    description="角色 (user 或 assistant)",
                    required=True
                ),
                "content": SchemaProperty(
                    type="string",
                    description="消息内容",
                    required=True
                ),
                "user_id": SchemaProperty(
                    type="string",
                    description="用户ID（用于长期记忆关联）",
                    required=False
                ),
            },
            type="object"
        )

        output_schema = ToolSchema(
            properties={
                "success": SchemaProperty(
                    type="boolean",
                    description="是否成功",
                    required=True
                ),
                "message_id": SchemaProperty(
                    type="string",
                    description="消息ID",
                    required=True
                ),
                "conversation_id": SchemaProperty(
                    type="string",
                    description="对话ID",
                    required=True
                ),
                "memory_extraction_triggered": SchemaProperty(
                    type="boolean",
                    description="是否触发了长期记忆提取",
                    required=False
                ),
            },
            type="object"
        )

        metadata = ToolMetadata(
            timeout_ms=5000,
            max_retries=1,
            permission="user",
            description="写入L0工作记忆，累积超过阈值时触发L0→L1提取"
        )

        super().__init__(
            name="working_memory_write",
            description="写入L0工作记忆",
            input_schema=input_schema,
            output_schema=output_schema,
            metadata=metadata
        )

        self._scheduler = None

    @property
    def scheduler(self):
        """延迟加载 MemoryScheduler"""
        if self._scheduler is None:
            from memory.memory_scheduler import get_memory_scheduler
            self._scheduler = get_memory_scheduler()
        return self._scheduler

    def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """执行对话记忆写入（双写 Redis + MySQL，检查是否触发 L1 提取）"""
        conversation_id = parameters.get("conversation_id")
        role = parameters.get("role")
        content = parameters.get("content")
        user_id_str = parameters.get("user_id")

        if role not in ["user", "assistant"]:
            raise ValueError(f"Invalid role: {role}, must be 'user' or 'assistant'")

        config.logger.info(
            f"Writing message to conversation {conversation_id}: role={role}"
        )

        # 写入 Redis（L0 工作记忆，供会话内快速读取）
        redis_client.add_message(conversation_id, role, content, user_id=user_id_str)

        # 持久化到 MySQL（L0 数据源，供提取调度器使用）
        if user_id_str:
            try:
                from core.mysql_client import user_memory_client
                user_memory_client.add_conversation_message(
                    int(user_id_str), conversation_id, role, content
                )
            except Exception as e:
                config.logger.warning(f"MySQL 消息写入失败: {e}")

        # 触发提取（仅在 assistant 写入时，MySQL 的 summary_id 天然防重复）
        extraction_triggered = False
        if user_id_str and role == "assistant":
            try:
                self._trigger_extraction_if_needed(
                    int(user_id_str), conversation_id
                )
                extraction_triggered = True
            except Exception as e:
                config.logger.warning(f"检查记忆提取阈值失败: {e}")

        return {
            "success": True,
            "message_id": str(uuid.uuid4()),
            "conversation_id": conversation_id,
            "memory_extraction_triggered": extraction_triggered,
        }

    def _trigger_extraction_if_needed(self, user_id: int, conversation_id: str):
        """从 MySQL 读未提取消息，超过阈值则异步提取"""
        from core.mysql_client import user_memory_client
        messages = user_memory_client.get_unsummarized_messages(
            user_id, conversation_id
        )
        if not messages:
            return

        formatted = [
            {"id": m["id"], "role": m["role"], "content": m["content"],
             "created_at": str(m.get("created_at", ""))}
            for m in messages
        ]

        if not self.scheduler.should_extract(formatted):
            return

        config.logger.info(
            f"Conversation {conversation_id} 超过 Token 阈值，触发长期记忆提取"
        )
        import threading
        t = threading.Thread(
            target=self.scheduler.extract_and_store,
            args=(formatted, user_id, conversation_id),
            daemon=True,
        )
        t.start()
