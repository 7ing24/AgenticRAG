from typing import Dict, Any, List
from tools.base import Tool, ToolSchema, SchemaProperty, ToolMetadata
from core.config import config
from core.redis_client import redis_client
from core.llm import LLMService

# 压缩阈值配置
COMPRESS_THRESHOLD = 10  # 超过10轮触发压缩
KEEP_RECENT = 5          # 保留最近5轮完整对话


class WorkingMemoryReadTool(Tool):
    """对话记忆读取工具（L0 工作记忆，含上下文压缩）"""

    def __init__(self):
        input_schema = ToolSchema(
            properties={
                "conversation_id": SchemaProperty(
                    type="string",
                    description="对话ID",
                    required=True
                ),
                "limit": SchemaProperty(
                    type="number",
                    description="返回消息数量限制",
                    required=False,
                    default=10
                )
            },
            type="object"
        )

        output_schema = ToolSchema(
            properties={
                "messages": SchemaProperty(
                    type="array",
                    description="对话消息列表",
                    required=True
                ),
                "conversation_id": SchemaProperty(
                    type="string",
                    description="对话ID",
                    required=True
                ),
                "total_count": SchemaProperty(
                    type="number",
                    description="总消息数量",
                    required=True
                ),
                "compressed": SchemaProperty(
                    type="boolean",
                    description="是否已压缩",
                    required=False
                )
            },
            type="object"
        )

        metadata = ToolMetadata(
            timeout_ms=10000,
            max_retries=1,
            permission="user",
            description="读取L0工作记忆（当前会话对话历史）"
        )

        super().__init__(
            name="working_memory_read",
            description="读取L0工作记忆（当前会话对话历史）",
            input_schema=input_schema,
            output_schema=output_schema,
            metadata=metadata
        )

        self.llm_service = LLMService()

    def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """执行 L0 工作记忆读取（含上下文压缩）"""
        conversation_id = parameters.get("conversation_id")
        limit = int(parameters.get("limit", 10))

        config.logger.info(f"Reading conversation memory for ID: {conversation_id}")

        total_count = redis_client.get_message_count(conversation_id)

        if total_count == 0:
            return {
                "messages": [],
                "conversation_id": conversation_id,
                "total_count": 0,
                "compressed": False
            }

        if total_count <= COMPRESS_THRESHOLD:
            messages = redis_client.get_messages(conversation_id, limit)
            return {
                "messages": messages,
                "conversation_id": conversation_id,
                "total_count": total_count,
                "compressed": False
            }

        # 超过阈值，触发压缩
        recent_messages = redis_client.get_messages(conversation_id, KEEP_RECENT)
        cached_summary = redis_client.get_summary(conversation_id)

        if cached_summary:
            summary = cached_summary
            config.logger.info(
                f"Using cached summary for conversation {conversation_id}"
            )
        else:
            all_messages = redis_client.get_all_messages(conversation_id)
            early_messages = all_messages[:-KEEP_RECENT]
            summary = self._compress_history(early_messages, conversation_id)
            redis_client.set_summary(conversation_id, summary)
            config.logger.info(
                f"Compressed {len(early_messages)} messages into summary"
            )

        return {
            "messages": [
                {"role": "system", "content": f"[历史对话摘要] {summary}"}
            ] + recent_messages,
            "conversation_id": conversation_id,
            "total_count": total_count,
            "compressed": True,
            "original_count": total_count
        }

    def _compress_history(self, messages: List[Dict], conversation_id: str) -> str:
        """用 LLM 压缩早期对话为摘要"""
        if not messages:
            return "无历史对话记录。"

        history_text = "\n".join([
            f"{m.get('role', 'unknown')}: {m.get('content', '')}"
            for m in messages
        ])

        prompt = f"""请将以下对话压缩为结构化摘要，按用户提问逐条总结，不要遗漏任何一个问题。

对话内容：
{history_text}

要求：
1. 用"用户询问了以下问题："开头
2. 对每个用户问题，用一行总结：问题是什么、AI 的核心回答是啥
3. 如果有反复出现的主题或用户偏好，在最后单独标注
4. 不使用 Markdown，纯文本输出"""

        try:
            summary = self.llm_service.generate(prompt)
            return summary
        except Exception as e:
            config.logger.warning(f"Failed to compress history: {e}")
            return f"用户进行了 {len(messages)} 轮对话，讨论了相关知识问题。"


class LongTermMemoryReadTool(Tool):
    """长期记忆读取工具（L1 分层记忆检索）

    从 Milvus 中检索三种类型的长期记忆：
    - semantic（语义记忆）：稳定的事实、知识、偏好
    - episodic（情景记忆）：具体的历史互动事件
    - procedural（程序记忆）：可复用的方法、工作流
    """

    def __init__(self):
        input_schema = ToolSchema(
            properties={
                "query": SchemaProperty(
                    type="string",
                    description="检索查询（用户当前问题或其改写）",
                    required=True
                ),
                "user_id": SchemaProperty(
                    type="number",
                    description="用户ID",
                    required=True
                ),
                "semantic_k": SchemaProperty(
                    type="number",
                    description="语义记忆召回数量（0-5）",
                    required=False,
                    default=3
                ),
                "episodic_k": SchemaProperty(
                    type="number",
                    description="情景记忆召回数量（0-5）",
                    required=False,
                    default=2
                ),
                "procedural_k": SchemaProperty(
                    type="number",
                    description="程序记忆召回数量（0-5）",
                    required=False,
                    default=2
                ),
            },
            type="object"
        )

        output_schema = ToolSchema(
            properties={
                "memories": SchemaProperty(
                    type="object",
                    description="按类型分组的记忆字典",
                    required=True
                ),
                "total_count": SchemaProperty(
                    type="number",
                    description="检索到的记忆总数",
                    required=True
                ),
            },
            type="object"
        )

        metadata = ToolMetadata(
            timeout_ms=15000,
            max_retries=1,
            permission="user",
            description="检索L1长期分层记忆（语义/情景/程序）"
        )

        super().__init__(
            name="long_term_memory_read",
            description="从长期分层记忆中检索语义、情景、程序记忆",
            input_schema=input_schema,
            output_schema=output_schema,
            metadata=metadata
        )

        self._memory_manager = None

    @property
    def memory_manager(self):
        """延迟加载 MemoryManager"""
        if self._memory_manager is None:
            from memory.memory_manager import get_memory_manager
            self._memory_manager = get_memory_manager()
        return self._memory_manager

    def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """执行 L1 长期记忆检索"""
        query = parameters.get("query", "")
        user_id = int(parameters.get("user_id", 0))
        semantic_k = min(int(parameters.get("semantic_k", 3)), 5)
        episodic_k = min(int(parameters.get("episodic_k", 2)), 5)
        procedural_k = min(int(parameters.get("procedural_k", 2)), 5)

        config.logger.info(
            f"L1 记忆检索 — query: {query[:50]}..., "
            f"semantic_k={semantic_k}, episodic_k={episodic_k}, "
            f"procedural_k={procedural_k}"
        )

        try:
            memories = self.memory_manager.hybrid_retrieval_memories(
                query=query,
                user_id=user_id,
                semantic_k=semantic_k,
                episodic_k=episodic_k,
                procedural_k=procedural_k,
            )
        except Exception as e:
            config.logger.error(f"L1 记忆检索失败: {e}")
            memories = {"semantic": [], "episodic": [], "procedural": []}

        total = (
            len(memories.get("semantic", []))
            + len(memories.get("episodic", []))
            + len(memories.get("procedural", []))
        )

        config.logger.info(f"L1 记忆检索完成，共 {total} 条")

        return {
            "memories": memories,
            "total_count": total,
        }
