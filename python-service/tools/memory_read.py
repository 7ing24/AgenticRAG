from typing import Dict, Any, List
from tools.base import Tool, ToolSchema, SchemaProperty, ToolMetadata
from core.config import config
from core.redis_client import redis_client
from core.llm import LLMService
from core.text_utils import estimate_tokens

# 压缩阈值：实际喂给大模型的上下文（摘要 + 会话列表）token 数 > 此值触发压缩。
# 压缩后摘要并入、列表被物理裁剪，输入总量回落，阈值重新武装——之后输入要再次
# 超过 COMPRESS_TOKEN_THRESHOLD 才会再次触发（相当于"每攒约 10000 token 的新输入就压一次"）。
COMPRESS_TOKEN_THRESHOLD = 10000
KEEP_RECENT_MSG = 20  # 保留最近 10 轮完整对话（每轮 1 user + 1 assistant，共 20 条消息）不压缩


class WorkingMemoryReadTool(Tool):
    """对话记忆读取工具（L0 工作记忆，含 Token 驱动的上下文压缩）"""

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
                ),
                "token_count": SchemaProperty(
                    type="number",
                    description="当前上下文 token 数",
                    required=False
                ),
            },
            type="object"
        )

        metadata = ToolMetadata(
            timeout_ms=10000,
            max_retries=1,
            permission="user",
            description="读取L0工作记忆（Token驱动压缩）"
        )

        super().__init__(
            name="working_memory_read",
            description="读取L0工作记忆",
            input_schema=input_schema,
            output_schema=output_schema,
            metadata=metadata
        )

        self.llm_service = LLMService()

    def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        conversation_id = parameters.get("conversation_id")

        config.logger.info(f"Reading conversation memory for ID: {conversation_id}")

        total_count = redis_client.get_message_count(conversation_id)

        if total_count == 0:
            return {
                "messages": [], "conversation_id": conversation_id,
                "total_count": 0, "compressed": False, "token_count": 0,
            }

        all_messages = redis_client.get_all_messages(conversation_id)
        cached_summary = redis_client.get_summary(conversation_id)

        full_text = "\n".join(
            f"{m.get('role', 'unknown')}: {m.get('content', '')}"
            for m in all_messages
        )
        list_tokens = estimate_tokens(full_text)
        # 实际喂给大模型的是「摘要 + 会话列表」，因此阈值判断也要以实际输入为准
        summary_tokens = estimate_tokens(cached_summary) if cached_summary else 0
        input_tokens = summary_tokens + list_tokens

        # ── 未超过 token 阈值，或消息不足一个窗口：不压缩 ──
        # 若已有摘要，则「摘要 + 全部原始消息」一起作为上下文返回，否则返回全部原始消息
        if input_tokens <= COMPRESS_TOKEN_THRESHOLD or len(all_messages) <= KEEP_RECENT_MSG:
            if cached_summary:
                return {
                    "messages": [{"role": "system", "content": f"[历史对话摘要] {cached_summary}"}] + all_messages,
                    "conversation_id": conversation_id,
                    "total_count": total_count, "compressed": False,
                    "token_count": input_tokens,
                }
            return {
                "messages": all_messages, "conversation_id": conversation_id,
                "total_count": total_count, "compressed": False,
                "token_count": list_tokens,
            }

        # ── 超过 token 阈值：压缩窗口之外的消息，并物理裁剪列表 ──
        # 裁剪后列表 token 总量回落、阈值重新武装——之后输入要再次超过阈值才会压缩。
        early = all_messages[:-KEEP_RECENT_MSG]
        recent = all_messages[-KEEP_RECENT_MSG:]

        if cached_summary:
            # 旧摘要 + 本次新增消息 → 滚动新摘要
            summary = self._recompress(cached_summary, early, conversation_id)
            config.logger.info(
                f"[{conversation_id}] Recompress: old_summary + "
                f"{len(early)} msgs → new (input={input_tokens}t)"
            )
        else:
            summary = self._compress(early, conversation_id)
            config.logger.info(
                f"[{conversation_id}] First compress: "
                f"{len(early)} msgs → summary (input={input_tokens}t)"
            )

        # 摘要成为被裁剪消息的唯一记录，TTL 与消息列表一致（24h），避免先于消息过期
        redis_client.set_summary(conversation_id, summary, expire=86400)
        # 物理裁剪：Redis 列表只保留最近 KEEP_RECENT_MSG 条，token 总量打回
        redis_client.trim_messages(conversation_id, KEEP_RECENT_MSG)

        # ── 构建返回 ──
        context = f"[历史对话摘要] {summary}\n"
        context += "\n".join(
            f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in recent
        )
        return {
            "messages": [
                {"role": "system", "content": f"[历史对话摘要] {summary}"}
            ] + recent,
            "conversation_id": conversation_id,
            "total_count": len(recent),
            "compressed": True,
            "token_count": estimate_tokens(context),
        }

    def _compress(self, messages: List[Dict], conversation_id: str) -> str:
        """首次压缩：将消息列表压缩为摘要"""
        return self._call_llm_compress(messages, conversation_id)

    def _recompress(self, old_summary: str, new_messages: List[Dict],
                    conversation_id: str) -> str:
        """增量重压缩：旧摘要 + 新增消息 → 新摘要"""
        history_text = f"[之前的对话摘要]\n{old_summary}\n\n"
        history_text += "[最近新增的对话]\n"
        history_text += "\n".join(
            f"{m.get('role', 'unknown')}: {m.get('content', '')}"
            for m in new_messages
        )
        return self._call_llm_compress(history_text, conversation_id, is_recompress=True)

    def _call_llm_compress(self, input_text, conversation_id: str,
                           is_recompress: bool = False) -> str:
        """调 LLM 生成摘要"""
        if isinstance(input_text, list):
            input_text = "\n".join(
                f"{m.get('role', 'unknown')}: {m.get('content', '')}"
                for m in input_text
            )

        label = "[重新压缩]" if is_recompress else "[压缩]"

        prompt = f"""请将以下对话压缩为结构化摘要，按用户提问逐条总结，不要遗漏任何一个问题。

对话内容：
{input_text}

要求：
1. 用"用户询问了以下问题："开头
2. 对每个用户问题，用一行总结：问题是什么、AI 的核心回答是啥
3. 如果有反复出现的主题或用户偏好，在最后单独标注
4. 不使用 Markdown，纯文本输出"""

        try:
            summary = self.llm_service.generate(prompt)
            config.logger.info(
                f"{label} {conversation_id}: "
                f"input ~{estimate_tokens(input_text)}t → "
                f"summary ~{estimate_tokens(summary)}t"
            )
            return summary
        except Exception as e:
            config.logger.warning(f"Failed to compress history: {e}")
            return f"用户进行了多轮对话，讨论了相关知识问题。"


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
                    description="检索查询：用户当前问题，若过于口语化或模糊可做适当改写",
                    required=True
                ),
                "user_id": SchemaProperty(
                    type="number",
                    description="由系统自动注入，禁止传入",
                    required=False
                ),
                "semantic_k": SchemaProperty(
                    type="number",
                    description="语义记忆（稳定事实/偏好/概念）召回数量，0=不需要。事实问答3-5，操作问题2-3，历史回忆1-2",
                    required=False,
                    default=3
                ),
                "episodic_k": SchemaProperty(
                    type="number",
                    description="情景记忆（历史事件/对话）召回数量，0=不需要。重复提问/回忆3-5，事实问答2-3，操作问题1-2",
                    required=False,
                    default=2
                ),
                "procedural_k": SchemaProperty(
                    type="number",
                    description="程序记忆（可复用步骤/方法）召回数量，0=不需要。操作问题3-5，事实问答2-3，历史回忆1-2",
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
            description="检索用户长期记忆（语义/情景/程序），用于个性化回答与历史连贯性。调用时机：用户提到之前/上次/刚才等回忆性词汇、重复提问、询问个人偏好/习惯/操作步骤。用户ID由系统自动注入，禁止传入"
        )

        super().__init__(
            name="long_term_memory_read",
            description="检索用户长期记忆（语义/情景/程序），用于个性化回答与历史连贯性。调用时机：用户提到之前/上次/刚才等回忆性词汇、重复提问、询问个人偏好/习惯/操作步骤。用户ID由系统自动注入，禁止传入",
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
