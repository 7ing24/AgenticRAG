"""Memory Agent - 独立的记忆管理Agent"""

from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor
from tools.registry import tool_registry
from core.llm import LLMService
import logging
import json

logger = logging.getLogger(__name__)


class MemoryAgent:
    """独立的记忆管理 Agent - 主动管理记忆生命周期"""

    # 压缩配置
    COMPRESS_THRESHOLD = 10  # 超过10轮触发压缩
    KEEP_RECENT = 5          # 保留最近5轮完整对话

    def __init__(self):
        self.llm_service = LLMService()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="memory")

    def load_memory(self, state, max_rounds: int = 10) -> dict:
        """加载记忆上下文（会话记忆 + 用户画像）

        Args:
            state: AgentState，需有 conversation_id / user_id / run_id
            max_rounds: 最大加载轮数，默认 10

        Returns:
            {"text": str, "token_count": int}
        """
        context_parts = []

        # 1. 加载会话记忆
        if state.conversation_id and tool_registry.has_tool("conversation_memory_read"):
            try:
                history = tool_registry.invoke_tool(
                    "conversation_memory_read",
                    {
                        "conversation_id": state.conversation_id,
                        "limit": max_rounds
                    },
                    run_id=state.run_id
                )
                messages = history.get("messages", [])
                if messages:
                    formatted = self._format_history(messages)
                    context_parts.append(formatted)
                    logger.info(f"[{state.run_id}] MemoryAgent loaded {len(messages)} messages"
                                f" (compressed: {history.get('compressed', False)}, "
                                f"max_rounds: {max_rounds})")
            except Exception as e:
                logger.warning(f"[{state.run_id}] MemoryAgent failed to load conversation: {e}")

        # 2. 加载用户画像（如果有 user_id）
        if state.user_id:
            user_profile = self._load_user_profile(state.user_id)
            if user_profile:
                context_parts.append(f"[用户画像] {json.dumps(user_profile, ensure_ascii=False)}")

        text = "\n\n".join(context_parts) if context_parts else ""
        return {"text": text, "token_count": self._estimate_tokens(text)}

    def save_memory(self, state, question: str, answer: str):
        """保存记忆（用户问题 + AI回答 + 提取偏好）"""
        if not state.conversation_id:
            return

        # 1. 写入用户问题
        if tool_registry.has_tool("conversation_memory_write"):
            try:
                tool_registry.invoke_tool(
                    "conversation_memory_write",
                    {
                        "conversation_id": state.conversation_id,
                        "role": "user",
                        "content": question
                    },
                    run_id=state.run_id
                )
            except Exception as e:
                logger.warning(f"[{state.run_id}] MemoryAgent failed to write user message: {e}")

        # 2. 写入 AI 回答
        if tool_registry.has_tool("conversation_memory_write"):
            try:
                tool_registry.invoke_tool(
                    "conversation_memory_write",
                    {
                        "conversation_id": state.conversation_id,
                        "role": "assistant",
                        "content": answer
                    },
                    run_id=state.run_id
                )
            except Exception as e:
                logger.warning(f"[{state.run_id}] MemoryAgent failed to write assistant message: {e}")

        # 3. 线程池异步提取用户偏好（不阻塞主流程）
        if state.user_id:
            self._executor.submit(
                self._extract_user_preference, state, question, answer
            )

    def _load_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """加载用户画像"""
        try:
            from core.mysql_client import user_memory_client
            return user_memory_client.get_user_memory(user_id)
        except Exception as e:
            logger.warning(f"Failed to load user profile: {e}")
            return None

    def _extract_user_preference(self, state, question: str, answer: str):
        """在线程池中异步提取用户偏好"""
        try:
            from core.mysql_client import user_memory_client

            prompt = f"""分析以下问答，判断用户是否有明显的偏好特征。

用户问题：{question}
AI回答：{answer}

从以下维度自由描述用户的偏好（不必全部填写，只写能观察到的）：
- 回答风格（简洁精炼 / 详细展开 / 先结论后推导 / 喜欢代码示例 / ...）
- 知识深度（入门科普 / 进阶原理 / 实战操作 / ...）
- 关注领域（哪些技术主题反复出现）
- 其他任何值得记录的偏好

如果有明显偏好，返回 JSON：
{{"preferences": {{"回答风格": "...", "知识深度": "...", "关注领域": [...]}}}}
如果没有明显偏好，返回：null
只返回 JSON 或 null，不要解释。"""

            result = self.llm_service.generate(prompt)
            if result and result.strip() != "null":
                cleaned = result.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
                    cleaned = cleaned.rsplit("```", 1)[0]

                preference = json.loads(cleaned.strip())
                user_memory_client.update_user_memory(
                    state.user_id, "preferences", preference,
                    source="agent", confidence=0.8
                )
                logger.info(f"[{state.run_id}] Extracted user preference: {preference}")
        except Exception as e:
            logger.warning(f"[{state.run_id}] Extract user preference failed: {e}")

    def _format_history(self, messages: list) -> str:
        """格式化对话历史"""
        if not messages:
            return ""

        formatted = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role == "system":
                formatted.append(content)
            elif role == "user":
                formatted.append(f"user: {content}")
            elif role == "assistant":
                formatted.append(f"assistant: {content}")
            else:
                formatted.append(content)

        return "\n".join(formatted)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """估算文本 token 数（中英文混合的简单估算）

        中文约 1.5 字符/token，英文约 4 字符/token。
        对混合文本取折中值 ~2.5 字符/token。
        后续可替换为 qwen tokenizer 精确计数。
        """
        if not text:
            return 0
        # 粗略区分中英文字符比例
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        # 中文 ~1.5 chars/token, 英文/其他 ~4 chars/token
        tokens = chinese_chars / 1.5 + other_chars / 4.0
        return int(tokens) + 1  # +1 向上取整
