"""Memory Agent - 三层记忆管理Agent

整合三层记忆的加载和保存：
  L0: Working Memory     — 当前会话上下文 (Redis)
  L1: Long-term Memory   — 语义/情景/程序记忆 (Milvus)
  L2: User Profile       — 用户画像 (MySQL + Redis)
"""

import logging
from typing import Dict, Any, Optional, List

from tools.registry import tool_registry
from core.llm import LLMService
from memory.memory_prompts import MEMORY_USAGE_PROMPT
from core.config import config

logger = logging.getLogger(__name__)


class MemoryAgent:
    """三层记忆管理 Agent — 协调 L0/L1/L2 的加载与保存"""

    # L1 检索默认参数
    DEFAULT_SEMANTIC_K = 3
    DEFAULT_EPISODIC_K = 2
    DEFAULT_PROCEDURAL_K = 2

    def __init__(self):
        self.llm_service = LLMService()
        self._l1_manager = None

    @property
    def l1_manager(self):
        """延迟加载 L1 MemoryManager"""
        if self._l1_manager is None:
            from memory.memory_manager import get_memory_manager
            self._l1_manager = get_memory_manager()
        return self._l1_manager

    # =========================================================================
    # 记忆加载（三层合并）
    # =========================================================================

    def load_memory(self, state, max_rounds: int = 10) -> dict:
        """加载完整记忆上下文：L1 长期记忆 + L0 工作记忆 + L2 用户画像

        Args:
            state: AgentState
            max_rounds: L0 最大加载轮数

        Returns:
            {"text": str, "token_count": int}
        """
        context_parts = []

        # 1. L2 用户画像（先加载，作为全局上下文注入最前面）
        if state.user_id:
            user_profile = self._load_user_profile(state.user_id)
            if user_profile:
                context_parts.append(f"[用户画像] {user_profile}")

        # 2. L1 长期记忆检索
        l1_memories = None
        if state.user_id and state.original_input:
            logger.info(
                f"[{state.run_id}] L1 长期记忆检索开始 — "
                f"query: {state.original_input[:50]}..."
            )
            l1_memories = self._load_long_term_memory(
                query=state.original_input,
                user_id=int(state.user_id),
            )

        # 3. L0 工作记忆
        l0_context = ""
        if state.conversation_id and tool_registry.has_tool(
            "working_memory_read"
        ):
            try:
                history = tool_registry.invoke_tool(
                    "working_memory_read",
                    {
                        "conversation_id": state.conversation_id,
                        "limit": max_rounds,
                    },
                    run_id=state.run_id,
                )
                messages = history.get("messages", [])
                if messages:
                    l0_context = self._format_history(messages)
                    context_parts.append(
                        f"[当前会话]\n{l0_context}"
                    )
                logger.info(
                    f"[{state.run_id}] L0 工作记忆: {len(messages)} 条消息"
                    f" (compressed: {history.get('compressed', False)})"
                )
            except Exception as e:
                logger.warning(
                    f"[{state.run_id}] L0 工作记忆加载失败: {e}"
                )

        # 4. 将 L1 记忆格式化为 LLM 可用的上下文
        if l1_memories and self._has_memories(l1_memories):
            import time
            now = int(time.time())
            l1_text = MEMORY_USAGE_PROMPT.format(
                memories_dict=l1_memories,
                current_timestamp=now,
            )
            # 插入到当前会话前面（作为历史上下文）
            if context_parts:
                # 找到 [当前会话] 的位置，插入到它前面
                for i, part in enumerate(context_parts):
                    if part.startswith("[当前会话]"):
                        context_parts.insert(i, l1_text)
                        break
                else:
                    context_parts.append(l1_text)
            else:
                context_parts.append(l1_text)

            logger.info(
                f"[{state.run_id}] L1 长期记忆: "
                f"semantic={len(l1_memories.get('semantic', []))}, "
                f"episodic={len(l1_memories.get('episodic', []))}, "
                f"procedural={len(l1_memories.get('procedural', []))}"
            )

        text = "\n\n".join(context_parts) if context_parts else ""
        return {"text": text, "token_count": self._estimate_tokens(text)}

    def _load_long_term_memory(
        self, query: str, user_id: int
    ) -> Optional[Dict[str, List[Dict]]]:
        """从 L1 Milvus 检索长期记忆"""
        try:
            return self.l1_manager.hybrid_retrieval_memories(
                query=query,
                user_id=user_id,
                semantic_k=self.DEFAULT_SEMANTIC_K,
                episodic_k=self.DEFAULT_EPISODIC_K,
                procedural_k=self.DEFAULT_PROCEDURAL_K,
            )
        except Exception as e:
            logger.warning(f"L1 长期记忆检索失败: {e}")
            return None

    def _load_user_profile(self, user_id: str) -> Optional[str]:
        """从 L2 MySQL/Redis 加载用户画像（纯文本）

        返回画像文本字符串，而不是 JSON dict。
        画像文本可以直接注入到 System Prompt 中。
        """
        try:
            from core.mysql_client import user_memory_client
            profile_text = user_memory_client.get_user_profile_text(user_id)
            if profile_text:
                logger.info(f"L2 用户画像加载成功 (user_id={user_id})")
                return profile_text
            return None
        except Exception as e:
            logger.warning(f"L2 用户画像加载失败: {e}")
            return None

    # =========================================================================
    # 记忆保存
    # =========================================================================

    def save_memory(self, state, question: str, answer: str):
        """保存记忆：L0 会话写入（L1/L2 提取由 memory_write 触发）"""
        if not state.conversation_id:
            return

        if tool_registry.has_tool("working_memory_write"):
            try:
                tool_registry.invoke_tool(
                    "working_memory_write",
                    {
                        "conversation_id": state.conversation_id,
                        "role": "user",
                        "content": question,
                        "user_id": str(state.user_id) if state.user_id else None,
                    },
                    run_id=state.run_id,
                )
            except Exception as e:
                logger.warning(
                    f"[{state.run_id}] L0 写入用户消息失败: {e}"
                )

            try:
                tool_registry.invoke_tool(
                    "working_memory_write",
                    {
                        "conversation_id": state.conversation_id,
                        "role": "assistant",
                        "content": answer,
                        "user_id": str(state.user_id) if state.user_id else None,
                    },
                    run_id=state.run_id,
                )
            except Exception as e:
                logger.warning(
                    f"[{state.run_id}] L0 写入 AI 消息失败: {e}"
                )

    # =========================================================================
    # 工具方法
    # =========================================================================

    def _format_history(self, messages: list) -> str:
        """格式化 L0 对话历史"""
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
    def _has_memories(memories: Dict[str, List]) -> bool:
        """检查是否有非空记忆"""
        return any(
            memories.get(k)
            for k in ["semantic", "episodic", "procedural"]
        )

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """估算文本 token 数（中英文混合的简单估算）

        中文约 1.5 字符/token，英文约 4 字符/token。
        对混合文本取折中值 ~2.5 字符/token。
        """
        if not text:
            return 0
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        tokens = chinese_chars / 1.5 + other_chars / 4.0
        return int(tokens) + 1
