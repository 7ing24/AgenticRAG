"""L0 → L1+L2 记忆提取器

从原始对话（L0）中调用 LLM 一次性提取：
- L1: semantic_memory / episodic_memory / procedural_memory
- L2: user_profile

借鉴 EchoMind 的提取 Prompt 设计，输出 5 字段 JSON。
"""

import json
import logging
from typing import List, Dict, Any, Optional

from core.llm import LLMService
from memory.memory_prompts import MEMORY_EXTRACT_PROMPT

logger = logging.getLogger(__name__)


class MemoryExtractor:
    """从原始对话中提取长期记忆和用户画像"""

    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service or LLMService()

    def extract(
        self,
        messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """从对话消息列表中提取记忆

        Args:
            messages: 消息列表，每项含 id / role / content / created_at

        Returns:
            {
                "semantic_memory": [{"content": str, "importance_score": float}, ...],
                "episodic_memory": [{"content": str, "importance_score": float}, ...],
                "procedural_memory": [{"content": str, "importance_score": float}, ...],
                "user_profile": str,
                "filtered_message_ids": [int, ...]
            }
        """
        if not messages:
            return self._empty_result()

        # 格式化对话文本
        conversation_text = self._format_messages(messages)

        # 调用 LLM 提取
        prompt = MEMORY_EXTRACT_PROMPT.format(conversation_text=conversation_text)
        response = self.llm_service.generate(prompt, temperature=0.3, max_tokens=2000)

        # 解析响应
        return self._parse_response(response)

    def _format_messages(self, messages: List[Dict[str, Any]]) -> str:
        """格式化消息列表为 LLM 输入文本"""
        lines = []
        for msg in messages:
            msg_id = msg.get("id", "")
            role = msg.get("role", "")
            content = msg.get("content", "")
            created_at = msg.get("created_at", "N/A")
            lines.append(f"{msg_id} | {role} | {content} | {created_at}")
        return "\n\n".join(lines)

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """解析 LLM 输出的 JSON，带容错处理"""
        if not response:
            logger.warning("LLM 返回空响应")
            return self._empty_result()

        content = response.strip()

        # 移除 Markdown 代码块包裹
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        try:
            result = json.loads(content.strip())
        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析失败: {e}，原始响应: {content[:200]}...")
            return self._empty_result()

        return {
            "semantic_memory": self._parse_items(result.get("semantic_memory")),
            "episodic_memory": self._parse_items(result.get("episodic_memory")),
            "procedural_memory": self._parse_items(result.get("procedural_memory")),
            "user_profile": result.get("user_profile", "").strip(),
            "filtered_message_ids": result.get("filtered_message_ids", []),
        }

    def _parse_items(self, items) -> List[Dict[str, Any]]:
        """解析并校验记忆条目列表"""
        if not isinstance(items, list):
            return []
        return [
            {
                "content": item["content"].strip(),
                "importance_score": self._safe_score(
                    item.get("importance_score", 0.5)
                ),
            }
            for item in items
            if isinstance(item, dict)
            and item.get("content", "").strip()
        ]

    @staticmethod
    def _safe_score(value) -> float:
        """安全地将重要性分数限制在 0.0-1.0 并保留 1 位小数"""
        try:
            return round(max(0.0, min(1.0, float(value))), 1)
        except (ValueError, TypeError):
            return 0.5

    @staticmethod
    def _empty_result() -> Dict[str, Any]:
        """返回空的提取结果"""
        return {
            "semantic_memory": [],
            "episodic_memory": [],
            "procedural_memory": [],
            "user_profile": "",
            "filtered_message_ids": [],
        }
