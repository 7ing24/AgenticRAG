"""ReAct LLM 输出解析器 — 将 Thought/Action/Final Answer 文本解析为结构化数据"""

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ParsedReActOutput:
    """ReAct LLM 输出的结构化解析结果"""
    raw_text: str
    thought: Optional[str] = None
    action: Optional[str] = None           # 原始 action 字符串: "tool_name(k=v, k=v)"
    tool_name: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    is_final_answer: bool = False
    final_answer: Optional[str] = None
    is_valid: bool = False
    parse_error: Optional[str] = None


class ReActParser:
    """解析 LLM 的 ReAct 格式输出"""

    # 多行匹配：从 Thought: 到下一个关键字之前
    THOUGHT_PATTERN = re.compile(
        r'Thought:\s*(.+?)(?=\n(?:Action:|Final Answer:|Observation:|$)|\Z)',
        re.DOTALL | re.IGNORECASE
    )
    # Action: tool_name(params)
    ACTION_PATTERN = re.compile(
        r'Action:\s*(\w+)\((.*?)\)',
        re.DOTALL | re.IGNORECASE
    )
    # Final Answer: 之后到末尾的所有内容
    FINAL_ANSWER_PATTERN = re.compile(
        r'Final Answer:\s*(.+)',
        re.DOTALL | re.IGNORECASE
    )

    def parse(self, text: str) -> ParsedReActOutput:
        """解析 LLM 输出，返回结构化结果"""
        if not text or not text.strip():
            return ParsedReActOutput(
                raw_text=text or "",
                is_valid=False,
                parse_error="Empty LLM output"
            )

        result = ParsedReActOutput(raw_text=text)

        # 优先匹配 Final Answer（终止状态）
        fa_match = self.FINAL_ANSWER_PATTERN.search(text)
        if fa_match:
            result.is_final_answer = True
            result.final_answer = fa_match.group(1).strip()
            result.is_valid = True
            thought = self._extract_thought(text)
            if thought:
                result.thought = thought
            logger.info(f"[ReActParser] Parsed Final Answer ({len(result.final_answer)} chars)")
            return result

        # 匹配 Thought
        thought = self._extract_thought(text)
        if thought:
            result.thought = thought

        # 匹配 Action
        action_match = self.ACTION_PATTERN.search(text)
        if action_match:
            result.action = action_match.group(0).strip()
            result.tool_name = action_match.group(1)
            try:
                result.parameters = self._parse_parameters(action_match.group(2))
            except Exception as e:
                result.parse_error = f"Parameter parsing failed: {e}"
                logger.warning(f"[ReActParser] {result.parse_error}")
                return result
            result.is_valid = True
            logger.info(f"[ReActParser] Parsed Action: {result.tool_name}({result.parameters})")
            return result

        # 既没 Final Answer 也没 Action — 可能是 LLM 直接给了答案
        if not result.is_valid:
            if len(text.strip()) > 20:
                # 长文本可能是没有格式标记的答案，当作 Final Answer
                result.is_final_answer = True
                result.final_answer = text.strip()
                result.is_valid = True
                logger.info("[ReActParser] No ReAct tags found, treating as Final Answer")
            else:
                result.parse_error = "No Action or Final Answer found"
                logger.warning(f"[ReActParser] {result.parse_error}: {text[:100]}")

        return result

    def _extract_thought(self, text: str) -> Optional[str]:
        """提取 Thought 部分"""
        match = self.THOUGHT_PATTERN.search(text)
        return match.group(1).strip() if match else None

    def _parse_parameters(self, params_str: str) -> Dict[str, Any]:
        """
        解析参数字符串: key="value", key=5, key=true, key=0.7

        支持:
        - 双引号字符串: key="hello world"
        - 单引号字符串: key='hello world'
        - 整数: key=5
        - 浮点: key=0.7
        - 布尔: key=true / key=false
        - 无引号标识符: key=value
        """
        params = {}
        if not params_str or not params_str.strip():
            return params

        # 匹配 key=value 对
        pattern = re.compile(
            r'(\w+)\s*=\s*'
            r'(?:'
            r'"([^"]*)"'          # 双引号字符串
            r'|\'([^\']*)\''       # 单引号字符串
            r'|(\[[^\]]*\])'       # 数组 [...]
            r'|(\{[^\}]*\})'       # 对象 {...}
            r'|(true|false)'       # 布尔
            r'|([^,\s)]+)'        # 无引号值
            r')',
            re.IGNORECASE
        )

        for match in pattern.finditer(params_str):
            key = match.group(1)
            groups = match.groups()[1:]  # 跳过 match.group(1)

            if groups[0] is not None:          # 双引号字符串
                value = groups[0]
            elif groups[1] is not None:        # 单引号字符串
                value = groups[1]
            elif groups[2] is not None:        # 数组
                value = groups[2]  # 保留字符串形式
            elif groups[3] is not None:        # 对象
                value = groups[3]  # 保留字符串形式
            elif groups[4] is not None:        # 布尔
                value = groups[4].lower() == 'true'
            elif groups[5] is not None:        # 无引号值
                token = groups[5]
                if token.lower() in ('true', 'false'):
                    value = token.lower() == 'true'
                else:
                    try:
                        value = int(token)
                    except ValueError:
                        try:
                            value = float(token)
                        except ValueError:
                            value = token
            else:
                continue

            params[key] = value

        return params
