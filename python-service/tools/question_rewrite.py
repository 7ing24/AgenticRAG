from typing import Dict, Any
from tools.base import Tool, ToolSchema, SchemaProperty, ToolMetadata
from core.llm import LLMService
from core.config import config


class QuestionRewriteTool(Tool):
    """问题重写工具 — 支持多轮检索的不同改写策略"""

    def __init__(self):
        input_schema = ToolSchema(
            properties={
                "question": SchemaProperty(
                    type="string",
                    description="原始问题",
                    required=True
                ),
                "conversation_context": SchemaProperty(
                    type="string",
                    description="对话上下文（可选）",
                    required=False,
                    default=""
                ),
            },
            type="object"
        )

        output_schema = ToolSchema(
            properties={
                "rewritten_question": SchemaProperty(
                    type="string",
                    description="重写后的问题",
                    required=True
                ),
                "original_question": SchemaProperty(
                    type="string",
                    description="原始问题",
                    required=True
                )
            },
            type="object"
        )

        metadata = ToolMetadata(
            timeout_ms=15000,
            max_retries=2,
            permission="user",
            description="重写用户问题以提高检索效果"
        )

        super().__init__(
            name="question_rewrite",
            description="重写用户问题以提高检索效果",
            input_schema=input_schema,
            output_schema=output_schema,
            metadata=metadata
        )

        self.llm_service = LLMService()

    def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """执行问题重写，根据 retry_round 使用不同策略"""
        question = parameters.get("question")
        conversation_context = parameters.get("conversation_context", "")
        retry_round = int(parameters.get("retry_round", 0))

        if retry_round == 0:
            prompt = self._prompt_round0(question, conversation_context)
        elif retry_round == 1:
            prompt = self._prompt_round1(question)
        else:
            prompt = self._prompt_round2(question)

        try:
            rewritten = self.llm_service.generate(prompt).strip()

            config.logger.info(
                f"Question rewritten (r{retry_round}): "
                f"'{question[:30]}...' -> '{rewritten[:50]}...'"
            )
            return {"rewritten_question": rewritten, "original_question": question}
        except Exception as e:
            config.logger.error(f"Question rewrite failed: {e}")
            return {"rewritten_question": question, "original_question": question}

    # ── 三轮改写 Prompt ──

    @staticmethod
    def _prompt_round0(question: str, context: str) -> str:
        """Round 0: 指代消解 + 关键词提取"""
        return (
            f"你是搜索查询改写专家。请将用户问题改写为适合知识库向量检索的查询。\n\n"
            f"原始问题：{question}\n\n"
            f"对话上下文：{context if context else '无'}\n\n"
            f"要求：\n"
            f"1. 如果问题包含指代词（如'它'、'这个'、'上面'），根据上下文替换为具体对象\n"
            f"2. 【禁止】根据对话上下文改变原始问题的主题或核心概念\n"
            f"3. 对话上下文仅用于指代消解，不用于推断用户意图或改变话题\n"
            f"4. 提取原始问题的核心概念和关键词（2-5个词）\n"
            f"5. 保持查询简洁，控制在25字以内\n"
            f"6. 直接返回改写后的查询，不要添加任何前缀或解释"
        )

    @staticmethod
    def _prompt_round1(question: str) -> str:
        """Round 1: 换表述角度"""
        return (
            f"请将以下检索查询用完全不同的表述重新描述。保持查询意图不变，"
            f"但使用不同的关键词、同义词或换个搜索角度。\n\n"
            f"原始查询：{question}\n\n"
            f"要求：\n"
            f"1. 必须使用与原始查询不同的关键词\n"
            f"2. 可以换个角度描述同一个问题\n"
            f"3. 保持简洁，25字以内\n"
            f"4. 直接返回改写后的查询，不要解释"
        )

    @staticmethod
    def _prompt_round2(question: str) -> str:
        """Round 2+: 关键词提取，宽泛搜索"""
        return (
            f"请从以下查询中提取最核心的2-3个关键词，用空格分隔返回。\n\n"
            f"查询：{question}\n\n"
            f"要求：\n"
            f"1. 只返回关键词，用空格分隔\n"
            f"2. 去掉修饰词，只保留核心概念\n"
            f"3. 不要添加任何解释"
        )
