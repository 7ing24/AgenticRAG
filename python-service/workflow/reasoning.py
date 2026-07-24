from typing import Dict, Any, Optional, List, Generator
from types import SimpleNamespace
from core.llm import LLMService
from core.vector_store import vector_store
from service.retrieval import RetrievalAgent
from engine.planner import Planner
from tools.registry import tool_registry
from tools.question_rewrite import QuestionRewriteTool
from agent.memory_agent import MemoryAgent
import logging
import json

logger = logging.getLogger(__name__)


class ReasoningAgent:
    """Reasoning Agent - 处理复杂问题的分步推理（Chain-of-Thought）

    策略：分解 → 逐个子问题检索+推理 → 汇总
    """

    def __init__(self):
        self.llm_service = LLMService()
        self.vector_store = vector_store
        self.retrieval_agent = RetrievalAgent()
        self.planner = Planner()
        self.question_rewrite_tool = QuestionRewriteTool()
        self.memory_agent = MemoryAgent()
        self.max_retrieval_rounds = 2  # 每个子问题最多检索重试轮数

    def reason(self, question: str, context: str = "",
               conversation_id: str = None, user_id: str = None) -> Dict[str, Any]:
        """执行推理流程：分解 → 逐个检索+推理 → 汇总"""
        logger.info(f"[ReasoningAgent] Starting reasoning for: {question[:50]}...")

        # 如果调用方没传 context，自己加载
        if not context and conversation_id:
            memory_state = SimpleNamespace(
                conversation_id=conversation_id, user_id=user_id,
                original_input=question, run_id="reasoning_memory"
            )
            context = self.memory_agent.load_memory(memory_state, max_rounds=10).get("text", "")
            logger.info(f"[ReasoningAgent] Self-loaded context, {len(context)} chars")

        try:
            # Step 1：问题分解
            sub_questions = self._decompose_question(question)
            logger.info(f"[ReasoningAgent] Decomposed into {len(sub_questions)} sub-questions")

            # Step 2：逐个子问题检索 + 推理
            reasoning_steps = []
            for sub_q in sub_questions:
                docs = self._retrieve_for_sub_question(sub_q, context)
                sub_answer = self._reason_sub_question(sub_q, docs, context)
                reasoning_steps.append({
                    "sub_question": sub_q,
                    "sources": [
                        {"doc_id": getattr(doc, 'metadata', {}).get("doc_id"),
                         "doc": getattr(doc, 'metadata', {}).get("source", "未知文档")}
                        for doc in docs
                    ],
                    "reasoning": sub_answer
                })

            # Step 3：汇总生成最终答案
            final_answer = self._synthesize_answer(question, reasoning_steps)

            # 合并所有来源（按 doc_id 去重）
            all_sources = []
            seen_ids = set()
            for step in reasoning_steps:
                for src in step["sources"]:
                    doc_id = src.get("doc_id")
                    if doc_id and doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        all_sources.append(src)

            # 统一通过 MemoryAgent 写入会话记忆
            if conversation_id:
                memory_state = SimpleNamespace(
                    conversation_id=conversation_id, user_id=user_id, run_id="reasoning_save"
                )
                self.memory_agent.save_memory(memory_state, question, final_answer)

            return {
                "answer": final_answer,
                "reasoning_steps": reasoning_steps,
                "sources": all_sources,
                "has_sources": len(all_sources) > 0,
                "task_type": "knowledge_qa"
            }

        except Exception as e:
            logger.error(f"[ReasoningAgent] Reasoning failed: {e}", exc_info=True)
            return {
                "answer": "抱歉，处理您的复杂问题时遇到了错误，请稍后再试。",
                "reasoning_steps": [],
                "sources": [],
                "has_sources": False,
                "task_type": "knowledge_qa",
                "error": True
            }

    def reason_stream(
        self,
        question: str,
        context: str = "",
        conversation_id: str = None,
        user_id: str = None,
    ) -> Generator[str, None, None]:
        """流式执行推理流程"""
        logger.info(f"[ReasoningAgent] Stream reasoning for: {question[:50]}...")

        try:
            yield json.dumps({"type": "start", "task_type": "reasoning"})

            # 加载记忆
            if not context and conversation_id:
                memory_state = SimpleNamespace(
                    conversation_id=conversation_id, user_id=user_id,
                original_input=question, run_id="reasoning_memory"
                )
                context = self.memory_agent.load_memory(memory_state, max_rounds=10).get("text", "")

            # Step 1：分解
            yield json.dumps({"type": "step", "step_name": "decompose", "status": "started"})
            sub_questions = self._decompose_question(question)
            yield json.dumps({
                "type": "step", "step_name": "decompose", "status": "completed",
                "sub_questions": sub_questions
            })

            # Step 2：逐个子问题
            reasoning_steps = []
            for i, sub_q in enumerate(sub_questions):
                yield json.dumps({
                    "type": "step", "step_name": f"sub_question_{i+1}",
                    "status": "started", "question": sub_q
                })

                docs = self._retrieve_for_sub_question(sub_q, context)
                sub_answer = self._reason_sub_question(sub_q, docs, context)

                step_result = {
                    "sub_question": sub_q,
                    "sources": [
                        {"doc_id": getattr(doc, 'metadata', {}).get("doc_id"),
                         "doc": getattr(doc, 'metadata', {}).get("source", "未知文档")}
                        for doc in docs
                    ],
                    "reasoning": sub_answer
                }
                reasoning_steps.append(step_result)

                yield json.dumps({
                    "type": "step", "step_name": f"sub_question_{i+1}",
                    "status": "completed", "sub_answer_preview": sub_answer[:200]
                })

            # Step 3：汇总
            yield json.dumps({"type": "step", "step_name": "synthesis", "status": "started"})
            final_answer = self._synthesize_answer(question, reasoning_steps)
            yield json.dumps({"type": "step", "step_name": "synthesis", "status": "completed"})

            # 合并来源
            all_sources = []
            seen_ids = set()
            for step in reasoning_steps:
                for src in step["sources"]:
                    doc_id = src.get("doc_id")
                    if doc_id and doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        all_sources.append(src)

            yield json.dumps({"type": "sources", "sources": all_sources})

            # 流式输出最终答案
            for char in final_answer:
                yield json.dumps({"type": "token", "content": char})

            yield json.dumps({"type": "end", "content": {
                "answer": final_answer,
                "reasoning_steps": [
                    {"sub_question": s["sub_question"], "reasoning": s["reasoning"][:200]}
                    for s in reasoning_steps
                ],
                "sources": all_sources,
                "has_sources": len(all_sources) > 0,
                "task_type": "knowledge_qa"
            }})

            # 写入记忆
            if conversation_id:
                memory_state = SimpleNamespace(
                    conversation_id=conversation_id, user_id=user_id, run_id="reasoning_save"
                )
                self.memory_agent.save_memory(memory_state, question, final_answer)

        except Exception as e:
            logger.error(f"[ReasoningAgent] Stream reasoning failed: {e}", exc_info=True)
            yield json.dumps({"type": "error", "content": str(e)})

    def _decompose_question(self, question: str) -> List[str]:
        """将复杂问题分解为子问题"""
        if not self.llm_service.llm:
            return [question]

        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        prompt = PromptTemplate.from_template(
            """请将以下复杂问题分解为 2-4 个简单的子问题，便于逐个检索和推理。

问题：{question}

请以 JSON 数组格式输出子问题列表，例如：["子问题1", "子问题2"]
只输出 JSON 数组，不要其他内容。"""
        )

        try:
            chain = prompt | self.llm_service.llm | StrOutputParser()
            response = chain.invoke({"question": question}).strip()

            # 处理可能的 markdown 代码块
            if response.startswith("```"):
                response = response.split("\n", 1)[1] if "\n" in response else response[3:]
                response = response.rsplit("```", 1)[0]

            sub_questions = json.loads(response.strip())
            if isinstance(sub_questions, list) and len(sub_questions) > 0:
                return sub_questions[:4]
        except Exception as e:
            logger.warning(f"[ReasoningAgent] Failed to decompose question: {e}")

        return [question]

    def _retrieve_for_sub_question(self, question: str, context: str) -> list:
        """为子问题执行检索，使用完整检索管道（改写 → 混合检索 → 重排），带充分性重试"""
        current_query = question

        for round_num in range(self.max_retrieval_rounds):
            retrieval_result = self.retrieval_agent.retrieve(
                query=current_query,
                conversation_context=context,
                use_rewrite=(round_num == 0),  # 首轮改写，后续轮次用已改写的 query
                use_rerank=True,
                top_k=5,
                similarity_threshold=0.5
            )

            docs = retrieval_result.reranked_documents
            scores = retrieval_result.scores
            logger.info(f"[ReasoningAgent] Sub-q '{question[:30]}...' round {round_num + 1}: "
                        f"retrieved {len(docs)} docs"
                        + (f", avg_score={sum(scores)/len(scores):.2f}" if scores else ""))

            # 充分性判断
            if docs and self.planner.evaluate_retrieval_sufficiency(
                docs, question, scores, score_threshold=0.3
            ).is_sufficient:
                logger.info(f"[ReasoningAgent] Sufficient at round {round_num + 1}")
                return docs

            # 首轮不改写（已在 retrieve 中完成），后续轮次尝试改写
            if round_num > 0 and round_num < self.max_retrieval_rounds - 1:
                try:
                    rewritten = self.question_rewrite_tool.execute({
                        "question": current_query,
                        "conversation_context": context
                    })
                    current_query = rewritten.get("rewritten_question", current_query)
                except Exception:
                    pass

        # 返回最后一轮的结果
        retrieval_result = self.retrieval_agent.retrieve(
            query=current_query,
            conversation_context=context,
            use_rewrite=False,
            use_rerank=True,
            top_k=5,
            similarity_threshold=0.5
        )
        return retrieval_result.reranked_documents

    def _reason_sub_question(self, question: str, docs: list, context: str) -> str:
        """对单个子问题进行推理"""
        if not self.llm_service.llm:
            return "无法推理（LLM 不可用）"

        doc_text = "\n".join([
            getattr(doc, 'page_content', str(doc)) if hasattr(doc, 'page_content')
            else doc.get('content', str(doc)) if isinstance(doc, dict) else str(doc)
            for doc in docs
        ]) if docs else "（无相关文档）"

        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        prompt = PromptTemplate.from_template(
            """基于以下参考资料，回答子问题。

参考资料：
{doc_text}

对话上下文：{context}

子问题：{question}

请给出简洁准确的回答。"""
        )

        chain = prompt | self.llm_service.llm | StrOutputParser()
        return chain.invoke({
            "doc_text": doc_text,
            "context": context or "无",
            "question": question
        }).strip()

    def _synthesize_answer(self, original_question: str, reasoning_steps: list) -> str:
        """汇总推理结果，生成最终答案"""
        if not self.llm_service.llm:
            return "\n\n".join([step["reasoning"] for step in reasoning_steps])

        steps_text = "\n".join([
            f"Q: {step['sub_question']}\nA: {step['reasoning']}"
            for step in reasoning_steps
        ])

        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        prompt = PromptTemplate.from_template(
            """基于以下分步推理结果，回答用户的原始问题。

分步推理：
{steps_text}

原始问题：{original_question}

请给出完整、准确、结构化的回答。"""
        )

        chain = prompt | self.llm_service.llm | StrOutputParser()
        return chain.invoke({
            "steps_text": steps_text,
            "original_question": original_question
        }).strip()
