"""ReAct Agent — 真正的 ReAct 循环: Observe → Think → Act → Observe → ... → Final Answer"""

import uuid
import logging
import json
from typing import Dict, Any, Optional, List, Generator
from datetime import datetime

from engine.state import AgentState, AgentStatus, AgentStep, StepType, TerminationCondition
from engine.events import (
    EventBus, event_bus,
    RunStartedEvent, RunCompletedEvent, RunFailedEvent,
    StepStartedEvent, StepCompletedEvent, StepFailedEvent,
    ToolCallCompletedEvent, ToolCallFailedEvent,
)
from agent.memory_agent import MemoryAgent
from agent.react_parser import ReActParser, ParsedReActOutput
from agent.react_prompts import ReActPrompts
from core.llm import LLMService, llm_service
from tools.registry import tool_registry

logger = logging.getLogger(__name__)


class ReActAgent:
    """ReAct 循环 Agent — LLM 动态决定调用哪些工具，自主判断何时输出最终答案"""

    # qwen-plus 上下文窗口约 32K，留 ~8K 给输出，24K 给输入
    MAX_CONTEXT_TOKENS = 24000

    def __init__(
        self,
        max_iterations: int = 10,
        timeout_seconds: int = 300,
    ):
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds
        self.llm_service = llm_service
        self.tool_registry = tool_registry
        self.event_bus = event_bus
        self.memory_agent = MemoryAgent()
        self.parser = ReActParser()
        self.prompts = ReActPrompts()

    # ── Public API ──────────────────────────────────────────────

    def run(
        self,
        question: str,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        context: str = "",
        context_token_count: int = 0,
        skip_memory: bool = False,
        trace_id: str = "",
        parent_run_id: str = "",
        **kwargs,
    ) -> Dict[str, Any]:
        """同步执行 ReAct 循环，返回最终答案

        Args:
            skip_memory: 为 True 时跳过保存记忆和提取偏好（multi-agent 中作为 worker 时使用）
        """
        state = self._create_state(question, conversation_id, user_id,
                                   trace_id=trace_id, parent_run_id=parent_run_id)

        try:
            initial_prompt = self.prompts.build_initial_messages(
                question, context, "", self.tool_registry
            )
            messages = [{"role": "user", "content": initial_prompt}]

            self.event_bus.publish(RunStartedEvent(
                run_id=state.run_id, goal=state.goal, input_data=question,
                trace_id=state.trace_id
            ))
            state.start()

            return self._run_loop(state, messages, question, conversation_id,
                                  context_token_count=context_token_count,
                                  skip_memory=skip_memory)

        except Exception as e:
            logger.error(f"[{state.run_id}] ReAct run failed: {e}", exc_info=True)
            state.fail(str(e), "REACT_AGENT_ERROR")
            self.event_bus.publish(RunFailedEvent(
                run_id=state.run_id, error=str(e), error_code="REACT_AGENT_ERROR",
                trace_id=state.trace_id
            ))
            return self._build_error_response(state)

    def run_stream(
        self,
        question: str,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        context: str = "",
        **kwargs,
    ) -> Generator[str, None, None]:
        """流式执行 ReAct 循环，yield JSON 事件"""
        state = self._create_state(question, conversation_id, user_id,
                                   trace_id=trace_id, parent_run_id=parent_run_id)

        try:
            yield json.dumps({"type": "start", "task_type": "react"})

            initial_prompt = self.prompts.build_initial_messages(
                question, context, "", self.tool_registry
            )
            messages = [{"role": "user", "content": initial_prompt}]

            state.start()
            self.event_bus.publish(RunStartedEvent(
                run_id=state.run_id, goal=state.goal, input_data=question,
                trace_id=state.trace_id
            ))

            # ReAct loop with streaming
            last_tool_calls = set()
            for iteration in range(self.max_iterations):
                if TerminationCondition.should_terminate(state):
                    break

                step = state.add_step(StepType.TOOL_CALL, f"react_iteration_{iteration}")
                step.start()
                yield json.dumps({"type": "step_started", "step_name": step.step_name})

                llm_response = self._call_llm(messages)
                parsed = self.parser.parse(llm_response)

                if parsed.thought:
                    yield json.dumps({"type": "thought", "content": parsed.thought})

                if parsed.is_final_answer:
                    final_answer = parsed.final_answer
                    step.complete({"thought": parsed.thought, "final_answer": final_answer})
                    logger.info(f"[{state.run_id}] Iteration {iteration + 1}: "
                                f"Thought → Final Answer ({len(final_answer)} chars)")
                    break

                if parsed.action and parsed.tool_name:
                    logger.info(f"[{state.run_id}] Iteration {iteration + 1}: "
                                f"Thought → Action({parsed.tool_name})")
                    # Loop detection
                    call_key = (parsed.tool_name, json.dumps(parsed.parameters, sort_keys=True))
                    if call_key in last_tool_calls:
                        messages.append({
                            "role": "user",
                            "content": "You just called this exact tool with the same parameters. "
                                       "Try a different approach or provide your Final Answer."
                        })
                        step.complete({"thought": parsed.thought, "loop_detected": True})
                        last_tool_calls = {call_key}  # allow one more attempt after warning
                        continue
                    last_tool_calls.add(call_key)

                    yield json.dumps({
                        "type": "tool_call",
                        "tool_name": parsed.tool_name,
                        "params": parsed.parameters,
                    })

                    tool_result = self._execute_tool(
                        state, step, parsed.tool_name, parsed.parameters
                    )

                    observation = self.prompts.format_observation(parsed.tool_name, tool_result)
                    yield json.dumps({"type": "observation", "content": observation[:300]})

                    messages.append({"role": "assistant", "content": llm_response})
                    messages.append({"role": "user", "content": f"Observation: {observation}"})
                else:
                    messages.append({"role": "user", "content": (
                        "Please provide either an Action with a tool call "
                        "or your Final Answer."
                    )})
                    step.complete({"thought": parsed.thought, "parse_error": parsed.parse_error})
            else:
                final_answer = self._force_final_answer(messages, question, state.run_id)

            # Build and return response
            sources = self._extract_sources_from_state(state)
            steps = self._extract_steps_from_state(state)
            response = self._build_response(state, final_answer, sources, steps)

            yield json.dumps({"type": "sources", "sources": sources})

            # Stream tokens of final answer
            for char in final_answer:
                yield json.dumps({"type": "token", "content": char})

            yield json.dumps({"type": "end", "content": {
                "answer": final_answer, "sources": sources,
                "task_type": "knowledge_qa", "steps": steps
            }})

            state.complete(response)
            self.event_bus.publish(RunCompletedEvent(run_id=state.run_id, output=response,
                                                 trace_id=state.trace_id))
            self.memory_agent.save_memory(state, question, final_answer)

        except Exception as e:
            logger.error(f"[{state.run_id}] ReAct stream failed: {e}", exc_info=True)
            state.fail(str(e), "REACT_AGENT_ERROR")
            yield json.dumps({"type": "error", "content": str(e)})

    # ── Core Loop ───────────────────────────────────────────────

    def _run_loop(
        self,
        state: AgentState,
        messages: List[Dict[str, str]],
        question: str,
        conversation_id: Optional[str],
        context_token_count: int = 0,
        skip_memory: bool = False,
    ) -> Dict[str, Any]:
        """ReAct 主循环"""
        final_answer = None
        last_tool_calls = set()

        # 估算初始 token 数：system prompt + user message + context
        estimated_tokens = context_token_count
        for msg in messages:
            estimated_tokens += MemoryAgent._estimate_tokens(msg.get("content", ""))

        for iteration in range(self.max_iterations):
            if TerminationCondition.should_terminate(state):
                logger.info(f"[{state.run_id}] Termination condition met")
                break

            step = state.add_step(StepType.TOOL_CALL, f"react_iteration_{iteration}")
            step.start()
            self.event_bus.publish(StepStartedEvent(
                run_id=state.run_id, step_id=step.step_id,
                step_name=step.step_name, step_type=step.step_type.value,
                trace_id=state.trace_id
            ))

            # 1. Call LLM
            llm_response = self._call_llm(messages)
            parsed = self.parser.parse(llm_response)

            if parsed.thought:
                state.add_intermediate_conclusion(
                    step_id=step.step_id, conclusion_type="thought",
                    content=parsed.thought, confidence=0.8
                )

            # 2. Final Answer → done
            if parsed.is_final_answer:
                final_answer = parsed.final_answer
                step.step_name = f"final_answer"
                step.step_type = StepType.ANSWER_GENERATION
                step.complete({"thought": parsed.thought, "final_answer": final_answer})
                logger.info(f"[{state.run_id}] Iteration {iteration + 1}: "
                            f"Thought → Final Answer ({len(final_answer)} chars)")
                break

            # 3. Action → execute tool
            if parsed.action and parsed.tool_name:
                logger.info(f"[{state.run_id}] Iteration {iteration + 1}: "
                            f"Thought → Action({parsed.tool_name})")
                step.step_name = f"{parsed.tool_name}"
                step.step_type = StepType.TOOL_CALL
                # Loop detection
                # Loop detection
                call_key = (parsed.tool_name, json.dumps(parsed.parameters, sort_keys=True))
                if call_key in last_tool_calls:
                    messages.append({
                        "role": "user",
                        "content": "You just called this same tool with the same parameters. "
                                   "Try a different approach or give your Final Answer."
                    })
                    step.complete({"thought": parsed.thought, "loop_detected": True})
                    last_tool_calls = {call_key}
                    continue
                last_tool_calls.add(call_key)

                tool_result = self._execute_tool(
                    state, step, parsed.tool_name, parsed.parameters
                )
                observation = self.prompts.format_observation(parsed.tool_name, tool_result)

                # Append to conversation
                messages.append({"role": "assistant", "content": llm_response})
                messages.append({"role": "user", "content": f"Observation: {observation}"})

                # Token 追踪：超出阈值时裁剪最早的 observation 轮次
                estimated_tokens += MemoryAgent._estimate_tokens(llm_response)
                estimated_tokens += MemoryAgent._estimate_tokens(f"Observation: {observation}")
                if estimated_tokens > self.MAX_CONTEXT_TOKENS:
                    self._trim_old_observations(messages)
                    estimated_tokens = sum(
                        MemoryAgent._estimate_tokens(m.get("content", ""))
                        for m in messages
                    )
                    logger.info(f"[{state.run_id}] Trimmed old observations, "
                                f"estimated tokens: {estimated_tokens}")

                step.complete({"thought": parsed.thought, "action": parsed.action})

            else:
                # Parsing failed — prompt LLM to retry
                messages.append({"role": "user", "content": (
                    "Please provide either an Action with one of the available tools, "
                    "or your Final Answer."
                )})
                step.complete({"thought": parsed.thought, "parse_error": parsed.parse_error})
                logger.warning(f"[{state.run_id}] Parse failed: {parsed.parse_error}")

        # 4. Max iterations → force final answer
        if final_answer is None:
            logger.info(f"[{state.run_id}] Max iterations reached, forcing final answer")
            final_answer = self._force_final_answer(messages, question, state.run_id)

        # 5. Build response with steps
        sources = self._extract_sources_from_state(state)
        steps = self._extract_steps_from_state(state)
        response = self._build_response(state, final_answer, sources, steps)

        state.complete(response)
        self.event_bus.publish(RunCompletedEvent(run_id=state.run_id, output=response,
                                                 trace_id=state.trace_id))
        if not skip_memory:
            self.memory_agent.save_memory(state, question, final_answer)
        else:
            logger.info(f"[{state.run_id}] Skipped memory save (multi-agent worker)")

        return response

    # ── LLM ────────────────────────────────────────────────────

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """调用 LLM，将消息列表转换为单次生成请求"""
        # 构建完整的对话文本
        conversation_text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                conversation_text += f"\n\n{content}"
            elif role == "assistant":
                conversation_text += f"\n\n{content}"

        try:
            if self.llm_service.llm:
                from langchain_core.output_parsers import StrOutputParser
                from langchain_core.prompts import PromptTemplate
                chain = PromptTemplate.from_template("{input}") | self.llm_service.llm | StrOutputParser()
                return chain.invoke({"input": conversation_text.strip()})
            else:
                return "Final Answer: 抱歉，AI 服务当前不可用，请稍后再试。"
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"Final Answer: 抱歉，调用 AI 服务时出错：{str(e)}"

    # ── Tool Execution ──────────────────────────────────────────

    def _execute_tool(
        self,
        state: AgentState,
        step: AgentStep,
        tool_name: str,
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """执行工具调用，记录到 state，发布事件"""
        if not self.tool_registry.has_tool(tool_name):
            error_msg = f"Tool '{tool_name}' not found. Available: {list(self.tool_registry.get_all_tools().keys())}"
            return {"error": error_msg}

        try:
            result = self.tool_registry.invoke_tool(
                tool_name, parameters, run_id=state.run_id
            )
            step.tool_call_id = getattr(result, 'tool_call_id', str(uuid.uuid4()))

            state.add_tool_call(
                tool_call_id=step.tool_call_id or str(uuid.uuid4()),
                tool_name=tool_name,
                input_params=parameters,
                output=result,
                status="success",
                duration_ms=step.duration_ms or 0,
            )

            self.event_bus.publish(ToolCallCompletedEvent(
                run_id=state.run_id,
                tool_call_id=step.tool_call_id or "",
                tool_name=tool_name,
                output=result,
                duration_ms=step.duration_ms or 0,
                trace_id=state.trace_id,
            ))

            return result

        except Exception as e:
            logger.error(f"[{state.run_id}] Tool '{tool_name}' failed: {e}")
            self.event_bus.publish(ToolCallFailedEvent(
                run_id=state.run_id,
                tool_call_id=step.tool_call_id or "",
                tool_name=tool_name,
                error=str(e),
                trace_id=state.trace_id,
            ))
            return {"error": f"Tool '{tool_name}' failed: {str(e)}"}

    # ── State & Memory ─────────────────────────────────────────

    def _create_state(
        self, question: str, conversation_id: Optional[str], user_id: Optional[str],
        trace_id: str = "", parent_run_id: str = "",
    ) -> AgentState:
        return AgentState(
            run_id=str(uuid.uuid4()),
            trace_id=trace_id,
            parent_run_id=parent_run_id if parent_run_id else None,
            conversation_id=conversation_id,
            user_id=user_id,
            goal=f"Answer: {question[:50]}...",
            original_input=question,
            max_steps=self.max_iterations * 2,
            timeout_seconds=self.timeout_seconds,
        )

    # ── Helpers ─────────────────────────────────────────────────

    @staticmethod
    def _trim_old_observations(messages: List[Dict[str, str]]):
        """裁剪最早的 assistant+observation 轮次，保留最近 5 轮完整交互

        messages 结构：[user(初始prompt), assistant1, user(obs1), assistant2, user(obs2), ...]
        第 0 条是初始 prompt，不能删。从第 1 条开始，按 assistant + user(observation) 成对裁剪。
        """
        MIN_KEEP_PAIRS = 5
        # 第 0 条是初始 user prompt
        pairs = messages[1:]  # [assistant, user(obs), assistant, user(obs), ...]
        total_pairs = len(pairs) // 2
        if total_pairs <= MIN_KEEP_PAIRS:
            return

        # 保留最近 MIN_KEEP_PAIRS 对
        keep_start = (total_pairs - MIN_KEEP_PAIRS) * 2
        trimmed = messages[:1] + pairs[keep_start:]
        messages.clear()
        messages.extend(trimmed)

    def _force_final_answer(
        self, messages: List[Dict[str, str]], question: str, run_id: str
    ) -> str:
        """达到最大迭代次数时，强制 LLM 生成最终答案"""
        messages.append({
            "role": "user",
            "content": (
                "You have reached the maximum number of steps. "
                "Based on all the information gathered so far, "
                "please provide your Final Answer now.\n\n"
                f"Original question: {question}"
            )
        })
        try:
            response = self._call_llm(messages)
            parsed = self.parser.parse(response)
            return parsed.final_answer or response.strip()
        except Exception as e:
            logger.error(f"[{run_id}] Force final answer failed: {e}")
            return "抱歉，处理您的问题超时，请稍后重试。"

    def _extract_sources_from_state(self, state: AgentState) -> List[Dict[str, Any]]:
        """从 state 的工具调用记录中提取引用来源"""
        sources = []
        seen_ids = set()
        for tc in state.tool_calls:
            if tc.tool_name not in ("knowledge_search", "rerank"):
                continue
            docs = (
                tc.output.get("documents")
                or tc.output.get("reranked_documents")
                or tc.output.get("chunks")
                or []
            )
            for doc in docs:
                if isinstance(doc, dict):
                    metadata = doc.get("metadata", {})
                    doc_id = metadata.get("doc_id")
                else:
                    metadata = getattr(doc, "metadata", {})
                    doc_id = metadata.get("doc_id")

                if doc_id and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    sources.append({
                        "doc_id": doc_id,
                        "doc_name": metadata.get("source", metadata.get("doc_name", "Unknown")),
                        "page": metadata.get("page") or metadata.get("page_number"),
                        "score": metadata.get("score", 0),
                    })
        return sources

    def _build_response(
        self, state: AgentState, answer: str, sources: List[Dict],
        steps: List[Dict] = None
    ) -> Dict[str, Any]:
        return {
            "answer": answer,
            "sources": sources,
            "has_sources": len(sources) > 0,
            "task_type": "knowledge_qa",
            "question": state.original_input or "",
            "iterations": len(state.tool_calls),
            "steps": steps or [],
            "trace_id": state.trace_id,
            "run_id": state.run_id,
            "parent_run_id": state.parent_run_id or None,
            "agent_type": "react_agent",
        }

    def _extract_steps_from_state(self, state: AgentState) -> List[Dict[str, Any]]:
        """从 AgentState 中提取步骤轨迹"""
        steps = []
        for step in state.steps:
            step_info = {
                "step_name": step.step_name,
                "step_type": step.step_type.value if step.step_type else "unknown",
                "status": step.status.value if step.status else "unknown",
            }
            if step.output_data:
                if "final_answer" in step.output_data:
                    step_info["output"] = f"Final Answer ({len(str(step.output_data.get('final_answer', '')))} chars)"
                elif "thought" in step.output_data:
                    step_info["output"] = str(step.output_data.get("thought", ""))[:200]
                elif "action" in step.output_data:
                    step_info["output"] = str(step.output_data.get("action", ""))[:200]
            steps.append(step_info)
        return steps

    def _build_error_response(self, state: AgentState) -> Dict[str, Any]:
        logger.error(f"[{state.run_id}] Error response: {state.error_message} (code: {state.error_code})")
        return {
            "answer": "抱歉，服务暂时不可用，请稍后再试。",
            "sources": [],
            "has_sources": False,
            "error": True,
        }


# 模块级单例
react_agent = ReActAgent()
