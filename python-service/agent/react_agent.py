"""ReAct Agent — 真正的 ReAct 循环: Observe → Think → Act → Observe → ... → Final Answer"""

import uuid
import time
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

    @staticmethod
    def _safe_str(obj, budget=400):
        """将 tool_result 格式化为可读字符串，总长度控制在 budget 内，保证括号闭合"""
        def _fmt(o, remaining):
            """递归格式化，remaining 为剩余可用的字符预算"""
            if remaining <= 0:
                return "..."
            if isinstance(o, dict):
                parts = []
                for k, v in o.items():
                    if remaining <= 2:
                        parts.append("...")
                        break
                    v_str = _fmt(v, remaining - len(k) - 3)
                    item = f"{k}={v_str}"
                    if len(item) > remaining:
                        parts.append("...")
                        break
                    parts.append(item)
                    remaining -= len(item) + 2  # ", " separator
                return "{" + ", ".join(parts) + "}"
            if isinstance(o, list):
                if not o:
                    return "[]"
                first = _fmt(o[0], remaining - 15)
                total = f"[{first}, ...({len(o)} items)]"
                if len(total) > remaining:
                    return f"[...({len(o)} items)]"
                return total
            if isinstance(o, str):
                if len(o) > 35:
                    return repr(o[:35] + "...")
                return repr(o)
            if isinstance(o, float):
                return f"{o:.3f}"
            s = str(o)
            if len(s) > 50:
                return s[:50] + "..."
            return s
        return _fmt(obj, budget)

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
        conversation_history: str = "",
        context_token_count: int = 0,
        skip_memory: bool = False,
        trace_id: str = "",
        parent_run_id: str = "",
        **kwargs,
    ) -> Dict[str, Any]:
        """同步执行 ReAct 循环，返回最终答案

        Args:
            conversation_history: 对话历史文本，来自 MemoryAgent
            skip_memory: 为 True 时跳过保存记忆和提取偏好（multi-agent 中作为 worker 时使用）
        """
        state = self._create_state(question, conversation_id, user_id,
                                   trace_id=trace_id, parent_run_id=parent_run_id)

        try:
            initial_prompt = self.prompts.build_initial_messages(
                question=question, conversation_history=conversation_history,
                context="", tool_registry=self.tool_registry,
            )
            messages = [{"role": "user", "content": initial_prompt}]

            self.event_bus.publish(RunStartedEvent(
                run_id=state.run_id, goal=state.goal, input_data=question,
                trace_id=state.trace_id
            ))
            state.start()

            trace_collector = kwargs.get("trace_collector")
            worker_label = kwargs.get("worker_label", "")
            return self._run_loop(state, messages, question, conversation_id,
                                  context_token_count=context_token_count,
                                  skip_memory=skip_memory,
                                  trace_collector=trace_collector,
                                  worker_label=worker_label)

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
        conversation_history: str = "",
        **kwargs,
    ) -> Generator[str, None, None]:
        """流式执行 ReAct 循环，yield JSON 事件"""
        collector = kwargs.get("trace_collector")
        state = self._create_state(question, conversation_id, user_id,
                                   trace_id=kwargs.get("trace_id", ""),
                                   parent_run_id=kwargs.get("parent_run_id", ""))

        react_start = time.time() if collector else 0

        try:
            if collector:
                collector.start_timer("react")
            yield json.dumps({"type": "start", "task_type": "react"})

            initial_prompt = self.prompts.build_initial_messages(
                question=question, conversation_history=conversation_history,
                context="", tool_registry=self.tool_registry,
            )
            messages = [{"role": "user", "content": initial_prompt}]

            state.start()
            self.event_bus.publish(RunStartedEvent(
                run_id=state.run_id, goal=state.goal, input_data=question,
                trace_id=state.trace_id
            ))

            # ── ORCHESTRATOR_PLAN（对齐非流式 MultiAgentOrchestrator）──
            if collector:
                collector.record_event(
                    event_type="ORCHESTRATOR_PLAN",
                    phase="REACT",
                    input_data={"question": question},
                    output_data={"decomposable": True, "sub_task_count": 1,
                                 "reasoning": f"ReActAgent 流式循环，最大迭代 {self.max_iterations} 次",
                                 "sub_questions": [question[:150]]},
                    agent_name="ReActAgent",
                )
                collector.record_event(
                    event_type="ORCHESTRATOR_WORKERS_START",
                    phase="REACT",
                    input_data={"sub_task_count": 1},
                    output_data={"status": "dispatching"},
                    agent_name="ReActAgent",
                )

            # ReAct loop with streaming
            last_tool_calls = set()
            tool_call_count = 0
            for iteration in range(self.max_iterations):
                if TerminationCondition.should_terminate(state):
                    logger.info(f"[{state.run_id}] Termination condition met at iteration {iteration}")
                    break

                if collector:
                    collector.start_timer(f"react_iter_{iteration}")

                step = state.add_step(StepType.TOOL_CALL, f"react_iteration_{iteration}")
                step.start()
                yield json.dumps({"type": "step_started", "step_name": step.step_name})

                # 流式 LLM + 状态机：检测到 Final Answer: 前静默缓冲，之后才 yield
                llm_response = ""
                streaming = False
                last_yielded_pos = 0
                for chunk in self._call_llm_stream(messages):
                    llm_response += chunk
                    if not streaming:
                        keyword_idx = llm_response.lower().find("final answer:")
                        if keyword_idx >= 0:
                            streaming = True
                            last_yielded_pos = keyword_idx + len("final answer:")
                            while last_yielded_pos < len(llm_response) and llm_response[last_yielded_pos] in (' ', '\n', '\r'):
                                last_yielded_pos += 1
                    if streaming and len(llm_response) > last_yielded_pos:
                        new_text = llm_response[last_yielded_pos:]
                        yield json.dumps({"type": "token", "content": new_text})
                        last_yielded_pos = len(llm_response)
                parsed = self.parser.parse(llm_response)

                iter_latency, iter_start = collector.stop_timer(f"react_iter_{iteration}") if collector else (0, "")

                if parsed.thought:
                    yield json.dumps({"type": "thought", "content": parsed.thought})

                if parsed.is_final_answer:
                    final_answer = parsed.final_answer
                    step.complete({"thought": parsed.thought,
                                   "final_answer": final_answer})
                    logger.info(f"[{state.run_id}] Iteration {iteration + 1}: "
                                f"Thought → Final Answer ({len(final_answer)} chars)")
                    # 记录 REACT_ITERATION + ANSWER_GENERATED（对齐非流式）
                    if collector:
                        token_usage = self.llm_service.get_last_token_usage()
                        collector.record_event(
                            event_type="REACT_ITERATION",
                            phase="REACT",
                            input_data={"question": question, "iteration": iteration + 1,
                                        "thought": parsed.thought if parsed.thought else ""},
                            output_data={"final_answer": final_answer},
                            agent_name="ReActAgent",
                            model_name="qwen-plus",
                            latency_ms=iter_latency,
                            event_time=iter_start,
                            input_tokens=token_usage.get("input_tokens") if token_usage else None,
                            output_tokens=token_usage.get("output_tokens") if token_usage else None,
                            total_tokens=token_usage.get("total_tokens") if token_usage else None,
                        )
                        collector.record_event(
                            event_type="ANSWER_GENERATED",
                            phase="GENERATION",
                            input_data={"question": question, "iterations": iteration + 1},
                            output_data={"answer": final_answer},
                            agent_name="ReActAgent",
                            model_name="qwen-plus",
                            latency_ms=iter_latency,
                            event_time=iter_start,
                            input_tokens=token_usage.get("input_tokens") if token_usage else None,
                            output_tokens=token_usage.get("output_tokens") if token_usage else None,
                            total_tokens=token_usage.get("total_tokens") if token_usage else None,
                        )
                        collector.record_event(
                            event_type="ORCHESTRATOR_SYNTHESIS",
                            phase="GENERATION",
                            input_data={"question": question},
                            output_data={"answer": final_answer},
                            agent_name="ReActAgent",
                            model_name="qwen-plus",
                            latency_ms=iter_latency,
                            event_time=iter_start,
                        )
                    break

                if parsed.action and parsed.tool_name:
                    tool_call_count += 1
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

                    # 记录 REACT_ITERATION trace 事件（对齐非流式字段）
                    if collector:
                        token_usage = self.llm_service.get_last_token_usage()
                        collector.record_event(
                            event_type="REACT_ITERATION",
                            phase="REACT",
                            input_data={"question": question, "iteration": iteration + 1,
                                        "thought": parsed.thought if parsed.thought else ""},
                            output_data={"action": parsed.tool_name,
                                         "observation": observation},
                            agent_name="ReActAgent",
                            model_name="qwen-plus",
                            latency_ms=iter_latency,
                            event_time=iter_start,
                            input_tokens=token_usage.get("input_tokens") if token_usage else None,
                            output_tokens=token_usage.get("output_tokens") if token_usage else None,
                            total_tokens=token_usage.get("total_tokens") if token_usage else None,
                        )

                    messages.append({"role": "assistant", "content": llm_response})
                    messages.append({"role": "user", "content": f"Observation: {observation}"})
                else:
                    messages.append({"role": "user", "content": (
                        "Please provide either an Action with a tool call "
                        "or your Final Answer."
                    )})
                    step.complete({"thought": parsed.thought, "parse_error": parsed.parse_error})
            else:
                # 达到最大迭代次数，强制生成最终答案
                logger.info(f"[{state.run_id}] Max iterations reached, forcing final answer")
                final_answer = self._force_final_answer(messages, question, state.run_id)
                for char in final_answer:
                    yield json.dumps({"type": "token", "content": char})

            # ── ORCHESTRATOR_WORKERS_END（对齐非流式）──
            if collector:
                collector.record_event(
                    event_type="ORCHESTRATOR_WORKERS_END",
                    phase="REACT",
                    input_data={"sub_task_count": 1},
                    output_data={"completed": tool_call_count},
                    agent_name="ReActAgent",
                )

            # Build and return response（仅在非正常 break 退出时执行）
            sources = self._extract_sources_from_state(state)
            steps = self._extract_steps_from_state(state)
            response = self._build_response(state, final_answer, sources, steps)

            yield json.dumps({"type": "sources", "content": sources})

            # Record REACT_EXECUTED trace event
            if collector:
                react_latency, react_start_time = collector.stop_timer("react")
                collector.record_event(
                    event_type="REACT_EXECUTED",
                    phase="REACT",
                    input_data={"question": question, "iterations": len(steps)},
                    output_data={"answer": final_answer},
                    agent_name="ReActAgent",
                    model_name="qwen-plus",
                    latency_ms=react_latency,
                    event_time=react_start_time,
                )

            yield json.dumps({"type": "end", "content": final_answer,
                               "task_type": "knowledge_qa"})

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
        trace_collector=None,
        worker_label: str = "",
    ) -> Dict[str, Any]:
        """ReAct 主循环"""
        agent_label = f"ReActAgent({worker_label})" if worker_label else "ReActAgent"
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

            # 1. Call LLM（计时器 key 加 worker_label 避免并行时冲突）
            timer_key = f"react_{worker_label}_iter_{iteration}"
            if trace_collector:
                trace_collector.start_timer(timer_key)
            llm_response = self._call_llm(messages)
            parsed = self.parser.parse(llm_response)
            it_latency, it_start = trace_collector.stop_timer(timer_key) if trace_collector else (0, "")

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
                if trace_collector:
                    tu = self.llm_service.get_last_token_usage()
                    trace_collector.record_event(
                        event_type="REACT_ITERATION",
                        phase="REACT",
                        input_data={"question": question, "iteration": iteration + 1,
                                    "thought": parsed.thought if parsed.thought else ""},
                        output_data={"final_answer": final_answer},
                        agent_name=agent_label,
                        model_name="qwen-plus",
                        latency_ms=it_latency,
                        event_time=it_start,
                        input_tokens=tu.get("input_tokens"),
                        output_tokens=tu.get("output_tokens"),
                        total_tokens=tu.get("total_tokens"),
                    )
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
                if trace_collector:
                    tu = self.llm_service.get_last_token_usage()
                    output_data = {
                        "action": parsed.action,
                        "tool": parsed.tool_name,
                        "result": ReActAgent._safe_str(tool_result),
                    }
                    if parsed.tool_name == "knowledge_search" and isinstance(tool_result, dict):
                        tool_docs = tool_result.get("documents", [])
                        tool_scores = tool_result.get("scores", [])
                        trace_chunks = []
                        for i, doc in enumerate(tool_docs):
                            meta = doc.get("metadata", {}) if isinstance(doc, dict) else getattr(doc, "metadata", {})
                            score = tool_scores[i] if i < len(tool_scores) else doc.get("score", 0) if isinstance(doc, dict) else getattr(doc, "score", 0)
                            trace_chunks.append({
                                "doc_id": meta.get("doc_id"),
                                "chunk_index": meta.get("chunk_index") or meta.get("parent_chunk_index", "-"),
                                "page": meta.get("page"),
                                "score": round(score, 3) if isinstance(score, (int, float)) else score,
                            })
                        output_data["chunk_count"] = len(tool_docs)
                        output_data["avg_score"] = round(sum(tool_scores) / len(tool_scores), 3) if tool_scores else 0
                        output_data["chunks"] = trace_chunks
                    trace_collector.record_event(
                        event_type="REACT_ITERATION",
                        phase="REACT",
                        input_data={"question": question, "iteration": iteration + 1,
                                    "thought": parsed.thought if parsed.thought else ""},
                        output_data=output_data,
                        agent_name=agent_label,
                        model_name="qwen-plus",
                        latency_ms=it_latency,
                        event_time=it_start,
                        input_tokens=tu.get("input_tokens"),
                        output_tokens=tu.get("output_tokens"),
                        total_tokens=tu.get("total_tokens"),
                    )

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
                result, token_cb = self.llm_service._dashscope_generate(conversation_text.strip())
                self.llm_service._last_token_callback = token_cb
                return result
            else:
                from core.llm_fallback import fallback_handler
                return f"Final Answer: {fallback_handler.registry.get('generation')}"
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            from core.llm_fallback import fallback_handler
            return f"Final Answer: {fallback_handler.registry.get('generation')}"

    def _call_llm_stream(self, messages: List[Dict[str, str]]) -> Generator[str, None, str]:
        """流式调用 LLM，逐 token yield，返回完整文本"""
        conversation_text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role in ("user", "assistant"):
                conversation_text += f"\n\n{content}"

        full_text = ""
        prompt = conversation_text.strip()
        if self.llm_service.llm:
            try:
                input_tokens = self.llm_service.llm.get_num_tokens(prompt)
                for chunk in self.llm_service.llm.stream(prompt):
                    full_text += chunk
                    yield chunk
                output_tokens = self.llm_service.llm.get_num_tokens(full_text)
                from core.llm import _TokenUsage
                self.llm_service._last_token_callback = _TokenUsage(input_tokens, output_tokens)
            except Exception as e:
                logger.error(f"Stream LLM call failed: {e}")
                from core.llm_fallback import fallback_handler
                full_text = f"Final Answer: {fallback_handler.registry.get('generation')}"
                for char in full_text:
                    yield char
        else:
            from core.llm_fallback import fallback_handler
            full_text = f"Final Answer: {fallback_handler.registry.get('generation')}"
            for char in full_text:
                yield char

        return full_text

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
            from core.llm_fallback import fallback_handler
            return fallback_handler.registry.get("timeout")

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
