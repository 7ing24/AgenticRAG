"""Multi-Agent 编排器 — 替代 ReActAgent 作为 KnowledgeQAAgent 的 complex 分支

流程：Analyze → Plan → Dispatch → Execute(parallel) → Synthesize
Agent 间通过 EventBus（兼黑板）共享中间结果，EventBus 线程安全。
"""

import uuid
import logging
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.llm import llm_service
from engine.events import (
    EventBus, event_bus,
    EventType,
    RunStartedEvent, RunCompletedEvent, RunFailedEvent,
    SubTaskStartedEvent, SubTaskCompletedEvent,
    FindingPublishedEvent, SynthesisCompletedEvent,
)
from engine.agent_registry import agent_registry
from agent.memory_agent import MemoryAgent

logger = logging.getLogger(__name__)


@dataclass
class SubTask:
    """子任务"""
    id: str
    question: str
    worker: str = ""          # agent 名称


@dataclass
class Plan:
    """执行计划"""
    is_decomposable: bool
    sub_tasks: List[SubTask] = field(default_factory=list)
    reasoning: str = ""


class MultiAgentOrchestrator:
    """多 Agent 编排器

    复杂问题入口：
    - 可分解 → 拆解为 sub-tasks，并行分派给 worker，汇总
    - 不可分解 → 委托 ReActAgent 单独执行
    """

    def __init__(self, max_workers: int = 4):
        self.event_bus = event_bus
        self.registry = agent_registry
        self.llm_service = llm_service
        self.memory_agent = MemoryAgent()
        self.max_workers = max_workers
        self._react_agent = None

    @property
    def react_agent(self):
        """延迟加载 ReActAgent"""
        if self._react_agent is None:
            from agent.react_agent import ReActAgent
            self._react_agent = ReActAgent(max_iterations=10)
        return self._react_agent

    # ── Public API ──────────────────────────────────────────────

    def run(
        self,
        question: str,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        conversation_history: str = "",
        trace_id: str = "",
        **kwargs,
    ) -> Dict[str, Any]:
        """执行多 Agent 协作，返回最终答案"""
        run_id = str(uuid.uuid4())
        self_run_id = run_id
        orch_steps: list = []  # 记录 orchestrator 自身的步骤
        collector = kwargs.get("trace_collector")

        try:
            self.event_bus.publish(RunStartedEvent(
                run_id=run_id, goal=f"Multi-Agent: {question[:50]}...", input_data=question,
                trace_id=trace_id
            ))

            # 1. Analyze & Plan
            if collector:
                collector.start_timer("orchestrator_plan")
            plan = self._analyze_and_plan(question)
            if collector:
                plan_latency, plan_start = collector.stop_timer("orchestrator_plan")
                sub_questions = [st.question[:150] for st in plan.sub_tasks]
                collector.record_event(
                    event_type="ORCHESTRATOR_PLAN",
                    phase="REACT",
                    input_data={"question": question[:200]},
                    output_data={"decomposable": plan.is_decomposable,
                                 "sub_task_count": len(plan.sub_tasks),
                                 "reasoning": plan.reasoning[:200],
                                 "sub_questions": sub_questions},
                    agent_name="MultiAgentOrchestrator",
                    latency_ms=plan_latency,
                    event_time=plan_start,
                )

            orch_steps.append({"step_name": "analyze_and_plan", "step_type": "planning", "status": "completed",
                                "output": f"decomposable={plan.is_decomposable}, {plan.reasoning}"})

            if not plan.is_decomposable:
                logger.info(f"[{run_id}] Not decomposable, delegating to ReActAgent")
                worker_result = self._run_single_worker(
                    question=question,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    conversation_history=conversation_history,
                    run_id=run_id,
                    trace_id=trace_id,
                    **kwargs,
                )
                # worker 已包含自己的 run_id, agent_type 等字段
                worker_result["runs"] = [{
                    "run_id": self_run_id,
                    "parent_run_id": None,
                    "agent_type": "agent_orchestrator",
                    "steps": orch_steps,
                }, {
                    "run_id": worker_result.get("run_id", ""),
                    "parent_run_id": self_run_id,
                    "agent_type": worker_result.get("agent_type", "react_agent"),
                    "question": worker_result.get("question", question),
                    "steps": worker_result.get("steps", []),
                }]
                worker_result["trace_id"] = trace_id
                return worker_result

            logger.info(f"[{run_id}] Decomposed into {len(plan.sub_tasks)} sub-tasks: {plan.reasoning}")

            # 2. Execute parallel — 先记录 START（插入时间线正确位置），等执行完再补 END
            if collector:
                collector.start_timer("orchestrator_workers")
                collector.record_event(
                    event_type="ORCHESTRATOR_WORKERS_START",
                    phase="REACT",
                    input_data={"sub_task_count": len(plan.sub_tasks)},
                    output_data={"status": "dispatching"},
                    agent_name="MultiAgentOrchestrator",
                )
            results = self._execute_parallel(
                plan=plan,
                run_id=run_id,
                conversation_id=conversation_id,
                user_id=user_id,
                conversation_history=conversation_history,
                trace_id=trace_id,
                **kwargs,
            )
            if collector:
                workers_latency, workers_start = collector.stop_timer("orchestrator_workers")
                collector.record_event(
                    event_type="ORCHESTRATOR_WORKERS_END",
                    phase="REACT",
                    input_data={"sub_task_count": len(plan.sub_tasks)},
                    output_data={"worker_count": len(results),
                                 "completed": sum(1 for r in results.values()
                                                  if isinstance(r.get("result", {}), dict)
                                                  and not r.get("result", {}).get("error"))},
                    agent_name="MultiAgentOrchestrator",
                    latency_ms=workers_latency,
                    event_time=workers_start,
                )

            # 3. Synthesize
            if collector:
                collector.start_timer("orchestrator_synthesis")
            final_answer = self._synthesize(question, results, run_id, trace_id)
            orch_steps.append({"step_name": "synthesis", "step_type": "synthesis", "status": "completed"})
            if collector:
                syn_latency, syn_start = collector.stop_timer("orchestrator_synthesis")
                worker_answers = []
                for sub_id, entry in results.items():
                    r = entry.get("result", {})
                    st = entry.get("sub_task")
                    worker_answers.append({
                        "worker": sub_id,
                        "question": st.question if st else "",
                        "answer": (r.get("answer", "") if isinstance(r, dict) else str(r))[:100],
                    })
                collector.record_event(
                    event_type="ORCHESTRATOR_SYNTHESIS",
                    phase="GENERATION",
                    input_data={"question": question[:200], "worker_answers": worker_answers},
                    output_data={"answer": final_answer[:200]},
                    agent_name="MultiAgentOrchestrator",
                    model_name="qwen-plus",
                    latency_ms=syn_latency,
                    event_time=syn_start,
                )

            # 4. 统一保存记忆（原始问题 + 最终合成答案，只写一次）
            if conversation_id:
                from types import SimpleNamespace
                memory_state = SimpleNamespace(
                    conversation_id=conversation_id, user_id=user_id, run_id=run_id
                )
                self.memory_agent.save_memory(memory_state, question, final_answer)

            # 5. Build response with runs array
            sources = self._collect_sources(results)
            # orchestrator run entry
            runs = [{
                "run_id": self_run_id,
                "parent_run_id": None,
                "agent_type": "agent_orchestrator",
                "steps": orch_steps,
            }]
            # collect child worker run entries
            for entry in results.values():
                w_result = entry.get("result", {})
                sub_task_obj = entry.get("sub_task")
                if isinstance(w_result, dict):
                    runs.append({
                        "run_id": w_result.get("run_id", ""),
                        "parent_run_id": self_run_id,
                        "agent_type": w_result.get("agent_type", "react_agent"),
                        "question": w_result.get("question", "") or (
                            sub_task_obj.question if sub_task_obj else ""
                        ),
                        "steps": w_result.get("steps", []),
                    })

            response = {
                "answer": final_answer,
                "sources": sources,
                "has_sources": len(sources) > 0,
                "task_type": "knowledge_qa",
                "multi_agent": True,
                "sub_task_count": len(plan.sub_tasks),
                "trace_id": trace_id,
                "run_id": self_run_id,
                "runs": runs,
            }

            self.event_bus.publish(RunCompletedEvent(run_id=run_id, output=response, trace_id=trace_id))
            self.event_bus.clear_run(run_id)
            return response

        except Exception as e:
            logger.error(f"[{run_id}] Multi-agent run failed: {e}", exc_info=True)
            self.event_bus.publish(RunFailedEvent(
                run_id=run_id, error=str(e), error_code="MULTI_AGENT_ERROR",
                trace_id=trace_id
            ))
            self.event_bus.clear_run(run_id)
            return {
                "answer": "抱歉，处理您的问题时遇到了错误，请稍后再试。",
                "sources": [],
                "has_sources": False,
                "task_type": "knowledge_qa",
                "error": True,
                "trace_id": trace_id,
                "runs": [{
                    "run_id": self_run_id,
                    "parent_run_id": None,
                    "agent_type": "agent_orchestrator",
                    "steps": orch_steps,
                }],
            }

    # ── Analyze & Plan ──────────────────────────────────────────

    def _analyze_and_plan(self, question: str) -> Plan:
        """LLM 判断问题是否可分解，生成执行计划"""
        try:
            prompt = (
                "判断以下问题是否可以分解为 2-4 个独立的子问题，分别检索后再汇总。\n\n"
                "【可以分解的标准】\n"
                "- 问题明确要求对比多个事物（如 A vs B、X和Y的区别）\n"
                "- 问题要求分别说明/介绍多个独立概念\n"
                "- 问题包含多个并列的、可独立检索的疑问\n\n"
                "【不可以分解的标准】\n"
                "- 问题是单一的操作方法/解决方案（如'如何处理X'、'为什么Y'）\n"
                "- 问题需要逐步推理且步骤之间强依赖\n"
                "- 问题是定义/概念解释类\n\n"
                f"问题：{question}\n\n"
                "请以 JSON 格式回复：\n"
                '{"decomposable": true/false, "reasoning": "判断理由", '
                '"sub_questions": ["子问题1", "子问题2", ...]}\n\n'
                "只输出 JSON，不要其他内容。"
            )

            result = self.llm_service.generate(prompt)
            if not result:
                return Plan(is_decomposable=False)

            result = result.strip()
            if result.startswith("```"):
                result = result.split("\n", 1)[1] if "\n" in result else result[3:]
                result = result.rsplit("```", 1)[0]

            data = json.loads(result.strip())

            decomposable = data.get("decomposable", False)
            if not decomposable:
                return Plan(is_decomposable=False, reasoning=data.get("reasoning", ""))

            sub_questions = data.get("sub_questions", [])
            if not isinstance(sub_questions, list) or len(sub_questions) <= 1:
                return Plan(is_decomposable=False)

            sub_tasks = [
                SubTask(id=f"sub_{i}", question=q)
                for i, q in enumerate(sub_questions[:4])
            ]

            return Plan(
                is_decomposable=True,
                sub_tasks=sub_tasks,
                reasoning=data.get("reasoning", ""),
            )

        except Exception as e:
            logger.warning(f"[MultiAgentOrchestrator] Plan failed: {e}")
            return Plan(is_decomposable=False)

    # ── Execute ─────────────────────────────────────────────────

    def _run_single_worker(
        self,
        question: str,
        conversation_id: Optional[str],
        user_id: Optional[str],
        conversation_history: str,
        run_id: str,
        trace_id: str = "",
        **kwargs,
    ) -> Dict[str, Any]:
        """不可分解时，委托 ReActAgent 单独执行"""
        return self.react_agent.run(
            question=question,
            conversation_id=conversation_id,
            user_id=user_id,
            conversation_history=conversation_history,
            trace_id=trace_id,
            parent_run_id=run_id,
            **kwargs,
        )

    def _execute_parallel(
        self,
        plan: Plan,
        run_id: str,
        conversation_id: Optional[str],
        user_id: Optional[str],
        conversation_history: str,
        trace_id: str = "",
        **kwargs,
    ) -> Dict[str, Dict[str, Any]]:
        """并行执行所有 sub-tasks，通过 EventBus 共享中间结果"""
        results: Dict[str, Dict[str, Any]] = {}

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(plan.sub_tasks))) as executor:
            futures = {}
            for sub_task in plan.sub_tasks:
                future = executor.submit(
                    self._execute_sub_task,
                    sub_task=sub_task,
                    run_id=run_id,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    conversation_history=conversation_history,
                    trace_id=trace_id,
                    trace_collector=kwargs.get("trace_collector"),
                    worker_label=f"worker-{sub_task.id}",
                )
                futures[future] = sub_task

            for future in as_completed(futures):
                sub_task = futures[future]
                try:
                    result = future.result()
                    results[sub_task.id] = {
                        "sub_task": sub_task,
                        "result": result,
                        "status": "completed",
                    }
                except Exception as e:
                    logger.error(f"[{run_id}] Sub-task '{sub_task.id}' failed: {e}")
                    results[sub_task.id] = {
                        "sub_task": sub_task,
                        "result": {"answer": f"子任务执行失败: {str(e)}", "sources": []},
                        "status": "failed",
                    }

        return results

    def _execute_sub_task(
        self,
        sub_task: SubTask,
        run_id: str,
        conversation_id: Optional[str],
        user_id: Optional[str],
        conversation_history: str,
        trace_id: str = "",
        worker_label: str = "",
        **kwargs,
    ) -> Dict[str, Any]:
        """执行单个 sub-task，结果发布到 EventBus 黑板"""
        logger.info(f"[{run_id}] Executing sub-task '{sub_task.id}': {sub_task.question[:50]}...")

        self.event_bus.publish(SubTaskStartedEvent(
            run_id=run_id,
            sub_task_id=sub_task.id,
            sub_task_name=sub_task.question[:80],
            worker="ReActAgent",
            trace_id=trace_id,
        ))

        t0 = __import__("time").time()

        # 使用 ReActAgent 作为 worker，跳过记忆保存（由 Orchestrator 统一处理）
        worker_result = self.react_agent.run(
            question=sub_task.question,
            conversation_id=conversation_id,
            user_id=user_id,
            conversation_history=conversation_history,
            skip_memory=True,
            trace_id=trace_id,
            parent_run_id=run_id,
            worker_label=worker_label,
            **kwargs,
        )

        duration_ms = (__import__("time").time() - t0) * 1000

        # 发布结果到 EventBus 黑板，供其他 worker / synthesizer 读取
        self.event_bus.publish(FindingPublishedEvent(
            run_id=run_id,
            key=sub_task.id,
            value={
                "question": sub_task.question,
                "answer": worker_result.get("answer", ""),
                "sources": worker_result.get("sources", []),
            },
            publisher="ReActAgent",
            trace_id=trace_id,
        ))

        self.event_bus.publish(SubTaskCompletedEvent(
            run_id=run_id,
            sub_task_id=sub_task.id,
            sub_task_name=sub_task.question[:80],
            worker="ReActAgent",
            result=worker_result,
            duration_ms=duration_ms,
            trace_id=trace_id,
        ))

        return worker_result

    # ── Synthesize ──────────────────────────────────────────────

    def _synthesize(
        self,
        original_question: str,
        results: Dict[str, Dict[str, Any]],
        run_id: str,
        trace_id: str = "",
    ) -> str:
        """LLM 汇总所有 sub-task 结果，生成最终答案"""
        # 收集所有 sub-answers
        parts = []
        for task_id, entry in results.items():
            sub_task = entry["sub_task"]
            answer = entry["result"].get("answer", "") if isinstance(entry["result"], dict) else str(entry["result"])
            parts.append(f"【{sub_task.question}】\n{answer}")

        sub_answers_text = "\n\n".join(parts)

        try:
            prompt = (
                "你是一个知识问答汇总专家。请基于以下分步检索结果，回答用户的原始问题。\n\n"
                f"原始问题：{original_question}\n\n"
                "分步检索结果：\n"
                f"{sub_answers_text}\n\n"
                "要求：\n"
                "- 综合所有子问题的答案，给出结构化的完整回答\n"
                "- 如果子问题之间有矛盾，请明确指出\n"
                "- 保留引用的来源信息\n"
                "- 使用与原始问题相同的语言回复"
            )

            result = self.llm_service.generate(prompt)
            if result:
                self.event_bus.publish(SynthesisCompletedEvent(
                    run_id=run_id,
                    answer=result,
                    sub_task_count=len(results),
                    trace_id=trace_id,
                ))
                return result.strip()

        except Exception as e:
            logger.error(f"[{run_id}] Synthesis failed: {e}")

        # fallback：简单拼接
        fallback = "\n\n".join([
            entry["result"].get("answer", "")
            if isinstance(entry["result"], dict)
            else str(entry["result"])
            for entry in results.values()
        ])

        self.event_bus.publish(SynthesisCompletedEvent(
            run_id=run_id,
            answer=fallback,
            sub_task_count=len(results),
            trace_id=trace_id,
        ))
        return fallback

    # ── Helpers ─────────────────────────────────────────────────

    def _collect_sources(self, results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """从所有 sub-task 结果中收集引用来源（按 doc_id 去重）"""
        sources = []
        seen_ids = set()
        for entry in results.values():
            result = entry.get("result", {})
            if not isinstance(result, dict):
                continue
            for src in result.get("sources", []):
                doc_id = src.get("doc_id")
                if doc_id and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    sources.append(src)
        return sources


# 模块级单例
multi_agent_orchestrator = MultiAgentOrchestrator()
