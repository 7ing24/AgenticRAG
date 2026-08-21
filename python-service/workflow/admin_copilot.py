from typing import Dict, Any, Optional, Generator
from types import SimpleNamespace
from core.mysql_client import mysql_client
from service.ops import ops_agent
from agent.memory_agent import MemoryAgent
from engine.trace_collector import TraceCollector
import logging
import json

logger = logging.getLogger(__name__)


class AdminCopilotAgent:
    """管理助手Agent - 专门处理管理端运营分析的工作流"""

    def __init__(self):
        self.ops_agent = ops_agent
        self.memory_agent = MemoryAgent()
        self.admin_operations = {
            "stats": "统计分析",
            "knowledge_inspection": "知识巡检",
            "knowledge_gap": "知识缺口分析（P4-3）",
            "unanswered_analysis": "未命中分析",
            "user_activity": "用户活跃度分析",
            "full_ops_report": "完整运营报告（P4-3）",
            "hot_questions": "热门问题日报/周报",
            "knowledge_growth": "知识库增长趋势",
            "agent_success_rate": "Agent成功率分析",
            "tool_call_failures": "工具调用失败排行",
        }
        self._llm_service = None

    @property
    def llm_service(self):
        """延迟加载 LLM 服务"""
        if self._llm_service is None:
            try:
                from core.llm import LLMService
                self._llm_service = LLMService()
            except Exception as e:
                logger.warning(f"[AdminCopilotAgent] LLM service unavailable: {e}")
                self._llm_service = False
        return self._llm_service if self._llm_service is not False else None

    # 有效的操作类型（对应 _execute_operation 的分支）
    VALID_OPERATIONS = [
        ("stats", "系统统计（文档数、问答次数、用户数、未命中问题数等）"),
        ("knowledge_gap", "知识缺口分析（用户提问但系统未回答的问题）"),
        ("hot_questions", "热门问题排行（高频问题 TOP 榜）"),
        ("knowledge_inspection", "知识巡检（重复文档、低质量片段、过期知识、冷门文档）"),
        ("full_ops_report", "完整运营报告（综合知识缺口、问答趋势、用户活跃度）"),
        ("user_activity", "用户活跃度分析（活跃用户统计）"),
        ("knowledge_growth", "知识库增长趋势（文档和片段的新增情况）"),
        ("agent_success_rate", "Agent 成功率分析"),
        ("tool_call_failures", "工具调用失败排行"),
    ]

    def handle(self, question: str, conversation_id: Optional[str] = None,
               user_id: Optional[str] = None, context: str = "",
               trace_id: str = "",
               **kwargs) -> Dict[str, Any]:
        """
        处理管理助手请求

        Args:
            question: 用户问题
            conversation_id: 会话ID
            user_id: 用户ID
            context: 对话上下文
            **kwargs: 其他参数

        Returns:
            包含answer和sources的字典
        """
        import uuid
        run_id = str(uuid.uuid4())
        logger.info(f"[AdminCopilotAgent] Processing admin request: {question[:50]}...")
        collector = kwargs.get("trace_collector")  # type: Optional[TraceCollector]

        # 如果调用方没传 context，自己加载
        if not context and conversation_id:
            memory_state = SimpleNamespace(
                conversation_id=conversation_id, user_id=user_id,
                original_input=question, run_id="admin_memory"
            )
            context = self.memory_agent.load_memory(memory_state, max_rounds=5, include_l1=False).get("text", "")

        try:
            if collector:
                collector.start_timer("admin_op")
            operation = self._parse_operation(question, context)
            steps = [{"step_name": "parse_operation", "step_type": "intent_parsing",
                      "status": "completed", "output": operation}]
            result = self._execute_operation(operation, question)
            admin_latency, admin_start = collector.stop_timer("admin_op") if collector else (0, "")

            if collector:
                collector.record_event(
                    event_type="ADMIN_OPERATION",
                    phase="GENERATION",
                    input_data={"question": question, "operation": operation},
                    output_data={"answer": result.get("answer", "")},
                    agent_name="AdminCopilotAgent",
                    latency_ms=admin_latency,
                    event_time=admin_start,
                )

            result["trace_id"] = trace_id
            result["run_id"] = run_id
            result["runs"] = [{"run_id": run_id, "parent_run_id": None,
                               "agent_type": "admin_copilot", "steps": steps}]
            return result

        except Exception as e:
            logger.error(f"[AdminCopilotAgent] Error: {e}", exc_info=True)
            return {
                "answer": f"抱歉，处理管理请求时出错：{str(e)}",
                "sources": [],
                "has_sources": False,
                "task_type": "admin_copilot",
                "error": True,
                "trace_id": trace_id, "run_id": run_id,
                "runs": [{"run_id": run_id, "parent_run_id": None,
                          "agent_type": "admin_copilot", "steps": []}]
            }

    def handle_stream(self, question: str, conversation_id: Optional[str] = None,
                     user_id: Optional[str] = None, context: str = "",
                     **kwargs) -> Generator[str, None, None]:
        """流式处理管理助手请求"""
        logger.info(f"[AdminCopilotAgent] Stream admin request: {question[:50]}...")

        # 如果调用方没传 context，自己加载
        if not context and conversation_id:
            memory_state = SimpleNamespace(
                conversation_id=conversation_id, user_id=user_id,
                original_input=question, run_id="admin_stream_memory"
            )
            context = self.memory_agent.load_memory(memory_state, max_rounds=5, include_l1=False).get("text", "")

        try:
            operation = self._parse_operation(question, context)

            result = self.handle(question, conversation_id, user_id, context, **kwargs)
            answer = result.get("answer", "")

            for char in answer:
                yield json.dumps({"type": "token", "content": char})

            yield json.dumps({"type": "end", "content": answer, "task_type": "admin_copilot"})

        except Exception as e:
            logger.error(f"[AdminCopilotAgent] Stream error: {e}", exc_info=True)
            yield json.dumps({"type": "error", "content": str(e)})

    def _parse_operation(self, question: str, context: str = "") -> str:
        """解析操作类型：LLM 优先，关键词 fallback"""
        # 如果是跟进请求，从上下文推断上一轮操作
        if any(phrase in question for phrase in ["重新回答", "再列", "换格式", "每行", "重新列", "重列", "再回答", "换个方式"]):
            prev_op = self._infer_operation_from_context(context)
            if prev_op:
                logger.info(f"[AdminCopilotAgent] Follow-up detected, inferred operation: {prev_op}")
                return prev_op

        # 1. LLM 优先判断
        operation = self._classify_operation_with_llm(question)
        if operation:
            return operation

        # 2. 关键词 fallback
        return self._keyword_parse(question)

    def _classify_operation_with_llm(self, question: str) -> Optional[str]:
        """LLM 判断操作类型，失败返回 None"""
        try:
            llm = self.llm_service
            if not llm:
                return None

            op_list = "\n".join([f"- {op}: {desc}" for op, desc in self.VALID_OPERATIONS])
            prompt = f"""判断以下管理操作请求属于哪类操作。

用户请求：{question}

可选操作类型：
{op_list}

只回复操作类型的英文标识（如 stats、hot_questions），不要其他内容。"""

            result = llm.generate(prompt)
            if result:
                result = result.strip().lower()
                for op, _ in self.VALID_OPERATIONS:
                    if op in result:
                        logger.info(f"[AdminCopilotAgent] LLM classified operation: {op}")
                        return op
        except Exception as e:
            logger.warning(f"[AdminCopilotAgent] LLM classification failed: {e}")
        return None

    def _keyword_parse(self, question: str) -> str:
        """关键词 fallback（具体词优先于宽泛词）"""
        lower_question = question.lower()

        if any(kw in lower_question for kw in ["热门问题", "问题排行", "top问题", "常见问题"]):
            return "hot_questions"
        if any(kw in lower_question for kw in ["知识缺口", "缺口", "未命中", "知识缺口分析"]):
            return "knowledge_gap"
        if any(kw in lower_question for kw in ["运营报告", "完整报告", "全报告", "运营分析"]):
            return "full_ops_report"
        if any(kw in lower_question for kw in ["用户", "活跃度", "活跃用户"]):
            return "user_activity"
        if any(kw in lower_question for kw in ["知识库增长", "文档增长", "增长趋势", "新增文档"]):
            return "knowledge_growth"
        if any(kw in lower_question for kw in ["成功率", "失败率", "agent成功", "运行成功"]):
            return "agent_success_rate"
        if any(kw in lower_question for kw in ["工具调用", "工具失败", "工具错误", "工具排行"]):
            return "tool_call_failures"
        if any(kw in lower_question for kw in ["知识", "文档", "巡检", "检查", "质量"]):
            return "knowledge_inspection"
        if any(kw in lower_question for kw in ["统计", "报表", "数据", "分析", "多少", "数量"]):
            return "stats"

        return "stats"

    def _infer_operation_from_context(self, context: str) -> Optional[str]:
        """从对话上下文中推断上一轮的操作类型"""
        if not context:
            return None

        # 提取上下文中用户说的内容
        user_lines = []
        for line in context.split("\n"):
            if line.startswith("user:") or line.startswith("用户:"):
                user_lines.append(line.split(":", 1)[-1].strip())

        # 从最近一条用户消息开始，反向查找匹配的操作关键词
        keyword_to_operation = [
            (["热门问题", "问题排行", "top问题", "常见问题"], "hot_questions"),
            (["知识缺口", "缺口", "未命中"], "knowledge_gap"),
            (["运营报告", "完整报告", "全报告"], "full_ops_report"),
            (["活跃度", "活跃用户", "用户活动"], "user_activity"),
            (["知识库增长", "文档增长", "增长趋势"], "knowledge_growth"),
            (["成功率", "失败率", "agent成功"], "agent_success_rate"),
            (["工具调用", "工具失败", "工具错误"], "tool_call_failures"),
            (["统计", "报表", "仪表盘"], "stats"),
            (["知识", "文档", "巡检", "检查", "质量"], "knowledge_inspection"),
        ]

        for user_text in reversed(user_lines):
            for keywords, operation in keyword_to_operation:
                if any(kw in user_text for kw in keywords):
                    return operation

        return None

    def _execute_operation(self, operation: str, question: str) -> Dict[str, Any]:
        """执行管理操作"""
        try:
            if operation == "stats":
                return self._get_stats()
            elif operation == "knowledge_inspection":
                return self._knowledge_inspection()
            elif operation == "knowledge_gap":
                return self._analyze_knowledge_gap()
            elif operation == "user_activity":
                return self._analyze_user_activity()
            elif operation == "full_ops_report":
                return self._generate_full_ops_report()
            elif operation == "hot_questions":
                period = "week" if "周" in question else "day"
                return self._analyze_hot_questions(period)
            elif operation == "knowledge_growth":
                period = "week" if "周" in question else "month"
                return self._analyze_knowledge_growth(period)
            elif operation == "agent_success_rate":
                period = "week" if "周" in question else "month"
                return self._analyze_agent_success_rate(period)
            elif operation == "tool_call_failures":
                return self._analyze_tool_call_failures()
            else:
                return {
                    "answer": "抱歉，我暂时无法处理这类管理请求。",
                    "sources": [],
                    "has_sources": False,
                    "task_type": "admin_copilot"
                }
        except Exception as e:
            logger.error(f"[AdminCopilotAgent] Operation error: {e}", exc_info=True)
            return {
                "answer": f"执行操作时出错：{str(e)}",
                "sources": [],
                "has_sources": False,
                "task_type": "admin_copilot",
                "error": True
            }

    def _get_stats(self) -> Dict[str, Any]:
        """获取统计数据"""
        try:
            doc_count = mysql_client.fetch_one("SELECT COUNT(*) as count FROM knowledge_doc") or {}
            chunk_count = mysql_client.fetch_one("SELECT COUNT(*) as count FROM knowledge_chunk") or {}
            qa_count = mysql_client.fetch_one("SELECT COUNT(*) as count FROM qa_log") or {}
            user_count = mysql_client.fetch_one("SELECT COUNT(*) as count FROM user") or {}
            unanswered_count = mysql_client.fetch_one("SELECT COUNT(*) as count FROM qa_unanswered") or {}

            answer = f"""📊 系统统计信息

━━━━━━━━━━━━━━━━━━━━━━━━━

📚 知识库：
- 文档数量：{doc_count.get('count', 0)}
- 知识片段：{chunk_count.get('count', 0)}

💬 问答系统：
- 总问答次数：{qa_count.get('count', 0)}
- 未命中问题：{unanswered_count.get('count', 0)}

👥 用户管理：
- 注册用户：{user_count.get('count', 0)}

━━━━━━━━━━━━━━━━━━━━━━━━━

💡 提示：
- 说"知识缺口分析"可以查看知识缺口（P4-3功能）
- 说"完整运营报告"可以获取完整分析（P4-3功能）
"""

            return {
                "answer": answer,
                "sources": [],
                "has_sources": False,
                "task_type": "admin_copilot",
                "data": {
                    "doc_count": doc_count.get('count', 0),
                    "chunk_count": chunk_count.get('count', 0),
                    "qa_count": qa_count.get('count', 0),
                    "user_count": user_count.get('count', 0),
                    "unanswered_count": unanswered_count.get('count', 0)
                }
            }
        except Exception as e:
            logger.error(f"[AdminCopilotAgent] Stats error: {e}", exc_info=True)
            return {
                "answer": "获取统计数据失败，请稍后重试。",
                "sources": [],
                "has_sources": False,
                "task_type": "admin_copilot",
                "error": True
            }

    def _knowledge_inspection(self) -> Dict[str, Any]:
        """知识巡检 - 调用InspectionAgent"""
        from agent.inspection_agent import InspectionAgent
        inspection_agent = InspectionAgent()
        return inspection_agent.inspect("full")

    def _analyze_knowledge_gap(self) -> Dict[str, Any]:
        """知识缺口分析 - P4-3功能 - 调用Ops Agent"""
        logger.info("[AdminCopilotAgent] Analyzing knowledge gap via Ops Agent")
        result = self.ops_agent.analyze("knowledge_gap")
        if result.get("success"):
            return {
                "answer": result.get("answer", ""),
                "sources": [],
                "has_sources": False,
                "task_type": "admin_copilot",
                "data": result.get("data", {})
            }
        else:
            return {
                "answer": "知识缺口分析失败：" + str(result.get("error", "")),
                "sources": [],
                "has_sources": False,
                "task_type": "admin_copilot",
                "error": True
            }

    def _analyze_user_activity(self) -> Dict[str, Any]:
        """用户活跃度分析 - 调用Ops Agent"""
        logger.info("[AdminCopilotAgent] Analyzing user activity via Ops Agent")
        result = self.ops_agent.analyze("user_activity")
        if result.get("success"):
            return {
                "answer": result.get("answer", ""),
                "sources": [],
                "has_sources": False,
                "task_type": "admin_copilot",
                "data": result.get("data", {})
            }
        else:
            return {
                "answer": "用户活跃度分析失败：" + str(result.get("error", "")),
                "sources": [],
                "has_sources": False,
                "task_type": "admin_copilot",
                "error": True
            }

    def _generate_full_ops_report(self) -> Dict[str, Any]:
        """生成完整运营报告 - P4-3功能 - 调用Ops Agent"""
        logger.info("[AdminCopilotAgent] Generating full ops report via Ops Agent")
        result = self.ops_agent.analyze("full_report")
        if result.get("success"):
            return {
                "answer": result.get("answer", ""),
                "sources": [],
                "has_sources": False,
                "task_type": "admin_copilot",
                "data": result.get("data", {})
            }
        else:
            return {
                "answer": "完整运营报告生成失败：" + str(result.get("error", "")),
                "sources": [],
                "has_sources": False,
                "task_type": "admin_copilot",
                "error": True
            }

    def _analyze_hot_questions(self, period: str) -> Dict[str, Any]:
        """分析热门问题 - 调用Ops Agent"""
        logger.info(f"[AdminCopilotAgent] Analyzing hot questions, period: {period}")
        result = self.ops_agent.analyze("hot_questions", period=period)
        if result.get("success"):
            return {
                "answer": result.get("answer", ""),
                "sources": [],
                "has_sources": False,
                "task_type": "admin_copilot",
                "data": result.get("data", {})
            }
        else:
            return {
                "answer": "热门问题分析失败：" + str(result.get("error", "")),
                "sources": [],
                "has_sources": False,
                "task_type": "admin_copilot",
                "error": True
            }

    def _analyze_knowledge_growth(self, period: str) -> Dict[str, Any]:
        """分析知识库增长趋势 - 调用Ops Agent"""
        logger.info(f"[AdminCopilotAgent] Analyzing knowledge growth, period: {period}")
        result = self.ops_agent.analyze("knowledge_growth", period=period)
        if result.get("success"):
            return {
                "answer": result.get("answer", ""),
                "sources": [],
                "has_sources": False,
                "task_type": "admin_copilot",
                "data": result.get("data", {})
            }
        else:
            return {
                "answer": "知识库增长趋势分析失败：" + str(result.get("error", "")),
                "sources": [],
                "has_sources": False,
                "task_type": "admin_copilot",
                "error": True
            }

    def _analyze_agent_success_rate(self, period: str) -> Dict[str, Any]:
        """分析Agent成功率 - 调用Ops Agent"""
        logger.info(f"[AdminCopilotAgent] Analyzing agent success rate, period: {period}")
        result = self.ops_agent.analyze("agent_success_rate", period=period)
        if result.get("success"):
            return {
                "answer": result.get("answer", ""),
                "sources": [],
                "has_sources": False,
                "task_type": "admin_copilot",
                "data": result.get("data", {})
            }
        else:
            return {
                "answer": "Agent成功率分析失败：" + str(result.get("error", "")),
                "sources": [],
                "has_sources": False,
                "task_type": "admin_copilot",
                "error": True
            }

    def _analyze_tool_call_failures(self) -> Dict[str, Any]:
        """分析工具调用失败排行 - 调用Ops Agent"""
        logger.info("[AdminCopilotAgent] Analyzing tool call failures")
        result = self.ops_agent.analyze("tool_call_failures")
        if result.get("success"):
            return {
                "answer": result.get("answer", ""),
                "sources": [],
                "has_sources": False,
                "task_type": "admin_copilot",
                "data": result.get("data", {})
            }
        else:
            return {
                "answer": "工具调用失败分析失败：" + str(result.get("error", "")),
                "sources": [],
                "has_sources": False,
                "task_type": "admin_copilot",
                "error": True
            }
