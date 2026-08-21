"""模型自主检索（EchoMind 风格）改造的单元测试

覆盖：
- long_term_memory_read 已暴露给模型（REACT_TOOL_NAMES）
- 工具 schema：user_id 非必填 + 描述含调用引导与 k 决策规则
- format_observation 对记忆检索结果的两种渲染（空/有）+ error 容错
- load_memory(include_l1=False) 跳过 L1 预载，保留 L2/L0
- _execute_tool：user_id 注入、单轮限流 1 次、无 user_id 报错
"""

from unittest.mock import patch, MagicMock

import pytest

from agent.react_prompts import ReActPrompts
from agent.memory_agent import MemoryAgent
from agent.react_agent import ReActAgent
from engine.state import AgentState, AgentStep, StepType


class TestToolExposure:
    """工具暴露给模型"""

    def test_long_term_memory_read_in_react_tool_names(self):
        assert "long_term_memory_read" in ReActPrompts.REACT_TOOL_NAMES

    def test_tool_user_id_not_required_and_guidance_in_schema(self):
        from tools.memory_read import LongTermMemoryReadTool

        tool = LongTermMemoryReadTool()
        assert tool.input_schema.properties["user_id"].required is False
        assert "调用时机" in tool.description
        assert "0=不需要" in tool.input_schema.properties["semantic_k"].description


class TestObservationFormatting:
    """记忆检索结果的 observation 渲染"""

    def test_empty_memories(self):
        out = ReActPrompts().format_observation(
            "long_term_memory_read",
            {"memories": {"semantic": [], "episodic": [], "procedural": []}, "total_count": 0},
        )
        assert "未检索到相关长期记忆" in out

    def test_with_memories_renders_content(self):
        memories = {
            "semantic": [{"content": "用户喜欢喝美式咖啡", "memory_type": "semantic"}],
            "episodic": [],
            "procedural": [],
        }
        out = ReActPrompts().format_observation(
            "long_term_memory_read", {"memories": memories, "total_count": 1}
        )
        assert "用户喜欢喝美式咖啡" in out

    def test_error_result_tolerated(self):
        out = ReActPrompts().format_observation("long_term_memory_read", {"error": "x"})
        assert "未检索到相关长期记忆" in out


class TestLoadMemoryGate:
    """load_memory 的 include_l1 门控"""

    def test_include_l1_false_skips_l1_keeps_l2_l0(self):
        ma = MemoryAgent()
        state = AgentState(run_id="t", conversation_id="c", user_id=1, original_input="q")
        with patch.object(ma, "_load_user_profile", return_value="画像"), \
             patch.object(ma, "_load_long_term_memory") as mock_l1, \
             patch("agent.memory_agent.tool_registry") as mock_reg:
            mock_reg.has_tool.return_value = True
            mock_reg.invoke_tool.return_value = {
                "messages": [{"role": "user", "content": "你好"}], "compressed": False
            }
            res = ma.load_memory(state, include_l1=False)
            assert mock_l1.call_count == 0
            assert "[用户画像]" in res["text"]
            assert "[当前会话]" in res["text"]


class TestLoadMemorySplit:
    """load_memory 返回 profile/context 分离（画像进 system，历史进 user）"""

    def test_returns_profile_and_context_fields(self):
        ma = MemoryAgent()
        state = AgentState(run_id="t", conversation_id="c", user_id=1, original_input="q")
        with patch.object(ma, "_load_user_profile", return_value="用户画像文本"), \
             patch.object(ma, "_load_long_term_memory"), \
             patch("agent.memory_agent.tool_registry") as mock_reg:
            mock_reg.has_tool.return_value = True
            mock_reg.invoke_tool.return_value = {
                "messages": [{"role": "user", "content": "你好"}], "compressed": False
            }
            res = ma.load_memory(state, include_l1=False)
            assert res["profile"] == "用户画像文本"
            # context 仅含对话历史，不含画像标记
            assert "[用户画像]" not in res["context"]
            assert "[当前会话]" in res["context"]
            # text 保持旧格式兼容（画像在前）
            assert "[用户画像]" in res["text"]
            assert res["text"].startswith("[用户画像]")


class TestExecuteToolInjection:
    """_execute_tool 对 long_term_memory_read 的注入与限流"""

    @staticmethod
    def _agent():
        ra = ReActAgent.__new__(ReActAgent)
        ra.tool_registry = MagicMock()
        ra.tool_registry.has_tool.return_value = True
        ra.tool_registry.invoke_tool.return_value = {"memories": {}, "total_count": 0}
        ra.event_bus = MagicMock()
        ra.prompts = ReActPrompts()
        return ra

    def test_user_id_injected_from_state(self):
        ra = self._agent()
        state = AgentState(run_id="r", conversation_id="c", user_id="42", original_input="q")
        step = AgentStep(step_id="s1", step_name="t", step_type=StepType.TOOL_CALL)
        step.start()
        ra._execute_tool(state, step, "long_term_memory_read", {"query": "q", "semantic_k": 3})
        injected = ra.tool_registry.invoke_tool.call_args[0][1]
        assert injected["user_id"] == 42

    def test_single_call_per_turn_limit(self):
        ra = self._agent()
        state = AgentState(run_id="r", conversation_id="c", user_id="42", original_input="q")
        state.add_tool_call("id1", "long_term_memory_read", {}, {}, "success", 0)
        step = AgentStep(step_id="s2", step_name="t", step_type=StepType.TOOL_CALL)
        step.start()
        res = ra._execute_tool(state, step, "long_term_memory_read", {"query": "q"})
        assert "已检索过" in res.get("error", "")
        assert ra.tool_registry.invoke_tool.call_count == 0

    def test_missing_user_id_returns_error(self):
        ra = self._agent()
        state = AgentState(run_id="r", conversation_id="c", user_id=None, original_input="q")
        step = AgentStep(step_id="s3", step_name="t", step_type=StepType.TOOL_CALL)
        step.start()
        res = ra._execute_tool(state, step, "long_term_memory_read", {"query": "q"})
        assert "用户ID" in res.get("error", "")
        assert ra.tool_registry.invoke_tool.call_count == 0
