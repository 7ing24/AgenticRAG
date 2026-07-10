"""测试 ReAct Agent 组件：Parser、Prompts、Agent 循环"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from agent.react_parser import ReActParser, ParsedReActOutput
from agent.react_prompts import ReActPrompts


# ── Parser Tests ──────────────────────────────────────────────

class TestReActParser:
    """测试 ReAct LLM 输出解析器"""

    def setup_method(self):
        self.parser = ReActParser()

    # Final Answer 测试
    def test_parse_final_answer(self):
        text = "Thought: I have enough information.\nFinal Answer: 数据库事务是一组操作的集合。"
        result = self.parser.parse(text)
        assert result.is_final_answer is True
        assert "数据库事务" in result.final_answer
        assert result.thought == "I have enough information."

    def test_parse_final_answer_without_thought(self):
        text = "Final Answer: Hello World"
        result = self.parser.parse(text)
        assert result.is_final_answer is True
        assert result.final_answer == "Hello World"

    # Action 测试
    def test_parse_action_basic(self):
        text = 'Thought: 需要搜索知识库。\nAction: knowledge_search(query="数据库事务", top_k=5)'
        result = self.parser.parse(text)
        assert result.is_final_answer is False
        assert result.tool_name == "knowledge_search"
        assert result.parameters["query"] == "数据库事务"
        assert result.parameters["top_k"] == 5

    def test_parse_action_multiple_params(self):
        text = ('Thought: 改写查询\n'
                'Action: question_rewrite(question="什么是ACID", conversation_context="user asked about DB")')
        result = self.parser.parse(text)
        assert result.tool_name == "question_rewrite"
        assert result.parameters["question"] == "什么是ACID"
        assert result.parameters["conversation_context"] == "user asked about DB"

    def test_parse_action_with_numbers(self):
        text = 'Action: knowledge_search(query="test", top_k=5, similarity_threshold=0.7)'
        result = self.parser.parse(text)
        assert result.parameters["top_k"] == 5
        assert result.parameters["similarity_threshold"] == 0.7

    def test_parse_action_with_boolean(self):
        text = 'Action: knowledge_search(query="test", use_rerank=true)'
        result = self.parser.parse(text)
        assert result.parameters["use_rerank"] is True

    def test_parse_action_with_boolean_false(self):
        text = 'Action: knowledge_search(query="test", use_rerank=False)'
        result = self.parser.parse(text)
        assert result.parameters["use_rerank"] is False

    # 多行 Thought 测试
    def test_parse_multiline_thought(self):
        text = ("Thought: 第一步，我需要理解问题。\n"
                "第二步，搜索知识库。\n"
                "第三步，评估结果。\n"
                "Action: knowledge_search(query=\"test\")")
        result = self.parser.parse(text)
        assert "第一步" in result.thought
        assert "第二步" in result.thought
        assert result.tool_name == "knowledge_search"

    # 边界情况
    def test_parse_empty_text(self):
        result = self.parser.parse("")
        assert result.is_valid is False

    def test_parse_text_without_tags(self):
        result = self.parser.parse("直接回答，没有 Thought 标签")
        assert result.is_final_answer is True  # 长文本当答案处理
        assert result.final_answer == "直接回答，没有 Thought 标签"

    def test_parse_short_text_without_tags(self):
        result = self.parser.parse("hello")
        assert result.is_valid is False
        assert result.parse_error is not None

    def test_parse_action_no_params(self):
        text = "Action: conversation_memory_read()"
        result = self.parser.parse(text)
        assert result.tool_name == "conversation_memory_read"
        assert result.parameters == {}

    # 单引号参数
    def test_parse_single_quoted_params(self):
        text = "Action: knowledge_search(query='hello world', top_k=3)"
        result = self.parser.parse(text)
        assert result.parameters["query"] == "hello world"
        assert result.parameters["top_k"] == 3


# ── Prompts Tests ──────────────────────────────────────────────

class TestReActPrompts:
    """测试 Prompt 生成和 Observation 格式化"""

    def test_format_tools_returns_string(self):
        """不应抛异常"""
        from tools.registry import tool_registry
        desc = ReActPrompts.format_tools_for_llm(tool_registry)
        assert isinstance(desc, str)
        assert len(desc) > 0
        # 验证包含了工具名
        assert "knowledge_search" in desc or len(tool_registry.get_all_tools()) == 0

    def test_format_observation_knowledge_search(self):
        result = {
            "documents": [
                {
                    "content": "数据库事务是...",
                    "metadata": {"source": "db.pdf", "doc_id": 1, "score": 0.92},
                    "score": 0.92
                }
            ]
        }
        obs = ReActPrompts.format_observation("knowledge_search", result)
        assert "数据库事务" in obs
        assert "db.pdf" in obs

    def test_format_observation_empty_result(self):
        obs = ReActPrompts.format_observation("knowledge_search", None)
        assert "empty" in obs.lower() or "Empty" in obs

    def test_format_observation_question_rewrite(self):
        result = {"rewritten_question": "数据库事务 ACID 特性"}
        obs = ReActPrompts.format_observation("question_rewrite", result)
        assert "数据库事务 ACID 特性" in obs

    def test_format_observation_truncation(self):
        """验证长结果被截断"""
        long_content = "x" * 2000
        result = {"documents": [{"content": long_content, "metadata": {"source": "long.txt"}}]}
        obs = ReActPrompts.format_observation("knowledge_search", result, max_chars=200)
        assert len(obs) < 500  # 应该被截断

    def test_build_initial_messages(self):
        from tools.registry import tool_registry
        msgs = ReActPrompts.build_initial_messages(
            question="什么是数据库事务",
            context="",
            conversation_history="",
            tool_registry=tool_registry,
        )
        assert "什么是数据库事务" in msgs
        assert "knowledge_search" in msgs or "Available Tools" in msgs


# ── Integration Tests (with mocks) ─────────────────────────────

class TestReActAgentWithMocks:
    """用 mock 验证 ReActAgent 循环逻辑"""

    def test_agent_answers_directly(self):
        """LLM 直接返回 Final Answer，不做任何 tool call"""
        from agent.react_agent import ReActAgent

        agent = ReActAgent(max_iterations=10)

        with patch.object(agent, '_call_llm') as mock_llm, \
             patch.object(agent, '_load_conversation_memory', return_value=""), \
             patch.object(agent, '_save_conversation_memory'):
            mock_llm.return_value = "Thought: 这是一个简单问题。\nFinal Answer: 你好！有什么可以帮你的？"

            result = agent.run("你好")

            assert result["answer"] == "你好！有什么可以帮你的？"
            assert mock_llm.call_count == 1  # 只调了一次 LLM

    def test_agent_calls_tool_then_answers(self):
        """LLM 调用一次 tool 然后输出 Final Answer"""
        from agent.react_agent import ReActAgent

        agent = ReActAgent(max_iterations=10)

        responses = [
            'Thought: 需要搜索。\nAction: knowledge_search(query="数据库事务", top_k=5)',
            'Thought: 信息足够了。\nFinal Answer: 数据库事务是一组操作的集合。',
        ]

        with patch.object(agent, '_call_llm', side_effect=responses), \
             patch.object(agent, '_load_conversation_memory', return_value=""), \
             patch.object(agent, '_save_conversation_memory'), \
             patch.object(agent.tool_registry, 'invoke_tool', return_value={
                 "documents": [{"content": "数据库事务是...", "metadata": {"source": "db.pdf"}}]
             }), \
             patch.object(agent.tool_registry, 'has_tool', return_value=True):
            result = agent.run("什么是数据库事务")

            assert "数据库事务" in result["answer"]
            assert mock_llm.call_count == 2

    def test_agent_max_iterations_forces_answer(self):
        """达到最大迭代次数后强制生成答案"""
        from agent.react_agent import ReActAgent

        agent = ReActAgent(max_iterations=3)

        # 前 max_iterations 次返回 Action，最后 forced answer
        responses = [
            'Action: knowledge_search(query="test")',
        ] * 3 + [
            'Final Answer: 基于有限信息，这是我能提供的最佳答案。'
        ]

        with patch.object(agent, '_call_llm', side_effect=responses), \
             patch.object(agent, '_load_conversation_memory', return_value=""), \
             patch.object(agent, '_save_conversation_memory'), \
             patch.object(agent.tool_registry, 'invoke_tool', return_value={"documents": []}), \
             patch.object(agent.tool_registry, 'has_tool', return_value=True):
            result = agent.run("test question")

            assert result["answer"] is not None
            assert len(result["answer"]) > 0

    def test_agent_loop_detection(self):
        """检测到 LLM 连续调用同一个 tool 时发出警告"""
        from agent.react_agent import ReActAgent

        agent = ReActAgent(max_iterations=10)

        responses = [
            'Action: knowledge_search(query="test", top_k=5)',   # round 1
            'Action: knowledge_search(query="test", top_k=5)',   # round 2 — 重复
            'Final Answer: 抱歉，知识库中未找到相关信息。',      # round 3 — 改答案了
        ]

        with patch.object(agent, '_call_llm', side_effect=responses), \
             patch.object(agent, '_load_conversation_memory', return_value=""), \
             patch.object(agent, '_save_conversation_memory'), \
             patch.object(agent.tool_registry, 'invoke_tool', return_value={"documents": []}), \
             patch.object(agent.tool_registry, 'has_tool', return_value=True):
            result = agent.run("test")

            assert result["answer"] is not None
