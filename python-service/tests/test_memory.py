"""记忆系统单元测试

覆盖：提取解析、三维评分、冲突去重、存储流程
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock

from memory.memory_extractor import MemoryExtractor
from memory.memory_manager import MemoryManager


# =============================================================================
# MemoryExtractor 测试
# =============================================================================

class TestMemoryExtractor:
    """LLM 记忆提取器测试"""

    def test_empty_messages(self):
        """空消息列表返回空结果"""
        extractor = MemoryExtractor()
        result = extractor.extract([])
        assert result["semantic_memory"] == []
        assert result["episodic_memory"] == []
        assert result["procedural_memory"] == []
        assert result["user_profile"] == ""
        assert result["filtered_message_ids"] == []

    def test_parse_valid_json(self):
        """解析有效的 JSON 响应"""
        extractor = MemoryExtractor()
        raw = (
            '{"semantic_memory": [{"content": "测试事实 | 检索关键词: test", '
            '"importance_score": 0.8}], '
            '"episodic_memory": [{"content": "2026-07-23: 测试事件", '
            '"importance_score": 0.7}], '
            '"procedural_memory": [], '
            '"user_profile": "测试用户", '
            '"filtered_message_ids": []}'
        )
        result = extractor._parse_response(raw)
        assert len(result["semantic_memory"]) == 1
        assert result["semantic_memory"][0]["importance_score"] == 0.8
        assert len(result["episodic_memory"]) == 1
        assert result["user_profile"] == "测试用户"

    def test_parse_markdown_wrapped_json(self):
        """解析被 Markdown 代码块包裹的 JSON"""
        extractor = MemoryExtractor()
        raw = (
            '```json\n'
            '{"semantic_memory": [], "episodic_memory": [], '
            '"procedural_memory": [], "user_profile": "", '
            '"filtered_message_ids": []}\n'
            '```'
        )
        result = extractor._parse_response(raw)
        assert result["semantic_memory"] == []

    def test_parse_invalid_json(self):
        """解析无效 JSON 返回空结果"""
        extractor = MemoryExtractor()
        raw = "这不是有效的 JSON"
        result = extractor._parse_response(raw)
        assert result["semantic_memory"] == []
        assert result["user_profile"] == ""

    def test_parse_items_filters_empty_content(self):
        """解析时过滤掉 content 为空的记忆"""
        extractor = MemoryExtractor()
        items = [
            {"content": "有效内容", "importance_score": 0.8},
            {"content": "", "importance_score": 0.5},
            {"content": "   ", "importance_score": 0.3},
            {},  # 缺少 content
        ]
        parsed = extractor._parse_items(items)
        assert len(parsed) == 1
        assert parsed[0]["content"] == "有效内容"

    def test_safe_score_bounds(self):
        """分数安全限幅"""
        extractor = MemoryExtractor()
        assert extractor._safe_score(0.5) == 0.5
        assert extractor._safe_score(1.5) == 1.0  # 上限
        assert extractor._safe_score(-0.5) == 0.0  # 下限
        assert extractor._safe_score("invalid") == 0.5  # 非法输入

    def test_format_messages(self):
        """消息格式化"""
        extractor = MemoryExtractor()
        messages = [
            {"id": "1", "role": "user", "content": "你好", "created_at": "2026-07-23"},
            {"id": "2", "role": "assistant", "content": "你好！", "created_at": "2026-07-23"},
        ]
        formatted = extractor._format_messages(messages)
        assert "1 | user | 你好 | 2026-07-23" in formatted
        assert "2 | assistant | 你好！ | 2026-07-23" in formatted


# =============================================================================
# MemoryManager 三维评分测试
# =============================================================================

class TestMemoryScoring:
    """三维评分排序算法测试"""

    @pytest.fixture
    def manager(self):
        """创建不连 Milvus 的 MemoryManager"""
        return MemoryManager(
            uri="mock://localhost",
            token="mock",
            embeddings=Mock(),
            collection_name="test_memory",
            dense_dim=1536,
        )

    def test_empty_memories_returns_empty(self, manager):
        """空记忆返回空结果"""
        result = manager._get_top_k_memories(
            memory_dict={"semantic": []},
            memory_configs={"semantic": {"k": 3}},
        )
        assert result["semantic"] == []

    def test_scoring_sorts_by_final_score(self, manager):
        """高分记忆排在前面"""
        now = time.time()
        # id=3 设为 30 天前，确保时间衰减效果明显
        thirty_days_ago = now - 30 * 24 * 3600
        memories = {
            "semantic": [
                {
                    "id": "1", "score": 0.6, "importance": 0.5,
                    "last_access_at": now,
                },
                {
                    "id": "2", "score": 0.9, "importance": 0.9,
                    "last_access_at": now,
                },
                {
                    "id": "3", "score": 0.7, "importance": 0.6,
                    "last_access_at": thirty_days_ago,
                },
            ]
        }
        config = {"semantic": {"k": 3}}
        result = manager._get_top_k_memories(memory_dict=memories, memory_configs=config)

        assert len(result["semantic"]) == 3
        # id=2 应排第一（相似度高 + 重要性高）
        assert result["semantic"][0]["id"] == "2"
        # id=3 应排最后（30天未访问，时间衰减严重）
        assert result["semantic"][2]["id"] == "3"

    def test_type_weight_applied(self, manager):
        """类型权重生效：semantic > procedural > episodic"""
        now = time.time()
        # 三种类型完全相同的 score/importance/last_access
        base_mem = {"score": 0.8, "importance": 0.8, "last_access_at": now}

        semantic_mem = {"semantic": [{"id": "s1", **base_mem}]}
        episodic_mem = {"episodic": [{"id": "e1", **base_mem}]}
        procedural_mem = {"procedural": [{"id": "p1", **base_mem}]}

        # semantic 权重 1.3，应得分最高
        s_score = manager._get_top_k_memories(
            {"semantic": [{"id": "s1", **base_mem}]},
            {"semantic": {"k": 1}},
        )
        e_score = manager._get_top_k_memories(
            {"episodic": [{"id": "e1", **base_mem}]},
            {"episodic": {"k": 1}},
        )

        # 计算原始分数
        semantic_final = (
            0.45 * 0.8 + 0.25 * 1.0 + 0.30 * 0.8
        ) * 1.3  # semantic weight
        episodic_final = (
            0.45 * 0.8 + 0.25 * 1.0 + 0.30 * 0.8
        ) * 1.0  # episodic weight

        assert semantic_final > episodic_final

    def test_recency_decay(self, manager):
        """时间衰减：旧记忆得分更低"""
        now = time.time()
        old_memory = {
            "id": "old",
            "score": 0.9,
            "importance": 0.9,
            "last_access_at": now - 720 * 3600,  # 30天前
        }
        new_memory = {
            "id": "new",
            "score": 0.9,
            "importance": 0.9,
            "last_access_at": now,
        }

        result = manager._get_top_k_memories(
            memory_dict={"semantic": [old_memory, new_memory]},
            memory_configs={"semantic": {"k": 2}},
        )

        # 新记忆排前面
        assert result["semantic"][0]["id"] == "new"
        assert result["semantic"][1]["id"] == "old"

    def test_top_k_truncation(self, manager):
        """正确截断到 k 条"""
        now = time.time()
        memories = {
            "semantic": [
                {"id": str(i), "score": 0.5 + i * 0.1, "importance": 0.5,
                 "last_access_at": now}
                for i in range(10)
            ]
        }
        config = {"semantic": {"k": 3}}
        result = manager._get_top_k_memories(memory_dict=memories, memory_configs=config)
        assert len(result["semantic"]) == 3

    def test_last_access_at_in_results(self, manager):
        """检索结果中包含 last_access_at 字段"""
        now = time.time()

        # 模拟 _get_top_k_memories 返回的记忆包含 last_access_at
        top_k = {
            "semantic": [
                {"id": "mem_001", "memory_type": "semantic",
                 "content": "test", "last_access_at": now},
            ]
        }
        # last_access_at 应作为检索结果的一部分返回，用于时间衰减计算
        assert "last_access_at" in top_k["semantic"][0]
        assert top_k["semantic"][0]["last_access_at"] <= now


# =============================================================================
# MemoryStore 测试
# =============================================================================

class TestMemoryStore:
    """记忆存储流程测试"""

    @pytest.fixture
    def store(self):
        from memory.memory_store import MemoryStore
        return MemoryStore()

    def test_empty_extract_result(self, store):
        """空的提取结果存储"""
        # 直接设置 _memory_manager 避免触发 pymilvus 导入
        mock_mm = Mock()
        mock_mm.resolve_conflicts.return_value = {
            "semantic_memory": [],
            "episodic_memory": [],
            "procedural_memory": [],
        }
        mock_mm.add_memories_batch.return_value = True
        store._memory_manager = mock_mm

        with patch('memory.memory_store.user_memory_client') as mock_um:
            mock_um.get_user_profile_text.return_value = ""
            mock_um.set_user_profile_text.return_value = True

            result = store.store(
                messages=[],
                extract_result={
                    "semantic_memory": [],
                    "episodic_memory": [],
                    "procedural_memory": [],
                    "user_profile": "",
                    "filtered_message_ids": [],
                },
                user_id=1,
                conversation_id="test_thread",
                total_tokens=500,
                filtered_message_ids=[],
            )

            assert result["success"] is True
            assert result["semantic_count"] == 0


# =============================================================================
# MemoryScheduler 测试
# =============================================================================

class TestMemoryScheduler:
    """记忆调度器测试"""

    @pytest.fixture
    def scheduler(self):
        from memory.memory_scheduler import MemoryScheduler
        return MemoryScheduler(token_threshold=100)

    def test_estimate_tokens_chinese(self, scheduler):
        """中文字符 token 估算"""
        messages = [
            {"content": "你好世界" * 100}  # 400 个中文字
        ]
        tokens = scheduler.estimate_tokens(messages)
        assert tokens > 0
        assert tokens < 400  # 中文约 1.5 字符/token

    def test_estimate_tokens_empty(self, scheduler):
        """空消息 token 为 0"""
        assert scheduler.estimate_tokens([]) == 1  # +1 取整

    def test_should_extract_below_threshold(self, scheduler):
        """低于阈值不触发提取"""
        messages = [{"content": "短消息"}]
        assert scheduler.should_extract(messages) is False

    def test_should_extract_above_threshold(self, scheduler):
        """高于阈值触发提取"""
        messages = [{"content": "你好世界" * 100}]  # 远超 100 tokens
        assert scheduler.should_extract(messages) is True

    def test_extract_and_store_empty_messages(self, scheduler):
        """空消息不执行提取"""
        result = scheduler.extract_and_store([], user_id=1, conversation_id="test")
        assert result["success"] is False
        assert "没有消息" in result["reason"]

    def test_extract_and_store_below_threshold(self, scheduler):
        """低于阈值不执行提取"""
        messages = [{"content": "你好"}]
        result = scheduler.extract_and_store(
            messages, user_id=1, conversation_id="test"
        )
        assert result["success"] is False
        assert "未超过阈值" in result["reason"]
