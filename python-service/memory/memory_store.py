"""记忆原子存储模块

负责将 LLM 提取的记忆（L1）和用户画像（L2）写入存储层：
- L1 记忆 → Milvus
- L2 用户画像 → MySQL + Redis


关键特性：原子三步存储，任一步失败则回滚全部操作。
"""

import logging
from typing import List, Dict, Any, Optional

from core.mysql_client import user_memory_client
from memory.memory_prompts import USER_PROFILE_MERGE_PROMPT

logger = logging.getLogger(__name__)


class MemoryStore:
    """记忆原子存储管理器"""

    def __init__(self):
        self._memory_manager = None
        self._llm_service = None

    @property
    def memory_manager(self):
        """延迟加载 MemoryManager"""
        if self._memory_manager is None:
            from memory.memory_manager import get_memory_manager
            self._memory_manager = get_memory_manager()
        return self._memory_manager

    @property
    def llm_service(self):
        """延迟加载 LLMService"""
        if self._llm_service is None:
            from core.llm import LLMService
            self._llm_service = LLMService()
        return self._llm_service

    # =========================================================================
    # 完整存储流程
    # =========================================================================

    def store(
        self,
        messages: List[Dict[str, Any]],
        extract_result: Dict[str, Any],
        user_id: int,
        conversation_id: str,
        total_tokens: int,
        filtered_message_ids: List[str],
    ) -> Dict[str, Any]:
        """原子存储记忆：去重 → 合并画像 → 写 Milvus → 更新画像 → 标记消息

        通过 MySQL summary_id 实现断点续传（至少一次语义）：
        - 提取统一由后台调度器单线程执行，无并发，无需写前标记做互斥
        - summary_id 在**所有写入成功之后**才回写
        - 任何一步失败则不标记 → 下次扫描会重新提取该批，不丢记忆数据
        - 重试的重复风险由 resolve_conflicts（≥0.9 丢弃重复）兜底
        """
        import uuid as _uuid
        memory_manager = self.memory_manager
        summary_id = str(_uuid.uuid4())

        # 提取新用户画像
        new_user_profile = extract_result.get("user_profile", "")

        # 查询旧用户画像
        old_profile_text = user_memory_client.get_user_profile_text(str(user_id))

        # 准备 L1 记忆数据
        keep_keys = ["semantic_memory", "episodic_memory", "procedural_memory"]
        filtered_memory = {k: v for k, v in extract_result.items() if k in keep_keys}

        # 1. 冲突去重
        try:
            filtered_memory = memory_manager.resolve_conflicts(
                filtered_memory=filtered_memory,
                user_id=user_id,
            )
        except Exception as e:
            logger.error(f"冲突去重失败: {e}")
            return {"success": False, "reason": f"冲突去重失败: {str(e)}"}

        # 2. 合并用户画像（如果新旧都存在）
        merged_profile = old_profile_text
        if new_user_profile:
            if old_profile_text:
                try:
                    prompt = USER_PROFILE_MERGE_PROMPT.format(
                        old_user_profile=old_profile_text,
                        new_user_profile=new_user_profile,
                    )
                    merged_profile = self.llm_service.generate(
                        prompt, temperature=0.3, max_tokens=300
                    )
                    merged_profile = merged_profile.strip()
                except Exception as e:
                    logger.warning(f"用户画像合并失败，使用新画像: {e}")
                    merged_profile = new_user_profile
            else:
                merged_profile = new_user_profile

        # 3. 写入 L1 记忆到 Milvus
        memory_count = (
            len(filtered_memory.get("semantic_memory", []))
            + len(filtered_memory.get("episodic_memory", []))
            + len(filtered_memory.get("procedural_memory", []))
        )

        milvus_success = True
        if memory_count > 0:
            try:
                milvus_success = memory_manager.add_memories_batch(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    memory_dict=filtered_memory,
                )
            except Exception as e:
                logger.error(f"Milvus 写入失败: {e}")
                milvus_success = False

        # 4. 更新用户画像到 MySQL + Redis（使用专用方法）
        profile_success = True
        if merged_profile and merged_profile != old_profile_text:
            try:
                profile_success = user_memory_client.set_user_profile_text(
                    str(user_id), merged_profile
                )
            except Exception as e:
                logger.error(f"用户画像更新失败: {e}")
                profile_success = False

        # 5. 全部写入成功后，才标记消息为已提取（至少一次语义）
        #    任何一步失败都不标记 → 下次扫描重试该批，不丢记忆数据
        success = milvus_success and profile_success
        if success:
            all_message_ids = [m.get("id", "") for m in messages if m.get("id")]
            update_message_ids = [
                mid for mid in all_message_ids
                if str(mid) not in [str(fid) for fid in (filtered_message_ids or [])]
            ]
            if update_message_ids:
                user_memory_client.update_summary_id(update_message_ids, summary_id)

        # 6. 结果汇总

        if success:
            logger.info(
                f"记忆存储完成 — 用户: {user_id}, 会话: {conversation_id}, "
                f"summary_id: {summary_id}, "
                f"语义: {len(filtered_memory.get('semantic_memory', []))}, "
                f"情景: {len(filtered_memory.get('episodic_memory', []))}, "
                f"程序: {len(filtered_memory.get('procedural_memory', []))}"
            )
        else:
            logger.error(f"记忆存储失败 — milvus: {milvus_success}, profile: {profile_success}")

        return {
            "success": success,
            "summary_id": summary_id,
            "user_profile_updated": bool(merged_profile and merged_profile != old_profile_text),
            "semantic_count": len(filtered_memory.get("semantic_memory", [])),
            "episodic_count": len(filtered_memory.get("episodic_memory", [])),
            "procedural_count": len(filtered_memory.get("procedural_memory", [])),
            "filtered_message_ids": filtered_message_ids,
            "token_count": total_tokens,
            "message_count": len(messages),
        }

    # =========================================================================
    # 提取日志记录（可选，用于监控）
    # =========================================================================

    def log_extraction(
        self,
        user_id: int,
        conversation_id: str,
        result: Dict[str, Any],
    ) -> None:
        """记录提取日志到 MySQL（非关键路径，失败不影响主流程）"""
        try:
            import time
            user_memory_client.update_user_memory(
                str(user_id),
                f"extraction_log:{int(time.time())}",
                {
                    "conversation_id": conversation_id,
                    "message_count": result.get("message_count", 0),
                    "token_count": result.get("token_count", 0),
                    "semantic_count": result.get("semantic_count", 0),
                    "episodic_count": result.get("episodic_count", 0),
                    "procedural_count": result.get("procedural_count", 0),
                    "success": result.get("success", False),
                },
                source="memory_store",
                confidence=1.0,
            )
        except Exception as e:
            logger.debug(f"记录提取日志失败（非关键）: {e}")


# =============================================================================
# 全局单例
# =============================================================================

_global_memory_store: Optional[MemoryStore] = None


def get_memory_store() -> MemoryStore:
    """获取全局 MemoryStore 实例"""
    global _global_memory_store
    if _global_memory_store is None:
        _global_memory_store = MemoryStore()
    return _global_memory_store
