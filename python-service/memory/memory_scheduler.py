"""记忆提取调度器

负责定时扫描 Redis 中累积的未提取对话消息，
当 Token 数超过阈值时触发 L0→L1 记忆提取。

两种触发方式：
1. 定时轮询：后台协程每 N 秒扫描所有活跃会话
2. 实时触发：消息写入后检查当前会话是否超过 Token 阈值
"""

import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class MemoryScheduler:
    """记忆提取调度器"""

    # 默认 Token 阈值（可由配置覆盖）
    DEFAULT_TOKEN_THRESHOLD = 2000
    # Token 估算：中文约 1.5 字符/token
    CHINESE_CHARS_PER_TOKEN = 1.5
    ENGLISH_CHARS_PER_TOKEN = 4.0

    def __init__(self, token_threshold: int = None):
        self.token_threshold = token_threshold or self.DEFAULT_TOKEN_THRESHOLD
        self._extractor = None
        self._store = None
        self._last_scan_time: Optional[datetime] = None
        self._running = False

    @property
    def extractor(self):
        """延迟加载 MemoryExtractor"""
        if self._extractor is None:
            from memory.memory_extractor import MemoryExtractor
            self._extractor = MemoryExtractor()
        return self._extractor

    @property
    def store(self):
        """延迟加载 MemoryStore"""
        if self._store is None:
            from memory.memory_store import get_memory_store
            self._store = get_memory_store()
        return self._store

    # =========================================================================
    # Token 估算
    # =========================================================================

    def estimate_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """估算消息列表的 Token 数（混合中英文）"""
        total_text = ""
        for msg in messages:
            total_text += msg.get("content", "") + " "

        chinese_chars = sum(1 for c in total_text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(total_text) - chinese_chars

        tokens = (
            chinese_chars / self.CHINESE_CHARS_PER_TOKEN
            + other_chars / self.ENGLISH_CHARS_PER_TOKEN
        )
        return int(tokens) + 1

    def should_extract(self, messages: List[Dict[str, Any]]) -> bool:
        """判断消息是否达到提取阈值"""
        if not messages:
            return False
        token_count = self.estimate_tokens(messages)
        return token_count >= self.token_threshold

    # =========================================================================
    # 单会话提取
    # =========================================================================

    def extract_and_store(
        self,
        messages: List[Dict[str, Any]],
        user_id: int,
        conversation_id: str,
    ) -> Dict[str, Any]:
        """对单个会话执行提取并存储

        Args:
            messages: 未提取的消息列表
            user_id: 用户 ID
            conversation_id: 会话 ID

        Returns:
            提取结果字典，含 success 等
        """
        if not messages:
            return {"success": False, "reason": "没有消息需要提取"}

        # Token 阈值检查
        token_count = self.estimate_tokens(messages)
        if token_count < self.token_threshold:
            return {
                "success": False,
                "reason": f"Token 数未超过阈值 ({token_count} < {self.token_threshold})",
                "token_count": token_count,
                "message_count": len(messages),
            }

        # LLM 提取
        try:
            extract_result = self.extractor.extract(messages)
        except Exception as e:
            logger.error(f"记忆提取失败 — 用户: {user_id}, 会话: {conversation_id}: {e}")
            return {
                "success": False,
                "reason": f"记忆提取失败: {str(e)}",
                "token_count": token_count,
                "message_count": len(messages),
            }

        # 存储
        filtered_message_ids = extract_result.get("filtered_message_ids", [])
        try:
            result = self.store.store(
                messages=messages,
                extract_result=extract_result,
                user_id=user_id,
                conversation_id=conversation_id,
                total_tokens=token_count,
                filtered_message_ids=filtered_message_ids,
            )
            return result
        except Exception as e:
            logger.error(f"记忆存储失败 — 用户: {user_id}, 会话: {conversation_id}: {e}")
            return {
                "success": False,
                "reason": f"记忆存储失败: {str(e)}",
                "token_count": token_count,
                "message_count": len(messages),
            }

    # =========================================================================
    # 定时扫描
    # =========================================================================

    def scan_all_conversations(self) -> Dict[str, Any]:
        """扫描所有有未提取消息的会话并提取记忆

        由定时任务调用。通过 MySQL 查询找出有 summary_id IS NULL 消息的会话。
        summary_id 天然防止重复提取。

        Returns:
            汇总统计
        """
        from core.mysql_client import user_memory_client

        results = {
            "total_scanned": 0,
            "extracted": 0,
            "skipped": 0,
            "failed": 0,
            "details": [],
        }

        try:
            # MySQL 查询：所有有未提取消息的会话（天然去重）
            conversations = user_memory_client.get_unsummarized_conversations()

            for conv in conversations:
                user_id = conv["user_id"]
                conversation_id = conv["conversation_id"]
                results["total_scanned"] += 1

                try:
                    # 获取未提取的消息
                    messages = user_memory_client.get_unsummarized_messages(
                        user_id, conversation_id
                    )
                    if not messages:
                        results["skipped"] += 1
                        continue

                    # Token 阈值检查
                    if not self.should_extract(messages):
                        results["skipped"] += 1
                        continue

                    # 执行提取（store 内部会回写 summary_id）
                    extract_result = self.extract_and_store(
                        messages=messages,
                        user_id=user_id,
                        conversation_id=conversation_id,
                    )

                    if extract_result.get("success"):
                        results["extracted"] += 1
                    else:
                        results["failed"] += 1

                    results["details"].append({
                        "user_id": user_id,
                        "conversation_id": conversation_id,
                        **extract_result,
                    })

                except Exception as e:
                    logger.error(
                        f"处理会话失败 user={user_id}, conv={conversation_id}: {e}"
                    )
                    results["failed"] += 1

        except Exception as e:
            logger.error(f"扫描会话失败: {e}")
            results["error"] = str(e)

        self._last_scan_time = datetime.now()

        if results["extracted"] > 0 or results["failed"] > 0:
            logger.info(
                f"记忆提取扫描完成 — 扫描: {results['total_scanned']}, "
                f"提取: {results['extracted']}, 跳过: {results['skipped']}, "
                f"失败: {results['failed']}"
            )

        return results

    # =========================================================================
    # 后台调度
    # =========================================================================

    def start_background(self, interval_seconds: int = None):
        """启动后台定时扫描（在独立线程中运行）

        Args:
            interval_seconds: 扫描间隔（秒），默认从配置读取
        """
        import threading

        if self._running:
            return

        if interval_seconds is None:
            from core.config import config
            interval_seconds = config.MEMORY_EXTRACTION_INTERVAL_SEC

        self._running = True

        def _loop():
            while self._running:
                try:
                    self.scan_all_conversations()
                except Exception as e:
                    logger.error(f"后台扫描异常: {e}")
                time.sleep(interval_seconds)

        thread = threading.Thread(target=_loop, daemon=True, name="memory-scheduler")
        thread.start()
        logger.info(f"记忆提取后台调度已启动，间隔 {interval_seconds}s，阈值 {self.token_threshold}tokens")

    def stop(self):
        """停止后台调度"""
        self._running = False
        logger.info("记忆提取后台调度已停止")


# =============================================================================
# 全局单例
# =============================================================================

_global_scheduler: Optional[MemoryScheduler] = None


def get_memory_scheduler() -> MemoryScheduler:
    """获取全局 MemoryScheduler 实例"""
    global _global_scheduler
    if _global_scheduler is None:
        from core.config import config
        _global_scheduler = MemoryScheduler(
            token_threshold=config.MEMORY_EXTRACTION_TOKEN_THRESHOLD,
        )
    return _global_scheduler
