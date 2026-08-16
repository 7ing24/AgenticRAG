import redis
import json
import time
import logging
from core.config import config

logger = logging.getLogger(__name__)

class RedisClient:
    """Redis 客户端，用于会话记忆存储"""

    def __init__(self):
        """初始化Redis连接池"""
        self.pool = redis.ConnectionPool(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            password=config.REDIS_PASSWORD if config.REDIS_PASSWORD else None,
            db=config.REDIS_DB,
            decode_responses=True,
            max_connections=20
        )
        self.client = redis.Redis(connection_pool=self.pool)
        logger.info(f"Redis client initialized: {config.REDIS_HOST}:{config.REDIS_PORT}")

    def add_message(self, conversation_id: str, role: str, content: str,
                    user_id: str = None):
        """追加消息到会话"""
        key = f"conversation:{conversation_id}:messages"
        message = json.dumps({
            "role": role,
            "content": content,
            "timestamp": time.time()
        }, ensure_ascii=False)
        self.client.rpush(key, message)
        self.client.expire(key, 86400)  # 24小时过期
        # 记录用户关联
        if user_id:
            user_key = f"conversation:{conversation_id}:user_id"
            self.client.setex(user_key, 86400, str(user_id))
        logger.debug(f"Added message to conversation {conversation_id}: {role}")

    def get_messages(self, conversation_id: str, limit: int = 10) -> list:
        """获取最近 N 条消息"""
        key = f"conversation:{conversation_id}:messages"
        messages = self.client.lrange(key, -limit, -1)
        return [json.loads(m) for m in messages]

    def get_all_messages(self, conversation_id: str) -> list:
        """获取所有消息"""
        key = f"conversation:{conversation_id}:messages"
        messages = self.client.lrange(key, 0, -1)
        return [json.loads(m) for m in messages]

    def get_message_count(self, conversation_id: str) -> int:
        """获取消息总数"""
        key = f"conversation:{conversation_id}:messages"
        return self.client.llen(key)

    def _trim_conversation(self, conversation_id: str, keep_recent: int = 10):
        """裁剪会话，只保留最近 N 条消息"""
        key = f"conversation:{conversation_id}:messages"
        total = self.client.llen(key)
        if total > keep_recent:
            trim_count = total - keep_recent
            self.client.ltrim(key, trim_count, -1)
            logger.info(
                f"Trimmed conversation {conversation_id}: "
                f"removed {trim_count} messages, kept {keep_recent}"
            )

    def clear_conversation(self, conversation_id: str):
        """清空会话"""
        key = f"conversation:{conversation_id}:messages"
        summary_key = f"conversation:{conversation_id}:summary"
        user_key = f"conversation:{conversation_id}:user_id"
        self.client.delete(key)
        self.client.delete(summary_key)
        self.client.delete(user_key)
        logger.info(f"Cleared conversation {conversation_id}")

    def get_summary(self, conversation_id: str) -> str:
        """获取会话摘要"""
        summary_key = f"conversation:{conversation_id}:summary"
        return self.client.get(summary_key)

    def set_summary(self, conversation_id: str, summary: str, expire: int = 3600):
        """设置会话摘要"""
        summary_key = f"conversation:{conversation_id}:summary"
        self.client.setex(summary_key, expire, summary)
        logger.debug(f"Set summary for conversation {conversation_id}")

    def get_compressed_count(self, conversation_id: str) -> int:
        """获取已被压缩的消息数（用于增量重压缩判断）"""
        key = f"conversation:{conversation_id}:compressed_count"
        val = self.client.get(key)
        return int(val) if val else 0

    def set_compressed_count(self, conversation_id: str, count: int, expire: int = 3600):
        """设置已被压缩的消息数"""
        key = f"conversation:{conversation_id}:compressed_count"
        self.client.setex(key, expire, str(count))

# 创建全局实例
redis_client = RedisClient()
