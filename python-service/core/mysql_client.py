import os
import json
import uuid
import mysql.connector
from mysql.connector import Error, pooling
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class MySQLClient:
    """MySQL 数据库客户端"""
    
    def __init__(self):
        self.host = os.getenv("MYSQL_HOST", "192.168.100.129")
        self.port = int(os.getenv("MYSQL_PORT", "3306"))
        self.database = os.getenv("MYSQL_DATABASE", "ai_knowledge_db")
        self.username = os.getenv("MYSQL_USERNAME", "root")
        self.password = os.getenv("MYSQL_PASSWORD", "Wqj.120224")
        self.connection = None
    
    def connect(self):
        """建立数据库连接"""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.username,
                password=self.password,
                use_unicode=True,
                charset='utf8mb4',
                time_zone='+08:00'
            )
            if self.connection.is_connected():
                logger.info("Successfully connected to MySQL database")
                # 设置会话时区
                cursor = self.connection.cursor()
                cursor.execute("SET time_zone = '+08:00'")
                cursor.close()
        except Error as e:
            logger.error(f"Error connecting to MySQL: {e}")
    
    def disconnect(self):
        """关闭数据库连接"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("MySQL connection closed")
    
    def insert_chunks(self, doc_id: int, chunks: List[Dict[str, Any]],
                      parent_mode: bool = False):
        """批量插入父块到 knowledge_chunk 表（父子块模式）

        父子块模式下，chunks 是父块，含 parent_id 和 chunk_text。
        """
        if not chunks:
            return 0

        if not self.connection or not self.connection.is_connected():
            self.connect()

        try:
            cursor = self.connection.cursor()

            delete_sql = "DELETE FROM knowledge_chunk WHERE doc_id = %s"
            cursor.execute(delete_sql, (doc_id,))

            if parent_mode:
                insert_sql = """
                    INSERT INTO knowledge_chunk (doc_id, parent_id, chunk_text, chunk_index, page_number, source, create_time)
                    VALUES (%s, %s, %s, %s, %s, %s, NOW())
                """
                data = [
                    (doc_id, chunk.get('parent_id', ''), chunk.get('chunk_text', chunk.get('page_content', '')),
                     chunk.get('chunk_index', i), chunk.get('page_number', 1), chunk.get('source', ''))
                    for i, chunk in enumerate(chunks)
                ]
            else:
                insert_sql = """
                    INSERT INTO knowledge_chunk (doc_id, chunk_index, chunk_text, page_number, create_time)
                    VALUES (%s, %s, %s, %s, NOW())
                """
                data = [
                    (doc_id, chunk.get('chunk_index', i), chunk.get('page_content', ''),
                     chunk.get('page_number', 1),)
                    for i, chunk in enumerate(chunks)
                ]

            cursor.executemany(insert_sql, data)
            self.connection.commit()
            inserted_count = cursor.rowcount
            logger.info(f"Inserted {inserted_count} parent chunks for doc_id {doc_id}")
            
            cursor.close()
            return inserted_count
        
        except Error as e:
            logger.error(f"Error inserting chunks: {e}")
            if self.connection:
                self.connection.rollback()
            return 0
    
    def get_parent_chunks_by_ids(self, parent_ids: List[str]) -> List[Dict[str, Any]]:
        """按 parent_id 批量查父块文本"""
        if not parent_ids:
            return []
        if not self.connection or not self.connection.is_connected():
            self.connect()
        try:
            cursor = self.connection.cursor(dictionary=True)
            placeholders = ",".join(["%s"] * len(parent_ids))
            cursor.execute(
                f"SELECT parent_id, chunk_text, doc_id, chunk_index, page_number, source "
                f"FROM knowledge_chunk WHERE parent_id IN ({placeholders})",
                parent_ids,
            )
            rows = cursor.fetchall()
            cursor.close()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error fetching parent chunks: {e}")
            return []

    def get_chunk_count(self, doc_id: int = None) -> int:
        """获取 chunk 数量"""
        if not self.connection or not self.connection.is_connected():
            self.connect()
        
        try:
            cursor = self.connection.cursor()
            
            if doc_id:
                sql = "SELECT COUNT(*) FROM knowledge_chunk WHERE doc_id = %s"
                cursor.execute(sql, (doc_id,))
            else:
                sql = "SELECT COUNT(*) FROM knowledge_chunk"
                cursor.execute(sql)
            
            result = cursor.fetchone()
            cursor.close()
            return result[0] if result else 0
        
        except Error as e:
            logger.error(f"Error getting chunk count: {e}")
            return 0

    def fetch_one(self, sql: str, params: tuple = None) -> Dict[str, Any]:
        """执行查询并返回单行结果"""
        if not self.connection or not self.connection.is_connected():
            self.connect()
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            
            result = cursor.fetchone()
            cursor.close()
            return result
        
        except Error as e:
            logger.error(f"Error fetching one: {e}")
            return None

    def fetch_all(self, sql: str, params: tuple = None) -> List[Dict[str, Any]]:
        """执行查询并返回所有结果"""
        if not self.connection or not self.connection.is_connected():
            self.connect()
        
        try:
            cursor = self.connection.cursor(dictionary=True)
            
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            
            result = cursor.fetchall()
            cursor.close()
            return result
        
        except Error as e:
            logger.error(f"Error fetching all: {e}")
            return []

    def execute(self, sql: str, params: tuple = None) -> int:
        """执行SQL语句（INSERT/UPDATE/DELETE）"""
        if not self.connection or not self.connection.is_connected():
            self.connect()
        
        try:
            cursor = self.connection.cursor()
            
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            
            self.connection.commit()
            affected_rows = cursor.rowcount
            cursor.close()
            return affected_rows
        
        except Error as e:
            logger.error(f"Error executing SQL: {e}")
            if self.connection:
                self.connection.rollback()
            return 0

# 创建全局实例
mysql_client = MySQLClient()


class UserMemoryClient:
    """用户记忆客户端（连接池版本）"""

    _pool = None

    @classmethod
    def get_pool(cls):
        """获取连接池（懒初始化，单例）"""
        if cls._pool is None:
            cls._pool = pooling.MySQLConnectionPool(
                pool_name="user_memory_pool",
                pool_size=10,               # 连接池大小
                pool_reset_session=True,     # 归还连接时重置会话
                host=os.getenv("MYSQL_HOST", "localhost"),
                port=int(os.getenv("MYSQL_PORT", "3306")),
                user=os.getenv("MYSQL_USERNAME", "root"),
                password=os.getenv("MYSQL_PASSWORD", "123456"),
                database=os.getenv("MYSQL_DATABASE", "ai_knowledge_db"),
                charset="utf8mb4",
                autocommit=True
            )
        return cls._pool

    _tables_initialized = False

    @classmethod
    def _ensure_tables(cls):
        """自动创建记忆系统所需的表（幂等）"""
        if cls._tables_initialized:
            return
        conn = cls.get_pool().get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS `raw_conversations` (
                    `id` varchar(50) NOT NULL,
                    `user_id` int NOT NULL,
                    `conversation_id` varchar(100) NOT NULL,
                    `summary_id` varchar(50) DEFAULT NULL,
                    `role` varchar(10) NOT NULL,
                    `content` text NOT NULL,
                    `created_at` datetime DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (`id`),
                    INDEX `idx_user_conv` (`user_id`, `conversation_id`),
                    INDEX `idx_summary_id` (`summary_id`),
                    INDEX `idx_created_at` (`created_at`)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            conn.commit()
            cursor.close()
            cls._tables_initialized = True
        except Exception as e:
            logger.error(f"创建记忆系统表失败: {e}")
        finally:
            conn.close()

    def _get_connection(self):
        """从连接池获取连接"""
        self._ensure_tables()
        return self.get_pool().get_connection()

    # =========================================================================
    # L0 原始对话持久化
    # =========================================================================

    def add_conversation_message(self, user_id: int, conversation_id: str,
                                  role: str, content: str) -> str:
        """写入 L0 消息到 MySQL"""
        msg_id = str(uuid.uuid4())
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO raw_conversations (id, user_id, conversation_id, role, content)
                   VALUES (%s, %s, %s, %s, %s)""",
                (msg_id, user_id, conversation_id, role, content),
            )
            conn.commit()
            cursor.close()
            return msg_id
        finally:
            conn.close()

    def get_unsummarized_messages(self, user_id: int, conversation_id: str
                                   ) -> List[Dict[str, Any]]:
        """获取某个会话中未提取的消息（summary_id IS NULL）"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                """SELECT id, role, content, created_at
                   FROM raw_conversations
                   WHERE user_id = %s AND conversation_id = %s AND summary_id IS NULL
                   ORDER BY created_at ASC""",
                (user_id, conversation_id),
            )
            rows = cursor.fetchall()
            cursor.close()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_unsummarized_conversations(self) -> List[Dict[str, Any]]:
        """获取所有有未提取消息的会话（user_id, conversation_id 去重）"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                """SELECT DISTINCT user_id, conversation_id
                   FROM raw_conversations
                   WHERE summary_id IS NULL"""
            )
            rows = cursor.fetchall()
            cursor.close()
            return [{"user_id": row["user_id"], "conversation_id": row["conversation_id"]}
                    for row in rows]
        finally:
            conn.close()

    def update_summary_id(self, message_ids: List[str], summary_id: str) -> bool:
        """标记消息为已提取"""
        if not message_ids:
            return False
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            placeholders = ",".join(["%s"] * len(message_ids))
            cursor.execute(
                f"UPDATE raw_conversations SET summary_id = %s WHERE id IN ({placeholders})",
                [summary_id] + message_ids,
            )
            conn.commit()
            cursor.close()
            logger.info(f"标记 {len(message_ids)} 条消息 summary_id={summary_id}")
            return True
        except Exception as e:
            logger.error(f"更新 summary_id 失败: {e}")
            return False
        finally:
            conn.close()

    # =========================================================================
    # L2 用户画像
    # =========================================================================

    def get_user_memory(self, user_id: str) -> Dict[str, Any]:
        """获取用户所有记忆"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT memory_key, memory_value FROM user_memory WHERE user_id = %s",
                (user_id,)
            )
            rows = cursor.fetchall()
            cursor.close()
        finally:
            conn.close()  # 归还到连接池，不是真正关闭

        memory = {}
        for row in rows:
            try:
                memory[row["memory_key"]] = json.loads(row["memory_value"])
            except:
                memory[row["memory_key"]] = row["memory_value"]
        return memory

    def update_user_memory(self, user_id: str, key: str, value: Dict[str, Any],
                           source: str = "agent", confidence: float = 0.6):
        """更新用户记忆（加权合并，非简单覆盖）

        对 dict 类型的 value 做逐键合并：
        - 同一维度多次出现 → 提升 confidence（上限 0.95）
        - 新维度 → 初始 confidence 较低
        - 旧维度只在新值中出现才替换，否则保留
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # 读取旧值
            old_value = {}
            old_confidence = 0.0
            cursor.execute(
                "SELECT memory_value, confidence FROM user_memory "
                "WHERE user_id = %s AND memory_key = %s",
                (user_id, key)
            )
            row = cursor.fetchone()
            if row:
                try:
                    old_value = json.loads(row[0]) if isinstance(row[0], str) else row[0]
                except (json.JSONDecodeError, TypeError):
                    old_value = {}
                old_confidence = float(row[1]) if row[1] else 0.0

            # 逐键合并：按值类型选择策略
            merged = {}
            all_keys = set(old_value.keys()) | set(value.keys())
            for k in all_keys:
                new_val = value.get(k)
                old_val = old_value.get(k)

                if new_val is not None and old_val is not None:
                    # 两边都有 → 按类型合并
                    if isinstance(new_val, list) and isinstance(old_val, list):
                        # 列表去重合并：旧项在前，新项追加
                        merged[k] = list(dict.fromkeys(old_val + new_val))
                    elif isinstance(new_val, str) and isinstance(old_val, str):
                        merged[k] = new_val  # 字符串：新替旧
                    else:
                        merged[k] = new_val  # 其他：新值优先
                elif new_val is not None:
                    merged[k] = new_val
                elif old_val is not None:
                    merged[k] = old_val

            # 置信度：新旧有重叠维度 → 累积提升，否则用较高值
            common_keys = set(old_value.keys()) & set(value.keys())
            if common_keys and old_confidence > 0:
                merged_confidence = min(0.95, max(confidence, old_confidence + 0.05))
            else:
                merged_confidence = max(confidence, old_confidence)

            cursor.execute("""
                INSERT INTO user_memory (user_id, memory_key, memory_value, source, confidence)
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    memory_value = VALUES(memory_value),
                    source = VALUES(source),
                    confidence = VALUES(confidence)
            """, (user_id, key, json.dumps(merged, ensure_ascii=False),
                  source, merged_confidence))
            conn.commit()
            cursor.close()
        finally:
            conn.close()  # 归还到连接池

    def batch_get_user_memories(self, user_ids: list) -> Dict[str, Dict]:
        """批量获取多个用户的记忆（减少连接获取次数）"""
        if not user_ids:
            return {}

        conn = self._get_connection()
        try:
            cursor = conn.cursor(dictionary=True)
            placeholders = ",".join(["%s"] * len(user_ids))
            cursor.execute(
                f"SELECT user_id, memory_key, memory_value FROM user_memory WHERE user_id IN ({placeholders})",
                user_ids
            )
            rows = cursor.fetchall()
            cursor.close()
        finally:
            conn.close()

        result = {uid: {} for uid in user_ids}
        for row in rows:
            try:
                result[row["user_id"]][row["memory_key"]] = json.loads(row["memory_value"])
            except:
                result[row["user_id"]][row["memory_key"]] = row["memory_value"]
        return result


    # =========================================================================
    # L2 用户画像专用方法
    # =========================================================================

    def get_user_profile_text(self, user_id: str) -> str:
        """获取用户的自然语言画像（纯文本）

        优先读取 memory_key='profile' 中的 text 字段。
        如果不存在，尝试从旧的 'preferences' key 构造简单的画像文本。

        Args:
            user_id: 用户 ID

        Returns:
            用户画像文本，如果没有画像则返回空字符串
        """
        try:
            all_memory = self.get_user_memory(user_id)
            profile_data = all_memory.get("profile", {})
            if isinstance(profile_data, dict):
                text = profile_data.get("text", "")
                if text:
                    return text
            elif isinstance(profile_data, str):
                return profile_data

            # Fallback: 从旧的 preferences 构造简单画像
            preferences = all_memory.get("preferences", {})
            if isinstance(preferences, dict) and preferences.get("preferences"):
                prefs = preferences["preferences"]
                parts = []
                for key, val in prefs.items():
                    if isinstance(val, list):
                        parts.append(f"{key}: {', '.join(val)}")
                    else:
                        parts.append(f"{key}: {val}")
                if parts:
                    return "用户偏好 — " + "; ".join(parts)

            return ""
        except Exception as e:
            logger.warning(f"获取用户画像文本失败 (user_id={user_id}): {e}")
            return ""

    def set_user_profile_text(self, user_id: str, profile_text: str) -> bool:
        """设置用户的自然语言画像

        将纯文本画像存储到 memory_key='profile' 的 text 字段中。

        Args:
            user_id: 用户 ID
            profile_text: 用户画像文本

        Returns:
            更新成功返回 True
        """
        if not profile_text or not profile_text.strip():
            return False

        try:
            self.update_user_memory(
                user_id,
                "profile",
                {"text": profile_text.strip()},
                source="memory_extraction",
                confidence=0.85,
            )
            logger.info(f"用户画像已更新 (user_id={user_id})")
            return True
        except Exception as e:
            logger.error(f"设置用户画像失败 (user_id={user_id}): {e}")
            return False


# 创建全局实例
user_memory_client = UserMemoryClient()
