import os
import logging
from typing import Dict, Any

class ConfigManager:
    """统一配置管理模块"""
    
    def __init__(self):
        """初始化配置管理器"""
        self._load_config()
        self._setup_logging()
    
    def _load_config(self):
        """加载配置项"""
        # AI Embeddings
        self.DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
        # LLM 模型名（DashScope），trace 事件 model_name 与 LLM 调用统一使用
        self.DASHSCOPE_MODEL = os.getenv("DASHSCOPE_MODEL", "qwen-plus")
        # Embedding模型选择: "dashscope" 使用云端API, "local" 使用本地中文模型
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "dashscope")
        # 本地Embedding模型名称（仅当EMBEDDING_MODEL=local时生效）
        self.LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-m3")

        # Milvus Configuration
        self.MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
        self.MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
        self.MILVUS_USER = os.getenv("MILVUS_USER", "")
        self.MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD", "")

        # Redis Configuration
        self.REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
        self.REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
        self.REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
        self.REDIS_DB = int(os.getenv("REDIS_DB", "0"))

        # MySQL Configuration
        self.DB_HOST = os.getenv("MYSQL_HOST", "localhost")
        self.DB_PORT = int(os.getenv("MYSQL_PORT", "3306"))
        self.DB_USER = os.getenv("MYSQL_USERNAME", "root")
        self.DB_PASSWORD = os.getenv("MYSQL_PASSWORD", "123456")
        self.DB_NAME = os.getenv("MYSQL_DATABASE", "ai_knowledge_db")

        # Vector Store Configuration (Milvus)
        self.VECTOR_STORE_COLLECTION_NAME = os.getenv("VECTOR_STORE_COLLECTION_NAME", "ai_knowledge_collection")

        # Milvus 度量类型: L2 / IP / COSINE（默认 L2）
        self.MILVUS_METRIC_TYPE = os.getenv("MILVUS_METRIC_TYPE", "L2").upper()

        # ============================================================
        # Memory System Configuration (三层记忆系统)
        # ============================================================
        # L1 长期记忆集合名称
        self.MEMORY_COLLECTION = os.getenv("MEMORY_COLLECTION", "long_term_memory")
        # L1 记忆 Embedding 维度（text-embedding-v3=1024）
        self.MEMORY_EMBEDDING_DIM = int(os.getenv("MEMORY_EMBEDDING_DIM", "1024"))
        # L0→L1 提取触发 Token 阈值
        self.MEMORY_EXTRACTION_TOKEN_THRESHOLD = int(os.getenv(
            "MEMORY_EXTRACTION_TOKEN_THRESHOLD", "1000"
        ))
        # 定时提取间隔（秒），默认 3 分钟
        self.MEMORY_EXTRACTION_INTERVAL_SEC = int(os.getenv(
            "MEMORY_EXTRACTION_INTERVAL_SEC", "180"
        ))
        # 冲突去重余弦相似度阈值
        self.MEMORY_CONFLICT_THRESHOLD = float(os.getenv(
            "MEMORY_CONFLICT_THRESHOLD", "0.9"
        ))
        # 时间衰减因子（每小时）
        self.MEMORY_DECAY_RATE = float(os.getenv("MEMORY_DECAY_RATE", "0.995"))
        # 三维评分权重
        self.MEMORY_ALPHA = float(os.getenv("MEMORY_ALPHA", "0.45"))  # 语义相似度
        self.MEMORY_BETA = float(os.getenv("MEMORY_BETA", "0.25"))    # 时间衰减
        self.MEMORY_GAMMA = float(os.getenv("MEMORY_GAMMA", "0.30"))  # 重要性
        # L1 检索时各记忆类型的最大召回数
        self.MEMORY_MAX_SEMANTIC_K = int(os.getenv("MEMORY_MAX_SEMANTIC_K", "3"))
        self.MEMORY_MAX_EPISODIC_K = int(os.getenv("MEMORY_MAX_EPISODIC_K", "2"))
        self.MEMORY_MAX_PROCEDURAL_K = int(os.getenv("MEMORY_MAX_PROCEDURAL_K", "2"))
        # 记忆提取用的 LLM 模型（成本较低的小模型即可）
        self.MEMORY_EXTRACTION_MODEL = os.getenv(
            "MEMORY_EXTRACTION_MODEL", "qwen-turbo"
        )

        # Rerank Configuration
        self.RERANKER_TYPE = os.getenv("RERANKER_TYPE", "bge")
        # gte rerank 模型名（DashScope），默认 gte-rerank-v2（v1 多数账号未开通返回 403）
        self.RERANK_MODEL = os.getenv("RERANK_MODEL", "gte-rerank-v2")
        self.COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")

        # Semantic cache Configuration（知识问答答案缓存：Redis 精确 + Milvus 语义）
        self.QA_CACHE_ENABLED = os.getenv("QA_CACHE_ENABLED", "true").lower() == "true"
        self.QA_CACHE_TTL = int(os.getenv("QA_CACHE_TTL", "86400"))

        # Text Chunking Configuration
        self.CHUNK_STRATEGY = os.getenv("CHUNK_STRATEGY", "semantic")
        self.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
        self.CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
        self.MIN_CHUNK_SIZE = int(os.getenv("MIN_CHUNK_SIZE", "100"))

        # Parent-Child Chunk Retrieval Configuration (父块语义切分，子块机械细切)
        self.CHILD_CHUNK_SIZE = int(os.getenv("CHILD_CHUNK_SIZE", "300"))
        self.CHILD_CHUNK_OVERLAP = int(os.getenv("CHILD_CHUNK_OVERLAP", "50"))
        self.PARENT_CHILD_MAX_PARENTS = int(os.getenv("PARENT_CHILD_MAX_PARENTS", "5"))

        # Semantic Chunking Tuning (only used when CHUNK_STRATEGY=semantic)
        self.SEMANTIC_BREAKPOINT_TYPE = os.getenv("SEMANTIC_BREAKPOINT_TYPE", "percentile")
        self.SEMANTIC_BREAKPOINT_AMOUNT = float(os.getenv("SEMANTIC_BREAKPOINT_AMOUNT")) if os.getenv("SEMANTIC_BREAKPOINT_AMOUNT") else None
        self.SEMANTIC_BUFFER_SIZE = int(os.getenv("SEMANTIC_BUFFER_SIZE", "1"))

        # Tesseract OCR Configuration（默认为空，由 parser.py 自动检测）
        self.TESSERACT_PATH = os.getenv("TESSERACT_PATH", "")

        # Document Parsing Feature Switches（默认开启，便于回退）
        self.ENABLE_TABLE_EXTRACTION = os.getenv("ENABLE_TABLE_EXTRACTION", "true").lower() == "true"
        self.ENABLE_CLEANING = os.getenv("ENABLE_CLEANING", "true").lower() == "true"
        
        # Temporary Files Configuration
        self.TEMP_DIR = os.getenv("TEMP_DIR", "./temp")
        # 确保临时目录存在
        os.makedirs(self.TEMP_DIR, exist_ok=True)
        
        # Logging Configuration
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        
        # API Configuration
        self.API_HOST = os.getenv("API_HOST", "0.0.0.0")
        self.API_PORT = int(os.getenv("API_PORT", "8080"))
        
        # CORS Configuration
        self.CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
    
    def _setup_logging(self):
        """设置日志配置"""
        log_level = getattr(logging, self.LOG_LEVEL.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Configuration loaded successfully")
    
    def get_config(self) -> Dict[str, Any]:
        """获取所有配置项"""
        return {
            "DASHSCOPE_API_KEY": "***" if self.DASHSCOPE_API_KEY else "",
            "MILVUS_HOST": self.MILVUS_HOST,
            "MILVUS_PORT": self.MILVUS_PORT,
            "VECTOR_STORE_COLLECTION_NAME": self.VECTOR_STORE_COLLECTION_NAME,
            "EMBEDDING_MODEL": self.EMBEDDING_MODEL,
            "LOCAL_EMBEDDING_MODEL": self.LOCAL_EMBEDDING_MODEL,
            "RERANKER_TYPE": self.RERANKER_TYPE,
            "CHUNK_STRATEGY": self.CHUNK_STRATEGY,
            "CHUNK_SIZE": self.CHUNK_SIZE,
            "CHUNK_OVERLAP": self.CHUNK_OVERLAP,
            "MIN_CHUNK_SIZE": self.MIN_CHUNK_SIZE,
            "SEMANTIC_BREAKPOINT_TYPE": self.SEMANTIC_BREAKPOINT_TYPE,
            "SEMANTIC_BREAKPOINT_AMOUNT": self.SEMANTIC_BREAKPOINT_AMOUNT,
            "SEMANTIC_BUFFER_SIZE": self.SEMANTIC_BUFFER_SIZE,
            "TESSERACT_PATH": self.TESSERACT_PATH,
            "TEMP_DIR": self.TEMP_DIR,
            "LOG_LEVEL": self.LOG_LEVEL,
            "API_HOST": self.API_HOST,
            "API_PORT": self.API_PORT,
            "CORS_ORIGINS": self.CORS_ORIGINS
        }
    
    def validate_config(self) -> bool:
        """验证配置项"""
        is_valid = True
        
        # 验证必要的配置项
        if not self.DASHSCOPE_API_KEY:
            self.logger.warning("DASHSCOPE_API_KEY not set, will use local embeddings")
        
        # 验证Tesseract路径（空值表示由 parser.py 自动检测）
        if self.TESSERACT_PATH and not os.path.exists(self.TESSERACT_PATH):
            self.logger.warning(f"Tesseract not found at {self.TESSERACT_PATH}, will try other locations")
        
        return is_valid

# 创建全局配置实例
config = ConfigManager()
