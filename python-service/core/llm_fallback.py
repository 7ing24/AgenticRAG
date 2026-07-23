"""LLM 全链路 Fallback 引擎

重试 → 熔断 → 降级 → 可观测，收敛到一处控制全局 LLM fallback 行为。

用法:
    from core.llm_fallback import fallback_handler

    def _call_llm(prompt):
        resp = dashscope.Generation.call(...)
        if resp.status_code != 200:
            raise LLMServiceError(...)
        return resp

    result = fallback_handler.invoke(_call_llm, scene="generation")
"""

import time
import logging
import threading
from typing import Callable, Dict, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 自定义异常
# ═══════════════════════════════════════════════════════════════

class LLMServiceError(Exception):
    """LLM 调用失败（可重试）"""


class CircuitBreakerOpenError(Exception):
    """熔断器已打开，拒绝调用"""


class NonRetryableError(Exception):
    """不可重试的错误（如参数错误）"""


# ═══════════════════════════════════════════════════════════════
# 熔断器
# ═══════════════════════════════════════════════════════════════

class CircuitBreakerState(Enum):
    CLOSED = "CLOSED"          # 正常通行
    OPEN = "OPEN"              # 熔断，拒绝调用
    HALF_OPEN = "HALF_OPEN"    # 试探性放行


class CircuitBreaker:
    """熔断器：连续失败 N 次后打开，冷却期后进入半开试探

    状态机: CLOSED → (连续失败>=threshold) → OPEN → (冷却期过) → HALF_OPEN → (成功) → CLOSED
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 1,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._half_open_calls = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        return self._state.value

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """包装一次调用，自动处理熔断逻辑"""
        with self._lock:
            if self._state == CircuitBreakerState.OPEN:
                if time.time() - self._last_failure_time >= self.recovery_timeout:
                    self._state = CircuitBreakerState.HALF_OPEN
                    self._half_open_calls = 0
                    logger.info("[CircuitBreaker] OPEN → HALF_OPEN (recovery timeout elapsed)")
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker is OPEN. "
                        f"Retry in {self.recovery_timeout - (time.time() - self._last_failure_time):.0f}s"
                    )

            if self._state == CircuitBreakerState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    raise CircuitBreakerOpenError("Circuit breaker is HALF_OPEN, max probe calls reached")

        try:
            result = func(*args, **kwargs)
        except (CircuitBreakerOpenError, NonRetryableError):
            raise
        except Exception as e:
            self._on_failure(e)
            raise

        self._on_success()
        return result

    def _on_success(self):
        with self._lock:
            if self._state == CircuitBreakerState.HALF_OPEN:
                logger.info("[CircuitBreaker] HALF_OPEN → CLOSED (probe succeeded)")
            self._state = CircuitBreakerState.CLOSED
            self._failure_count = 0
            self._half_open_calls = 0

    def _on_failure(self, error: Exception):
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._half_open_calls += 1

            if (self._state == CircuitBreakerState.CLOSED
                    and self._failure_count >= self.failure_threshold):
                self._state = CircuitBreakerState.OPEN
                logger.warning(
                    f"[CircuitBreaker] CLOSED → OPEN "
                    f"(failures: {self._failure_count}, error: {error})"
                )

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "state": self._state.value,
                "failure_count": self._failure_count,
                "last_failure_time": self._last_failure_time,
            }


# ═══════════════════════════════════════════════════════════════
# 重试配置
# ═══════════════════════════════════════════════════════════════

class RetryConfig:
    """重试配置"""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    def get_delay(self, attempt: int) -> float:
        """指数退避：1s → 2s → 4s → ..."""
        return min(self.base_delay * (2 ** attempt), self.max_delay)

    def is_retryable(self, error: Exception) -> bool:
        """判断异常是否可重试"""
        if isinstance(error, NonRetryableError):
            return False
        retryable_keywords = [
            "timeout", "connection", "network", "rate_limit",
            "throttl", "server error", "service unavailable",
            "internal server error", "bad gateway", "gateway",
        ]
        error_str = str(error).lower()
        return any(kw in error_str for kw in retryable_keywords)


# ═══════════════════════════════════════════════════════════════
# 降级文案注册表
# ═══════════════════════════════════════════════════════════════

class FallbackMessageRegistry:
    """统一降级文案，按场景分类"""

    SCENES: Dict[str, str] = {
        # 通用生成
        "generation":         "抱歉，我暂时无法回答这个问题，请稍后再试。",
        # 无 API 密钥
        "no_api_key":         "AI服务未配置API密钥。请联系管理员配置DASHSCOPE_API_KEY以启用完整功能。",
        # 熔断中
        "circuit_open":       "AI服务暂时繁忙，请稍后再试。",
        # 超时
        "timeout":            "AI服务响应超时，请稍后重试。",
        # 闲聊
        "chitchat":           "哈哈，这个话题挺有意思的！你最近有什么新鲜事想分享吗？",
        # 标题生成
        "title":              "New Chat",
        # 摘要
        "summary":            "生成摘要失败，请稍后再试。",
        # 记忆压缩
        "memory_compress":    "用户进行了多轮对话。",
        # 意图分类 — 空字符串 = 调用方走关键词规则兜底
        "classification":     "",
        # 问题改写 — 空字符串 = 调用方返回原始 query
        "rewrite":            "",
        # 检索后生成
        "retrieval_answer":   "知识库检索暂时不可用。请稍后再试，或尝试换个方式提问。",
        # 无检索结果的 LLM 兜底
        "no_knowledge":       "知识库中未找到相关信息，以下是基于通用知识的回答：",
    }

    def get(self, scene: str = "generation", **kwargs) -> str:
        """获取降级文案

        Args:
            scene: 场景标识
            **kwargs: 可选的格式化参数（如默认消息覆盖）
        """
        if "default_message" in kwargs and kwargs["default_message"]:
            return kwargs["default_message"]
        return self.SCENES.get(scene, self.SCENES["generation"])

    def register(self, scene: str, message: str):
        """注册自定义场景文案"""
        self.SCENES[scene] = message


# ═══════════════════════════════════════════════════════════════
# 统一 Fallback Handler
# ═══════════════════════════════════════════════════════════════

class LLMFallbackHandler:
    """LLM 调用统一入口：重试 → 熔断 → 降级

    用法：
        handler = LLMFallbackHandler()

        def my_llm_call():
            ...

        result, cb = handler.invoke(my_llm_call, scene="generation")
    """

    def __init__(
        self,
        circuit_breaker: Optional[CircuitBreaker] = None,
        retry_config: Optional[RetryConfig] = None,
        registry: Optional[FallbackMessageRegistry] = None,
    ):
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.retry_config = retry_config or RetryConfig()
        self.registry = registry or FallbackMessageRegistry()

        # 可观测指标
        self._metrics_lock = threading.Lock()
        self._total_calls: Dict[str, int] = {}       # scene → total calls
        self._failure_counts: Dict[str, int] = {}     # scene → failure count
        self._retry_counts: Dict[str, int] = {}       # scene → retry count
        self._circuit_open_count: int = 0

    # ── 主入口 ──────────────────────────────────────────────────

    def invoke(
        self,
        func: Callable,
        scene: str = "generation",
    ) -> Any:
        """包装 LLM 调用，自动重试 + 熔断

        成功返回 func 的结果；失败时抛出异常（由上层调用方决定如何降级）。

        Args:
            func: 实际 LLM 调用的 callable（无参数）
            scene: 场景标识，用于分场景统计

        Returns:
            func 的成功返回值

        Raises:
            CircuitBreakerOpenError: 熔断器打开
            LLMServiceError: LLM 调用失败（重试耗尽后）
        """
        self._incr_dict(self._total_calls, scene)
        try:
            return self.circuit_breaker.call(self._try_with_retry, func, scene)
        except CircuitBreakerOpenError:
            self._circuit_open_count += 1
            raise

    def invoke_or_fallback(
        self,
        func: Callable,
        scene: str = "generation",
        default_message: str = "",
    ) -> Any:
        """invoke + 自动降级：失败时返回 Registry 中的降级文案

        用于 get_answer / generate 等希望直接拿到可用返回值的场景。
        """
        try:
            return self.invoke(func, scene)
        except (CircuitBreakerOpenError, LLMServiceError, Exception) as e:
            logger.warning(f"[LLMFallback] scene={scene} fallback triggered: {e}")
            return self._wrap_fallback(
                self.registry.get(scene, default_message=default_message),
                scene,
            )

    def _try_with_retry(self, func: Callable, scene: str) -> Any:
        """带重试的执行"""
        last_error = None

        for attempt in range(self.retry_config.max_retries + 1):
            try:
                return func()
            except (CircuitBreakerOpenError, NonRetryableError):
                raise
            except Exception as e:
                last_error = e
                if attempt < self.retry_config.max_retries and self.retry_config.is_retryable(e):
                    delay = self.retry_config.get_delay(attempt)
                    self._incr_dict(self._retry_counts, scene)
                    logger.warning(
                        f"[LLMFallback] scene={scene} attempt={attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    break

        # 重试耗尽
        self._incr_dict(self._failure_counts, scene)
        logger.error(
            f"[LLMFallback] scene={scene} all retries exhausted. "
            f"Last error: {last_error}"
        )
        raise last_error  # 由外层熔断器捕获

    def _wrap_fallback(self, fallback_text: str, scene: str) -> Any:
        """将降级文案包装成与正常返回值兼容的格式

        对于 _dashscope_generate：返回 (text, TokenCallback())
        对于其他方法：直接返回字符串
        """
        if scene in ("generation", "title", "chitchat", "summary",
                      "memory_compress", "retrieval_answer"):
            # 这些场景返回 (text, callback) 元组
            return (fallback_text, TokenCallback())
        return fallback_text

    # ── Metrics ───────────────────────────────────────────────────

    def _incr_dict(self, d: Dict, key: str):
        with self._metrics_lock:
            d[key] = d.get(key, 0) + 1

    def get_stats(self) -> dict:
        """获取 LLM 调用统计（供 health check 等使用）"""
        with self._metrics_lock:
            return {
                "circuit_breaker": self.circuit_breaker.get_stats(),
                "total_calls": dict(self._total_calls),
                "failure_counts": dict(self._failure_counts),
                "retry_counts": dict(self._retry_counts),
                "circuit_open_count": self._circuit_open_count,
            }


# ═══════════════════════════════════════════════════════════════
# TokenCallback（被降级时返回空 token 统计）
# ═══════════════════════════════════════════════════════════════

class TokenCallback:
    """空的 TokenCallback，用于降级返回值"""
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0

    def to_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


# ═══════════════════════════════════════════════════════════════
# 全局单例
# ═══════════════════════════════════════════════════════════════

fallback_handler = LLMFallbackHandler()
