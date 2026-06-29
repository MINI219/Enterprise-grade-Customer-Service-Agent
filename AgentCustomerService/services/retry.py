"""
弹性设计工具 —— 重试、退避、超时

为 ChromaDB HTTP 调用提供网络层的容错能力：
  - 指数退避 + 随机抖动（避免惊群效应）
  - 可重试异常白名单（网络抖动 / 临时不可用）
  - 不可重试异常黑名单（认证失败 / 参数错误）
  - 超时控制
  - 断路器模式的轻量实现（连续失败计数）

使用方式：
    from services.retry import retry_on_network_error

    @retry_on_network_error
    def call_chromadb():
        ...

    # 异步版本
    @retry_on_network_error_async
    async def call_chromadb_async():
        ...
"""

import asyncio
import functools
import time
from typing import Callable, Optional, Set, Type, TypeVar

from tenacity import (
    RetryError,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)
from tenacity.retry import retry_any

from app.core.logger import logger

# ═══════════════════════════════════════════════════════════════
# 可重试异常白名单
# ═══════════════════════════════════════════════════════════════

RETRYABLE_EXCEPTIONS: Set[Type[BaseException]] = {
    ConnectionError,           # DNS / TCP 连接失败
    TimeoutError,              # 通用超时
    asyncio.TimeoutError,      # asyncio 超时
    OSError,                   # 底层 I/O 错误（Errno 101/104/110 等）
    ConnectionRefusedError,    # 端口未监听 / Docker 重启中
    ConnectionResetError,      # TCP RST
    BrokenPipeError,           # 管道断开
}

# 某些第三方库可能抛出 HTTPError / RequestException
try:
    import httpx
    RETRYABLE_EXCEPTIONS.add(httpx.HTTPStatusError)   # 5xx 服务端错误
    RETRYABLE_EXCEPTIONS.add(httpx.ConnectError)
    RETRYABLE_EXCEPTIONS.add(httpx.ReadError)
    RETRYABLE_EXCEPTIONS.add(httpx.WriteError)
    RETRYABLE_EXCEPTIONS.add(httpx.RemoteProtocolError)
except ImportError:
    pass

try:
    import requests
    RETRYABLE_EXCEPTIONS.add(requests.ConnectionError)
    RETRYABLE_EXCEPTIONS.add(requests.Timeout)
except ImportError:
    pass

# ChromaDB 自身的异常（网络相关）
try:
    import chromadb
    RETRYABLE_EXCEPTIONS.add(chromadb.errors.ChromaError)  # 临时服务端错误
except (ImportError, AttributeError):
    pass

# ═══════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════

# 最大重试次数（含首次 = 总共 3 次）
MAX_RETRY_ATTEMPTS = 3

# 最小等待时间（秒），首次重试前等待
MIN_WAIT_SECONDS = 0.5

# 最大等待时间（秒），指数退避上限
MAX_WAIT_SECONDS = 10.0

# 连续失败阈值（超过则触发熔断告警）
CIRCUIT_BREAKER_THRESHOLD = 5


# ═══════════════════════════════════════════════════════════════
# 重试装饰器
# ═══════════════════════════════════════════════════════════════

F = TypeVar("F", bound=Callable)


def retry_on_network_error(func: F) -> F:
    """
    装饰器：为同步函数添加网络错误自动重试。

    重试策略：
      - 指数退避：0.5s → 2s → 10s（含 40% 随机抖动）
      - 最多 3 次尝试
      - 仅对网络类异常重试，业务异常直接抛出
    """
    decorator = retry(
        retry=retry_if_exception_type(tuple(RETRYABLE_EXCEPTIONS)),
        stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
        wait=wait_exponential_jitter(
            initial=MIN_WAIT_SECONDS,
            max=MAX_WAIT_SECONDS,
            jitter=0.4,
        ),
        before_sleep=before_sleep_log(logger, "WARNING"),
        reraise=True,
    )
    return decorator(func)  # type: ignore[return-value]


def retry_on_network_error_async(func: Callable) -> Callable:
    """
    装饰器：为异步函数添加网络错误自动重试。

    与同步版本相同的退避策略，适配 asyncio 调用链。

    使用方式：
        @retry_on_network_error_async
        async def my_async_func():
            ...
    """
    decorator = retry(
        retry=retry_if_exception_type(tuple(RETRYABLE_EXCEPTIONS)),
        stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
        wait=wait_exponential_jitter(
            initial=MIN_WAIT_SECONDS,
            max=MAX_WAIT_SECONDS,
            jitter=0.4,
        ),
        before_sleep=before_sleep_log(logger, "WARNING"),
        reraise=True,
    )
    return decorator(func)


# ═══════════════════════════════════════════════════════════════
# 异步超时包装器
# ═══════════════════════════════════════════════════════════════

async def with_timeout(
    coro,
    timeout_seconds: float = 15.0,
    operation_name: str = "operation",
) -> object:
    """
    为异步协程添加超时控制。

    若超时，抛出 asyncio.TimeoutError（属于 RETRYABLE_EXCEPTIONS，
    会被上层的 retry_on_network_error_async 捕获并重试）。

    Args:
        coro:            协程对象
        timeout_seconds: 超时秒数（默认 15s，适配 ChromaDB 向量检索延迟）
        operation_name:  操作描述（用于日志）

    Returns:
        协程的返回值

    Raises:
        asyncio.TimeoutError: 超时
    """
    try:
        return await asyncio.wait_for(
            coro,
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.warning(
            f"[Retry] 操作超时 | op={operation_name} | timeout={timeout_seconds}s"
        )
        raise


# ═══════════════════════════════════════════════════════════════
# 轻量断路器（Circuit Breaker）
# ═══════════════════════════════════════════════════════════════

class CircuitBreaker:
    """
    轻量断路器 —— 当连续失败达到阈值时，短暂熔断以避免雪崩。

    不引入额外依赖，使用内存计数器实现。

    状态机：
        CLOSED ──连续失败达到阈值──→ OPEN
        OPEN   ──冷却时间过后─────→ HALF_OPEN
        HALF_OPEN ──成功──────→ CLOSED
        HALF_OPEN ──失败──────→ OPEN（重置冷却）
    """

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = CIRCUIT_BREAKER_THRESHOLD,
        cooldown_seconds: float = 30.0,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds

        self._failure_count: int = 0
        self._last_failure_time: float = 0.0
        self._state: str = "CLOSED"  # CLOSED | OPEN | HALF_OPEN

    @property
    def is_open(self) -> bool:
        """断路器是否开路（拒绝请求）。"""
        if self._state == "CLOSED":
            return False
        if self._state == "HALF_OPEN":
            return False
        # OPEN 状态：检查冷却时间
        if time.monotonic() - self._last_failure_time >= self.cooldown_seconds:
            self._state = "HALF_OPEN"
            logger.info(
                f"[CB:{self.name}] 冷却完成，进入 HALF_OPEN 试探状态"
            )
            return False
        return True

    def record_success(self) -> None:
        """记录一次成功调用。"""
        self._failure_count = 0
        if self._state in ("HALF_OPEN", "OPEN"):
            logger.info(f"[CB:{self.name}] 试探成功，恢复 CLOSED")
        self._state = "CLOSED"

    def record_failure(self) -> None:
        """记录一次失败调用。"""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._state == "HALF_OPEN":
            logger.warning(
                f"[CB:{self.name}] 试探失败，重新 OPEN | failures={self._failure_count}"
            )
            self._state = "OPEN"
        elif self._failure_count >= self.failure_threshold and self._state == "CLOSED":
            logger.error(
                f"[CB:{self.name}] 连续失败 {self._failure_count} 次，熔断开路！"
                f"冷却 {self.cooldown_seconds}s"
            )
            self._state = "OPEN"

    def reset(self) -> None:
        """手动重置（用于测试或运维干预）。"""
        self._failure_count = 0
        self._state = "CLOSED"
        logger.info(f"[CB:{self.name}] 已手动重置")


# ═══════════════════════════════════════════════════════════════
# 全局 ChromaDB 断路器
# ═══════════════════════════════════════════════════════════════

chromadb_breaker = CircuitBreaker(
    name="chromadb",
    failure_threshold=CIRCUIT_BREAKER_THRESHOLD,
    cooldown_seconds=30.0,
)
