"""
并发限流器 - 面试核心问题：
1. 本地 embedding 推理如何做并发控制？
   答：embedding 是 CPU 密集型，无限并发会打满 CPU 导致所有请求变慢
       设置最大并发数，让每条请求都有合理响应时间
2. 为什么用令牌桶而不是信号量？
   答：令牌桶更灵活，可以控制最大并发数还能控制速率
"""
import time
import threading
from typing import Optional

from internal.model.config import get_config
from internal.pkg.logger import get_logger


class RateLimiter:
    """
    令牌桶限流器 - 面试点：
    - 为什么要限流 embedding？CPU 推理是计算密集型，并发过高响应崩溃
    - max_concurrent=4：4核 CPU 最优，1核给系统，3核给推理
    """

    def __init__(self, max_concurrent: int = None):
        cfg = get_config()
        self.max_concurrent = max_concurrent or cfg.embedding.max_concurrent
        self._current = 0
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self.logger = get_logger()

    def acquire(self, timeout: float = 30.0) -> bool:
        """
        获取令牌 - 超时返回 False
        面试点：为什么是阻塞等待而不是直接拒绝？
        答：RAG 查询是同步操作，用户期望得到结果
            短暂等待比直接返回"系统繁忙"用户体验更好
        """
        deadline = time.time() + timeout
        with self._condition:
            while self._current >= self.max_concurrent:
                remaining = deadline - time.time()
                if remaining <= 0:
                    self.logger.warn("acquire", "rate_limiter",
                                     "获取令牌超时",
                                     timeout=timeout, current=self._current)
                    return False
                self._condition.wait(timeout=min(remaining, 1.0))

            self._current += 1
            return True

    def release(self):
        """释放令牌"""
        with self._condition:
            self._current -= 1
            self._condition.notify()

    @property
    def available(self) -> int:
        with self._lock:
            return self.max_concurrent - self._current

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        self.release()


# 全局单例
_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    global _limiter
    if _limiter is None:
        _limiter = RateLimiter()
    return _limiter
