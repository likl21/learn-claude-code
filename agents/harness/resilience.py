"""
Resilience layer for agent harness.

Three mechanisms:
1. retry_with_backoff  — Exponential backoff with jitter
2. CircuitBreaker      — CLOSED → OPEN → HALF_OPEN state machine
3. ResilientLLMClient  — Wraps Anthropic client with retry + breaker + fallback
"""

from __future__ import annotations

import functools
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)
T = TypeVar("T")


# ── Retry with Exponential Backoff ─────────────────────────

def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retryable_errors: tuple[type[Exception], ...] = (Exception,),
) -> Callable:
    """
    Decorator: exponential backoff with jitter.

    Delay formula:  min(base_delay * 2^attempt, max_delay) * uniform(0.75, 1.25)

    Usage::

        @retry_with_backoff(max_retries=3, retryable_errors=(RateLimitError,))
        def call_api():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_error: Optional[Exception] = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_errors as e:
                    last_error = e
                    if attempt == max_retries:
                        break
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    delay *= 0.75 + random.random() * 0.5  # jitter ±25%
                    logger.warning(
                        "Retry %d/%d after %.1fs: %s",
                        attempt + 1, max_retries, delay, e,
                    )
                    time.sleep(delay)
            raise last_error  # type: ignore[misc]
        return wrapper
    return decorator


# ── Circuit Breaker ────────────────────────────────────────

class CircuitBreakerOpen(Exception):
    """Raised when the circuit breaker is in OPEN state."""
    pass


@dataclass
class CircuitBreaker:
    """
    Three-state circuit breaker.

    State machine::

        CLOSED ──[failure_threshold consecutive failures]──→ OPEN
        OPEN   ──[recovery_timeout elapsed]──────────────→ HALF_OPEN
        HALF_OPEN ──[success]──→ CLOSED
        HALF_OPEN ──[failure]──→ OPEN

    Usage::

        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        result = breaker.call(some_function, arg1, arg2)
    """

    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    _state: str = field(default="closed", repr=False)
    _failures: int = field(default=0, repr=False)
    _last_failure_time: float = field(default=0.0, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def state(self) -> str:
        with self._lock:
            if self._state == "open":
                if time.time() - self._last_failure_time > self.recovery_timeout:
                    self._state = "half_open"
            return self._state

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute func through the circuit breaker."""
        current_state = self.state
        if current_state == "open":
            raise CircuitBreakerOpen(
                f"Circuit breaker OPEN. Recovery in "
                f"{self.recovery_timeout - (time.time() - self._last_failure_time):.0f}s"
            )
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception:
            self._on_failure()
            raise

    def _on_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._state = "closed"

    def _on_failure(self) -> None:
        with self._lock:
            self._failures += 1
            self._last_failure_time = time.time()
            if self._failures >= self.failure_threshold:
                self._state = "open"
                logger.error(
                    "Circuit breaker OPEN after %d consecutive failures",
                    self._failures,
                )

    def reset(self) -> None:
        """Manually reset to closed state."""
        with self._lock:
            self._state = "closed"
            self._failures = 0
            self._last_failure_time = 0.0


# ── Resilient LLM Client ──────────────────────────────────

class ResilientLLMClient:
    """
    Wraps an Anthropic client with retry + circuit breaker + model fallback.

    Usage::

        from anthropic import Anthropic
        raw = Anthropic()
        client = ResilientLLMClient(raw, fallback_model="claude-haiku-3-20250307")

        response = client.create(
            model="claude-sonnet-4-20250514",
            messages=[...],
            max_tokens=8000,
        )
    """

    def __init__(
        self,
        client: Any,
        fallback_model: Optional[str] = None,
        max_retries: int = 3,
        breaker_threshold: int = 5,
        breaker_timeout: float = 60.0,
    ):
        self.client = client
        self.fallback_model = fallback_model
        self._breaker = CircuitBreaker(
            failure_threshold=breaker_threshold,
            recovery_timeout=breaker_timeout,
        )
        self._max_retries = max_retries

    def create(self, **kwargs: Any) -> Any:
        """
        Messages.create with retry + circuit breaker + fallback.

        Falls back to fallback_model when circuit breaker is open.
        """
        @retry_with_backoff(
            max_retries=self._max_retries,
            base_delay=1.0,
            retryable_errors=(Exception,),
        )
        def _call(**kw: Any) -> Any:
            try:
                return self._breaker.call(
                    self.client.messages.create, **kw,
                )
            except CircuitBreakerOpen:
                if (
                    self.fallback_model
                    and kw.get("model") != self.fallback_model
                ):
                    logger.warning(
                        "Circuit breaker open — falling back to %s",
                        self.fallback_model,
                    )
                    kw["model"] = self.fallback_model
                    return self.client.messages.create(**kw)
                raise

        return _call(**kwargs)

    @property
    def breaker_state(self) -> str:
        return self._breaker.state
