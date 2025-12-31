"""
Production Resilience Module
============================

Provides circuit breakers, retry logic, and graceful shutdown
for production deployment.
"""

import os
import time
import signal
import asyncio
import logging
from typing import Callable, Optional, Any, Dict, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import random

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# Circuit Breaker
# =============================================================================

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes before closing
    timeout: float = 30.0  # Seconds before trying half-open
    excluded_exceptions: tuple = ()  # Exceptions that don't count as failures


class CircuitBreakerError(Exception):
    """Raised when circuit is open."""
    def __init__(self, service_name: str, retry_after: float):
        self.service_name = service_name
        self.retry_after = retry_after
        super().__init__(f"Circuit breaker open for {service_name}, retry after {retry_after:.1f}s")


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.

    States:
    - CLOSED: Normal operation, all requests pass through
    - OPEN: Service is failing, reject all requests
    - HALF_OPEN: Testing if service recovered
    """

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    async def _update_state(self):
        """Update circuit state based on current conditions."""
        if self._state == CircuitState.OPEN:
            if self._last_failure_time:
                time_since_failure = time.time() - self._last_failure_time
                if time_since_failure >= self.config.timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                    logger.info(f"Circuit {self.name}: OPEN -> HALF_OPEN")

    async def record_success(self):
        """Record a successful call."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    logger.info(f"Circuit {self.name}: HALF_OPEN -> CLOSED")
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = max(0, self._failure_count - 1)

    async def record_failure(self, exception: Exception):
        """Record a failed call."""
        # Check if this exception type should be excluded
        if isinstance(exception, self.config.excluded_exceptions):
            return

        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._state = CircuitState.OPEN
                    logger.warning(f"Circuit {self.name}: CLOSED -> OPEN")
            elif self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning(f"Circuit {self.name}: HALF_OPEN -> OPEN")

    async def can_proceed(self) -> bool:
        """Check if a request can proceed."""
        async with self._lock:
            await self._update_state()

            if self._state == CircuitState.CLOSED:
                return True
            elif self._state == CircuitState.HALF_OPEN:
                return True
            else:  # OPEN
                return False

    def retry_after(self) -> float:
        """Get seconds until circuit might close."""
        if self._state != CircuitState.OPEN:
            return 0.0
        if self._last_failure_time:
            elapsed = time.time() - self._last_failure_time
            return max(0.0, self.config.timeout - elapsed)
        return self.config.timeout


# Global circuit breakers registry
_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """Get or create a circuit breaker by name."""
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name, config)
    return _circuit_breakers[name]


def circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """Decorator to apply circuit breaker to a function."""
    def decorator(func):
        cb = get_circuit_breaker(name, config)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not await cb.can_proceed():
                raise CircuitBreakerError(name, cb.retry_after())

            try:
                result = await func(*args, **kwargs)
                await cb.record_success()
                return result
            except Exception as e:
                await cb.record_failure(e)
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, use a simple check
            if cb._state == CircuitState.OPEN:
                raise CircuitBreakerError(name, cb.retry_after())

            try:
                result = func(*args, **kwargs)
                asyncio.get_event_loop().run_until_complete(cb.record_success())
                return result
            except Exception as e:
                asyncio.get_event_loop().run_until_complete(cb.record_failure(e))
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# =============================================================================
# Retry Logic
# =============================================================================

@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_attempts: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 30.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple = (Exception,)


def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate delay for the next retry attempt."""
    delay = config.base_delay * (config.exponential_base ** attempt)
    delay = min(delay, config.max_delay)

    if config.jitter:
        # Add random jitter up to 25% of delay
        delay = delay * (0.75 + random.random() * 0.5)

    return delay


async def retry_async(
    func: Callable,
    config: Optional[RetryConfig] = None,
    *args,
    **kwargs
) -> Any:
    """Execute a function with retry logic."""
    config = config or RetryConfig()
    last_exception = None

    for attempt in range(config.max_attempts):
        try:
            return await func(*args, **kwargs)
        except config.retryable_exceptions as e:
            last_exception = e
            if attempt < config.max_attempts - 1:
                delay = calculate_delay(attempt, config)
                logger.warning(
                    f"Attempt {attempt + 1}/{config.max_attempts} failed: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )
                await asyncio.sleep(delay)
            else:
                logger.error(f"All {config.max_attempts} attempts failed")

    raise last_exception


def retry(config: Optional[RetryConfig] = None):
    """Decorator to add retry logic to a function."""
    cfg = config or RetryConfig()

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await retry_async(func, cfg, *args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(cfg.max_attempts):
                try:
                    return func(*args, **kwargs)
                except cfg.retryable_exceptions as e:
                    last_exception = e
                    if attempt < cfg.max_attempts - 1:
                        delay = calculate_delay(attempt, cfg)
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                        time.sleep(delay)
            raise last_exception

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# =============================================================================
# Graceful Shutdown
# =============================================================================

class GracefulShutdown:
    """
    Manages graceful shutdown of the application.

    Features:
    - Signal handling (SIGTERM, SIGINT)
    - Request draining period
    - Resource cleanup callbacks
    """

    def __init__(self, drain_seconds: float = 10.0):
        self.drain_seconds = drain_seconds
        self._shutdown_event = asyncio.Event()
        self._cleanup_callbacks: list[Callable] = []
        self._active_requests = 0
        self._lock = asyncio.Lock()
        self._shutting_down = False

    @property
    def is_shutting_down(self) -> bool:
        return self._shutting_down

    def register_cleanup(self, callback: Callable):
        """Register a cleanup callback to run during shutdown."""
        self._cleanup_callbacks.append(callback)

    async def track_request(self):
        """Track an active request."""
        async with self._lock:
            self._active_requests += 1

    async def untrack_request(self):
        """Untrack a completed request."""
        async with self._lock:
            self._active_requests -= 1

    async def wait_for_shutdown(self):
        """Wait for the shutdown signal."""
        await self._shutdown_event.wait()

    def trigger_shutdown(self):
        """Trigger the shutdown process."""
        if self._shutting_down:
            return
        self._shutting_down = True
        logger.info("Shutdown triggered")
        self._shutdown_event.set()

    async def graceful_shutdown(self):
        """Execute graceful shutdown sequence."""
        logger.info(f"Starting graceful shutdown (drain period: {self.drain_seconds}s)")

        # Mark as shutting down
        self._shutting_down = True

        # Wait for active requests to complete
        start_time = time.time()
        while time.time() - start_time < self.drain_seconds:
            async with self._lock:
                if self._active_requests == 0:
                    break
                logger.info(f"Waiting for {self._active_requests} active requests...")
            await asyncio.sleep(1.0)

        async with self._lock:
            if self._active_requests > 0:
                logger.warning(f"Forcing shutdown with {self._active_requests} active requests")

        # Run cleanup callbacks
        logger.info("Running cleanup callbacks...")
        for callback in self._cleanup_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                logger.error(f"Cleanup callback error: {e}")

        logger.info("Graceful shutdown complete")


# Global shutdown handler
_shutdown_handler: Optional[GracefulShutdown] = None


def get_shutdown_handler(drain_seconds: float = 10.0) -> GracefulShutdown:
    """Get or create the global shutdown handler."""
    global _shutdown_handler
    if _shutdown_handler is None:
        _shutdown_handler = GracefulShutdown(drain_seconds)
    return _shutdown_handler


def setup_signal_handlers(shutdown_handler: GracefulShutdown):
    """Set up signal handlers for graceful shutdown."""
    loop = asyncio.get_event_loop()

    def handle_signal(sig):
        logger.info(f"Received signal {sig.name}")
        shutdown_handler.trigger_shutdown()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, handle_signal, sig)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            signal.signal(sig, lambda s, f: handle_signal(signal.Signals(s)))


# =============================================================================
# Timeout Wrapper
# =============================================================================

class TimeoutError(Exception):
    """Raised when an operation times out."""
    pass


async def with_timeout(coro, timeout_seconds: float, operation_name: str = "operation"):
    """Execute a coroutine with a timeout."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        raise TimeoutError(f"{operation_name} timed out after {timeout_seconds}s")


def timeout(seconds: float, operation_name: str = "operation"):
    """Decorator to add timeout to async functions."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await with_timeout(
                func(*args, **kwargs),
                seconds,
                operation_name
            )
        return wrapper
    return decorator


# =============================================================================
# Bulkhead Pattern (Resource Isolation)
# =============================================================================

class Bulkhead:
    """
    Bulkhead pattern for resource isolation.

    Limits concurrent access to a resource to prevent cascade failures.
    """

    def __init__(self, name: str, max_concurrent: int, max_waiting: int = 0):
        self.name = name
        self.max_concurrent = max_concurrent
        self.max_waiting = max_waiting
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._waiting = 0
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """Try to acquire a slot in the bulkhead."""
        async with self._lock:
            if self._waiting >= self.max_waiting:
                return False
            self._waiting += 1

        try:
            await self._semaphore.acquire()
            return True
        finally:
            async with self._lock:
                self._waiting -= 1

    def release(self):
        """Release a slot in the bulkhead."""
        self._semaphore.release()

    async def __aenter__(self):
        acquired = await self.acquire()
        if not acquired:
            raise RuntimeError(f"Bulkhead {self.name} is full")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False


# Global bulkheads registry
_bulkheads: Dict[str, Bulkhead] = {}


def get_bulkhead(name: str, max_concurrent: int = 10, max_waiting: int = 5) -> Bulkhead:
    """Get or create a bulkhead by name."""
    if name not in _bulkheads:
        _bulkheads[name] = Bulkhead(name, max_concurrent, max_waiting)
    return _bulkheads[name]
