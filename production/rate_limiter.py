"""
Production Rate Limiter
=======================

Token bucket rate limiter with per-tenant limits and Redis backend support.
"""

import time
import logging
import asyncio
from typing import Dict, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""
    allowed: bool
    remaining: int
    reset_at: float
    retry_after: Optional[float] = None


class RateLimiterBackend(ABC):
    """Abstract base for rate limiter storage backends."""

    @abstractmethod
    async def check_and_consume(
        self,
        key: str,
        tokens_requested: int,
        max_tokens: int,
        refill_rate: float
    ) -> RateLimitResult:
        """Check rate limit and consume tokens if allowed."""
        pass


class InMemoryRateLimiter(RateLimiterBackend):
    """
    In-memory token bucket rate limiter.

    Suitable for single-instance deployments or development.
    For multi-instance production, use RedisRateLimiter.
    """

    def __init__(self):
        self._buckets: Dict[str, dict] = {}
        self._lock = asyncio.Lock()

    async def check_and_consume(
        self,
        key: str,
        tokens_requested: int,
        max_tokens: int,
        refill_rate: float  # tokens per second
    ) -> RateLimitResult:
        async with self._lock:
            now = time.time()

            if key not in self._buckets:
                self._buckets[key] = {
                    "tokens": max_tokens,
                    "last_refill": now
                }

            bucket = self._buckets[key]

            # Refill tokens based on elapsed time
            elapsed = now - bucket["last_refill"]
            tokens_to_add = elapsed * refill_rate
            bucket["tokens"] = min(max_tokens, bucket["tokens"] + tokens_to_add)
            bucket["last_refill"] = now

            # Check if we have enough tokens
            if bucket["tokens"] >= tokens_requested:
                bucket["tokens"] -= tokens_requested
                return RateLimitResult(
                    allowed=True,
                    remaining=int(bucket["tokens"]),
                    reset_at=now + (max_tokens - bucket["tokens"]) / refill_rate
                )
            else:
                # Calculate when we'll have enough tokens
                tokens_needed = tokens_requested - bucket["tokens"]
                retry_after = tokens_needed / refill_rate
                return RateLimitResult(
                    allowed=False,
                    remaining=int(bucket["tokens"]),
                    reset_at=now + retry_after,
                    retry_after=retry_after
                )

    async def reset(self, key: str):
        """Reset a bucket (for testing)."""
        async with self._lock:
            if key in self._buckets:
                del self._buckets[key]


class RedisRateLimiter(RateLimiterBackend):
    """
    Redis-backed token bucket rate limiter.

    Uses Lua script for atomic operations across multiple instances.
    """

    TOKEN_BUCKET_SCRIPT = """
    local key = KEYS[1]
    local max_tokens = tonumber(ARGV[1])
    local refill_rate = tonumber(ARGV[2])
    local tokens_requested = tonumber(ARGV[3])
    local now = tonumber(ARGV[4])

    local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
    local tokens = tonumber(bucket[1])
    local last_refill = tonumber(bucket[2])

    if tokens == nil then
        tokens = max_tokens
        last_refill = now
    end

    -- Refill tokens
    local elapsed = now - last_refill
    local tokens_to_add = elapsed * refill_rate
    tokens = math.min(max_tokens, tokens + tokens_to_add)

    -- Check and consume
    if tokens >= tokens_requested then
        tokens = tokens - tokens_requested
        redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
        redis.call('EXPIRE', key, 3600)  -- 1 hour TTL
        return {1, math.floor(tokens), now + (max_tokens - tokens) / refill_rate}
    else
        redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
        redis.call('EXPIRE', key, 3600)
        local tokens_needed = tokens_requested - tokens
        local retry_after = tokens_needed / refill_rate
        return {0, math.floor(tokens), now + retry_after, retry_after}
    end
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self._redis_url = redis_url
        self._redis = None
        self._script_sha = None

    async def _get_redis(self):
        if self._redis is None:
            import redis.asyncio as aioredis
            self._redis = await aioredis.from_url(self._redis_url)
            self._script_sha = await self._redis.script_load(self.TOKEN_BUCKET_SCRIPT)
        return self._redis

    async def check_and_consume(
        self,
        key: str,
        tokens_requested: int,
        max_tokens: int,
        refill_rate: float
    ) -> RateLimitResult:
        try:
            redis = await self._get_redis()
            result = await redis.evalsha(
                self._script_sha,
                1,
                f"ratelimit:{key}",
                max_tokens,
                refill_rate,
                tokens_requested,
                time.time()
            )

            allowed = bool(result[0])
            remaining = int(result[1])
            reset_at = float(result[2])
            retry_after = float(result[3]) if len(result) > 3 else None

            return RateLimitResult(
                allowed=allowed,
                remaining=remaining,
                reset_at=reset_at,
                retry_after=retry_after
            )

        except Exception as e:
            logger.error(f"Redis rate limiter error: {e}")
            # Fail open - allow request but log the issue
            return RateLimitResult(
                allowed=True,
                remaining=0,
                reset_at=time.time()
            )


class TenantRateLimiter:
    """
    Per-tenant rate limiter with configurable limits.

    Enforces:
    - Events per second (ingest_eps)
    - Alerts per minute (alerts_per_min)
    """

    def __init__(self, backend: Optional[RateLimiterBackend] = None):
        self._backend = backend or InMemoryRateLimiter()
        self._tenant_limits: Dict[str, dict] = {}

    def configure_tenant(
        self,
        tenant_id: str,
        ingest_eps: int,
        alerts_per_min: int
    ):
        """Configure rate limits for a tenant."""
        self._tenant_limits[tenant_id] = {
            "ingest_eps": ingest_eps,
            "alerts_per_min": alerts_per_min
        }
        logger.info(f"Configured rate limits for tenant {tenant_id}: "
                    f"ingest_eps={ingest_eps}, alerts_per_min={alerts_per_min}")

    async def check_ingest_rate(
        self,
        tenant_id: str,
        event_count: int = 1
    ) -> RateLimitResult:
        """Check if event ingestion is within rate limits."""
        limits = self._tenant_limits.get(tenant_id)
        if not limits:
            # No limits configured - allow with warning
            logger.warning(f"No rate limits configured for tenant {tenant_id}")
            return RateLimitResult(allowed=True, remaining=0, reset_at=time.time())

        return await self._backend.check_and_consume(
            key=f"ingest:{tenant_id}",
            tokens_requested=event_count,
            max_tokens=limits["ingest_eps"],
            refill_rate=limits["ingest_eps"]  # Refill at max rate per second
        )

    async def check_alert_rate(
        self,
        tenant_id: str
    ) -> RateLimitResult:
        """Check if alert generation is within rate limits."""
        limits = self._tenant_limits.get(tenant_id)
        if not limits:
            return RateLimitResult(allowed=True, remaining=0, reset_at=time.time())

        return await self._backend.check_and_consume(
            key=f"alerts:{tenant_id}",
            tokens_requested=1,
            max_tokens=limits["alerts_per_min"],
            refill_rate=limits["alerts_per_min"] / 60.0  # Per minute rate
        )


# Global rate limiter instance
_rate_limiter: Optional[TenantRateLimiter] = None


def get_rate_limiter() -> TenantRateLimiter:
    """Get the global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        import os
        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            try:
                backend = RedisRateLimiter(redis_url)
                logger.info("Using Redis rate limiter backend")
            except ImportError:
                logger.warning("Redis not available, using in-memory rate limiter")
                backend = InMemoryRateLimiter()
        else:
            backend = InMemoryRateLimiter()
            logger.info("Using in-memory rate limiter backend")
        _rate_limiter = TenantRateLimiter(backend)
    return _rate_limiter


def configure_tenant_limits(tenant_manager) -> TenantRateLimiter:
    """Configure rate limits from tenant manager."""
    rate_limiter = get_rate_limiter()

    for tenant in tenant_manager.list_tenants():
        rate_limiter.configure_tenant(
            tenant_id=tenant.tenant_id,
            ingest_eps=tenant.rate_limits.ingest_eps,
            alerts_per_min=tenant.rate_limits.alerts_per_min
        )

    return rate_limiter
