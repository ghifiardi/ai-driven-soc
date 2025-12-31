"""
Production Hardening Module
===========================

This module provides production-ready components for:
- Security: Secret management, service authentication, input validation
- Rate Limiting: Per-tenant token bucket rate limiting
- Observability: Structured logging, metrics, health checks
- Resilience: Circuit breakers, retry logic, graceful shutdown
- Tenant Isolation: Context propagation, scoped clients
"""

from production.security import (
    SecretManager,
    secret_manager,
    get_jwt_secret,
    get_service_api_key,
    ServiceAuthenticator,
    InputValidator,
    hash_api_key,
    verify_api_key,
)

from production.rate_limiter import (
    TenantRateLimiter,
    RateLimitResult,
    get_rate_limiter,
    configure_tenant_limits,
)

from production.observability import (
    generate_correlation_id,
    get_correlation_id,
    set_correlation_id,
    set_tenant_context,
    get_tenant_context,
    configure_structured_logging,
    StructuredLogger,
    MetricsCollector,
    metrics,
    HealthChecker,
    HealthCheckResult,
    timed_operation,
)

from production.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    circuit_breaker,
    get_circuit_breaker,
    RetryConfig,
    retry,
    retry_async,
    GracefulShutdown,
    get_shutdown_handler,
    setup_signal_handlers,
    timeout,
    TimeoutError,
    Bulkhead,
    get_bulkhead,
)

from production.tenant_context import (
    TenantContext,
    set_current_tenant,
    get_current_tenant_id,
    get_current_tenant_config,
    clear_tenant_context,
    TenantContextMiddleware,
    require_tenant_context,
    TenantAwareClient,
    TenantScopedBigQueryClient,
)

__all__ = [
    # Security
    "SecretManager",
    "secret_manager",
    "get_jwt_secret",
    "get_service_api_key",
    "ServiceAuthenticator",
    "InputValidator",
    "hash_api_key",
    "verify_api_key",
    # Rate Limiting
    "TenantRateLimiter",
    "RateLimitResult",
    "get_rate_limiter",
    "configure_tenant_limits",
    # Observability
    "generate_correlation_id",
    "get_correlation_id",
    "set_correlation_id",
    "set_tenant_context",
    "get_tenant_context",
    "configure_structured_logging",
    "StructuredLogger",
    "MetricsCollector",
    "metrics",
    "HealthChecker",
    "HealthCheckResult",
    "timed_operation",
    # Resilience
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerError",
    "circuit_breaker",
    "get_circuit_breaker",
    "RetryConfig",
    "retry",
    "retry_async",
    "GracefulShutdown",
    "get_shutdown_handler",
    "setup_signal_handlers",
    "timeout",
    "TimeoutError",
    "Bulkhead",
    "get_bulkhead",
    # Tenant Context
    "TenantContext",
    "set_current_tenant",
    "get_current_tenant_id",
    "get_current_tenant_config",
    "clear_tenant_context",
    "TenantContextMiddleware",
    "require_tenant_context",
    "TenantAwareClient",
    "TenantScopedBigQueryClient",
]
