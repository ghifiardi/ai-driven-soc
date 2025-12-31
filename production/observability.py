"""
Production Observability Module
===============================

Provides structured logging, correlation IDs, metrics, and health checks
for production deployment.
"""

import os
import time
import json
import logging
import asyncio
import uuid
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from contextvars import ContextVar
from functools import wraps

# Context variable for correlation ID propagation
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")
tenant_id_var: ContextVar[str] = ContextVar("tenant_id", default="")


# =============================================================================
# Correlation ID Management
# =============================================================================

def generate_correlation_id() -> str:
    """Generate a new correlation ID."""
    return str(uuid.uuid4())


def get_correlation_id() -> str:
    """Get the current correlation ID."""
    cid = correlation_id_var.get()
    if not cid:
        cid = generate_correlation_id()
        correlation_id_var.set(cid)
    return cid


def set_correlation_id(correlation_id: str):
    """Set the correlation ID for the current context."""
    correlation_id_var.set(correlation_id)


def get_tenant_context() -> str:
    """Get the current tenant ID."""
    return tenant_id_var.get()


def set_tenant_context(tenant_id: str):
    """Set the tenant ID for the current context."""
    tenant_id_var.set(tenant_id)


# =============================================================================
# Structured Logging
# =============================================================================

class StructuredFormatter(logging.Formatter):
    """JSON-formatted log output with correlation IDs and tenant context."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": correlation_id_var.get() or None,
            "tenant_id": tenant_id_var.get() or None,
        }

        # Add source location for errors
        if record.levelno >= logging.ERROR:
            log_entry["source"] = {
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName
            }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add any extra fields
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)

        return json.dumps(log_entry)


class StructuredLogger:
    """Logger wrapper with structured output and context propagation."""

    def __init__(self, name: str):
        self._logger = logging.getLogger(name)

    def _log(self, level: int, message: str, **kwargs):
        extra_fields = kwargs.pop("extra", {})
        extra_fields.update(kwargs)

        record = self._logger.makeRecord(
            self._logger.name,
            level,
            "(unknown)",
            0,
            message,
            (),
            None
        )
        record.extra_fields = extra_fields
        self._logger.handle(record)

    def debug(self, message: str, **kwargs):
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        self._log(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs):
        self._log(logging.CRITICAL, message, **kwargs)


def configure_structured_logging(level: str = "INFO"):
    """Configure structured logging for the application."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Create handler with structured formatter
    handler = logging.StreamHandler()
    handler.setFormatter(StructuredFormatter())

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers = [handler]

    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)


# =============================================================================
# Metrics Collection
# =============================================================================

@dataclass
class MetricValue:
    """A single metric data point."""
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """
    Simple metrics collector for export to monitoring systems.

    Supports counters, gauges, and histograms.
    """

    def __init__(self, namespace: str = "soc"):
        self._namespace = namespace
        self._counters: Dict[str, float] = {}
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = {}
        self._labels: Dict[str, Dict[str, str]] = {}

    def _make_key(self, name: str, labels: Dict[str, str]) -> str:
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{self._namespace}_{name}_{label_str}"

    def increment_counter(self, name: str, value: float = 1.0, **labels):
        """Increment a counter metric."""
        key = self._make_key(name, labels)
        self._counters[key] = self._counters.get(key, 0) + value
        self._labels[key] = labels

    def set_gauge(self, name: str, value: float, **labels):
        """Set a gauge metric."""
        key = self._make_key(name, labels)
        self._gauges[key] = value
        self._labels[key] = labels

    def observe_histogram(self, name: str, value: float, **labels):
        """Record a histogram observation."""
        key = self._make_key(name, labels)
        if key not in self._histograms:
            self._histograms[key] = []
        self._histograms[key].append(value)
        self._labels[key] = labels

    def get_metrics(self) -> List[MetricValue]:
        """Get all metrics for export."""
        metrics = []

        for key, value in self._counters.items():
            name = key.split("_", 2)[1] if "_" in key else key
            metrics.append(MetricValue(
                name=f"{self._namespace}_{name}_total",
                value=value,
                labels=self._labels.get(key, {})
            ))

        for key, value in self._gauges.items():
            name = key.split("_", 2)[1] if "_" in key else key
            metrics.append(MetricValue(
                name=f"{self._namespace}_{name}",
                value=value,
                labels=self._labels.get(key, {})
            ))

        for key, values in self._histograms.items():
            if values:
                name = key.split("_", 2)[1] if "_" in key else key
                # Export count, sum, and percentiles
                metrics.append(MetricValue(
                    name=f"{self._namespace}_{name}_count",
                    value=len(values),
                    labels=self._labels.get(key, {})
                ))
                metrics.append(MetricValue(
                    name=f"{self._namespace}_{name}_sum",
                    value=sum(values),
                    labels=self._labels.get(key, {})
                ))
                sorted_values = sorted(values)
                for p in [0.5, 0.9, 0.99]:
                    idx = int(len(sorted_values) * p)
                    metrics.append(MetricValue(
                        name=f"{self._namespace}_{name}_p{int(p*100)}",
                        value=sorted_values[idx],
                        labels=self._labels.get(key, {})
                    ))

        return metrics

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        for metric in self.get_metrics():
            labels_str = ",".join(f'{k}="{v}"' for k, v in metric.labels.items())
            if labels_str:
                lines.append(f"{metric.name}{{{labels_str}}} {metric.value}")
            else:
                lines.append(f"{metric.name} {metric.value}")
        return "\n".join(lines)


# Global metrics collector
metrics = MetricsCollector(namespace=os.getenv("METRICS_NAMESPACE", "soc"))


# =============================================================================
# Health Checks
# =============================================================================

@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    healthy: bool
    message: str = ""
    latency_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


class HealthChecker:
    """
    Comprehensive health check system.

    Supports:
    - Liveness checks (is the service running?)
    - Readiness checks (is the service ready to receive traffic?)
    - Dependency checks (are external services available?)
    """

    def __init__(self, service_name: str):
        self.service_name = service_name
        self._checks: Dict[str, Callable] = {}
        self._startup_time = time.time()
        self._ready = False

    def register_check(self, name: str, check_func: Callable):
        """Register a health check function."""
        self._checks[name] = check_func

    def set_ready(self, ready: bool = True):
        """Set the service readiness state."""
        self._ready = ready

    async def check_liveness(self) -> HealthCheckResult:
        """Basic liveness check - is the service running?"""
        return HealthCheckResult(
            name="liveness",
            healthy=True,
            message="Service is alive",
            details={
                "uptime_seconds": time.time() - self._startup_time,
                "service": self.service_name
            }
        )

    async def check_readiness(self) -> tuple[bool, List[HealthCheckResult]]:
        """
        Readiness check - is the service ready to receive traffic?

        Returns (overall_healthy, individual_results)
        """
        if not self._ready:
            return False, [HealthCheckResult(
                name="readiness",
                healthy=False,
                message="Service not yet ready"
            )]

        results = []
        all_healthy = True

        for name, check_func in self._checks.items():
            start_time = time.time()
            try:
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()

                latency_ms = (time.time() - start_time) * 1000

                if isinstance(result, HealthCheckResult):
                    result.latency_ms = latency_ms
                    results.append(result)
                    if not result.healthy:
                        all_healthy = False
                elif isinstance(result, bool):
                    results.append(HealthCheckResult(
                        name=name,
                        healthy=result,
                        latency_ms=latency_ms
                    ))
                    if not result:
                        all_healthy = False
                else:
                    results.append(HealthCheckResult(
                        name=name,
                        healthy=True,
                        latency_ms=latency_ms,
                        details={"result": result}
                    ))

            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000
                results.append(HealthCheckResult(
                    name=name,
                    healthy=False,
                    message=str(e),
                    latency_ms=latency_ms
                ))
                all_healthy = False

        return all_healthy, results

    def to_dict(self, results: List[HealthCheckResult]) -> Dict[str, Any]:
        """Convert health check results to dictionary."""
        return {
            "service": self.service_name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "checks": [
                {
                    "name": r.name,
                    "status": "healthy" if r.healthy else "unhealthy",
                    "message": r.message,
                    "latency_ms": round(r.latency_ms, 2),
                    "details": r.details
                }
                for r in results
            ]
        }


# Common health check functions
async def check_bigquery(client) -> HealthCheckResult:
    """Check BigQuery connectivity."""
    try:
        # Simple query to check connectivity
        query = "SELECT 1"
        query_job = client.client.query(query)
        list(query_job.result())
        return HealthCheckResult(name="bigquery", healthy=True)
    except Exception as e:
        return HealthCheckResult(name="bigquery", healthy=False, message=str(e))


async def check_redis(redis_url: str) -> HealthCheckResult:
    """Check Redis connectivity."""
    try:
        import redis.asyncio as aioredis
        client = await aioredis.from_url(redis_url)
        await client.ping()
        await client.close()
        return HealthCheckResult(name="redis", healthy=True)
    except Exception as e:
        return HealthCheckResult(name="redis", healthy=False, message=str(e))


async def check_service(url: str, service_name: str) -> HealthCheckResult:
    """Check external service health."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{url}/health")
            if response.status_code == 200:
                return HealthCheckResult(name=service_name, healthy=True)
            else:
                return HealthCheckResult(
                    name=service_name,
                    healthy=False,
                    message=f"HTTP {response.status_code}"
                )
    except Exception as e:
        return HealthCheckResult(name=service_name, healthy=False, message=str(e))


# =============================================================================
# Request Timing Decorator
# =============================================================================

def timed_operation(operation_name: str):
    """Decorator to time operations and record metrics."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            correlation_id = get_correlation_id()
            tenant_id = get_tenant_context()

            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000

                metrics.observe_histogram(
                    f"{operation_name}_duration_ms",
                    duration_ms,
                    tenant_id=tenant_id or "unknown"
                )
                metrics.increment_counter(
                    f"{operation_name}_total",
                    status="success",
                    tenant_id=tenant_id or "unknown"
                )

                return result

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                metrics.increment_counter(
                    f"{operation_name}_total",
                    status="error",
                    tenant_id=tenant_id or "unknown"
                )
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            tenant_id = get_tenant_context()

            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000

                metrics.observe_histogram(
                    f"{operation_name}_duration_ms",
                    duration_ms,
                    tenant_id=tenant_id or "unknown"
                )
                metrics.increment_counter(
                    f"{operation_name}_total",
                    status="success",
                    tenant_id=tenant_id or "unknown"
                )

                return result

            except Exception:
                metrics.increment_counter(
                    f"{operation_name}_total",
                    status="error",
                    tenant_id=tenant_id or "unknown"
                )
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
