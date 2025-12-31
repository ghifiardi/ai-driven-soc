"""
Tenant Context Module
=====================

Provides tenant context propagation across services and async boundaries.
"""

import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from contextvars import ContextVar

logger = logging.getLogger(__name__)

# Context variables for tenant propagation
_tenant_id: ContextVar[str] = ContextVar("tenant_id", default="")
_tenant_config: ContextVar[Optional[Dict[str, Any]]] = ContextVar("tenant_config", default=None)


@dataclass
class TenantContext:
    """Immutable tenant context for request processing."""
    tenant_id: str
    dataset: str
    region: str
    service_level: str
    rate_limits: Dict[str, int]

    def to_headers(self) -> Dict[str, str]:
        """Convert to HTTP headers for service-to-service calls."""
        return {
            "X-Tenant-ID": self.tenant_id,
            "X-Tenant-Dataset": self.dataset,
            "X-Tenant-Region": self.region,
            "X-Tenant-Service-Level": self.service_level,
        }

    @classmethod
    def from_headers(cls, headers: Dict[str, str]) -> Optional["TenantContext"]:
        """Create from HTTP headers."""
        tenant_id = headers.get("X-Tenant-ID") or headers.get("x-tenant-id")
        if not tenant_id:
            return None

        return cls(
            tenant_id=tenant_id,
            dataset=headers.get("X-Tenant-Dataset", headers.get("x-tenant-dataset", "")),
            region=headers.get("X-Tenant-Region", headers.get("x-tenant-region", "")),
            service_level=headers.get("X-Tenant-Service-Level", headers.get("x-tenant-service-level", "starter")),
            rate_limits={}
        )

    @classmethod
    def from_tenant_config(cls, config) -> "TenantContext":
        """Create from TenantConfig object."""
        return cls(
            tenant_id=config.tenant_id,
            dataset=config.dataset,
            region=config.region,
            service_level=config.service_level,
            rate_limits={
                "ingest_eps": config.rate_limits.ingest_eps,
                "alerts_per_min": config.rate_limits.alerts_per_min
            }
        )


def set_current_tenant(context: TenantContext):
    """Set the current tenant context."""
    _tenant_id.set(context.tenant_id)
    _tenant_config.set({
        "tenant_id": context.tenant_id,
        "dataset": context.dataset,
        "region": context.region,
        "service_level": context.service_level,
        "rate_limits": context.rate_limits
    })

    # Also set in observability context
    from production.observability import set_tenant_context
    set_tenant_context(context.tenant_id)


def get_current_tenant_id() -> str:
    """Get the current tenant ID."""
    return _tenant_id.get()


def get_current_tenant_config() -> Optional[Dict[str, Any]]:
    """Get the current tenant configuration."""
    return _tenant_config.get()


def clear_tenant_context():
    """Clear the tenant context."""
    _tenant_id.set("")
    _tenant_config.set(None)


class TenantContextMiddleware:
    """
    FastAPI middleware for tenant context propagation.

    Extracts tenant context from JWT or headers and makes it
    available throughout the request lifecycle.
    """

    def __init__(self, app, tenant_manager):
        self.app = app
        self.tenant_manager = tenant_manager

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Try to extract tenant from headers first (for service-to-service)
        headers = dict(scope.get("headers", []))
        headers = {k.decode(): v.decode() for k, v in headers.items()}

        tenant_context = TenantContext.from_headers(headers)

        if tenant_context:
            set_current_tenant(tenant_context)
            logger.debug(f"Tenant context set from headers: {tenant_context.tenant_id}")

        try:
            await self.app(scope, receive, send)
        finally:
            clear_tenant_context()


def require_tenant_context(func):
    """Decorator that ensures tenant context is available."""
    from functools import wraps
    import asyncio

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        tenant_id = get_current_tenant_id()
        if not tenant_id:
            raise ValueError("Tenant context required but not set")
        return await func(*args, **kwargs)

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        tenant_id = get_current_tenant_id()
        if not tenant_id:
            raise ValueError("Tenant context required but not set")
        return func(*args, **kwargs)

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


# =============================================================================
# Service Client with Tenant Context
# =============================================================================

class TenantAwareClient:
    """
    HTTP client that automatically propagates tenant context.
    """

    def __init__(self, base_url: str, service_name: str):
        self.base_url = base_url.rstrip("/")
        self.service_name = service_name

    def _get_headers(self) -> Dict[str, str]:
        """Get headers including tenant context and service auth."""
        headers = {}

        # Add tenant context
        config = get_current_tenant_config()
        if config:
            headers["X-Tenant-ID"] = config["tenant_id"]
            headers["X-Tenant-Dataset"] = config.get("dataset", "")
            headers["X-Tenant-Region"] = config.get("region", "")
            headers["X-Tenant-Service-Level"] = config.get("service_level", "")

        # Add correlation ID
        from production.observability import get_correlation_id
        headers["X-Correlation-ID"] = get_correlation_id()

        # Add service authentication
        try:
            from production.security import ServiceAuthenticator
            auth = ServiceAuthenticator(self.service_name)
            headers.update(auth.create_auth_headers())
        except Exception as e:
            logger.warning(f"Failed to add service auth headers: {e}")

        return headers

    async def post(self, path: str, json_data: Dict[str, Any], timeout: float = 10.0) -> Dict[str, Any]:
        """Make a POST request with tenant context."""
        import httpx
        import json as json_module

        url = f"{self.base_url}{path}"
        headers = self._get_headers()
        body = json_module.dumps(json_data).encode()

        # Update auth headers with body
        try:
            from production.security import ServiceAuthenticator
            auth = ServiceAuthenticator(self.service_name)
            headers.update(auth.create_auth_headers(body))
        except Exception:
            pass

        headers["Content-Type"] = "application/json"

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, content=body, headers=headers)
            response.raise_for_status()
            return response.json()

    async def get(self, path: str, timeout: float = 10.0) -> Dict[str, Any]:
        """Make a GET request with tenant context."""
        import httpx

        url = f"{self.base_url}{path}"
        headers = self._get_headers()

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            return response.json()


# =============================================================================
# Tenant-Scoped BigQuery Client
# =============================================================================

class TenantScopedBigQueryClient:
    """
    BigQuery client that enforces tenant isolation.

    All queries are automatically scoped to the tenant's dataset.
    """

    def __init__(self, base_client, tenant_manager):
        self._base_client = base_client
        self._tenant_manager = tenant_manager

    def for_current_tenant(self):
        """Get a client scoped to the current tenant."""
        tenant_id = get_current_tenant_id()
        if not tenant_id:
            raise ValueError("No tenant context set")

        from bigquery_client import BigQueryClient
        return BigQueryClient.for_tenant(self._tenant_manager, tenant_id)

    def insert_rows_for_tenant(self, rows: list, tenant_id: Optional[str] = None):
        """Insert rows into the tenant's table."""
        tid = tenant_id or get_current_tenant_id()
        if not tid:
            raise ValueError("Tenant ID required")

        from bigquery_client import BigQueryClient
        client = BigQueryClient.for_tenant(self._tenant_manager, tid)
        return client.insert_rows_json(rows)

    def query_tenant_data(self, query_template: str, tenant_id: Optional[str] = None):
        """Execute a query scoped to the tenant's dataset."""
        tid = tenant_id or get_current_tenant_id()
        if not tid:
            raise ValueError("Tenant ID required")

        tenant = self._tenant_manager.get_tenant(tid)

        # Replace placeholders with tenant-specific values
        query = query_template.format(
            project_id=self._tenant_manager.project_id,
            dataset=tenant.dataset,
            results_dataset=tenant.results_dataset,
            events_table=tenant.tables.events,
            alerts_table=tenant.tables.alerts,
            results_table=tenant.tables.results
        )

        from bigquery_client import BigQueryClient
        client = BigQueryClient.for_tenant(self._tenant_manager, tid)
        return client.client.query(query).result()
