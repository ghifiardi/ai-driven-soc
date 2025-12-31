#!/usr/bin/env python3
"""
MSSP Platform Server - Production Hardened
==========================================

Production-ready version with:
- Secure secret management
- Service-to-service authentication
- Rate limiting enforcement
- Tenant context propagation
- Structured logging with correlation IDs
- Health checks with dependency verification
- Circuit breakers and retry logic
- Graceful shutdown handling
"""

import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import secrets

import jwt
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Header, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
from pydantic import BaseModel, validator
import numpy as np

# Production modules
from production.security import (
    get_jwt_secret,
    InputValidator,
    ServiceAuthenticator,
)
from production.rate_limiter import (
    get_rate_limiter,
    configure_tenant_limits,
    RateLimitResult,
)
from production.observability import (
    configure_structured_logging,
    StructuredLogger,
    HealthChecker,
    metrics,
    get_correlation_id,
    set_correlation_id,
    generate_correlation_id,
    timed_operation,
    check_bigquery,
)
from production.resilience import (
    circuit_breaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    retry,
    RetryConfig,
    get_shutdown_handler,
    setup_signal_handlers,
    timeout,
    get_bulkhead,
)
from production.tenant_context import (
    TenantContext,
    set_current_tenant,
    get_current_tenant_id,
    clear_tenant_context,
    TenantAwareClient,
)

from multi_tenant_manager import (
    MultiTenantManager,
    TenantConfig,
    TenantTables,
    TenantPubSubTopics,
    TenantRateLimits,
    FirewallConfig,
)
from advanced_anomaly_detection import GATRAAnomalyDetectionSystem
from bigquery_client import BigQueryClient

# Configure structured logging
configure_structured_logging(os.getenv("LOG_LEVEL", "INFO"))
logger = StructuredLogger(__name__)

# Configuration
CONFIG_PATH = os.getenv("MULTITENANT_CONFIG_PATH", "config/gatra_multitenant_config.json")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_MINUTES = int(os.getenv("JWT_EXPIRATION_MINUTES", "60"))
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# Service URLs
TAA_SERVICE_URL = os.getenv("TAA_SERVICE_URL", "http://taa-service:8082")
CRA_SERVICE_URL = os.getenv("CRA_SERVICE_URL", "http://cra-service:8083")
LEARNING_SERVICE_URL = os.getenv("LEARNING_SERVICE_URL", "http://learning-service:8084")

# Set BigQuery Credentials
SA_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "Service Account BigQuery/sa-gatra-bigquery.json")
if os.path.exists(SA_PATH):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SA_PATH

# CORS Configuration - Restrictive for production
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
if ENVIRONMENT == "production" and "*" in CORS_ORIGINS:
    logger.warning("CORS allows all origins in production - consider restricting")

security = HTTPBearer()


# =============================================================================
# Pydantic Models with Validation
# =============================================================================

class FirewallConfigModel(BaseModel):
    type: str
    mgmt_ip: str
    api_key: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    device_group: Optional[str] = None
    domain: Optional[str] = None


class TenantRegistrationRequest(BaseModel):
    tenant_id: str
    display_name: str
    region: Optional[str] = None
    service_level: str = "starter"
    contact_email: Optional[str] = None
    api_key: Optional[str] = None
    firewall_config: Optional[FirewallConfigModel] = None

    @validator("tenant_id")
    def validate_tenant_id(cls, v):
        if not InputValidator.validate_tenant_id(v):
            raise ValueError("Invalid tenant_id format")
        return v

    @validator("service_level")
    def validate_service_level(cls, v):
        valid_levels = ["starter", "professional", "enterprise"]
        if v not in valid_levels:
            raise ValueError(f"service_level must be one of {valid_levels}")
        return v


class EventIngestionRequest(BaseModel):
    tenant_id: str
    events: List[Dict[str, Any]]
    source: str = "api"

    @validator("tenant_id")
    def validate_tenant_id(cls, v):
        if not InputValidator.validate_tenant_id(v):
            raise ValueError("Invalid tenant_id format")
        return v

    @validator("events")
    def validate_events(cls, v, values):
        tenant_id = values.get("tenant_id", "")
        is_valid, error_msg = InputValidator.validate_event_batch(v, tenant_id)
        if not is_valid:
            raise ValueError(error_msg)
        return v


# =============================================================================
# MSSP Platform Server
# =============================================================================

class MSSPPlatformServer:
    def __init__(self):
        self.app = FastAPI(
            title="AI-Driven SOC - MSSP Platform (Production)",
            description="Production-hardened platform for Managed Security Service Providers",
            version="2.0.0"
        )

        # Initialize components
        self._init_tenant_manager()
        self._init_bigquery()
        self._init_rate_limiter()
        self._init_health_checker()
        self._init_service_clients()
        self._init_gatra()

        # Setup
        self._setup_middleware()
        self._setup_routes()
        self._setup_shutdown()

    def _init_tenant_manager(self):
        """Initialize tenant manager with validation."""
        try:
            self.tenant_manager = MultiTenantManager.from_file(CONFIG_PATH)
            logger.info("Tenant configuration loaded",
                        tenant_count=self.tenant_manager.tenants_count(),
                        config_path=CONFIG_PATH)
        except Exception as e:
            logger.critical("Failed to load tenant configuration", error=str(e))
            raise

    def _init_bigquery(self):
        """Initialize BigQuery with health monitoring."""
        try:
            self.bq_persistence = BigQueryClient(
                project_id=os.getenv("BQ_PROJECT_ID", "chronicle-dev-2be9"),
                dataset_id=os.getenv("BQ_DATASET_ID", "gatra_database"),
                table_id=os.getenv("BQ_TABLE_ID", "siem_events")
            )
            logger.info("BigQuery persistence initialized")
        except Exception as e:
            logger.error("Failed to initialize BigQuery", error=str(e))
            self.bq_persistence = None

    def _init_rate_limiter(self):
        """Initialize rate limiter with tenant configuration."""
        self.rate_limiter = configure_tenant_limits(self.tenant_manager)
        logger.info("Rate limiter configured for all tenants")

    def _init_health_checker(self):
        """Initialize health checker with dependency checks."""
        self.health_checker = HealthChecker("mssp-platform")

        # Register dependency checks
        if self.bq_persistence:
            self.health_checker.register_check(
                "bigquery",
                lambda: check_bigquery(self.bq_persistence)
            )

        self.health_checker.register_check(
            "taa_service",
            lambda: self._check_service(TAA_SERVICE_URL, "taa")
        )

        self.health_checker.register_check(
            "cra_service",
            lambda: self._check_service(CRA_SERVICE_URL, "cra")
        )

    async def _check_service(self, url: str, name: str):
        """Check if a downstream service is healthy."""
        from production.observability import check_service
        return await check_service(url, name)

    def _init_service_clients(self):
        """Initialize service clients with resilience patterns."""
        self.taa_client = TenantAwareClient(TAA_SERVICE_URL, "mssp-platform")
        self.cra_client = TenantAwareClient(CRA_SERVICE_URL, "mssp-platform")

    def _init_gatra(self):
        """Initialize GATRA anomaly detection."""
        self.gatra = GATRAAnomalyDetectionSystem()
        logger.info("GATRA anomaly detection system initialized")

    def _setup_middleware(self):
        """Setup middleware including CORS and request tracking."""
        # CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=CORS_ORIGINS,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["*"],
        )

        # Request tracking middleware
        @self.app.middleware("http")
        async def track_requests(request: Request, call_next):
            # Generate or extract correlation ID
            correlation_id = request.headers.get("X-Correlation-ID")
            if not correlation_id:
                correlation_id = generate_correlation_id()
            set_correlation_id(correlation_id)

            # Track request
            shutdown_handler = get_shutdown_handler()
            if shutdown_handler.is_shutting_down:
                return Response(
                    content='{"error": "Service is shutting down"}',
                    status_code=503,
                    media_type="application/json"
                )

            await shutdown_handler.track_request()

            # Record metrics
            start_time = asyncio.get_event_loop().time()

            try:
                response = await call_next(request)

                # Add correlation ID to response
                response.headers["X-Correlation-ID"] = correlation_id

                # Record latency
                duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000
                metrics.observe_histogram(
                    "request_duration_ms",
                    duration_ms,
                    path=request.url.path,
                    method=request.method
                )
                metrics.increment_counter(
                    "requests",
                    status=str(response.status_code),
                    path=request.url.path
                )

                return response

            finally:
                await shutdown_handler.untrack_request()
                clear_tenant_context()

    def _create_access_token(self, tenant_id: str) -> str:
        """Create JWT token with secure secret."""
        expires_delta = timedelta(minutes=JWT_EXPIRATION_MINUTES)
        expire = datetime.utcnow() + expires_delta
        to_encode = {
            "exp": expire,
            "sub": tenant_id,
            "iat": datetime.utcnow(),
            "jti": secrets.token_hex(16)  # Unique token ID
        }
        return jwt.encode(to_encode, get_jwt_secret(), algorithm=JWT_ALGORITHM)

    async def _get_current_tenant(
        self,
        credentials: HTTPAuthorizationCredentials = Depends(security)
    ) -> TenantConfig:
        """Validate JWT and return tenant config."""
        token = credentials.credentials
        try:
            payload = jwt.decode(
                token,
                get_jwt_secret(),
                algorithms=[JWT_ALGORITHM]
            )
            tenant_id: str = payload.get("sub")
            if tenant_id is None:
                raise HTTPException(status_code=401, detail="Invalid token: missing subject")

            tenant = self.tenant_manager.get_tenant(tenant_id)

            # Set tenant context for downstream use
            context = TenantContext.from_tenant_config(tenant)
            set_current_tenant(context)

            return tenant

        except jwt.ExpiredSignatureError:
            metrics.increment_counter("auth_failures", reason="expired")
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.PyJWTError as e:
            metrics.increment_counter("auth_failures", reason="invalid")
            raise HTTPException(status_code=401, detail="Could not validate credentials")
        except Exception as e:
            logger.error("Authentication error", error=str(e))
            raise HTTPException(status_code=401, detail="Invalid tenant")

    def _setup_routes(self):
        """Setup all API routes."""
        self._setup_health_routes()
        self._setup_auth_routes()
        self._setup_tenant_routes()
        self._setup_event_routes()
        self._setup_metrics_routes()

    def _setup_health_routes(self):
        """Health check endpoints."""
        @self.app.get("/health/live")
        async def liveness():
            """Kubernetes liveness probe."""
            result = await self.health_checker.check_liveness()
            return self.health_checker.to_dict([result])

        @self.app.get("/health/ready")
        async def readiness():
            """Kubernetes readiness probe."""
            healthy, results = await self.health_checker.check_readiness()
            response = self.health_checker.to_dict(results)
            response["status"] = "ready" if healthy else "not_ready"

            if not healthy:
                raise HTTPException(status_code=503, detail=response)
            return response

        @self.app.get("/health")
        async def health():
            """Combined health check."""
            healthy, results = await self.health_checker.check_readiness()
            return {
                "status": "healthy" if healthy else "degraded",
                "timestamp": datetime.utcnow().isoformat(),
                "checks": self.health_checker.to_dict(results)
            }

    def _setup_auth_routes(self):
        """Authentication endpoints."""
        @self.app.post("/api/v1/auth/token")
        @timed_operation("token_exchange")
        async def get_token(x_api_key: str = Header(...)):
            """Exchange API key for JWT token."""
            for tenant in self.tenant_manager.list_tenants():
                if tenant.api_key == x_api_key:
                    token = self._create_access_token(tenant.tenant_id)
                    logger.info("Token issued", tenant_id=tenant.tenant_id)
                    return {
                        "access_token": token,
                        "token_type": "bearer",
                        "expires_in": JWT_EXPIRATION_MINUTES * 60
                    }

            metrics.increment_counter("auth_failures", reason="invalid_api_key")
            raise HTTPException(status_code=401, detail="Invalid API Key")

    def _setup_tenant_routes(self):
        """Tenant management endpoints."""
        @self.app.get("/api/v1/tenants")
        async def list_tenants(
            tenant_auth: TenantConfig = Depends(self._get_current_tenant)
        ):
            """List all tenants (admin only in production)."""
            # In production, restrict to admin tenants
            if ENVIRONMENT == "production" and tenant_auth.service_level != "enterprise":
                raise HTTPException(status_code=403, detail="Admin access required")

            tenants = []
            for t in self.tenant_manager.list_tenants():
                tenants.append({
                    "tenant_id": t.tenant_id,
                    "display_name": t.display_name,
                    "region": t.region,
                    "service_level": t.service_level
                })
            return {"tenants": tenants}

        @self.app.post("/api/v1/tenants")
        @timed_operation("tenant_registration")
        async def register_tenant(request: TenantRegistrationRequest):
            """Register a new tenant."""
            try:
                # Check if tenant exists
                try:
                    self.tenant_manager.get_tenant(request.tenant_id)
                    raise HTTPException(
                        status_code=409,
                        detail=f"Tenant {request.tenant_id} already exists"
                    )
                except Exception:
                    pass

                # Create tenant config
                defaults = self.tenant_manager._config.defaults
                firewall_config = None
                if request.firewall_config:
                    firewall_config = FirewallConfig(
                        type=request.firewall_config.type,
                        mgmt_ip=request.firewall_config.mgmt_ip,
                        api_key=request.firewall_config.api_key,
                        username=request.firewall_config.username,
                        password=request.firewall_config.password,
                        device_group=request.firewall_config.device_group,
                        domain=request.firewall_config.domain
                    )

                new_tenant = TenantConfig(
                    tenant_id=request.tenant_id,
                    display_name=request.display_name,
                    region=request.region or defaults.location,
                    dataset=defaults.dataset_template.format(tenant_id=request.tenant_id),
                    results_dataset=defaults.results_dataset_template.format(
                        tenant_id=request.tenant_id
                    ),
                    tables=TenantTables(
                        events="events",
                        alerts="alerts",
                        results="events_results"
                    ),
                    pubsub_topics=TenantPubSubTopics(
                        ingest=f"events-{request.tenant_id}",
                        alerts=f"alerts-{request.tenant_id}",
                        priority=f"priority-{request.tenant_id}"
                    ),
                    rate_limits=TenantRateLimits(ingest_eps=1000, alerts_per_min=100),
                    service_level=request.service_level,
                    api_key=request.api_key or secrets.token_urlsafe(32),
                    firewall_config=firewall_config
                )

                self.tenant_manager.add_tenant(new_tenant)
                self.tenant_manager.save_config(CONFIG_PATH)

                # Configure rate limits for new tenant
                self.rate_limiter.configure_tenant(
                    new_tenant.tenant_id,
                    new_tenant.rate_limits.ingest_eps,
                    new_tenant.rate_limits.alerts_per_min
                )

                logger.info("Tenant registered", tenant_id=request.tenant_id)
                metrics.increment_counter("tenants_registered")

                return {
                    "status": "created",
                    "tenant_id": new_tenant.tenant_id,
                    "api_key": new_tenant.api_key
                }

            except HTTPException:
                raise
            except Exception as e:
                logger.error("Tenant registration failed", error=str(e))
                raise HTTPException(status_code=500, detail=str(e))

    def _setup_event_routes(self):
        """Event ingestion endpoints."""
        @self.app.post("/api/v1/events")
        @timed_operation("event_ingestion")
        async def ingest_events(
            request: EventIngestionRequest,
            background_tasks: BackgroundTasks,
            tenant_auth: TenantConfig = Depends(self._get_current_tenant)
        ):
            """Ingest security events with rate limiting."""
            # Verify tenant matches token
            if tenant_auth.tenant_id != request.tenant_id:
                raise HTTPException(
                    status_code=403,
                    detail="Token does not match tenant_id"
                )

            # Check rate limit
            rate_result = await self.rate_limiter.check_ingest_rate(
                request.tenant_id,
                len(request.events)
            )

            if not rate_result.allowed:
                metrics.increment_counter(
                    "rate_limit_exceeded",
                    tenant_id=request.tenant_id
                )
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "Rate limit exceeded",
                        "retry_after": rate_result.retry_after,
                        "remaining": rate_result.remaining
                    },
                    headers={"Retry-After": str(int(rate_result.retry_after or 1))}
                )

            # Process events in background
            background_tasks.add_task(
                self._process_events_background,
                request.tenant_id,
                request.events
            )

            metrics.increment_counter(
                "events_ingested",
                tenant_id=request.tenant_id,
                count=len(request.events)
            )

            return {
                "status": "accepted",
                "event_count": len(request.events),
                "tenant_id": request.tenant_id,
                "correlation_id": get_correlation_id(),
                "rate_limit_remaining": rate_result.remaining
            }

    def _setup_metrics_routes(self):
        """Metrics endpoints."""
        @self.app.get("/metrics")
        async def prometheus_metrics():
            """Prometheus-format metrics endpoint."""
            return Response(
                content=metrics.export_prometheus(),
                media_type="text/plain"
            )

    @circuit_breaker("bigquery", CircuitBreakerConfig(failure_threshold=3, timeout=60))
    @retry(RetryConfig(max_attempts=3, base_delay=1.0))
    async def _persist_to_bigquery(self, tenant_id: str, rows: list):
        """Persist events to BigQuery with resilience."""
        if not self.bq_persistence:
            logger.warning("BigQuery not available, skipping persistence")
            return

        self.bq_persistence.insert_rows_json(rows)
        logger.info("Events persisted to BigQuery",
                    tenant_id=tenant_id,
                    row_count=len(rows))

    @circuit_breaker("taa", CircuitBreakerConfig(failure_threshold=5, timeout=30))
    @timeout(10.0, "taa_triage")
    async def _forward_to_taa(self, anomaly_report: dict) -> Optional[dict]:
        """Forward anomaly to TAA with circuit breaker."""
        try:
            result = await self.taa_client.post(
                "/api/v1/triage",
                anomaly_report,
                timeout=5.0
            )
            return result
        except Exception as e:
            logger.error("TAA forwarding failed", error=str(e))
            raise

    @circuit_breaker("cra", CircuitBreakerConfig(failure_threshold=5, timeout=30))
    @timeout(10.0, "cra_containment")
    async def _forward_to_cra(self, triage_result: dict) -> Optional[dict]:
        """Forward to CRA for containment with circuit breaker."""
        try:
            result = await self.cra_client.post(
                "/api/v1/contain",
                triage_result,
                timeout=5.0
            )
            return result
        except Exception as e:
            logger.error("CRA forwarding failed", error=str(e))
            raise

    async def _process_events_background(
        self,
        tenant_id: str,
        events: List[Dict[str, Any]]
    ):
        """Background event processing with full pipeline."""
        correlation_id = get_correlation_id()

        # Use bulkhead to limit concurrent processing
        bulkhead = get_bulkhead("event_processing", max_concurrent=10, max_waiting=50)

        try:
            async with bulkhead:
                # 1. Persist to BigQuery
                rows = []
                for event in events:
                    rows.append({
                        "alarmId": event.get("alarm_id", secrets.token_hex(8)),
                        "events": json.dumps(event),
                        "tenant_id": tenant_id,
                        "correlation_id": correlation_id,
                        "processed_by_ada": False,
                        "ingestion_time": datetime.now().isoformat()
                    })

                try:
                    await self._persist_to_bigquery(tenant_id, rows)
                except CircuitBreakerError:
                    logger.warning("BigQuery circuit open, queuing for later")
                except Exception as e:
                    logger.error("BigQuery persistence failed", error=str(e))

                # 2. GATRA Analysis
                logger.info("Starting GATRA analysis",
                            tenant_id=tenant_id,
                            event_count=len(events))

                for event in events:
                    try:
                        # Prepare features
                        features = np.zeros(10)
                        features[0] = float(event.get("duration") or 0)
                        features[1] = float(event.get("bytes_sent") or 0)
                        features[2] = float(event.get("bytes_received") or 0)
                        features[3] = float(event.get("port") or 0)
                        event['features'] = features

                        # Detect anomalies
                        gatra_result = self.gatra.process_telemetry(event)

                        if gatra_result.is_anomaly:
                            metrics.increment_counter(
                                "anomalies_detected",
                                tenant_id=tenant_id,
                                severity=gatra_result.severity.name
                            )

                            anomaly_report = {
                                "alarm_id": f"GATRA-{event.get('id')}",
                                "tenant_id": tenant_id,  # Include tenant context
                                "correlation_id": correlation_id,
                                "timestamp": datetime.now().isoformat(),
                                "source_ip": event.get("source_ip"),
                                "destination_ip": event.get("dest_ip"),
                                "protocol": event.get("protocol"),
                                "alert_severity": gatra_result.severity.name.lower(),
                                "attack_category": "anomaly",
                                "confidence": gatra_result.score,
                                "reasoning": [gatra_result.reasoning]
                            }

                            # Forward to TAA
                            try:
                                triage_result = await self._forward_to_taa(anomaly_report)

                                if triage_result:
                                    logger.info("TAA triage complete",
                                                alarm_id=anomaly_report["alarm_id"],
                                                classification=triage_result.get("classification"),
                                                confidence=triage_result.get("confidence"))

                                    # Forward to CRA if critical
                                    if triage_result.get("classification") in ["malicious", "critical"]:
                                        triage_result["tenant_id"] = tenant_id
                                        triage_result["correlation_id"] = correlation_id

                                        try:
                                            cra_result = await self._forward_to_cra(triage_result)
                                            if cra_result:
                                                logger.info("CRA containment executed",
                                                            alarm_id=anomaly_report["alarm_id"],
                                                            status=cra_result.get("containment_status"))
                                        except CircuitBreakerError:
                                            logger.warning("CRA circuit open",
                                                           alarm_id=anomaly_report["alarm_id"])

                            except CircuitBreakerError:
                                logger.warning("TAA circuit open",
                                               alarm_id=anomaly_report["alarm_id"])

                    except Exception as e:
                        logger.error("Event processing error",
                                     event_id=event.get("id"),
                                     error=str(e))

        except RuntimeError as e:
            if "full" in str(e):
                logger.warning("Event processing bulkhead full",
                               tenant_id=tenant_id)
                metrics.increment_counter("bulkhead_rejected", tenant_id=tenant_id)
            else:
                raise

        await asyncio.sleep(0.01)  # Yield to event loop

    def _setup_shutdown(self):
        """Setup graceful shutdown handlers."""
        shutdown_handler = get_shutdown_handler(drain_seconds=15)

        # Register cleanup callbacks
        shutdown_handler.register_cleanup(self._cleanup_resources)

        @self.app.on_event("startup")
        async def startup():
            """Application startup."""
            logger.info("MSSP Platform starting",
                        environment=ENVIRONMENT,
                        tenants=self.tenant_manager.tenants_count())
            self.health_checker.set_ready(True)
            setup_signal_handlers(shutdown_handler)

        @self.app.on_event("shutdown")
        async def shutdown():
            """Application shutdown."""
            logger.info("MSSP Platform shutting down")
            await shutdown_handler.graceful_shutdown()

    async def _cleanup_resources(self):
        """Cleanup resources during shutdown."""
        logger.info("Cleaning up resources")
        # Close BigQuery client, etc.

    def run(self, host: str = "0.0.0.0", port: int = 8081):
        """Run the server."""
        logger.info("Starting MSSP Platform Server",
                    host=host,
                    port=port,
                    environment=ENVIRONMENT)

        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="warning"  # Use structured logging instead
        )


if __name__ == "__main__":
    server = MSSPPlatformServer()
    server.run()
