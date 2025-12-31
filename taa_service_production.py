#!/usr/bin/env python3
"""
Triage & Analysis Agent (TAA) Service - Production Hardened
============================================================

Production-ready TAA service with:
- Tenant context awareness
- Service authentication
- Structured logging
- Health checks with dependencies
- Graceful shutdown
"""

import os
import time
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import FastAPI, Body, HTTPException, Request, Response, Header
from pydantic import BaseModel, validator
import uvicorn

# Production modules
from production.security import (
    ServiceAuthenticator,
    InputValidator,
)
from production.observability import (
    configure_structured_logging,
    StructuredLogger,
    HealthChecker,
    HealthCheckResult,
    metrics,
    get_correlation_id,
    set_correlation_id,
    timed_operation,
)
from production.resilience import (
    get_shutdown_handler,
    setup_signal_handlers,
)
from production.tenant_context import (
    TenantContext,
    set_current_tenant,
    get_current_tenant_id,
    clear_tenant_context,
    require_tenant_context,
)

from bigquery_client import BigQueryClient
from enhanced_taa_agent import EnhancedTAAgent

# Configure structured logging
configure_structured_logging(os.getenv("LOG_LEVEL", "INFO"))
logger = StructuredLogger("TAA-Service")

# Configuration
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
SERVICE_NAME = "taa-service"

# Set BigQuery Credentials
SA_PATH = os.getenv(
    "GOOGLE_APPLICATION_CREDENTIALS",
    "Service Account BigQuery/sa-gatra-bigquery.json"
)
if os.path.exists(SA_PATH):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SA_PATH


# =============================================================================
# Request Models
# =============================================================================

class TriageRequest(BaseModel):
    """Alert triage request with tenant context."""
    alarm_id: str
    tenant_id: Optional[str] = None
    correlation_id: Optional[str] = None
    timestamp: Optional[str] = None
    source_ip: Optional[str] = None
    destination_ip: Optional[str] = None
    protocol: Optional[str] = None
    alert_severity: Optional[str] = None
    attack_category: Optional[str] = None
    confidence: Optional[float] = None
    reasoning: Optional[List[str]] = None

    @validator("alarm_id")
    def validate_alarm_id(cls, v):
        if not v or len(v) > 256:
            raise ValueError("Invalid alarm_id")
        return v


class BatchTriageRequest(BaseModel):
    """Batch triage request."""
    alerts: List[Dict[str, Any]]
    tenant_id: Optional[str] = None

    @validator("alerts")
    def validate_alerts(cls, v):
        if len(v) > 100:
            raise ValueError("Maximum 100 alerts per batch")
        return v


# =============================================================================
# TAA Service
# =============================================================================

class TAAService:
    def __init__(self):
        self.app = FastAPI(
            title="GATRA SOC - Triage & Analysis Service (Production)",
            description="Production-hardened TAA with tenant isolation",
            version="2.0.0"
        )

        self._init_components()
        self._setup_middleware()
        self._setup_routes()
        self._setup_shutdown()

    def _init_components(self):
        """Initialize service components."""
        # TAA Agent
        self.agent = EnhancedTAAgent()
        logger.info("TAA Agent initialized")

        # BigQuery (optional)
        try:
            self.bq_client = BigQueryClient(
                project_id=os.getenv("BQ_PROJECT_ID", "chronicle-dev-2be9"),
                dataset_id=os.getenv("BQ_DATASET_ID", "gatra_database"),
                table_id=os.getenv("BQ_TABLE_ID", "taa_state")
            )
            logger.info("BigQuery persistence initialized")
        except Exception as e:
            logger.warning("BigQuery not available", error=str(e))
            self.bq_client = None

        # Health checker
        self.health_checker = HealthChecker(SERVICE_NAME)
        self.health_checker.register_check("agent", self._check_agent)
        if self.bq_client:
            self.health_checker.register_check("bigquery", self._check_bigquery)

        # Threat intelligence cache
        self.intelligence_cache = {
            "malicious_ips": set(),
            "last_update": 0
        }

    async def _check_agent(self) -> HealthCheckResult:
        """Check if TAA agent is healthy."""
        try:
            stats = self.agent.get_statistics()
            return HealthCheckResult(
                name="agent",
                healthy=True,
                details=stats
            )
        except Exception as e:
            return HealthCheckResult(
                name="agent",
                healthy=False,
                message=str(e)
            )

    async def _check_bigquery(self) -> HealthCheckResult:
        """Check BigQuery connectivity."""
        try:
            query = "SELECT 1"
            self.bq_client.client.query(query).result()
            return HealthCheckResult(name="bigquery", healthy=True)
        except Exception as e:
            return HealthCheckResult(
                name="bigquery",
                healthy=False,
                message=str(e)
            )

    def _setup_middleware(self):
        """Setup request middleware."""
        @self.app.middleware("http")
        async def process_request(request: Request, call_next):
            # Extract correlation ID
            correlation_id = request.headers.get("X-Correlation-ID")
            if correlation_id:
                set_correlation_id(correlation_id)

            # Extract tenant context from headers
            tenant_id = request.headers.get("X-Tenant-ID")
            if tenant_id:
                context = TenantContext(
                    tenant_id=tenant_id,
                    dataset=request.headers.get("X-Tenant-Dataset", ""),
                    region=request.headers.get("X-Tenant-Region", ""),
                    service_level=request.headers.get("X-Tenant-Service-Level", "starter"),
                    rate_limits={}
                )
                set_current_tenant(context)

            # Verify service authentication (optional in dev)
            if ENVIRONMENT == "production":
                service_name = request.headers.get("X-Service-Name")
                timestamp = request.headers.get("X-Service-Timestamp")
                signature = request.headers.get("X-Service-Signature")

                if service_name and timestamp and signature:
                    body = await request.body()
                    if not ServiceAuthenticator.verify_request(
                        service_name, timestamp, signature, body
                    ):
                        return Response(
                            content='{"error": "Invalid service credentials"}',
                            status_code=403,
                            media_type="application/json"
                        )

            # Check shutdown status
            shutdown_handler = get_shutdown_handler()
            if shutdown_handler.is_shutting_down:
                return Response(
                    content='{"error": "Service is shutting down"}',
                    status_code=503,
                    media_type="application/json"
                )

            await shutdown_handler.track_request()

            try:
                response = await call_next(request)
                return response
            finally:
                await shutdown_handler.untrack_request()
                clear_tenant_context()

    def _setup_routes(self):
        """Setup API routes."""
        self._setup_health_routes()
        self._setup_triage_routes()
        self._setup_metrics_routes()

    def _setup_health_routes(self):
        """Health check endpoints."""
        @self.app.get("/health/live")
        async def liveness():
            result = await self.health_checker.check_liveness()
            return self.health_checker.to_dict([result])

        @self.app.get("/health/ready")
        async def readiness():
            healthy, results = await self.health_checker.check_readiness()
            response = self.health_checker.to_dict(results)
            if not healthy:
                raise HTTPException(status_code=503, detail=response)
            return response

        @self.app.get("/health")
        async def health():
            """Legacy health endpoint."""
            try:
                stats = self.agent.get_statistics()
                return {"status": "healthy", "stats": stats}
            except Exception as e:
                return {"status": "degraded", "error": str(e)}

    def _setup_triage_routes(self):
        """Triage endpoints."""
        @self.app.post("/api/v1/triage")
        @timed_operation("triage_alert")
        async def triage_alert(
            request: Request,
            alert_data: Dict[str, Any] = Body(...)
        ):
            """Analyze an alert and produce a classification."""
            try:
                alarm_id = alert_data.get("alarm_id", "unknown")
                tenant_id = alert_data.get("tenant_id") or get_current_tenant_id()
                correlation_id = get_correlation_id()

                logger.info("Triage request received",
                            alarm_id=alarm_id,
                            tenant_id=tenant_id)

                # Threat intelligence check
                src_ip = alert_data.get("source_ip")
                if src_ip and src_ip in self.intelligence_cache["malicious_ips"]:
                    logger.warning("Threat intel match",
                                   source_ip=src_ip,
                                   alarm_id=alarm_id)
                    alert_data["alert_severity"] = "critical"
                    metrics.increment_counter(
                        "threat_intel_matches",
                        tenant_id=tenant_id or "unknown"
                    )

                # Run triage
                result = await self.agent.analyze_alert(alert_data)

                # Record metrics
                metrics.increment_counter(
                    "triage_completed",
                    tenant_id=tenant_id or "unknown",
                    classification=result.classification
                )

                # Persist result (optional)
                if self.bq_client and tenant_id:
                    try:
                        self.bq_client.insert_rows_json([{
                            "alarm_id": alarm_id,
                            "tenant_id": tenant_id,
                            "correlation_id": correlation_id,
                            "classification": result.classification,
                            "confidence": result.confidence,
                            "threat_score": result.threat_score,
                            "is_anomaly": result.is_anomaly,
                            "processing_time": result.processing_time,
                            "event_time": datetime.now().isoformat()
                        }])
                    except Exception as e:
                        logger.warning("Failed to persist triage result",
                                       error=str(e))

                logger.info("Triage complete",
                            alarm_id=alarm_id,
                            classification=result.classification,
                            confidence=result.confidence)

                return {
                    "alarm_id": result.alarm_id,
                    "tenant_id": tenant_id,
                    "correlation_id": correlation_id,
                    "classification": result.classification,
                    "confidence": result.confidence,
                    "threat_score": result.threat_score,
                    "is_anomaly": result.is_anomaly,
                    "processing_time": result.processing_time
                }

            except Exception as e:
                logger.error("Triage error",
                             alarm_id=alert_data.get("alarm_id"),
                             error=str(e))
                metrics.increment_counter("triage_errors")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/v1/triage/batch")
        @timed_operation("triage_batch")
        async def triage_batch(
            request: BatchTriageRequest
        ):
            """Analyze a batch of alerts."""
            try:
                tenant_id = request.tenant_id or get_current_tenant_id()

                logger.info("Batch triage request",
                            alert_count=len(request.alerts),
                            tenant_id=tenant_id)

                results = await self.agent.analyze_batch(request.alerts)

                metrics.observe_histogram(
                    "batch_size",
                    len(request.alerts),
                    tenant_id=tenant_id or "unknown"
                )

                return [{
                    "alarm_id": r.alarm_id,
                    "classification": r.classification,
                    "confidence": r.confidence,
                    "threat_score": r.threat_score,
                    "is_anomaly": r.is_anomaly
                } for r in results]

            except Exception as e:
                logger.error("Batch triage error", error=str(e))
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/v1/threat-intel/update")
        async def update_threat_intel(intel_data: Dict[str, Any] = Body(...)):
            """Update threat intelligence cache."""
            try:
                if "malicious_ips" in intel_data:
                    new_ips = set(intel_data["malicious_ips"])
                    self.intelligence_cache["malicious_ips"].update(new_ips)
                    self.intelligence_cache["last_update"] = time.time()
                    logger.info("Threat intel updated",
                                new_ips=len(new_ips),
                                total_ips=len(self.intelligence_cache["malicious_ips"]))

                return {
                    "status": "updated",
                    "total_ips": len(self.intelligence_cache["malicious_ips"]),
                    "last_update": self.intelligence_cache["last_update"]
                }
            except Exception as e:
                logger.error("Threat intel update failed", error=str(e))
                raise HTTPException(status_code=500, detail=str(e))

    def _setup_metrics_routes(self):
        """Metrics endpoint."""
        @self.app.get("/metrics")
        async def prometheus_metrics():
            return Response(
                content=metrics.export_prometheus(),
                media_type="text/plain"
            )

    def _setup_shutdown(self):
        """Setup graceful shutdown."""
        shutdown_handler = get_shutdown_handler(drain_seconds=10)

        @self.app.on_event("startup")
        async def startup():
            logger.info("TAA Service starting", environment=ENVIRONMENT)
            self.health_checker.set_ready(True)
            setup_signal_handlers(shutdown_handler)

            # Start threat intel refresh task
            asyncio.create_task(self._refresh_threat_intel())

        @self.app.on_event("shutdown")
        async def shutdown():
            logger.info("TAA Service shutting down")
            await shutdown_handler.graceful_shutdown()

    async def _refresh_threat_intel(self):
        """Background task to refresh threat intelligence."""
        while True:
            try:
                now = time.time()
                if now - self.intelligence_cache["last_update"] > 300:
                    # In production, fetch from real threat intel feed
                    logger.info("Refreshing threat intelligence")
                    self.intelligence_cache["last_update"] = now
            except Exception as e:
                logger.error("Threat intel refresh failed", error=str(e))

            await asyncio.sleep(60)

    def run(self, host: str = "0.0.0.0", port: int = 8082):
        """Run the service."""
        logger.info("Starting TAA Service", host=host, port=port)
        uvicorn.run(self.app, host=host, port=port, log_level="warning")


# Create app instance for uvicorn
service = TAAService()
app = service.app

if __name__ == "__main__":
    service.run()
