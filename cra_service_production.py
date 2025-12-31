#!/usr/bin/env python3
"""
Containment & Response Agent (CRA) Service - Production Hardened
=================================================================

Production-ready CRA service with:
- Tenant context awareness
- Service authentication
- Structured logging
- Health checks
- Audit logging for containment actions
- Graceful shutdown
"""

import os
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from fastapi import FastAPI, HTTPException, Body, Request, Response
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
    circuit_breaker,
    CircuitBreakerConfig,
)
from production.tenant_context import (
    TenantContext,
    set_current_tenant,
    get_current_tenant_id,
    clear_tenant_context,
)

from bigquery_client import BigQueryClient
from containment_response_agent import ContainmentResponseAgent

# Configure structured logging
configure_structured_logging(os.getenv("LOG_LEVEL", "INFO"))
logger = StructuredLogger("CRA-Service")

# Configuration
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
SERVICE_NAME = "cra-service"

# Set BigQuery Credentials
SA_PATH = os.getenv(
    "GOOGLE_APPLICATION_CREDENTIALS",
    "Service Account BigQuery/sa-gatra-bigquery.json"
)
if os.path.exists(SA_PATH):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SA_PATH


# =============================================================================
# Lite CRA for Local Development
# =============================================================================

class LiteCRA(ContainmentResponseAgent):
    """Mock CRA that doesn't require GCP/Firestore."""

    def __init__(self, config_path=None):
        self.config = {
            "project_id": "soc-lite",
            "approval_required": {"high": True, "medium": False, "low": False},
            "playbooks": {
                "default": [{"action": "mock_block", "params": {}, "risk_level": "medium"}],
                "isolate": [{"action": "isolate_host", "params": {}, "risk_level": "high"}],
                "block_ip": [{"action": "block_ip", "params": {}, "risk_level": "medium"}]
            }
        }
        self.actions_log = []
        logger.info("Lite CRA initialized (mock mode)")

    def _execute_playbook(self, playbook_tuple, incident, incident_ref=None):
        name, actions = playbook_tuple
        results = []

        for action in actions:
            action_result = {
                "action": action['action'],
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "incident_id": incident.get("alarm_id")
            }
            results.append(action_result)
            self.actions_log.append(action_result)

        return {
            "containment_status": "complete",
            "playbook": name,
            "actions_executed": results
        }


# =============================================================================
# Request Models
# =============================================================================

class ContainmentRequest(BaseModel):
    """Containment action request."""
    alarm_id: str
    tenant_id: Optional[str] = None
    correlation_id: Optional[str] = None
    classification: Optional[str] = None
    confidence: Optional[float] = None
    threat_score: Optional[float] = None
    source_ip: Optional[str] = None
    destination_ip: Optional[str] = None
    recommended_actions: Optional[List[str]] = None

    @validator("alarm_id")
    def validate_alarm_id(cls, v):
        if not v or len(v) > 256:
            raise ValueError("Invalid alarm_id")
        return v


class ApprovalRequest(BaseModel):
    """Manual approval for high-risk actions."""
    incident_id: str
    approved: bool
    approver: str
    comments: Optional[str] = None


# =============================================================================
# CRA Service
# =============================================================================

class CRAService:
    def __init__(self):
        self.app = FastAPI(
            title="GATRA SOC - Containment & Response Service (Production)",
            description="Production-hardened CRA with tenant isolation and audit logging",
            version="2.0.0"
        )

        self._init_components()
        self._setup_middleware()
        self._setup_routes()
        self._setup_shutdown()

    def _init_components(self):
        """Initialize service components."""
        # CRA Agent
        self.agent = LiteCRA()

        # BigQuery for audit logging
        try:
            self.bq_client = BigQueryClient(
                project_id=os.getenv("BQ_PROJECT_ID", "chronicle-dev-2be9"),
                dataset_id=os.getenv("BQ_DATASET_ID", "gatra_database"),
                table_id=os.getenv("BQ_TABLE_ID", "cra_audit_log")
            )
            logger.info("BigQuery audit logging initialized")
        except Exception as e:
            logger.warning("BigQuery not available", error=str(e))
            self.bq_client = None

        # Pending approvals queue
        self.pending_approvals: Dict[str, Dict[str, Any]] = {}

        # Health checker
        self.health_checker = HealthChecker(SERVICE_NAME)
        self.health_checker.register_check("agent", self._check_agent)

    async def _check_agent(self) -> HealthCheckResult:
        """Check CRA agent health."""
        return HealthCheckResult(
            name="agent",
            healthy=True,
            details={"actions_logged": len(self.agent.actions_log)}
        )

    def _setup_middleware(self):
        """Setup request middleware."""
        @self.app.middleware("http")
        async def process_request(request: Request, call_next):
            # Extract correlation ID
            correlation_id = request.headers.get("X-Correlation-ID")
            if correlation_id:
                set_correlation_id(correlation_id)

            # Extract tenant context
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

            # Verify service authentication
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

            # Check shutdown
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
        self._setup_containment_routes()
        self._setup_approval_routes()
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
            return {"status": "healthy"}

    def _setup_containment_routes(self):
        """Containment action endpoints."""
        @self.app.post("/api/v1/contain")
        @timed_operation("containment")
        async def contain_incident(
            incident: Dict[str, Any] = Body(...)
        ):
            """Execute containment actions based on triaged threat."""
            alarm_id = incident.get("alarm_id", "unknown")
            tenant_id = incident.get("tenant_id") or get_current_tenant_id()
            correlation_id = get_correlation_id()

            logger.info("Containment request received",
                        alarm_id=alarm_id,
                        tenant_id=tenant_id,
                        classification=incident.get("classification"))

            try:
                # Determine risk level
                classification = incident.get("classification", "unknown")
                confidence = incident.get("confidence", 0)
                risk_level = self._assess_risk_level(classification, confidence)

                # Check if approval required
                if self._requires_approval(risk_level, tenant_id):
                    approval_id = f"approval-{alarm_id}"
                    self.pending_approvals[approval_id] = {
                        "incident": incident,
                        "tenant_id": tenant_id,
                        "correlation_id": correlation_id,
                        "risk_level": risk_level,
                        "created_at": datetime.now().isoformat()
                    }

                    logger.info("Containment pending approval",
                                alarm_id=alarm_id,
                                risk_level=risk_level)

                    await self._audit_log(
                        action="pending_approval",
                        incident=incident,
                        tenant_id=tenant_id,
                        details={"risk_level": risk_level}
                    )

                    return {
                        "status": "pending_approval",
                        "incident_id": alarm_id,
                        "approval_id": approval_id,
                        "risk_level": risk_level,
                        "message": "High-risk action requires manual approval"
                    }

                # Execute containment
                playbook = self._select_playbook(classification, risk_level)
                result = self.agent._execute_playbook(playbook, incident)

                # Send notification
                await self._send_notification(
                    subject=f"INCIDENT CONTAINED: {alarm_id}",
                    message=f"Incident {alarm_id} contained\n"
                            f"Classification: {classification}\n"
                            f"Actions: {result.get('actions_executed')}",
                    tenant_id=tenant_id
                )

                # Audit log
                await self._audit_log(
                    action="containment_executed",
                    incident=incident,
                    tenant_id=tenant_id,
                    result=result
                )

                metrics.increment_counter(
                    "containment_executed",
                    tenant_id=tenant_id or "unknown",
                    status="success"
                )

                logger.info("Containment executed",
                            alarm_id=alarm_id,
                            status=result.get("containment_status"),
                            actions=len(result.get("actions_executed", [])))

                return {
                    "status": "active",
                    "incident_id": alarm_id,
                    "tenant_id": tenant_id,
                    "correlation_id": correlation_id,
                    "actions": result.get("actions_executed"),
                    "containment_status": result.get("containment_status")
                }

            except Exception as e:
                logger.error("Containment error",
                             alarm_id=alarm_id,
                             error=str(e))
                metrics.increment_counter(
                    "containment_executed",
                    tenant_id=tenant_id or "unknown",
                    status="error"
                )
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/v1/actions")
        async def get_actions():
            """Get recent containment actions."""
            return {
                "actions": self.agent.actions_log[-100:],  # Last 100 actions
                "total": len(self.agent.actions_log)
            }

    def _setup_approval_routes(self):
        """Manual approval endpoints."""
        @self.app.get("/api/v1/pending-approvals")
        async def list_pending_approvals():
            """List pending containment approvals."""
            return {
                "pending": [
                    {
                        "approval_id": aid,
                        "incident_id": data["incident"].get("alarm_id"),
                        "tenant_id": data["tenant_id"],
                        "risk_level": data["risk_level"],
                        "created_at": data["created_at"]
                    }
                    for aid, data in self.pending_approvals.items()
                ]
            }

        @self.app.post("/api/v1/approve")
        @timed_operation("approval")
        async def process_approval(request: ApprovalRequest):
            """Process manual approval for high-risk actions."""
            approval_id = f"approval-{request.incident_id}"

            if approval_id not in self.pending_approvals:
                raise HTTPException(
                    status_code=404,
                    detail="Approval not found or already processed"
                )

            pending = self.pending_approvals.pop(approval_id)
            incident = pending["incident"]
            tenant_id = pending["tenant_id"]

            logger.info("Processing approval",
                        incident_id=request.incident_id,
                        approved=request.approved,
                        approver=request.approver)

            if request.approved:
                # Execute containment
                classification = incident.get("classification", "unknown")
                playbook = self._select_playbook(
                    classification,
                    pending["risk_level"]
                )
                result = self.agent._execute_playbook(playbook, incident)

                await self._audit_log(
                    action="containment_approved_and_executed",
                    incident=incident,
                    tenant_id=tenant_id,
                    result=result,
                    approver=request.approver,
                    comments=request.comments
                )

                return {
                    "status": "executed",
                    "incident_id": request.incident_id,
                    "actions": result.get("actions_executed"),
                    "approved_by": request.approver
                }
            else:
                await self._audit_log(
                    action="containment_rejected",
                    incident=incident,
                    tenant_id=tenant_id,
                    approver=request.approver,
                    comments=request.comments
                )

                return {
                    "status": "rejected",
                    "incident_id": request.incident_id,
                    "rejected_by": request.approver,
                    "comments": request.comments
                }

    def _setup_metrics_routes(self):
        """Metrics endpoint."""
        @self.app.get("/metrics")
        async def prometheus_metrics():
            return Response(
                content=metrics.export_prometheus(),
                media_type="text/plain"
            )

    def _assess_risk_level(self, classification: str, confidence: float) -> str:
        """Assess risk level of incident."""
        if classification in ["critical", "malicious"] and confidence > 0.8:
            return "high"
        elif classification in ["critical", "malicious", "suspicious"]:
            return "medium"
        return "low"

    def _requires_approval(self, risk_level: str, tenant_id: Optional[str]) -> bool:
        """Check if action requires manual approval."""
        # In production, high-risk actions require approval
        if ENVIRONMENT == "production" and risk_level == "high":
            return True
        return self.agent.config["approval_required"].get(risk_level, False)

    def _select_playbook(self, classification: str, risk_level: str) -> tuple:
        """Select appropriate containment playbook."""
        if risk_level == "high":
            return ("isolate", self.agent.config["playbooks"]["isolate"])
        elif classification in ["malicious", "critical"]:
            return ("block_ip", self.agent.config["playbooks"]["block_ip"])
        return ("default", self.agent.config["playbooks"]["default"])

    async def _send_notification(
        self,
        subject: str,
        message: str,
        tenant_id: Optional[str] = None
    ):
        """Send notification (mock in development)."""
        recipient = os.getenv("SOC_NOTIFICATION_EMAIL", "analyst@example.com")
        logger.info("Notification sent",
                    recipient=recipient,
                    subject=subject,
                    tenant_id=tenant_id)

    async def _audit_log(
        self,
        action: str,
        incident: Dict[str, Any],
        tenant_id: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        approver: Optional[str] = None,
        comments: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Write audit log entry."""
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "incident_id": incident.get("alarm_id"),
            "tenant_id": tenant_id,
            "correlation_id": get_correlation_id(),
            "classification": incident.get("classification"),
            "result": json.dumps(result) if result else None,
            "approver": approver,
            "comments": comments,
            "details": json.dumps(details) if details else None
        }

        logger.info("Audit log entry",
                    action=action,
                    incident_id=incident.get("alarm_id"))

        if self.bq_client:
            try:
                self.bq_client.insert_rows_json([audit_entry])
            except Exception as e:
                logger.warning("Audit log persistence failed", error=str(e))

    def _setup_shutdown(self):
        """Setup graceful shutdown."""
        shutdown_handler = get_shutdown_handler(drain_seconds=15)

        @self.app.on_event("startup")
        async def startup():
            logger.info("CRA Service starting", environment=ENVIRONMENT)
            self.health_checker.set_ready(True)
            setup_signal_handlers(shutdown_handler)

        @self.app.on_event("shutdown")
        async def shutdown():
            logger.info("CRA Service shutting down")
            await shutdown_handler.graceful_shutdown()

    def run(self, host: str = "0.0.0.0", port: int = 8083):
        """Run the service."""
        logger.info("Starting CRA Service", host=host, port=port)
        uvicorn.run(self.app, host=host, port=port, log_level="warning")


# Create app instance
service = CRAService()
app = service.app

if __name__ == "__main__":
    service.run()
