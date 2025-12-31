#!/usr/bin/env python3
"""
MSSP Platform Server
====================

Unified platform interface for MSSPs to integrate with the AI-driven SOC.
Exposes REST API for event ingestion and tenant management, and MCP tools for AI agent integration.
"""

import os
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import secrets
import jwt
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel

from multi_tenant_manager import MultiTenantManager, TenantConfig, TenantTables, TenantPubSubTopics, TenantRateLimits, FirewallConfig
try:
    from taa_a2a_mcp_agent import TAAA2AMCPAgent
except ImportError:
    TAAA2AMCPAgent = None
from advanced_anomaly_detection import GATRAAnomalyDetectionSystem
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CONFIG_PATH = os.getenv("MULTITENANT_CONFIG_PATH", "config/gatra_multitenant_config.json")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_MINUTES = 60

# JWT Secret - MUST be provided via environment variable in production
_jwt_secret = os.getenv("JWT_SECRET")
if not _jwt_secret:
    if os.getenv("ENVIRONMENT", "development") == "production":
        raise RuntimeError(
            "JWT_SECRET environment variable is required in production. "
            "Generate one with: openssl rand -hex 32"
        )
    # Development only: generate ephemeral secret (will change on restart)
    import warnings
    _jwt_secret = secrets.token_hex(32)
    warnings.warn(
        "JWT_SECRET not set - using ephemeral secret. "
        "Set JWT_SECRET environment variable for production.",
        RuntimeWarning
    )
JWT_SECRET = _jwt_secret

LEARNING_SERVICE_URL = os.getenv("LEARNING_SERVICE_URL", "http://learning-service:8084")

# Set BigQuery Credentials for Baseline Integration
SA_PATH = "Service Account BigQuery/sa-gatra-bigquery.json"
if os.path.exists(SA_PATH):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SA_PATH

import requests
from google.cloud import bigquery
from bigquery_client import BigQueryClient

security = HTTPBearer()

# Pydantic Models
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

class EventIngestionRequest(BaseModel):
    tenant_id: str
    events: List[Dict[str, Any]]
    source: str = "api"

class MSSPPlatformServer:
    def __init__(self):
        self.app = FastAPI(
            title="AI-Driven SOC - MSSP Platform",
            description="Unified platform for Managed Security Service Providers",
            version="1.0.0"
        )
        
        # Initialize Tenant Manager
        try:
            self.tenant_manager = MultiTenantManager.from_file(CONFIG_PATH)
            logger.info(f"Loaded tenant configuration from {CONFIG_PATH}")
        except Exception as e:
            logger.error(f"Failed to load tenant configuration: {e}")
            raise

        if TAAA2AMCPAgent:
            try:
                self.agent = TAAA2AMCPAgent()
            except Exception as e:
                logger.warning(f"Failed to initialize TAAA2AMCPAgent: {e}")
                self.agent = None
        
        # Initialize BigQuery Persistence
        try:
            self.bq_persistence = BigQueryClient(
                project_id="chronicle-dev-2be9",
                dataset_id="gatra_database",
                table_id="siem_events"
            )
            logger.info("BigQuery Ingestion Persistence active")
        except Exception as e:
            logger.error(f"Failed to initialize BQ Persistence: {e}")
            self.bq_persistence = None
        else:
            self.agent = None
        
        # Initialize GATRA Engine
        self.gatra = GATRAAnomalyDetectionSystem()
        
        self._setup_middleware()
        self._setup_routes()

    def _create_access_token(self, tenant_id: str):
        expires_delta = timedelta(minutes=JWT_EXPIRATION_MINUTES)
        expire = datetime.utcnow() + expires_delta
        to_encode = {"exp": expire, "sub": tenant_id}
        encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
        return encoded_jwt

    def _get_current_tenant(self, credentials: HTTPAuthorizationCredentials = Depends(security)):
        token = credentials.credentials
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            tenant_id: str = payload.get("sub")
            if tenant_id is None:
                raise HTTPException(status_code=401, detail="Invalid token")
            return self.tenant_manager.get_tenant(tenant_id)
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.PyJWTError:
            raise HTTPException(status_code=401, detail="Could not validate credentials")
        except Exception:
            raise HTTPException(status_code=401, detail="Invalid tenant")

    def _setup_middleware(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self):
        @self.app.get("/")
        async def root():
            return {
                "platform": "AI-Driven SOC MSSP Platform",
                "status": "operational",
                "tenants_active": self.tenant_manager.tenants_count(),
                "timestamp": datetime.now().isoformat()
            }

        @self.app.post("/api/v1/feedback")
        async def submit_soc_feedback(payload: Dict[str, Any]):
            """Analyst feedback for Reinforcement Learning."""
            try:
                resp = requests.post(f"{LEARNING_SERVICE_URL}/api/v1/feedback", json=payload, timeout=5)
                return resp.json()
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/api/v1/auth/token")
        async def get_token(x_api_key: str = Header(...)):
            """Exchange an API Key for a JWT token."""
            for tenant in self.tenant_manager.list_tenants():
                if tenant.api_key == x_api_key:
                    token = self._create_access_token(tenant.tenant_id)
                    return {"access_token": token, "token_type": "bearer"}
            
            raise HTTPException(status_code=401, detail="Invalid API Key")

        # --- Tenant Management ---

        @self.app.get("/api/v1/tenants")
        async def list_tenants():
            """List all registered tenants."""
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
        async def register_tenant(request: TenantRegistrationRequest):
            """Register a new tenant dynamically."""
            try:
                # Check if tenant already exists
                try:
                    self.tenant_manager.get_tenant(request.tenant_id)
                    raise HTTPException(status_code=409, detail=f"Tenant {request.tenant_id} already exists")
                except Exception:
                    pass # Tenant does not exist, proceed

                # Create default resources structure
                defaults = self.tenant_manager._config.defaults
                # Configure firewall if provided
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
                    results_dataset=defaults.results_dataset_template.format(tenant_id=request.tenant_id),
                    tables=TenantTables(events="events", alerts="alerts", results="events_results"),
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
                
                logger.info(f"Registered new tenant: {request.tenant_id}")
                return {
                    "status": "created", 
                    "tenant_id": new_tenant.tenant_id,
                    "api_key": new_tenant.api_key
                }

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error registering tenant: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # --- Event Ingestion ---

        @self.app.post("/api/v1/events")
        async def ingest_events(
            request: EventIngestionRequest, 
            background_tasks: BackgroundTasks,
            tenant_auth: TenantConfig = Depends(self._get_current_tenant)
        ):
            """Ingest security events for a specific tenant (Requires JWT)."""
            try:
                # Ensure the token matches the requested tenant_id
                if tenant_auth.tenant_id != request.tenant_id:
                    raise HTTPException(status_code=403, detail="Token does not match tenant_id")
                
                # In a real implementation, this would push to Pub/Sub
                # For now, we'll simulate processing
                background_tasks.add_task(self._process_events_background, request.tenant_id, request.events)
                
                return {
                    "status": "accepted",
                    "event_count": len(request.events),
                    "tenant_id": request.tenant_id
                }
            except Exception as e:
                logger.error(f"Ingestion error: {e}")
                raise HTTPException(status_code=400, detail=str(e))

        # --- MCP Tools ---

        @self.app.post("/mcp/tools/get_tenant_status")
        async def mcp_get_tenant_status(payload: Dict[str, Any]):
            """MCP Tool: Get status and configuration of a tenant."""
            tenant_id = payload.get("tenant_id")
            if not tenant_id:
                raise HTTPException(status_code=400, detail="Missing tenant_id")
            
            try:
                tenant = self.tenant_manager.get_tenant(tenant_id)
                return {
                    "tenant_id": tenant.tenant_id,
                    "status": "active",
                    "config": {
                        "region": tenant.region,
                        "service_level": tenant.service_level
                    }
                }
            except Exception as e:
                return {"error": str(e)}

    async def _process_events_background(self, tenant_id: str, events: List[Dict[str, Any]]):
        """Background task for analysis and persistence."""
        # 1. Persist to BigQuery Baseline
        if self.bq_persistence:
            try:
                rows = []
                for event in events:
                    rows.append({
                        "alarmId": event.get("alarm_id", str(secrets.token_hex(8))),
                        "events": json.dumps(event),
                        "processed_by_ada": False,
                        "ingestion_time": datetime.now().isoformat()
                    })
                self.bq_persistence.insert_rows_json(rows)
            except Exception as e:
                logger.error(f"BQ Persistence error: {e}")

        # 2. GATRA Analysis
        logger.info(f"GATRA analyzing {len(events)} events for tenant {tenant_id}")
        
        results = []
        for event in events:
            # Prepare feature vector (10D)
            # In a real scenario, this would use a more sophisticated feature extractor
            features = np.zeros(10)
            features[0] = float(event.get("duration") or 0)
            features[1] = float(event.get("bytes_sent") or 0)
            features[2] = float(event.get("bytes_received") or 0)
            features[3] = float(event.get("port") or 0)
            # ... additional mapping ...
            
            event['features'] = features
            
            # Detect anomalies
            gatra_result = self.gatra.process_telemetry(event)
            
            if gatra_result.is_anomaly:
                logger.warning(f"ALERT: GATRA detected anomaly for {tenant_id}: {gatra_result.reasoning}")
                
                # Create an anomaly report
                anomaly_report = {
                    "alarm_id": f"GATRA-{event.get('id')}",
                    "timestamp": datetime.now().isoformat(),
                    "source_ip": event.get("source_ip"),
                    "destination_ip": event.get("dest_ip"),
                    "protocol": event.get("protocol"),
                    "alert_severity": gatra_result.severity.name.lower(),
                    "attack_category": "anomaly",
                    "confidence": gatra_result.score,
                    "reasoning": [gatra_result.reasoning]
                }
                
                # --- ORCHESTRATION: Forward to TAA for Triage ---
                try:
                    logger.info(f"Forwarding {anomaly_report['alarm_id']} to TAA for triage...")
                    taa_resp = requests.post(f"{TAA_SERVICE_URL}/api/v1/triage", json=anomaly_report, timeout=5)
                    if taa_resp.status_code == 200:
                        triage_result = taa_resp.json()
                        logger.info(f"TAA Triage Outcome: {triage_result['classification']} (Confidence: {triage_result['confidence']:.2f})")
                        
                        # --- ORCHESTRATION: If Malicious/Critical, forward to CRA for Containment ---
                        if triage_result['classification'] in ['malicious', 'critical']:
                            logger.warning(f"THREAT CONFIRMED: Forwarding to CRA for autonomous response.")
                            cra_resp = requests.post(f"{CRA_SERVICE_URL}/api/v1/contain", json=triage_result, timeout=5)
                            if cra_resp.status_code == 200:
                                cra_outcome = cra_resp.json()
                                logger.info(f"CRA Response: {cra_outcome['containment_status']} - Actions: {cra_outcome['actions']}")
                except Exception as e:
                    logger.error(f"SOC Orchestration error: {e}")

                results.append({
                    "event_id": event.get("id"),
                    "score": gatra_result.score,
                    "severity": gatra_result.severity.name,
                    "reasoning": gatra_result.reasoning
                })
        
        # In a real implementation, we would write these results to BigQuery/Firestore
        if results:
            logger.info(f"GATRA found {len(results)} anomalies for tenant {tenant_id}")
        
        await asyncio.sleep(0.1) # Simulate I/O latency

    def run(self, host: str = "0.0.0.0", port: int = 8081):
        logger.info(f"Starting MSSP Platform Server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)

if __name__ == "__main__":
    server = MSSPPlatformServer()
    server.run()
