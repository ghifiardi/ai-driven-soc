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
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel

from multi_tenant_manager import MultiTenantManager, TenantConfig, TenantTables, TenantPubSubTopics, TenantRateLimits, FirewallConfig
from taa_a2a_mcp_agent import TAAA2AMCPAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CONFIG_PATH = os.getenv("MULTITENANT_CONFIG_PATH", "config/gatra_multitenant_config.json")

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

        # Initialize MCP Agent (reusing TAA logic for intelligence)
        self.agent = TAAA2AMCPAgent()
        
        self._setup_middleware()
        self._setup_routes()

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
                    firewall_config=firewall_config
                )

                self.tenant_manager.add_tenant(new_tenant)
                self.tenant_manager.save_config(CONFIG_PATH)
                
                logger.info(f"Registered new tenant: {request.tenant_id}")
                return {"status": "created", "tenant_id": request.tenant_id}

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error registering tenant: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # --- Event Ingestion ---

        @self.app.post("/api/v1/events")
        async def ingest_events(request: EventIngestionRequest, background_tasks: BackgroundTasks):
            """Ingest security events for a specific tenant."""
            try:
                # Validate tenant
                tenant = self.tenant_manager.get_tenant(request.tenant_id)
                
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
        """Simulate background event processing."""
        logger.info(f"Processing {len(events)} events for tenant {tenant_id}")
        # Here we would use the BigQueryClient to insert rows
        # or PublisherClient to publish to Pub/Sub
        await asyncio.sleep(0.1) # Simulate latency

    def run(self, host: str = "0.0.0.0", port: int = 8081):
        logger.info(f"Starting MSSP Platform Server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)

if __name__ == "__main__":
    server = MSSPPlatformServer()
    server.run()
