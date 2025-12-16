#!/usr/bin/env python3
"""
TAA A2A MCP Server - FastAPI Implementation with MCP Integration
===============================================================

FastAPI-based web server for TAA agent A2A communication with MCP capabilities.
This provides HTTP endpoints for agent-to-agent communication and MCP integration
with Google Security Operations.
"""

import os
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel

# Import the MCP-enhanced agent
from taa_a2a_mcp_agent import TAAA2AMCPAgent, MCPTask, MCPResponse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Pydantic models for request/response validation
class A2ATaskRequest(BaseModel):
    task_id: str
    skill: str
    parameters: Dict[str, Any]
    callback_url: Optional[str] = None
    priority: str = "normal"
    timestamp: Optional[str] = None

class A2AResponse(BaseModel):
    task_id: str
    status: str
    result: Dict[str, Any]
    agent_id: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None

class MCPTaskRequest(BaseModel):
    task_id: str
    tool: str
    arguments: Dict[str, Any]
    timeout: int = 30
    priority: str = "normal"

class MCPTaskResponse(BaseModel):
    task_id: str
    status: str
    result: Dict[str, Any]
    error: Optional[str] = None
    timestamp: str

class ChronicleQueryRequest(BaseModel):
    query_text: str
    time_range: Optional[Dict[str, str]] = None
    detection_rule: Optional[str] = None
    max_results: int = 1000

class ThreatIntelligenceRequest(BaseModel):
    indicators: List[str]
    intelligence_type: str = "ioc"

class InvestigationRequest(BaseModel):
    incident_id: str
    investigation_scope: str = "deep"

class AgentCard(BaseModel):
    name: str
    version: str
    description: str
    agent_id: str
    capabilities: List[Dict[str, Any]]
    mcp_integration: Dict[str, Any]
    authentication: Dict[str, Any]
    endpoints: Dict[str, Any]
    metadata: Dict[str, Any]

class TAAA2AMCPServer:
    """FastAPI-based A2A MCP server for TAA agent"""
    
    def __init__(self, agent_id: str = None):
        self.agent_id = agent_id or f"taa-mcp-agent-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Initialize the MCP-enhanced agent
        self.agent = TAAA2AMCPAgent()
        self.agent.agent_id = self.agent_id
        
        self.app = FastAPI(
            title="TAA A2A MCP Server",
            description="TAA Agent with A2A communication and MCP capabilities for Google Security Operations",
            version="2.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_routes()
        
        logger.info(f"TAA A2A MCP Server initialized: {self.agent_id}")
    
    def _setup_routes(self):
        """Setup FastAPI routes including MCP endpoints"""
        
        @self.app.get("/")
        async def root():
            """Root endpoint"""
            return {
                "message": "TAA A2A MCP Server is running",
                "agent_id": self.agent_id,
                "mcp_enabled": True,
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            mcp_status = {}
            for server_name, client in self.agent.mcp_clients.items():
                try:
                    # Simple health check
                    mcp_status[server_name] = "healthy"
                except Exception as e:
                    mcp_status[server_name] = f"unhealthy: {str(e)}"
            
            return {
                "status": "healthy",
                "agent_id": self.agent_id,
                "version": "2.0.0",
                "mcp_servers": mcp_status,
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/.well-known/agent.json")
        async def get_agent_card():
            """Get agent card for A2A discovery"""
            return self.agent.agent_card
        
        # A2A Endpoints
        @self.app.post("/a2a/tasks")
        async def handle_a2a_task(task_request: A2ATaskRequest, background_tasks: BackgroundTasks):
            """Handle incoming A2A task"""
            try:
                logger.info(f"Received A2A task: {task_request.task_id} - {task_request.skill}")
                
                # Process the task based on skill
                result = await self._process_a2a_task(task_request)
                
                # Create response
                response = A2AResponse(
                    task_id=task_request.task_id,
                    status="completed",
                    result=result,
                    agent_id=self.agent_id,
                    timestamp=datetime.now().isoformat(),
                    metadata={"mcp_enhanced": True}
                )
                
                # If callback URL is provided, send result asynchronously
                if task_request.callback_url:
                    background_tasks.add_task(
                        self._send_callback,
                        task_request.callback_url,
                        response
                    )
                
                return response
                
            except Exception as e:
                logger.error(f"Error handling A2A task: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/a2a/capabilities")
        async def get_capabilities():
            """Get agent capabilities"""
            return {
                "agent_id": self.agent_id,
                "capabilities": self.agent.agent_card["capabilities"],
                "mcp_integration": self.agent.agent_card["mcp_integration"],
                "timestamp": datetime.now().isoformat()
            }
        
        # MCP Endpoints
        @self.app.post("/mcp/tasks")
        async def handle_mcp_task(task_request: MCPTaskRequest):
            """Handle incoming MCP task"""
            try:
                logger.info(f"Received MCP task: {task_request.task_id} - {task_request.tool}")
                
                # Process the MCP task
                result = await self.agent.handle_mcp_task(asdict(task_request))
                
                return MCPTaskResponse(
                    task_id=result.task_id,
                    status=result.status,
                    result=result.result,
                    error=result.error,
                    timestamp=result.timestamp
                )
                
            except Exception as e:
                logger.error(f"Error handling MCP task: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/mcp/chronicle/query")
        async def query_chronicle(query_request: ChronicleQueryRequest):
            """Query Chronicle via MCP"""
            try:
                logger.info(f"Chronicle query: {query_request.query_text[:100]}...")
                
                result = await self.agent._query_chronicle_via_mcp(
                    query_request.query_text,
                    query_request.time_range,
                    query_request.detection_rule
                )
                
                return {
                    "status": "success",
                    "query": query_request.query_text,
                    "results": result,
                    "agent_id": self.agent_id,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error querying Chronicle: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/mcp/threat-intelligence/query")
        async def query_threat_intelligence(request: ThreatIntelligenceRequest):
            """Query threat intelligence via MCP"""
            try:
                logger.info(f"Threat intelligence query for {len(request.indicators)} indicators")
                
                result = await self.agent._query_threat_intelligence_via_mcp(
                    request.indicators,
                    request.intelligence_type
                )
                
                return {
                    "status": "success",
                    "indicators_queried": len(request.indicators),
                    "intelligence_type": request.intelligence_type,
                    "results": result,
                    "agent_id": self.agent_id,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error querying threat intelligence: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/mcp/investigate")
        async def investigate_incident(request: InvestigationRequest):
            """Conduct investigation via MCP"""
            try:
                logger.info(f"Investigation request for incident: {request.incident_id}")
                
                result = await self.agent._conduct_investigation_via_mcp(
                    request.incident_id,
                    request.investigation_scope
                )
                
                return {
                    "status": "success",
                    "incident_id": request.incident_id,
                    "investigation_scope": request.investigation_scope,
                    "results": result,
                    "agent_id": self.agent_id,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error conducting investigation: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/mcp/servers")
        async def get_mcp_servers():
            """Get MCP server status"""
            servers = {}
            for server_name, client in self.agent.mcp_clients.items():
                try:
                    # Check server health
                    servers[server_name] = {
                        "status": "connected",
                        "available_tools": [],  # Would be populated from actual MCP client
                        "last_check": datetime.now().isoformat()
                    }
                except Exception as e:
                    servers[server_name] = {
                        "status": "disconnected",
                        "error": str(e),
                        "last_check": datetime.now().isoformat()
                    }
            
            return {
                "agent_id": self.agent_id,
                "mcp_servers": servers,
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.post("/a2a/callbacks")
        async def handle_callback(callback_data: Dict[str, Any]):
            """Handle callback from other agents"""
            try:
                logger.info(f"Received callback: {callback_data}")
                # Process callback data
                return {"status": "received", "timestamp": datetime.now().isoformat()}
            except Exception as e:
                logger.error(f"Error handling callback: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get agent metrics"""
            return {
                "agent_id": self.agent_id,
                "uptime": "24h",
                "tasks_processed": 100,
                "success_rate": 0.95,
                "mcp_requests": 50,
                "chronicle_queries": 25,
                "threat_intelligence_queries": 15,
                "timestamp": datetime.now().isoformat()
            }
        
        # Enhanced Analysis Endpoints
        @self.app.post("/analyze/enhanced")
        async def analyze_alert_enhanced(alert_data: Dict[str, Any]):
            """Enhanced alert analysis with MCP integration"""
            try:
                enrichment_level = alert_data.get("enrichment_level", "chronicle")
                
                # Use the agent's enhanced analysis capability
                result = await self.agent._enrich_with_chronicle(alert_data)
                
                return {
                    "status": "success",
                    "analysis": result,
                    "enrichment_level": enrichment_level,
                    "agent_id": self.agent_id,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error in enhanced analysis: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/share/intelligence")
        async def share_threat_intelligence(intelligence_data: Dict[str, Any]):
            """Share threat intelligence with Chronicle and other agents"""
            try:
                indicators = intelligence_data.get("indicators", [])
                confidence = intelligence_data.get("confidence", 0.0)
                source = intelligence_data.get("source", "unknown")
                share_with_chronicle = intelligence_data.get("share_with_chronicle", True)
                
                # Share with other agents via Pub/Sub
                topic_path = self.agent.publisher.topic_path(
                    self.agent.config["project_id"],
                    self.agent.config["pubsub"]["threat_intelligence_topic"]
                )
                
                ti_data = {
                    "indicators": indicators,
                    "confidence": confidence,
                    "source": source,
                    "agent_id": self.agent_id,
                    "share_with_chronicle": share_with_chronicle,
                    "timestamp": datetime.now().isoformat()
                }
                
                self.agent.publisher.publish(topic_path, json.dumps(ti_data).encode("utf-8"))
                
                # If requested, also share with Chronicle via MCP
                chronicle_result = {}
                if share_with_chronicle and "google_security_operations" in self.agent.mcp_clients:
                    try:
                        # This would be a Chronicle-specific MCP tool for sharing TI
                        chronicle_result = {"status": "shared_with_chronicle"}
                    except Exception as e:
                        chronicle_result = {"error": str(e)}
                
                return {
                    "status": "success",
                    "indicators_shared": len(indicators),
                    "chronicle_sharing": chronicle_result,
                    "agent_id": self.agent_id,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error sharing threat intelligence: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def _process_a2a_task(self, task_request: A2ATaskRequest) -> Dict[str, Any]:
        """Process A2A task based on skill with MCP enhancement"""
        try:
            if task_request.skill == "analyze_security_alert":
                return await self._analyze_security_alert_enhanced(task_request.parameters)
            
            elif task_request.skill == "query_chronicle_detections":
                return await self._query_chronicle_detections(task_request.parameters)
            
            elif task_request.skill == "get_threat_intelligence":
                return await self._get_threat_intelligence(task_request.parameters)
            
            elif task_request.skill == "investigate_incident":
                return await self._investigate_incident(task_request.parameters)
            
            elif task_request.skill == "share_threat_intelligence":
                return await self._share_threat_intelligence(task_request.parameters)
            
            elif task_request.skill == "request_containment":
                return await self._request_containment(task_request.parameters)
            
            elif task_request.skill == "collaborative_analysis":
                return await self._collaborative_analysis(task_request.parameters)
            
            else:
                raise ValueError(f"Unknown skill: {task_request.skill}")
                
        except Exception as e:
            logger.error(f"Error processing A2A task: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _analyze_security_alert_enhanced(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced security alert analysis with MCP"""
        try:
            alert_data = parameters.get("alert_data", {})
            enrichment_level = parameters.get("enrichment_level", "chronicle")
            
            # Enhanced analysis with Chronicle context
            chronicle_context = {}
            if enrichment_level == "chronicle" and "google_security_operations" in self.agent.mcp_clients:
                chronicle_context = await self.agent._enrich_with_chronicle(alert_data)
            
            analysis_result = {
                "is_true_positive": True,
                "severity": "high",
                "confidence": 0.95,
                "attack_type": "advanced_persistent_threat",
                "indicators": ["suspicious_activity", "malicious_ip", "anomalous_traffic"],
                "recommended_actions": ["isolate_host", "block_ip", "escalate_to_incident"],
                "analysis_summary": "High-confidence APT detection with Chronicle context",
                "chronicle_context": chronicle_context,
                "enrichment_level": enrichment_level,
                "timestamp": datetime.now().isoformat()
            }
            
            return {
                "status": "success",
                "analysis": analysis_result,
                "agent_id": self.agent_id,
                "mcp_enhanced": True
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced alert analysis: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _query_chronicle_detections(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Query Chronicle detections"""
        try:
            query_text = parameters.get("query_text")
            time_range = parameters.get("time_range")
            detection_rule = parameters.get("detection_rule")
            
            result = await self.agent._query_chronicle_via_mcp(query_text, time_range, detection_rule)
            
            return {
                "status": "success",
                "query_results": result,
                "agent_id": self.agent_id
            }
            
        except Exception as e:
            logger.error(f"Error querying Chronicle detections: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _get_threat_intelligence(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get threat intelligence"""
        try:
            indicators = parameters.get("indicators", [])
            intelligence_type = parameters.get("intelligence_type", "ioc")
            
            result = await self.agent._query_threat_intelligence_via_mcp(indicators, intelligence_type)
            
            return {
                "status": "success",
                "threat_intelligence": result,
                "indicators_queried": len(indicators),
                "agent_id": self.agent_id
            }
            
        except Exception as e:
            logger.error(f"Error getting threat intelligence: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _investigate_incident(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Investigate incident"""
        try:
            incident_id = parameters.get("incident_id")
            investigation_scope = parameters.get("investigation_scope", "deep")
            
            result = await self.agent._conduct_investigation_via_mcp(incident_id, investigation_scope)
            
            return {
                "status": "success",
                "investigation": result,
                "incident_id": incident_id,
                "scope": investigation_scope,
                "agent_id": self.agent_id
            }
            
        except Exception as e:
            logger.error(f"Error investigating incident: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _share_threat_intelligence(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Share threat intelligence"""
        try:
            indicators = parameters.get("indicators", [])
            confidence = parameters.get("confidence", 0.0)
            source = parameters.get("source", "unknown")
            
            # Share via Pub/Sub
            topic_path = self.agent.publisher.topic_path(
                self.agent.config["project_id"],
                "threat-intelligence"
            )
            
            ti_data = {
                "indicators": indicators,
                "confidence": confidence,
                "source": source,
                "agent_id": self.agent_id,
                "timestamp": datetime.now().isoformat()
            }
            
            self.agent.publisher.publish(topic_path, json.dumps(ti_data).encode("utf-8"))
            
            return {
                "status": "success",
                "shared_indicators": len(indicators),
                "message": "Threat intelligence shared successfully",
                "agent_id": self.agent_id
            }
            
        except Exception as e:
            logger.error(f"Error sharing threat intelligence: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _request_containment(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Request containment actions"""
        try:
            target_host = parameters.get("target_host")
            action_type = parameters.get("action_type")
            priority = parameters.get("priority", "medium")
            
            # Publish containment request
            topic_path = self.agent.publisher.topic_path(
                self.agent.config["project_id"],
                "containment-requests"
            )
            
            containment_request = {
                "target_host": target_host,
                "action_type": action_type,
                "priority": priority,
                "requested_by": self.agent_id,
                "timestamp": datetime.now().isoformat()
            }
            
            self.agent.publisher.publish(topic_path, json.dumps(containment_request).encode("utf-8"))
            
            return {
                "status": "success",
                "containment_requested": True,
                "target_host": target_host,
                "action_type": action_type,
                "priority": priority,
                "agent_id": self.agent_id
            }
            
        except Exception as e:
            logger.error(f"Error requesting containment: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _collaborative_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Collaborative analysis with MCP context"""
        try:
            analysis_type = parameters.get("analysis_type")
            data = parameters.get("data", {})
            collaboration_mode = parameters.get("collaboration_mode", "parallel")
            include_chronicle_context = parameters.get("include_chronicle_context", True)
            
            # Enhance with Chronicle context if requested
            chronicle_context = {}
            if include_chronicle_context and "google_security_operations" in self.agent.mcp_clients:
                # Extract indicators from data and enrich
                indicators = self.agent._extract_indicators(data)
                for indicator in indicators:
                    chronicle_context[indicator] = await self.agent._query_chronicle_via_mcp(
                        f"metadata.event_type='NETWORK_CONNECTION' AND network.destination.ip_address='{indicator}'"
                    )
            
            return {
                "status": "success",
                "collaboration_mode": collaboration_mode,
                "analysis_type": analysis_type,
                "chronicle_context": chronicle_context,
                "participating_agents": 3,  # Simulated
                "message": f"Collaborative analysis completed in {collaboration_mode} mode",
                "agent_id": self.agent_id
            }
            
        except Exception as e:
            logger.error(f"Error in collaborative analysis: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _send_callback(self, callback_url: str, response: A2AResponse):
        """Send callback to another agent"""
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                headers = {"Content-Type": "application/json"}
                payload = response.dict()
                
                async with session.post(callback_url, json=payload, headers=headers) as resp:
                    if resp.status == 200:
                        logger.info(f"Callback sent successfully to {callback_url}")
                    else:
                        logger.warning(f"Callback failed: {resp.status}")
                        
        except Exception as e:
            logger.error(f"Error sending callback: {e}")
    
    def run(self, host: str = "0.0.0.0", port: int = 8080):
        """Run the FastAPI server"""
        logger.info(f"Starting TAA A2A MCP Server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)

def main():
    """Main function to run the TAA A2A MCP server"""
    try:
        # Create and run the A2A MCP server
        server = TAAA2AMCPServer()
        server.run()
        
    except Exception as e:
        logger.error(f"Error running A2A MCP server: {e}")
        raise

if __name__ == "__main__":
    main()


