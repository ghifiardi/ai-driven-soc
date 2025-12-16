#!/usr/bin/env python3
"""
TAA A2A Server - FastAPI Implementation
=======================================

FastAPI-based web server for TAA agent A2A communication.
This provides HTTP endpoints for agent-to-agent communication.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel

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

class AgentCard(BaseModel):
    name: str
    version: str
    description: str
    agent_id: str
    capabilities: List[Dict[str, Any]]
    authentication: Dict[str, Any]
    endpoints: Dict[str, Any]
    metadata: Dict[str, Any]

class TAAA2AServer:
    """FastAPI-based A2A server for TAA agent"""
    
    def __init__(self, agent_id: str = None):
        self.agent_id = agent_id or f"taa-agent-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.app = FastAPI(
            title="TAA A2A Server",
            description="TAA Agent with A2A communication capabilities",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Create agent card
        self.agent_card = self._create_agent_card()
        
        # Setup routes
        self._setup_routes()
        
        logger.info(f"TAA A2A Server initialized: {self.agent_id}")
    
    def _create_agent_card(self) -> Dict:
        """Create A2A agent card for discovery"""
        return {
            "name": "taa-security-agent",
            "version": "1.0.0",
            "description": "Triage and Analysis Agent for security alerts with A2A capabilities",
            "agent_id": self.agent_id,
            "capabilities": [
                {
                    "name": "analyze_security_alert",
                    "description": "Analyze security alerts using LLM and threat intelligence",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "alert_data": {"type": "object"},
                            "enrichment_level": {"type": "string", "enum": ["basic", "full"]}
                        },
                        "required": ["alert_data"]
                    }
                },
                {
                    "name": "share_threat_intelligence",
                    "description": "Share threat intelligence with other agents",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "indicators": {"type": "array"},
                            "confidence": {"type": "number"},
                            "source": {"type": "string"}
                        }
                    }
                },
                {
                    "name": "request_containment",
                    "description": "Request containment actions from other agents",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "target_host": {"type": "string"},
                            "action_type": {"type": "string", "enum": ["isolate", "block", "monitor"]},
                            "priority": {"type": "string", "enum": ["low", "medium", "high", "critical"]}
                        },
                        "required": ["target_host", "action_type"]
                    }
                },
                {
                    "name": "collaborative_analysis",
                    "description": "Collaborate with other agents for complex analysis",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "analysis_type": {"type": "string"},
                            "data": {"type": "object"},
                            "collaboration_mode": {"type": "string", "enum": ["parallel", "sequential", "hierarchical"]}
                        }
                    }
                }
            ],
            "authentication": {
                "type": "oauth2",
                "scopes": ["https://www.googleapis.com/auth/cloud-platform"]
            },
            "endpoints": {
                "base_url": f"https://{self.agent_id}.agentspace.googleapis.com",
                "a2a_endpoint": "/a2a/tasks",
                "health_check": "/health",
                "agent_card": "/.well-known/agent.json"
            },
            "metadata": {
                "domain": "cybersecurity",
                "specialization": "threat_analysis",
                "supported_formats": ["json", "xml"],
                "response_time": "5s",
                "availability": "24/7"
            }
        }
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/")
        async def root():
            """Root endpoint"""
            return {
                "message": "TAA A2A Server is running",
                "agent_id": self.agent_id,
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "agent_id": self.agent_id,
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0"
            }
        
        @self.app.get("/.well-known/agent.json")
        async def get_agent_card():
            """Get agent card for A2A discovery"""
            return self.agent_card
        
        @self.app.post("/a2a/tasks")
        async def handle_a2a_task(task_request: A2ATaskRequest, background_tasks: BackgroundTasks):
            """Handle incoming A2A task"""
            try:
                logger.info(f"Received A2A task: {task_request.task_id} - {task_request.skill}")
                
                # Process the task based on skill
                result = await self._process_task(task_request)
                
                # Create response
                response = A2AResponse(
                    task_id=task_request.task_id,
                    status="completed",
                    result=result,
                    agent_id=self.agent_id,
                    timestamp=datetime.now().isoformat()
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
                "capabilities": self.agent_card["capabilities"],
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
                "timestamp": datetime.now().isoformat()
            }
    
    async def _process_task(self, task_request: A2ATaskRequest) -> Dict[str, Any]:
        """Process A2A task based on skill"""
        try:
            if task_request.skill == "analyze_security_alert":
                return await self._analyze_security_alert(task_request.parameters)
            
            elif task_request.skill == "share_threat_intelligence":
                return await self._share_threat_intelligence(task_request.parameters)
            
            elif task_request.skill == "request_containment":
                return await self._request_containment(task_request.parameters)
            
            elif task_request.skill == "collaborative_analysis":
                return await self._collaborative_analysis(task_request.parameters)
            
            else:
                raise ValueError(f"Unknown skill: {task_request.skill}")
                
        except Exception as e:
            logger.error(f"Error processing task: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _analyze_security_alert(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze security alert using LLM and threat intelligence"""
        try:
            alert_data = parameters.get("alert_data", {})
            enrichment_level = parameters.get("enrichment_level", "full")
            
            # Simulate LLM analysis (replace with actual implementation)
            analysis_result = {
                "is_true_positive": True,
                "severity": "high",
                "confidence": 0.92,
                "attack_type": "malware",
                "indicators": ["suspicious_activity", "malicious_ip"],
                "recommended_actions": ["isolate_host", "block_ip"],
                "analysis_summary": "High-confidence malware detection",
                "enrichment_level": enrichment_level,
                "timestamp": datetime.now().isoformat()
            }
            
            return {
                "status": "success",
                "analysis": analysis_result,
                "agent_id": self.agent_id
            }
            
        except Exception as e:
            logger.error(f"Error in alert analysis: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _share_threat_intelligence(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Share threat intelligence with other agents"""
        try:
            indicators = parameters.get("indicators", [])
            confidence = parameters.get("confidence", 0.0)
            source = parameters.get("source", "unknown")
            
            # Simulate threat intelligence sharing
            ti_data = {
                "indicators": indicators,
                "confidence": confidence,
                "source": source,
                "agent_id": self.agent_id,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Sharing threat intelligence: {len(indicators)} indicators")
            
            return {
                "status": "success",
                "shared_indicators": len(indicators),
                "message": "Threat intelligence shared successfully",
                "data": ti_data
            }
            
        except Exception as e:
            logger.error(f"Error sharing threat intelligence: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _request_containment(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Request containment actions from other agents"""
        try:
            target_host = parameters.get("target_host")
            action_type = parameters.get("action_type")
            priority = parameters.get("priority", "medium")
            
            # Simulate containment request
            containment_request = {
                "target_host": target_host,
                "action_type": action_type,
                "priority": priority,
                "requested_by": self.agent_id,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Requesting containment: {action_type} on {target_host}")
            
            return {
                "status": "success",
                "containment_requested": True,
                "target_host": target_host,
                "action_type": action_type,
                "priority": priority,
                "message": f"Containment request submitted for {target_host}"
            }
            
        except Exception as e:
            logger.error(f"Error requesting containment: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _collaborative_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle collaborative analysis with other agents"""
        try:
            analysis_type = parameters.get("analysis_type")
            data = parameters.get("data", {})
            collaboration_mode = parameters.get("collaboration_mode", "parallel")
            
            logger.info(f"Collaborative analysis: {analysis_type} in {collaboration_mode} mode")
            
            # Simulate collaborative analysis
            return {
                "status": "success",
                "collaboration_mode": collaboration_mode,
                "analysis_type": analysis_type,
                "participating_agents": 3,  # Simulated
                "message": f"Collaborative analysis completed in {collaboration_mode} mode",
                "results": {
                    "primary_analysis": "completed",
                    "secondary_analysis": "completed",
                    "consensus": "high_confidence"
                }
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
        logger.info(f"Starting TAA A2A Server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)

def main():
    """Main function to run the TAA A2A server"""
    try:
        # Create and run the A2A server
        server = TAAA2AServer()
        server.run()
        
    except Exception as e:
        logger.error(f"Error running A2A server: {e}")
        raise

if __name__ == "__main__":
    main() 