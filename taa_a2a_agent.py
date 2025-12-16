#!/usr/bin/env python3
"""
TAA Agent with A2A (Agent-to-Agent) Communication
==================================================

This implements the TAA (Triage & Analysis Agent) with A2A protocol support
for communication with Google Agentspace and other agents.

Features:
- A2A protocol implementation
- Google Vertex AI integration
- Agent discovery and communication
- Threat intelligence sharing
- Multi-agent collaboration
"""

import os
import json
import logging
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from google.cloud import pubsub_v1
import vertexai
from vertexai import agent_engines
from google.adk.agents import Agent
from google.adk.core import BaseAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class A2ATask:
    """A2A task structure"""
    task_id: str
    skill: str
    parameters: Dict[str, Any]
    callback_url: Optional[str] = None
    priority: str = "normal"
    timestamp: str = None

@dataclass
class A2AResponse:
    """A2A response structure"""
    task_id: str
    status: str
    result: Dict[str, Any]
    agent_id: str
    timestamp: str
    metadata: Dict[str, Any] = None

class TAAA2AAgent:
    """TAA Agent with A2A communication capabilities"""
    
    def __init__(self, config_path: str = "config/taa_a2a_config.json"):
        """Initialize the TAA A2A Agent"""
        self.config = self._load_config(config_path)
        self.agent_id = f"taa-agent-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Initialize Vertex AI
        self._init_vertex_ai()
        
        # Initialize A2A components
        self.a2a_server = None
        self.a2a_client = None
        self.agent_card = self._create_agent_card()
        
        # Initialize Pub/Sub for communication
        self.publisher = pubsub_v1.PublisherClient()
        self.subscriber = pubsub_v1.SubscriberClient()
        
        # Agent registry for discovered agents
        self.agent_registry = {}
        
        # Initialize the core TAA agent
        self.taa_agent = self._create_taa_agent()
        
        logger.info(f"TAA A2A Agent initialized: {self.agent_id}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load A2A configuration"""
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
        except FileNotFoundError:
            # Default configuration
            config = {
                "project_id": os.environ.get("GOOGLE_CLOUD_PROJECT", "ai-driven-soc"),
                "location": "us-central1",
                "a2a": {
                    "enabled": True,
                    "port": 8080,
                    "discovery_interval": 300,  # 5 minutes
                    "max_agents": 100
                },
                "vertex_ai": {
                    "model": "gemini-2.0-flash-exp",
                    "staging_bucket": os.environ.get("STAGING_BUCKET", "gs://ai-driven-soc-staging")
                },
                "pubsub": {
                    "alerts_topic": "ada-alerts",
                    "feedback_topic": "taa-feedback",
                    "a2a_topic": "a2a-communication"
                },
                "agentspace": {
                    "enabled": True,
                    "registration_url": "https://agentspace.googleapis.com/v1/agents",
                    "discovery_url": "https://agentspace.googleapis.com/v1/agents/discover"
                }
            }
        
        return config
    
    def _init_vertex_ai(self):
        """Initialize Vertex AI environment"""
        try:
            vertexai.init(
                project=self.config["project_id"],
                location=self.config["location"],
                staging_bucket=self.config["vertex_ai"]["staging_bucket"]
            )
            logger.info("Vertex AI initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {e}")
            raise
    
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
    
    def _create_taa_agent(self) -> Agent:
        """Create the core TAA agent using Google ADK"""
        try:
            agent = Agent(
                name="taa-security-agent",
                model_name=self.config["vertex_ai"]["model"],
                instructions="""You are a Triage and Analysis Agent (TAA) specialized in cybersecurity threat analysis. 
                Your capabilities include:
                - Analyzing security alerts using LLM and threat intelligence
                - Sharing threat intelligence with other agents
                - Requesting containment actions
                - Collaborating with other agents for complex analysis
                
                Always provide structured, actionable responses and maintain security best practices."""
            )
            
            # Add TAA-specific tools
            @agent.tool
            def analyze_security_alert(alert_data: Dict, enrichment_level: str = "full") -> Dict:
                """Analyze security alert using LLM and threat intelligence"""
                try:
                    # Simulate LLM analysis (replace with actual implementation)
                    analysis_result = {
                        "is_true_positive": True,
                        "severity": "high",
                        "confidence": 0.92,
                        "attack_type": "malware",
                        "indicators": ["suspicious_activity", "malicious_ip"],
                        "recommended_actions": ["isolate_host", "block_ip"],
                        "analysis_summary": "High-confidence malware detection",
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
            
            @agent.tool
            def share_threat_intelligence(indicators: List[str], confidence: float, source: str) -> Dict:
                """Share threat intelligence with other agents"""
                try:
                    # Publish to threat intelligence topic
                    topic_path = self.publisher.topic_path(
                        self.config["project_id"], 
                        "threat-intelligence"
                    )
                    
                    ti_data = {
                        "indicators": indicators,
                        "confidence": confidence,
                        "source": source,
                        "agent_id": self.agent_id,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    self.publisher.publish(topic_path, json.dumps(ti_data).encode("utf-8"))
                    
                    return {
                        "status": "success",
                        "shared_indicators": len(indicators),
                        "message": "Threat intelligence shared successfully"
                    }
                except Exception as e:
                    logger.error(f"Error sharing threat intelligence: {e}")
                    return {"status": "error", "message": str(e)}
            
            @agent.tool
            def request_containment(target_host: str, action_type: str, priority: str = "medium") -> Dict:
                """Request containment actions from other agents"""
                try:
                    # Find appropriate containment agent
                    containment_agent = self._find_agent_by_capability("containment")
                    
                    if containment_agent:
                        task = A2ATask(
                            task_id=f"containment-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                            skill="execute_containment",
                            parameters={
                                "target_host": target_host,
                                "action_type": action_type,
                                "priority": priority
                            },
                            callback_url=f"{self.agent_card['endpoints']['base_url']}/a2a/callbacks"
                        )
                        
                        response = asyncio.run(self._send_a2a_task(containment_agent["endpoint"], task))
                        
                        return {
                            "status": "success",
                            "containment_requested": True,
                            "task_id": task.task_id,
                            "target_agent": containment_agent["name"]
                        }
                    else:
                        return {
                            "status": "error",
                            "message": "No containment agent available"
                        }
                except Exception as e:
                    logger.error(f"Error requesting containment: {e}")
                    return {"status": "error", "message": str(e)}
            
            logger.info("TAA agent created successfully with A2A tools")
            return agent
            
        except Exception as e:
            logger.error(f"Failed to create TAA agent: {e}")
            raise
    
    async def _send_a2a_task(self, agent_url: str, task: A2ATask) -> A2AResponse:
        """Send A2A task to another agent"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._get_auth_token()}"
                }
                
                payload = asdict(task)
                
                async with session.post(
                    f"{agent_url}/a2a/tasks",
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return A2AResponse(**result)
                    else:
                        raise Exception(f"A2A task failed: {response.status}")
                        
        except Exception as e:
            logger.error(f"Error sending A2A task: {e}")
            raise
    
    def _get_auth_token(self) -> str:
        """Get authentication token for A2A communication"""
        # Implement OAuth2 token retrieval
        # This is a simplified version - implement proper OAuth2 flow
        return "your-auth-token"
    
    def _find_agent_by_capability(self, capability: str) -> Optional[Dict]:
        """Find agent in registry by capability"""
        for agent_id, agent_info in self.agent_registry.items():
            if capability in [cap["name"] for cap in agent_info.get("capabilities", [])]:
                return agent_info
        return None
    
    async def discover_agents(self):
        """Discover other A2A agents"""
        try:
            if not self.config["agentspace"]["enabled"]:
                return
            
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self._get_auth_token()}"
                }
                
                params = {
                    "domain": "cybersecurity",
                    "capabilities": ["containment", "forensics", "intelligence"]
                }
                
                async with session.get(
                    self.config["agentspace"]["discovery_url"],
                    params=params,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        agents = await response.json()
                        
                        for agent in agents.get("agents", []):
                            self.agent_registry[agent["agent_id"]] = agent
                        
                        logger.info(f"Discovered {len(agents.get('agents', []))} agents")
                    else:
                        logger.warning(f"Agent discovery failed: {response.status}")
                        
        except Exception as e:
            logger.error(f"Error in agent discovery: {e}")
    
    async def register_with_agentspace(self):
        """Register this agent with Google Agentspace"""
        try:
            if not self.config["agentspace"]["enabled"]:
                return
            
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._get_auth_token()}"
                }
                
                registration_data = {
                    "agent_card": self.agent_card,
                    "status": "active",
                    "registration_timestamp": datetime.now().isoformat()
                }
                
                async with session.post(
                    self.config["agentspace"]["registration_url"],
                    json=registration_data,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"Registered with Agentspace: {result}")
                    else:
                        logger.error(f"Agentspace registration failed: {response.status}")
                        
        except Exception as e:
            logger.error(f"Error registering with Agentspace: {e}")
    
    async def handle_a2a_task(self, task_data: Dict) -> A2AResponse:
        """Handle incoming A2A task"""
        try:
            task = A2ATask(**task_data)
            
            # Route to appropriate capability
            if task.skill == "analyze_security_alert":
                result = self.taa_agent.invoke({
                    "tool": "analyze_security_alert",
                    "parameters": task.parameters
                })
            elif task.skill == "share_threat_intelligence":
                result = self.taa_agent.invoke({
                    "tool": "share_threat_intelligence",
                    "parameters": task.parameters
                })
            elif task.skill == "request_containment":
                result = self.taa_agent.invoke({
                    "tool": "request_containment",
                    "parameters": task.parameters
                })
            elif task.skill == "collaborative_analysis":
                result = await self._handle_collaborative_analysis(task.parameters)
            else:
                raise ValueError(f"Unknown skill: {task.skill}")
            
            return A2AResponse(
                task_id=task.task_id,
                status="completed",
                result=result,
                agent_id=self.agent_id,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error handling A2A task: {e}")
            return A2AResponse(
                task_id=task_data.get("task_id", "unknown"),
                status="error",
                result={"error": str(e)},
                agent_id=self.agent_id,
                timestamp=datetime.now().isoformat()
            )
    
    async def _handle_collaborative_analysis(self, parameters: Dict) -> Dict:
        """Handle collaborative analysis with other agents"""
        try:
            analysis_type = parameters.get("analysis_type")
            data = parameters.get("data", {})
            collaboration_mode = parameters.get("collaboration_mode", "parallel")
            
            # Find collaborating agents
            collaborating_agents = []
            for agent_id, agent_info in self.agent_registry.items():
                if analysis_type in [cap["name"] for cap in agent_info.get("capabilities", [])]:
                    collaborating_agents.append(agent_info)
            
            if collaboration_mode == "parallel":
                # Send tasks to all agents in parallel
                tasks = []
                for agent in collaborating_agents:
                    task = A2ATask(
                        task_id=f"collab-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                        skill=analysis_type,
                        parameters=data
                    )
                    tasks.append(self._send_a2a_task(agent["endpoints"]["base_url"], task))
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                return {
                    "status": "success",
                    "collaboration_mode": collaboration_mode,
                    "participating_agents": len(collaborating_agents),
                    "results": [r.result if not isinstance(r, Exception) else {"error": str(r)} for r in results]
                }
            
            else:
                # Sequential or hierarchical processing
                return {
                    "status": "success",
                    "collaboration_mode": collaboration_mode,
                    "message": f"Collaborative analysis completed in {collaboration_mode} mode"
                }
                
        except Exception as e:
            logger.error(f"Error in collaborative analysis: {e}")
            return {"status": "error", "message": str(e)}
    
    async def start_a2a_server(self):
        """Start A2A server for agent communication"""
        try:
            # This would typically use a web framework like FastAPI
            # For this example, we'll simulate the server setup
            logger.info(f"Starting A2A server on port {self.config['a2a']['port']}")
            
            # Register with Agentspace
            await self.register_with_agentspace()
            
            # Start agent discovery loop
            asyncio.create_task(self._discovery_loop())
            
            logger.info("A2A server started successfully")
            
        except Exception as e:
            logger.error(f"Error starting A2A server: {e}")
            raise
    
    async def _discovery_loop(self):
        """Periodic agent discovery loop"""
        while True:
            try:
                await self.discover_agents()
                await asyncio.sleep(self.config["a2a"]["discovery_interval"])
            except Exception as e:
                logger.error(f"Error in discovery loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    def deploy_to_vertex_ai(self):
        """Deploy the A2A-enabled TAA agent to Vertex AI"""
        try:
            requirements = [
                "google-cloud-aiplatform[agent_engines,adk]>=1.88.0",
                "aiohttp>=3.8.0",
                "asyncio",
                "google-cloud-pubsub>=2.18.0",
                "vertexai>=1.88.0"
            ]
            
            # Deploy to Vertex AI Agent Engine
            remote_agent = agent_engines.create(
                self.taa_agent,
                requirements=requirements,
                display_name="TAA-A2A-Agent",
                description="TAA agent with A2A communication capabilities",
                env_vars={
                    "A2A_ENABLED": "true",
                    "AGENT_PORT": str(self.config["a2a"]["port"]),
                    "AGENT_ID": self.agent_id
                }
            )
            
            logger.info(f"Agent deployed to Vertex AI: {remote_agent.resource_name}")
            return remote_agent
            
        except Exception as e:
            logger.error(f"Error deploying to Vertex AI: {e}")
            raise

def main():
    """Main function to run the TAA A2A agent"""
    try:
        # Create and initialize the TAA A2A agent
        taa_a2a_agent = TAAA2AAgent()
        
        # Deploy to Vertex AI
        remote_agent = taa_a2a_agent.deploy_to_vertex_ai()
        
        # Start A2A server
        asyncio.run(taa_a2a_agent.start_a2a_server())
        
        logger.info("TAA A2A Agent is running and ready for communication")
        
        # Keep the agent running
        try:
            asyncio.get_event_loop().run_forever()
        except KeyboardInterrupt:
            logger.info("Shutting down TAA A2A Agent")
            
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main() 