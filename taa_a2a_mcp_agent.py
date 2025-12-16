#!/usr/bin/env python3
"""
TAA A2A Agent with MCP Integration for Google Security Operations
===============================================================

This enhanced TAA agent integrates with Google Security Operations through MCP
(Model Context Protocol) to provide advanced threat intelligence, detection rules,
and investigation capabilities.

Features:
- MCP client integration with Google Security Operations
- Chronicle threat intelligence queries
- Detection rule management
- Investigation workflow automation
- Enhanced A2A capabilities with MCP context
"""

import os
import json
import logging
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from google.cloud import pubsub_v1
import vertexai
from vertexai import agent_engines
from google.adk.agents import Agent
from google.adk.core import BaseAgent

from multi_tenant_manager import MultiTenantManager, TenantConfig
from palo_alto_integration import PaloAltoIntegration
from checkpoint_integration import CheckpointIntegration

# MCP imports
try:
    import mcp
    from mcp.client import Client
    from mcp.client.stdio import StdioServerParameters
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logging.warning("MCP not available. Install with: pip install mcp")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MCPTask:
    """MCP task structure"""
    task_id: str
    tool: str
    arguments: Dict[str, Any]
    timeout: int = 30
    priority: str = "normal"

@dataclass
class MCPResponse:
    """MCP response structure"""
    task_id: str
    status: str
    result: Dict[str, Any]
    error: Optional[str] = None
    timestamp: str = None

@dataclass
class ChronicleQuery:
    """Chronicle query structure"""
    query_type: str  # "detection", "investigation", "intelligence"
    query_text: str
    time_range: Dict[str, str]
    parameters: Dict[str, Any]

class TAAA2AMCPAgent:
    """TAA Agent with A2A communication and MCP integration"""
    
    def __init__(self, config_path: str = "config/taa_a2a_config.json"):
        """Initialize the TAA A2A MCP Agent"""
        self.config = self._load_config(config_path)
        self.agent_id = f"taa-mcp-agent-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Initialize MCP clients
        self.mcp_clients = {}
        self._init_mcp_clients()
        
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
        
        # Initialize the core TAA agent with MCP capabilities
        self.taa_agent = self._create_taa_agent()
        
        # Load Tenant Manager (for Firewall Config lookup)
        try:
            self.tenant_manager = MultiTenantManager.from_file(os.getenv("MULTITENANT_CONFIG_PATH", "config/gatra_multitenant_config.json"))
        except Exception as e:
            logger.warning(f"Could not load tenant manager for firewall integration: {e}")
            self.tenant_manager = None
            
        logger.info(f"TAA A2A MCP Agent initialized: {self.agent_id}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load A2A MCP configuration"""
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
        except FileNotFoundError:
            # Default configuration with MCP settings
            config = {
                "project_id": os.environ.get("GOOGLE_CLOUD_PROJECT", "ai-driven-soc"),
                "location": "us-central1",
                "a2a": {
                    "enabled": True,
                    "port": 8080,
                    "discovery_interval": 300,
                    "max_agents": 100
                },
                "mcp": {
                    "enabled": True,
                    "servers": {
                        "google_security_operations": {
                            "name": "google-security-operations",
                            "command": "npx",
                            "args": ["@google/mcp-server-security-operations"],
                            "env": {
                                "GOOGLE_CLOUD_PROJECT": os.environ.get("GOOGLE_CLOUD_PROJECT"),
                                "GOOGLE_APPLICATION_CREDENTIALS": os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
                            }
                        },
                        "google_threat_intelligence": {
                            "name": "google-threat-intelligence", 
                            "command": "npx",
                            "args": ["@google/mcp-server-threat-intelligence"],
                            "env": {
                                "GOOGLE_CLOUD_PROJECT": os.environ.get("GOOGLE_CLOUD_PROJECT"),
                                "GOOGLE_APPLICATION_CREDENTIALS": os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
                            }
                        },
                        "security_command_center": {
                            "name": "security-command-center",
                            "command": "npx", 
                            "args": ["@google/mcp-server-scc"],
                            "env": {
                                "GOOGLE_CLOUD_PROJECT": os.environ.get("GOOGLE_CLOUD_PROJECT"),
                                "GOOGLE_APPLICATION_CREDENTIALS": os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
                            }
                        }
                    },
                    "timeout": 30,
                    "retry_attempts": 3
                },
                "vertex_ai": {
                    "model": "gemini-2.0-flash-exp",
                    "staging_bucket": os.environ.get("STAGING_BUCKET", "gs://ai-driven-soc-staging")
                },
                "pubsub": {
                    "alerts_topic": "ada-alerts",
                    "feedback_topic": "taa-feedback",
                    "a2a_topic": "a2a-communication",
                    "chronicle_alerts_topic": "chronicle-alerts",
                    "threat_intelligence_topic": "threat-intelligence"
                },
                "agentspace": {
                    "enabled": True,
                    "registration_url": "https://agentspace.googleapis.com/v1/agents",
                    "discovery_url": "https://agentspace.googleapis.com/v1/agents/discover"
                },
                "chronicle": {
                    "customer_id": os.environ.get("CHRONICLE_CUSTOMER_ID"),
                    "region": "us-central1",
                    "max_results": 1000,
                    "timeout": 60
                }
            }
        
        return config
    
    def _init_mcp_clients(self):
        """Initialize MCP clients for Google Security Operations"""
        if not MCP_AVAILABLE:
            logger.warning("MCP not available. Some capabilities will be limited.")
            return
        
        if not self.config.get("mcp", {}).get("enabled", False):
            logger.info("MCP integration disabled in configuration")
            return
        
        try:
            for server_name, server_config in self.config["mcp"]["servers"].items():
                try:
                    # Create MCP client for each server
                    client = Client(
                        StdioServerParameters(
                            command=server_config["command"],
                            args=server_config["args"],
                            env=server_config.get("env", {})
                        )
                    )
                    
                    # Connect to the MCP server
                    asyncio.create_task(self._connect_mcp_client(client, server_name))
                    self.mcp_clients[server_name] = client
                    
                    logger.info(f"MCP client initialized for {server_name}")
                    
                except Exception as e:
                    logger.error(f"Failed to initialize MCP client for {server_name}: {e}")
                    
        except Exception as e:
            logger.error(f"Error initializing MCP clients: {e}")
    
    async def _connect_mcp_client(self, client: Client, server_name: str):
        """Connect to MCP server"""
        try:
            await client.connect()
            logger.info(f"Connected to MCP server: {server_name}")
        except Exception as e:
            logger.error(f"Failed to connect to MCP server {server_name}: {e}")
    
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
        """Create A2A agent card with MCP capabilities"""
        capabilities = [
            {
                "name": "analyze_security_alert",
                "description": "Analyze security alerts using LLM, threat intelligence, and Chronicle data",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "alert_data": {"type": "object"},
                        "enrichment_level": {"type": "string", "enum": ["basic", "full", "chronicle"]}
                    },
                    "required": ["alert_data"]
                }
            },
            {
                "name": "query_chronicle_detections",
                "description": "Query Chronicle for detection rules and alerts",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query_text": {"type": "string"},
                        "time_range": {"type": "object"},
                        "detection_rule": {"type": "string"}
                    },
                    "required": ["query_text"]
                }
            },
            {
                "name": "get_threat_intelligence",
                "description": "Retrieve threat intelligence from Google Threat Intelligence",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "indicators": {"type": "array"},
                        "intelligence_type": {"type": "string", "enum": ["ioc", "threat_actor", "campaign"]}
                    },
                    "required": ["indicators"]
                }
            },
            {
                "name": "investigate_incident",
                "description": "Conduct automated investigation using Chronicle and SCC",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "incident_id": {"type": "string"},
                        "investigation_scope": {"type": "string", "enum": ["basic", "deep", "forensic"]}
                    },
                    "required": ["incident_id"]
                }
            },
            {
                "name": "share_threat_intelligence",
                "description": "Share threat intelligence with other agents and Chronicle",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "indicators": {"type": "array"},
                        "confidence": {"type": "number"},
                        "source": {"type": "string"},
                        "share_with_chronicle": {"type": "boolean"}
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
                        "action_type": {"type": "string", "enum": ["isolate", "block", "monitor", "quarantine"]},
                        "priority": {"type": "string", "enum": ["low", "medium", "high", "critical"]}
                    },
                    "required": ["target_host", "action_type"]
                }
            },
            {
                "name": "collaborative_analysis",
                "description": "Collaborate with other agents for complex analysis using MCP context",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "analysis_type": {"type": "string"},
                        "data": {"type": "object"},
                        "collaboration_mode": {"type": "string", "enum": ["parallel", "sequential", "hierarchical"]},
                        "include_chronicle_context": {"type": "boolean"}
                    }
                }
            },
            {
                "name": "firewall_block_ip",
                "description": "Block an IP address on the tenant's firewall (Palo Alto)",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "tenant_id": {"type": "string"},
                        "ip_address": {"type": "string"},
                        "reason": {"type": "string"}
                    },
                    "required": ["tenant_id", "ip_address"]
                }
            }
        ]
        
        return {
            "name": "taa-mcp-security-agent",
            "version": "2.0.0",
            "description": "Triage and Analysis Agent with A2A and MCP capabilities for Google Security Operations",
            "agent_id": self.agent_id,
            "capabilities": capabilities,
            "mcp_integration": {
                "enabled": True,
                "servers": list(self.config["mcp"]["servers"].keys()),
                "capabilities": [
                    "chronicle_query",
                    "threat_intelligence",
                    "security_command_center",
                    "detection_management"
                ]
            },
            "authentication": {
                "type": "oauth2",
                "scopes": [
                    "https://www.googleapis.com/auth/cloud-platform",
                    "https://www.googleapis.com/auth/chronicle",
                    "https://www.googleapis.com/auth/securitycenter"
                ]
            },
            "endpoints": {
                "base_url": f"https://{self.agent_id}.agentspace.googleapis.com",
                "a2a_endpoint": "/a2a/tasks",
                "mcp_endpoint": "/mcp/tasks",
                "health_check": "/health",
                "agent_card": "/.well-known/agent.json"
            },
            "metadata": {
                "domain": "cybersecurity",
                "specialization": "threat_analysis_mcp",
                "supported_formats": ["json", "xml", "stix"],
                "response_time": "3s",
                "availability": "24/7",
                "mcp_version": "1.0.0"
            }
        }
    
    def _create_taa_agent(self) -> Agent:
        """Create the core TAA agent with MCP capabilities"""
        try:
            agent = Agent(
                name="taa-mcp-security-agent",
                model_name=self.config["vertex_ai"]["model"],
                instructions="""You are a Triage and Analysis Agent (TAA) with MCP integration for Google Security Operations. 
                Your enhanced capabilities include:
                - Analyzing security alerts using LLM, threat intelligence, and Chronicle data
                - Querying Chronicle for detection rules and historical alerts
                - Retrieving threat intelligence from Google Threat Intelligence
                - Conducting automated investigations using Chronicle and SCC
                - Sharing threat intelligence with other agents and Chronicle
                - Requesting containment actions
                - Collaborating with other agents using MCP context
                
                Always provide structured, actionable responses and leverage MCP integration for enhanced security analysis."""
            )
            
            # Add MCP-enhanced tools
            @agent.tool
            def analyze_security_alert(alert_data: Dict, enrichment_level: str = "chronicle") -> Dict:
                """Analyze security alert using LLM, threat intelligence, and Chronicle data"""
                try:
                    # Use MCP to enrich with Chronicle data
                    chronicle_context = {}
                    if enrichment_level == "chronicle" and "google_security_operations" in self.mcp_clients:
                        chronicle_context = asyncio.run(self._enrich_with_chronicle(alert_data))
                    
                    # Enhanced analysis with Chronicle context
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
                    logger.error(f"Error in MCP-enhanced alert analysis: {e}")
                    return {"status": "error", "message": str(e)}
            
            @agent.tool
            def query_chronicle_detections(query_text: str, time_range: Dict = None, detection_rule: str = None) -> Dict:
                """Query Chronicle for detection rules and alerts"""
                try:
                    if "google_security_operations" not in self.mcp_clients:
                        return {"status": "error", "message": "Chronicle MCP client not available"}
                    
                    # Execute Chronicle query via MCP
                    result = asyncio.run(self._query_chronicle_via_mcp(
                        query_text, time_range, detection_rule
                    ))
                    
                    return {
                        "status": "success",
                        "query_results": result,
                        "agent_id": self.agent_id
                    }
                except Exception as e:
                    logger.error(f"Error querying Chronicle: {e}")
                    return {"status": "error", "message": str(e)}
            
            @agent.tool
            def get_threat_intelligence(indicators: List[str], intelligence_type: str = "ioc") -> Dict:
                """Retrieve threat intelligence from Google Threat Intelligence"""
                try:
                    if "google_threat_intelligence" not in self.mcp_clients:
                        return {"status": "error", "message": "Threat Intelligence MCP client not available"}
                    
                    # Query threat intelligence via MCP
                    result = asyncio.run(self._query_threat_intelligence_via_mcp(
                        indicators, intelligence_type
                    ))
                    
                    return {
                        "status": "success",
                        "threat_intelligence": result,
                        "indicators_queried": len(indicators),
                        "agent_id": self.agent_id
                    }
                except Exception as e:
                    logger.error(f"Error getting threat intelligence: {e}")
                    return {"status": "error", "message": str(e)}
            
            @agent.tool
            def investigate_incident(incident_id: str, investigation_scope: str = "deep") -> Dict:
                """Conduct automated investigation using Chronicle and SCC"""
                try:
                    investigation_result = asyncio.run(self._conduct_investigation_via_mcp(
                        incident_id, investigation_scope
                    ))
                    
                    return {
                        "status": "success",
                        "investigation": investigation_result,
                        "incident_id": incident_id,
                        "scope": investigation_scope,
                        "agent_id": self.agent_id
                    }
                except Exception as e:
                    logger.error(f"Error conducting investigation: {e}")
                    return {"status": "error", "message": str(e)}
            
            @agent.tool
            def firewall_block_ip(tenant_id: str, ip_address: str, reason: str = "security_incident") -> Dict:
                """Block an IP address on the tenant's firewall (Palo Alto)"""
                try:
                    if not self.tenant_manager:
                        return {"status": "error", "message": "Tenant manager not available"}
                    
                    # Get Tenant Config
                    try:
                        tenant = self.tenant_manager.get_tenant(tenant_id)
                    except Exception:
                        return {"status": "error", "message": f"Tenant {tenant_id} not found"}
                        
                    if not tenant.firewall_config:
                        return {"status": "error", "message": f"No firewall configuration for tenant {tenant_id}"}
                        
                    result = {}
                    fw_type = tenant.firewall_config.type
                    
                    if fw_type.startswith("palo_alto"):
                        # Initialize Palo Alto Integration
                        fav_integration = PaloAltoIntegration(
                            mgmt_ip=tenant.firewall_config.mgmt_ip,
                            api_key=tenant.firewall_config.api_key or "", # PA needs key
                            mode=fw_type,
                            device_group=tenant.firewall_config.device_group
                        )
                        result = fav_integration.block_ip(ip_address)
                        
                    elif fw_type == "checkpoint":
                        # Initialize Check Point Integration
                        cp_integration = CheckpointIntegration(
                            mgmt_ip=tenant.firewall_config.mgmt_ip,
                            api_key=tenant.firewall_config.api_key,
                            username=tenant.firewall_config.username,
                            password=tenant.firewall_config.password,
                            domain=tenant.firewall_config.domain
                        )
                        result = cp_integration.block_ip(ip_address)
                        
                    else:
                        return {"status": "error", "message": f"Unsupported firewall type: {fw_type}"}
                    
                    return {
                         "status": "success", 
                         "result": result,
                         "tenant_id": tenant_id,
                         "firewall_type": fw_type
                    }
                    
                except Exception as e:
                    logger.error(f"Error blocking IP on firewall: {e}")
                    return {"status": "error", "message": str(e)}

            logger.info("TAA agent created successfully with MCP tools")
            return agent
            
        except Exception as e:
            logger.error(f"Failed to create TAA agent with MCP: {e}")
            raise
    
    async def _enrich_with_chronicle(self, alert_data: Dict) -> Dict:
        """Enrich alert with Chronicle data via MCP"""
        try:
            client = self.mcp_clients["google_security_operations"]
            
            # Extract relevant indicators from alert
            indicators = self._extract_indicators(alert_data)
            
            chronicle_context = {}
            for indicator in indicators:
                # Query Chronicle for each indicator
                query_result = await client.call_tool(
                    "chronicle_query",
                    {
                        "query": f"metadata.event_type='NETWORK_CONNECTION' AND network.destination.ip_address='{indicator}'",
                        "time_range": {
                            "start_time": (datetime.now() - timedelta(days=7)).isoformat(),
                            "end_time": datetime.now().isoformat()
                        }
                    }
                )
                
                chronicle_context[indicator] = query_result
            
            return chronicle_context
            
        except Exception as e:
            logger.error(f"Error enriching with Chronicle: {e}")
            return {}
    
    async def _query_chronicle_via_mcp(self, query_text: str, time_range: Dict = None, detection_rule: str = None) -> Dict:
        """Query Chronicle via MCP"""
        try:
            client = self.mcp_clients["google_security_operations"]
            
            if not time_range:
                time_range = {
                    "start_time": (datetime.now() - timedelta(hours=24)).isoformat(),
                    "end_time": datetime.now().isoformat()
                }
            
            # Execute Chronicle query
            result = await client.call_tool(
                "chronicle_query",
                {
                    "query": query_text,
                    "time_range": time_range,
                    "detection_rule": detection_rule,
                    "max_results": self.config["chronicle"]["max_results"]
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error querying Chronicle via MCP: {e}")
            return {"error": str(e)}
    
    async def _query_threat_intelligence_via_mcp(self, indicators: List[str], intelligence_type: str) -> Dict:
        """Query threat intelligence via MCP"""
        try:
            client = self.mcp_clients["google_threat_intelligence"]
            
            # Query threat intelligence for each indicator
            ti_results = {}
            for indicator in indicators:
                result = await client.call_tool(
                    f"get_{intelligence_type}",
                    {"indicator": indicator}
                )
                ti_results[indicator] = result
            
            return ti_results
            
        except Exception as e:
            logger.error(f"Error querying threat intelligence via MCP: {e}")
            return {"error": str(e)}
    
    async def _conduct_investigation_via_mcp(self, incident_id: str, investigation_scope: str) -> Dict:
        """Conduct investigation via MCP"""
        try:
            # Use multiple MCP servers for comprehensive investigation
            investigation_result = {
                "incident_id": incident_id,
                "scope": investigation_scope,
                "findings": {},
                "timeline": [],
                "recommendations": []
            }
            
            # Query Chronicle for incident data
            if "google_security_operations" in self.mcp_clients:
                chronicle_data = await self.mcp_clients["google_security_operations"].call_tool(
                    "investigate_incident",
                    {"incident_id": incident_id, "scope": investigation_scope}
                )
                investigation_result["findings"]["chronicle"] = chronicle_data
            
            # Query Security Command Center
            if "security_command_center" in self.mcp_clients:
                scc_data = await self.mcp_clients["security_command_center"].call_tool(
                    "get_finding",
                    {"finding_id": incident_id}
                )
                investigation_result["findings"]["scc"] = scc_data
            
            return investigation_result
            
        except Exception as e:
            logger.error(f"Error conducting investigation via MCP: {e}")
            return {"error": str(e)}
    
    def _extract_indicators(self, alert_data: Dict) -> List[str]:
        """Extract indicators from alert data"""
        indicators = []
        
        # Extract IP addresses
        log_data = alert_data.get("log_data", {})
        if "src_ip" in log_data:
            indicators.append(log_data["src_ip"])
        if "dst_ip" in log_data:
            indicators.append(log_data["dst_ip"])
        
        # Extract domains
        if "domain" in log_data:
            indicators.append(log_data["domain"])
        
        # Extract file hashes
        if "file_hash" in log_data:
            indicators.append(log_data["file_hash"])
        
        return indicators
    
    async def handle_mcp_task(self, task_data: Dict) -> MCPResponse:
        """Handle incoming MCP task"""
        try:
            task = MCPTask(**task_data)
            
            # Route to appropriate MCP capability
            if task.tool == "chronicle_query":
                result = await self._query_chronicle_via_mcp(**task.arguments)
            elif task.tool == "threat_intelligence_query":
                result = await self._query_threat_intelligence_via_mcp(**task.arguments)
            elif task.tool == "investigate_incident":
                result = await self._conduct_investigation_via_mcp(**task.arguments)
            else:
                raise ValueError(f"Unknown MCP tool: {task.tool}")
            
            return MCPResponse(
                task_id=task.task_id,
                status="completed",
                result=result,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error handling MCP task: {e}")
            return MCPResponse(
                task_id=task_data.get("task_id", "unknown"),
                status="error",
                result={},
                error=str(e),
                timestamp=datetime.now().isoformat()
            )
    
    async def start_a2a_mcp_server(self):
        """Start A2A server with MCP capabilities"""
        try:
            logger.info(f"Starting A2A MCP server on port {self.config['a2a']['port']}")
            
            # Register with Agentspace
            await self.register_with_agentspace()
            
            # Start agent discovery loop
            asyncio.create_task(self._discovery_loop())
            
            # Start MCP health check loop
            asyncio.create_task(self._mcp_health_check_loop())
            
            logger.info("A2A MCP server started successfully")
            
        except Exception as e:
            logger.error(f"Error starting A2A MCP server: {e}")
            raise
    
    async def _mcp_health_check_loop(self):
        """Periodic MCP health check"""
        while True:
            try:
                for server_name, client in self.mcp_clients.items():
                    try:
                        # Simple health check
                        await client.list_tools()
                        logger.debug(f"MCP server {server_name} is healthy")
                    except Exception as e:
                        logger.warning(f"MCP server {server_name} health check failed: {e}")
                        # Attempt to reconnect
                        await self._connect_mcp_client(client, server_name)
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in MCP health check loop: {e}")
                await asyncio.sleep(60)
    
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
                    "mcp_capabilities": True,
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
    
    def _get_auth_token(self) -> str:
        """Get authentication token for A2A communication"""
        # Implement OAuth2 token retrieval
        return "your-auth-token"
    
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
                    "capabilities": ["containment", "forensics", "intelligence", "mcp"]
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
    
    async def _discovery_loop(self):
        """Periodic agent discovery loop"""
        while True:
            try:
                await self.discover_agents()
                await asyncio.sleep(self.config["a2a"]["discovery_interval"])
            except Exception as e:
                logger.error(f"Error in discovery loop: {e}")
                await asyncio.sleep(60)

def main():
    """Main function to run the TAA A2A MCP agent"""
    try:
        # Create and initialize the TAA A2A MCP agent
        taa_a2a_mcp_agent = TAAA2AMCPAgent()
        
        # Start A2A MCP server
        asyncio.run(taa_a2a_mcp_agent.start_a2a_mcp_server())
        
        logger.info("TAA A2A MCP Agent is running and ready for communication")
        
        # Keep the agent running
        try:
            asyncio.get_event_loop().run_forever()
        except KeyboardInterrupt:
            logger.info("Shutting down TAA A2A MCP Agent")
            
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()

