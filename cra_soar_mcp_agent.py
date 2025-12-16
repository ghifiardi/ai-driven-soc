#!/usr/bin/env python3
"""
Enhanced Containment Response Agent (CRA) with SOAR and MCP Integration
=====================================================================

This enhanced CRA integrates with:
- MCP Threat Intelligence for informed containment decisions
- SOAR workflow engine for automated response orchestration
- A2A communication for multi-agent coordination
- Advanced playbook automation with ML-driven decision making

Key Features:
- MCP-enhanced threat intelligence for containment decisions
- SOAR workflow orchestration with conditional logic
- Automated playbook execution with approval workflows
- Integration with multiple security tools and platforms
- Real-time incident response coordination
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path

import aiohttp
import structlog
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from google.cloud import pubsub_v1, firestore
from google.cloud import aiplatform
import vertexai
from vertexai.generative_models import GenerativeModel

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger("CRA_SOAR_MCP")

class RiskLevel(Enum):
    """Enhanced risk levels for containment actions"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class ActionType(Enum):
    """Types of containment actions"""
    ISOLATE = "isolate"
    BLOCK = "block"
    QUARANTINE = "quarantine"
    MONITOR = "monitor"
    PATCH = "patch"
    DISABLE = "disable"
    RESET = "reset"
    ESCALATE = "escalate"

class PlaybookStatus(Enum):
    """Playbook execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    APPROVAL_REQUIRED = "approval_required"
    CANCELLED = "cancelled"

@dataclass
class ThreatIntelligence:
    """Threat intelligence data structure"""
    indicator: str
    indicator_type: str  # ip, domain, hash, url
    reputation: str
    risk_score: float
    threat_types: List[str]
    confidence: float
    sources: List[str]
    last_updated: str
    metadata: Dict[str, Any]

@dataclass
class ContainmentAction:
    """Containment action data structure"""
    action_id: str
    action_type: ActionType
    target: str
    parameters: Dict[str, Any]
    risk_level: RiskLevel
    justification: str
    requires_approval: bool
    approval_status: Optional[str] = None
    executed_by: Optional[str] = None
    executed_at: Optional[str] = None
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None

@dataclass
class SOARPlaybook:
    """SOAR playbook data structure"""
    playbook_id: str
    name: str
    description: str
    trigger_conditions: Dict[str, Any]
    actions: List[ContainmentAction]
    approval_workflow: Dict[str, Any]
    timeout_minutes: int
    retry_count: int
    enabled: bool = True

@dataclass
class Incident:
    """Incident data structure"""
    incident_id: str
    alert_id: str
    severity: RiskLevel
    title: str
    description: str
    affected_assets: List[str]
    indicators: List[str]
    threat_intelligence: List[ThreatIntelligence]
    containment_actions: List[ContainmentAction]
    playbook_id: Optional[str] = None
    status: str = "new"
    created_at: str = ""
    updated_at: str = ""

class MCPThreatIntelligenceClient:
    """Client for MCP Threat Intelligence integration"""
    
    def __init__(self, mcp_config: Dict[str, Any]):
        self.config = mcp_config
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = await self._connect_mcp()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def _connect_mcp(self) -> ClientSession:
        """Connect to MCP threat intelligence server"""
        try:
            server_params = StdioServerParameters(
                command="python3",
                args=["threat_intelligence_mcp_server.py"],
                env=self.config.get("env", {})
            )
            
            session = await stdio_client(server_params)
            await session.initialize()
            
            logger.info("Connected to MCP Threat Intelligence server")
            return session
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            raise
    
    async def query_indicator(self, indicator: str, indicator_type: str) -> ThreatIntelligence:
        """Query threat intelligence for an indicator"""
        try:
            if indicator_type == "ip":
                result = await self.session.call_tool("query_ip", {"ip": indicator})
            elif indicator_type == "domain":
                result = await self.session.call_tool("query_domain", {"domain": indicator})
            elif indicator_type in ["hash", "md5", "sha1", "sha256"]:
                result = await self.session.call_tool("query_hash", {"hash": indicator})
            else:
                raise ValueError(f"Unsupported indicator type: {indicator_type}")
            
            # Parse the result
            if result and len(result) > 0:
                ti_data = json.loads(result[0].text)
                return ThreatIntelligence(
                    indicator=indicator,
                    indicator_type=indicator_type,
                    reputation=ti_data.get("summary", {}).get("reputation", "unknown"),
                    risk_score=ti_data.get("summary", {}).get("risk_score", 0.0),
                    threat_types=ti_data.get("summary", {}).get("threat_types", []),
                    confidence=0.8,  # Default confidence
                    sources=list(ti_data.get("sources", {}).keys()),
                    last_updated=datetime.now().isoformat(),
                    metadata=ti_data
                )
            else:
                # Return default unknown threat intelligence
                return ThreatIntelligence(
                    indicator=indicator,
                    indicator_type=indicator_type,
                    reputation="unknown",
                    risk_score=0.0,
                    threat_types=[],
                    confidence=0.0,
                    sources=[],
                    last_updated=datetime.now().isoformat(),
                    metadata={}
                )
                
        except Exception as e:
            logger.error(f"Error querying threat intelligence: {e}")
            # Return default unknown threat intelligence
            return ThreatIntelligence(
                indicator=indicator,
                indicator_type=indicator_type,
                reputation="unknown",
                risk_score=0.0,
                threat_types=[],
                confidence=0.0,
                sources=[],
                last_updated=datetime.now().isoformat(),
                metadata={"error": str(e)}
            )
    
    async def enrich_indicators(self, indicators: List[str]) -> List[ThreatIntelligence]:
        """Enrich multiple indicators with threat intelligence"""
        tasks = []
        
        for indicator in indicators:
            # Determine indicator type
            indicator_type = self._detect_indicator_type(indicator)
            if indicator_type:
                tasks.append(self.query_indicator(indicator, indicator_type))
        
        # Execute all queries concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and return valid results
        valid_results = []
        for result in results:
            if isinstance(result, ThreatIntelligence):
                valid_results.append(result)
            else:
                logger.warning(f"Failed to enrich indicator: {result}")
        
        return valid_results
    
    def _detect_indicator_type(self, indicator: str) -> Optional[str]:
        """Detect the type of indicator"""
        indicator = indicator.strip()
        
        # IP address detection
        if self._is_ip_address(indicator):
            return "ip"
        
        # Domain detection
        if "." in indicator and not indicator.startswith("http"):
            return "domain"
        
        # Hash detection
        if len(indicator) in [32, 40, 64] and all(c in "0123456789abcdefABCDEF" for c in indicator):
            return "hash"
        
        # URL detection
        if indicator.startswith(("http://", "https://")):
            return "url"
        
        return None
    
    def _is_ip_address(self, indicator: str) -> bool:
        """Check if indicator is an IP address"""
        try:
            parts = indicator.split(".")
            if len(parts) != 4:
                return False
            return all(0 <= int(part) <= 255 for part in parts)
        except:
            return False

class SOARWorkflowEngine:
    """SOAR Workflow Engine for automated response orchestration"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.playbooks: Dict[str, SOARPlaybook] = {}
        self.running_workflows: Dict[str, Dict] = {}
        self._load_playbooks()
    
    def _load_playbooks(self):
        """Load SOAR playbooks from configuration"""
        playbooks_config = self.config.get("playbooks", {})
        
        for playbook_id, playbook_data in playbooks_config.items():
            actions = []
            for action_data in playbook_data.get("actions", []):
                actions.append(ContainmentAction(
                    action_id=action_data["action_id"],
                    action_type=ActionType(action_data["action_type"]),
                    target=action_data["target"],
                    parameters=action_data["parameters"],
                    risk_level=RiskLevel(action_data["risk_level"]),
                    justification=action_data["justification"],
                    requires_approval=action_data["requires_approval"]
                ))
            
            self.playbooks[playbook_id] = SOARPlaybook(
                playbook_id=playbook_id,
                name=playbook_data["name"],
                description=playbook_data["description"],
                trigger_conditions=playbook_data["trigger_conditions"],
                actions=actions,
                approval_workflow=playbook_data.get("approval_workflow", {}),
                timeout_minutes=playbook_data.get("timeout_minutes", 60),
                retry_count=playbook_data.get("retry_count", 3),
                enabled=playbook_data.get("enabled", True)
            )
        
        logger.info(f"Loaded {len(self.playbooks)} SOAR playbooks")
    
    def select_playbook(self, incident: Incident) -> Optional[SOARPlaybook]:
        """Select appropriate playbook based on incident characteristics"""
        best_match = None
        best_score = 0
        
        for playbook in self.playbooks.values():
            if not playbook.enabled:
                continue
                
            score = self._calculate_playbook_score(playbook, incident)
            if score > best_score:
                best_score = score
                best_match = playbook
        
        logger.info(f"Selected playbook: {best_match.name if best_match else 'None'} (score: {best_score})")
        return best_match
    
    def _calculate_playbook_score(self, playbook: SOARPlaybook, incident: Incident) -> float:
        """Calculate match score for a playbook"""
        score = 0.0
        conditions = playbook.trigger_conditions
        
        # Severity matching
        if conditions.get("severity") == incident.severity.value:
            score += 0.4
        
        # Threat type matching
        incident_threat_types = set()
        for ti in incident.threat_intelligence:
            incident_threat_types.update(ti.threat_types)
        
        playbook_threat_types = set(conditions.get("threat_types", []))
        if incident_threat_types & playbook_threat_types:
            score += 0.3
        
        # Asset type matching
        playbook_asset_types = set(conditions.get("asset_types", []))
        if playbook_asset_types:
            # Simple asset type detection (can be enhanced)
            score += 0.2
        
        # Risk score matching
        max_risk_score = max([ti.risk_score for ti in incident.threat_intelligence], default=0)
        min_risk_threshold = conditions.get("min_risk_score", 0)
        if max_risk_score >= min_risk_threshold:
            score += 0.1
        
        return score
    
    async def execute_playbook(self, playbook: SOARPlaybook, incident: Incident) -> Dict[str, Any]:
        """Execute a SOAR playbook"""
        workflow_id = f"{incident.incident_id}_{playbook.playbook_id}_{int(time.time())}"
        
        logger.info(f"Executing playbook {playbook.name} for incident {incident.incident_id}")
        
        workflow_state = {
            "workflow_id": workflow_id,
            "playbook_id": playbook.playbook_id,
            "incident_id": incident.incident_id,
            "status": PlaybookStatus.RUNNING.value,
            "started_at": datetime.now().isoformat(),
            "actions_completed": [],
            "actions_failed": [],
            "actions_pending_approval": [],
            "current_step": 0,
            "retry_count": 0
        }
        
        self.running_workflows[workflow_id] = workflow_state
        
        try:
            # Execute actions sequentially or in parallel based on playbook configuration
            for i, action in enumerate(playbook.actions):
                workflow_state["current_step"] = i
                
                # Check if action requires approval
                if action.requires_approval:
                    workflow_state["actions_pending_approval"].append(asdict(action))
                    workflow_state["status"] = PlaybookStatus.APPROVAL_REQUIRED.value
                    logger.info(f"Action {action.action_id} requires approval")
                    break
                
                # Execute action
                action_result = await self._execute_action(action, incident)
                
                if action_result["success"]:
                    workflow_state["actions_completed"].append({
                        "action_id": action.action_id,
                        "result": action_result
                    })
                else:
                    workflow_state["actions_failed"].append({
                        "action_id": action.action_id,
                        "error": action_result.get("error")
                    })
                    
                    # Check if we should continue or stop on failure
                    if playbook.trigger_conditions.get("stop_on_failure", True):
                        workflow_state["status"] = PlaybookStatus.FAILED.value
                        break
            
            # Update final status
            if workflow_state["status"] == PlaybookStatus.RUNNING.value:
                if workflow_state["actions_failed"]:
                    workflow_state["status"] = PlaybookStatus.FAILED.value
                else:
                    workflow_state["status"] = PlaybookStatus.COMPLETED.value
            
            workflow_state["completed_at"] = datetime.now().isoformat()
            
            return workflow_state
            
        except Exception as e:
            workflow_state["status"] = PlaybookStatus.FAILED.value
            workflow_state["error"] = str(e)
            workflow_state["completed_at"] = datetime.now().isoformat()
            logger.error(f"Playbook execution failed: {e}")
            return workflow_state
        
        finally:
            # Clean up completed workflow
            if workflow_state["status"] in [PlaybookStatus.COMPLETED.value, PlaybookStatus.FAILED.value]:
                self.running_workflows.pop(workflow_id, None)
    
    async def _execute_action(self, action: ContainmentAction, incident: Incident) -> Dict[str, Any]:
        """Execute a containment action"""
        logger.info(f"Executing action: {action.action_type.value} on {action.target}")
        
        try:
            if action.action_type == ActionType.ISOLATE:
                return await self._isolate_asset(action, incident)
            elif action.action_type == ActionType.BLOCK:
                return await self._block_indicator(action, incident)
            elif action.action_type == ActionType.QUARANTINE:
                return await self._quarantine_file(action, incident)
            elif action.action_type == ActionType.MONITOR:
                return await self._monitor_asset(action, incident)
            elif action.action_type == ActionType.ESCALATE:
                return await self._escalate_incident(action, incident)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported action type: {action.action_type.value}"
                }
        
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _isolate_asset(self, action: ContainmentAction, incident: Incident) -> Dict[str, Any]:
        """Isolate an asset (simulated)"""
        # In a real implementation, this would integrate with network security tools
        # like firewalls, NAC systems, or endpoint security platforms
        
        await asyncio.sleep(1)  # Simulate API call delay
        
        return {
            "success": True,
            "action": "isolate",
            "target": action.target,
            "timestamp": datetime.now().isoformat(),
            "details": {
                "isolation_method": "network_quarantine",
                "isolation_duration": "until_manual_release"
            }
        }
    
    async def _block_indicator(self, action: ContainmentAction, incident: Incident) -> Dict[str, Any]:
        """Block an indicator (simulated)"""
        # In a real implementation, this would integrate with firewalls, DNS filters, etc.
        
        await asyncio.sleep(1)  # Simulate API call delay
        
        return {
            "success": True,
            "action": "block",
            "target": action.target,
            "timestamp": datetime.now().isoformat(),
            "details": {
                "block_type": "firewall_rule",
                "rule_id": f"block_{action.target}_{int(time.time())}"
            }
        }
    
    async def _quarantine_file(self, action: ContainmentAction, incident: Incident) -> Dict[str, Any]:
        """Quarantine a file (simulated)"""
        # In a real implementation, this would integrate with endpoint security tools
        
        await asyncio.sleep(1)  # Simulate API call delay
        
        return {
            "success": True,
            "action": "quarantine",
            "target": action.target,
            "timestamp": datetime.now().isoformat(),
            "details": {
                "quarantine_location": "/quarantine/",
                "original_location": action.parameters.get("original_path")
            }
        }
    
    async def _monitor_asset(self, action: ContainmentAction, incident: Incident) -> Dict[str, Any]:
        """Monitor an asset (simulated)"""
        # In a real implementation, this would integrate with monitoring systems
        
        await asyncio.sleep(1)  # Simulate API call delay
        
        return {
            "success": True,
            "action": "monitor",
            "target": action.target,
            "timestamp": datetime.now().isoformat(),
            "details": {
                "monitoring_level": "enhanced",
                "duration": "24h"
            }
        }
    
    async def _escalate_incident(self, action: ContainmentAction, incident: Incident) -> Dict[str, Any]:
        """Escalate an incident (simulated)"""
        # In a real implementation, this would integrate with ticketing systems
        
        await asyncio.sleep(1)  # Simulate API call delay
        
        return {
            "success": True,
            "action": "escalate",
            "target": action.target,
            "timestamp": datetime.now().isoformat(),
            "details": {
                "escalation_level": "tier2",
                "assigned_to": "security_team",
                "ticket_id": f"SEC-{incident.incident_id}"
            }
        }

class CRASOARMCPAgent:
    """Enhanced CRA with SOAR and MCP integration"""
    
    def __init__(self, config_path: str = "config/cra_soar_mcp_config.json"):
        """Initialize the enhanced CRA SOAR MCP agent"""
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.mcp_client = MCPThreatIntelligenceClient(self.config["mcp"])
        self.soar_engine = SOARWorkflowEngine(self.config)
        
        # Initialize Google Cloud clients
        self.publisher = pubsub_v1.PublisherClient()
        self.subscriber = pubsub_v1.SubscriberClient()
        self.db = firestore.Client(project=self.config["project_id"])
        
        # Initialize Vertex AI
        vertexai.init(
            project=self.config["project_id"],
            location=self.config["location"]
        )
        self.llm_model = GenerativeModel(self.config["vertex_ai"]["model"])
        
        logger.info("CRA SOAR MCP Agent initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            # Return default configuration
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "project_id": "ai-driven-soc",
            "location": "us-central1",
            "mcp": {
                "enabled": True,
                "env": {
                    "VT_API_KEY": "",
                    "AIPDB_API_KEY": "",
                    "SHODAN_API_KEY": ""
                }
            },
            "vertex_ai": {
                "model": "gemini-2.0-flash-exp"
            },
            "pubsub": {
                "incidents_topic": "confirmed-incidents",
                "reports_topic": "incident-reports",
                "a2a_topic": "a2a-communication"
            },
            "playbooks": {
                "malware_containment": {
                    "name": "Malware Containment",
                    "description": "Contain malware infections",
                    "trigger_conditions": {
                        "severity": "high",
                        "threat_types": ["malware", "ransomware"]
                    },
                    "actions": [
                        {
                            "action_id": "isolate_host",
                            "action_type": "isolate",
                            "target": "${affected_hosts}",
                            "parameters": {},
                            "risk_level": "high",
                            "justification": "Isolate infected hosts to prevent spread",
                            "requires_approval": True
                        },
                        {
                            "action_id": "block_malware_ips",
                            "action_type": "block",
                            "target": "${malicious_ips}",
                            "parameters": {},
                            "risk_level": "medium",
                            "justification": "Block known malicious IPs",
                            "requires_approval": False
                        }
                    ],
                    "timeout_minutes": 30,
                    "retry_count": 2,
                    "enabled": True
                }
            }
        }
    
    async def process_incident(self, incident_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a confirmed security incident with MCP-enhanced intelligence"""
        incident_id = incident_data.get("alert_id", f"inc_{int(time.time())}")
        
        logger.info(f"Processing incident: {incident_id}")
        
        try:
            # Create incident object
            incident = Incident(
                incident_id=incident_id,
                alert_id=incident_data.get("alert_id", ""),
                severity=RiskLevel(incident_data.get("severity", "medium")),
                title=incident_data.get("title", "Security Incident"),
                description=incident_data.get("description", ""),
                affected_assets=incident_data.get("affected_assets", []),
                indicators=incident_data.get("indicators", []),
                threat_intelligence=[],
                containment_actions=[],
                created_at=datetime.now().isoformat()
            )
            
            # Enrich with threat intelligence using MCP
            if self.config["mcp"]["enabled"] and incident.indicators:
                async with self.mcp_client as mcp:
                    incident.threat_intelligence = await mcp.enrich_indicators(incident.indicators)
                    logger.info(f"Enriched {len(incident.threat_intelligence)} indicators with threat intelligence")
            
            # Select and execute appropriate playbook
            playbook = self.soar_engine.select_playbook(incident)
            
            if playbook:
                incident.playbook_id = playbook.playbook_id
                workflow_result = await self.soar_engine.execute_playbook(playbook, incident)
                
                # Update incident with containment actions
                incident.containment_actions = [
                    ContainmentAction(**action_data) 
                    for action_data in workflow_result.get("actions_completed", [])
                ]
                
                incident.status = workflow_result["status"]
                incident.updated_at = datetime.now().isoformat()
                
                logger.info(f"Playbook execution completed with status: {workflow_result['status']}")
            else:
                logger.warning(f"No suitable playbook found for incident {incident_id}")
                incident.status = "no_playbook"
            
            # Store incident in Firestore
            await self._store_incident(incident)
            
            # Generate incident report
            report = await self._generate_incident_report(incident, workflow_result if playbook else None)
            
            # Send report to reporting topic
            await self._send_incident_report(report)
            
            return {
                "incident_id": incident_id,
                "status": incident.status,
                "playbook_executed": playbook.name if playbook else None,
                "containment_actions": len(incident.containment_actions),
                "threat_intelligence_sources": len(set(
                    ti.sources for ti in incident.threat_intelligence
                )),
                "report_generated": True,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing incident {incident_id}: {e}")
            return {
                "incident_id": incident_id,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _store_incident(self, incident: Incident):
        """Store incident in Firestore"""
        try:
            incident_ref = self.db.collection("incidents").document(incident.incident_id)
            incident_ref.set(asdict(incident))
            logger.info(f"Incident {incident.incident_id} stored in Firestore")
        except Exception as e:
            logger.error(f"Error storing incident: {e}")
    
    async def _generate_incident_report(self, incident: Incident, workflow_result: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate comprehensive incident report using LLM"""
        try:
            # Prepare context for LLM
            context = {
                "incident": asdict(incident),
                "workflow_result": workflow_result,
                "threat_intelligence_summary": {
                    "total_indicators": len(incident.threat_intelligence),
                    "high_risk_indicators": len([ti for ti in incident.threat_intelligence if ti.risk_score > 70]),
                    "threat_types": list(set(
                        threat_type 
                        for ti in incident.threat_intelligence 
                        for threat_type in ti.threat_types
                    )),
                    "sources": list(set(
                        source 
                        for ti in incident.threat_intelligence 
                        for source in ti.sources
                    ))
                }
            }
            
            prompt = f"""
            Generate a comprehensive security incident report based on the following data:
            
            Incident Details:
            - ID: {incident.incident_id}
            - Severity: {incident.severity.value}
            - Title: {incident.title}
            - Description: {incident.description}
            - Affected Assets: {', '.join(incident.affected_assets)}
            
            Threat Intelligence:
            - Indicators Analyzed: {len(incident.threat_intelligence)}
            - High Risk Indicators: {len([ti for ti in incident.threat_intelligence if ti.risk_score > 70])}
            - Threat Types: {', '.join(set(threat_type for ti in incident.threat_intelligence for threat_type in ti.threat_types))}
            
            Response Actions:
            - Containment Actions Taken: {len(incident.containment_actions)}
            - Playbook Status: {workflow_result.get('status', 'N/A') if workflow_result else 'No playbook executed'}
            
            Please provide:
            1. Executive Summary
            2. Incident Analysis
            3. Threat Assessment
            4. Response Actions Taken
            5. Recommendations
            6. Lessons Learned
            """
            
            response = self.llm_model.generate_content(prompt)
            
            return {
                "incident_id": incident.incident_id,
                "report_type": "incident_response",
                "generated_at": datetime.now().isoformat(),
                "content": response.text,
                "metadata": {
                    "llm_model": self.config["vertex_ai"]["model"],
                    "threat_intelligence_enriched": len(incident.threat_intelligence) > 0,
                    "automated_response": workflow_result is not None
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating incident report: {e}")
            return {
                "incident_id": incident.incident_id,
                "report_type": "incident_response",
                "generated_at": datetime.now().isoformat(),
                "content": f"Error generating report: {str(e)}",
                "metadata": {"error": True}
            }
    
    async def _send_incident_report(self, report: Dict[str, Any]):
        """Send incident report to reporting topic"""
        try:
            topic_path = self.publisher.topic_path(
                self.config["project_id"],
                self.config["pubsub"]["reports_topic"]
            )
            
            message_data = json.dumps(report).encode("utf-8")
            future = self.publisher.publish(topic_path, message_data)
            future.result()  # Wait for publish to complete
            
            logger.info(f"Incident report sent for incident {report['incident_id']}")
            
        except Exception as e:
            logger.error(f"Error sending incident report: {e}")
    
    async def handle_a2a_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle A2A task requests"""
        try:
            task_type = task_data.get("task_type")
            
            if task_type == "process_incident":
                return await self.process_incident(task_data.get("incident_data", {}))
            elif task_type == "get_playbook_status":
                workflow_id = task_data.get("workflow_id")
                return self.soar_engine.running_workflows.get(workflow_id, {"error": "Workflow not found"})
            elif task_type == "approve_action":
                workflow_id = task_data.get("workflow_id")
                action_id = task_data.get("action_id")
                return await self._approve_action(workflow_id, action_id)
            else:
                return {"error": f"Unknown task type: {task_type}"}
                
        except Exception as e:
            logger.error(f"Error handling A2A task: {e}")
            return {"error": str(e)}

async def main():
    """Main function for testing the CRA SOAR MCP agent"""
    agent = CRASOARMCPAgent()
    
    # Test incident processing
    test_incident = {
        "alert_id": "test_incident_001",
        "severity": "high",
        "title": "Malware Detection",
        "description": "Suspicious executable detected on multiple hosts",
        "affected_assets": ["host-001", "host-002"],
        "indicators": ["192.168.1.100", "malware.example.com", "abc123def456"]
    }
    
    result = await agent.process_incident(test_incident)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main())


