"""
Containment and Response Agent (CRA) for AI-driven SOC

Core Responsibilities:
- Execution of automated containment actions for confirmed incidents
- Creation of tickets in IT service management systems
- Integration with security controls and firewall systems
- Notification of human analysts for incidents requiring manual intervention
- Tracking of response actions and outcomes
"""

import os
import json
import base64
import time
import logging
import requests
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from google.cloud import pubsub_v1, firestore
import googleapiclient.discovery
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ContainmentResponseAgent")

class RiskLevel(Enum):
    """Risk levels for containment actions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class ContainmentResponseAgent:
    """Containment and Response Agent for security incidents"""
    
    def __init__(self, config_path: str = "config/cra_config.json"):
        """Initialize the Containment and Response Agent
        
        Args:
            config_path: Path to the agent configuration file
        """
        self.config = self._load_config(config_path)
        
        # Initialize publishers and subscribers
        self.publisher = pubsub_v1.PublisherClient()
        self.subscriber = pubsub_v1.SubscriberClient()
        
        # Topic for reporting
        self.reporting_topic_path = self.publisher.topic_path(
            self.config["project_id"], 
            self.config["reporting_topic"]
        )
        
        # Firestore for state management
        self.db = firestore.Client(project=self.config["project_id"])
        
        # Initialize API clients
        self._init_api_clients()
        
        logger.info("Containment and Response Agent initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load the agent configuration from a JSON file
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Dict containing the configuration
        """
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            # Use default configuration if file not found
            return {
                "project_id": "ai-driven-soc",
                "incident_subscription": "confirmed-incidents-sub",
                "reporting_topic": "incident-reports",
                "approval_required": {
                    "high": True,
                    "medium": False,
                    "low": False
                },
                "ticket_system": {
                    "type": "jira",  # or "servicenow"
                    "url": "https://example.atlassian.net",
                    "project_key": "SEC",
                    "issue_type": "Security Incident",
                    "api_token": os.environ.get("JIRA_API_TOKEN", ""),
                    "username": os.environ.get("JIRA_USERNAME", "")
                },
                "firewall_api": {
                    "type": "gcp",  # or "aws", "azure", "palo_alto", etc.
                    "network": "default",
                    "region": "us-central1"
                },
                # Playbook definitions for different attack types
                "playbooks": {
                    "default": [
                        {"action": "block_ip", "parameters": {"ip_field": "src_ip"}, "risk_level": "medium"},
                        {"action": "create_ticket", "parameters": {}, "risk_level": "low"}
                    ],
                    "brute_force": [
                        {"action": "block_ip", "parameters": {"ip_field": "src_ip"}, "risk_level": "medium"},
                        {"action": "reset_credentials", "parameters": {"user_field": "target_user"}, "risk_level": "high"},
                        {"action": "create_ticket", "parameters": {}, "risk_level": "low"}
                    ],
                    "data_exfiltration": [
                        {"action": "block_ip", "parameters": {"ip_field": "dst_ip"}, "risk_level": "medium"},
                        {"action": "isolate_host", "parameters": {"host_field": "src_hostname"}, "risk_level": "high"},
                        {"action": "create_ticket", "parameters": {"priority": "high"}, "risk_level": "low"}
                    ]
                }
            }
    
    def _init_api_clients(self):
        """Initialize API clients for external systems"""
        # Initialize firewall API client based on type
        firewall_type = self.config["firewall_api"]["type"]
        if firewall_type == "gcp":
            self.firewall_client = googleapiclient.discovery.build('compute', 'v1')
        else:
            self.firewall_client = None
            logger.warning(f"Firewall type {firewall_type} not implemented yet")
        
        # Initialize ticket system client
        ticket_type = self.config["ticket_system"]["type"]
        self.ticket_client = None  # Will be initialized on demand
    
    def process_incident(self, incident: Dict) -> Dict:
        """Process a confirmed security incident
        
        Args:
            incident: Confirmed incident data from the Triage Agent
            
        Returns:
            Dict containing response results
        """
        incident_id = incident.get("alert_id", "unknown")
        logger.info(f"Processing incident: {incident_id}")
        
        # Create an incident record in Firestore
        incident_ref = self._create_incident_record(incident)
        
        # Select the appropriate playbook
        playbook = self._select_playbook(incident)
        
        # Execute the playbook
        execution_results = self._execute_playbook(playbook, incident, incident_ref)
        
        # Prepare response results
        response_results = {
            "incident_id": incident_id,
            "timestamp": datetime.now().isoformat(),
            "incident": incident,
            "playbook_name": execution_results.get("playbook_name", "default"),
            "actions_executed": execution_results.get("actions_executed", []),
            "actions_pending_approval": execution_results.get("actions_pending_approval", []),
            "ticket_id": execution_results.get("ticket_id"),
            "ticket_url": execution_results.get("ticket_url"),
            "containment_status": execution_results.get("containment_status", "partial")
        }
        
        # Send to reporting topic
        self._send_to_reporting(response_results)
        
        return response_results
    
    def _create_incident_record(self, incident: Dict) -> firestore.DocumentReference:
        """Create an incident record in Firestore
        
        Args:
            incident: Incident data
            
        Returns:
            Firestore document reference
        """
        incident_id = incident.get("alert_id", f"INC-{int(time.time())}")
        incident_data = {
            "incident_id": incident_id,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "incident_data": incident,
            "status": "new",
            "actions": [],
            "ticket_id": None,
            "ticket_url": None,
            "containment_status": "pending"
        }
        
        # Create or update the incident record
        incident_ref = self.db.collection("incidents").document(incident_id)
        incident_ref.set(incident_data)
        
        return incident_ref
    
    def _select_playbook(self, incident: Dict) -> Tuple[str, List[Dict]]:
        """Select the appropriate playbook for the incident
        
        Args:
            incident: Incident data
            
        Returns:
            Tuple of (playbook_name, playbook_actions)
        """
        # Extract the attack type from the incident
        analysis = incident.get("analysis", {})
        attack_type = analysis.get("attack_type", "").lower().replace(" ", "_")
        
        # Find a matching playbook or use default
        if attack_type in self.config["playbooks"]:
            playbook_name = attack_type
            playbook = self.config["playbooks"][attack_type]
        else:
            playbook_name = "default"
            playbook = self.config["playbooks"]["default"]
        
        logger.info(f"Selected playbook: {playbook_name}")
        return (playbook_name, playbook)
    
    def _execute_playbook(self, playbook_tuple: Tuple[str, List[Dict]], incident: Dict, 
                         incident_ref: firestore.DocumentReference) -> Dict:
        """Execute a playbook for an incident
        
        Args:
            playbook_tuple: Tuple of (playbook_name, playbook_actions)
            incident: Incident data
            incident_ref: Firestore reference to the incident record
            
        Returns:
            Dict containing execution results
        """
        playbook_name, playbook = playbook_tuple
        incident_id = incident.get("alert_id", "unknown")
        severity = incident.get("severity", "low")
        
        execution_results = {
            "playbook_name": playbook_name,
            "actions_executed": [],
            "actions_pending_approval": [],
            "ticket_id": None,
            "ticket_url": None,
            "containment_status": "partial"
        }
        
        # Update incident record with playbook info
        incident_ref.update({
            "playbook": playbook_name,
            "updated_at": datetime.now()
        })
        
        # Execute each action in the playbook
        for action in playbook:
            action_name = action.get("action")
            action_params = action.get("parameters", {})
            risk_level = action.get("risk_level", "medium")
            
            # Check if approval is required
            requires_approval = self._requires_approval(risk_level, severity)
            
            if requires_approval:
                # Add to pending approval list
                execution_results["actions_pending_approval"].append({
                    "action": action_name,
                    "parameters": action_params,
                    "risk_level": risk_level
                })
                
                # Create approval request
                self._create_approval_request(incident_id, action_name, action_params, risk_level, incident_ref)
                
                logger.info(f"Action {action_name} requires approval, request created")
                continue
            
            # Execute the action
            try:
                action_result = self._execute_action(action_name, action_params, incident)
                
                # Record the action
                action_record = {
                    "action": action_name,
                    "parameters": action_params,
                    "risk_level": risk_level,
                    "timestamp": datetime.now().isoformat(),
                    "status": "completed",
                    "result": action_result
                }
                
                # Update incident record
                incident_ref.update({
                    "actions": firestore.ArrayUnion([action_record]),
                    "updated_at": datetime.now()
                })
                
                # Add to executed actions list
                execution_results["actions_executed"].append(action_record)
                
                # Update ticket ID and URL if available
                if action_name == "create_ticket" and "ticket_id" in action_result:
                    execution_results["ticket_id"] = action_result["ticket_id"]
                    execution_results["ticket_url"] = action_result.get("ticket_url")
                    
                    # Update incident record
                    incident_ref.update({
                        "ticket_id": action_result["ticket_id"],
                        "ticket_url": action_result.get("ticket_url"),
                        "updated_at": datetime.now()
                    })
            
            except Exception as e:
                logger.error(f"Error executing action {action_name}: {e}")
                
                # Record the failed action
                action_record = {
                    "action": action_name,
                    "parameters": action_params,
                    "risk_level": risk_level,
                    "timestamp": datetime.now().isoformat(),
                    "status": "failed",
                    "error": str(e)
                }
                
                # Update incident record
                incident_ref.update({
                    "actions": firestore.ArrayUnion([action_record]),
                    "updated_at": datetime.now()
                })
        
        # Determine containment status
        if len(execution_results["actions_executed"]) > 0 and len(execution_results["actions_pending_approval"]) == 0:
            containment_status = "complete"
        elif len(execution_results["actions_executed"]) > 0:
            containment_status = "partial"
        else:
            containment_status = "pending"
        
        execution_results["containment_status"] = containment_status
        
        # Update incident record with final status
        incident_ref.update({
            "containment_status": containment_status,
            "status": "contained" if containment_status in ["complete", "partial"] else "new",
            "updated_at": datetime.now()
        })
        
        return execution_results
    
    def _requires_approval(self, risk_level: str, severity: str) -> bool:
        """Check if an action requires approval based on risk level and incident severity
        
        Args:
            risk_level: Risk level of the action
            severity: Severity of the incident
            
        Returns:
            True if approval is required, False otherwise
        """
        # Get the approval configuration
        approval_config = self.config["approval_required"]
        
        # High-risk actions always require approval
        if risk_level == "high":
            return approval_config.get("high", True)
        
        # For medium-risk actions, it depends on severity
        if risk_level == "medium":
            if severity == "high":
                # Medium-risk actions on high-severity incidents might not need approval
                return approval_config.get("medium", True)
            else:
                # On medium/low severity, follow the config
                return approval_config.get("medium", True)
        
        # For low-risk actions
        return approval_config.get("low", False)
    
    def _create_approval_request(self, incident_id: str, action_name: str, 
                               action_params: Dict, risk_level: str,
                               incident_ref: firestore.DocumentReference) -> None:
        """Create an approval request for a high-risk action
        
        Args:
            incident_id: Incident ID
            action_name: Name of the action
            action_params: Parameters for the action
            risk_level: Risk level of the action
            incident_ref: Firestore reference to the incident
        """
        approval_id = f"APR-{incident_id}-{action_name}-{int(time.time())}"
        
        approval_request = {
            "approval_id": approval_id,
            "incident_id": incident_id,
            "action": action_name,
            "parameters": action_params,
            "risk_level": risk_level,
            "created_at": datetime.now(),
            "status": "pending",
            "approver": None,
            "approval_time": None
        }
        
        # Store the approval request
        self.db.collection("approval_requests").document(approval_id).set(approval_request)
        
        # Add the pending approval to the incident record
        incident_ref.update({
            "pending_approvals": firestore.ArrayUnion([approval_id]),
            "updated_at": datetime.now()
        })
        
        # TODO: Send notification to approvers (email, SMS, Slack, etc.)
        logger.info(f"Approval request created: {approval_id}")
    
    def _execute_action(self, action_name: str, parameters: Dict, incident: Dict) -> Dict:
        """Execute a containment or response action
        
        Args:
            action_name: Name of the action to execute
            parameters: Parameters for the action
            incident: Incident data
            
        Returns:
            Dict containing the result of the action
        """
        # Map of action names to handler methods
        action_handlers = {
            "block_ip": self._action_block_ip,
            "create_ticket": self._action_create_ticket,
            "isolate_host": self._action_isolate_host,
            "reset_credentials": self._action_reset_credentials
        }
        
        # Check if the action is supported
        if action_name not in action_handlers:
            raise ValueError(f"Unsupported action: {action_name}")
        
        # Execute the handler
        handler = action_handlers[action_name]
        result = handler(parameters, incident)
        
        logger.info(f"Executed action: {action_name}")
        return result
    
    def _action_block_ip(self, parameters: Dict, incident: Dict) -> Dict:
        """Block an IP address in the firewall
        
        Args:
            parameters: Action parameters (ip_field: field in incident containing the IP)
            incident: Incident data
            
        Returns:
            Result of the action
        """
        # Extract the IP to block
        ip_field = parameters.get("ip_field", "src_ip")
        ip_to_block = None
        
        if "log_data" in incident and ip_field in incident["log_data"]:
            ip_to_block = incident["log_data"][ip_field]
        elif ip_field in incident:
            ip_to_block = incident[ip_field]
        
        if not ip_to_block:
            raise ValueError(f"IP address not found in incident using field: {ip_field}")
        
        # Block the IP using the appropriate firewall API
        firewall_type = self.config["firewall_api"]["type"]
        
        if firewall_type == "gcp":
            result = self._block_ip_gcp(ip_to_block)
        else:
            raise ValueError(f"Unsupported firewall type: {firewall_type}")
        
        return {
            "ip_blocked": ip_to_block,
            "firewall_type": firewall_type,
            "rule_id": result.get("rule_id")
        }
    
    def _block_ip_gcp(self, ip_address: str) -> Dict:
        """Block an IP address using GCP Firewall
        
        Args:
            ip_address: IP address to block
            
        Returns:
            Result of the firewall rule creation
        """
        # Create a firewall rule to block the IP
        project = self.config["project_id"]
        network = self.config["firewall_api"]["network"]
        rule_name = f"block-ip-{ip_address.replace('.', '-')}-{int(time.time())}"
        
        firewall_rule = {
            "name": rule_name,
            "network": f"projects/{project}/global/networks/{network}",
            "direction": "INGRESS",
            "priority": 1000,
            "sourceRanges": [f"{ip_address}/32"],
            "denied": [{"IPProtocol": "all"}],
            "description": f"Block malicious IP detected in incident"
        }
        
        operation = self.firewall_client.firewalls().insert(
            project=project,
            body=firewall_rule
        ).execute()
        
        # This is async, so we just return the operation ID
        return {
            "rule_id": rule_name,
            "operation_id": operation["id"]
        }
    
    def _action_create_ticket(self, parameters: Dict, incident: Dict) -> Dict:
        """Create a ticket in the ticket system
        
        Args:
            parameters: Action parameters
            incident: Incident data
            
        Returns:
            Result of the ticket creation
        """
        ticket_system = self.config["ticket_system"]["type"]
        
        if ticket_system == "jira":
            result = self._create_ticket_jira(parameters, incident)
        elif ticket_system == "servicenow":
            result = self._create_ticket_servicenow(parameters, incident)
        else:
            raise ValueError(f"Unsupported ticket system: {ticket_system}")
        
        return result
    
    def _create_ticket_jira(self, parameters: Dict, incident: Dict) -> Dict:
        """Create a ticket in Jira
        
        Args:
            parameters: Action parameters
            incident: Incident data
            
        Returns:
            Result of the ticket creation
        """
        # Initialize Jira client if not already
        if not self.ticket_client:
            from jira import JIRA
            
            jira_config = self.config["ticket_system"]
            self.ticket_client = JIRA(
                server=jira_config["url"],
                basic_auth=(jira_config["username"], jira_config["api_token"])
            )
        
        # Prepare ticket data
        incident_id = incident.get("alert_id", "unknown")
        severity = incident.get("severity", "low")
        analysis = incident.get("analysis", {})
        
        # Map severity to priority
        priority_map = {
            "high": "High",
            "medium": "Medium",
            "low": "Low"
        }
        
        # Override priority if specified in parameters
        priority = parameters.get("priority", priority_map.get(severity, "Medium"))
        
        # Create the ticket
        issue_dict = {
            "project": {"key": self.config["ticket_system"]["project_key"]},
            "summary": f"Security Incident: {analysis.get('attack_type', 'Unknown')} - {incident_id}",
            "description": self._format_ticket_description(incident),
            "issuetype": {"name": self.config["ticket_system"]["issue_type"]},
            "priority": {"name": priority}
        }
        
        new_issue = self.ticket_client.create_issue(fields=issue_dict)
        
        return {
            "ticket_id": new_issue.key,
            "ticket_url": f"{self.config['ticket_system']['url']}/browse/{new_issue.key}"
        }
    
    def _create_ticket_servicenow(self, parameters: Dict, incident: Dict) -> Dict:
        """Create a ticket in ServiceNow
        
        Args:
            parameters: Action parameters
            incident: Incident data
            
        Returns:
            Result of the ticket creation
        """
        # ServiceNow implementation would go here
        # For now, this is a placeholder
        logger.warning("ServiceNow ticket creation not implemented yet")
        
        return {
            "ticket_id": f"INC{int(time.time())}",
            "ticket_url": f"{self.config['ticket_system']['url']}/incident.do?sysparm_id=INC{int(time.time())}"
        }
    
    def _format_ticket_description(self, incident: Dict) -> str:
        """Format the incident data for a ticket description
        
        Args:
            incident: Incident data
            
        Returns:
            Formatted description
        """
        analysis = incident.get("analysis", {})
        
        description = f"""
## Security Incident

**Incident ID:** {incident.get('alert_id', 'Unknown')}
**Severity:** {incident.get('severity', 'Unknown')}
**Attack Type:** {analysis.get('attack_type', 'Unknown')}
**Confidence:** {incident.get('confidence', 0.0):.2f}
**Detection Time:** {incident.get('timestamp', 'Unknown')}

### Summary
{analysis.get('analysis_summary', 'No summary available')}

### Indicators of Compromise
{self._format_list(analysis.get('indicators', []))}

### Recommended Actions
{self._format_list(analysis.get('recommended_actions', []))}

### Technical Details
```
{json.dumps(incident.get('log_data', {}), indent=2)}
```
"""
        return description
    
    def _format_list(self, items: List[str]) -> str:
        """Format a list of items as Markdown bullet points
        
        Args:
            items: List of items
            
        Returns:
            Formatted bullet points
        """
        if not items:
            return "None"
        
        return "\n".join([f"- {item}" for item in items])
    
    def _action_isolate_host(self, parameters: Dict, incident: Dict) -> Dict:
        """Isolate a host from the network
        
        Args:
            parameters: Action parameters
            incident: Incident data
            
        Returns:
            Result of the action
        """
        # Extract the host to isolate
        host_field = parameters.get("host_field", "src_hostname")
        host_to_isolate = None
        
        if "log_data" in incident and host_field in incident["log_data"]:
            host_to_isolate = incident["log_data"][host_field]
        elif host_field in incident:
            host_to_isolate = incident[host_field]
        
        if not host_to_isolate:
            raise ValueError(f"Host not found in incident using field: {host_field}")
        
        # This would normally involve calling an EDR API or similar
        # For now, just log it as a placeholder
        logger.info(f"Would isolate host: {host_to_isolate}")
        
        return {
            "host_isolated": host_to_isolate,
            "status": "simulated"
        }
    
    def _action_reset_credentials(self, parameters: Dict, incident: Dict) -> Dict:
        """Reset credentials for a user
        
        Args:
            parameters: Action parameters
            incident: Incident data
            
        Returns:
            Result of the action
        """
        # Extract the user to reset
        user_field = parameters.get("user_field", "target_user")
        user_to_reset = None
        
        if "log_data" in incident and user_field in incident["log_data"]:
            user_to_reset = incident["log_data"][user_field]
        elif user_field in incident:
            user_to_reset = incident[user_field]
        
        if not user_to_reset:
            raise ValueError(f"User not found in incident using field: {user_field}")
        
        # This would normally involve calling an IdP API or similar
        # For now, just log it as a placeholder
        logger.info(f"Would reset credentials for user: {user_to_reset}")
        
        return {
            "user": user_to_reset,
            "status": "simulated"
        }
    
    def _send_to_reporting(self, response_results: Dict) -> None:
        """Send response results to the reporting topic
        
        Args:
            response_results: Results of incident response
        """
        try:
            message_data = json.dumps(response_results).encode("utf-8")
            future = self.publisher.publish(self.reporting_topic_path, message_data)
            message_id = future.result()
            logger.info(f"Report published with message ID: {message_id}")
        except Exception as e:
            logger.error(f"Error sending to reporting: {e}")
    
    def process_approval(self, approval_id: str, approved: bool, approver: str) -> Dict:
        """Process an approval decision
        
        Args:
            approval_id: ID of the approval request
            approved: Whether the action was approved
            approver: ID or name of the approver
            
        Returns:
            Result of the approval processing
        """
        # Get the approval request
        approval_doc = self.db.collection("approval_requests").document(approval_id).get()
        
        if not approval_doc.exists:
            raise ValueError(f"Approval request not found: {approval_id}")
        
        approval_data = approval_doc.to_dict()
        
        # Update the approval request
        self.db.collection("approval_requests").document(approval_id).update({
            "status": "approved" if approved else "rejected",
            "approver": approver,
            "approval_time": datetime.now()
        })
        
        # If approved, execute the action
        if approved:
            incident_id = approval_data.get("incident_id")
            incident_doc = self.db.collection("incidents").document(incident_id).get()
            
            if not incident_doc.exists:
                logger.error(f"Incident not found: {incident_id}")
                return {"status": "error", "message": f"Incident not found: {incident_id}"}
            
            incident_data = incident_doc.to_dict()
            
            try:
                # Execute the action
                action_name = approval_data.get("action")
                action_params = approval_data.get("parameters", {})
                
                action_result = self._execute_action(action_name, action_params, incident_data.get("incident_data", {}))
                
                # Record the action
                action_record = {
                    "action": action_name,
                    "parameters": action_params,
                    "risk_level": approval_data.get("risk_level", "high"),
                    "timestamp": datetime.now().isoformat(),
                    "status": "completed",
                    "result": action_result,
                    "approval_id": approval_id,
                    "approver": approver
                }
                
                # Update incident record
                incident_ref = self.db.collection("incidents").document(incident_id)
                incident_ref.update({
                    "actions": firestore.ArrayUnion([action_record]),
                    "pending_approvals": firestore.ArrayRemove([approval_id]),
                    "updated_at": datetime.now()
                })
                
                # Check if all pending approvals are done
                # If so, update containment status
                updated_incident = incident_ref.get().to_dict()
                if not updated_incident.get("pending_approvals"):
                    incident_ref.update({
                        "containment_status": "complete",
                        "status": "contained",
                        "updated_at": datetime.now()
                    })
                
                logger.info(f"Approved action {action_name} executed for incident {incident_id}")
                
                # If ticket creation, update incident with ticket info
                if action_name == "create_ticket" and "ticket_id" in action_result:
                    incident_ref.update({
                        "ticket_id": action_result["ticket_id"],
                        "ticket_url": action_result.get("ticket_url"),
                        "updated_at": datetime.now()
                    })
                
                return {
                    "status": "success",
                    "action": action_name,
                    "result": action_result
                }
            
            except Exception as e:
                logger.error(f"Error executing approved action: {e}")
                
                # Record the failed action
                action_record = {
                    "action": approval_data.get("action"),
                    "parameters": approval_data.get("parameters", {}),
                    "risk_level": approval_data.get("risk_level", "high"),
                    "timestamp": datetime.now().isoformat(),
                    "status": "failed",
                    "error": str(e),
                    "approval_id": approval_id,
                    "approver": approver
                }
                
                # Update incident record
                self.db.collection("incidents").document(incident_id).update({
                    "actions": firestore.ArrayUnion([action_record]),
                    "pending_approvals": firestore.ArrayRemove([approval_id]),
                    "updated_at": datetime.now()
                })
                
                return {
                    "status": "error",
                    "message": str(e)
                }
        else:
            # If rejected, update the incident record
            incident_id = approval_data.get("incident_id")
            
            # Record the rejected action
            action_record = {
                "action": approval_data.get("action"),
                "parameters": approval_data.get("parameters", {}),
                "risk_level": approval_data.get("risk_level", "high"),
                "timestamp": datetime.now().isoformat(),
                "status": "rejected",
                "approval_id": approval_id,
                "approver": approver
            }
            
            # Update incident record
            self.db.collection("incidents").document(incident_id).update({
                "actions": firestore.ArrayUnion([action_record]),
                "pending_approvals": firestore.ArrayRemove([approval_id]),
                "updated_at": datetime.now()
            })
            
            logger.info(f"Action {approval_data.get('action')} rejected for incident {incident_id}")
            
            return {
                "status": "rejected",
                "message": "Action was rejected by approver"
            }
    
    def listen_for_incidents(self):
        """Listen for incidents from the subscription"""
        subscription_path = self.subscriber.subscription_path(
            self.config["project_id"],
            self.config["incident_subscription"]
        )
        
        def callback(message):
            try:
                # Decode the message data
                incident_data = json.loads(message.data.decode("utf-8"))
                
                # Process the incident
                self.process_incident(incident_data)
                
                # Acknowledge the message
                message.ack()
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                # Don't acknowledge to allow retry
        
        streaming_pull_future = self.subscriber.subscribe(
            subscription_path, callback=callback
        )
        logger.info(f"Listening for messages on {subscription_path}")
        
        try:
            # Keep the main thread alive
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            streaming_pull_future.cancel()
            logger.info("Listener stopped")

# Cloud Function entry point for processing incidents
def process_incident_function(event, context):
    """Cloud Function entry point for processing incidents
    
    Args:
        event: Pub/Sub event
        context: Event context
    """
    agent = ContainmentResponseAgent()
    
    # Decode the incident data
    incident_data = json.loads(base64.b64decode(event["data"]).decode("utf-8"))
    
    # Process the incident
    result = agent.process_incident(incident_data)
    
    return "Incident processed successfully"

# Cloud Function entry point for processing approvals
def process_approval_function(request):
    """Cloud Function entry point for processing approvals
    
    Args:
        request: HTTP request
    """
    request_json = request.get_json()
    
    if not request_json:
        return {"status": "error", "message": "No JSON data received"}
    
    approval_id = request_json.get("approval_id")
    approved = request_json.get("approved", False)
    approver = request_json.get("approver", "unknown")
    
    if not approval_id:
        return {"status": "error", "message": "No approval_id provided"}
    
    agent = ContainmentResponseAgent()
    
    # Process the approval
    result = agent.process_approval(approval_id, approved, approver)
    
    return result

if __name__ == "__main__":
    # For testing
    agent = ContainmentResponseAgent()
    
    # Test incident
    test_incident = {
        "alert_id": "ADA-20230503123456-7890",
        "timestamp": "2023-05-03T12:34:56Z",
        "severity": "high",
        "confidence": 0.85,
        "analysis": {
            "is_true_positive": True,
            "severity": "high",
            "confidence": 0.85,
            "attack_type": "brute_force",
            "indicators": ["Repeated failed login attempts", "Unusual source IP"],
            "recommended_actions": ["Block source IP", "Reset affected user credentials"],
            "analysis_summary": "Multiple failed login attempts from unusual IP address"
        },
        "log_data": {
            "log_id": "log-123456",
            "timestamp": "2023-05-03T12:34:56Z",
            "src_ip": "10.0.0.1",
            "dst_ip": "203.0.113.1",
            "target_user": "admin",
            "src_hostname": "workstation-123"
        }
    }
    
    # Process the test incident
    result = agent.process_incident(test_incident)
    
    # Print the result
    print(json.dumps(result, indent=2))
