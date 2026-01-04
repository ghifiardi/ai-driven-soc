#!/usr/bin/env python3
"""
ASGARD Orchestration Agent
===========================
Mimics Nextron Systems ASGARD management platform capabilities:
- Fleet-wide THOR scan deployment and orchestration
- Centralized IOC and YARA rule distribution
- Scheduled and on-demand threat hunting campaigns
- Real-time scan monitoring and result aggregation
- Integration with SIEM and ticketing systems
- Multi-tenant scan isolation

Integration with AI-Driven SOC:
- Orchestrates THOR agents across multiple endpoints
- Coordinates with VALHALLA for rule updates
- Sends aggregated findings to TAA for analysis
- Triggers CRA playbooks based on fleet-wide patterns
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict

# Google Cloud imports
from google.cloud import pubsub_v1, firestore, bigquery, compute_v1
import google.api_core.exceptions

# LangGraph for workflow orchestration
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CampaignStatus(Enum):
    """Threat hunting campaign status"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ScanPriority(Enum):
    """Scan priority levels"""
    CRITICAL = "critical"  # Execute immediately
    HIGH = "high"  # Execute within 1 hour
    MEDIUM = "medium"  # Execute within 24 hours
    LOW = "low"  # Execute during maintenance window
    SCHEDULED = "scheduled"  # Execute at specified time


class TargetSelectionMode(Enum):
    """How to select scan targets"""
    ALL = "all"  # All registered endpoints
    GROUP = "group"  # Specific endpoint group (e.g., "web_servers")
    LABEL = "label"  # By labels (e.g., "env:production")
    CUSTOM = "custom"  # Custom list of hostnames/IPs
    REGEX = "regex"  # Hostname pattern matching


@dataclass
class EndpointInfo:
    """Registered endpoint information"""
    endpoint_id: str
    hostname: str
    ip_address: str
    os_type: str  # linux, windows, macos
    os_version: str
    agent_version: str
    last_seen: str
    labels: Dict[str, str]
    groups: List[str]
    status: str  # online, offline, scanning
    capabilities: List[str]  # yara, ioc, sigma, etc.

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ThreatHuntingCampaign:
    """Threat hunting campaign definition"""
    campaign_id: str
    name: str
    description: str
    created_at: str
    created_by: str
    status: CampaignStatus
    priority: ScanPriority

    # Target selection
    target_selection_mode: TargetSelectionMode
    target_criteria: Dict[str, Any]  # Depends on selection mode

    # Scan configuration
    scan_types: List[str]  # filesystem, process, network, etc.
    yara_rule_sets: List[str]  # YARA rule names to use
    ioc_feeds: List[str]  # IOC feeds to include
    sigma_rules: List[str]  # Sigma rules for log analysis

    # Scheduling
    schedule_type: str  # immediate, scheduled, recurring
    scheduled_time: Optional[str]
    recurrence_pattern: Optional[str]  # cron expression

    # Execution settings
    max_parallel_scans: int
    timeout_minutes: int
    retry_on_failure: bool

    # Results
    total_targets: int
    scanned_count: int
    in_progress_count: int
    failed_count: int
    total_threats_found: int
    critical_threats_found: int

    # Timestamps
    started_at: Optional[str]
    completed_at: Optional[str]

    def to_dict(self) -> Dict:
        data = asdict(self)
        data['status'] = self.status.value
        data['priority'] = self.priority.value
        data['target_selection_mode'] = self.target_selection_mode.value
        return data


@dataclass
class ScanTask:
    """Individual scan task for an endpoint"""
    task_id: str
    campaign_id: str
    endpoint_id: str
    hostname: str
    status: str  # queued, running, completed, failed
    scan_config: Dict[str, Any]
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    error_message: Optional[str]
    result_summary: Optional[Dict[str, Any]]

    def to_dict(self) -> Dict:
        return asdict(self)


class ASGARDState(TypedDict):
    """LangGraph state for ASGARD orchestration workflow"""
    campaign: ThreatHuntingCampaign
    endpoints: List[EndpointInfo]
    scan_tasks: List[ScanTask]
    workflow_status: str
    errors: List[str]


class ASGARDOrchestrationAgent:
    """
    ASGARD-like orchestration platform for fleet-wide threat hunting

    Capabilities:
    - Central management console for THOR deployments
    - Fleet-wide scan campaigns with flexible targeting
    - Real-time scan monitoring and result aggregation
    - Scheduled and recurring threat hunts
    - IOC and YARA rule distribution to endpoints
    - Integration with cloud compute APIs (GCP, AWS, Azure)
    """

    def __init__(self, config_path: str = "config/asgard_config.json"):
        self.config = self._load_config(config_path)
        self.project_id = self.config.get("gcp_project_id", os.getenv("GCP_PROJECT_ID"))
        self.tenant_id = self.config.get("tenant_id", "default")

        # GCP clients
        self.publisher = pubsub_v1.PublisherClient()
        self.subscriber = pubsub_v1.SubscriberClient()
        self.firestore_client = firestore.Client(project=self.project_id)
        self.bq_client = bigquery.Client(project=self.project_id)

        # Optional: GCP Compute client for VM discovery
        try:
            self.compute_client = compute_v1.InstancesClient()
        except Exception as e:
            logger.warning(f"GCP Compute client not available: {e}")
            self.compute_client = None

        # Pub/Sub topics
        self.campaign_topic = self.config["pubsub_topics"]["campaigns"]
        self.scan_tasks_topic = self.config["pubsub_topics"]["scan_tasks"]
        self.scan_results_topic = self.config["pubsub_topics"]["scan_results"]

        # In-memory state (in production, use Firestore/Redis)
        self.registered_endpoints: Dict[str, EndpointInfo] = {}
        self.active_campaigns: Dict[str, ThreatHuntingCampaign] = {}
        self.scan_tasks: Dict[str, ScanTask] = {}

        # Load registered endpoints from Firestore
        self._load_endpoints()

        logger.info(f"ASGARD Orchestration Agent initialized for tenant {self.tenant_id}")
        logger.info(f"Registered endpoints: {len(self.registered_endpoints)}")

    def _load_config(self, config_path: str) -> Dict:
        """Load ASGARD configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self._default_config()

    def _default_config(self) -> Dict:
        """Default configuration"""
        import os
        return {
            "tenant_id": "default",
            "gcp_project_id": os.getenv("GCP_PROJECT_ID", "my-soc-project"),
            "pubsub_topics": {
                "campaigns": "asgard-campaigns",
                "scan_tasks": "asgard-scan-tasks",
                "scan_results": "thor-findings"
            },
            "orchestration": {
                "max_parallel_scans_per_campaign": 50,
                "default_scan_timeout_minutes": 120,
                "retry_failed_scans": True,
                "max_retries": 2
            },
            "endpoint_discovery": {
                "auto_discover_gcp_vms": True,
                "auto_discover_interval_hours": 1,
                "required_labels": {"soc_agent": "enabled"}
            },
            "integration": {
                "publish_to_taa": True,
                "publish_to_siem": True,
                "create_tickets_for_critical": True
            }
        }

    def _load_endpoints(self) -> None:
        """Load registered endpoints from Firestore"""
        try:
            endpoints_ref = self.firestore_client.collection('endpoints')
            docs = endpoints_ref.where('tenant_id', '==', self.tenant_id).stream()

            for doc in docs:
                data = doc.to_dict()
                endpoint = EndpointInfo(**data)
                self.registered_endpoints[endpoint.endpoint_id] = endpoint

            logger.info(f"Loaded {len(self.registered_endpoints)} registered endpoints")

        except Exception as e:
            logger.error(f"Failed to load endpoints: {e}")

    def register_endpoint(self, endpoint_info: EndpointInfo) -> bool:
        """
        Register a new endpoint with ASGARD

        Args:
            endpoint_info: Endpoint information

        Returns:
            True if successful
        """
        try:
            # Store in Firestore
            doc_ref = self.firestore_client.collection('endpoints').document(endpoint_info.endpoint_id)
            data = endpoint_info.to_dict()
            data['tenant_id'] = self.tenant_id
            doc_ref.set(data)

            # Update in-memory cache
            self.registered_endpoints[endpoint_info.endpoint_id] = endpoint_info

            logger.info(f"Registered endpoint: {endpoint_info.hostname} ({endpoint_info.endpoint_id})")
            return True

        except Exception as e:
            logger.error(f"Failed to register endpoint: {e}")
            return False

    def discover_endpoints_gcp(self, zone: str = None) -> List[EndpointInfo]:
        """
        Auto-discover GCP VM instances as potential endpoints

        Args:
            zone: GCP zone to search (None for all zones)

        Returns:
            List of discovered endpoints
        """
        if not self.compute_client:
            logger.warning("GCP Compute client not available")
            return []

        discovered = []

        try:
            # List all zones if none specified
            zones = [zone] if zone else self._list_gcp_zones()

            for z in zones:
                request = compute_v1.ListInstancesRequest(
                    project=self.project_id,
                    zone=z
                )

                instances = self.compute_client.list(request=request)

                for instance in instances:
                    # Check if instance has required labels
                    required_labels = self.config["endpoint_discovery"]["required_labels"]
                    if not all(instance.labels.get(k) == v for k, v in required_labels.items()):
                        continue

                    # Extract endpoint info
                    endpoint_id = instance.name
                    hostname = instance.name

                    # Get primary network interface IP
                    ip_address = ""
                    if instance.network_interfaces:
                        ip_address = instance.network_interfaces[0].network_i_p

                    # Determine OS type from metadata or machine image
                    os_type = "linux"  # Default
                    if instance.metadata and instance.metadata.items:
                        for item in instance.metadata.items:
                            if item.key == "os_type":
                                os_type = item.value

                    endpoint_info = EndpointInfo(
                        endpoint_id=endpoint_id,
                        hostname=hostname,
                        ip_address=ip_address,
                        os_type=os_type,
                        os_version="unknown",
                        agent_version="unknown",
                        last_seen=datetime.utcnow().isoformat(),
                        labels=dict(instance.labels) if instance.labels else {},
                        groups=[],
                        status="online" if instance.status == "RUNNING" else "offline",
                        capabilities=["yara", "ioc", "process", "network"]
                    )

                    discovered.append(endpoint_info)
                    self.register_endpoint(endpoint_info)

            logger.info(f"Discovered {len(discovered)} GCP endpoints")
            return discovered

        except Exception as e:
            logger.error(f"GCP endpoint discovery failed: {e}")
            return []

    def _list_gcp_zones(self) -> List[str]:
        """List all available GCP zones"""
        # Simplified - in production, query GCP API
        return ["us-central1-a", "us-central1-b", "us-east1-b"]

    def create_campaign(
        self,
        name: str,
        description: str,
        target_selection_mode: TargetSelectionMode,
        target_criteria: Dict[str, Any],
        scan_types: List[str],
        yara_rule_sets: List[str] = None,
        ioc_feeds: List[str] = None,
        priority: ScanPriority = ScanPriority.MEDIUM,
        schedule_type: str = "immediate",
        created_by: str = "asgard_api"
    ) -> ThreatHuntingCampaign:
        """
        Create a new threat hunting campaign

        Args:
            name: Campaign name
            description: Campaign description
            target_selection_mode: How to select targets
            target_criteria: Target selection criteria
            scan_types: Types of scans to perform
            yara_rule_sets: YARA rules to use
            ioc_feeds: IOC feeds to include
            priority: Campaign priority
            schedule_type: immediate, scheduled, recurring
            created_by: Creator identifier

        Returns:
            Created campaign
        """
        campaign_id = f"campaign_{uuid.uuid4().hex[:16]}"

        campaign = ThreatHuntingCampaign(
            campaign_id=campaign_id,
            name=name,
            description=description,
            created_at=datetime.utcnow().isoformat(),
            created_by=created_by,
            status=CampaignStatus.PENDING,
            priority=priority,
            target_selection_mode=target_selection_mode,
            target_criteria=target_criteria,
            scan_types=scan_types,
            yara_rule_sets=yara_rule_sets or [],
            ioc_feeds=ioc_feeds or ["valhalla", "threatfox"],
            sigma_rules=[],
            schedule_type=schedule_type,
            scheduled_time=None,
            recurrence_pattern=None,
            max_parallel_scans=self.config["orchestration"]["max_parallel_scans_per_campaign"],
            timeout_minutes=self.config["orchestration"]["default_scan_timeout_minutes"],
            retry_on_failure=self.config["orchestration"]["retry_failed_scans"],
            total_targets=0,
            scanned_count=0,
            in_progress_count=0,
            failed_count=0,
            total_threats_found=0,
            critical_threats_found=0,
            started_at=None,
            completed_at=None
        )

        # Store in Firestore
        try:
            doc_ref = self.firestore_client.collection('campaigns').document(campaign_id)
            data = campaign.to_dict()
            data['tenant_id'] = self.tenant_id
            doc_ref.set(data)

            self.active_campaigns[campaign_id] = campaign

            logger.info(f"Created campaign: {name} ({campaign_id})")

            # If immediate, start execution
            if schedule_type == "immediate":
                asyncio.create_task(self.execute_campaign(campaign_id))

            return campaign

        except Exception as e:
            logger.error(f"Failed to create campaign: {e}")
            raise

    def select_targets(
        self,
        selection_mode: TargetSelectionMode,
        criteria: Dict[str, Any]
    ) -> List[EndpointInfo]:
        """
        Select target endpoints based on criteria

        Args:
            selection_mode: Selection mode
            criteria: Selection criteria

        Returns:
            List of selected endpoints
        """
        targets = []

        if selection_mode == TargetSelectionMode.ALL:
            targets = list(self.registered_endpoints.values())

        elif selection_mode == TargetSelectionMode.GROUP:
            group_name = criteria.get("group_name")
            targets = [
                ep for ep in self.registered_endpoints.values()
                if group_name in ep.groups
            ]

        elif selection_mode == TargetSelectionMode.LABEL:
            required_labels = criteria.get("labels", {})
            targets = [
                ep for ep in self.registered_endpoints.values()
                if all(ep.labels.get(k) == v for k, v in required_labels.items())
            ]

        elif selection_mode == TargetSelectionMode.CUSTOM:
            hostnames = set(criteria.get("hostnames", []))
            targets = [
                ep for ep in self.registered_endpoints.values()
                if ep.hostname in hostnames
            ]

        elif selection_mode == TargetSelectionMode.REGEX:
            import re
            pattern = criteria.get("hostname_pattern", ".*")
            regex = re.compile(pattern)
            targets = [
                ep for ep in self.registered_endpoints.values()
                if regex.match(ep.hostname)
            ]

        # Filter by status (only online endpoints)
        targets = [ep for ep in targets if ep.status == "online"]

        logger.info(f"Selected {len(targets)} targets using mode {selection_mode.value}")
        return targets

    async def execute_campaign(self, campaign_id: str) -> None:
        """
        Execute a threat hunting campaign

        Args:
            campaign_id: Campaign ID to execute
        """
        try:
            # Get campaign
            campaign = self.active_campaigns.get(campaign_id)
            if not campaign:
                # Try loading from Firestore
                doc = self.firestore_client.collection('campaigns').document(campaign_id).get()
                if doc.exists:
                    data = doc.to_dict()
                    campaign = ThreatHuntingCampaign(**data)
                    self.active_campaigns[campaign_id] = campaign
                else:
                    logger.error(f"Campaign {campaign_id} not found")
                    return

            logger.info(f"Executing campaign: {campaign.name} ({campaign_id})")

            # Update status
            campaign.status = CampaignStatus.RUNNING
            campaign.started_at = datetime.utcnow().isoformat()
            self._update_campaign_firestore(campaign)

            # Select targets
            targets = self.select_targets(
                campaign.target_selection_mode,
                campaign.target_criteria
            )

            campaign.total_targets = len(targets)
            self._update_campaign_firestore(campaign)

            if not targets:
                logger.warning(f"No targets selected for campaign {campaign_id}")
                campaign.status = CampaignStatus.COMPLETED
                campaign.completed_at = datetime.utcnow().isoformat()
                self._update_campaign_firestore(campaign)
                return

            # Create scan tasks for each target
            scan_tasks = []
            for endpoint in targets:
                task = ScanTask(
                    task_id=f"task_{uuid.uuid4().hex[:16]}",
                    campaign_id=campaign_id,
                    endpoint_id=endpoint.endpoint_id,
                    hostname=endpoint.hostname,
                    status="queued",
                    scan_config={
                        "scan_types": campaign.scan_types,
                        "yara_rule_sets": campaign.yara_rule_sets,
                        "ioc_feeds": campaign.ioc_feeds,
                        "timeout_minutes": campaign.timeout_minutes
                    },
                    created_at=datetime.utcnow().isoformat(),
                    started_at=None,
                    completed_at=None,
                    error_message=None,
                    result_summary=None
                )
                scan_tasks.append(task)
                self.scan_tasks[task.task_id] = task

            # Publish scan tasks to Pub/Sub for THOR agents to consume
            await self._distribute_scan_tasks(scan_tasks)

            # Monitor campaign progress
            asyncio.create_task(self._monitor_campaign(campaign_id))

        except Exception as e:
            logger.error(f"Campaign execution failed: {e}")
            if campaign:
                campaign.status = CampaignStatus.FAILED
                campaign.completed_at = datetime.utcnow().isoformat()
                self._update_campaign_firestore(campaign)

    async def _distribute_scan_tasks(self, tasks: List[ScanTask]) -> None:
        """Distribute scan tasks to THOR agents via Pub/Sub"""
        try:
            topic_path = self.publisher.topic_path(self.project_id, self.scan_tasks_topic)

            for task in tasks:
                message_data = {
                    "task_id": task.task_id,
                    "campaign_id": task.campaign_id,
                    "endpoint_id": task.endpoint_id,
                    "hostname": task.hostname,
                    "scan_config": task.scan_config,
                    "tenant_id": self.tenant_id
                }

                future = self.publisher.publish(
                    topic_path,
                    json.dumps(message_data).encode('utf-8'),
                    task_id=task.task_id,
                    campaign_id=task.campaign_id
                )

                logger.info(f"Published scan task {task.task_id} for {task.hostname}")

        except Exception as e:
            logger.error(f"Failed to distribute scan tasks: {e}")

    async def _monitor_campaign(self, campaign_id: str) -> None:
        """Monitor campaign progress and update status"""
        campaign = self.active_campaigns.get(campaign_id)
        if not campaign:
            return

        logger.info(f"Monitoring campaign {campaign_id}")

        # Poll task status until all complete
        while campaign.status == CampaignStatus.RUNNING:
            await asyncio.sleep(30)  # Check every 30 seconds

            # Count task statuses
            tasks = [t for t in self.scan_tasks.values() if t.campaign_id == campaign_id]

            campaign.scanned_count = len([t for t in tasks if t.status == "completed"])
            campaign.in_progress_count = len([t for t in tasks if t.status == "running"])
            campaign.failed_count = len([t for t in tasks if t.status == "failed"])

            # Update campaign in Firestore
            self._update_campaign_firestore(campaign)

            # Check if all tasks complete
            if campaign.scanned_count + campaign.failed_count >= campaign.total_targets:
                campaign.status = CampaignStatus.COMPLETED
                campaign.completed_at = datetime.utcnow().isoformat()
                self._update_campaign_firestore(campaign)

                logger.info(f"Campaign {campaign_id} completed: "
                          f"{campaign.scanned_count} scanned, {campaign.failed_count} failed, "
                          f"{campaign.total_threats_found} threats found")

                # Generate campaign report
                self._generate_campaign_report(campaign)
                break

    def _update_campaign_firestore(self, campaign: ThreatHuntingCampaign) -> None:
        """Update campaign in Firestore"""
        try:
            doc_ref = self.firestore_client.collection('campaigns').document(campaign.campaign_id)
            doc_ref.update(campaign.to_dict())
        except Exception as e:
            logger.error(f"Failed to update campaign in Firestore: {e}")

    def process_scan_result(self, result_data: Dict[str, Any]) -> None:
        """
        Process scan result from THOR agent

        Args:
            result_data: Scan result data from Pub/Sub
        """
        try:
            task_id = result_data.get("task_id")
            campaign_id = result_data.get("campaign_id")

            if task_id not in self.scan_tasks:
                logger.warning(f"Unknown task {task_id}")
                return

            task = self.scan_tasks[task_id]
            task.status = "completed"
            task.completed_at = datetime.utcnow().isoformat()
            task.result_summary = {
                "total_threats": len(result_data.get("matches", [])),
                "critical_threats": len([m for m in result_data.get("matches", [])
                                        if m.get("severity") == "critical"]),
                "scan_duration": result_data.get("scan_duration"),
                "files_scanned": result_data.get("files_scanned", 0)
            }

            # Update campaign statistics
            campaign = self.active_campaigns.get(campaign_id)
            if campaign:
                campaign.total_threats_found += task.result_summary["total_threats"]
                campaign.critical_threats_found += task.result_summary["critical_threats"]
                self._update_campaign_firestore(campaign)

            logger.info(f"Processed result for task {task_id}: "
                       f"{task.result_summary['total_threats']} threats found")

        except Exception as e:
            logger.error(f"Failed to process scan result: {e}")

    def _generate_campaign_report(self, campaign: ThreatHuntingCampaign) -> None:
        """Generate comprehensive campaign report"""
        try:
            # Query all task results
            tasks = [t for t in self.scan_tasks.values() if t.campaign_id == campaign.campaign_id]

            report = {
                "campaign_id": campaign.campaign_id,
                "campaign_name": campaign.name,
                "description": campaign.description,
                "created_at": campaign.created_at,
                "completed_at": campaign.completed_at,
                "duration_minutes": self._calculate_duration(campaign.started_at, campaign.completed_at),
                "statistics": {
                    "total_targets": campaign.total_targets,
                    "successfully_scanned": campaign.scanned_count,
                    "failed_scans": campaign.failed_count,
                    "total_threats": campaign.total_threats_found,
                    "critical_threats": campaign.critical_threats_found
                },
                "tasks": [t.to_dict() for t in tasks],
                "top_threats": self._get_top_threats(campaign.campaign_id)
            }

            # Store report in Firestore
            doc_ref = self.firestore_client.collection('campaign_reports').document(campaign.campaign_id)
            doc_ref.set(report)

            # Store in BigQuery for analytics
            self._store_report_bigquery(report)

            logger.info(f"Generated report for campaign {campaign.campaign_id}")

        except Exception as e:
            logger.error(f"Failed to generate campaign report: {e}")

    def _calculate_duration(self, start: str, end: str) -> float:
        """Calculate duration in minutes"""
        try:
            start_dt = datetime.fromisoformat(start)
            end_dt = datetime.fromisoformat(end)
            return (end_dt - start_dt).total_seconds() / 60
        except:
            return 0

    def _get_top_threats(self, campaign_id: str, limit: int = 10) -> List[Dict]:
        """Get top threats from campaign"""
        # Query BigQuery for top threats
        # Simplified implementation
        return []

    def _store_report_bigquery(self, report: Dict) -> None:
        """Store campaign report in BigQuery"""
        try:
            table_id = f"{self.project_id}.soc_data.asgard_campaign_reports"

            rows_to_insert = [{
                "campaign_id": report["campaign_id"],
                "campaign_name": report["campaign_name"],
                "created_at": report["created_at"],
                "completed_at": report["completed_at"],
                "duration_minutes": report["duration_minutes"],
                "total_targets": report["statistics"]["total_targets"],
                "successfully_scanned": report["statistics"]["successfully_scanned"],
                "failed_scans": report["statistics"]["failed_scans"],
                "total_threats": report["statistics"]["total_threats"],
                "critical_threats": report["statistics"]["critical_threats"],
                "tenant_id": self.tenant_id
            }]

            errors = self.bq_client.insert_rows_json(table_id, rows_to_insert)

            if not errors:
                logger.info(f"Stored campaign report in BigQuery")

        except Exception as e:
            logger.error(f"Failed to store report in BigQuery: {e}")

    def get_campaign_status(self, campaign_id: str) -> Optional[Dict]:
        """Get current campaign status"""
        campaign = self.active_campaigns.get(campaign_id)
        if campaign:
            return {
                "campaign_id": campaign.campaign_id,
                "name": campaign.name,
                "status": campaign.status.value,
                "progress": {
                    "total": campaign.total_targets,
                    "scanned": campaign.scanned_count,
                    "in_progress": campaign.in_progress_count,
                    "failed": campaign.failed_count
                },
                "threats": {
                    "total": campaign.total_threats_found,
                    "critical": campaign.critical_threats_found
                }
            }
        return None


async def main():
    """Main entry point"""
    import os
    agent = ASGARDOrchestrationAgent()

    # Example: Create a threat hunting campaign
    campaign = agent.create_campaign(
        name="Ransomware Hunt Q4 2025",
        description="Hunt for ransomware indicators across production fleet",
        target_selection_mode=TargetSelectionMode.LABEL,
        target_criteria={"labels": {"env": "production"}},
        scan_types=["filesystem", "process", "network"],
        yara_rule_sets=["ransomware", "apt"],
        priority=ScanPriority.HIGH
    )

    print(f"Created campaign: {campaign.campaign_id}")

    # Monitor status
    await asyncio.sleep(5)
    status = agent.get_campaign_status(campaign.campaign_id)
    print(f"Campaign status: {json.dumps(status, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())
