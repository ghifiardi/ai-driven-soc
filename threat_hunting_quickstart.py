#!/usr/bin/env python3
"""
Threat Hunting Quick Start Script
==================================
Demonstrates integration of THOR, ASGARD, and VALHALLA with existing SOC agents

This script:
1. Initializes VALHALLA and updates threat intelligence feeds
2. Configures ASGARD for campaign management
3. Launches a sample threat hunting campaign
4. Monitors results and demonstrates TAA/CRA integration
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Dict, Any

# Import threat hunting agents
from valhalla_feed_manager import VALHALLAFeedManager
from asgard_orchestration_agent import (
    ASGARDOrchestrationAgent,
    ThreatHuntingCampaign,
    ScanPriority,
    TargetSelectionMode,
    EndpointInfo
)
from thor_endpoint_agent import THOREndpointAgent, ScanType


class ThreatHuntingOrchestrator:
    """
    Orchestrates the full threat hunting workflow
    Integrates THOR/ASGARD/VALHALLA with TAA/CRA/CLA
    """

    def __init__(self, project_id: str):
        self.project_id = project_id

        print("🚀 Initializing Threat Hunting Platform...")

        # Initialize components
        self.valhalla = VALHALLAFeedManager()
        self.asgard = ASGARDOrchestrationAgent()

        print("✅ VALHALLA Feed Manager initialized")
        print("✅ ASGARD Orchestration Agent initialized")

    async def setup_threat_intelligence(self) -> Dict[str, int]:
        """
        Step 1: Update threat intelligence feeds via VALHALLA
        """
        print("\n📡 Step 1: Updating Threat Intelligence Feeds...")

        # Update all feeds
        stats = self.valhalla.update_all_feeds()

        print(f"   📥 ThreatFox IOCs: {stats.get('threatfox', 0)}")
        print(f"   📥 MalwareBazaar Hashes: {stats.get('malwarebazaar', 0)}")
        print(f"   📥 Emerging Threats YARA: {stats.get('emergingthreats_yara', 0)}")

        # Export IOCs for THOR agents
        print("\n   📤 Exporting IOCs for THOR agents...")
        ioc_export = self.valhalla.export_iocs_for_thor()

        print(f"   ✅ Exported {ioc_export['metadata']['total_iocs']} IOCs")
        print(f"      - IP IOCs: {len(ioc_export['ips'])}")
        print(f"      - Domain IOCs: {len(ioc_export['domains'])}")
        print(f"      - Hash IOCs: {len(ioc_export['file_hashes'])}")

        return stats

    async def register_demo_endpoints(self) -> int:
        """
        Step 2: Register demo endpoints (in production, auto-discover from cloud)
        """
        print("\n🖥️  Step 2: Registering Endpoints...")

        # Demo endpoints
        demo_endpoints = [
            {
                "hostname": "web-server-01",
                "ip": "10.0.1.10",
                "os": "linux",
                "labels": {"env": "production", "role": "webserver"}
            },
            {
                "hostname": "web-server-02",
                "ip": "10.0.1.11",
                "os": "linux",
                "labels": {"env": "production", "role": "webserver"}
            },
            {
                "hostname": "db-server-01",
                "ip": "10.0.2.10",
                "os": "linux",
                "labels": {"env": "production", "role": "database"}
            },
            {
                "hostname": "app-server-01",
                "ip": "10.0.3.10",
                "os": "linux",
                "labels": {"env": "production", "role": "application"}
            },
            {
                "hostname": "dev-server-01",
                "ip": "10.0.4.10",
                "os": "linux",
                "labels": {"env": "development", "role": "development"}
            }
        ]

        for ep_data in demo_endpoints:
            endpoint = EndpointInfo(
                endpoint_id=ep_data["hostname"],
                hostname=ep_data["hostname"],
                ip_address=ep_data["ip"],
                os_type=ep_data["os"],
                os_version="Ubuntu 22.04",
                agent_version="1.0.0",
                last_seen=datetime.utcnow().isoformat(),
                labels=ep_data["labels"],
                groups=["production"] if ep_data["labels"]["env"] == "production" else ["development"],
                status="online",
                capabilities=["yara", "ioc", "process", "network"]
            )

            self.asgard.register_endpoint(endpoint)
            print(f"   ✅ Registered: {ep_data['hostname']} ({ep_data['labels']['role']})")

        print(f"\n   📊 Total endpoints registered: {len(self.asgard.registered_endpoints)}")
        return len(self.asgard.registered_endpoints)

    async def create_threat_hunt_campaign(
        self,
        name: str,
        target_mode: TargetSelectionMode,
        criteria: Dict[str, Any],
        priority: ScanPriority = ScanPriority.HIGH
    ) -> ThreatHuntingCampaign:
        """
        Step 3: Create and launch threat hunting campaign
        """
        print(f"\n🎯 Step 3: Creating Threat Hunting Campaign: '{name}'...")

        campaign = self.asgard.create_campaign(
            name=name,
            description=f"Automated threat hunt created at {datetime.utcnow().isoformat()}",
            target_selection_mode=target_mode,
            target_criteria=criteria,
            scan_types=["filesystem", "process", "network"],
            yara_rule_sets=["ransomware", "trojan", "backdoor"],
            ioc_feeds=["valhalla", "threatfox"],
            priority=priority,
            schedule_type="immediate",
            created_by="quickstart_demo"
        )

        print(f"   ✅ Campaign created: {campaign.campaign_id}")
        print(f"   📋 Priority: {campaign.priority.value}")
        print(f"   🎯 Target mode: {campaign.target_selection_mode.value}")
        print(f"   📊 Criteria: {json.dumps(criteria, indent=6)}")

        return campaign

    async def monitor_campaign(self, campaign_id: str, duration_seconds: int = 60):
        """
        Step 4: Monitor campaign progress
        """
        print(f"\n📊 Step 4: Monitoring Campaign Progress...")

        start_time = datetime.utcnow()
        iteration = 0

        while (datetime.utcnow() - start_time).total_seconds() < duration_seconds:
            iteration += 1

            # Get campaign status
            status = self.asgard.get_campaign_status(campaign_id)

            if status:
                print(f"\n   [Iteration {iteration}] Campaign Status:")
                print(f"   Status: {status['status']}")
                print(f"   Progress: {status['progress']['scanned']}/{status['progress']['total']} endpoints")
                print(f"   In Progress: {status['progress']['in_progress']}")
                print(f"   Failed: {status['progress']['failed']}")
                print(f"   Threats: {status['threats']['total']} (Critical: {status['threats']['critical']})")

                # Check if complete
                if status['status'] in ['completed', 'failed']:
                    print(f"\n   ✅ Campaign {status['status']}!")
                    break

            await asyncio.sleep(10)

    def demonstrate_taa_integration(self, scan_result: Dict[str, Any]):
        """
        Step 5: Demonstrate TAA (Triage and Analysis Agent) integration
        """
        print("\n🔍 Step 5: TAA Integration - Enriching Findings...")

        # In production, this would be automatic via Pub/Sub
        # TAA subscribes to 'thor-findings' topic

        print("   📡 THOR finding published to 'thor-findings' Pub/Sub topic")
        print("   🔄 TAA receives finding and begins enrichment...")

        # Simulate TAA enrichment
        print("\n   TAA Enrichment Pipeline:")
        print("   1️⃣  Extract IOCs from THOR findings")
        print("   2️⃣  Query VirusTotal for IP/domain/hash reputation")
        print("   3️⃣  Check AbuseIPDB for IP abuse history")
        print("   4️⃣  Perform LLM-based threat analysis (Gemini/Llama)")
        print("   5️⃣  Calculate threat score and confidence")
        print("   6️⃣  Classify as true positive / false positive")
        print("   7️⃣  Prioritize based on severity and context")

        # Example enriched output
        enriched_finding = {
            "original_finding": scan_result,
            "taa_enrichment": {
                "virustotal_score": 8.5,
                "abuseipdb_confidence": 95,
                "llm_analysis": "High confidence ransomware detection. File hash matches known Lockbit 3.0 variant.",
                "threat_score": 9.2,
                "classification": "true_positive",
                "severity": "critical",
                "recommended_actions": [
                    "Immediate endpoint isolation",
                    "Block source IP at firewall",
                    "Initiate incident response",
                    "Forensic data collection"
                ]
            }
        }

        print(f"\n   ✅ TAA Enrichment Complete")
        print(f"      Threat Score: {enriched_finding['taa_enrichment']['threat_score']}/10")
        print(f"      Classification: {enriched_finding['taa_enrichment']['classification']}")
        print(f"      Severity: {enriched_finding['taa_enrichment']['severity'].upper()}")

        return enriched_finding

    def demonstrate_cra_integration(self, enriched_finding: Dict[str, Any]):
        """
        Step 6: Demonstrate CRA (Containment and Response Agent) integration
        """
        print("\n🛡️  Step 6: CRA Integration - Automated Response...")

        # In production, TAA would publish to CRA topic if severity is critical

        taa_data = enriched_finding['taa_enrichment']

        if taa_data['severity'] == 'critical' and taa_data['classification'] == 'true_positive':
            print("   🚨 CRITICAL THREAT DETECTED - Triggering CRA playbook...")

            # Simulate CRA response
            print("\n   CRA Response Playbook: 'Ransomware Containment'")
            print("   1️⃣  Isolate endpoint from network (GCP firewall rule)")
            print("   2️⃣  Block malicious IP across all firewalls")
            print("   3️⃣  Disable compromised user account")
            print("   4️⃣  Create Jira incident ticket: INCIDENT-2025-001")
            print("   5️⃣  Send PagerDuty alert to on-call SOC analyst")
            print("   6️⃣  Capture memory dump for forensics")
            print("   7️⃣  Snapshot disk for investigation")

            response_actions = {
                "playbook": "ransomware_containment",
                "actions_taken": [
                    {"action": "isolate_endpoint", "status": "success", "endpoint": "web-server-01"},
                    {"action": "block_ip", "status": "success", "ip": "192.0.2.100"},
                    {"action": "disable_account", "status": "success", "user": "compromised_user"},
                    {"action": "create_ticket", "status": "success", "ticket_id": "INCIDENT-2025-001"},
                    {"action": "alert_oncall", "status": "success", "recipient": "soc-team@company.com"}
                ],
                "timestamp": datetime.utcnow().isoformat()
            }

            print(f"\n   ✅ CRA Response Complete - {len(response_actions['actions_taken'])} actions executed")
            print(f"      Incident Ticket: {response_actions['actions_taken'][3]['ticket_id']}")

            return response_actions

        else:
            print("   ℹ️  Threat severity below CRA threshold - Manual review flagged")
            return {"playbook": "manual_review", "actions_taken": []}

    def demonstrate_cla_integration(self, scan_results: Dict[str, Any]):
        """
        Step 7: Demonstrate CLA (Continuous Learning Agent) integration
        """
        print("\n🧠 Step 7: CLA Integration - Feedback Loop...")

        print("   📊 CLA receives detection data from THOR/TAA/CRA")
        print("   🔄 Processing feedback for model improvement...")

        # Simulate CLA learning
        print("\n   CLA Learning Pipeline:")
        print("   1️⃣  Analyze detection patterns and analyst feedback")
        print("   2️⃣  Update anomaly detection model with new threat vectors")
        print("   3️⃣  Identify emerging threat patterns not covered by YARA rules")
        print("   4️⃣  Recommend new YARA rules to VALHALLA")
        print("   5️⃣  Optimize false positive rate based on historical data")
        print("   6️⃣  Retrain classification models")

        cla_feedback = {
            "model_updates": [
                "Added ransomware behavioral signature to ADA model",
                "Updated process tree anomaly detection weights",
                "Incorporated new C2 beacon patterns"
            ],
            "rule_recommendations": [
                "Suggest YARA rule for new Lockbit variant",
                "Recommend Sigma rule for lateral movement technique"
            ],
            "performance_metrics": {
                "detection_rate_improvement": "+12%",
                "false_positive_reduction": "-8%",
                "mean_time_to_detect": "-15 minutes"
            }
        }

        print(f"\n   ✅ CLA Feedback Processed")
        print(f"      Model Updates: {len(cla_feedback['model_updates'])}")
        print(f"      Rule Recommendations: {len(cla_feedback['rule_recommendations'])}")
        print(f"      Detection Rate: {cla_feedback['performance_metrics']['detection_rate_improvement']}")

        return cla_feedback


async def main():
    """
    Main quick-start demonstration
    """
    print("=" * 80)
    print("🎯 THREAT HUNTING PLATFORM QUICK START")
    print("   Nextron-Style Integration with AI-Driven SOC")
    print("=" * 80)

    # Initialize
    project_id = os.getenv("GCP_PROJECT_ID", "demo-soc-project")
    orchestrator = ThreatHuntingOrchestrator(project_id)

    try:
        # Step 1: Setup threat intelligence
        await orchestrator.setup_threat_intelligence()

        # Step 2: Register endpoints
        await orchestrator.register_demo_endpoints()

        # Step 3: Create threat hunting campaigns
        print("\n" + "=" * 80)
        print("🎯 LAUNCHING THREAT HUNTING CAMPAIGNS")
        print("=" * 80)

        # Campaign 1: Hunt across all production servers
        campaign1 = await orchestrator.create_threat_hunt_campaign(
            name="Production Fleet - Ransomware Hunt",
            target_mode=TargetSelectionMode.LABEL,
            criteria={"labels": {"env": "production"}},
            priority=ScanPriority.HIGH
        )

        # Campaign 2: Webshell hunt on web servers
        campaign2 = await orchestrator.create_threat_hunt_campaign(
            name="Web Servers - Webshell Detection",
            target_mode=TargetSelectionMode.LABEL,
            criteria={"labels": {"role": "webserver"}},
            priority=ScanPriority.HIGH
        )

        # Step 4: Monitor campaign (shortened for demo)
        await orchestrator.monitor_campaign(campaign1.campaign_id, duration_seconds=30)

        # Step 5-7: Demonstrate integration with TAA/CRA/CLA
        print("\n" + "=" * 80)
        print("🔄 DEMONSTRATING SOC AGENT INTEGRATION")
        print("=" * 80)

        # Simulate a THOR finding
        simulated_finding = {
            "scan_id": "thor_scan_20250101_120000",
            "hostname": "web-server-01",
            "match_type": "yara",
            "rule_name": "Ransomware_Lockbit_3_0",
            "severity": "critical",
            "confidence": 0.95,
            "target": "/var/www/uploads/suspicious.exe",
            "file_hash": "a1b2c3d4e5f6789...",
            "timestamp": datetime.utcnow().isoformat()
        }

        # TAA enrichment
        enriched = orchestrator.demonstrate_taa_integration(simulated_finding)

        # CRA response
        response = orchestrator.demonstrate_cra_integration(enriched)

        # CLA learning
        feedback = orchestrator.demonstrate_cla_integration({
            "finding": simulated_finding,
            "enrichment": enriched,
            "response": response
        })

        # Summary
        print("\n" + "=" * 80)
        print("✅ QUICK START COMPLETE")
        print("=" * 80)
        print("\n📊 Summary:")
        print(f"   ✅ Threat Intelligence: Updated from 3 sources")
        print(f"   ✅ Endpoints Registered: {len(orchestrator.asgard.registered_endpoints)}")
        print(f"   ✅ Campaigns Launched: 2")
        print(f"   ✅ SOC Integration: TAA → CRA → CLA pipeline demonstrated")
        print("\n🎯 Next Steps:")
        print("   1. Configure GCP project ID in config files")
        print("   2. Deploy THOR agents on production endpoints")
        print("   3. Set up scheduled VALHALLA feed updates")
        print("   4. Configure TAA/CRA integration in production")
        print("   5. Review BigQuery dashboards for analytics")
        print("\n📚 See THREAT_HUNTING_README.md for full documentation")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error during quick start: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
