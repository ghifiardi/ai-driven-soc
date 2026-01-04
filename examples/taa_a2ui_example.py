"""
Example: TAA Agent with A2UI Integration

This example shows how the Triage and Analysis Agent (TAA) uses A2UI
to generate interactive investigation interfaces instead of text-only outputs.
"""

import logging
from datetime import datetime
from typing import Dict, Any

from gatra_a2ui import (
    AttackNarrativeBuilder,
    DecisionPanelBuilder,
    A2UIPublisher,
)
from gatra_a2ui.components import Severity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TAAWithA2UI:
    """
    Enhanced TAA agent that generates A2UI interfaces.

    Instead of outputting text-only analysis, this agent builds
    rich, interactive UIs that guide analysts through investigations.
    """

    def __init__(self, project_id: str = "chronicle-dev-2be9"):
        self.project_id = project_id
        self.a2ui_publisher = A2UIPublisher(project_id=project_id)

    def analyze_incident(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze incident and generate A2UI investigation interface.

        Args:
            alert_data: Alert from ADA agent

        Returns:
            Analysis results with A2UI schema
        """
        incident_id = alert_data.get("alert_id", "UNKNOWN")
        logger.info(f"Analyzing incident {incident_id} with A2UI generation")

        # Step 1: Perform traditional analysis
        analysis_result = self._perform_analysis(alert_data)

        # Step 2: Build Attack Narrative UI
        narrative_schema = self._build_attack_narrative_ui(
            incident_id=incident_id,
            analysis=analysis_result
        )

        # Step 3: Publish narrative UI
        self.a2ui_publisher.publish(
            schema=narrative_schema,
            screen_type="attack_narrative",
            agent_id="TAA",
            incident_id=incident_id,
            priority="high" if analysis_result.get("severity") == "critical" else "normal"
        )

        # Step 4: If high severity, also build Decision Panel
        if analysis_result.get("severity") in ["high", "critical"]:
            decision_schema = self._build_decision_panel_ui(
                incident_id=incident_id,
                analysis=analysis_result
            )

            self.a2ui_publisher.publish(
                schema=decision_schema,
                screen_type="decision_panel",
                agent_id="TAA",
                incident_id=incident_id,
                priority="high"
            )

        return {
            "incident_id": incident_id,
            "analysis": analysis_result,
            "a2ui_published": True
        }

    def _perform_analysis(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform traditional TAA analysis.

        In a real implementation, this would:
        - Enrich with threat intel (VirusTotal, AbuseIPDB)
        - Run LLM-based analysis
        - Map to MITRE ATT&CK
        - Calculate confidence scores
        """
        # Simulated analysis
        return {
            "is_true_positive": True,
            "confidence": 0.92,
            "severity": "high",
            "threat_actor": "UNC3886",
            "attack_chain": [
                {
                    "step": 1,
                    "title": "Initial Access",
                    "description": "Phishing email delivered to user@company.com at 2025-01-03 09:32 UTC",
                    "mitre_technique": "T1566.001",
                    "mitre_name": "Spearphishing Attachment",
                    "mitre_tactic": "Initial Access",
                    "confidence": 0.92,
                    "evidence": ["email_log_id_12345", "attachment_sha256_abc123"]
                },
                {
                    "step": 2,
                    "title": "Execution",
                    "description": "Malicious macro executed, dropped payload to %TEMP%\\update.exe",
                    "mitre_technique": "T1204.002",
                    "mitre_name": "Malicious File",
                    "mitre_tactic": "Execution",
                    "confidence": 0.89,
                    "evidence": ["process_create_id_67890", "file_write_id_11111"]
                },
                {
                    "step": 3,
                    "title": "Lateral Movement",
                    "description": "SMB connections from compromised host to 5 internal servers",
                    "mitre_technique": "T1021.002",
                    "mitre_name": "SMB/Windows Admin Shares",
                    "mitre_tactic": "Lateral Movement",
                    "confidence": 0.87,
                    "evidence": ["smb_log_id_22222", "netflow_id_33333"]
                },
                {
                    "step": 4,
                    "title": "Data Staging",
                    "description": "Large archive created in C:\\temp\\export.zip (2.3 GB)",
                    "mitre_technique": "T1560.001",
                    "mitre_name": "Archive via Utility",
                    "mitre_tactic": "Collection",
                    "confidence": 0.61,
                    "evidence": ["file_create_id_44444"],
                    "uncertainty": "Could be legitimate backup activity - requires verification"
                }
            ],
            "recommended_actions": [
                {
                    "action_id": "isolate_host",
                    "title": "Isolate Host",
                    "description": "Disconnect WORKSTATION-42 from network",
                    "risk": "medium",
                    "impact": "User unable to work (estimated 2-4 hours)",
                    "recommended": True,
                    "estimated_time": "5 minutes"
                },
                {
                    "action_id": "block_ip",
                    "title": "Block C2 IP",
                    "description": "Add firewall rule to block 185.220.101.42",
                    "risk": "low",
                    "impact": "Minimal - external IP",
                    "recommended": True,
                    "estimated_time": "2 minutes"
                },
                {
                    "action_id": "disable_account",
                    "title": "Disable User Account",
                    "description": "Disable user@company.com in Active Directory",
                    "risk": "high",
                    "impact": "User locked out, may affect ongoing projects",
                    "recommended": False,
                    "estimated_time": "1 minute"
                }
            ]
        }

    def _build_attack_narrative_ui(
        self,
        incident_id: str,
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build Attack Narrative UI using A2UI builders"""
        builder = AttackNarrativeBuilder(incident_id=incident_id)

        # Add each step from the attack chain
        for step_data in analysis.get("attack_chain", []):
            builder.add_step(
                step=step_data["step"],
                title=step_data["title"],
                description=step_data["description"],
                mitre_technique=step_data.get("mitre_technique"),
                mitre_name=step_data.get("mitre_name"),
                mitre_tactic=step_data.get("mitre_tactic"),
                confidence=step_data.get("confidence"),
                evidence=step_data.get("evidence"),
                uncertainty=step_data.get("uncertainty")
            )

        # Add action button for uncertain steps
        uncertain_steps = [
            s for s in analysis.get("attack_chain", [])
            if s.get("uncertainty") or s.get("confidence", 1.0) < 0.7
        ]

        if uncertain_steps:
            builder.add_expand_action(step_number=uncertain_steps[0]["step"])

        return builder.build()

    def _build_decision_panel_ui(
        self,
        incident_id: str,
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build Decision Panel UI for containment recommendations"""
        builder = DecisionPanelBuilder(incident_id=incident_id)

        # Add recommended actions
        for action in analysis.get("recommended_actions", []):
            # Map risk string to Severity enum
            risk_map = {
                "low": Severity.LOW,
                "medium": Severity.MEDIUM,
                "high": Severity.HIGH,
                "critical": Severity.CRITICAL
            }

            builder.add_action(
                action_id=action["action_id"],
                title=action["title"],
                description=action["description"],
                risk=risk_map.get(action["risk"], Severity.MEDIUM),
                impact=action["impact"],
                recommended=action.get("recommended", False),
                estimated_time=action.get("estimated_time")
            )

        # Add approval workflow
        builder.add_approval_workflow(
            approve_label="Execute Containment",
            reject_label="Manual Review First",
            note_required=True
        )

        # Add impact simulation
        builder.add_impact_simulation(
            affected_users=1,
            affected_systems=1,
            estimated_downtime="2-4 hours",
            business_impact="Low - non-critical user"
        )

        return builder.build()


def main():
    """
    Example usage of TAA with A2UI.

    This demonstrates how an alert from ADA is transformed into
    interactive investigation UIs instead of text-only output.
    """
    # Simulated alert from ADA
    alert_from_ada = {
        "alert_id": "INC-2025-001",
        "timestamp": datetime.utcnow().isoformat(),
        "source_ip": "10.5.3.12",
        "destination_ip": "185.220.101.42",
        "alert_type": "suspicious_network_activity",
        "severity": "high",
        "description": "Outbound connection to known C2 IP address",
        "host": "WORKSTATION-42",
        "user": "user@company.com"
    }

    # Initialize TAA with A2UI
    taa = TAAWithA2UI(project_id="chronicle-dev-2be9")

    # Analyze incident (will generate and publish A2UI)
    result = taa.analyze_incident(alert_from_ada)

    logger.info(f"Analysis complete: {result['incident_id']}")
    logger.info(f"A2UI schemas published: {result['a2ui_published']}")
    logger.info(f"Severity: {result['analysis']['severity']}")
    logger.info(f"Confidence: {result['analysis']['confidence']}")

    # Analyst now sees:
    # 1. Attack Narrative screen with timeline and MITRE mapping
    # 2. Decision Panel screen with recommended containment actions
    # 3. All fully interactive, no chat messages

    print("\n" + "="*70)
    print("TAA A2UI Example Complete")
    print("="*70)
    print(f"\nGenerated UI screens for incident: {result['incident_id']}")
    print("\nAnalyst would now see:")
    print("  1. Attack Narrative: 4-step timeline with MITRE ATT&CK mapping")
    print("  2. Decision Panel: 3 containment actions with approval workflow")
    print("\nNo chat messages. No text-only output. Pure agent-driven UI.")
    print("="*70)


if __name__ == "__main__":
    main()
