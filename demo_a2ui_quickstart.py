#!/usr/bin/env python3
"""
Quick A2UI Demo - Standalone Example
Generates your first A2UI screen without external dependencies
"""

import json
import sys
from datetime import datetime

# Add gatra_a2ui to path
sys.path.insert(0, '/Users/raditio.ghifiardigmail.com/Downloads/ai-driven-soc.backup')

from gatra_a2ui.components import (
    MetricCard, TimelineStep, MITRETechnique, Severity, TextComponent
)
from gatra_a2ui.builders import AttackNarrativeBuilder, SituationBriefBuilder


def demo_situation_brief():
    """Demo: Situation Brief Screen (Landing View)"""
    print("\n" + "="*70)
    print("DEMO 1: SITUATION BRIEF SCREEN")
    print("="*70)
    print("\nBuilding Situation Brief for GATRA SOC...\n")

    builder = SituationBriefBuilder()

    # Add metrics
    builder.add_metric_card(
        title="Active Incidents",
        value="7",
        severity=Severity.HIGH,
        trend="up"
    )

    builder.add_threat_actor_card(
        threat_actor="UNC3886 (APT)",
        severity=Severity.CRITICAL
    )

    builder.add_metric_card(
        title="Events/Second",
        value="12,450",
        severity=Severity.MEDIUM,
        trend="stable"
    )

    # Add incident list
    incidents = [
        "Suspicious East-West Traffic (10.5.3.12 → 10.5.6.20)",
        "Privileged Account Abuse (svc_admin)",
        "Data Staging Detected (C:\\temp\\export.zip)"
    ]
    builder.add_incident_list(incidents)

    # Add primary action
    builder.add_primary_action(
        label="Investigate Priority Incident",
        action_name="open_investigation",
        action_params={"incident_id": "INC-2025-001"}
    )

    # Build schema
    schema = builder.build()

    print("✅ Situation Brief A2UI Schema Generated!\n")
    print(json.dumps(schema, indent=2))

    return schema


def demo_attack_narrative():
    """Demo: Attack Narrative Screen (Explainability View)"""
    print("\n" + "="*70)
    print("DEMO 2: ATTACK NARRATIVE SCREEN")
    print("="*70)
    print("\nBuilding Attack Narrative for incident INC-2025-001...\n")

    builder = AttackNarrativeBuilder(incident_id="INC-2025-001")

    # Step 1: Initial Access
    builder.add_step(
        step=1,
        title="Initial Access",
        description="Phishing email delivered to user@company.com at 2025-01-03 09:32 UTC",
        mitre_technique="T1566.001",
        mitre_name="Spearphishing Attachment",
        mitre_tactic="Initial Access",
        confidence=0.92,
        evidence=["email_log_12345", "attachment_sha256_abc123"]
    )

    # Step 2: Execution
    builder.add_step(
        step=2,
        title="Execution",
        description="Malicious macro executed, dropped payload to %TEMP%\\update.exe",
        mitre_technique="T1204.002",
        mitre_name="Malicious File",
        mitre_tactic="Execution",
        confidence=0.89,
        evidence=["process_create_67890", "file_write_11111"]
    )

    # Step 3: Lateral Movement
    builder.add_step(
        step=3,
        title="Lateral Movement",
        description="SMB connections from compromised host to 5 internal servers",
        mitre_technique="T1021.002",
        mitre_name="SMB/Windows Admin Shares",
        mitre_tactic="Lateral Movement",
        confidence=0.87,
        evidence=["smb_log_22222", "netflow_33333"]
    )

    # Step 4: Data Staging (with uncertainty)
    builder.add_step(
        step=4,
        title="Data Staging",
        description="Large archive created in C:\\temp\\export.zip (2.3 GB)",
        mitre_technique="T1560.001",
        mitre_name="Archive via Utility",
        mitre_tactic="Collection",
        confidence=0.61,
        evidence=["file_create_44444"],
        uncertainty="Could be legitimate backup activity - requires verification"
    )

    # Add action button for uncertain step
    builder.add_expand_action(step_number=4)

    # Build schema
    schema = builder.build()

    print("✅ Attack Narrative A2UI Schema Generated!\n")
    print(json.dumps(schema, indent=2))

    return schema


def demo_manual_components():
    """Demo: Building components manually"""
    print("\n" + "="*70)
    print("DEMO 3: MANUAL COMPONENT BUILDING")
    print("="*70)
    print("\nBuilding individual A2UI components...\n")

    # Create a metric card
    metric_card = MetricCard(
        title="Mean Time to Respond (MTTR)",
        value="23 minutes",
        severity=Severity.LOW,
        trend="down",  # Lower is better!
        icon="timer"
    )

    component = metric_card.to_a2ui("mttr_card")
    print("📊 MetricCard Component:")
    print(json.dumps(component, indent=2))

    # Create a timeline step
    timeline_step = TimelineStep(
        step_number=1,
        title="Reconnaissance",
        description="Attacker scanned network ports 22, 80, 443, 3389",
        technique=MITRETechnique(
            technique_id="T1046",
            technique_name="Network Service Discovery",
            tactic="Discovery"
        ),
        confidence=0.95,
        evidence_ids=["netflow_scan_001", "ids_alert_002"]
    )

    component = timeline_step.to_a2ui("recon_step")
    print("\n📅 TimelineStep Component:")
    print(json.dumps(component, indent=2))

    return {"metric_card": metric_card, "timeline_step": timeline_step}


def show_analyst_view(schema_type, schema):
    """Simulate what the analyst would see in the UI"""
    print("\n" + "="*70)
    print(f"ANALYST VIEW: {schema_type.upper()}")
    print("="*70)

    if schema_type == "situation_brief":
        print("""
┌────────────────────────────────────────────────────────────────┐
│                  GATRA SOC - Situation Brief                   │
└────────────────────────────────────────────────────────────────┘

  ╔═══════════════════╗  ╔═══════════════════════════════════╗
  ║ Active Incidents  ║  ║ Likely Threat Actor               ║
  ║       7           ║  ║ UNC3886 (APT)                     ║
  ║   [HIGH] ↑        ║  ║ [CRITICAL] ⚠                      ║
  ╚═══════════════════╝  ╚═══════════════════════════════════╝

  ╔═══════════════════╗
  ║ Events/Second     ║
  ║     12,450        ║
  ║   [MEDIUM] →      ║
  ╚═══════════════════╝

  Priority Alerts:
  • Suspicious East-West Traffic (10.5.3.12 → 10.5.6.20)
  • Privileged Account Abuse (svc_admin)
  • Data Staging Detected (C:\\temp\\export.zip)

  ┌──────────────────────────────────────────┐
  │  [Investigate Priority Incident]         │
  └──────────────────────────────────────────┘
        """)

    elif schema_type == "attack_narrative":
        print("""
┌────────────────────────────────────────────────────────────────┐
│            Attack Narrative: INC-2025-001                      │
└────────────────────────────────────────────────────────────────┘

  1. Initial Access ━━━━━━━━━━━━━━━━━━━━ [92% confident]
     Phishing email delivered to user@company.com
     MITRE: T1566.001 - Spearphishing Attachment
     Evidence: email_log_12345, attachment_sha256_abc123

  2. Execution ━━━━━━━━━━━━━━━━━━━━━━━━━ [89% confident]
     Malicious macro executed, dropped payload
     MITRE: T1204.002 - Malicious File
     Evidence: process_create_67890, file_write_11111

  3. Lateral Movement ━━━━━━━━━━━━━━━━━━ [87% confident]
     SMB connections to 5 internal servers
     MITRE: T1021.002 - SMB/Windows Admin Shares
     Evidence: smb_log_22222, netflow_33333

  4. Data Staging ━━━━━━━━━━━━━━━━━━━━━━ [61% confident] ⚠
     Large archive created (2.3 GB)
     MITRE: T1560.001 - Archive via Utility
     Evidence: file_create_44444
     ⚠ Uncertainty: Could be legitimate backup - needs verification

  ┌──────────────────────────────────────────┐
  │    [Request More Evidence]               │
  └──────────────────────────────────────────┘
        """)


def main():
    """Run all A2UI demos"""
    print("\n" + "🚀"*35)
    print("GATRA A2UI QUICKSTART DEMO")
    print("Building Your First Agent-Driven SOC Screens")
    print("🚀"*35)

    # Demo 1: Situation Brief
    situation_schema = demo_situation_brief()
    show_analyst_view("situation_brief", situation_schema)

    input("\nPress ENTER to see next demo...")

    # Demo 2: Attack Narrative
    narrative_schema = demo_attack_narrative()
    show_analyst_view("attack_narrative", narrative_schema)

    input("\nPress ENTER to see manual component building...")

    # Demo 3: Manual components
    components = demo_manual_components()

    # Summary
    print("\n" + "="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    print("""
✅ You've successfully generated A2UI schemas for:

   1. Situation Brief Screen
      - Metric cards showing active incidents and threat actors
      - Priority alert list
      - Quick action buttons

   2. Attack Narrative Screen
      - 4-step attack timeline
      - MITRE ATT&CK technique mapping
      - Confidence indicators per step
      - Evidence linking
      - Uncertainty disclosure

   3. Individual Components
      - MetricCard for KPIs
      - TimelineStep for attack progression

WHAT JUST HAPPENED:
───────────────────
Instead of chat messages like:
  "TAA: I detected 7 incidents. The threat actor is UNC3886..."

Your agents generated DECLARATIVE UI schemas that render as:
  ✓ Interactive dashboards
  ✓ Timeline visualizations
  ✓ Evidence cards
  ✓ Action buttons

NEXT STEPS:
───────────
1. ✅ You've built A2UI schemas (this demo)
2. 📤 Next: Publish to Pub/Sub (see A2UI_QUICKSTART.md)
3. 🖥️  Then: Build frontend renderer (React/Flutter)
4. 🔄 Finally: Connect agents end-to-end

SECURITY:
─────────
Notice: No code execution! Only declarative JSON.
Your frontend controls what renders from a trusted catalog.

The schemas above would be published to:
  Topic: gatra-a2ui-updates
  Audit: gatra_audit.a2ui_states (BigQuery)

Want to test with real Pub/Sub? Run:
  python examples/taa_a2ui_example.py

(Requires GCP credentials configured)
    """)

    print("\n" + "="*70)
    print("A2UI schemas saved to memory - ready to publish!")
    print("="*70 + "\n")

    return {
        "situation_brief": situation_schema,
        "attack_narrative": narrative_schema,
        "components": components
    }


if __name__ == "__main__":
    try:
        schemas = main()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
