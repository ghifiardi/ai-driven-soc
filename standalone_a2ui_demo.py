#!/usr/bin/env python3
"""
Standalone A2UI Demo - Zero Dependencies
Demonstrates GATRA A2UI screen building without any external libraries
"""

import json
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional


# ============================================================================
# MINIMAL A2UI COMPONENTS (Standalone)
# ============================================================================

class Severity(str, Enum):
    """Severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


def build_metric_card(component_id: str, title: str, value: str,
                      severity: Optional[str] = None, trend: Optional[str] = None) -> Dict:
    """Build a metric card component"""
    component = {
        "id": component_id,
        "component": {
            "MetricCard": {
                "title": {"literalString": title},
                "value": {"literalString": value}
            }
        }
    }

    if severity:
        component["component"]["MetricCard"]["severity"] = severity
    if trend:
        component["component"]["MetricCard"]["trend"] = trend

    return component


def build_timeline_step(component_id: str, step_number: int, title: str,
                        description: str, mitre_id: Optional[str] = None,
                        mitre_name: Optional[str] = None, confidence: Optional[float] = None,
                        evidence: Optional[List[str]] = None, uncertainty: Optional[str] = None) -> Dict:
    """Build a timeline step component"""
    component_data = {
        "step_number": step_number,
        "title": {"literalString": title},
        "description": {"literalString": description}
    }

    if mitre_id and mitre_name:
        component_data["technique"] = {
            "mitre_id": mitre_id,
            "name": mitre_name
        }

    if confidence is not None:
        component_data["confidence"] = confidence

    if evidence:
        component_data["evidence_ids"] = evidence

    if uncertainty:
        component_data["uncertainty"] = {"literalString": uncertainty}

    return {
        "id": component_id,
        "component": {
            "TimelineStep": component_data
        }
    }


def build_text(component_id: str, text: str, style: str = "body") -> Dict:
    """Build a text component"""
    return {
        "id": component_id,
        "component": {
            "Text": {
                "text": {"literalString": text},
                "style": style
            }
        }
    }


def build_button(component_id: str, label: str, action_name: str,
                 action_params: Optional[Dict] = None, style: str = "primary") -> Dict:
    """Build a button component"""
    button_data = {
        "label": {"literalString": label},
        "action": {"name": action_name},
        "style": style
    }

    if action_params:
        button_data["action"]["params"] = action_params

    return {
        "id": component_id,
        "component": {
            "Button": button_data
        }
    }


# ============================================================================
# DEMO 1: SITUATION BRIEF
# ============================================================================

def demo_situation_brief():
    """Build Situation Brief A2UI screen"""
    print("\n" + "="*70)
    print("DEMO 1: SITUATION BRIEF SCREEN (Landing View)")
    print("="*70)
    print("\n🔨 Building agent-generated SOC dashboard...\n")

    components = []

    # Header
    components.append(build_text("header", "GATRA SOC - Situation Brief", "heading1"))

    # Metric cards
    components.append(build_metric_card(
        "active_incidents",
        "Active Incidents",
        "7",
        severity="high",
        trend="up"
    ))

    components.append(build_metric_card(
        "threat_actor",
        "Likely Threat Actor",
        "UNC3886 (APT)",
        severity="critical"
    ))

    components.append(build_metric_card(
        "events_per_sec",
        "Events/Second",
        "12,450",
        severity="medium",
        trend="stable"
    ))

    # Alert list
    components.append(build_text(
        "alert_1",
        "• Suspicious East-West Traffic (10.5.3.12 → 10.5.6.20)",
        "body"
    ))
    components.append(build_text(
        "alert_2",
        "• Privileged Account Abuse (svc_admin)",
        "body"
    ))
    components.append(build_text(
        "alert_3",
        "• Data Staging Detected (C:\\temp\\export.zip)",
        "body"
    ))

    # Primary action
    components.append(build_button(
        "primary_action",
        "Investigate Priority Incident",
        "open_investigation",
        {"incident_id": "INC-2025-001"},
        "primary"
    ))

    schema = {
        "surfaceUpdate": {
            "components": components
        }
    }

    print("✅ Generated A2UI Schema with components:")
    print(f"   - Header text")
    print(f"   - 3 metric cards (Active Incidents, Threat Actor, Events/Sec)")
    print(f"   - 3 priority alerts")
    print(f"   - 1 action button\n")

    print("📄 A2UI JSON Schema:")
    print(json.dumps(schema, indent=2))

    return schema


# ============================================================================
# DEMO 2: ATTACK NARRATIVE
# ============================================================================

def demo_attack_narrative():
    """Build Attack Narrative A2UI screen"""
    print("\n" + "="*70)
    print("DEMO 2: ATTACK NARRATIVE SCREEN (Explainability)")
    print("="*70)
    print("\n🔨 Building explainable attack timeline for INC-2025-001...\n")

    components = []

    # Header
    components.append(build_text(
        "narrative_header",
        "Attack Narrative: INC-2025-001",
        "heading2"
    ))

    # Timeline steps
    components.append(build_timeline_step(
        "step_1",
        step_number=1,
        title="Initial Access",
        description="Phishing email delivered to user@company.com at 2025-01-03 09:32 UTC",
        mitre_id="T1566.001",
        mitre_name="Spearphishing Attachment",
        confidence=0.92,
        evidence=["email_log_12345", "attachment_sha256_abc123"]
    ))

    components.append(build_timeline_step(
        "step_2",
        step_number=2,
        title="Execution",
        description="Malicious macro executed, dropped payload to %TEMP%\\update.exe",
        mitre_id="T1204.002",
        mitre_name="Malicious File",
        confidence=0.89,
        evidence=["process_create_67890", "file_write_11111"]
    ))

    components.append(build_timeline_step(
        "step_3",
        step_number=3,
        title="Lateral Movement",
        description="SMB connections from compromised host to 5 internal servers",
        mitre_id="T1021.002",
        mitre_name="SMB/Windows Admin Shares",
        confidence=0.87,
        evidence=["smb_log_22222", "netflow_33333"]
    ))

    components.append(build_timeline_step(
        "step_4",
        step_number=4,
        title="Data Staging",
        description="Large archive created in C:\\temp\\export.zip (2.3 GB)",
        mitre_id="T1560.001",
        mitre_name="Archive via Utility",
        confidence=0.61,
        evidence=["file_create_44444"],
        uncertainty="Could be legitimate backup activity - requires verification"
    ))

    # Action button for uncertain step
    components.append(build_button(
        "expand_action",
        "Request More Evidence",
        "expand_analysis",
        {"step": 4},
        "secondary"
    ))

    schema = {
        "surfaceUpdate": {
            "components": components
        }
    }

    print("✅ Generated A2UI Schema with components:")
    print(f"   - Header text")
    print(f"   - 4 timeline steps with MITRE ATT&CK mapping")
    print(f"   - Confidence indicators (92%, 89%, 87%, 61%)")
    print(f"   - Evidence linking")
    print(f"   - Uncertainty disclosure for step 4")
    print(f"   - Action button\n")

    print("📄 A2UI JSON Schema:")
    print(json.dumps(schema, indent=2))

    return schema


# ============================================================================
# VISUAL REPRESENTATION
# ============================================================================

def show_analyst_view(schema_type: str):
    """Show what the analyst would see in the rendered UI"""
    print("\n" + "="*70)
    print(f"👤 ANALYST VIEW: {schema_type.upper()}")
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

  💡 No chat messages! Pure agent-driven interface.
        """)

    elif schema_type == "attack_narrative":
        print("""
┌────────────────────────────────────────────────────────────────┐
│            Attack Narrative: INC-2025-001                      │
└────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────┐
  │ 1. Initial Access ━━━━━━━━━━━━━━━━ [92% confident]         │
  │    Phishing email delivered to user@company.com             │
  │    🎯 MITRE: T1566.001 - Spearphishing Attachment           │
  │    📎 Evidence: email_log_12345, attachment_sha256_abc123   │
  └─────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────┐
  │ 2. Execution ━━━━━━━━━━━━━━━━━━━━ [89% confident]          │
  │    Malicious macro executed, dropped payload                │
  │    🎯 MITRE: T1204.002 - Malicious File                     │
  │    📎 Evidence: process_create_67890, file_write_11111      │
  └─────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────┐
  │ 3. Lateral Movement ━━━━━━━━━━━━━ [87% confident]          │
  │    SMB connections to 5 internal servers                    │
  │    🎯 MITRE: T1021.002 - SMB/Windows Admin Shares           │
  │    📎 Evidence: smb_log_22222, netflow_33333                │
  └─────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────┐
  │ 4. Data Staging ━━━━━━━━━━━━━━━━━ [61% confident] ⚠        │
  │    Large archive created (2.3 GB)                           │
  │    🎯 MITRE: T1560.001 - Archive via Utility                │
  │    📎 Evidence: file_create_44444                           │
  │    ⚠  UNCERTAINTY: Could be legitimate backup activity      │
  └─────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────┐
  │    [Request More Evidence]               │
  └──────────────────────────────────────────┘

  💡 Agent shows its reasoning, confidence, and uncertainty!
        """)


# ============================================================================
# MAIN DEMO
# ============================================================================

def main():
    """Run the complete A2UI demo"""
    print("\n" + "🚀"*35)
    print("         GATRA A2UI QUICKSTART DEMO")
    print("   Building Your First Agent-Driven SOC Screens")
    print("🚀"*35)

    print("""
What you're about to see:
─────────────────────────
Your AI agents (ADA, TAA, CRA) will generate DECLARATIVE UI schemas
instead of chat messages. These schemas render as interactive screens.

Benefits:
✓ Security: No code execution, only declarative JSON
✓ Explainability: Agents show their reasoning visually
✓ Efficiency: No 20-message chat conversations
✓ Trust: Human-in-the-loop workflows built-in
    """)

    input("Press ENTER to start Demo 1...")

    # Demo 1: Situation Brief
    schema1 = demo_situation_brief()
    show_analyst_view("situation_brief")

    print("\n" + "💾"*35)
    print("WHAT JUST HAPPENED:")
    print("─" * 70)
    print("""
Instead of this (old way):
  Agent: "You have 7 active incidents."
  Analyst: "What are the incidents?"
  Agent: "The first incident is..."
  Analyst: "What's the threat actor?"
  Agent: "The threat actor is UNC3886"

The agent generated ONE A2UI schema that contains:
  ✓ All metrics (incidents, threat actor, events/sec)
  ✓ Alert list
  ✓ Action buttons
  ✓ Visual hierarchy

This schema would be published to Pub/Sub and rendered by your frontend.
    """)

    input("\nPress ENTER for Demo 2...")

    # Demo 2: Attack Narrative
    schema2 = demo_attack_narrative()
    show_analyst_view("attack_narrative")

    print("\n" + "💾"*35)
    print("WHAT JUST HAPPENED:")
    print("─" * 70)
    print("""
The TAA agent analyzed an incident and instead of describing it in text,
it generated a structured attack narrative with:

  ✓ 4-step timeline (Initial Access → Execution → Lateral Movement → Data Staging)
  ✓ MITRE ATT&CK technique mapping for each step
  ✓ Confidence scores (92%, 89%, 87%, 61%)
  ✓ Evidence linking to source logs
  ✓ Explicit uncertainty disclosure (step 4 is only 61% confident)
  ✓ "Request More Evidence" button for uncertain steps

This is EXPLAINABLE AI in action:
  - Analyst understands WHY the agent thinks this
  - Regulator can audit the decision trail
  - GDPR Article 22 compliant (automated decision explanation)
    """)

    # Summary
    print("\n" + "="*70)
    print("✅ DEMO COMPLETE!")
    print("="*70)
    print(f"""
You successfully generated {len(schema1['surfaceUpdate']['components']) + len(schema2['surfaceUpdate']['components'])} A2UI components across 2 screens:

1️⃣  Situation Brief:
   - {len(schema1['surfaceUpdate']['components'])} components (metrics, alerts, actions)
   - Agent: ADA (Anomaly Detection Agent)
   - Purpose: 10-second situational awareness

2️⃣  Attack Narrative:
   - {len(schema2['surfaceUpdate']['components'])} components (timeline, MITRE, evidence)
   - Agent: TAA (Triage & Analysis Agent)
   - Purpose: Explainable threat analysis

NEXT STEPS:
───────────
✅ You've built A2UI schemas (this demo)
📤 Next: Publish to Pub/Sub
   └─ See: gatra_a2ui/publisher.py
   └─ Command: publisher.publish(schema, screen_type="attack_narrative", agent_id="TAA")

🖥️  Then: Build frontend renderer
   └─ React: Use @gatra/a2ui-react
   └─ Flutter: Use a2ui_flutter package
   └─ Map catalog components to native widgets

🔄 Finally: Integrate with agents
   └─ See: examples/taa_a2ui_example.py
   └─ Pattern: analyze() → build_ui() → publish()

ARCHITECTURE:
─────────────
  Telemetry → Agents → A2UI Schema → Pub/Sub → Frontend → Analyst

SECURITY MODEL:
───────────────
  ✓ Declarative only (no code execution)
  ✓ Component whitelisting (trusted catalog)
  ✓ Content sanitization
  ✓ Audit trail (BigQuery logging)

FILES TO EXPLORE:
─────────────────
  📖 docs/A2UI_GATRA_SOC_IMPLEMENTATION.md  (Full guide)
  📄 A2UI_QUICKSTART.md                      (15-min tutorial)
  🐍 gatra_a2ui/                             (Python package)
  💡 examples/taa_a2ui_example.py            (TAA integration)

RESOURCES:
──────────
  • Google A2UI: https://a2ui.org/
  • GitHub: https://github.com/google/A2UI
  • GATRA Docs: See README.md
    """)

    print("\n" + "🎉"*35)
    print("Welcome to agent-native SOC interfaces!")
    print("🎉"*35 + "\n")

    return {
        "situation_brief_schema": schema1,
        "attack_narrative_schema": schema2,
        "total_components": len(schema1['surfaceUpdate']['components']) + len(schema2['surfaceUpdate']['components'])
    }


if __name__ == "__main__":
    try:
        result = main()
        print(f"✅ Demo completed successfully!")
        print(f"   Generated {result['total_components']} A2UI components")
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
