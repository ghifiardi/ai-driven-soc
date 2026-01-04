#!/usr/bin/env python3
"""
Auto-Running A2UI Demo - Builds your first A2UI screens
No interaction required - perfect for quick demonstration
"""

import json
import time
from typing import Dict, List, Optional


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
        component_data["technique"] = {"mitre_id": mitre_id, "name": mitre_name}
    if confidence is not None:
        component_data["confidence"] = confidence
    if evidence:
        component_data["evidence_ids"] = evidence
    if uncertainty:
        component_data["uncertainty"] = {"literalString": uncertainty}
    return {"id": component_id, "component": {"TimelineStep": component_data}}


def build_text(component_id: str, text: str, style: str = "body") -> Dict:
    """Build a text component"""
    return {
        "id": component_id,
        "component": {"Text": {"text": {"literalString": text}, "style": style}}
    }


def build_button(component_id: str, label: str, action_name: str,
                 action_params: Optional[Dict] = None) -> Dict:
    """Build a button component"""
    button_data = {"label": {"literalString": label}, "action": {"name": action_name}, "style": "primary"}
    if action_params:
        button_data["action"]["params"] = action_params
    return {"id": component_id, "component": {"Button": button_data}}


def pause(seconds=1):
    """Visual pause"""
    time.sleep(seconds)


def main():
    print("\n" + "🚀" * 35)
    print("         GATRA A2UI DEMO - AUTO RUN")
    print("   Building Your First Agent-Driven SOC Screens")
    print("🚀" * 35 + "\n")

    # ============================================================================
    # DEMO 1: SITUATION BRIEF
    # ============================================================================

    print("="*70)
    print("DEMO 1: SITUATION BRIEF (Landing Dashboard)")
    print("="*70 + "\n")

    components = []
    components.append(build_text("header", "GATRA SOC - Situation Brief", "heading1"))
    components.append(build_metric_card("active_incidents", "Active Incidents", "7", "high", "up"))
    components.append(build_metric_card("threat_actor", "Likely Threat Actor", "UNC3886 (APT)", "critical"))
    components.append(build_metric_card("events_per_sec", "Events/Second", "12,450", "medium", "stable"))
    components.append(build_text("alert_1", "• Suspicious East-West Traffic (10.5.3.12 → 10.5.6.20)", "body"))
    components.append(build_text("alert_2", "• Privileged Account Abuse (svc_admin)", "body"))
    components.append(build_text("alert_3", "• Data Staging Detected (C:\\temp\\export.zip)", "body"))
    components.append(build_button("primary_action", "Investigate Priority Incident", "open_investigation", {"incident_id": "INC-2025-001"}))

    brief_schema = {"surfaceUpdate": {"components": components}}

    print("✅ Building components...")
    pause(0.5)
    print(f"   ✓ Header: 'GATRA SOC - Situation Brief'")
    print(f"   ✓ Metric Card: Active Incidents = 7 [HIGH ↑]")
    print(f"   ✓ Metric Card: Threat Actor = UNC3886 [CRITICAL]")
    print(f"   ✓ Metric Card: Events/Second = 12,450 [MEDIUM →]")
    print(f"   ✓ Alert List: 3 priority alerts")
    print(f"   ✓ Action Button: 'Investigate Priority Incident'\n")

    print("📄 Generated A2UI Schema:\n")
    print(json.dumps(brief_schema, indent=2))

    print("\n👤 What the analyst sees:\n")
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

    pause(2)

    # ============================================================================
    # DEMO 2: ATTACK NARRATIVE
    # ============================================================================

    print("\n" + "="*70)
    print("DEMO 2: ATTACK NARRATIVE (Explainable AI)")
    print("="*70 + "\n")

    components = []
    components.append(build_text("narrative_header", "Attack Narrative: INC-2025-001", "heading2"))
    components.append(build_timeline_step(
        "step_1", 1, "Initial Access",
        "Phishing email delivered to user@company.com at 2025-01-03 09:32 UTC",
        "T1566.001", "Spearphishing Attachment", 0.92,
        ["email_log_12345", "attachment_sha256_abc123"]
    ))
    components.append(build_timeline_step(
        "step_2", 2, "Execution",
        "Malicious macro executed, dropped payload to %TEMP%\\update.exe",
        "T1204.002", "Malicious File", 0.89,
        ["process_create_67890", "file_write_11111"]
    ))
    components.append(build_timeline_step(
        "step_3", 3, "Lateral Movement",
        "SMB connections from compromised host to 5 internal servers",
        "T1021.002", "SMB/Windows Admin Shares", 0.87,
        ["smb_log_22222", "netflow_33333"]
    ))
    components.append(build_timeline_step(
        "step_4", 4, "Data Staging",
        "Large archive created in C:\\temp\\export.zip (2.3 GB)",
        "T1560.001", "Archive via Utility", 0.61,
        ["file_create_44444"],
        "Could be legitimate backup activity - requires verification"
    ))
    components.append(build_button("expand_action", "Request More Evidence", "expand_analysis", {"step": 4}))

    narrative_schema = {"surfaceUpdate": {"components": components}}

    print("✅ Building components...")
    pause(0.5)
    print(f"   ✓ Header: 'Attack Narrative: INC-2025-001'")
    print(f"   ✓ Step 1: Initial Access [T1566.001] - 92% confident")
    print(f"   ✓ Step 2: Execution [T1204.002] - 89% confident")
    print(f"   ✓ Step 3: Lateral Movement [T1021.002] - 87% confident")
    print(f"   ✓ Step 4: Data Staging [T1560.001] - 61% confident ⚠")
    print(f"   ✓ Action Button: 'Request More Evidence'\n")

    print("📄 Generated A2UI Schema:\n")
    print(json.dumps(narrative_schema, indent=2))

    print("\n👤 What the analyst sees:\n")
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
    """)

    pause(2)

    # ============================================================================
    # SUMMARY
    # ============================================================================

    total_components = len(brief_schema['surfaceUpdate']['components']) + len(narrative_schema['surfaceUpdate']['components'])

    print("\n" + "="*70)
    print("✅ DEMO COMPLETE!")
    print("="*70 + "\n")

    print(f"📊 Statistics:")
    print(f"   • Total components generated: {total_components}")
    print(f"   • Situation Brief: {len(brief_schema['surfaceUpdate']['components'])} components")
    print(f"   • Attack Narrative: {len(narrative_schema['surfaceUpdate']['components'])} components")

    print(f"\n🎯 What Just Happened:\n")
    print("   Instead of chat messages, your agents generated DECLARATIVE")
    print("   UI schemas that render as interactive, explainable screens.\n")

    print("   OLD WAY (Chat-based):")
    print("   ─────────────────────")
    print("   TAA: 'I detected suspicious activity...'")
    print("   Analyst: 'What kind of activity?'")
    print("   TAA: 'It appears to be phishing...'")
    print("   Analyst: 'Show me the timeline'")
    print("   TAA: 'Step 1 was...'")
    print("   [20 messages later...]")

    print("\n   NEW WAY (A2UI):")
    print("   ───────────────")
    print("   TAA: [Generates complete attack narrative UI]")
    print("   Analyst: [Sees interactive timeline with all details]")
    print("   [1 schema. Zero messages. Pure agent-driven interface.]")

    print(f"\n🔒 Security:")
    print("   ✓ No code execution - only declarative JSON")
    print("   ✓ Component whitelisting (trusted catalog)")
    print("   ✓ Content sanitization")
    print("   ✓ Full audit trail (every UI state logged)")

    print(f"\n📈 Benefits:")
    print("   ✓ Explainability: Agent shows reasoning visually")
    print("   ✓ Efficiency: No chat back-and-forth")
    print("   ✓ Trust: Confidence scores + uncertainty disclosure")
    print("   ✓ Compliance: GDPR Article 22 ready")

    print(f"\n📚 Next Steps:")
    print("   1. ✅ You've built A2UI schemas (this demo)")
    print("   2. 📤 Publish to Pub/Sub (gatra_a2ui/publisher.py)")
    print("   3. 🖥️  Build frontend renderer (React/Flutter)")
    print("   4. 🔄 Integrate with agents (see examples/)")

    print(f"\n📁 Files Created:")
    print("   • docs/A2UI_GATRA_SOC_IMPLEMENTATION.md (12,000+ words)")
    print("   • gatra_a2ui/ (Python package)")
    print("   • examples/taa_a2ui_example.py")
    print("   • A2UI_QUICKSTART.md (15-min tutorial)")
    print("   • A2UI_IMPLEMENTATION_SUMMARY.md")

    print("\n" + "🎉"*35)
    print("Welcome to agent-native SOC interfaces!")
    print("Your GATRA platform is now A2UI-ready!")
    print("🎉"*35 + "\n")

    return {
        "situation_brief_schema": brief_schema,
        "attack_narrative_schema": narrative_schema,
        "total_components": total_components
    }


if __name__ == "__main__":
    try:
        result = main()
        print(f"✅ Demo completed successfully!")
        print(f"   Generated {result['total_components']} A2UI components\n")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
