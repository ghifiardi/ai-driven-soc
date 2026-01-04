# How to Use A2UI in GATRA SOC - Practical Guide

## Quick Start (5 Minutes)

### Step 1: Run the Demo

```bash
cd /Users/raditio.ghifiardigmail.com/Downloads/ai-driven-soc.backup

# Run the demo
python3 run_a2ui_demo.py
```

You'll see:
- ✅ Situation Brief screen generated
- ✅ Attack Narrative screen generated
- ✅ JSON schemas displayed
- ✅ Visual mockups of what analysts see

---

## How to Integrate A2UI into Your Agents

### Option 1: Quick Integration (Copy-Paste Ready)

Add this to your existing TAA agent:

```python
# At the top of your taa_langgraph_agent.py
import sys
sys.path.insert(0, '/Users/raditio.ghifiardigmail.com/Downloads/ai-driven-soc.backup')

from run_a2ui_demo import (
    build_attack_narrative_schema,
    build_situation_brief_schema
)

# Inside your TAA analysis function
def analyze_alert(alert_data):
    # Your existing analysis
    analysis_result = perform_analysis(alert_data)

    # NEW: Generate A2UI
    a2ui_schema = build_attack_narrative_schema(
        incident_id=alert_data['alert_id'],
        attack_chain=analysis_result['attack_chain']
    )

    # Print or publish the schema
    print(json.dumps(a2ui_schema, indent=2))

    return analysis_result, a2ui_schema
```

### Option 2: Full Integration (Production Ready)

```python
# 1. Import the builders
import sys
sys.path.insert(0, '/Users/raditio.ghifiardigmail.com/Downloads/ai-driven-soc.backup')

from run_a2ui_demo import (
    build_metric_card,
    build_timeline_step,
    build_text,
    build_button
)

# 2. Create your custom screen
def build_custom_soc_screen(incident_data):
    components = []

    # Add header
    components.append(build_text(
        "header",
        f"Incident Analysis: {incident_data['id']}",
        "heading1"
    ))

    # Add metrics from your data
    components.append(build_metric_card(
        "severity",
        "Severity",
        incident_data['severity'].upper(),
        severity=incident_data['severity']
    ))

    # Add timeline steps
    for i, step in enumerate(incident_data['attack_steps'], 1):
        components.append(build_timeline_step(
            f"step_{i}",
            step_number=i,
            title=step['title'],
            description=step['description'],
            mitre_id=step.get('mitre_id'),
            mitre_name=step.get('mitre_name'),
            confidence=step.get('confidence'),
            evidence=step.get('evidence', [])
        ))

    # Add action button
    components.append(build_button(
        "investigate",
        "Start Investigation",
        "open_investigation",
        {"incident_id": incident_data['id']}
    ))

    return {"surfaceUpdate": {"components": components}}

# 3. Use it
incident = {
    "id": "INC-2025-002",
    "severity": "high",
    "attack_steps": [
        {
            "title": "Brute Force Attack",
            "description": "Multiple failed SSH login attempts",
            "mitre_id": "T1110.001",
            "mitre_name": "Password Guessing",
            "confidence": 0.95,
            "evidence": ["ssh_log_001", "auth_log_002"]
        }
    ]
}

schema = build_custom_soc_screen(incident)
print(json.dumps(schema, indent=2))
```

---

## Practical Use Cases

### Use Case 1: Real-Time Incident Dashboard

```python
# In your ADA agent (Anomaly Detection)
from run_a2ui_demo import build_metric_card, build_text

def generate_situation_brief(current_incidents):
    components = []

    # Header
    components.append(build_text("header", "Live SOC Dashboard", "heading1"))

    # Active incidents
    components.append(build_metric_card(
        "active",
        "Active Incidents",
        str(len(current_incidents)),
        severity="high" if len(current_incidents) > 5 else "medium",
        trend="up" if len(current_incidents) > len(previous_incidents) else "down"
    ))

    # Critical threats
    critical = [i for i in current_incidents if i['severity'] == 'critical']
    components.append(build_metric_card(
        "critical",
        "Critical Threats",
        str(len(critical)),
        severity="critical" if critical else "low"
    ))

    return {"surfaceUpdate": {"components": components}}
```

### Use Case 2: Threat Investigation Timeline

```python
# In your TAA agent (Triage & Analysis)
from run_a2ui_demo import build_timeline_step, build_text

def generate_investigation_timeline(incident_id, events):
    components = []

    components.append(build_text(
        "header",
        f"Investigation Timeline: {incident_id}",
        "heading2"
    ))

    # Sort events chronologically
    sorted_events = sorted(events, key=lambda x: x['timestamp'])

    # Generate timeline steps
    for i, event in enumerate(sorted_events, 1):
        components.append(build_timeline_step(
            f"event_{i}",
            step_number=i,
            title=event['event_type'],
            description=f"{event['description']} at {event['timestamp']}",
            mitre_id=event.get('mitre_technique'),
            mitre_name=event.get('technique_name'),
            confidence=event.get('confidence', 0.8),
            evidence=[event['log_id']]
        ))

    return {"surfaceUpdate": {"components": components}}

# Example usage
events = [
    {
        "timestamp": "2025-01-03T10:15:00Z",
        "event_type": "Suspicious Login",
        "description": "Admin login from unusual location",
        "mitre_technique": "T1078.004",
        "technique_name": "Valid Accounts: Cloud Accounts",
        "confidence": 0.87,
        "log_id": "cloudtrail_001"
    },
    {
        "timestamp": "2025-01-03T10:20:00Z",
        "event_type": "Data Exfiltration",
        "description": "Large data transfer to external S3 bucket",
        "mitre_technique": "T1537",
        "technique_name": "Transfer Data to Cloud Account",
        "confidence": 0.92,
        "log_id": "s3_access_002"
    }
]

schema = generate_investigation_timeline("INC-2025-003", events)
```

### Use Case 3: Containment Decision Panel

```python
# In your CRA agent (Containment & Response)
from run_a2ui_demo import build_text, build_button

def generate_containment_options(incident_id, recommended_actions):
    components = []

    components.append(build_text(
        "header",
        "Containment Recommendations",
        "heading2"
    ))

    components.append(build_text(
        "description",
        "Select actions to contain this threat:",
        "body"
    ))

    # Action buttons
    for i, action in enumerate(recommended_actions):
        components.append(build_button(
            f"action_{i}",
            action['label'],
            "execute_containment",
            {
                "action_type": action['type'],
                "incident_id": incident_id,
                "risk_level": action['risk']
            }
        ))

    return {"surfaceUpdate": {"components": components}}

# Example usage
actions = [
    {"type": "isolate_host", "label": "Isolate Compromised Host", "risk": "medium"},
    {"type": "block_ip", "label": "Block Malicious IP", "risk": "low"},
    {"type": "reset_credentials", "label": "Reset User Credentials", "risk": "high"}
]

schema = generate_containment_options("INC-2025-004", actions)
```

---

## Testing Your A2UI Screens

### Test 1: Validate JSON Schema

```python
import json

# Your schema
schema = build_custom_soc_screen(your_data)

# Validate it's valid JSON
try:
    json_str = json.dumps(schema, indent=2)
    print("✅ Valid JSON schema generated!")
    print(json_str)
except Exception as e:
    print(f"❌ Invalid JSON: {e}")
```

### Test 2: Component Count

```python
schema = build_custom_soc_screen(your_data)

component_count = len(schema['surfaceUpdate']['components'])
print(f"Generated {component_count} components")

# List component types
for comp in schema['surfaceUpdate']['components']:
    comp_type = list(comp['component'].keys())[0]
    print(f"  - {comp['id']}: {comp_type}")
```

### Test 3: Save to File

```python
import json

schema = build_custom_soc_screen(your_data)

# Save for frontend testing
with open('test_a2ui_schema.json', 'w') as f:
    json.dump(schema, f, indent=2)

print("✅ Schema saved to test_a2ui_schema.json")
print("   You can now use this in your frontend renderer")
```

---

## Common Patterns

### Pattern 1: Multi-Step Workflow

```python
def build_phishing_investigation(email_data):
    components = []

    # Step 1: Email analysis
    components.append(build_text("step1_header", "1. Email Analysis", "heading3"))
    components.append(build_metric_card(
        "sender_risk",
        "Sender Risk Score",
        f"{email_data['sender_risk'] * 100:.0f}%",
        severity="high" if email_data['sender_risk'] > 0.7 else "medium"
    ))

    # Step 2: Attachment scan
    components.append(build_text("step2_header", "2. Attachment Scan", "heading3"))
    if email_data['attachment_malicious']:
        components.append(build_metric_card(
            "attachment",
            "Attachment Status",
            "MALICIOUS",
            severity="critical"
        ))

    # Step 3: Recommended action
    components.append(build_text("step3_header", "3. Recommended Action", "heading3"))
    components.append(build_button(
        "quarantine",
        "Quarantine Email",
        "quarantine_email",
        {"email_id": email_data['id']}
    ))

    return {"surfaceUpdate": {"components": components}}
```

### Pattern 2: Dynamic Alert List

```python
def build_alert_feed(alerts):
    components = []

    components.append(build_text("header", "Recent Alerts", "heading2"))

    for i, alert in enumerate(alerts[:10]):  # Show last 10
        severity_map = {
            "critical": "🔴",
            "high": "🟠",
            "medium": "🟡",
            "low": "🟢"
        }

        icon = severity_map.get(alert['severity'], "⚪")
        components.append(build_text(
            f"alert_{i}",
            f"{icon} {alert['title']} - {alert['timestamp']}",
            "body"
        ))

    return {"surfaceUpdate": {"components": components}}
```

### Pattern 3: Confidence-Based Display

```python
def build_ai_analysis(analysis_result):
    components = []

    # Show confidence level
    confidence = analysis_result['confidence']

    if confidence >= 0.9:
        components.append(build_text(
            "confidence_high",
            "✅ High Confidence Analysis",
            "heading3"
        ))
    elif confidence >= 0.7:
        components.append(build_text(
            "confidence_medium",
            "⚠️ Medium Confidence - Review Recommended",
            "heading3"
        ))
    else:
        components.append(build_text(
            "confidence_low",
            "❓ Low Confidence - Manual Investigation Required",
            "heading3"
        ))

    # Add the analysis
    components.append(build_text(
        "analysis",
        analysis_result['description'],
        "body"
    ))

    return {"surfaceUpdate": {"components": components}}
```

---

## Debugging Tips

### Tip 1: Pretty Print JSON

```python
import json

schema = your_function()
print(json.dumps(schema, indent=2, sort_keys=True))
```

### Tip 2: Validate Component IDs

```python
schema = your_function()
ids = [comp['id'] for comp in schema['surfaceUpdate']['components']]

# Check for duplicates
if len(ids) != len(set(ids)):
    print("❌ Duplicate component IDs found!")
    duplicates = [id for id in ids if ids.count(id) > 1]
    print(f"   Duplicates: {set(duplicates)}")
else:
    print("✅ All component IDs are unique")
```

### Tip 3: Component Type Summary

```python
from collections import Counter

schema = your_function()
types = [list(comp['component'].keys())[0]
         for comp in schema['surfaceUpdate']['components']]

type_counts = Counter(types)
print("Component breakdown:")
for comp_type, count in type_counts.items():
    print(f"  {comp_type}: {count}")
```

---

## Next Steps

### 1. Run More Examples

```bash
# Basic demo
python3 run_a2ui_demo.py

# Interactive demo (if you want pauses)
python3 standalone_a2ui_demo.py
```

### 2. Modify the Demo

Edit `run_a2ui_demo.py` to change:
- Metric values
- Alert messages
- Timeline steps
- Button labels

### 3. Create Your Own Screen

```python
# Copy this template
from run_a2ui_demo import *

def my_custom_screen():
    components = []

    # Your components here
    components.append(build_text("header", "My Screen", "heading1"))
    # ... add more ...

    return {"surfaceUpdate": {"components": components}}

# Generate and print
schema = my_custom_screen()
print(json.dumps(schema, indent=2))
```

### 4. Integrate with Real Data

```python
# Example: Use real BigQuery data
from google.cloud import bigquery

client = bigquery.Client()
query = "SELECT * FROM incidents WHERE status = 'active' LIMIT 10"
results = client.query(query).result()

# Convert to A2UI
components = []
for row in results:
    components.append(build_metric_card(
        f"incident_{row.id}",
        f"Incident {row.id}",
        row.severity,
        severity=row.severity
    ))

schema = {"surfaceUpdate": {"components": components}}
```

---

## Getting Help

### Check the Documentation

1. **Full Guide**: `docs/A2UI_GATRA_SOC_IMPLEMENTATION.md`
2. **Quick Start**: `A2UI_QUICKSTART.md`
3. **Demo Results**: `A2UI_DEMO_RESULTS.md`

### Example Files

1. **run_a2ui_demo.py** - Auto-running demo (best for learning)
2. **standalone_a2ui_demo.py** - Interactive version
3. **examples/taa_a2ui_example.py** - Full TAA integration

### External Resources

- [Google A2UI Docs](https://a2ui.org/)
- [A2UI GitHub](https://github.com/google/A2UI)

---

## Summary

**To use A2UI in GATRA:**

1. ✅ Run the demo: `python3 run_a2ui_demo.py`
2. ✅ Understand the JSON schema structure
3. ✅ Use the builder functions (`build_*`)
4. ✅ Generate schemas from your agent data
5. ✅ Test with `json.dumps(schema, indent=2)`
6. 📤 Publish to Pub/Sub (next step)
7. 🖥️ Build frontend renderer (next step)

**You're ready to build agent-native SOC interfaces!**
