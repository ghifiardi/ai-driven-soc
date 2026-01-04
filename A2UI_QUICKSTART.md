# GATRA A2UI Quick Start Guide

Get started with Agent-to-User Interface (A2UI) in your GATRA SOC in 15 minutes.

## What You'll Build

A TAA agent that generates an interactive Attack Narrative screen instead of text-only analysis.

**Before A2UI:**
```
TAA: "I detected a phishing attack. Initial access via email.
     Execution of malicious macro. Lateral movement detected..."
Analyst: "Can you show me the timeline?"
TAA: "Sure, step 1 was..."
```

**After A2UI:**
```
[Interactive timeline appears with:]
- 4-step attack progression
- MITRE ATT&CK badges
- Confidence indicators
- Evidence cards
- "Request More Evidence" button
```

---

## Prerequisites

1. Python 3.9+
2. GCP project with Pub/Sub and BigQuery
3. Existing GATRA installation

---

## Step 1: Install Dependencies

```bash
# Install A2UI package
pip install google-cloud-pubsub google-cloud-bigquery

# No additional dependencies needed - uses standard library
```

---

## Step 2: Create GCP Resources

### Create Pub/Sub Topic

```bash
gcloud pubsub topics create gatra-a2ui-updates

# Create subscription for frontend
gcloud pubsub subscriptions create gatra-a2ui-updates-sub \
  --topic=gatra-a2ui-updates
```

### Create BigQuery Audit Table

```bash
# Create dataset
bq mk --dataset gatra_audit

# Create audit table
bq mk --table gatra_audit.a2ui_states \
  timestamp:TIMESTAMP,\
  agent_id:STRING,\
  screen_type:STRING,\
  incident_id:STRING,\
  schema_hash:STRING,\
  schema:STRING,\
  priority:STRING,\
  metadata:STRING
```

---

## Step 3: Your First A2UI Screen

Create `my_first_a2ui.py`:

```python
from gatra_a2ui import AttackNarrativeBuilder, A2UIPublisher

# Initialize publisher
publisher = A2UIPublisher(project_id="YOUR-PROJECT-ID")

# Build attack narrative
builder = AttackNarrativeBuilder(incident_id="INC-2025-001")

builder.add_step(
    step=1,
    title="Initial Access",
    description="Phishing email delivered to user@company.com",
    mitre_technique="T1566.001",
    mitre_name="Spearphishing Attachment",
    mitre_tactic="Initial Access",
    confidence=0.92,
    evidence=["email_log_12345"]
)

builder.add_step(
    step=2,
    title="Execution",
    description="Malicious macro executed, dropped payload",
    mitre_technique="T1204.002",
    mitre_name="Malicious File",
    mitre_tactic="Execution",
    confidence=0.89,
    evidence=["process_create_67890"]
)

# Build schema
schema = builder.build()

# Publish to frontend
publisher.publish(
    schema=schema,
    screen_type="attack_narrative",
    agent_id="TAA",
    incident_id="INC-2025-001",
    priority="high"
)

print("A2UI Attack Narrative published!")
```

Run it:

```bash
python my_first_a2ui.py
```

---

## Step 4: Verify Publication

Check that the message was published:

```bash
# Pull message from Pub/Sub
gcloud pubsub subscriptions pull gatra-a2ui-updates-sub \
  --limit=1 \
  --format=json

# Should see A2UI schema JSON
```

Check audit trail:

```bash
# Query BigQuery audit log
bq query --use_legacy_sql=false '
SELECT
  timestamp,
  agent_id,
  screen_type,
  incident_id,
  LEFT(schema_hash, 16) as schema_hash_prefix
FROM `gatra_audit.a2ui_states`
ORDER BY timestamp DESC
LIMIT 5
'
```

---

## Step 5: Integrate with TAA Agent

Add A2UI to your existing TAA agent:

```python
# In taa_langgraph_agent.py

from gatra_a2ui import AttackNarrativeBuilder, A2UIPublisher

# Initialize in your agent
class TAAAgent:
    def __init__(self):
        # ... existing init ...
        self.a2ui_publisher = A2UIPublisher(project_id="YOUR-PROJECT-ID")

    def analyze_alert(self, alert):
        # Perform your analysis
        analysis_result = self._run_analysis(alert)

        # Build A2UI narrative
        builder = AttackNarrativeBuilder(incident_id=alert["alert_id"])

        for step in analysis_result["attack_chain"]:
            builder.add_step(
                step=step["number"],
                title=step["title"],
                description=step["description"],
                mitre_technique=step["mitre_id"],
                mitre_name=step["mitre_name"],
                confidence=step["confidence"],
                evidence=step["evidence_ids"]
            )

        # Publish UI
        self.a2ui_publisher.publish(
            schema=builder.build(),
            screen_type="attack_narrative",
            agent_id="TAA",
            incident_id=alert["alert_id"]
        )

        return analysis_result
```

---

## Step 6: Build Frontend Renderer

### Option A: React (Web)

```javascript
// Install A2UI React renderer
npm install @gatra/a2ui-react

// In your React component
import { A2UIRenderer } from '@gatra/a2ui-react';
import { gatraComponentCatalog } from './gatra-catalog';

function SOCDashboard() {
  const [schema, setSchema] = useState(null);

  useEffect(() => {
    // Subscribe to Pub/Sub (via WebSocket or polling)
    const subscription = pubsub.subscribe('gatra-a2ui-updates', (msg) => {
      setSchema(JSON.parse(msg.data));
    });

    return () => subscription.unsubscribe();
  }, []);

  return (
    <A2UIRenderer
      schema={schema}
      catalog={gatraComponentCatalog}
      onAction={handleUserAction}
    />
  );
}
```

### Option B: Flutter (Mobile)

```dart
// Install A2UI Flutter package
// pubspec.yaml: a2ui_flutter: ^1.0.0

import 'package:a2ui_flutter/a2ui_flutter.dart';

class SOCDashboard extends StatefulWidget {
  @override
  _SOCDashboardState createState() => _SOCDashboardState();
}

class _SOCDashboardState extends State<SOCDashboard> {
  Map<String, dynamic>? _schema;

  @override
  void initState() {
    super.initState();
    _subscribeToPubSub();
  }

  void _subscribeToPubSub() {
    // Subscribe to Pub/Sub
    pubsub.subscribe('gatra-a2ui-updates', (message) {
      setState(() {
        _schema = jsonDecode(message.data);
      });
    });
  }

  @override
  Widget build(BuildContext context) {
    return A2UIRenderer(
      schema: _schema,
      catalog: gatraComponentCatalog,
      onAction: _handleUserAction,
    );
  }
}
```

---

## Step 7: Test End-to-End

1. **Generate alert** (simulate or use real data)
2. **TAA processes alert** → generates A2UI
3. **A2UI published** to Pub/Sub
4. **Frontend receives** and renders
5. **Analyst sees** interactive timeline

Test flow:

```bash
# 1. Trigger alert (simulated)
python examples/taa_a2ui_example.py

# 2. Check Pub/Sub
gcloud pubsub subscriptions pull gatra-a2ui-updates-sub --limit=1

# 3. Frontend should render timeline
# (Open your React/Flutter app)

# 4. Verify audit trail
bq query --use_legacy_sql=false '
SELECT timestamp, agent_id, screen_type
FROM `gatra_audit.a2ui_states`
ORDER BY timestamp DESC LIMIT 1
'
```

---

## Next Steps

### 1. Add More Screen Types

```python
from gatra_a2ui import (
    SituationBriefBuilder,
    DecisionPanelBuilder,
    ConfidenceMonitorBuilder
)

# Situation Brief
brief = SituationBriefBuilder()
brief.add_metric_card("Active Incidents", value=7, severity="high")
brief.add_threat_actor_card("UNC3886", severity="critical")

# Decision Panel
decision = DecisionPanelBuilder(incident_id="INC-001")
decision.add_action(
    action_id="isolate",
    title="Isolate Host",
    description="Disconnect from network",
    risk=Severity.MEDIUM,
    impact="User downtime: 2-4 hours",
    recommended=True
)
decision.add_approval_workflow()

# Confidence Monitor
monitor = ConfidenceMonitorBuilder()
monitor.add_agent_gauge(
    agent_name="ADA",
    overall_confidence=0.94,
    data_coverage=0.98,
    model_drift=0.02,
    status=ComponentStatus.HEALTHY
)
```

### 2. Enable Streaming Updates

For long-running investigations:

```python
from gatra_a2ui import StreamingA2UIPublisher

publisher = StreamingA2UIPublisher(project_id="YOUR-PROJECT-ID")

with publisher.stream(screen_type="investigation", agent_id="TAA") as stream:
    # Agent starts analysis
    stream.add_component(header_component)

    # ... continues analyzing ...
    stream.add_component(timeline_step_1)

    # ... more analysis ...
    stream.add_component(timeline_step_2)

    # Finalize
    stream.finalize()

# Analyst sees UI build in real-time!
```

### 3. Add User Actions

Handle user interactions:

```python
# Frontend sends action back to agent
def handle_user_action(action):
    if action["name"] == "approve_containment":
        # Publish to CRA agent
        pubsub.publish("cra-actions", {
            "action": "isolate_host",
            "incident_id": action["params"]["incident_id"],
            "approved_by": current_user.id,
            "timestamp": datetime.utcnow().isoformat()
        })
```

### 4. Implement Component Catalog

Create your frontend component library:

```javascript
// gatra-catalog.js
export const gatraComponentCatalog = {
  TimelineStep: TimelineStepComponent,
  MetricCard: MetricCardComponent,
  MITREBadge: MITREBadgeComponent,
  RiskIndicator: RiskIndicatorComponent,
  // ... etc
};
```

---

## Troubleshooting

### A2UI not publishing

```bash
# Check service account permissions
gcloud projects add-iam-policy-binding YOUR-PROJECT-ID \
  --member="serviceAccount:YOUR-SA@YOUR-PROJECT-ID.iam.gserviceaccount.com" \
  --role="roles/pubsub.publisher"

# Check topic exists
gcloud pubsub topics list | grep gatra-a2ui-updates
```

### Audit logging fails

```bash
# Check BigQuery permissions
gcloud projects add-iam-policy-binding YOUR-PROJECT-ID \
  --member="serviceAccount:YOUR-SA@YOUR-PROJECT-ID.iam.gserviceaccount.com" \
  --role="roles/bigquery.dataEditor"

# Verify table schema
bq show --schema gatra_audit.a2ui_states
```

### Frontend not receiving updates

```bash
# Check subscription
gcloud pubsub subscriptions describe gatra-a2ui-updates-sub

# Test manually
gcloud pubsub subscriptions pull gatra-a2ui-updates-sub \
  --auto-ack \
  --limit=5
```

---

## Resources

- **Full Documentation**: [docs/A2UI_GATRA_SOC_IMPLEMENTATION.md](docs/A2UI_GATRA_SOC_IMPLEMENTATION.md)
- **Example Code**: [examples/taa_a2ui_example.py](examples/taa_a2ui_example.py)
- **Component Catalog**: [gatra_a2ui/catalog.py](gatra_a2ui/catalog.py)
- **A2UI Spec**: https://a2ui.org/
- **Google A2UI GitHub**: https://github.com/google/A2UI

---

## Getting Help

1. Check the [full documentation](docs/A2UI_GATRA_SOC_IMPLEMENTATION.md)
2. Review [example integrations](examples/)
3. Consult [A2UI official docs](https://a2ui.org/)
4. Open an issue in this repository

---

## What's Next?

You've successfully:
- ✅ Created your first A2UI screen
- ✅ Integrated with TAA agent
- ✅ Published to Pub/Sub
- ✅ Logged to audit trail

**Next milestones:**
1. Build frontend renderer (React/Flutter)
2. Add all 6 core SOC screens
3. Implement streaming updates
4. Enable human-in-the-loop workflows
5. Deploy to production

Welcome to agent-driven SOC interfaces!
