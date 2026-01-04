# Nextron-Style Threat Hunting Agents

A comprehensive threat hunting platform inspired by Nextron Systems (THOR/ASGARD/VALHALLA), integrated with the AI-Driven SOC ecosystem.

## Overview

This implementation provides three complementary agents that work together to deliver enterprise-grade threat hunting capabilities:

### 🔍 **THOR Endpoint Agent** (`thor_endpoint_agent.py`)
Endpoint scanning and threat detection agent with:
- **YARA rule scanning** (filesystem, memory, processes)
- **IOC matching** (IPs, domains, file hashes, filenames)
- **Sigma rule evaluation** for log events
- **Behavioral analysis** (suspicious processes, network patterns)
- **Multi-format scanning** (files, memory, network connections)

### 🎯 **ASGARD Orchestration Agent** (`asgard_orchestration_agent.py`)
Fleet-wide campaign management and orchestration:
- **Campaign management** (create, schedule, monitor threat hunts)
- **Target selection** (all endpoints, groups, labels, regex patterns)
- **Scan distribution** via Pub/Sub to THOR agents
- **Real-time monitoring** and progress tracking
- **Automated reporting** and analytics
- **Multi-cloud endpoint discovery** (GCP, AWS, Azure)

### 🗄️ **VALHALLA Feed Manager** (`valhalla_feed_manager.py`)
Threat intelligence feed aggregation and distribution:
- **IOC aggregation** from multiple sources (ThreatFox, MalwareBazaar, etc.)
- **YARA rule repository** with versioning and quality scoring
- **Sigma rule management** for log analysis
- **Automated feed updates** and synchronization
- **Rule validation** and compilation
- **Custom rule creation** and testing

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    VALHALLA Feed Manager                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │  ThreatFox  │  │MalwareBazaar│  │  Emerging   │            │
│  │     IOCs    │  │    Hashes   │  │   Threats   │  + Custom  │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘            │
│         │                 │                 │                    │
│         └─────────────────┴─────────────────┘                    │
│                           │                                      │
│              IOC/YARA Rule Distribution                          │
│                           │                                      │
└───────────────────────────┼──────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ASGARD Orchestration                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Campaign: "Ransomware Hunt Q4 2025"                     │  │
│  │  Targets: 500 production servers                          │  │
│  │  Scans: filesystem + process + network                    │  │
│  │  Priority: HIGH                                            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           │                                      │
│              Scan Task Distribution (Pub/Sub)                   │
│                           │                                      │
└───────────────────────────┼──────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  THOR Endpoint Agents (Fleet)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Endpoint 1  │  │  Endpoint 2  │  │  Endpoint N  │          │
│  │   YARA Scan  │  │   YARA Scan  │  │   YARA Scan  │          │
│  │   IOC Match  │  │   IOC Match  │  │   IOC Match  │          │
│  │  Behavioral  │  │  Behavioral  │  │  Behavioral  │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         └──────────────────┴──────────────────┘                  │
│                           │                                      │
│                    Findings (Pub/Sub)                            │
│                           │                                      │
└───────────────────────────┼──────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              AI-Driven SOC Agent Ecosystem                      │
│                                                                  │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐                │
│  │   TAA    │────▶│   CRA    │────▶│   CLA    │                │
│  │  Triage  │     │ Response │     │ Learning │                │
│  └──────────┘     └──────────┘     └──────────┘                │
│       │                 │                 │                      │
│   Enrichment      Containment        Feedback                   │
│   & Analysis      & Remediation      & Training                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Integration with Existing SOC Agents

### 1. **Triage and Analysis Agent (TAA) Integration**

THOR findings are automatically published to TAA for enrichment:

```python
# In thor_endpoint_agent.py
def _publish_findings(self, scan_result: ScanResult):
    """Publish to 'thor-findings' topic → TAA subscribes"""
    message_data = {
        "scan_id": scan_result.scan_id,
        "hostname": scan_result.hostname,
        "matches": [m.to_dict() for m in scan_result.matches],
        "source": "thor_endpoint_agent"
    }
    # TAA enriches with threat intel (VirusTotal, AbuseIPDB, etc.)
```

**TAA Actions:**
- Enriches IOC matches with external threat intelligence
- Performs LLM-based analysis of behavioral detections
- Prioritizes findings by severity and confidence
- Flags false positives for review

### 2. **Containment and Response Agent (CRA) Integration**

Critical threats trigger automated response playbooks:

```python
# TAA → CRA workflow
if threat_severity == "CRITICAL" and confidence > 0.9:
    cra.execute_playbook(
        playbook="isolate_endpoint_and_block_ip",
        targets=[endpoint_id],
        context={"scan_id": scan_id, "ioc": malicious_ip}
    )
```

**CRA Actions:**
- Isolate compromised endpoints
- Block malicious IPs at firewall
- Reset compromised credentials
- Create incident tickets
- Trigger forensic data collection

### 3. **Continuous Learning Agent (CLA) Integration**

Feed detection data back to improve ML models:

```python
# THOR → CLA feedback loop
cla.ingest_detection_feedback({
    "detection_type": "yara",
    "rule_name": "Ransomware_Lockbit",
    "outcome": "true_positive",
    "endpoint_context": {...},
    "analyst_feedback": "Confirmed Lockbit ransomware"
})
```

**CLA Actions:**
- Learn from YARA/IOC detection patterns
- Improve behavioral anomaly detection models
- Identify emerging threat patterns
- Recommend new YARA rules to VALHALLA

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_threat_hunting.txt
```

### 2. Configure GCP Project

Edit configuration files with your GCP project ID:

```bash
# config/thor_config.json
{
  "gcp_project_id": "your-gcp-project-id",
  ...
}

# config/asgard_config.json
{
  "gcp_project_id": "your-gcp-project-id",
  ...
}

# config/valhalla_config.json
{
  "gcp_project_id": "your-gcp-project-id",
  ...
}
```

### 3. Create GCP Resources

```bash
# Create Pub/Sub topics
gcloud pubsub topics create thor-scan-requests
gcloud pubsub topics create thor-findings
gcloud pubsub topics create asgard-campaigns
gcloud pubsub topics create asgard-scan-tasks
gcloud pubsub topics create valhalla-rule-updates
gcloud pubsub topics create valhalla-ioc-updates

# Create Firestore collections
# (Auto-created on first use)

# Create BigQuery tables
bq mk --dataset soc_data
bq mk --table soc_data.thor_scan_results \
  scan_id:STRING,hostname:STRING,start_time:TIMESTAMP,end_time:TIMESTAMP,...

bq mk --table soc_data.asgard_campaign_reports \
  campaign_id:STRING,campaign_name:STRING,created_at:TIMESTAMP,...

# Create GCS bucket for VALHALLA
gsutil mb gs://valhalla-threat-intel
```

### 4. Initialize VALHALLA Feed Manager

```bash
# Fetch initial threat intelligence feeds
python valhalla_feed_manager.py
```

This will:
- Fetch IOCs from ThreatFox and MalwareBazaar
- Download YARA rules from Emerging Threats
- Store rules in Firestore and GCS
- Export IOC set for THOR agents

### 5. Deploy THOR Agent on Endpoints

```bash
# On each endpoint to be monitored
sudo python thor_endpoint_agent.py \
  --config config/thor_config.json \
  --scan-type full \
  --load-yara \
  --load-iocs
```

Or run as a daemon to respond to ASGARD scan requests:

```bash
# Listen for scan tasks from ASGARD
python -c "
from thor_endpoint_agent import THOREndpointAgent
from google.cloud import pubsub_v1

agent = THOREndpointAgent()
agent.load_yara_rules()
agent.load_iocs()

subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path('your-project', 'thor-scan-requests-sub')

def callback(message):
    scan_request = json.loads(message.data)
    result = agent.perform_scan(...)
    message.ack()

subscriber.subscribe(subscription_path, callback=callback)
"
```

### 6. Launch Threat Hunting Campaign via ASGARD

```python
from asgard_orchestration_agent import ASGARDOrchestrationAgent, ScanPriority, TargetSelectionMode

# Initialize ASGARD
asgard = ASGARDOrchestrationAgent()

# Create campaign
campaign = asgard.create_campaign(
    name="Q1 2025 Ransomware Hunt",
    description="Hunt for ransomware indicators across production fleet",
    target_selection_mode=TargetSelectionMode.LABEL,
    target_criteria={"labels": {"env": "production"}},
    scan_types=["filesystem", "process", "network"],
    yara_rule_sets=["ransomware", "crypto_malware"],
    ioc_feeds=["valhalla", "threatfox"],
    priority=ScanPriority.HIGH,
    schedule_type="immediate"  # Or "scheduled" with a time
)

print(f"Campaign created: {campaign.campaign_id}")
print(f"Targeting {campaign.total_targets} endpoints")
```

---

## Use Cases

### 1. **Ransomware Threat Hunt**

Proactively hunt for ransomware across your fleet:

```python
# ASGARD campaign
campaign = asgard.create_campaign(
    name="Lockbit 3.0 Hunt",
    description="Hunt for Lockbit 3.0 ransomware indicators",
    target_selection_mode=TargetSelectionMode.ALL,
    scan_types=["filesystem", "process", "network"],
    yara_rule_sets=["ransomware"],
    priority=ScanPriority.CRITICAL
)

# VALHALLA ensures latest Lockbit YARA rules are distributed
# THOR scans all endpoints with Lockbit signatures
# TAA enriches findings and classifies severity
# CRA automatically isolates infected systems
```

### 2. **APT Detection Campaign**

Hunt for Advanced Persistent Threats:

```python
campaign = asgard.create_campaign(
    name="APT29 Indicators Hunt",
    description="Search for APT29 (Cozy Bear) TTPs",
    target_selection_mode=TargetSelectionMode.LABEL,
    target_criteria={"labels": {"criticality": "high"}},
    scan_types=["filesystem", "process", "network", "registry"],
    yara_rule_sets=["apt", "apt29", "backdoor"],
    ioc_feeds=["valhalla", "misp"],
    priority=ScanPriority.HIGH
)
```

### 3. **Webshell Detection**

Hunt for webshells on web servers:

```python
campaign = asgard.create_campaign(
    name="Webshell Hunt - Web Servers",
    description="Detect webshells on public-facing web servers",
    target_selection_mode=TargetSelectionMode.LABEL,
    target_criteria={"labels": {"role": "webserver"}},
    scan_types=["filesystem"],
    yara_rule_sets=["webshell"],
    priority=ScanPriority.HIGH,
    # Only scan web directories
    custom_scan_paths=["/var/www", "/usr/share/nginx", "/opt/app"]
)
```

### 4. **Cryptocurrency Miner Hunt**

Detect unauthorized cryptocurrency mining:

```python
campaign = asgard.create_campaign(
    name="Cryptominer Detection",
    description="Detect cryptomining malware and activity",
    target_selection_mode=TargetSelectionMode.ALL,
    scan_types=["process", "network"],
    yara_rule_sets=["cryptominer"],
    behavioral_checks=["high_cpu_usage", "mining_pool_connections"],
    priority=ScanPriority.MEDIUM
)
```

### 5. **Scheduled Recurring Hunt**

Set up weekly threat hunts:

```python
campaign = asgard.create_campaign(
    name="Weekly Security Sweep",
    description="Comprehensive weekly threat hunt",
    target_selection_mode=TargetSelectionMode.ALL,
    scan_types=["filesystem", "process", "network"],
    yara_rule_sets=["ransomware", "trojan", "backdoor"],
    ioc_feeds=["valhalla", "threatfox"],
    priority=ScanPriority.MEDIUM,
    schedule_type="recurring",
    recurrence_pattern="0 2 * * 0"  # Every Sunday at 2 AM
)
```

---

## Advanced Features

### Custom YARA Rule Creation

Add your own custom YARA rules to VALHALLA:

```python
from valhalla_feed_manager import VALHALLAFeedManager, YARARule, RuleQuality, ThreatCategory, FeedSource
import hashlib

valhalla = VALHALLAFeedManager()

# Create custom rule
custom_rule_content = """
rule Custom_Backdoor_Detection {
    meta:
        description = "Detects custom backdoor pattern"
        author = "SOC Team"
        severity = "high"
    strings:
        $s1 = "backdoor_signature_1" ascii
        $s2 = {6A 40 68 00 30 00 00}
    condition:
        any of them
}
"""

rule = YARARule(
    rule_id=f"custom_{hashlib.md5(custom_rule_content.encode()).hexdigest()[:16]}",
    rule_name="Custom_Backdoor_Detection",
    rule_content=custom_rule_content,
    description="Custom backdoor detection rule",
    author="SOC Team",
    reference="Internal investigation CASE-2025-001",
    created_at=datetime.utcnow().isoformat(),
    updated_at=datetime.utcnow().isoformat(),
    version="1.0",
    quality=RuleQuality.EXPERIMENTAL,
    categories=[ThreatCategory.BACKDOOR],
    tags=["custom", "backdoor"],
    severity="high",
    false_positive_rate=0.1,
    detection_rate=0.8,
    source=FeedSource.CUSTOM,
    is_compiled=False,
    compiled_path=None,
    sha256_hash=hashlib.sha256(custom_rule_content.encode()).hexdigest()
)

# Add to VALHALLA
valhalla.add_yara_rule(rule, validate=True)

# Rule is now available to all THOR agents
```

### IOC Whitelisting

Reduce false positives by whitelisting known-good IOCs:

```python
# In valhalla_config.json
{
  "ioc_management": {
    "whitelisting": {
      "enabled": true,
      "whitelist_path": "/opt/valhalla/whitelist.json",
      "auto_whitelist_fp": true
    }
  }
}

# whitelist.json
{
  "ips": ["8.8.8.8", "1.1.1.1"],  # Google DNS, Cloudflare DNS
  "domains": ["microsoft.com", "apple.com"],
  "file_hashes": {
    "sha256_of_legitimate_tool": "IT admin tool"
  }
}
```

### Performance Tuning

Optimize THOR scans for large endpoints:

```python
# In thor_config.json
{
  "scan_limits": {
    "max_files_per_scan": 1000000,
    "timeout_minutes": 120,
    "max_memory_mb": 4096
  },
  "performance": {
    "max_parallel_file_scans": 10,
    "io_nice_priority": 3,
    "cpu_nice_priority": 10,
    "enable_incremental_scans": true  # Only scan changed files
  }
}
```

---

## Monitoring and Metrics

### Campaign Status Dashboard

Monitor campaign progress:

```python
# Get campaign status
status = asgard.get_campaign_status(campaign_id)

print(f"""
Campaign: {status['name']}
Status: {status['status']}
Progress: {status['progress']['scanned']}/{status['progress']['total']}
Threats Found: {status['threats']['total']} (Critical: {status['threats']['critical']})
""")
```

### BigQuery Analytics

Query historical threat hunting data:

```sql
-- Top threats detected in last 30 days
SELECT
  rule_name,
  COUNT(*) as detection_count,
  AVG(confidence) as avg_confidence
FROM `soc_data.thor_scan_results`,
  UNNEST(JSON_EXTRACT_ARRAY(matches)) as match
WHERE
  DATE(start_time) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
GROUP BY rule_name
ORDER BY detection_count DESC
LIMIT 10;

-- Campaign effectiveness
SELECT
  campaign_name,
  total_targets,
  successfully_scanned,
  total_threats,
  critical_threats,
  ROUND(total_threats / successfully_scanned, 2) as threats_per_endpoint
FROM `soc_data.asgard_campaign_reports`
WHERE DATE(created_at) >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY)
ORDER BY created_at DESC;
```

---

## Comparison with Nextron Systems

| Feature | Nextron (THOR/ASGARD/VALHALLA) | This Implementation |
|---------|--------------------------------|---------------------|
| **YARA Scanning** | ✅ Commercial-grade rules | ✅ Open-source + custom |
| **IOC Matching** | ✅ Multi-source feeds | ✅ ThreatFox, MalwareBazaar, etc. |
| **Fleet Orchestration** | ✅ ASGARD platform | ✅ ASGARD-like orchestration |
| **Memory Scanning** | ✅ Advanced | ✅ Via yara-python |
| **Sigma Rules** | ✅ Integrated | ✅ Planned/experimental |
| **Cloud-Native** | ❌ On-prem focused | ✅ GCP-native (Pub/Sub, Firestore) |
| **AI Integration** | ❌ Limited | ✅ Full SOC agent integration |
| **Cost** | 💰 Commercial license | 💰 Open-source + cloud costs |
| **Threat Intel** | ✅ VALHALLA premium | ✅ Free feeds + custom |

---

## Troubleshooting

### THOR Agent Not Finding Threats

1. **Verify YARA rules loaded:**
   ```python
   agent = THOREndpointAgent()
   agent.load_yara_rules()
   print(f"Loaded rules: {agent.compiled_rules}")
   ```

2. **Check IOC freshness:**
   ```python
   valhalla = VALHALLAFeedManager()
   valhalla.update_all_feeds()
   ```

3. **Test rule syntax:**
   ```bash
   yara -c /opt/thor/yara_rules/test_rule.yar /path/to/test/file
   ```

### ASGARD Campaign Stuck

1. **Check Pub/Sub subscriptions:**
   ```bash
   gcloud pubsub subscriptions list
   gcloud pubsub subscriptions pull thor-scan-requests-sub --limit 10
   ```

2. **Verify endpoint registration:**
   ```python
   asgard = ASGARDOrchestrationAgent()
   print(f"Registered endpoints: {len(asgard.registered_endpoints)}")
   ```

3. **Review Firestore campaign status:**
   ```python
   campaign_doc = firestore_client.collection('campaigns').document(campaign_id).get()
   print(campaign_doc.to_dict())
   ```

### High False Positive Rate

1. **Adjust rule quality threshold:**
   ```json
   // valhalla_config.json
   {
     "yara": {
       "min_quality": "high"  // Change from "medium"
     }
   }
   ```

2. **Enable IOC whitelisting:**
   ```json
   {
     "ioc_management": {
       "whitelisting": {
         "enabled": true
       }
     }
   }
   ```

3. **Review and tune behavioral patterns:**
   ```json
   // thor_config.json - Remove overly broad patterns
   {
     "behavioral_analysis": {
       "suspicious_process_patterns": [
         "powershell.*-encodedcommand"  // Keep specific
         // Remove: ".*python.*" (too broad)
       ]
     }
   }
   ```

---

## Security Considerations

### Principle of Least Privilege

- Run THOR agents with minimal required permissions
- ASGARD should have read-only access to endpoints for discovery
- VALHALLA API keys stored in Secret Manager, not config files

### Data Privacy

- Scan results may contain sensitive file paths and process information
- Enable encryption for scan results in transit and at rest
- Configure data retention policies in BigQuery

### Audit Logging

- All campaigns are logged in Firestore with creator and timestamp
- THOR scan results include full audit trail
- CRA actions require approval workflow for high-impact operations

---

## Contributing

To add new threat intelligence sources:

1. Implement feed connector in `valhalla_feed_manager.py`:
   ```python
   def fetch_iocs_new_source(self) -> List[IOCEntry]:
       # Fetch from new API
       # Parse and return IOCEntry objects
   ```

2. Add configuration in `config/valhalla_config.json`

3. Update feed scheduler in `update_all_feeds()`

---

## License

This implementation is provided as part of the AI-Driven SOC platform. See main LICENSE file.

---

## References

- Nextron Systems: https://www.nextron-systems.com/
- YARA Documentation: https://yara.readthedocs.io/
- Sigma Rules: https://github.com/SigmaHQ/sigma
- ThreatFox API: https://threatfox.abuse.ch/
- MalwareBazaar API: https://bazaar.abuse.ch/

---

## Support

For issues, questions, or feature requests, please see the main project repository.
