# Testing Threat Hunting Platform on GCP Development VM

## Quick Start Guide for Your GCP VM

Based on your screenshot showing `xdgaisocapp01` VM in GCP, here's how to test the threat hunting platform.

---

## ✅ Prerequisites Check

Your VM appears to have:
- ✅ SSH access (cloud shell)
- ✅ Python environment (Python-3.13.0 visible)
- ✅ GCP connectivity
- ✅ Existing AI-SOC codebase

---

## 🚀 Step 1: Upload Threat Hunting Files to VM

### Option A: Using the SSH Browser Upload Feature

1. **Click "UPLOAD FILE" button** in your SSH session (visible in screenshot)
2. **Upload these files** from your local machine:
   ```
   ai-driven-soc.backup/
   ├── thor_endpoint_agent.py
   ├── asgard_orchestration_agent.py
   ├── valhalla_feed_manager.py
   ├── threat_hunting_quickstart.py
   ├── requirements_threat_hunting.txt
   └── config/
       ├── thor_config.json
       ├── asgard_config.json
       └── valhalla_config.json
   ```

3. **Or upload as ZIP** and extract:
   ```bash
   # After uploading ai-driven-soc.backup.zip
   unzip ai-driven-soc.backup.zip
   cd ai-driven-soc.backup
   ```

### Option B: Using Git (if you have a repository)

```bash
# Clone your repository
git clone https://github.com/your-org/ai-driven-soc.git
cd ai-driven-soc
```

### Option C: Using gcloud SCP (from your local machine)

```bash
# From your local terminal (not SSH)
gcloud compute scp --recurse \
  /Users/raditio.ghifiardigmail.com/Downloads/ai-driven-soc.backup \
  xdgaisocapp01:~/ \
  --zone=asia-southeast2-a \
  --project=chronicle-dev-2be9
```

---

## 🔧 Step 2: Install Dependencies

```bash
# SSH into your VM (already connected based on screenshot)
cd ai-driven-soc.backup

# Install threat hunting dependencies
pip3 install -r requirements_threat_hunting.txt

# Or if using a virtual environment (recommended)
python3 -m venv threat-hunting-env
source threat-hunting-env/bin/activate
pip install -r requirements_threat_hunting.txt
```

**Expected packages to install:**
- yara-python (YARA rule engine)
- psutil (process/system monitoring)
- google-cloud-pubsub
- google-cloud-firestore
- google-cloud-bigquery
- google-cloud-storage
- langgraph (workflow orchestration)

---

## ⚙️ Step 3: Configure for Your GCP Project

### Edit Configuration Files

```bash
# Edit THOR config
nano config/thor_config.json
# Change: "gcp_project_id": "chronicle-dev-2be9"

# Edit ASGARD config
nano config/asgard_config.json
# Change: "gcp_project_id": "chronicle-dev-2be9"

# Edit VALHALLA config
nano config/valhalla_config.json
# Change: "gcp_project_id": "chronicle-dev-2be9"
```

### Quick Config Update Script

```bash
# Create quick update script
cat > update_gcp_project.sh << 'EOF'
#!/bin/bash
PROJECT_ID="chronicle-dev-2be9"

for config in config/thor_config.json config/asgard_config.json config/valhalla_config.json; do
  if [ -f "$config" ]; then
    sed -i "s/your-gcp-project-id/$PROJECT_ID/g" "$config"
    echo "Updated $config"
  fi
done
EOF

chmod +x update_gcp_project.sh
./update_gcp_project.sh
```

---

## 🎯 Step 4: Set Up GCP Resources (One-Time Setup)

```bash
# Set your project
gcloud config set project chronicle-dev-2be9

# Create Pub/Sub topics
gcloud pubsub topics create thor-scan-requests
gcloud pubsub topics create thor-findings
gcloud pubsub topics create asgard-campaigns
gcloud pubsub topics create asgard-scan-tasks
gcloud pubsub topics create valhalla-rule-updates
gcloud pubsub topics create valhalla-ioc-updates

# Create subscriptions
gcloud pubsub subscriptions create thor-scan-requests-sub \
  --topic=thor-scan-requests

gcloud pubsub subscriptions create thor-findings-sub \
  --topic=thor-findings

# Create BigQuery dataset
bq mk --dataset chronicle-dev-2be9:soc_data

# Create BigQuery tables
bq mk --table chronicle-dev-2be9:soc_data.thor_scan_results \
  scan_id:STRING,hostname:STRING,start_time:TIMESTAMP,end_time:TIMESTAMP,\
  total_files_scanned:INTEGER,total_processes_scanned:INTEGER,\
  total_network_connections:INTEGER,total_threats:INTEGER,\
  critical_count:INTEGER,high_count:INTEGER,medium_count:INTEGER,\
  matches:STRING,statistics:STRING,tenant_id:STRING

bq mk --table chronicle-dev-2be9:soc_data.asgard_campaign_reports \
  campaign_id:STRING,campaign_name:STRING,created_at:TIMESTAMP,\
  completed_at:TIMESTAMP,duration_minutes:FLOAT,total_targets:INTEGER,\
  successfully_scanned:INTEGER,failed_scans:INTEGER,total_threats:INTEGER,\
  critical_threats:INTEGER,tenant_id:STRING

# Create GCS bucket for threat intel
gsutil mb -p chronicle-dev-2be9 -l asia-southeast2 gs://chronicle-valhalla-threat-intel
gsutil versioning set on gs://chronicle-valhalla-threat-intel
```

---

## 🧪 Step 5: Test Individual Components

### Test 1: VALHALLA Feed Manager

```bash
# Test fetching threat intelligence
python3 valhalla_feed_manager.py

# Expected output:
# ✅ Fetched X IOCs from ThreatFox
# ✅ Fetched Y hashes from MalwareBazaar
# ✅ Downloaded YARA rules from Emerging Threats
# ✅ Exported IOC set for THOR agents
```

### Test 2: THOR Endpoint Agent (Scan Your VM)

```bash
# Run a quick filesystem scan on your VM
python3 thor_endpoint_agent.py \
  --config config/thor_config.json \
  --scan-type filesystem \
  --target /tmp /var/tmp \
  --load-yara \
  --load-iocs

# Expected output:
# Starting THOR scan thor_scan_20250101_120000 on xdgaisocapp01
# Performing filesystem scan...
# Scanned 1000 files, found 0-5 threats
# THOR scan complete: X threats detected
```

### Test 3: ASGARD Orchestration (Register VM as Endpoint)

```bash
# Create test script
cat > test_asgard.py << 'EOF'
#!/usr/bin/env python3
import asyncio
from asgard_orchestration_agent import ASGARDOrchestrationAgent, EndpointInfo
from datetime import datetime

async def main():
    asgard = ASGARDOrchestrationAgent()

    # Register this VM as an endpoint
    endpoint = EndpointInfo(
        endpoint_id="xdgaisocapp01",
        hostname="xdgaisocapp01",
        ip_address="10.0.0.1",  # Update with actual IP
        os_type="linux",
        os_version="Ubuntu 22.04",
        agent_version="1.0.0",
        last_seen=datetime.utcnow().isoformat(),
        labels={"env": "development", "role": "test-vm"},
        groups=["development"],
        status="online",
        capabilities=["yara", "ioc", "process", "network"]
    )

    success = asgard.register_endpoint(endpoint)
    print(f"Endpoint registration: {'✅ Success' if success else '❌ Failed'}")

    # List registered endpoints
    print(f"\nRegistered endpoints: {len(asgard.registered_endpoints)}")
    for ep_id, ep in asgard.registered_endpoints.items():
        print(f"  - {ep.hostname} ({ep.status})")

if __name__ == "__main__":
    asyncio.run(main())
EOF

python3 test_asgard.py
```

---

## 🎬 Step 6: Run Complete Demo

```bash
# Run the full quick-start demonstration
python3 threat_hunting_quickstart.py

# This will:
# 1. Update threat intelligence feeds
# 2. Register demo endpoints (including your VM)
# 3. Create sample threat hunting campaigns
# 4. Demonstrate TAA/CRA/CLA integration
# 5. Show complete SOC workflow

# Expected runtime: 2-3 minutes
```

---

## 🔍 Step 7: Test Real Scanning on Your VM

### Scan Your VM for Common Threats

```bash
# Create a comprehensive scan script
cat > scan_my_vm.py << 'EOF'
#!/usr/bin/env python3
from thor_endpoint_agent import THOREndpointAgent, ScanType

# Initialize THOR agent
agent = THOREndpointAgent(config_path="config/thor_config.json")

# Load rules and IOCs
print("Loading YARA rules and IOCs...")
agent.load_yara_rules()
agent.load_iocs()

# Perform comprehensive scan
print("\nStarting comprehensive threat scan...")
result = agent.perform_scan(
    scan_types=[ScanType.FILESYSTEM, ScanType.PROCESS, ScanType.NETWORK],
    target_paths=["/tmp", "/var/tmp", "/home", "/opt"]
)

# Print results
print(f"\n{'='*60}")
print(f"Scan Results: {result.scan_id}")
print(f"{'='*60}")
print(f"Files scanned: {result.total_files_scanned}")
print(f"Processes scanned: {result.total_processes_scanned}")
print(f"Network connections: {result.total_network_connections}")
print(f"Threats detected: {len(result.matches)}")

if result.matches:
    print(f"\n⚠️  THREATS FOUND:")
    for match in result.matches[:10]:  # Show top 10
        print(f"  [{match.severity.value.upper()}] {match.rule_name}")
        print(f"    Target: {match.target}")
        print(f"    Confidence: {match.confidence:.2%}")
else:
    print(f"\n✅ No threats detected - System clean!")

print(f"\nResults stored in BigQuery: soc_data.thor_scan_results")
EOF

python3 scan_my_vm.py
```

---

## 📊 Step 8: Query Results in BigQuery

```bash
# Query scan results
bq query --use_legacy_sql=false '
SELECT
  scan_id,
  hostname,
  start_time,
  total_files_scanned,
  total_threats,
  critical_count
FROM `chronicle-dev-2be9.soc_data.thor_scan_results`
ORDER BY start_time DESC
LIMIT 10;
'

# Query threats by severity
bq query --use_legacy_sql=false '
SELECT
  JSON_VALUE(match, "$.severity") as severity,
  JSON_VALUE(match, "$.rule_name") as rule_name,
  COUNT(*) as count
FROM `chronicle-dev-2be9.soc_data.thor_scan_results`,
  UNNEST(JSON_EXTRACT_ARRAY(matches)) as match
GROUP BY severity, rule_name
ORDER BY count DESC
LIMIT 20;
'
```

---

## 🎯 Step 9: Create a Real Threat Hunting Campaign

```bash
# Create campaign script
cat > create_campaign.py << 'EOF'
#!/usr/bin/env python3
import asyncio
from asgard_orchestration_agent import (
    ASGARDOrchestrationAgent,
    ScanPriority,
    TargetSelectionMode
)

async def main():
    asgard = ASGARDOrchestrationAgent()

    # Create a threat hunting campaign
    campaign = asgard.create_campaign(
        name="Dev VM Threat Hunt - Test",
        description="Testing threat hunting on development VM",
        target_selection_mode=TargetSelectionMode.LABEL,
        target_criteria={"labels": {"env": "development"}},
        scan_types=["filesystem", "process", "network"],
        yara_rule_sets=["ransomware", "trojan", "malware"],
        ioc_feeds=["valhalla", "threatfox"],
        priority=ScanPriority.MEDIUM,
        schedule_type="immediate",
        created_by="test_user"
    )

    print(f"✅ Campaign created: {campaign.campaign_id}")
    print(f"   Name: {campaign.name}")
    print(f"   Target endpoints: {campaign.total_targets}")
    print(f"   Priority: {campaign.priority.value}")
    print(f"   Status: {campaign.status.value}")

    # Monitor for 60 seconds
    print(f"\nMonitoring campaign for 60 seconds...")
    for i in range(6):
        await asyncio.sleep(10)
        status = asgard.get_campaign_status(campaign.campaign_id)
        if status:
            print(f"  [{i*10}s] {status['progress']['scanned']}/{status['progress']['total']} scanned, "
                  f"{status['threats']['total']} threats")

if __name__ == "__main__":
    asyncio.run(main())
EOF

python3 create_campaign.py
```

---

## 🐛 Troubleshooting

### Issue: "Permission denied" errors

```bash
# Grant service account permissions
gcloud projects add-iam-policy-binding chronicle-dev-2be9 \
  --member="serviceAccount:YOUR_SERVICE_ACCOUNT@chronicle-dev-2be9.iam.gserviceaccount.com" \
  --role="roles/pubsub.publisher"

gcloud projects add-iam-policy-binding chronicle-dev-2be9 \
  --member="serviceAccount:YOUR_SERVICE_ACCOUNT@chronicle-dev-2be9.iam.gserviceaccount.com" \
  --role="roles/pubsub.subscriber"

gcloud projects add-iam-policy-binding chronicle-dev-2be9 \
  --member="serviceAccount:YOUR_SERVICE_ACCOUNT@chronicle-dev-2be9.iam.gserviceaccount.com" \
  --role="roles/bigquery.dataEditor"

gcloud projects add-iam-policy-binding chronicle-dev-2be9 \
  --member="serviceAccount:YOUR_SERVICE_ACCOUNT@chronicle-dev-2be9.iam.gserviceaccount.com" \
  --role="roles/datastore.user"
```

### Issue: YARA rules not loading

```bash
# Create sample YARA rules directory
mkdir -p /opt/thor/yara_rules

# Create a test rule
cat > /opt/thor/yara_rules/test_malware.yar << 'EOF'
rule Test_Malware_Pattern {
    meta:
        description = "Test YARA rule for demo"
        author = "SOC Team"
        severity = "medium"
    strings:
        $sus1 = "malware" nocase
        $sus2 = "backdoor" nocase
    condition:
        any of them
}
EOF

# Update thor_config.json to point to this directory
```

### Issue: Module import errors

```bash
# Ensure you're using Python 3.8+
python3 --version

# Reinstall dependencies
pip3 install --upgrade -r requirements_threat_hunting.txt

# Check installations
pip3 list | grep -E "yara|psutil|google-cloud"
```

---

## 💡 Testing Best Practices

### 1. Start Small
- ✅ Test VALHALLA first (fetch threat intel)
- ✅ Test THOR on limited paths (/tmp only)
- ✅ Verify BigQuery writes
- ✅ Then expand to full scans

### 2. Monitor Resource Usage

```bash
# Watch system resources during scan
htop

# Or simpler:
top

# Check disk I/O
iostat -x 1

# Monitor network
iftop
```

### 3. Test in Stages

**Stage 1: Component Testing (30 min)**
- VALHALLA feed updates
- THOR single file scan
- ASGARD endpoint registration

**Stage 2: Integration Testing (1 hour)**
- THOR → TAA integration (if TAA deployed)
- ASGARD → THOR campaign
- Results in BigQuery

**Stage 3: Full Workflow (2 hours)**
- Complete threat hunt campaign
- TAA → CRA → CLA pipeline
- Analytics and reporting

---

## 📈 Expected Performance on Your VM

Based on typical GCP VM specs:

| Scan Type | Files/Processes | Expected Time | CPU Usage | Memory |
|-----------|----------------|---------------|-----------|--------|
| /tmp only | ~1,000 files | 30 seconds | 10-20% | 200 MB |
| Full filesystem | ~100,000 files | 10-15 minutes | 30-50% | 500 MB |
| Process scan | ~100 processes | 5 seconds | 5-10% | 100 MB |
| Network scan | ~50 connections | 2 seconds | 5% | 50 MB |

---

## ✅ Success Checklist

After testing, you should have:
- ☐ VALHALLA successfully fetched threat intel
- ☐ THOR scanned your VM and stored results
- ☐ ASGARD registered your VM as endpoint
- ☐ Created at least one campaign
- ☐ Results visible in BigQuery
- ☐ No permission errors
- ☐ Understanding of how components work together

---

## 🎓 Next Steps After Testing

1. **Expand to Other VMs**
   - Install THOR on other development VMs
   - Test fleet-wide campaigns

2. **Integrate with Existing SOC**
   - Connect THOR findings to your TAA agent
   - Enable CRA automated response
   - Feed data to CLA for learning

3. **Production Deployment**
   - Follow Pure_Platform_Deployment_Guide.docx
   - Set up scheduled campaigns
   - Configure monitoring and alerting

4. **Evaluate Results**
   - Review detection rates
   - Assess false positives
   - Calculate ROI
   - Decide on Hybrid vs. Pure deployment

---

## 📞 Need Help?

If you encounter issues:
1. Check logs: `tail -f /var/log/syslog`
2. Enable debug logging in config files
3. Review THREAT_HUNTING_README.md
4. Check GCP console for quota/permission errors

---

**Your VM is ready to test! Start with Step 1 and work through each step.** 🚀
