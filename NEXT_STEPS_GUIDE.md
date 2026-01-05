# Next Steps Guide - Threat Hunting Testing

This guide covers the execution of the 4 next steps for complete threat hunting testing.

## Overview

1. **Register Endpoints** - Register the GCP instance with ASGARD
2. **Create Custom YARA Rules** - Add test YARA rules for detection
3. **Setup GCP Resources** - Create Pub/Sub topics and BigQuery tables
4. **Test Threat Detection** - Create test malware and verify detection

---

## Step 1: Register Endpoints for ASGARD Campaigns

### On GCP Instance

```bash
cd ~/threat-hunting-test
source venv/bin/activate

# Download and run registration script
curl -f -s -o register_endpoint.py https://raw.githubusercontent.com/ghifiardi/ai-driven-soc/main/register_endpoint.py
python register_endpoint.py
```

### What it does:
- Collects endpoint information (hostname, IP, OS)
- Registers with ASGARD via Firestore
- Makes the endpoint available for campaigns

### Expected Output:
```
✓ Successfully registered endpoint: xdgaisocapp01
  Endpoint ID: endpoint_xdgaisocapp01
Total registered endpoints: 1
```

---

## Step 2: Create Custom YARA Rules for Testing

### On GCP Instance

```bash
cd ~/threat-hunting-test

# Download YARA rules
curl -f -s -o data/yara_rules/test_malware_rules.yar https://raw.githubusercontent.com/ghifiardi/ai-driven-soc/main/data/yara_rules/test_malware_rules.yar

# Verify rules
ls -lh data/yara_rules/
```

### YARA Rules Included:
- `Test_Ransomware_Signature` - Detects ransomware keywords
- `Test_Backdoor_Detection` - Detects backdoor patterns
- `Test_Webshell_Detection` - Detects webshell signatures
- `Test_CryptoMiner_Detection` - Detects cryptocurrency miners
- `Test_Suspicious_Process` - Detects suspicious processes
- `Test_File_Hash_Match` - Matches known file hashes

### Update Config to Use Local Rules:

Edit `config/thor_config.json`:
```json
{
  "yara": {
    "rules_path": "/home/app/threat-hunting-test/data/yara_rules"
  }
}
```

---

## Step 3: Set Up GCP Resources

### Option A: Using gcloud CLI (if you have access)

```bash
# From your local machine or GCP Cloud Shell
bash setup_gcp_resources.sh
```

### Option B: Manual Creation

#### Pub/Sub Topics:
```bash
gcloud pubsub topics create thor-scan-requests --project=chronicle-dev-2be9
gcloud pubsub topics create thor-findings --project=chronicle-dev-2be9
gcloud pubsub topics create asgard-campaigns --project=chronicle-dev-2be9
gcloud pubsub topics create asgard-scan-tasks --project=chronicle-dev-2be9
gcloud pubsub topics create asgard-scan-results --project=chronicle-dev-2be9
gcloud pubsub topics create thor-heartbeat --project=chronicle-dev-2be9
gcloud pubsub topics create valhalla-rule-updates --project=chronicle-dev-2be9
gcloud pubsub topics create valhalla-ioc-updates --project=chronicle-dev-2be9
```

#### BigQuery Dataset and Tables:
```bash
# Create dataset
bq mk --dataset chronicle-dev-2be9:soc_data

# Create THOR scan results table
bq mk --table chronicle-dev-2be9:soc_data.thor_scan_results \
  scan_id:STRING,hostname:STRING,start_time:TIMESTAMP,end_time:TIMESTAMP,threats_detected:INTEGER,files_scanned:INTEGER,processes_scanned:INTEGER,network_connections:INTEGER,matches:JSON

# Create ASGARD campaign reports table
bq mk --table chronicle-dev-2be9:soc_data.asgard_campaign_reports \
  campaign_id:STRING,campaign_name:STRING,created_at:TIMESTAMP,status:STRING,total_targets:INTEGER,successfully_scanned:INTEGER,total_threats:INTEGER,critical_threats:INTEGER

# Create VALHALLA feed stats table
bq mk --table chronicle-dev-2be9:soc_data.valhalla_feed_stats \
  feed_name:STRING,update_time:TIMESTAMP,iocs_count:INTEGER,yara_rules_count:INTEGER,sigma_rules_count:INTEGER
```

#### GCS Bucket:
```bash
gsutil mb -p chronicle-dev-2be9 gs://valhalla-threat-intel-chronicle-dev-2be9
```

---

## Step 4: Test Threat Detection with Sample Malware

### On GCP Instance

```bash
cd ~/threat-hunting-test
source venv/bin/activate

# Download and run test malware creation script
curl -f -s -o create_test_malware.py https://raw.githubusercontent.com/ghifiardi/ai-driven-soc/main/create_test_malware.py
python create_test_malware.py
```

### Test Files Created:
- `test_ransomware.txt` - Triggers ransomware rule
- `test_backdoor.txt` - Triggers backdoor rule
- `test_webshell.txt` - Triggers webshell rule
- `test_cryptominer.txt` - Triggers cryptominer rule
- `test_suspicious.txt` - Triggers suspicious process rule
- `test_hash_match.txt` - Triggers hash match rule

### Run THOR Scan on Test Files:

```bash
# Scan the test malware directory
python thor_endpoint_agent.py \
  --config config/thor_config.json \
  --scan-type filesystem \
  --target data/test_malware \
  --load-yara
```

### Expected Results:
- THOR should detect multiple threats
- Each test file should trigger its corresponding YARA rule
- Scan results should show threats detected

---

## Complete Setup Script

### On GCP Instance

Run all steps at once:

```bash
cd ~/threat-hunting-test
source venv/bin/activate

# Download complete setup script
curl -f -s -o run_complete_setup.sh https://raw.githubusercontent.com/ghifiardi/ai-driven-soc/main/run_complete_setup.sh
chmod +x run_complete_setup.sh
bash run_complete_setup.sh
```

---

## Verification Steps

### 1. Verify Endpoint Registration:
```bash
python -c "
from asgard_orchestration_agent import ASGARDOrchestrationAgent
asgard = ASGARDOrchestrationAgent()
print(f'Registered endpoints: {len(asgard.registered_endpoints)}')
for ep_id, ep in asgard.registered_endpoints.items():
    print(f'  - {ep.hostname} ({ep.endpoint_id})')
"
```

### 2. Verify YARA Rules:
```bash
ls -lh data/yara_rules/
yara data/yara_rules/test_malware_rules.yar data/test_malware/test_ransomware.txt
```

### 3. Verify Test Files:
```bash
ls -lh data/test_malware/
```

### 4. Test Detection:
```bash
python thor_endpoint_agent.py \
  --config config/thor_config.json \
  --scan-type filesystem \
  --target data/test_malware \
  --load-yara
```

---

## Troubleshooting

### Endpoint Registration Fails
- **Issue:** Firestore not accessible
- **Solution:** Ensure GCP credentials are configured, or registration will work in-memory only

### YARA Rules Not Loading
- **Issue:** Rules path incorrect
- **Solution:** Update `config/thor_config.json` with correct path

### GCP Resources Creation Fails
- **Issue:** Insufficient permissions
- **Solution:** Ensure service account has required permissions or create resources manually via GCP Console

### Test Files Not Detected
- **Issue:** YARA rules not loaded
- **Solution:** Use `--load-yara` flag and verify rules path in config

---

## Next Steps After Setup

1. **Create ASGARD Campaign:**
   ```bash
   python asgard_orchestration_agent.py
   ```

2. **Monitor Campaign Progress:**
   - Check Firestore for campaign status
   - View BigQuery tables for scan results

3. **Add More YARA Rules:**
   - Place `.yar` files in `data/yara_rules/`
   - THOR will load them automatically

4. **Integrate with TAA:**
   - Ensure Pub/Sub topic `thor-findings` exists
   - TAA will automatically receive findings

---

## Quick Reference

### Files Created:
- `register_endpoint.py` - Endpoint registration script
- `data/yara_rules/test_malware_rules.yar` - Test YARA rules
- `create_test_malware.py` - Test malware generator
- `setup_gcp_resources.sh` - GCP resource setup
- `run_complete_setup.sh` - Complete setup automation

### Key Directories:
- `~/threat-hunting-test/data/yara_rules/` - YARA rules
- `~/threat-hunting-test/data/test_malware/` - Test files
- `~/threat-hunting-test/config/` - Configuration files

---

**End of Guide**

