# 🚀 Quick Start: Testing on Your GCP VM

## Copy-Paste Commands for Your SSH Session

Based on your screenshot showing VM `xdgaisocapp01`, here's the exact command sequence to execute.

---

## Step 1: Download and Extract Files (On Your VM)

```bash
# Navigate to home directory
cd ~

# If you uploaded the files via SSH browser, skip to setup
# Otherwise, create directory and prepare for file upload
mkdir -p ai-driven-soc-backup
```

**Now use the "UPLOAD FILE" button in your SSH session to upload:**
- `setup_threat_hunting_test.sh`
- `thor_endpoint_agent.py`
- `asgard_orchestration_agent.py`
- `valhalla_feed_manager.py`
- `threat_hunting_quickstart.py`

---

## Step 2: Run Automated Setup

```bash
# Make setup script executable
chmod +x setup_threat_hunting_test.sh

# Run setup (creates clean test environment)
bash setup_threat_hunting_test.sh
```

**This will:**
- ✅ Create `~/threat-hunting-test/` directory
- ✅ Create Python virtual environment
- ✅ Install all dependencies
- ✅ Create config files (pre-configured for chronicle-dev-2be9)
- ✅ Create sample YARA rules
- ✅ Create test data files
- ✅ Create helper scripts

**Time:** ~5-10 minutes (depending on package downloads)

---

## Step 3: Copy Agent Files to Test Directory

```bash
# Copy uploaded agent files to test directory
cp ~/thor_endpoint_agent.py ~/threat-hunting-test/
cp ~/asgard_orchestration_agent.py ~/threat-hunting-test/
cp ~/valhalla_feed_manager.py ~/threat-hunting-test/
cp ~/threat_hunting_quickstart.py ~/threat-hunting-test/

# Navigate to test directory
cd ~/threat-hunting-test
```

---

## Step 4: Setup GCP Resources (One-Time)

```bash
# Activate environment
source activate.sh

# Setup GCP Pub/Sub, BigQuery, Cloud Storage
bash scripts/setup_gcp_resources.sh
```

**This creates:**
- ✅ 6 Pub/Sub topics
- ✅ 2 Pub/Sub subscriptions
- ✅ BigQuery dataset `soc_data`
- ✅ 2 BigQuery tables
- ✅ GCS bucket for threat intel

**Time:** ~2-3 minutes

---

## Step 5: Test Individual Components

### Test 1: Update Threat Intelligence

```bash
bash scripts/update_threat_intel.sh
```

**Expected output:**
```
📡 Updating threat intelligence feeds...
Fetching IOCs from ThreatFox and MalwareBazaar...

📊 Feed Update Summary:
   threatfox: 150 new items
   malwarebazaar: 50 new items

✅ Total IOCs exported: 200
   - IPs: 75
   - Domains: 25
   - Hashes: 100
```

---

### Test 2: Run Quick Scan

```bash
bash scripts/quick_test_scan.sh
```

**Expected output:**
```
🔍 Running quick test scan...
Loading YARA rules...

Scanning test data directory...

============================================================
Test Scan Results: thor_scan_20250101_120000
============================================================
Files scanned: 3
Threats detected: 1

⚠️  DETECTIONS (Expected for test files):
  [MEDIUM] Test_Suspicious_Keywords
    File: test_data/suspicious/test_sample.txt

Note: test_data/suspicious/ SHOULD trigger detections
```

**✅ If you see detections, YARA scanning is working!**

---

## Step 6: Run Full Demonstration

```bash
# Activate environment if not already active
source activate.sh

# Run complete demo
python3 threat_hunting_quickstart.py
```

**This demonstrates:**
1. VALHALLA feed updates
2. Endpoint registration (your VM)
3. Campaign creation
4. TAA/CRA/CLA integration simulation

**Time:** ~2-3 minutes

---

## Step 7: Scan Your Actual VM

```bash
# Create full VM scan script
cat > scan_full_vm.py << 'EOF'
#!/usr/bin/env python3
from thor_endpoint_agent import THOREndpointAgent, ScanType

agent = THOREndpointAgent(config_path="config/thor_config.json")

print("Loading rules and IOCs...")
agent.load_yara_rules()
agent.load_iocs()

print("\nScanning VM (this may take 5-10 minutes)...")
result = agent.perform_scan(
    scan_types=[ScanType.FILESYSTEM, ScanType.PROCESS, ScanType.NETWORK],
    target_paths=["/tmp", "/var/tmp", "/home"]
)

print(f"\n{'='*60}")
print(f"VM Scan Complete: {result.scan_id}")
print(f"{'='*60}")
print(f"Files: {result.total_files_scanned}")
print(f"Processes: {result.total_processes_scanned}")
print(f"Connections: {result.total_network_connections}")
print(f"Threats: {len(result.matches)}")

if result.matches:
    print(f"\n⚠️  THREATS DETECTED:")
    for match in result.matches[:10]:
        print(f"  [{match.severity.value.upper()}] {match.rule_name}")
        print(f"    Target: {match.target}")
else:
    print(f"\n✅ No threats detected - VM is clean!")
EOF

python3 scan_full_vm.py
```

---

## Step 8: Query Results in BigQuery

```bash
# View recent scans
bq query --use_legacy_sql=false '
SELECT
  scan_id,
  hostname,
  start_time,
  total_files_scanned,
  total_threats
FROM `chronicle-dev-2be9.soc_data.thor_scan_results`
ORDER BY start_time DESC
LIMIT 5;
'

# View detected threats
bq query --use_legacy_sql=false '
SELECT
  JSON_VALUE(match, "$.rule_name") as threat,
  JSON_VALUE(match, "$.severity") as severity,
  COUNT(*) as count
FROM `chronicle-dev-2be9.soc_data.thor_scan_results`,
  UNNEST(JSON_EXTRACT_ARRAY(matches)) as match
GROUP BY threat, severity
ORDER BY count DESC
LIMIT 10;
'
```

---

## 🎯 Success Criteria

After running the above steps, you should see:

- ✅ Virtual environment activated
- ✅ All dependencies installed (no errors)
- ✅ GCP resources created successfully
- ✅ Threat intel fetched (100+ IOCs)
- ✅ Test scan detected suspicious file
- ✅ Full demo completed
- ✅ VM scan completed (check for threats)
- ✅ Results visible in BigQuery

---

## 🐛 Troubleshooting

### Issue: "Permission denied" on GCP resources

```bash
# Check your current GCP authentication
gcloud auth list

# If needed, re-authenticate
gcloud auth application-default login

# Or use service account
gcloud auth activate-service-account --key-file=/path/to/key.json
```

### Issue: "Module not found: yara"

```bash
# Ensure virtual environment is activated
source ~/threat-hunting-test/activate.sh

# Reinstall yara-python
pip install --upgrade yara-python

# Verify installation
python3 -c "import yara; print('YARA installed successfully')"
```

### Issue: "Bucket access denied"

```bash
# Check if bucket exists
gsutil ls gs://chronicle-dev-2be9-valhalla-threat-intel

# Create if missing
gsutil mb -p chronicle-dev-2be9 -l asia-southeast2 \
  gs://chronicle-dev-2be9-valhalla-threat-intel
```

### Issue: BigQuery errors

```bash
# Verify dataset exists
bq ls -d chronicle-dev-2be9:soc_data

# If not, create it
bq mk --dataset --location=asia-southeast2 chronicle-dev-2be9:soc_data
```

---

## 📊 Monitoring Resource Usage

```bash
# Watch CPU/Memory during scan
top

# Or use htop (if installed)
htop

# Check disk I/O
iostat -x 1

# Monitor network
iftop
```

---

## 🧹 Cleanup (When Done Testing)

```bash
# Delete GCP resources
bash ~/threat-hunting-test/scripts/cleanup_gcp.sh

# Remove test directory
rm -rf ~/threat-hunting-test

# Or keep for future testing!
```

---

## 📝 Notes for Your VM (xdgaisocapp01)

Based on your screenshot:
- ✅ You have Python 3.13.0 available
- ✅ VM is in `asia-southeast2-a` zone
- ✅ Project is `chronicle-dev-2be9`
- ✅ SSH access is working
- ✅ You can upload files via browser

**All set for testing!** 🎉

---

## 🎓 What to Test Next

After basic tests work:

1. **Create Custom YARA Rule**
   ```bash
   nano ~/threat-hunting-test/yara_rules/custom_rule.yar
   # Add your rule, then run scan
   ```

2. **Scan Different Directories**
   ```bash
   # Edit scan_full_vm.py to scan /opt or /usr
   ```

3. **Create Threat Hunting Campaign**
   ```bash
   python3 << 'EOF'
   import asyncio
   from asgard_orchestration_agent import *

   async def main():
       asgard = ASGARDOrchestrationAgent()
       campaign = asgard.create_campaign(
           name="Test Campaign",
           target_selection_mode=TargetSelectionMode.ALL,
           scan_types=["filesystem"],
           priority=ScanPriority.MEDIUM
       )
       print(f"Campaign created: {campaign.campaign_id}")

   asyncio.run(main())
   EOF
   ```

4. **Integrate with Existing SOC**
   - Connect THOR findings to your TAA agent
   - Test CRA automated response
   - Feed results to CLA

---

## 📞 Need Help?

Check logs:
```bash
# View THOR logs
tail -f ~/threat-hunting-test/logs/*.log

# Check system logs
sudo tail -f /var/log/syslog
```

Enable debug mode:
```bash
# Edit config files
nano ~/threat-hunting-test/config/thor_config.json
# Add: "log_level": "DEBUG"
```

---

**Ready to start? Copy-paste the commands above into your SSH session!** 🚀
