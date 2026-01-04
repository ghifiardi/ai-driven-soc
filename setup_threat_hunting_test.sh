#!/bin/bash
#
# Threat Hunting Platform - Test Environment Setup
# Creates a clean directory structure for testing on GCP VM
#
# Usage: bash setup_threat_hunting_test.sh
#

set -e  # Exit on error

echo "========================================================"
echo "  Threat Hunting Platform - Test Environment Setup"
echo "========================================================"
echo ""

# Configuration
TEST_DIR="$HOME/threat-hunting-test"
VENV_NAME="threat-hunting-env"
GCP_PROJECT="chronicle-dev-2be9"  # Update if different
GCP_REGION="asia-southeast2"

echo "📁 Creating directory structure..."
echo ""

# Create main test directory
mkdir -p "$TEST_DIR"
cd "$TEST_DIR"

# Create subdirectories
mkdir -p config
mkdir -p logs
mkdir -p yara_rules
mkdir -p sigma_rules
mkdir -p test_data
mkdir -p results
mkdir -p scripts

echo "✅ Created directory: $TEST_DIR"
echo "   ├── config/          (configuration files)"
echo "   ├── logs/            (scan logs and results)"
echo "   ├── yara_rules/      (YARA detection rules)"
echo "   ├── sigma_rules/     (Sigma log analysis rules)"
echo "   ├── test_data/       (test files for scanning)"
echo "   ├── results/         (scan result reports)"
echo "   └── scripts/         (helper scripts)"
echo ""

# Create Python virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv "$VENV_NAME"
source "$VENV_NAME/bin/activate"

echo "✅ Virtual environment created and activated"
echo ""

# Create requirements file
echo "📦 Creating requirements.txt..."
cat > requirements.txt << 'EOF'
# Core Dependencies for Threat Hunting Platform
yara-python>=4.3.0
psutil>=5.9.0
requests>=2.31.0

# Google Cloud Platform
google-cloud-pubsub>=2.18.0
google-cloud-firestore>=2.13.0
google-cloud-bigquery>=3.13.0
google-cloud-storage>=2.10.0
google-cloud-compute>=1.14.0

# LangGraph for workflow orchestration
langgraph>=0.0.20
typing-extensions>=4.8.0

# Additional utilities
pyyaml>=6.0.1
python-dateutil>=2.8.2
cryptography>=41.0.0
EOF

echo "✅ requirements.txt created"
echo ""

# Install dependencies
echo "📥 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Dependencies installed"
echo ""

# Create configuration files
echo "⚙️  Creating configuration files..."

# THOR config
cat > config/thor_config.json << EOF
{
  "tenant_id": "test-environment",
  "gcp_project_id": "$GCP_PROJECT",
  "yara": {
    "rules_path": "$TEST_DIR/yara_rules",
    "enable_memory_scan": false,
    "enable_process_scan": true,
    "auto_update_rules": true,
    "update_interval_hours": 6
  },
  "ioc": {
    "feeds": ["threatfox", "malwarebazaar"],
    "update_interval_hours": 6,
    "auto_sync": true
  },
  "scan_exclusions": {
    "paths": [
      "/proc",
      "/sys",
      "/dev",
      "$TEST_DIR/$VENV_NAME"
    ],
    "extensions": [
      ".pyc",
      ".log",
      ".tmp"
    ],
    "max_file_size_mb": 100
  },
  "scan_limits": {
    "max_depth": 5,
    "max_files_per_scan": 10000,
    "timeout_minutes": 30,
    "max_memory_mb": 2048
  },
  "pubsub_topics": {
    "scan_requests": "thor-scan-requests",
    "findings": "thor-findings",
    "heartbeat": "thor-heartbeat"
  },
  "reporting": {
    "store_in_bigquery": true,
    "bigquery_dataset": "soc_data",
    "bigquery_table": "thor_scan_results",
    "publish_to_pubsub": true
  },
  "performance": {
    "max_parallel_file_scans": 4,
    "enable_incremental_scans": true
  }
}
EOF

# ASGARD config
cat > config/asgard_config.json << EOF
{
  "tenant_id": "test-environment",
  "gcp_project_id": "$GCP_PROJECT",
  "pubsub_topics": {
    "campaigns": "asgard-campaigns",
    "scan_tasks": "asgard-scan-tasks",
    "scan_results": "thor-findings",
    "agent_heartbeat": "thor-heartbeat"
  },
  "orchestration": {
    "max_parallel_scans_per_campaign": 10,
    "default_scan_timeout_minutes": 30,
    "retry_failed_scans": true,
    "max_retries": 2
  },
  "endpoint_discovery": {
    "auto_discover_gcp_vms": true,
    "auto_discover_interval_hours": 1,
    "required_labels": {
      "env": "test"
    }
  },
  "reporting": {
    "store_in_bigquery": true,
    "bigquery_dataset": "soc_data",
    "bigquery_table": "asgard_campaign_reports"
  }
}
EOF

# VALHALLA config
cat > config/valhalla_config.json << EOF
{
  "tenant_id": "test-environment",
  "gcp_project_id": "$GCP_PROJECT",
  "gcs": {
    "rules_bucket": "$GCP_PROJECT-valhalla-threat-intel"
  },
  "pubsub_topics": {
    "rule_updates": "valhalla-rule-updates",
    "ioc_updates": "valhalla-ioc-updates"
  },
  "feeds": {
    "threatfox": {
      "enabled": true,
      "url": "https://threatfox-api.abuse.ch/api/v1/",
      "update_interval_hours": 6,
      "types": ["ip", "domain", "hash", "url"]
    },
    "malwarebazaar": {
      "enabled": true,
      "url": "https://mb-api.abuse.ch/api/v1/",
      "update_interval_hours": 24,
      "types": ["hash"]
    }
  },
  "yara": {
    "auto_compile": true,
    "validation": true,
    "min_quality": "medium",
    "custom_rules_path": "$TEST_DIR/yara_rules"
  }
}
EOF

echo "✅ Configuration files created in config/"
echo ""

# Create sample YARA rules
echo "📝 Creating sample YARA rules..."

cat > yara_rules/test_malware.yar << 'EOF'
rule Test_Suspicious_Keywords {
    meta:
        description = "Detects common suspicious keywords (for testing)"
        author = "SOC Test Team"
        severity = "medium"
        date = "2025-01-01"
    strings:
        $keyword1 = "backdoor" nocase
        $keyword2 = "malware" nocase
        $keyword3 = "ransomware" nocase
        $keyword4 = "exploit" nocase
    condition:
        any of them
}

rule Test_Suspicious_Script {
    meta:
        description = "Detects potentially malicious script patterns"
        author = "SOC Test Team"
        severity = "high"
    strings:
        $cmd1 = "eval(base64_decode" nocase
        $cmd2 = "system(" nocase
        $cmd3 = "exec(" nocase
        $cmd4 = "shell_exec(" nocase
    condition:
        2 of them
}
EOF

echo "✅ Sample YARA rules created in yara_rules/"
echo ""

# Create test data
echo "🧪 Creating test data files..."

mkdir -p test_data/safe
mkdir -p test_data/suspicious

# Safe test file
cat > test_data/safe/readme.txt << 'EOF'
This is a safe test file.
It contains no malicious content.
Used for testing the threat hunting platform.
EOF

# Suspicious test file (contains YARA rule keywords)
cat > test_data/suspicious/test_sample.txt << 'EOF'
This file contains test keywords for YARA detection:
- backdoor
- malware
This is NOT actual malware, just a test file.
EOF

echo "✅ Test data files created in test_data/"
echo ""

# Create helper scripts
echo "📜 Creating helper scripts..."

# Script 1: Setup GCP resources
cat > scripts/setup_gcp_resources.sh << 'GCPSCRIPT'
#!/bin/bash
# Setup GCP resources for threat hunting

set -e

PROJECT_ID="chronicle-dev-2be9"
REGION="asia-southeast2"

echo "Setting up GCP resources for project: $PROJECT_ID"
echo ""

# Set project
gcloud config set project "$PROJECT_ID"

echo "1️⃣  Creating Pub/Sub topics..."
for topic in thor-scan-requests thor-findings asgard-campaigns asgard-scan-tasks valhalla-rule-updates valhalla-ioc-updates; do
  if ! gcloud pubsub topics describe "$topic" &>/dev/null; then
    gcloud pubsub topics create "$topic"
    echo "   ✅ Created topic: $topic"
  else
    echo "   ⏭️  Topic already exists: $topic"
  fi
done

echo ""
echo "2️⃣  Creating Pub/Sub subscriptions..."
if ! gcloud pubsub subscriptions describe thor-scan-requests-sub &>/dev/null; then
  gcloud pubsub subscriptions create thor-scan-requests-sub --topic=thor-scan-requests
  echo "   ✅ Created subscription: thor-scan-requests-sub"
else
  echo "   ⏭️  Subscription already exists: thor-scan-requests-sub"
fi

if ! gcloud pubsub subscriptions describe thor-findings-sub &>/dev/null; then
  gcloud pubsub subscriptions create thor-findings-sub --topic=thor-findings
  echo "   ✅ Created subscription: thor-findings-sub"
else
  echo "   ⏭️  Subscription already exists: thor-findings-sub"
fi

echo ""
echo "3️⃣  Creating BigQuery dataset..."
if ! bq ls -d "$PROJECT_ID:soc_data" &>/dev/null; then
  bq mk --dataset --location="$REGION" "$PROJECT_ID:soc_data"
  echo "   ✅ Created dataset: soc_data"
else
  echo "   ⏭️  Dataset already exists: soc_data"
fi

echo ""
echo "4️⃣  Creating BigQuery tables..."

# THOR results table
if ! bq ls "$PROJECT_ID:soc_data" | grep -q "thor_scan_results"; then
  bq mk --table "$PROJECT_ID:soc_data.thor_scan_results" \
    scan_id:STRING,hostname:STRING,start_time:TIMESTAMP,end_time:TIMESTAMP,\
total_files_scanned:INTEGER,total_processes_scanned:INTEGER,\
total_network_connections:INTEGER,total_threats:INTEGER,\
critical_count:INTEGER,high_count:INTEGER,medium_count:INTEGER,\
matches:STRING,statistics:STRING,tenant_id:STRING
  echo "   ✅ Created table: thor_scan_results"
else
  echo "   ⏭️  Table already exists: thor_scan_results"
fi

# ASGARD campaign reports table
if ! bq ls "$PROJECT_ID:soc_data" | grep -q "asgard_campaign_reports"; then
  bq mk --table "$PROJECT_ID:soc_data.asgard_campaign_reports" \
    campaign_id:STRING,campaign_name:STRING,created_at:TIMESTAMP,\
completed_at:TIMESTAMP,duration_minutes:FLOAT,total_targets:INTEGER,\
successfully_scanned:INTEGER,failed_scans:INTEGER,total_threats:INTEGER,\
critical_threats:INTEGER,tenant_id:STRING
  echo "   ✅ Created table: asgard_campaign_reports"
else
  echo "   ⏭️  Table already exists: asgard_campaign_reports"
fi

echo ""
echo "5️⃣  Creating GCS bucket..."
BUCKET="gs://$PROJECT_ID-valhalla-threat-intel"
if ! gsutil ls "$BUCKET" &>/dev/null; then
  gsutil mb -p "$PROJECT_ID" -l "$REGION" "$BUCKET"
  gsutil versioning set on "$BUCKET"
  echo "   ✅ Created bucket: $BUCKET"
else
  echo "   ⏭️  Bucket already exists: $BUCKET"
fi

echo ""
echo "✅ GCP resources setup complete!"
GCPSCRIPT

chmod +x scripts/setup_gcp_resources.sh

# Script 2: Quick test scan
cat > scripts/quick_test_scan.sh << 'TESTSCRIPT'
#!/bin/bash
# Quick test scan of sample data

cd "$(dirname "$0")/.."
source threat-hunting-env/bin/activate

echo "🔍 Running quick test scan..."
echo ""

python3 << 'PYEOF'
import sys
sys.path.insert(0, '.')

from thor_endpoint_agent import THOREndpointAgent, ScanType

# Initialize agent
agent = THOREndpointAgent(config_path="config/thor_config.json")

# Load rules
print("Loading YARA rules...")
agent.load_yara_rules()

# Scan test data
print("\nScanning test data directory...")
result = agent.perform_scan(
    scan_types=[ScanType.FILESYSTEM],
    target_paths=["test_data/"]
)

# Print results
print(f"\n{'='*60}")
print(f"Test Scan Results: {result.scan_id}")
print(f"{'='*60}")
print(f"Files scanned: {result.total_files_scanned}")
print(f"Threats detected: {len(result.matches)}")

if result.matches:
    print(f"\n⚠️  DETECTIONS (Expected for test files):")
    for match in result.matches:
        print(f"  [{match.severity.value.upper()}] {match.rule_name}")
        print(f"    File: {match.target}")
else:
    print(f"\n✅ No threats detected")

print(f"\nNote: test_data/suspicious/ SHOULD trigger detections")
PYEOF

echo ""
echo "✅ Test scan complete!"
TESTSCRIPT

chmod +x scripts/quick_test_scan.sh

# Script 3: Fetch threat intel
cat > scripts/update_threat_intel.sh << 'INTELSCRIPT'
#!/bin/bash
# Update threat intelligence feeds

cd "$(dirname "$0")/.."
source threat-hunting-env/bin/activate

echo "📡 Updating threat intelligence feeds..."
echo ""

python3 << 'PYEOF'
import sys
sys.path.insert(0, '.')

from valhalla_feed_manager import VALHALLAFeedManager

# Initialize VALHALLA
valhalla = VALHALLAFeedManager(config_path="config/valhalla_config.json")

# Update feeds
print("Fetching IOCs from ThreatFox and MalwareBazaar...")
stats = valhalla.update_all_feeds()

print(f"\n📊 Feed Update Summary:")
for feed, count in stats.items():
    print(f"   {feed}: {count} new items")

# Export for THOR
print(f"\nExporting IOCs for THOR agents...")
ioc_export = valhalla.export_iocs_for_thor()

print(f"\n✅ Total IOCs exported: {ioc_export['metadata']['total_iocs']}")
print(f"   - IPs: {len(ioc_export['ips'])}")
print(f"   - Domains: {len(ioc_export['domains'])}")
print(f"   - Hashes: {len(ioc_export['file_hashes'])}")
PYEOF

echo ""
echo "✅ Threat intelligence updated!"
INTELSCRIPT

chmod +x scripts/update_threat_intel.sh

echo "✅ Helper scripts created in scripts/"
echo ""

# Create README
cat > README.md << 'README'
# Threat Hunting Test Environment

This directory contains a complete test environment for the AI-Driven SOC Threat Hunting Platform.

## Quick Start

### 1. Activate virtual environment
```bash
source threat-hunting-env/bin/activate
```

### 2. Setup GCP resources (one-time)
```bash
bash scripts/setup_gcp_resources.sh
```

### 3. Update threat intelligence
```bash
bash scripts/update_threat_intel.sh
```

### 4. Run test scan
```bash
bash scripts/quick_test_scan.sh
```

## Directory Structure

- `config/` - Configuration files for THOR, ASGARD, VALHALLA
- `logs/` - Scan logs and execution logs
- `yara_rules/` - YARA detection rules
- `test_data/` - Sample files for testing
- `results/` - Scan result reports
- `scripts/` - Helper scripts
- `threat-hunting-env/` - Python virtual environment

## Next Steps

1. Copy agent files (thor_endpoint_agent.py, etc.) to this directory
2. Run full platform test
3. Create real threat hunting campaigns

## Documentation

See parent directory for:
- THREAT_HUNTING_README.md
- GCP_VM_TESTING_GUIDE.md
- Pure_Platform_Deployment_Guide.docx
README

echo "✅ README.md created"
echo ""

# Create environment activation script
cat > activate.sh << 'ACTIVATE'
#!/bin/bash
# Activate threat hunting test environment

cd "$(dirname "$0")"
source threat-hunting-env/bin/activate

echo "✅ Threat Hunting environment activated"
echo ""
echo "Available scripts:"
echo "  ./scripts/setup_gcp_resources.sh    - Setup GCP resources"
echo "  ./scripts/update_threat_intel.sh    - Update threat intelligence"
echo "  ./scripts/quick_test_scan.sh        - Run quick test scan"
echo ""
echo "Documentation:"
echo "  cat README.md                       - Read this first"
echo ""
ACTIVATE

chmod +x activate.sh

# Summary
echo "========================================================"
echo "  ✅ Setup Complete!"
echo "========================================================"
echo ""
echo "📁 Test environment created at: $TEST_DIR"
echo ""
echo "📋 Next steps:"
echo ""
echo "1️⃣  Copy agent files to test directory:"
echo "   cd $TEST_DIR"
echo "   # Upload: thor_endpoint_agent.py"
echo "   # Upload: asgard_orchestration_agent.py"
echo "   # Upload: valhalla_feed_manager.py"
echo ""
echo "2️⃣  Setup GCP resources:"
echo "   bash scripts/setup_gcp_resources.sh"
echo ""
echo "3️⃣  Update threat intelligence:"
echo "   bash scripts/update_threat_intel.sh"
echo ""
echo "4️⃣  Run test scan:"
echo "   bash scripts/quick_test_scan.sh"
echo ""
echo "📚 Documentation:"
echo "   cat README.md"
echo ""
echo "🚀 Quick activate:"
echo "   source $TEST_DIR/activate.sh"
echo ""
echo "========================================================"
