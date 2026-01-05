#!/bin/bash

# Complete Setup Script for Threat Hunting Testing
# Executes all 4 next steps:
# 1. Register endpoints for ASGARD campaigns
# 2. Create custom YARA rules for testing
# 3. Set up GCP resources (Pub/Sub, BigQuery) for full integration
# 4. Test threat detection with sample malware signatures

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Complete Threat Hunting Test Setup"
echo "=========================================="
echo ""

# Step 1: Register Endpoint
echo "Step 1: Registering Endpoint with ASGARD..."
echo ""
cd ~/threat-hunting-test
source venv/bin/activate

if python "$SCRIPT_DIR/register_endpoint.py"; then
    echo "✓ Endpoint registered successfully"
else
    echo "⚠ Endpoint registration had issues (may need Firestore setup)"
fi

echo ""

# Step 2: Create Custom YARA Rules
echo "Step 2: Setting up Custom YARA Rules..."
echo ""

# Copy YARA rules to the test directory
if [ -f "$SCRIPT_DIR/data/yara_rules/test_malware_rules.yar" ]; then
    mkdir -p ~/threat-hunting-test/data/yara_rules
    cp "$SCRIPT_DIR/data/yara_rules/test_malware_rules.yar" ~/threat-hunting-test/data/yara_rules/
    echo "✓ YARA rules copied to ~/threat-hunting-test/data/yara_rules/"
else
    echo "⚠ YARA rules file not found"
fi

echo ""

# Step 3: Setup GCP Resources
echo "Step 3: Setting up GCP Resources..."
echo ""
echo "Note: This requires gcloud CLI with proper permissions"
echo "Run manually if needed: bash $SCRIPT_DIR/setup_gcp_resources.sh"
echo ""

# Step 4: Create Test Malware Files
echo "Step 4: Creating Test Malware Files..."
echo ""

if python3 "$SCRIPT_DIR/create_test_malware.py"; then
    echo "✓ Test malware files created"
    echo ""
    echo "Test files location: ~/threat-hunting-test/data/test_malware/"
else
    echo "⚠ Test file creation had issues"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Test THOR detection:"
echo "   cd ~/threat-hunting-test"
echo "   source venv/bin/activate"
echo "   python thor_endpoint_agent.py --config config/thor_config.json --scan-type filesystem --target data/test_malware"
echo ""
echo "2. Create ASGARD campaign:"
echo "   python asgard_orchestration_agent.py"
echo ""
echo "3. Update config to use local YARA rules:"
echo "   Edit config/thor_config.json to point yara_rules_path to ~/threat-hunting-test/data/yara_rules"
echo ""

