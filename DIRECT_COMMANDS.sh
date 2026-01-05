#!/bin/bash
# Run these commands directly in your SSH session
# Copy and paste each section one at a time

# ==========================================
# SECTION 1: Create Directory Structure
# ==========================================
mkdir -p ~/threat-hunting-test
cd ~/threat-hunting-test
mkdir -p config logs data/yara_rules data/iocs data/sigma_rules scripts tests docs

echo "✓ Directory structure created"
ls -la

# ==========================================
# SECTION 2: Create Helper Scripts
# ==========================================

# Create setup_env.sh
cat > scripts/setup_env.sh << 'ENVEOF'
#!/bin/bash
set -e
cd "$(dirname "$0")/.."

USE_VENV="${1:-yes}"

if [ "$USE_VENV" = "yes" ]; then
    echo "Setting up Python virtual environment..."
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        echo "✓ Virtual environment created"
    fi
    source venv/bin/activate
    pip install --upgrade pip
    echo "✓ Using virtual environment"
else
    echo "Setting up Python environment (system-wide)..."
    pip3 install --upgrade pip --user
fi

if [ -f "requirements_threat_hunting.txt" ]; then
    echo "Installing from requirements_threat_hunting.txt..."
    if [ "$USE_VENV" = "yes" ]; then
        pip install -r requirements_threat_hunting.txt
    else
        pip3 install --user -r requirements_threat_hunting.txt
    fi
else
    echo "Installing basic requirements..."
    if [ "$USE_VENV" = "yes" ]; then
        pip install google-cloud-pubsub google-cloud-firestore google-cloud-bigquery yara-python requests
    else
        pip3 install --user google-cloud-pubsub google-cloud-firestore google-cloud-bigquery yara-python requests
    fi
fi

echo "✓ Environment setup complete!"
if [ "$USE_VENV" = "yes" ]; then
    echo "To activate: source venv/bin/activate"
fi
ENVEOF

chmod +x scripts/setup_env.sh
echo "✓ setup_env.sh created"

# Create update_config.sh
cat > scripts/update_config.sh << 'CONFIGEOF'
#!/bin/bash
set -e
cd "$(dirname "$0")/.."
PROJECT_ID="chronicle-dev-2be9"

echo "Updating configuration files with project ID: $PROJECT_ID"

if [ -f "config/thor_config.json" ]; then
    if sed --version >/dev/null 2>&1; then
        sed -i "s/your-gcp-project-id/$PROJECT_ID/g" config/*.json
    else
        sed -i '' "s/your-gcp-project-id/$PROJECT_ID/g" config/*.json 2>/dev/null || \
        sed -i "s/your-gcp-project-id/$PROJECT_ID/g" config/*.json
    fi
    echo "✓ Configuration files updated"
    grep -h "project_id\|gcp_project_id" config/*.json | head -3
else
    echo "⚠ Config files not found. Please upload them first."
fi
CONFIGEOF

chmod +x scripts/update_config.sh
echo "✓ update_config.sh created"

# Create quick_test.sh
cat > scripts/quick_test.sh << 'TESTEOF'
#!/bin/bash
cd "$(dirname "$0")/.."

echo "Running quick tests..."
echo ""

if [ -d "venv" ]; then
    echo "Using virtual environment..."
    source venv/bin/activate
else
    echo "⚠ No virtual environment found. Using system Python."
fi

echo "Python version:"
python3 --version
echo ""

echo "Testing imports..."
python3 << 'PYTEST'
import sys
try:
    from google.cloud import pubsub_v1, firestore, bigquery
    print("✓ GCP libraries imported successfully")
except ImportError as e:
    print(f"✗ GCP libraries import failed: {e}")
    sys.exit(1)
try:
    import yara
    print("✓ YARA library imported successfully")
except ImportError as e:
    print(f"✗ YARA library import failed: {e}")
try:
    import requests
    print("✓ Requests library imported successfully")
except ImportError as e:
    print(f"✗ Requests library import failed: {e}")
print("\n✓ Quick test completed!")
PYTEST

echo ""
echo "Checking files..."
[ -f "thor_endpoint_agent.py" ] && echo "✓ thor_endpoint_agent.py found" || echo "⚠ thor_endpoint_agent.py not found"
[ -f "asgard_orchestration_agent.py" ] && echo "✓ asgard_orchestration_agent.py found" || echo "⚠ asgard_orchestration_agent.py not found"
[ -f "valhalla_feed_manager.py" ] && echo "✓ valhalla_feed_manager.py found" || echo "⚠ valhalla_feed_manager.py not found"
[ -f "config/thor_config.json" ] && echo "✓ Config files found" || echo "⚠ Config files not found"
TESTEOF

chmod +x scripts/quick_test.sh
echo "✓ quick_test.sh created"

# Create README
cat > README.md << 'READMEEOF'
# Threat Hunting Test Environment

## Quick Start

1. Upload threat hunting files (see UPLOAD_INSTRUCTIONS.txt)
2. Setup environment: `bash scripts/setup_env.sh`
3. Update config: `bash scripts/update_config.sh`
4. Test: `bash scripts/quick_test.sh`

## Directory Structure

- `config/` - Configuration files
- `logs/` - Log files
- `data/` - YARA rules, IOCs, Sigma rules
- `scripts/` - Helper scripts
- `tests/` - Test files
- `docs/` - Documentation
READMEEOF

echo "✓ README created"

# Create upload instructions
cat > UPLOAD_INSTRUCTIONS.txt << 'UPLOADEOF'
# Files to Upload

## Main Directory (~/threat-hunting-test/):
- thor_endpoint_agent.py
- asgard_orchestration_agent.py
- valhalla_feed_manager.py
- threat_hunting_quickstart.py
- requirements_threat_hunting.txt
- THREAT_HUNTING_README.md

## Config Directory (~/threat-hunting-test/config/):
- thor_config.json
- asgard_config.json
- valhalla_config.json
UPLOADEOF

echo "✓ UPLOAD_INSTRUCTIONS.txt created"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Directory: ~/threat-hunting-test"
echo ""
echo "Next steps:"
echo "1. Upload threat hunting files (see UPLOAD_INSTRUCTIONS.txt)"
echo "2. Run: bash scripts/setup_env.sh"
echo "3. Run: bash scripts/update_config.sh"
echo "4. Run: bash scripts/quick_test.sh"
echo ""

