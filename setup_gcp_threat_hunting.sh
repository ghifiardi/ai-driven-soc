#!/bin/bash

# Setup script for Threat Hunting test environment on GCP
# Usage: Run this script on your GCP instance or via: gcloud compute ssh xdgaisocapp01 --zone=asia-southeast2-a --command="bash -s" < setup_gcp_threat_hunting.sh

set -e

# Configuration
PROJECT_DIR="$HOME/threat-hunting-test"
INSTANCE_NAME="xdgaisocapp01"
ZONE="asia-southeast2-a"
PROJECT_ID="chronicle-dev-2be9"

echo "=========================================="
echo "Setting up Threat Hunting Test Environment"
echo "=========================================="
echo ""

# Create main directory
echo "Creating directory structure..."
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Create subdirectories
mkdir -p config
mkdir -p logs
mkdir -p data/yara_rules
mkdir -p data/iocs
mkdir -p data/sigma_rules
mkdir -p scripts
mkdir -p tests
mkdir -p docs

echo "✓ Directory structure created"
echo ""

# Create a README for the test environment
cat > "$PROJECT_DIR/README.md" << 'EOF'
# Threat Hunting Test Environment

This directory contains the threat hunting test setup for the AI-Driven SOC platform.

## Directory Structure

- `config/` - Configuration files for THOR, ASGARD, and VALHALLA agents
- `logs/` - Log files from test runs
- `data/yara_rules/` - YARA rule files
- `data/iocs/` - IOC (Indicators of Compromise) files
- `data/sigma_rules/` - Sigma rule files for log analysis
- `scripts/` - Utility scripts
- `tests/` - Test files and test data
- `docs/` - Documentation

## Quick Start

1. Copy threat hunting files from local repository:
   ```bash
   # From your local machine:
   gcloud compute scp thor_endpoint_agent.py asgard_orchestration_agent.py valhalla_feed_manager.py \
     app@xdgaisocapp01:~/threat-hunting-test/ --zone=asia-southeast2-a
   
   gcloud compute scp config/*.json \
     app@xdgaisocapp01:~/threat-hunting-test/config/ --zone=asia-southeast2-a
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements_threat_hunting.txt
   ```

3. Configure GCP project ID in config files:
   ```bash
   # Update config files with your project ID
   sed -i 's/your-gcp-project-id/chronicle-dev-2be9/g' config/*.json
   ```

4. Initialize VALHALLA feed manager:
   ```bash
   python valhalla_feed_manager.py
   ```

5. Run a test scan:
   ```bash
   python thor_endpoint_agent.py --config config/thor_config.json --scan-type quick
   ```

## Notes

- Ensure you have proper GCP credentials configured
- Pub/Sub topics and Firestore collections will be auto-created on first use
- Check logs/ directory for detailed execution logs
EOF

echo "✓ README created"
echo ""

# Create a .gitignore file
cat > "$PROJECT_DIR/.gitignore" << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# Logs
logs/
*.log

# Data files
data/yara_rules/*.yar
data/iocs/*.json
data/sigma_rules/*.yml

# Config (may contain secrets)
config/*_secrets.json

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
EOF

echo "✓ .gitignore created"
echo ""

# Create a quick test script
cat > "$PROJECT_DIR/scripts/quick_test.sh" << 'EOF'
#!/bin/bash
# Quick test script for threat hunting components

echo "Running quick tests..."

# Test Python environment
echo "Checking Python environment..."
python3 --version
pip list | grep -E "(yara|google-cloud)"

# Test imports
echo "Testing imports..."
python3 -c "
try:
    from google.cloud import pubsub_v1, firestore
    from google.cloud import bigquery
    print('✓ GCP libraries imported successfully')
except ImportError as e:
    print(f'✗ Import error: {e}')

try:
    import yara
    print('✓ YARA library imported successfully')
except ImportError as e:
    print(f'✗ YARA not installed: {e}')
"

echo "Quick test completed!"
EOF

chmod +x "$PROJECT_DIR/scripts/quick_test.sh"

echo "✓ Quick test script created"
echo ""

# Create environment setup script
cat > "$PROJECT_DIR/scripts/setup_env.sh" << 'EOF'
#!/bin/bash
# Environment setup script

set -e

echo "Setting up Python virtual environment..."

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements (if file exists)
if [ -f "requirements_threat_hunting.txt" ]; then
    echo "Installing requirements..."
    pip install -r requirements_threat_hunting.txt
else
    echo "Installing basic requirements..."
    pip install google-cloud-pubsub google-cloud-firestore google-cloud-bigquery
    pip install yara-python
    pip install requests
fi

echo "✓ Environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
EOF

chmod +x "$PROJECT_DIR/scripts/setup_env.sh"

echo "✓ Environment setup script created"
echo ""

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Directory created at: $PROJECT_DIR"
echo ""
echo "Next steps:"
echo "1. Copy threat hunting files from your local repository"
echo "2. Run: cd $PROJECT_DIR && bash scripts/setup_env.sh"
echo "3. Run: bash scripts/quick_test.sh"
echo ""
echo "To copy files from local machine, use:"
echo "  gcloud compute scp <file> app@$INSTANCE_NAME:$PROJECT_DIR/ --zone=$ZONE"
echo ""

