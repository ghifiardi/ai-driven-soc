#!/bin/bash

# Standalone setup script for Threat Hunting test environment on GCP
# Run this script directly on your GCP instance via SSH

set -e

# Configuration
PROJECT_DIR="$HOME/threat-hunting-test"

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

1. Copy threat hunting files from local repository (use gcloud compute scp or upload via SSH-in-browser)

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
pip list | grep -E "(yara|google-cloud)" || echo "Some packages may not be installed yet"

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

# Create a file upload helper script
cat > "$PROJECT_DIR/scripts/upload_instructions.txt" << 'EOF'
# Instructions for Uploading Files

## Method 1: Using SSH-in-browser Upload
1. Click the "UPLOAD FILE" button in the SSH-in-browser interface
2. Upload these files from your local machine:
   - thor_endpoint_agent.py
   - asgard_orchestration_agent.py
   - valhalla_feed_manager.py
   - threat_hunting_quickstart.py
   - requirements_threat_hunting.txt
   - THREAT_HUNTING_README.md

3. Upload config files to the config/ directory:
   - config/thor_config.json
   - config/asgard_config.json
   - config/valhalla_config.json

## Method 2: Using gcloud compute scp (from local machine)
gcloud compute scp thor_endpoint_agent.py asgard_orchestration_agent.py valhalla_feed_manager.py \
  app@xdgaisocapp01:~/threat-hunting-test/ --zone=asia-southeast2-a

gcloud compute scp config/*.json \
  app@xdgaisocapp01:~/threat-hunting-test/config/ --zone=asia-southeast2-a

gcloud compute scp requirements_threat_hunting.txt \
  app@xdgaisocapp01:~/threat-hunting-test/ --zone=asia-southeast2-a

## Method 3: Using git clone (if repo is accessible)
cd ~/threat-hunting-test
git clone https://github.com/ghifiardi/ai-driven-soc.git temp_repo
cp temp_repo/thor_endpoint_agent.py .
cp temp_repo/asgard_orchestration_agent.py .
cp temp_repo/valhalla_feed_manager.py .
cp temp_repo/config/*.json config/
cp temp_repo/requirements_threat_hunting.txt .
rm -rf temp_repo
EOF

echo "✓ Upload instructions created"
echo ""

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Directory created at: $PROJECT_DIR"
echo ""
echo "Directory structure:"
tree -L 2 "$PROJECT_DIR" 2>/dev/null || find "$PROJECT_DIR" -maxdepth 2 -type d | sed 's|[^/]*/| |g'
echo ""
echo "Next steps:"
echo "1. Upload threat hunting files (see scripts/upload_instructions.txt)"
echo "2. Run: cd $PROJECT_DIR && bash scripts/setup_env.sh"
echo "3. Run: bash scripts/quick_test.sh"
echo ""

