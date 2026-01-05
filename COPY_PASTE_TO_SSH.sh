#!/bin/bash
# ==========================================
# COPY AND PASTE THIS ENTIRE SCRIPT INTO YOUR SSH-IN-BROWSER SESSION
# ==========================================

set -e

PROJECT_DIR="$HOME/threat-hunting-test"
PROJECT_ID="chronicle-dev-2be9"

echo "=========================================="
echo "Complete Threat Hunting Setup"
echo "=========================================="
echo ""

# Step 1: Create directory structure
echo "Step 1: Creating directory structure..."
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"
mkdir -p config logs data/yara_rules data/iocs data/sigma_rules scripts tests docs
echo "✓ Directory structure created"
echo ""

# Step 2: Create README
echo "Step 2: Creating README..."
cat > "$PROJECT_DIR/README.md" << 'EOF'
# Threat Hunting Test Environment

This directory contains the threat hunting test setup for the AI-Driven SOC platform.

## Quick Start

1. Upload threat hunting files (see UPLOAD_INSTRUCTIONS.txt)
2. Activate virtual environment: `source venv/bin/activate`
3. Run quick test: `bash scripts/quick_test.sh`
4. Initialize VALHALLA: `python valhalla_feed_manager.py`
EOF
echo "✓ README created"
echo ""

# Step 3: Create upload instructions
echo "Step 3: Creating upload instructions..."
cat > "$PROJECT_DIR/UPLOAD_INSTRUCTIONS.txt" << 'EOF'
# File Upload Instructions

## Files to Upload (use SSH-in-browser UPLOAD FILE button):

### Main Directory (~/threat-hunting-test/):
- thor_endpoint_agent.py
- asgard_orchestration_agent.py
- valhalla_feed_manager.py
- threat_hunting_quickstart.py
- requirements_threat_hunting.txt
- THREAT_HUNTING_README.md

### Config Directory (~/threat-hunting-test/config/):
- thor_config.json
- asgard_config.json
- valhalla_config.json

## After Upload:
1. Update project ID: bash scripts/update_config.sh
2. Setup environment: bash scripts/setup_env.sh
3. Test: bash scripts/quick_test.sh
EOF
echo "✓ Upload instructions created"
echo ""

# Step 4: Create setup environment script (with virtual environment option)
echo "Step 4: Creating environment setup script..."
cat > "$PROJECT_DIR/scripts/setup_env.sh" << 'EOF'
#!/bin/bash
set -e
cd "$(dirname "$0")/.."

USE_VENV="${1:-yes}"  # Default to yes, but can be disabled

if [ "$USE_VENV" = "yes" ]; then
    echo "Setting up Python virtual environment..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        echo "✓ Virtual environment created"
    fi
    
    # Activate and upgrade pip
    source venv/bin/activate
    pip install --upgrade pip
    
    echo "✓ Using virtual environment"
else
    echo "Setting up Python environment (system-wide)..."
    echo "⚠ Installing to system Python (use 'yes' as argument for venv)"
    pip3 install --upgrade pip --user
fi

# Install requirements
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
        pip install google-cloud-pubsub google-cloud-firestore google-cloud-bigquery
        pip install yara-python
        pip install requests
    else
        pip3 install --user google-cloud-pubsub google-cloud-firestore google-cloud-bigquery
        pip3 install --user yara-python
        pip3 install --user requests
    fi
fi

echo "✓ Environment setup complete!"
if [ "$USE_VENV" = "yes" ]; then
    echo ""
    echo "To activate: source venv/bin/activate"
else
    echo ""
    echo "⚠ Using system Python. Make sure Python can find user-installed packages."
fi
EOF
chmod +x "$PROJECT_DIR/scripts/setup_env.sh"
echo "✓ Environment setup script created (supports both venv and system Python)"
echo ""

# Step 5: Create config update script
echo "Step 5: Creating config update script..."
cat > "$PROJECT_DIR/scripts/update_config.sh" << 'EOFSCRIPT'
#!/bin/bash
set -e
cd "$(dirname "$0")/.."
PROJECT_ID="chronicle-dev-2be9"

echo "Updating configuration files with project ID: $PROJECT_ID"

if [ -f "config/thor_config.json" ]; then
    # Use sed compatible with both GNU and BSD sed
    if sed --version >/dev/null 2>&1; then
        # GNU sed
        sed -i "s/your-gcp-project-id/$PROJECT_ID/g" config/*.json
    else
        # BSD sed (macOS/Linux)
        sed -i '' "s/your-gcp-project-id/$PROJECT_ID/g" config/*.json 2>/dev/null || \
        sed -i "s/your-gcp-project-id/$PROJECT_ID/g" config/*.json
    fi
    echo "✓ Configuration files updated"
    echo "Verifying changes:"
    grep -h "project_id\|gcp_project_id" config/*.json | head -3
else
    echo "⚠ Config files not found. Please upload them first."
fi
EOFSCRIPT
chmod +x "$PROJECT_DIR/scripts/update_config.sh"
echo "✓ Config update script created"
echo ""

# Step 6: Create quick test script
echo "Step 6: Creating quick test script..."
cat > "$PROJECT_DIR/scripts/quick_test.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "$0")/.."

echo "Running quick tests..."
echo ""

# Check if venv exists and activate if available
if [ -d "venv" ]; then
    echo "Using virtual environment..."
    source venv/bin/activate
else
    echo "⚠ No virtual environment found. Using system Python."
    echo "  (This is OK if you installed packages system-wide)"
fi

# Test Python version
echo "Python version:"
python3 --version
echo ""

# Test imports
echo "Testing imports..."
python3 << 'PYTEST'
import sys

# Test GCP libraries
try:
    from google.cloud import pubsub_v1, firestore, bigquery
    print("✓ GCP libraries imported successfully")
except ImportError as e:
    print(f"✗ GCP libraries import failed: {e}")
    sys.exit(1)

# Test YARA
try:
    import yara
    print("✓ YARA library imported successfully")
except ImportError as e:
    print(f"✗ YARA library import failed: {e}")
    print("  Install with: pip install yara-python")

# Test other dependencies
try:
    import requests
    print("✓ Requests library imported successfully")
except ImportError as e:
    print(f"✗ Requests library import failed: {e}")

print("\n✓ Quick test completed!")
PYTEST

echo ""
echo "Checking files..."
if [ -f "thor_endpoint_agent.py" ]; then
    echo "✓ thor_endpoint_agent.py found"
else
    echo "⚠ thor_endpoint_agent.py not found"
fi

if [ -f "asgard_orchestration_agent.py" ]; then
    echo "✓ asgard_orchestration_agent.py found"
else
    echo "⚠ asgard_orchestration_agent.py not found"
fi

if [ -f "valhalla_feed_manager.py" ]; then
    echo "✓ valhalla_feed_manager.py found"
else
    echo "⚠ valhalla_feed_manager.py not found"
fi

if [ -d "config" ] && [ -f "config/thor_config.json" ]; then
    echo "✓ Config files found"
else
    echo "⚠ Config files not found"
fi
EOF
chmod +x "$PROJECT_DIR/scripts/quick_test.sh"
echo "✓ Quick test script created"
echo ""

# Step 7: Create verification script
echo "Step 7: Creating verification script..."
cat > "$PROJECT_DIR/scripts/verify_setup.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "$0")/.."

echo "Verifying setup..."
echo ""

# Check directory structure
echo "Directory structure:"
find . -maxdepth 2 -type d | sort
echo ""

# Check Python files
echo "Python files:"
ls -lh *.py 2>/dev/null || echo "⚠ No Python files found"
echo ""

# Check config files
echo "Config files:"
ls -lh config/*.json 2>/dev/null || echo "⚠ No config files found"
echo ""

# Check virtual environment
if [ -d "venv" ]; then
    echo "✓ Virtual environment exists"
    source venv/bin/activate
    echo "Python: $(python3 --version)"
    echo "Installed packages:"
    pip list | grep -E "(google-cloud|yara|requests)" || echo "⚠ Some packages may be missing"
else
    echo "⚠ Virtual environment not found. Run: bash scripts/setup_env.sh"
fi

echo ""
echo "Setup verification complete!"
EOF
chmod +x "$PROJECT_DIR/scripts/verify_setup.sh"
echo "✓ Verification script created"
echo ""

# Step 8: Display summary
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Directory created at: $PROJECT_DIR"
echo ""
echo "Directory structure:"
tree -L 2 2>/dev/null || find . -maxdepth 2 -type d | sed 's|[^/]*/| |g'
echo ""
echo "Next steps:"
echo "1. Upload threat hunting files (see UPLOAD_INSTRUCTIONS.txt)"
echo "2. Setup environment:"
echo "   - With virtual env: bash scripts/setup_env.sh"
echo "   - Without virtual env: bash scripts/setup_env.sh no"
echo "3. Run: bash scripts/update_config.sh"
echo "4. Run: bash scripts/quick_test.sh"
echo "5. Run: bash scripts/verify_setup.sh"
echo ""
echo "Current directory: $(pwd)"
echo ""
echo "Note: Virtual environment is recommended but optional."
echo "      Use 'bash scripts/setup_env.sh no' to install system-wide."
echo ""

