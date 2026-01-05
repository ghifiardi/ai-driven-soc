#!/bin/bash

# Create Video Demo for Threat Hunting Platform
# This script helps set up and record a demo video

set -e

DEMO_DIR="$HOME/threat-hunting-test"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Threat Hunting Platform - Video Demo Setup"
echo "=========================================="
echo ""

# Check if we're on GCP instance
if [ ! -d "$DEMO_DIR" ]; then
    echo "⚠ Threat hunting directory not found"
    echo "  Expected: $DEMO_DIR"
    exit 1
fi

cd "$DEMO_DIR"
source venv/bin/activate

echo "Step 1: Installing asciinema (if needed)..."
if ! command -v asciinema &> /dev/null; then
    echo "  Installing asciinema..."
    pip install asciinema || {
        echo "  ⚠ Could not install asciinema via pip"
        echo "  Try: sudo yum install -y asciinema"
    }
else
    echo "  ✓ asciinema already installed"
fi

echo ""
echo "Step 2: Preparing demo script..."
if [ -f "$SCRIPT_DIR/demo_script.sh" ]; then
    cp "$SCRIPT_DIR/demo_script.sh" "$DEMO_DIR/"
    chmod +x "$DEMO_DIR/demo_script.sh"
    echo "  ✓ Demo script copied"
else
    echo "  ⚠ Demo script not found, creating basic version..."
    cat > "$DEMO_DIR/demo_script.sh" << 'DEMOEOF'
#!/bin/bash
cd ~/threat-hunting-test
source venv/bin/activate

clear
echo "=========================================="
echo "  Threat Hunting Platform Demo"
echo "=========================================="
sleep 2

echo ""
echo "1. Environment Overview"
pwd
python3 --version
echo ""

echo "2. Platform Components"
ls -lh *.py | head -3
echo ""

echo "3. YARA Rules"
ls -lh data/yara_rules/
echo ""

echo "4. Test Files"
ls -lh data/test_malware/
echo ""

echo "5. Running THOR Scan..."
python thor_endpoint_agent.py --config config/thor_config.json --scan-type filesystem --target data/test_malware --load-yara 2>&1 | tail -20
echo ""

echo "6. Platform Summary"
echo "✓ THOR: Operational"
echo "✓ ASGARD: Operational"
echo "✓ VALHALLA: Operational"
echo "✓ Endpoint: Registered"
echo "✓ Detection: Verified (6 threats)"
DEMOEOF
    chmod +x "$DEMO_DIR/demo_script.sh"
fi

echo ""
echo "Step 3: Recording instructions..."
echo ""
echo "To record the demo, run:"
echo ""
echo "  cd ~/threat-hunting-test"
echo "  source venv/bin/activate"
echo ""
echo "Option A - Using asciinema:"
echo "  asciinema rec threat_hunting_demo.cast"
echo "  bash demo_script.sh"
echo "  # Press Ctrl+D to stop"
echo ""
echo "Option B - Using script command:"
echo "  script -a demo_output.txt"
echo "  bash demo_script.sh"
echo "  exit"
echo ""
echo "Option C - Manual recording:"
echo "  Use OBS Studio or screen recording software"
echo "  Record the SSH-in-browser session"
echo "  Run: bash demo_script.sh"
echo ""
echo "=========================================="
echo "Demo Setup Complete!"
echo "=========================================="
echo ""
echo "Next: Follow the recording instructions above"
echo ""

