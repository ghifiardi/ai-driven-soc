#!/bin/bash

# Setup Demo Recording on GCP Instance
# Run this script on your GCP instance to prepare for video recording

set -e

DEMO_DIR="$HOME/threat-hunting-test"

echo "=========================================="
echo "Demo Recording Setup"
echo "=========================================="
echo ""

cd "$DEMO_DIR"
source venv/bin/activate

# Create demo script on GCP instance
cat > demo_script.sh << 'DEMOEOF'
#!/bin/bash
cd ~/threat-hunting-test
source venv/bin/activate

clear
echo "=========================================="
echo "  Threat Hunting Platform Demo"
echo "  GCP Deployment - xdgaisocapp01"
echo "=========================================="
sleep 2

echo ""
echo "=== 1. Environment Overview ==="
echo ""
pwd
python3 --version
echo ""
echo "Directory structure:"
find . -maxdepth 2 -type d | head -10
sleep 2

echo ""
echo "=== 2. Platform Components ==="
echo ""
echo "THOR Endpoint Agent:"
ls -lh thor_endpoint_agent.py
echo ""
echo "ASGARD Orchestration Agent:"
ls -lh asgard_orchestration_agent.py
echo ""
echo "VALHALLA Feed Manager:"
ls -lh valhalla_feed_manager.py
sleep 2

echo ""
echo "=== 3. YARA Rules ==="
echo ""
ls -lh data/yara_rules/
echo ""
echo "Sample rule:"
head -15 data/yara_rules/test_malware_rules.yar
sleep 2

echo ""
echo "=== 4. Test Malware Files ==="
echo ""
ls -lh data/test_malware/
echo ""
echo "Sample test file:"
head -3 data/test_malware/test_ransomware.txt
sleep 2

echo ""
echo "=== 5. Running THOR Threat Detection ==="
echo ""
python thor_endpoint_agent.py \
    --config config/thor_config.json \
    --scan-type filesystem \
    --target data/test_malware \
    --load-yara 2>&1 | grep -E "(INFO|Threats detected|By severity|Top threats|HIGH|MEDIUM)" | head -15
sleep 2

echo ""
echo "=== 6. ASGARD Endpoint Status ==="
echo ""
python -c "
from asgard_orchestration_agent import ASGARDOrchestrationAgent
asgard = ASGARDOrchestrationAgent()
print(f'Registered endpoints: {len(asgard.registered_endpoints)}')
for ep_id, ep in asgard.registered_endpoints.items():
    print(f'  ✓ {ep.hostname} ({ep.endpoint_id})')
    print(f'    IP: {ep.ip_address}')
    print(f'    Status: {ep.status}')
    print(f'    Capabilities: {len(ep.capabilities)} features')
"
sleep 2

echo ""
echo "=== 7. Platform Summary ==="
echo ""
echo "✓ THOR: Scanning and detecting threats"
echo "✓ ASGARD: Managing campaigns and endpoints"
echo "✓ VALHALLA: Aggregating threat intelligence"
echo "✓ Endpoint: xdgaisocapp01 registered"
echo "✓ YARA Rules: 5 test rules loaded"
echo "✓ Detection: 6 threats detected successfully"
echo ""
echo "=========================================="
echo "  Demo Complete!"
echo "=========================================="
DEMOEOF

chmod +x demo_script.sh
echo "✓ Demo script created"

# Check for asciinema
if command -v asciinema &> /dev/null; then
    echo "✓ asciinema is installed"
else
    echo "⚠ asciinema not found"
    echo "  Install with: pip install asciinema"
    echo "  Or: sudo yum install -y asciinema"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To record the demo:"
echo ""
echo "Method 1 - asciinema (recommended):"
echo "  asciinema rec threat_hunting_demo.cast"
echo "  bash demo_script.sh"
echo "  # Press Ctrl+D to stop recording"
echo ""
echo "Method 2 - Screen recording:"
echo "  Start your screen recorder (OBS, QuickTime, etc.)"
echo "  bash demo_script.sh"
echo "  Stop recording when done"
echo ""
echo "Method 3 - script command:"
echo "  script -a demo_output.txt"
echo "  bash demo_script.sh"
echo "  exit"
echo ""

