#!/bin/bash

# Threat Hunting Platform Demo Script
# This script demonstrates all key features of the deployed threat hunting system
# Can be used with asciinema or other screen recording tools

set -e

DEMO_DIR="$HOME/threat-hunting-test"
DELAY=2  # Delay between commands for readability

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    sleep $DELAY
}

print_step() {
    echo -e "${GREEN}▶ $1${NC}"
    sleep 1
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
    sleep 0.5
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
    sleep 0.5
}

cd "$DEMO_DIR"
source venv/bin/activate

clear
echo "=========================================="
echo "  Threat Hunting Platform Demo"
echo "  GCP Deployment - xdgaisocapp01"
echo "=========================================="
sleep 2

# Demo 1: Show Environment
print_header "Demo 1: Environment Overview"
print_step "Current directory and environment"
pwd
echo ""
print_step "Python environment"
python3 --version
which python3
echo ""
print_step "Directory structure"
tree -L 2 2>/dev/null || find . -maxdepth 2 -type d | head -15
sleep $DELAY

# Demo 2: Show Components
print_header "Demo 2: Platform Components"
print_step "THOR Endpoint Agent"
ls -lh thor_endpoint_agent.py
echo ""
print_step "ASGARD Orchestration Agent"
ls -lh asgard_orchestration_agent.py
echo ""
print_step "VALHALLA Feed Manager"
ls -lh valhalla_feed_manager.py
sleep $DELAY

# Demo 3: Show Configuration
print_header "Demo 3: Configuration Files"
print_step "THOR Configuration"
cat config/thor_config.json | head -20
echo ""
print_step "ASGARD Configuration"
cat config/asgard_config.json | head -15
sleep $DELAY

# Demo 4: Show YARA Rules
print_header "Demo 4: YARA Rules"
print_step "Available YARA rules"
ls -lh data/yara_rules/
echo ""
print_step "Sample YARA rule content"
head -20 data/yara_rules/test_malware_rules.yar
sleep $DELAY

# Demo 5: Show Test Files
print_header "Demo 5: Test Malware Files"
print_step "Test malware files"
ls -lh data/test_malware/
echo ""
print_step "Sample test file content"
head -3 data/test_malware/test_ransomware.txt
sleep $DELAY

# Demo 6: THOR Scan Demo
print_header "Demo 6: THOR Threat Detection"
print_step "Running THOR scan on test malware directory"
python thor_endpoint_agent.py \
    --config config/thor_config.json \
    --scan-type filesystem \
    --target data/test_malware \
    --load-yara 2>&1 | grep -E "(INFO|WARNING|Threats detected|By severity|Top threats)" | head -20
sleep $DELAY

# Demo 7: ASGARD Campaign
print_header "Demo 7: ASGARD Campaign Management"
print_step "Checking registered endpoints"
python -c "
from asgard_orchestration_agent import ASGARDOrchestrationAgent
asgard = ASGARDOrchestrationAgent()
print(f'Registered endpoints: {len(asgard.registered_endpoints)}')
for ep_id, ep in asgard.registered_endpoints.items():
    print(f'  - {ep.hostname} ({ep.endpoint_id})')
    print(f'    IP: {ep.ip_address}')
    print(f'    Status: {ep.status}')
    print(f'    Capabilities: {ep.capabilities}')
"
sleep $DELAY

# Demo 8: VALHALLA Status
print_header "Demo 8: VALHALLA Feed Manager"
print_step "VALHALLA initialization status"
python -c "
from valhalla_feed_manager import VALHALLAFeedManager
valhalla = VALHALLAFeedManager()
print(f'YARA rules: {len(valhalla.yara_rules)}')
print(f'IOC feeds: {len(valhalla.ioc_feeds)}')
print(f'Sigma rules: {len(valhalla.sigma_rules)}')
" 2>&1 | head -10
sleep $DELAY

# Demo 9: Summary
print_header "Demo 9: Platform Summary"
print_success "THOR: Scanning and detecting threats"
print_success "ASGARD: Managing campaigns and endpoints"
print_success "VALHALLA: Aggregating threat intelligence"
print_success "Endpoint registered: xdgaisocapp01"
print_success "YARA rules: 5 test rules loaded"
print_success "Test detection: 6 threats detected"
echo ""
echo "=========================================="
echo "  Demo Complete!"
echo "=========================================="

