#!/bin/bash

# 🛡️ Hybrid Cyber Defense Agent Dashboard Deployment Script
# =========================================================
# Deploys dashboard to integrate with existing SOC infrastructure

set -e

echo "🚀 Starting Hybrid Cyber Defense Dashboard Deployment..."

# Configuration
PROJECT_DIR="/Users/raditio.ghifiardigmail.com/Downloads/ai-driven-soc"
DASHBOARD_PORT="8529"
SERVICE_NAME="hybrid-defense-dashboard"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running in correct directory
if [ ! -f "hybrid_defense_dashboard.py" ]; then
    print_error "Dashboard file not found. Please run from project directory."
    exit 1
fi

# Install/update required packages
print_status "Installing required packages..."
pip3 install streamlit plotly requests pandas numpy --quiet

# Check port availability
print_status "Checking port availability..."
if lsof -Pi :$DASHBOARD_PORT -sTCP:LISTEN -t >/dev/null ; then
    print_warning "Port $DASHBOARD_PORT is already in use!"
    print_status "Finding alternative port..."
    for port in {8530..8540}; do
        if ! lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
            DASHBOARD_PORT=$port
            print_success "Using alternative port: $port"
            break
        fi
    done
fi

# Create systemd service file
print_status "Creating systemd service file..."
sudo tee /etc/systemd/system/${SERVICE_NAME}.service > /dev/null <<EOF
[Unit]
Description=Hybrid Cyber Defense Agent Dashboard
Documentation=SOC Dashboard for DQN + Gemini Agent
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$(whoami)
Group=$(whoami)
WorkingDirectory=${PROJECT_DIR}
Environment=PYTHONPATH=${PROJECT_DIR}
ExecStart=/usr/bin/python3 -m streamlit run hybrid_defense_dashboard.py --server.port=${DASHBOARD_PORT} --server.address=0.0.0.0 --server.headless=true
ExecReload=/bin/kill -HUP \$MAINPID
KillMode=mixed
KillSignal=SIGTERM
RestartSec=5
Restart=on-failure
RestartPreventExitStatus=0

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=${PROJECT_DIR}

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=${SERVICE_NAME}

# Resource limits
LimitNOFILE=65536
MemoryMax=1G

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd and start service
print_status "Reloading systemd and starting service..."
sudo systemctl daemon-reload
sudo systemctl enable ${SERVICE_NAME}.service
sudo systemctl start ${SERVICE_NAME}.service

# Wait a moment for service to start
sleep 3

# Check service status
if sudo systemctl is-active --quiet ${SERVICE_NAME}.service; then
    print_success "Dashboard service started successfully!"
else
    print_error "Failed to start dashboard service"
    sudo systemctl status ${SERVICE_NAME}.service
    exit 1
fi

# Display access information
echo
echo "🎉 Deployment Complete!"
echo "======================="
echo
echo -e "${GREEN}✅ Dashboard Status:${NC} Running"
echo -e "${BLUE}🌐 Access URL:${NC} http://localhost:${DASHBOARD_PORT}"
echo -e "${BLUE}🖥️ Service Name:${NC} ${SERVICE_NAME}.service"
echo -e "${BLUE}📊 Port:${NC} ${DASHBOARD_PORT}"
echo

# Integration with existing SOC infrastructure
echo -e "${YELLOW}🔗 SOC Integration:${NC}"
echo "   • Main SOC Dashboard: http://10.45.254.19:8511"
echo "   • AI Training Dashboard: http://10.45.254.19:8503"
echo "   • Real Data Dashboard: http://10.45.254.19:8527"
echo "   • Production Correlator: http://10.45.254.19:8528"
echo "   • NEW - Hybrid Defense: http://10.45.254.19:${DASHBOARD_PORT}"
echo

echo -e "${BLUE}📋 Management Commands:${NC}"
echo "   • Start:   sudo systemctl start ${SERVICE_NAME}"
echo "   • Stop:    sudo systemctl stop ${SERVICE_NAME}"
echo "   • Restart: sudo systemctl restart ${SERVICE_NAME}"
echo "   • Status:  sudo systemctl status ${SERVICE_NAME}"
echo "   • Logs:    sudo journalctl -u ${SERVICE_NAME} -f"
echo

# Test dashboard accessibility
print_status "Testing dashboard accessibility..."
sleep 2
if curl -s http://localhost:${DASHBOARD_PORT} > /dev/null; then
    print_success "Dashboard is accessible!"
else
    print_warning "Dashboard may still be starting up. Please wait 30 seconds and try accessing manually."
fi

# Optional: Open browser (if in desktop environment)
if command -v open &> /dev/null; then
    read -p "Open dashboard in browser? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        open http://localhost:${DASHBOARD_PORT}
    fi
fi

echo
print_success "Hybrid Cyber Defense Dashboard deployed successfully! 🎉"
echo -e "${GREEN}Dashboard URL: http://localhost:${DASHBOARD_PORT}${NC}"
