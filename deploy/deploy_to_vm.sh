#!/bin/bash

# AI Model Training Dashboard Deployment Script
# Deploy to VM: 10.45.254.19

echo "🚀 Deploying AI Model Training Dashboard to VM..."

# Configuration
VM_IP="10.45.254.19"
VM_USER="app"  # Update this to your VM username
DASHBOARD_PORT="8503"  # Different port to avoid conflicts
PROJECT_DIR="/home/app/ai-model-training-dashboard"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}📋 Deployment Configuration:${NC}"
echo "   VM IP: $VM_IP"
echo "   Port: $DASHBOARD_PORT"
echo "   Project Dir: $PROJECT_DIR"
echo ""

# Check if files exist locally
echo -e "${YELLOW}🔍 Checking local files...${NC}"
if [ ! -f "ai_model_training_dashboard.py" ]; then
    echo -e "${RED}❌ ai_model_training_dashboard.py not found!${NC}"
    exit 1
fi

if [ ! -f "ai_dashboard_requirements.txt" ]; then
    echo -e "${RED}❌ ai_dashboard_requirements.txt not found!${NC}"
    exit 1
fi

echo -e "${GREEN}✅ All required files found${NC}"
echo ""

# Create deployment package
echo -e "${YELLOW}📦 Creating deployment package...${NC}"
DEPLOY_DIR="ai_dashboard_deploy_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$DEPLOY_DIR"

# Copy files
cp ai_model_training_dashboard.py "$DEPLOY_DIR/"
cp ai_dashboard_requirements.txt "$DEPLOY_DIR/"
cp ai_dashboard_config.py "$DEPLOY_DIR/"

# Create production startup script
cat > "$DEPLOY_DIR/start_dashboard.sh" << 'EOF'
#!/bin/bash

# AI Model Training Dashboard Startup Script
cd "$(dirname "$0")"

# Set environment variables
export DASHBOARD_PORT=8502
export DASHBOARD_HOST=0.0.0.0

echo "🤖 Starting AI Model Training Dashboard..."
echo "   Port: $DASHBOARD_PORT"
echo "   Host: $DASHBOARD_HOST"
echo "   URL: http://10.45.254.19:$DASHBOARD_PORT"
echo ""

# Start Streamlit
python3 -m streamlit run ai_model_training_dashboard.py \
    --server.port $DASHBOARD_PORT \
    --server.address $DASHBOARD_HOST \
    --server.headless true \
    --browser.gatherUsageStats false
EOF

chmod +x "$DEPLOY_DIR/start_dashboard.sh"

# Create systemd service file
cat > "$DEPLOY_DIR/ai-dashboard.service" << EOF
[Unit]
Description=AI Model Training Dashboard
After=network.target

[Service]
User=app
Group=app
WorkingDirectory=$PROJECT_DIR
ExecStart=$PROJECT_DIR/start_dashboard.sh
Restart=always
RestartSec=10
Environment=DASHBOARD_PORT=$DASHBOARD_PORT
Environment=DASHBOARD_HOST=0.0.0.0

[Install]
WantedBy=multi-user.target
EOF

# Create nginx configuration
cat > "$DEPLOY_DIR/ai-dashboard-nginx.conf" << EOF
server {
    listen 80;
    server_name 10.45.254.19;
    
    # AI Model Training Dashboard
    location /ai-training/ {
        proxy_pass http://127.0.0.1:$DASHBOARD_PORT/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 86400;
    }
    
    # Existing dashboard (if any)
    location / {
        # Keep existing configuration
        return 200 "AI Dashboard available at /ai-training/";
        add_header Content-Type text/plain;
    }
}
EOF

echo -e "${GREEN}✅ Deployment package created: $DEPLOY_DIR${NC}"
echo ""

# Instructions for manual deployment
echo -e "${YELLOW}📋 Manual Deployment Instructions:${NC}"
echo ""
echo "1. Copy the deployment package to your VM:"
echo "   scp -r $DEPLOY_DIR $VM_USER@$VM_IP:/tmp/"
echo ""
echo "2. SSH into your VM:"
echo "   ssh $VM_USER@$VM_IP"
echo ""
echo "3. Install dependencies:"
echo "   sudo apt update"
echo "   sudo apt install -y python3 python3-pip python3-venv"
echo "   cd /tmp/$DEPLOY_DIR"
echo "   pip3 install -r ai_dashboard_requirements.txt"
echo ""
echo "4. Deploy to production directory:"
echo "   sudo mkdir -p $PROJECT_DIR"
echo "   sudo cp -r * $PROJECT_DIR/"
echo "   sudo chown -R $VM_USER:$VM_USER $PROJECT_DIR"
echo ""
echo "5. Install systemd service:"
echo "   sudo cp ai-dashboard.service /etc/systemd/system/"
echo "   sudo systemctl daemon-reload"
echo "   sudo systemctl enable ai-dashboard"
echo "   sudo systemctl start ai-dashboard"
echo ""
echo "6. Configure nginx (optional):"
echo "   sudo cp ai-dashboard-nginx.conf /etc/nginx/sites-available/ai-dashboard"
echo "   sudo ln -s /etc/nginx/sites-available/ai-dashboard /etc/nginx/sites-enabled/"
echo "   sudo nginx -t && sudo systemctl reload nginx"
echo ""
echo "7. Access your dashboard:"
echo "   Direct: http://$VM_IP:$DASHBOARD_PORT"
echo "   Via nginx: http://$VM_IP/ai-training/"
echo ""

echo -e "${GREEN}🎉 Deployment package ready!${NC}"
echo "   Package: $DEPLOY_DIR"
echo "   Follow the manual instructions above to deploy to your VM."
echo ""
