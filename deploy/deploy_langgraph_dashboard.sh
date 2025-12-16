#!/bin/bash

# LangGraph ADA Dashboard Deployment Script
# Deploys the specialized dashboard for monitoring LangGraph ADA agent integration

set -e

# Configuration
VM_IP="10.45.254.19"
VM_USER="app"
DASHBOARD_PORT="8507"
DASHBOARD_NAME="langgraph-ada-dashboard"

echo "🚀 Deploying LangGraph ADA Dashboard to VM..."

# Create deployment directory
DEPLOY_DIR="langgraph_dashboard_deploy_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$DEPLOY_DIR"

# Copy dashboard files
echo "📁 Copying dashboard files..."
cp langgraph_ada_dashboard.py "$DEPLOY_DIR/"
cp langgraph_dashboard_requirements.txt "$DEPLOY_DIR/"

# Create systemd service file
cat > "$DEPLOY_DIR/langgraph-dashboard.service" << EOF
[Unit]
Description=LangGraph ADA Dashboard
After=network.target

[Service]
Type=simple
User=app
WorkingDirectory=/home/app/langgraph-ada-dashboard
Environment=DASHBOARD_PORT=$DASHBOARD_PORT
ExecStart=/home/app/.local/bin/streamlit run langgraph_ada_dashboard.py --server.port $DASHBOARD_PORT --server.address 0.0.0.0 --server.headless true
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Create startup script
cat > "$DEPLOY_DIR/start_langgraph_dashboard.sh" << EOF
#!/bin/bash
cd /home/app/langgraph-ada-dashboard
export DASHBOARD_PORT=$DASHBOARD_PORT
/home/app/.local/bin/streamlit run langgraph_ada_dashboard.py --server.port \$DASHBOARD_PORT --server.address 0.0.0.0 --server.headless true
EOF

chmod +x "$DEPLOY_DIR/start_langgraph_dashboard.sh"

# Create Nginx configuration
cat > "$DEPLOY_DIR/langgraph-dashboard-nginx.conf" << EOF
server {
    listen 80;
    server_name $VM_IP;

    location /langgraph-dashboard/ {
        proxy_pass http://127.0.0.1:$DASHBOARD_PORT/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # WebSocket support for Streamlit
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
EOF

# Create README
cat > "$DEPLOY_DIR/README.md" << EOF
# LangGraph ADA Dashboard Deployment

## Overview
Specialized dashboard for monitoring LangGraph ADA agent integration, workflow states, and ML model performance.

## Files
- \`langgraph_ada_dashboard.py\` - Main dashboard application
- \`langgraph_dashboard_requirements.txt\` - Python dependencies
- \`langgraph-dashboard.service\` - Systemd service file
- \`start_langgraph_dashboard.sh\` - Startup script
- \`langgraph-dashboard-nginx.conf\` - Nginx configuration

## Features
- Real-time LangGraph workflow monitoring
- ML model performance analytics
- ADA agent metrics
- Workflow state transitions visualization
- System health monitoring

## Access
- Dashboard: http://$VM_IP:$DASHBOARD_PORT
- Nginx Proxy: http://$VM_IP/langgraph-dashboard/

## Deployment Steps
1. Copy files to VM
2. Install dependencies
3. Configure systemd service
4. Configure Nginx
5. Start dashboard
EOF

# Create deployment script
cat > "$DEPLOY_DIR/deploy.sh" << EOF
#!/bin/bash

echo "🚀 Deploying LangGraph ADA Dashboard..."

# Create directory
mkdir -p /home/app/langgraph-ada-dashboard
cd /home/app/langgraph-ada-dashboard

# Install dependencies
echo "📦 Installing dependencies..."
/home/app/.local/bin/pip install -r langgraph_dashboard_requirements.txt

# Setup systemd service
echo "⚙️ Setting up systemd service..."
sudo cp langgraph-dashboard.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable langgraph-dashboard
sudo systemctl start langgraph-dashboard

# Check service status
echo "🔍 Checking service status..."
sudo systemctl status langgraph-dashboard --no-pager

echo "✅ LangGraph ADA Dashboard deployed successfully!"
echo "🌐 Access at: http://$VM_IP:$DASHBOARD_PORT"
echo "📊 Service status: sudo systemctl status langgraph-dashboard"
EOF

chmod +x "$DEPLOY_DIR/deploy.sh"

# Deploy to VM
echo "🚀 Copying files to VM..."
gcloud compute scp --recurse "$DEPLOY_DIR" "$VM_USER@xdgaisocapp01:/home/app/" --zone=asia-southeast2-a

echo "📦 Installing dependencies and starting dashboard..."
gcloud compute ssh "$VM_USER@xdgaisocapp01" --zone=asia-southeast2-a --command="cd /home/app/$DEPLOY_DIR && chmod +x deploy.sh && ./deploy.sh"

echo "✅ LangGraph ADA Dashboard deployment completed!"
echo "🌐 Dashboard accessible at: http://$VM_IP:$DASHBOARD_PORT"
echo "📁 Deployment files in: /home/app/$DEPLOY_DIR"
echo "🔧 Service management: sudo systemctl {start|stop|restart|status} langgraph-dashboard"
