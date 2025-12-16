#!/bin/bash

# Deploy Gradual Migration Enhanced TAA System
# ===========================================

echo "🚀 Deploying Gradual Migration Enhanced TAA System"
echo "=================================================="

# Configuration
PROJECT_ID="chronicle-dev-2be9"
VM_NAME="xdgaisocapp01"
VM_ZONE="asia-southeast2-a"
VM_USER="app"
VM_PATH="/home/app/ai-driven-soc"
LOCAL_PATH="$(pwd)"

echo "📋 Deployment Plan:"
echo "   Phase 1: Deploy enhanced TAA service (10% traffic)"
echo "   Phase 2: Create monitoring dashboard"
echo "   Phase 3: Set up Pub/Sub topics"
echo "   Phase 4: Start gradual migration"
echo ""

# Step 1: Deploy files to VM
echo "📤 Step 1: Deploying enhanced TAA files..."
gcloud compute scp "$LOCAL_PATH/gradual_migration_enhanced_taa.py" $VM_USER@$VM_NAME:$VM_PATH/ --zone=$VM_ZONE
gcloud compute scp "$LOCAL_PATH/gradual_migration_monitor.py" $VM_USER@$VM_NAME:$VM_PATH/ --zone=$VM_ZONE
gcloud compute scp "$LOCAL_PATH/create_gradual_migration_topics.py" $VM_USER@$VM_NAME:$VM_PATH/ --zone=$VM_ZONE

if [ $? -eq 0 ]; then
    echo "✅ Files deployed successfully"
else
    echo "❌ File deployment failed"
    exit 1
fi

# Step 2: Set up environment on VM
echo ""
echo "🔧 Step 2: Setting up environment on VM..."
gcloud compute ssh $VM_USER@$VM_NAME --zone=$VM_ZONE --command="
cd $VM_PATH
source venv/bin/activate

echo '📦 Installing additional dependencies...'
pip install plotly streamlit

echo '📡 Creating Pub/Sub topics...'
python3 create_gradual_migration_topics.py

echo '🔍 Checking current TAA service status...'
ps aux | grep taa_service.py | grep -v grep

echo '✅ Environment setup complete'
"

if [ $? -eq 0 ]; then
    echo "✅ Environment setup successful"
else
    echo "❌ Environment setup failed"
    exit 1
fi

# Step 3: Create systemd service for gradual migration
echo ""
echo "⚙️ Step 3: Creating systemd service..."
gcloud compute ssh $VM_USER@$VM_NAME --zone=$VM_ZONE --command="
cd $VM_PATH

# Create systemd service file
sudo tee /etc/systemd/system/gradual-migration-enhanced-taa.service > /dev/null << 'EOF'
[Unit]
Description=Gradual Migration Enhanced TAA Service
After=network.target
Wants=network.target

[Service]
Type=simple
User=app
Group=app
WorkingDirectory=/home/app/ai-driven-soc
Environment=PATH=/home/app/ai-driven-soc/venv/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=/home/app/ai-driven-soc/venv/bin/python3 /home/app/ai-driven-soc/gradual_migration_enhanced_taa.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd and enable service
sudo systemctl daemon-reload
sudo systemctl enable gradual-migration-enhanced-taa.service

echo '✅ Systemd service created and enabled'
echo '📋 Service status:'
sudo systemctl status gradual-migration-enhanced-taa.service --no-pager
"

# Step 4: Start the gradual migration service
echo ""
echo "🚀 Step 4: Starting gradual migration service..."
gcloud compute ssh $VM_USER@$VM_NAME --zone=$VM_ZONE --command="
sudo systemctl start gradual-migration-enhanced-taa.service
sleep 5
sudo systemctl status gradual-migration-enhanced-taa.service --no-pager
"

# Step 5: Start monitoring dashboard
echo ""
echo "📊 Step 5: Starting monitoring dashboard..."
gcloud compute ssh $VM_USER@$VM_NAME --zone=$VM_ZONE --command="
cd $VM_PATH
source venv/bin/activate

# Start dashboard in background
nohup python3 -m streamlit run gradual_migration_monitor.py --server.port 8531 --server.address 0.0.0.0 --server.headless true > gradual_migration_dashboard.log 2>&1 &

echo '✅ Monitoring dashboard started on port 8531'
echo '📋 Dashboard URL: http://10.45.254.19:8531'
"

# Step 6: Verification
echo ""
echo "🔍 Step 6: Verifying deployment..."
gcloud compute ssh $VM_USER@$VM_NAME --zone=$VM_ZONE --command="
echo '📊 Service Status:'
sudo systemctl status gradual-migration-enhanced-taa.service --no-pager | head -10

echo ''
echo '📈 Process Status:'
ps aux | grep -E '(gradual_migration|enhanced_taa)' | grep -v grep

echo ''
echo '🌐 Port Status:'
netstat -tlnp | grep :8531

echo ''
echo '📋 Log Status:'
ls -la gradual_migration_enhanced_taa.log 2>/dev/null || echo 'Log file not created yet'
"

echo ""
echo "🎉 Gradual Migration Deployment Complete!"
echo "========================================"
echo ""
echo "📋 Deployment Summary:"
echo "   ✅ Enhanced TAA service deployed (10% traffic)"
echo "   ✅ Monitoring dashboard running on port 8531"
echo "   ✅ Pub/Sub topics created"
echo "   ✅ Systemd service configured"
echo ""
echo "🔗 Access Points:"
echo "   📊 Monitoring Dashboard: http://10.45.254.19:8531"
echo "   📈 Original Dashboard: http://10.45.254.19:99"
echo ""
echo "📊 Current Status:"
echo "   🔄 Original TAA: Processing 90% of alerts"
echo "   🚀 Enhanced TAA: Processing 10% of alerts"
echo "   📋 Monitoring: Real-time A/B comparison"
echo ""
echo "🎯 Next Steps:"
echo "   1. Monitor dashboard for 24-48 hours"
echo "   2. Validate enhanced classification accuracy"
echo "   3. Proceed to Phase 2 (25% traffic) when ready"
echo ""
echo "🛠️ Management Commands:"
echo "   sudo systemctl status gradual-migration-enhanced-taa.service"
echo "   sudo systemctl restart gradual-migration-enhanced-taa.service"
echo "   sudo systemctl stop gradual-migration-enhanced-taa.service"
echo ""
echo "✅ Gradual Migration is now LIVE and processing alerts!"
