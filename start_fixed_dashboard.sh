#!/bin/bash
# Start Fixed Alert Review Dashboard - PERMANENT

echo "🛑 Stopping all dashboard processes..."
pkill -9 -f "auto_restart_dashboard" || true
pkill -9 -f "complete_operational_dashboard" || true
pkill -9 -f "streamlit.*8535" || true

sleep 3

echo "🚀 Starting Fixed Alert Review Dashboard..."
cd /home/app/ai-driven-soc
source venv/bin/activate

nohup python3 -m streamlit run fixed_alert_review_dashboard.py \
    --server.port 8535 \
    --server.address 0.0.0.0 \
    --server.headless true \
    > /home/app/ai-driven-soc/fixed_dashboard_running.log 2>&1 &

sleep 3

echo "✅ Dashboard started!"
netstat -tlnp | grep 8535 || echo "⚠️  Dashboard may not be running on port 8535"

