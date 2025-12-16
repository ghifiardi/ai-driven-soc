#!/bin/bash
# Simple dashboard restart script

cd /home/app/ai-driven-soc
source venv/bin/activate

# Kill existing dashboard processes
pkill -f "streamlit run.*8535" || true
sleep 2

# Start dashboard in background
nohup python3 -m streamlit run complete_operational_dashboard.py \
    --server.port 8535 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.fileWatcherType none \
    --logger.level error \
    > dashboard.log 2>&1 &

echo "Dashboard restarted at $(date)"
sleep 5

# Check if it's running
if ss -tulpn | grep -q :8535; then
    echo "✅ Dashboard is running on port 8535"
else
    echo "❌ Dashboard failed to start"
fi


