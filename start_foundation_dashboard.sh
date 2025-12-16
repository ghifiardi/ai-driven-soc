#!/bin/bash
# Start FOUNDATION Dashboard - Stable Version

echo "🛑 Stopping all dashboard processes..."
pkill -9 -f "auto_restart_dashboard" || true
pkill -9 -f "streamlit" || true

sleep 5

echo "🚀 Starting FOUNDATION Dashboard (complete_operational_dashboard.py)..."
cd /home/app/ai-driven-soc
source venv/bin/activate

# Set Gemini API key if available
export GEMINI_API_KEY="AIzaSyCB1jWcqRrHUueOUxLA2kt0gJUuY7ng8Ac"

nohup python3 -m streamlit run complete_operational_dashboard.py \
    --server.port 8535 \
    --server.address 0.0.0.0 \
    --server.headless true \
    > /home/app/ai-driven-soc/foundation_dashboard.log 2>&1 &

sleep 5

echo "✅ Dashboard started!"
ps aux | grep "complete_operational_dashboard" | grep -v grep
netstat -tlnp | grep 8535

