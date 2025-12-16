#!/bin/bash

# AI Model Training Dashboard Startup Script
cd "$(dirname "$0")"

# Set environment variables
export DASHBOARD_PORT=8503
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
