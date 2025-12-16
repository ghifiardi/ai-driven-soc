#!/bin/bash

echo "🔄 Restarting Native Streamlit Dashboard"
echo "========================================"
echo ""

# Kill existing dashboard
echo "🛑 Stopping existing dashboard..."
gcloud compute ssh app@xdgaisocapp01 --zone=asia-southeast2-a --command="sudo pkill -f 'native_streamlit_dashboard.py'"

# Wait a moment
sleep 3

# Start new dashboard
echo "🚀 Starting new dashboard on port 8518..."
gcloud compute ssh app@xdgaisocapp01 --zone=asia-southeast2-a --command="cd /home/app/langgraph-ada-dashboard && nohup /home/app/.local/bin/streamlit run native_streamlit_dashboard.py --server.port 8518 --server.address 0.0.0.0 --server.headless true > dashboard.log 2>&1 &"

# Wait for startup
sleep 5

echo ""
echo "✅ Dashboard restarted successfully!"
echo "🌐 Access at: http://10.45.254.19:8518"
echo ""
echo "📊 This dashboard uses ONLY native Streamlit components"
echo "✅ No HTML/CSS rendering issues - should display perfectly!"
