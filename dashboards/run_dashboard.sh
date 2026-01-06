#!/bin/bash

# SOC Dashboard Launcher Script
# Makes it easy to run any dashboard without manually activating venv

DASHBOARD_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DASHBOARD_DIR"

# Activate virtual environment
source venv/bin/activate

# Check which dashboard to run
DASHBOARD=${1:-enhanced_soc_dashboard.py}
PORT=${2:-8501}

echo "🚀 Starting SOC Dashboard..."
echo "📊 Dashboard: $DASHBOARD"
echo "🌐 Port: $PORT"
echo ""
echo "Access at: http://localhost:$PORT"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run the dashboard
streamlit run "$DASHBOARD" --server.port "$PORT"

