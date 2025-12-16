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
echo "🌐 Access at: http://10.45.254.19:8507"
echo "📊 Service status: sudo systemctl status langgraph-dashboard"
