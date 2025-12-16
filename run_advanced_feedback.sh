#!/bin/bash
# Advanced Feedback and Dashboard Management Script

set -e

echo "🚀 Starting Advanced SOC Feedback and Monitoring..."

# 1. Generate sophisticated feedback
echo "🤖 Step 1: Generating sophisticated feedback..."
cd /home/app
source ai-driven-soc/venv/bin/activate
python3 advanced_feedback_monitor.py

# 2. Restart the Comprehensive SOC Dashboard
echo "📊 Step 2: Restarting dashboard with updated metrics..."
pkill -f "comprehensive_soc_dashboard.py" || echo "No dashboard to kill"
nohup python3 -m streamlit run comprehensive_soc_dashboard.py --server.port 8535 --server.address 0.0.0.0 --server.headless true > comprehensive_dashboard.log 2>&1 &

echo "✅ Step 3: Checking dashboard status..."
sleep 3
ss -tulpn | grep :8535 || netstat -tlnp | grep 8535 || echo "❌ Dashboard not listening on 8535"

echo ""
echo "🎉 Advanced feedback and monitoring complete!"
echo ""
echo "📍 Access your enhanced dashboard:"
echo "   🔗 http://10.45.254.19:8535"
echo ""
echo "📊 What you'll see:"
echo "   • Real-time accuracy trends"
echo "   • Advanced feedback controls"
echo "   • Sophisticated threat analysis"
echo "   • TI-enhanced reasoning"
echo ""
echo "🔄 The system will continue learning and improving accuracy over time!"
