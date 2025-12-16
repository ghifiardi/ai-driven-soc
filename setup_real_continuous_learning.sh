#!/bin/bash
"""
Setup REAL Continuous Learning System
This will create a working hourly retraining system without sklearn dependencies
"""

echo "🔧 Setting up REAL Continuous Learning System..."

# 1. Stop failing enhanced service
sudo systemctl stop cla_enhanced.service 2>/dev/null || true

# 2. Create a simple cron job for hourly retraining
echo "Creating hourly retraining cron job..."
(crontab -l 2>/dev/null | grep -v "cla_simple_enhancement.py"; echo "0 * * * * cd /home/app/ai-driven-soc && source venv/bin/activate && python3 cla_simple_enhancement.py >> cla_continuous_learning.log 2>&1") | crontab -

# 3. Create a monitoring script
cat > monitor_continuous_learning.sh << 'EOF'
#!/bin/bash
echo "📊 Continuous Learning Status Check"
echo "=================================="

# Check if cron job exists
if crontab -l | grep -q "cla_simple_enhancement.py"; then
    echo "✅ Hourly retraining cron job: ACTIVE"
else
    echo "❌ Hourly retraining cron job: NOT FOUND"
fi

# Check recent logs
echo ""
echo "📝 Recent Enhancement Logs:"
tail -n 10 cla_simple_enhancement.log 2>/dev/null || echo "No logs found"

# Check continuous learning logs
echo ""
echo "🔄 Continuous Learning Logs:"
tail -n 10 cla_continuous_learning.log 2>/dev/null || echo "No continuous learning logs found"

# Check BigQuery for latest metrics
echo ""
echo "📈 Latest Metrics in BigQuery:"
python3 -c "
from google.cloud import bigquery
import pandas as pd
try:
    client = bigquery.Client(project='chronicle-dev-2be9')
    query = 'SELECT timestamp, accuracy, false_positive_rate FROM \`chronicle-dev-2be9.soc_data.cla_metrics\` ORDER BY timestamp DESC LIMIT 3'
    df = client.query(query).to_dataframe()
    print(df.to_string(index=False))
except Exception as e:
    print(f'Error: {e}')
"

echo ""
echo "🎯 Next scheduled run:"
crontab -l | grep "cla_simple_enhancement.py"
EOF

chmod +x monitor_continuous_learning.sh

# 4. Run initial enhancement
echo "🚀 Running initial enhancement..."
source venv/bin/activate
python3 cla_simple_enhancement.py

echo ""
echo "✅ REAL Continuous Learning System Setup Complete!"
echo ""
echo "📋 What's Now ACTIVE:"
echo "- ✅ Hourly cron job for retraining"
echo "- ✅ Simple enhancement script (no sklearn issues)"
echo "- ✅ BigQuery integration"
echo "- ✅ Monitoring script"
echo ""
echo "🔍 To monitor: ./monitor_continuous_learning.sh"
echo "📝 Logs: tail -f cla_continuous_learning.log"
