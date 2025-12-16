#!/bin/bash

# Twitter API Integration Setup Script for VM
# This script sets up real Twitter monitoring on your VM

echo "🐦 Setting up Twitter API Integration for SOC Monitoring"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "twitter_real_monitor.py" ]; then
    echo "❌ Error: Please run this script from the ai-driven-soc directory"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source soc_env/bin/activate

# Install Twitter-specific requirements
echo "📦 Installing Twitter API requirements..."
pip install tweepy>=4.0.0 python-dotenv>=0.19.0

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file..."
    touch .env
    echo "# Twitter API Configuration" >> .env
    echo "TWITTER_BEARER_TOKEN=your_bearer_token_here" >> .env
fi

# Create Twitter alerts table in BigQuery
echo "🗄️ Creating Twitter alerts table in BigQuery..."
python3 -c "
from twitter_real_monitor import RealTwitterSOCMonitor
monitor = RealTwitterSOCMonitor()
if monitor.create_twitter_alerts_table():
    print('✅ Twitter alerts table created successfully')
else:
    print('❌ Failed to create Twitter alerts table')
"

# Set up environment variables
echo "🔧 Setting up environment variables..."
export GEMINI_API_KEY="AIzaSyCB1jWcqRrHUueOUxLA2kt0gJUuY7ng8Ac"

# Create startup script for Twitter monitor
echo "📝 Creating Twitter monitor startup script..."
cat > start_twitter_monitor.sh << 'EOF'
#!/bin/bash
cd /home/app/ai-driven-soc
source soc_env/bin/activate
export GEMINI_API_KEY="AIzaSyCB1jWcqRrHUueOUxLA2kt0gJUuY7ng8Ac"
export TWITTER_BEARER_TOKEN="$(grep TWITTER_BEARER_TOKEN .env | cut -d'=' -f2)"
python3 twitter_real_monitor.py
EOF

chmod +x start_twitter_monitor.sh

# Create startup script for Twitter dashboard
echo "📝 Creating Twitter dashboard startup script..."
cat > start_twitter_dashboard.sh << 'EOF'
#!/bin/bash
cd /home/app/ai-driven-soc
source soc_env/bin/activate
export GEMINI_API_KEY="AIzaSyCB1jWcqRrHUueOUxLA2kt0gJUuY7ng8Ac"
export TWITTER_BEARER_TOKEN="$(grep TWITTER_BEARER_TOKEN .env | cut -d'=' -f2)"
streamlit run twitter_real_dashboard.py --server.port 8505 --server.address 0.0.0.0
EOF

chmod +x start_twitter_dashboard.sh

echo ""
echo "✅ Twitter API Integration Setup Complete!"
echo ""
echo "📋 Next Steps:"
echo "1. Get Twitter API access: https://developer.twitter.com/en/portal/dashboard"
echo "2. Create a Twitter App with API v2 access"
echo "3. Generate Bearer Token from Keys & Tokens tab"
echo "4. Edit .env file and replace 'your_bearer_token_here' with your actual token"
echo "5. Run: ./start_twitter_monitor.sh (to collect tweets)"
echo "6. Run: ./start_twitter_dashboard.sh (to view dashboard on port 8505)"
echo ""
echo "🔗 Dashboard URLs:"
echo "- Twitter Real Dashboard: http://your-vm-ip:8505"
echo "- Main SOC Dashboard: http://your-vm-ip:8501"
echo "- TikTok Dashboard: http://your-vm-ip:8504"
echo ""
echo "📊 To test the setup:"
echo "python3 setup_twitter_api.py"
























