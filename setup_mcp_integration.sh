#!/bin/bash
# Setup script for TAA A2A MCP Integration
# This script sets up the environment and dependencies for MCP integration with Google Security Operations

set -e

echo "🚀 Setting up TAA A2A MCP Integration"
echo "======================================"

# Check if Python 3.9+ is available
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "📋 Python version: $python_version"

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  No virtual environment detected. Consider creating one:"
    echo "   python3 -m venv venv"
    echo "   source venv/bin/activate"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -U pip
pip install -r requirements_mcp.txt

# Install Node.js dependencies for MCP servers
echo "📦 Installing Node.js dependencies for MCP servers..."
if command -v npm &> /dev/null; then
    echo "✅ npm found, installing MCP servers..."
    
    # Install Google MCP servers
    npm install -g @google/mcp-server-security-operations
    npm install -g @google/mcp-server-threat-intelligence  
    npm install -g @google/mcp-server-scc
    
    echo "✅ MCP servers installed successfully"
else
    echo "⚠️  npm not found. Please install Node.js and npm to use MCP servers."
    echo "   Visit: https://nodejs.org/"
    echo "   After installing, run:"
    echo "   npm install -g @google/mcp-server-security-operations"
    echo "   npm install -g @google/mcp-server-threat-intelligence"
    echo "   npm install -g @google/mcp-server-scc"
fi

# Check Google Cloud credentials
echo "🔐 Checking Google Cloud credentials..."
if [[ -f "Service Account BigQuery/sa-gatra-bigquery.json" ]]; then
    echo "✅ Service account key found"
    export GOOGLE_APPLICATION_CREDENTIALS="Service Account BigQuery/sa-gatra-bigquery.json"
else
    echo "⚠️  Service account key not found at 'Service Account BigQuery/sa-gatra-bigquery.json'"
    echo "   Please ensure your service account key is in the correct location."
fi

# Set environment variables
echo "🌍 Setting up environment variables..."
export GOOGLE_CLOUD_PROJECT="ai-driven-soc"
export GOOGLE_APPLICATION_CREDENTIALS="Service Account BigQuery/sa-gatra-bigquery.json"

# Create .env file for easy loading
cat > .env << EOF
# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT=ai-driven-soc
GOOGLE_APPLICATION_CREDENTIALS=Service Account BigQuery/sa-gatra-bigquery.json

# MCP Configuration
MCP_ENABLED=true

# Chronicle Configuration (update with your customer ID)
CHRONICLE_CUSTOMER_ID=

# Server Configuration
HOST=0.0.0.0
PORT=8080
EOF

echo "✅ Environment variables configured in .env file"

# Create Pub/Sub topics if they don't exist
echo "📡 Setting up Pub/Sub topics..."
if command -v gcloud &> /dev/null; then
    echo "✅ gcloud CLI found"
    
    # Create topics
    gcloud pubsub topics create chronicle-alerts --project=ai-driven-soc || echo "Topic chronicle-alerts may already exist"
    gcloud pubsub topics create threat-intelligence --project=ai-driven-soc || echo "Topic threat-intelligence may already exist"
    gcloud pubsub topics create containment-requests --project=ai-driven-soc || echo "Topic containment-requests may already exist"
    
    echo "✅ Pub/Sub topics configured"
else
    echo "⚠️  gcloud CLI not found. Please install Google Cloud SDK:"
    echo "   Visit: https://cloud.google.com/sdk/docs/install"
    echo "   After installing, create the following topics:"
    echo "   gcloud pubsub topics create chronicle-alerts --project=ai-driven-soc"
    echo "   gcloud pubsub topics create threat-intelligence --project=ai-driven-soc"
    echo "   gcloud pubsub topics create containment-requests --project=ai-driven-soc"
fi

# Create logs directory
mkdir -p logs
echo "✅ Created logs directory"

# Make scripts executable
chmod +x test_mcp_integration.py
echo "✅ Made test script executable"

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Update your Chronicle customer ID in config/taa_a2a_config.json"
echo "2. Start the MCP server: python3 taa_a2a_mcp_server.py"
echo "3. Test the integration: python3 test_mcp_integration.py"
echo ""
echo "📚 Documentation:"
echo "- MCP Integration: https://github.com/google/mcp-security"
echo "- Google Security Operations: https://cloud.google.com/security-operations"
echo "- Chronicle: https://cloud.google.com/chronicle"
echo ""
echo "🔧 Troubleshooting:"
echo "- Check logs in the logs/ directory"
echo "- Verify Google Cloud credentials"
echo "- Ensure MCP servers are installed: npm list -g | grep mcp-server"
echo ""
echo "Happy hunting! 🕵️‍♂️"


