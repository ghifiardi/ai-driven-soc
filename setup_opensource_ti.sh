#!/bin/bash
# Setup script for Open-Source Threat Intelligence MCP Integration
# This script sets up the environment and dependencies for open-source threat intelligence

set -e

echo "🚀 Setting up Open-Source Threat Intelligence MCP Integration"
echo "============================================================="

# Check if Python 3.12+ is available
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "📋 Python version: $python_version"

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  No virtual environment detected. Consider creating one:"
    echo "   python3 -m venv venv_mcp"
    echo "   source venv_mcp/bin/activate"
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
    
    # Install filesystem MCP server locally
    npm install @modelcontextprotocol/server-filesystem
    
    echo "✅ MCP servers installed successfully"
else
    echo "⚠️  npm not found. Please install Node.js and npm to use MCP servers."
    echo "   Visit: https://nodejs.org/"
fi

# Create MCP data directory
echo "📁 Creating MCP data directory..."
mkdir -p /tmp/mcp-data
echo "✅ MCP data directory created"

# Set environment variables
echo "🌍 Setting up environment variables..."
export GOOGLE_CLOUD_PROJECT="ai-driven-soc"
export GOOGLE_APPLICATION_CREDENTIALS="Service Account BigQuery/sa-gatra-bigquery.json"

# Create environment file template
echo "📝 Creating environment file template..."
cp mcp_env_template.txt .env

echo "✅ Environment file template created at .env"
echo ""
echo "🔑 IMPORTANT: Configure your API keys in .env file:"
echo "   - VirusTotal: https://www.virustotal.com/gui/join-us"
echo "   - AbuseIPDB: https://www.abuseipdb.com/register"
echo "   - Shodan: https://account.shodan.io/register"
echo ""

# Test the threat intelligence server
echo "🧪 Testing threat intelligence server..."
if python3 -c "import threat_intelligence_mcp_server" 2>/dev/null; then
    echo "✅ Threat intelligence server imports successfully"
else
    echo "❌ Threat intelligence server import failed"
    exit 1
fi

# Test the simulated Chronicle server
echo "🧪 Testing simulated Chronicle server..."
if python3 -c "import simulated_chronicle_server" 2>/dev/null; then
    echo "✅ Simulated Chronicle server imports successfully"
else
    echo "❌ Simulated Chronicle server import failed"
    exit 1
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Configure API keys in .env file (optional but recommended)"
echo "2. Start the MCP server: python3 taa_a2a_mcp_server.py"
echo "3. Test the integration: python3 test_mcp_integration.py"
echo ""
echo "📚 Available Threat Intelligence Sources:"
echo "✅ VirusTotal (requires free API key)"
echo "✅ AbuseIPDB (requires free API key)"
echo "✅ Shodan (requires free API key)"
echo "✅ ThreatFox (free, no API key required)"
echo "✅ MalwareBazaar (free, no API key required)"
echo ""
echo "🔧 API Key Setup:"
echo "1. VirusTotal: Sign up at https://www.virustotal.com/gui/join-us"
echo "2. AbuseIPDB: Sign up at https://www.abuseipdb.com/register"
echo "3. Shodan: Sign up at https://account.shodan.io/register"
echo "4. Add keys to .env file: VT_API_KEY=your_key_here"
echo ""
echo "🚀 Start the server:"
echo "   source venv_mcp/bin/activate"
echo "   python3 taa_a2a_mcp_server.py"
echo ""
echo "Happy threat hunting! 🕵️‍♂️"
