#!/bin/bash
# Setup script for CRA SOAR MCP Integration
# This script sets up the enhanced CRA with SOAR and MCP capabilities

set -e

echo "🚀 Setting up CRA SOAR MCP Integration"
echo "====================================="

# Check if we're in the right directory
if [ ! -f "cra_soar_mcp_agent.py" ]; then
    echo "❌ Error: Please run this script from the ai-driven-soc directory"
    exit 1
fi

# Check Python version
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

# Install additional dependencies for CRA SOAR
echo "📦 Installing additional dependencies for CRA SOAR..."
pip install structlog aiohttp uvicorn

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p logs
mkdir -p data/incidents
mkdir -p data/workflows

# Set up environment variables
echo "🌍 Setting up environment variables..."
export GOOGLE_CLOUD_PROJECT="ai-driven-soc"
export GOOGLE_APPLICATION_CREDENTIALS="Service Account BigQuery/sa-gatra-bigquery.json"

# Create environment file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating environment file..."
    cp mcp_env_template.txt .env
    echo "✅ Environment file created at .env"
    echo ""
    echo "🔑 IMPORTANT: Configure your API keys in .env file:"
    echo "   - VirusTotal: https://www.virustotal.com/gui/join-us"
    echo "   - AbuseIPDB: https://www.abuseipdb.com/register"
    echo "   - Shodan: https://account.shodan.io/register"
    echo ""
fi

# Test the CRA SOAR MCP agent
echo "🧪 Testing CRA SOAR MCP agent..."
if python3 -c "import cra_soar_mcp_agent" 2>/dev/null; then
    echo "✅ CRA SOAR MCP agent imports successfully"
else
    echo "❌ CRA SOAR MCP agent import failed"
    exit 1
fi

# Test the SOAR workflow server
echo "🧪 Testing SOAR workflow server..."
if python3 -c "import soar_workflow_server" 2>/dev/null; then
    echo "✅ SOAR workflow server imports successfully"
else
    echo "❌ SOAR workflow server import failed"
    exit 1
fi

# Test the test client
echo "🧪 Testing integration test client..."
if python3 -c "import test_cra_soar_mcp" 2>/dev/null; then
    echo "✅ Integration test client imports successfully"
else
    echo "❌ Integration test client import failed"
    exit 1
fi

echo ""
echo "🎉 CRA SOAR MCP Integration setup completed successfully!"
echo ""
echo "📋 Available Components:"
echo "✅ CRA SOAR MCP Agent - Enhanced containment with TI integration"
echo "✅ SOAR Workflow Server - RESTful API for workflow management"
echo "✅ Threat Intelligence MCP Server - Multi-source TI queries"
echo "✅ Integration Test Suite - Comprehensive testing framework"
echo "✅ Configuration Files - Complete configuration templates"
echo ""
echo "🚀 Quick Start Commands:"
echo ""
echo "1. Start the SOAR Workflow Server:"
echo "   python3 soar_workflow_server.py"
echo ""
echo "2. Test the integration (in another terminal):"
echo "   python3 test_cra_soar_mcp.py"
echo ""
echo "3. Test individual components:"
echo "   python3 cra_soar_mcp_agent.py  # Test CRA agent"
echo ""
echo "📚 Available SOAR Playbooks:"
echo "✅ Malware Containment - Isolate infected hosts and block malicious IPs"
echo "✅ Phishing Response - Block phishing URLs and monitor accounts"
echo "✅ DDoS Mitigation - Enable DDoS protection and escalate to network team"
echo "✅ Insider Threat Response - Disable accounts and escalate to legal"
echo "✅ Critical Vulnerability Response - Apply patches and isolate systems"
echo ""
echo "🔧 Key Features:"
echo "✅ MCP-Enhanced Threat Intelligence - Multi-source TI enrichment"
echo "✅ Intelligent Playbook Selection - Automatic playbook matching"
echo "✅ Approval Workflows - Human approval for high-risk actions"
echo "✅ Real-time Monitoring - Live workflow status tracking"
echo "✅ RESTful API - Complete API for integration"
echo "✅ Comprehensive Testing - Full test suite for validation"
echo ""
echo "📖 Documentation:"
echo "📄 CRA_SOAR_INTEGRATION_GUIDE.md - Complete integration guide"
echo "📄 MCP_INTEGRATION_GUIDE.md - MCP integration documentation"
echo "📄 OPENSOURCE_TI_SETUP_COMPLETE.md - Threat intelligence setup"
echo ""
echo "🎯 Next Steps:"
echo "1. Configure API keys in .env file (optional but recommended)"
echo "2. Start the SOAR server: python3 soar_workflow_server.py"
echo "3. Run integration tests: python3 test_cra_soar_mcp.py"
echo "4. Integrate with your existing ADA and TAA agents"
echo ""
echo "🔗 Integration Points:"
echo "✅ MCP Threat Intelligence - Enriches incidents with TI data"
echo "✅ A2A Communication - Coordinates with other agents"
echo "✅ Google Cloud Services - Vertex AI, Pub/Sub, Firestore"
echo "✅ RESTful API - Easy integration with external systems"
echo ""
echo "Happy automating! 🤖🛡️"


