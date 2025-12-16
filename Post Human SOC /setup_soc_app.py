#!/usr/bin/env python3
"""
Setup script for AI-Driven SOC Operations Platform
Installs dependencies and configures the application
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required Python packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def create_env_file():
    """Create .env file for configuration"""
    env_content = """# GLM-4.6 API Configuration
GLM_API_KEY=your_glm_api_key_here
GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4/chat/completions

# SOC Platform Configuration
SOC_DEBUG=True
SOC_HOST=0.0.0.0
SOC_PORT=5000

# Security Configuration
SECRET_KEY=soc_glm_secret_key_2024
"""
    
    env_path = Path(".env")
    if not env_path.exists():
        with open(env_path, "w") as f:
            f.write(env_content)
        print("✅ Created .env file")
    else:
        print("ℹ️  .env file already exists")

def create_startup_script():
    """Create startup script"""
    startup_content = """#!/bin/bash
# AI-Driven SOC Operations Platform Startup Script

echo "🚀 Starting AI-Driven SOC Operations Platform with GLM-4.6"
echo "📊 Dashboard: http://localhost:5000"
echo "🔧 API Endpoints: /api/alerts, /api/incidents, /api/compliance"
echo "⚡ Real-time updates via WebSocket"
echo ""

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found. Please configure GLM_API_KEY"
    echo "   Edit .env file and set your GLM-4.6 API key"
fi

# Start the application
python3 soc_glm_app.py
"""
    
    with open("start_soc_platform.sh", "w") as f:
        f.write(startup_content)
    
    # Make executable
    os.chmod("start_soc_platform.sh", 0o755)
    print("✅ Created startup script: start_soc_platform.sh")

def create_documentation():
    """Create documentation file"""
    doc_content = """# AI-Driven SOC Operations Platform with GLM-4.6

## Overview
Custom SOC operations platform powered by GLM-4.6 AI for real-time threat analysis, incident response, and compliance monitoring.

## Features
- 🤖 **AI-Powered Threat Analysis**: GLM-4.6 analyzes security alerts in real-time
- 🚨 **Real-time Alert Processing**: Automated alert correlation and prioritization
- 📊 **Live Dashboard**: Real-time metrics and visualizations
- 🔄 **Incident Response**: AI-generated response plans
- 📋 **Compliance Monitoring**: Automated compliance checking
- ⚡ **WebSocket Updates**: Real-time notifications

## Setup Instructions

### 1. Install Dependencies
```bash
python3 setup_soc_app.py
```

### 2. Configure GLM-4.6 API
Edit `.env` file and set your GLM-4.6 API key:
```
GLM_API_KEY=your_actual_api_key_here
```

### 3. Start the Platform
```bash
./start_soc_platform.sh
```

Or manually:
```bash
python3 soc_glm_app.py
```

### 4. Access Dashboard
Open browser: http://localhost:5000

## API Endpoints

### Alerts
- `GET /api/alerts` - Get all alerts
- `POST /api/alerts` - Create new alert
- Real-time updates via WebSocket

### Incidents
- `POST /api/incidents` - Create new incident
- Real-time updates via WebSocket

### Compliance
- `POST /api/compliance` - Check compliance
- Returns compliance status and recommendations

## GLM-4.6 Integration

The platform uses GLM-4.6 for:
- **Threat Analysis**: Real-time alert analysis and classification
- **Incident Response**: Automated response plan generation
- **Compliance Checking**: Regulatory compliance assessment
- **AI Reasoning**: Natural language explanations for security events

## Dashboard Features

### Real-time Metrics
- Total alerts processed
- Active incidents
- Mean Time To Resolution (MTTR)
- Compliance score

### AI Analysis Display
- GLM-4.6 reasoning for each alert
- Confidence scores
- Recommended actions
- Threat level assessment

### Interactive Controls
- Create test alerts
- Generate incidents
- Compliance checks
- Real-time monitoring

## Security Features
- Encrypted API communications
- Secure WebSocket connections
- Audit trail logging
- Compliance monitoring

## Troubleshooting

### Common Issues
1. **GLM-4.6 API Errors**: Check API key and network connectivity
2. **WebSocket Connection**: Ensure port 5000 is available
3. **Dependencies**: Run `pip install -r requirements.txt`

### Logs
Check console output for detailed error messages and status updates.

## Support
For technical support or questions about GLM-4.6 integration, refer to the platform documentation or contact the development team.
"""
    
    with open("SOC_PLATFORM_README.md", "w") as f:
        f.write(doc_content)
    print("✅ Created documentation: SOC_PLATFORM_README.md")

def main():
    """Main setup function"""
    print("🚀 Setting up AI-Driven SOC Operations Platform with GLM-4.6")
    print("=" * 60)
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed during package installation")
        return False
    
    # Create configuration files
    create_env_file()
    create_startup_script()
    create_documentation()
    
    print("\n" + "=" * 60)
    print("✅ Setup completed successfully!")
    print("\n📋 Next Steps:")
    print("1. Edit .env file and set your GLM-4.6 API key")
    print("2. Run: ./start_soc_platform.sh")
    print("3. Open: http://localhost:5000")
    print("\n📚 Documentation: SOC_PLATFORM_README.md")
    print("🔧 Configuration: .env file")
    print("🚀 Startup: start_soc_platform.sh")
    
    return True

if __name__ == "__main__":
    main()