# AI-Driven SOC Operations Platform with GLM-4.6

## 🚀 Overview
Custom SOC operations platform powered by GLM-4.6 AI for real-time threat analysis, incident response, and compliance monitoring.

## ✨ Features
- 🤖 **AI-Powered Threat Analysis**: GLM-4.6 analyzes security alerts in real-time
- 🚨 **Real-time Alert Processing**: Automated alert correlation and prioritization
- 📊 **Live Dashboard**: Real-time metrics and visualizations
- 🔄 **Incident Response**: AI-generated response plans
- 📋 **Compliance Monitoring**: Automated compliance checking
- ⚡ **WebSocket Updates**: Real-time notifications

## 🛠️ Setup Instructions

### 1. Install Dependencies
```bash
python3 -m pip install flask flask-socketio requests python-socketio eventlet python-dotenv
```

### 2. Configure GLM-4.6 API
Create a `.env` file and set your GLM-4.6 API key:
```bash
echo "GLM_API_KEY=your_actual_glm_api_key_here" > .env
```

### 3. Start the Platform
```bash
python3 soc_glm_app_simple.py
```

### 4. Access Dashboard
Open browser: http://localhost:5000

## 🔧 API Endpoints

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

## 🤖 GLM-4.6 Integration

The platform uses GLM-4.6 for:
- **Threat Analysis**: Real-time alert analysis and classification
- **Incident Response**: Automated response plan generation
- **Compliance Checking**: Regulatory compliance assessment
- **AI Reasoning**: Natural language explanations for security events

## 📊 Dashboard Features

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

## 🔒 Security Features
- Encrypted API communications
- Secure WebSocket connections
- Audit trail logging
- Compliance monitoring

## 📁 File Structure
```
Post Human SOC/
├── soc_glm_app_simple.py          # Main application
├── templates/
│   └── soc_dashboard.html          # Dashboard interface
├── .env                           # Configuration file
└── SOC_PLATFORM_README.md         # This documentation
```

## 🚀 Quick Start

1. **Set API Key**:
   ```bash
   echo "GLM_API_KEY=your_glm_api_key_here" > .env
   ```

2. **Start Platform**:
   ```bash
   python3 soc_glm_app_simple.py
   ```

3. **Open Dashboard**: http://localhost:5000

4. **Test Features**:
   - Click "Create Test Alert" to generate sample alerts
   - Click "Critical Alert" for high-priority testing
   - Use "Compliance Check" to test regulatory assessment

## 🔧 Configuration

### Environment Variables
- `GLM_API_KEY`: Your GLM-4.6 API key (required)
- `SOC_DEBUG`: Enable debug mode (optional)
- `SOC_HOST`: Host address (default: 0.0.0.0)
- `SOC_PORT`: Port number (default: 5000)

### GLM-4.6 API Setup
1. Get API key from GLM-4.6 provider
2. Set in `.env` file
3. Restart application

## 📈 Usage Examples

### Create Alert
```bash
curl -X POST http://localhost:5000/api/alerts \
  -H "Content-Type: application/json" \
  -d '{
    "source": "Firewall",
    "severity": "High",
    "description": "Multiple failed login attempts detected"
  }'
```

### Create Incident
```bash
curl -X POST http://localhost:5000/api/incidents \
  -H "Content-Type: application/json" \
  -d '{
    "severity": "Critical",
    "attack_type": "Malware",
    "affected_systems": "Web Server, Database"
  }'
```

### Check Compliance
```bash
curl -X POST http://localhost:5000/api/compliance \
  -H "Content-Type: application/json" \
  -d '{
    "event": "Data access attempt",
    "data_type": "Personal Information",
    "user": "admin@company.com",
    "system": "Customer Database"
  }'
```

## 🐛 Troubleshooting

### Common Issues
1. **GLM-4.6 API Errors**: Check API key and network connectivity
2. **WebSocket Connection**: Ensure port 5000 is available
3. **Dependencies**: Run `pip install flask flask-socketio requests`

### Logs
Check console output for detailed error messages and status updates.

## 📞 Support
For technical support or questions about GLM-4.6 integration, refer to the platform documentation or contact the development team.

## 🎯 Next Steps
1. Configure your GLM-4.6 API key
2. Start the platform
3. Test with sample alerts
4. Integrate with your existing SOC tools
5. Customize for your organization's needs

Ready to transform your SOC operations with AI! 🚀