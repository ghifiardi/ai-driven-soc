# Quick Start Guide: TAA A2A Implementation

## 🚀 Get Started in 5 Minutes

This guide will help you quickly set up and test the TAA A2A (Agent-to-Agent) implementation.

## Prerequisites

- Python 3.8+ installed
- Google Cloud account (optional for full deployment)
- Basic knowledge of command line

## Step 1: Quick Demo (No Setup Required)

Run the demo script to see A2A communication in action:

```bash
python3 test_a2a_demo.py
```

This will show you:
- ✅ Agent discovery
- ✅ Security alert analysis
- ✅ Threat intelligence sharing
- ✅ Containment requests
- ✅ Collaborative analysis
- ✅ Multi-agent workflow

## Step 2: Local Development Setup

### 1. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install fastapi uvicorn aiohttp pydantic
```

### 3. Start A2A Server
```bash
python3 taa_a2a_server.py
```

The server will start on `http://localhost:8080`

### 4. Test the Server
```bash
# Health check
curl http://localhost:8080/health

# Agent discovery
curl http://localhost:8080/.well-known/agent.json

# Test A2A task
curl -X POST http://localhost:8080/a2a/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "test-001",
    "skill": "analyze_security_alert",
    "parameters": {
      "alert_data": {"alert_id": "test", "severity": "high"},
      "enrichment_level": "full"
    }
  }'
```

## Step 3: Full Deployment (Google Cloud)

### 1. Install Google Cloud SDK
```bash
# macOS
brew install google-cloud-sdk

# Ubuntu
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
```

### 2. Authenticate and Set Project
```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

### 3. Run Deployment Script
```bash
chmod +x deploy_taa_a2a.sh
./deploy_taa_a2a.sh
```

This will:
- ✅ Enable required APIs
- ✅ Create service accounts
- ✅ Set up Pub/Sub topics
- ✅ Install dependencies
- ✅ Deploy to Vertex AI

## Step 4: Test A2A Communication

### 1. Run Comprehensive Tests
```bash
python3 taa_a2a_test_client.py
```

### 2. Manual Testing
```bash
# Test all capabilities
curl -X POST http://localhost:8080/a2a/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "analysis-001",
    "skill": "analyze_security_alert",
    "parameters": {
      "alert_data": {
        "alert_id": "ALERT-001",
        "severity": "high",
        "source": "firewall"
      },
      "enrichment_level": "full"
    }
  }'
```

## 🔧 Configuration

### Basic Configuration
Edit `config/taa_a2a_config.json`:

```json
{
  "project_id": "your-project-id",
  "location": "us-central1",
  "a2a": {
    "enabled": true,
    "port": 8080
  }
}
```

### Environment Variables
```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"
export STAGING_BUCKET="gs://your-staging-bucket"
```

## 📊 Monitoring

### Health Check
```bash
curl http://localhost:8080/health
```

### Metrics
```bash
curl http://localhost:8080/metrics
```

### Logs
```bash
tail -f logs/taa_a2a.log
```

## 🛠️ Common Commands

### Start Services
```bash
# Start A2A server
python3 taa_a2a_server.py

# Start with custom port
python3 taa_a2a_server.py --port 9090

# Start with debug logging
LOG_LEVEL=DEBUG python3 taa_a2a_server.py
```

### Testing
```bash
# Run demo
python3 test_a2a_demo.py

# Run comprehensive tests
python3 taa_a2a_test_client.py

# Test specific capability
curl -X POST http://localhost:8080/a2a/tasks \
  -H "Content-Type: application/json" \
  -d '{"task_id": "test", "skill": "share_threat_intelligence", "parameters": {"indicators": ["test"], "confidence": 0.8, "source": "test"}}'
```

### Deployment
```bash
# Deploy to Google Cloud
./deploy_taa_a2a.sh

# Deploy to Cloud Run
gcloud run deploy taa-a2a-agent --source . --platform managed --region us-central1

# Deploy to Kubernetes
kubectl apply -f k8s/
```

## 🔍 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Find process using port 8080
   lsof -i :8080
   
   # Kill process
   kill -9 <PID>
   ```

2. **Authentication Errors**
   ```bash
   # Re-authenticate
   gcloud auth login
   gcloud auth application-default login
   ```

3. **Dependencies Missing**
   ```bash
   # Reinstall dependencies
   pip install -r requirements_a2a.txt
   ```

4. **API Not Enabled**
   ```bash
   # Enable required APIs
   gcloud services enable aiplatform.googleapis.com
   gcloud services enable pubsub.googleapis.com
   ```

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python3 taa_a2a_server.py

# Check logs
tail -f logs/taa_a2a.log
```

## 📚 Next Steps

1. **Explore the Documentation**
   - Read `TAA_A2A_IMPLEMENTATION.md` for detailed implementation
   - Review `A2A_IMPLEMENTATION_SUMMARY.md` for overview

2. **Customize the Agent**
   - Add new capabilities in `taa_a2a_agent.py`
   - Modify configuration in `config/taa_a2a_config.json`
   - Extend the API in `taa_a2a_server.py`

3. **Integrate with Your SOC**
   - Connect to existing ADA agent
   - Integrate with CLA agent
   - Add custom threat intelligence sources

4. **Deploy to Production**
   - Use the deployment script
   - Set up monitoring and alerting
   - Configure security policies

## 🆘 Support

- **Documentation**: Check the markdown files in this directory
- **Code Comments**: Review inline documentation in Python files
- **Configuration**: See `config/taa_a2a_config.json` for all options
- **Testing**: Use the provided test scripts to validate functionality

## 🎯 Success Criteria

You've successfully set up TAA A2A when you can:

- ✅ Run the demo script without errors
- ✅ Start the A2A server and access health endpoint
- ✅ Execute A2A tasks and receive responses
- ✅ See agent discovery working
- ✅ Test all four capabilities (analysis, sharing, containment, collaboration)

Congratulations! You now have a working A2A-enabled TAA agent ready for production use. 