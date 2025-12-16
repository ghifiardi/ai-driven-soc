# Hybrid Cyber Defense Agent - Production Deployment Guide

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Prerequisites](#prerequisites)
4. [Installation & Setup](#installation--setup)
5. [Configuration](#configuration)
6. [Deployment](#deployment)
7. [Monitoring & Operations](#monitoring--operations)
8. [Troubleshooting](#troubleshooting)
9. [Security Considerations](#security-considerations)
10. [Performance Optimization](#performance-optimization)
11. [Maintenance](#maintenance)
12. [Appendix](#appendix)

## System Overview

The Hybrid Cyber Defense Agent is an AI-powered cybersecurity system that combines:
- **DQN (Deep Q-Network)** for fast decision-making
- **Google Gemini** for explainable AI and natural language processing
- **Pub/Sub integration** for real-time alert processing
- **A2A Protocol** compliance for agent interoperability

### Key Features
- Real-time threat analysis and response
- Explainable AI with natural language explanations
- Circuit breaker patterns for resilience
- Comprehensive monitoring and alerting
- RESTful API for integration
- Streamlit dashboard for visualization

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   ADA Agent     │───▶│  Pub/Sub Topics  │───▶│ Hybrid Defense  │
│ (Anomaly Det.)  │    │   (ada-alerts)   │    │     Agent       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CRA Agent     │◀───│  Pub/Sub Topics  │◀───│  DQN + Gemini   │
│ (Containment)   │    │ (containment-req)│    │   Processing    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CLA Agent     │◀───│  Pub/Sub Topics  │◀───│  Feedback Loop  │
│ (Learning)      │    │  (taa-feedback)  │    │   & Analytics   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Component Responsibilities

#### Hybrid Defense Agent
- **DQN Engine**: Fast threat classification and action recommendation
- **Gemini Integration**: Natural language explanations and reasoning
- **Pub/Sub Handler**: Real-time alert ingestion and response publishing
- **Circuit Breakers**: Resilience and fault tolerance
- **A2A Compliance**: Agent-to-agent communication protocol

#### Dashboard
- **Real-time Monitoring**: Agent status, metrics, and performance
- **Alert Visualization**: Live threat processing and response tracking
- **System Health**: Component status and circuit breaker states
- **Analytics**: DQN performance and Gemini usage statistics

## Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows 10+
- **Python**: 3.8+ (recommended 3.9+)
- **Memory**: 4GB+ RAM
- **Storage**: 10GB+ free space
- **Network**: Internet connectivity for Google Cloud services

### Google Cloud Requirements
- **Project**: Valid GCP project with billing enabled
- **APIs**: Pub/Sub, Vertex AI, Cloud Storage
- **Service Account**: With appropriate permissions
- **Authentication**: Application Default Credentials or service account key

### Python Dependencies
```bash
# Core dependencies
torch>=2.0.0
torchvision>=0.15.0
vertexai>=1.38.0
google-cloud-pubsub>=2.18.0
fastapi>=0.104.0
uvicorn>=0.24.0
streamlit>=1.28.0
gymnasium>=0.29.0

# Additional dependencies
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
requests>=2.31.0
python-multipart>=0.0.6
```

## Installation & Setup

### 1. Environment Setup

```bash
# Create virtual environment
python3 -m venv hybrid-defense-env
source hybrid-defense-env/bin/activate  # Linux/macOS
# or
hybrid-defense-env\Scripts\activate     # Windows

# Upgrade pip
pip install --upgrade pip
```

### 2. Install Dependencies

```bash
# Install from requirements.txt
pip install -r requirements.txt

# Or install individually
pip install torch torchvision vertexai google-cloud-pubsub fastapi uvicorn streamlit gymnasium
```

### 3. Google Cloud Setup

```bash
# Install Google Cloud CLI
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Authenticate
gcloud auth login
gcloud auth application-default login

# Set project
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable pubsub.googleapis.com
gcloud services enable aiplatform.googleapis.com
gcloud services enable storage.googleapis.com
```

### 4. Create Service Account (Optional)

```bash
# Create service account
gcloud iam service-accounts create hybrid-defense-agent \
    --description="Service account for Hybrid Defense Agent" \
    --display-name="Hybrid Defense Agent"

# Grant permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:hybrid-defense-agent@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/pubsub.subscriber"
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:hybrid-defense-agent@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/pubsub.publisher"
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:hybrid-defense-agent@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

# Create and download key
gcloud iam service-accounts keys create hybrid-defense-key.json \
    --iam-account=hybrid-defense-agent@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

## Configuration

### 1. Agent Configuration (`config/hybrid_defense_config.json`)

```json
{
  "agent": {
    "agent_id": "hybrid-defense-agent",
    "log_level": "INFO",
    "api_port": 8083,
    "refresh_interval_seconds": 30,
    "max_alerts_in_memory": 1000
  },
  "dqn_model": {
    "state_size": 7,
    "action_size": 5,
    "model_path": "models/dqn_cyber_defense_model.pth",
    "fallback_enabled": true,
    "fallback_threshold": 0.6
  },
  "gemini": {
    "project_id": "YOUR_PROJECT_ID",
    "location": "us-central1",
    "model_name": "gemini-2.0-flash-exp",
    "temperature": 0.1,
    "max_output_tokens": 500,
    "timeout_seconds": 10
  },
  "pubsub": {
    "project_id": "YOUR_PROJECT_ID",
    "ada_alerts_topic": "ada-alerts",
    "ada_alerts_subscription": "hybrid-defense-ada-alerts-sub",
    "containment_requests_topic": "containment-requests",
    "taa_feedback_topic": "taa-feedback",
    "max_messages_per_pull": 10,
    "ack_deadline_seconds": 30
  },
  "circuit_breakers": {
    "dqn_model_failure_threshold": 5,
    "gemini_api_failure_threshold": 3,
    "pubsub_connection_failure_threshold": 5,
    "reset_timeout_seconds": 300
  }
}
```

### 2. Environment Variables

```bash
# Set environment variables
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account-key.json"
export GOOGLE_CLOUD_PROJECT="YOUR_PROJECT_ID"
export HYBRID_DEFENSE_CONFIG_PATH="config/hybrid_defense_config.json"
export HYBRID_DEFENSE_LOG_LEVEL="INFO"
```

### 3. Pub/Sub Topics Setup

```bash
# Create topics
gcloud pubsub topics create ada-alerts
gcloud pubsub topics create containment-requests
gcloud pubsub topics create taa-feedback

# Create subscriptions
gcloud pubsub subscriptions create hybrid-defense-ada-alerts-sub \
    --topic=ada-alerts
```

## Deployment

### 1. Development Deployment

```bash
# Start agent
python3 hybrid_cyber_defense_agent.py

# Start dashboard (separate terminal)
python3 -m streamlit run hybrid_defense_dashboard.py --server.port=8529
```

### 2. Production Deployment with systemd

#### Create Service File

```bash
sudo nano /etc/systemd/system/hybrid-defense-agent.service
```

```ini
[Unit]
Description=Hybrid Cyber Defense Agent
After=network.target

[Service]
Type=simple
User=app
Group=app
WorkingDirectory=/opt/hybrid-defense-agent
ExecStart=/opt/hybrid-defense-agent/venv/bin/python hybrid_cyber_defense_agent.py
Restart=always
RestartSec=10
StandardOutput=syslog
StandardError=syslog
SyslogIdentifier=hybrid-defense-agent

# Environment variables
Environment=GOOGLE_APPLICATION_CREDENTIALS=/opt/hybrid-defense-agent/config/service-account-key.json
Environment=GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID

[Install]
WantedBy=multi-user.target
```

#### Deploy Dashboard Service

```bash
sudo nano /etc/systemd/system/hybrid-defense-dashboard.service
```

```ini
[Unit]
Description=Hybrid Defense Dashboard
After=network.target hybrid-defense-agent.service

[Service]
Type=simple
User=app
Group=app
WorkingDirectory=/opt/hybrid-defense-agent
ExecStart=/opt/hybrid-defense-agent/venv/bin/streamlit run hybrid_defense_dashboard.py --server.port=8529 --server.headless=true
Restart=always
RestartSec=10
StandardOutput=syslog
StandardError=syslog
SyslogIdentifier=hybrid-defense-dashboard

[Install]
WantedBy=multi-user.target
```

#### Enable and Start Services

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable services
sudo systemctl enable hybrid-defense-agent
sudo systemctl enable hybrid-defense-dashboard

# Start services
sudo systemctl start hybrid-defense-agent
sudo systemctl start hybrid-defense-dashboard

# Check status
sudo systemctl status hybrid-defense-agent
sudo systemctl status hybrid-defense-dashboard
```

### 3. Docker Deployment

#### Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 app && chown -R app:app /app
USER app

# Expose ports
EXPOSE 8083 8529

# Start command
CMD ["python", "hybrid_cyber_defense_agent.py"]
```

#### Docker Compose

```yaml
version: '3.8'

services:
  hybrid-defense-agent:
    build: .
    ports:
      - "8083:8083"
    environment:
      - GOOGLE_APPLICATION_CREDENTIALS=/app/config/service-account-key.json
      - GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID
    volumes:
      - ./config:/app/config
      - ./models:/app/models
    restart: unless-stopped

  hybrid-defense-dashboard:
    build: .
    ports:
      - "8529:8529"
    command: ["streamlit", "run", "hybrid_defense_dashboard.py", "--server.port=8529", "--server.headless=true"]
    depends_on:
      - hybrid-defense-agent
    restart: unless-stopped
```

### 4. Kubernetes Deployment

#### Deployment YAML

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hybrid-defense-agent
spec:
  replicas: 2
  selector:
    matchLabels:
      app: hybrid-defense-agent
  template:
    metadata:
      labels:
        app: hybrid-defense-agent
    spec:
      containers:
      - name: hybrid-defense-agent
        image: hybrid-defense-agent:latest
        ports:
        - containerPort: 8083
        env:
        - name: GOOGLE_APPLICATION_CREDENTIALS
          value: "/app/config/service-account-key.json"
        - name: GOOGLE_CLOUD_PROJECT
          value: "YOUR_PROJECT_ID"
        volumeMounts:
        - name: config
          mountPath: /app/config
        - name: models
          mountPath: /app/models
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
      volumes:
      - name: config
        secret:
          secretName: hybrid-defense-config
      - name: models
        persistentVolumeClaim:
          claimName: hybrid-defense-models
---
apiVersion: v1
kind: Service
metadata:
  name: hybrid-defense-agent-service
spec:
  selector:
    app: hybrid-defense-agent
  ports:
  - port: 8083
    targetPort: 8083
  type: LoadBalancer
```

## Monitoring & Operations

### 1. Health Checks

```bash
# Agent health
curl http://localhost:8083/health

# Agent status
curl http://localhost:8083/status

# Dashboard health
curl http://localhost:8529/healthz
```

### 2. Logging

```bash
# View agent logs
sudo journalctl -u hybrid-defense-agent -f

# View dashboard logs
sudo journalctl -u hybrid-defense-dashboard -f

# View specific log levels
sudo journalctl -u hybrid-defense-agent --since "1 hour ago" | grep ERROR
```

### 3. Metrics Collection

The agent provides comprehensive metrics via the `/status` endpoint:

- **Performance Metrics**: DQN inference duration, Gemini API response times
- **Business Metrics**: Alerts processed, actions recommended, explanations generated
- **Error Metrics**: Error counts by type, circuit breaker states
- **System Metrics**: Uptime, memory usage, connection status

### 4. Alerting

Set up monitoring for:
- Agent downtime
- High error rates
- Circuit breaker activations
- DQN model failures
- Gemini API failures
- Pub/Sub connection issues

## Troubleshooting

### Common Issues

#### 1. Google Cloud Authentication

**Problem**: `404 Requested project not found or user does not have access`

**Solution**:
```bash
# Verify project ID
gcloud config get-value project

# Re-authenticate
gcloud auth application-default login

# Check service account permissions
gcloud projects get-iam-policy YOUR_PROJECT_ID
```

#### 2. Pub/Sub Connection Issues

**Problem**: `Connection refused` or `Timeout`

**Solution**:
```bash
# Check topic existence
gcloud pubsub topics list

# Check subscription
gcloud pubsub subscriptions list

# Test connectivity
gcloud pubsub topics publish ada-alerts --message="test"
```

#### 3. DQN Model Loading

**Problem**: `No pre-trained model found`

**Solution**:
```bash
# Check model file exists
ls -la models/dqn_cyber_defense_model.pth

# Train model if needed
python3 train_dqn_model.py

# Verify model loading
python3 simple_validation.py
```

#### 4. Gemini API Issues

**Problem**: `Vertex AI API not available`

**Solution**:
```bash
# Check API enablement
gcloud services list --enabled | grep aiplatform

# Verify quota
gcloud compute project-info describe --project=YOUR_PROJECT_ID

# Test API access
python3 -c "import vertexai; vertexai.init(); print('OK')"
```

### Debug Mode

Enable debug logging:

```bash
export HYBRID_DEFENSE_LOG_LEVEL=DEBUG
python3 hybrid_cyber_defense_agent.py
```

### Circuit Breaker Diagnostics

```bash
# Check circuit breaker states
curl http://localhost:8083/status | jq '.circuit_breakers'

# Reset circuit breakers (if needed)
curl -X POST http://localhost:8083/reset-circuit-breakers
```

## Security Considerations

### 1. Authentication & Authorization

- Use service accounts with minimal required permissions
- Rotate service account keys regularly
- Implement API key authentication for external access
- Use HTTPS in production

### 2. Network Security

- Deploy behind load balancer with SSL termination
- Use VPC for internal communication
- Implement firewall rules to restrict access
- Monitor network traffic for anomalies

### 3. Data Protection

- Encrypt sensitive data at rest
- Use secure communication channels
- Implement data retention policies
- Regular security audits

### 4. Access Control

- Principle of least privilege
- Regular access reviews
- Multi-factor authentication
- Audit logging for all access

## Performance Optimization

### 1. DQN Model Optimization

- Use GPU acceleration when available
- Implement model quantization
- Batch processing for multiple alerts
- Model caching and preloading

### 2. Gemini API Optimization

- Implement request batching
- Use connection pooling
- Cache frequent explanations
- Implement rate limiting

### 3. Pub/Sub Optimization

- Increase message batch size
- Optimize acknowledgment timing
- Use flow control settings
- Monitor message backlog

### 4. System Optimization

- Use SSD storage for models
- Increase memory allocation
- Optimize Python GIL usage
- Implement connection pooling

## Maintenance

### 1. Regular Tasks

#### Daily
- Monitor system health and performance
- Check error logs and circuit breaker states
- Verify Pub/Sub message flow
- Review alert processing metrics

#### Weekly
- Update DQN model with new training data
- Review and optimize Gemini prompts
- Analyze performance trends
- Update documentation

#### Monthly
- Security audit and vulnerability assessment
- Performance tuning and optimization
- Backup and disaster recovery testing
- Capacity planning review

### 2. Model Updates

```bash
# Retrain DQN model
python3 train_dqn_model.py --episodes=1000 --save-path=models/dqn_cyber_defense_model.pth

# Validate new model
python3 simple_validation.py

# Deploy new model
sudo systemctl restart hybrid-defense-agent
```

### 3. Configuration Updates

```bash
# Update configuration
sudo nano /opt/hybrid-defense-agent/config/hybrid_defense_config.json

# Reload configuration
sudo systemctl reload hybrid-defense-agent
```

### 4. Backup & Recovery

```bash
# Backup configuration
tar -czf hybrid-defense-backup-$(date +%Y%m%d).tar.gz \
    config/ models/ *.py *.json

# Restore from backup
tar -xzf hybrid-defense-backup-YYYYMMDD.tar.gz
sudo systemctl restart hybrid-defense-agent
```

## Appendix

### A. API Endpoints

#### Agent Endpoints
- `GET /health` - Health check
- `GET /status` - Comprehensive status and metrics
- `POST /a2a/process_alert` - Process alert via A2A protocol
- `GET /agent-card` - A2A agent card

#### Dashboard Endpoints
- `GET /` - Main dashboard interface
- `GET /healthz` - Dashboard health check

### B. Configuration Reference

#### Agent Configuration
- `agent_id`: Unique identifier for the agent
- `log_level`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `api_port`: Port for the FastAPI server
- `refresh_interval_seconds`: Dashboard refresh interval
- `max_alerts_in_memory`: Maximum alerts to keep in memory

#### DQN Configuration
- `state_size`: Size of the state vector
- `action_size`: Number of possible actions
- `model_path`: Path to the pre-trained model
- `fallback_enabled`: Enable rule-based fallback
- `fallback_threshold`: Confidence threshold for fallback

#### Gemini Configuration
- `project_id`: Google Cloud project ID
- `location`: Vertex AI region
- `model_name`: Gemini model name
- `temperature`: Response creativity (0.0-1.0)
- `max_output_tokens`: Maximum response length
- `timeout_seconds`: API timeout

#### Pub/Sub Configuration
- `project_id`: Google Cloud project ID
- `ada_alerts_topic`: Topic for ADA alerts
- `ada_alerts_subscription`: Subscription for ADA alerts
- `containment_requests_topic`: Topic for containment requests
- `taa_feedback_topic`: Topic for TAA feedback
- `max_messages_per_pull`: Maximum messages per pull
- `ack_deadline_seconds`: Acknowledgment deadline

#### Circuit Breaker Configuration
- `dqn_model_failure_threshold`: DQN failure threshold
- `gemini_api_failure_threshold`: Gemini API failure threshold
- `pubsub_connection_failure_threshold`: Pub/Sub failure threshold
- `reset_timeout_seconds`: Circuit breaker reset timeout

### C. Error Codes

#### HTTP Status Codes
- `200`: Success
- `400`: Bad Request
- `401`: Unauthorized
- `403`: Forbidden
- `404`: Not Found
- `500`: Internal Server Error
- `503`: Service Unavailable

#### Agent Error Types
- `DQNError`: DQN model inference failure
- `GeminiError`: Gemini API failure
- `PubSubError`: Pub/Sub communication failure
- `ConfigError`: Configuration error
- `ValidationError`: Input validation error

### D. Monitoring Queries

#### Prometheus Queries (if using Prometheus)
```promql
# Agent uptime
hybrid_defense_agent_uptime_seconds

# Alerts processed per minute
rate(hybrid_defense_agent_alerts_processed_total[1m])

# DQN inference duration
histogram_quantile(0.95, rate(hybrid_defense_agent_dqn_inference_duration_seconds_bucket[5m]))

# Circuit breaker states
hybrid_defense_agent_circuit_breaker_state
```

### E. Support Contacts

- **Technical Support**: [Your support email]
- **Documentation**: [Your documentation URL]
- **Issue Tracking**: [Your issue tracking system]
- **Emergency Contact**: [Your emergency contact]

---

**Document Version**: 1.0  
**Last Updated**: [Current Date]  
**Author**: Hybrid Defense Team  
**Review Date**: [Review Date]
