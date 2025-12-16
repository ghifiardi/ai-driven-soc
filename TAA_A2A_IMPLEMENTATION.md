# TAA A2A (Agent-to-Agent) Implementation

## Overview

This document describes the implementation of A2A (Agent-to-Agent) communication for the TAA (Triage & Analysis Agent) using Google's A2A protocol and Vertex AI Agent Engine. The implementation enables the TAA agent to communicate with other agents in the Google Agentspace ecosystem and participate in multi-agent workflows.

## Architecture

### Components

1. **TAA A2A Agent** (`taa_a2a_agent.py`)
   - Core agent implementation with A2A capabilities
   - Google Vertex AI integration using ADK
   - Agent discovery and registry management
   - Task routing and execution

2. **A2A Server** (`taa_a2a_server.py`)
   - FastAPI-based HTTP server for A2A communication
   - RESTful endpoints for agent discovery and task execution
   - Health monitoring and metrics collection
   - Callback handling for asynchronous communication

3. **Test Client** (`taa_a2a_test_client.py`)
   - Comprehensive testing framework for A2A communication
   - Agent discovery testing
   - Capability testing for all TAA skills
   - Performance and reliability validation

4. **Configuration** (`config/taa_a2a_config.json`)
   - Centralized configuration management
   - Google Cloud settings
   - A2A protocol parameters
   - Security and monitoring configuration

### A2A Protocol Implementation

The implementation follows Google's A2A protocol specification:

#### Agent Card
```json
{
  "name": "taa-security-agent",
  "version": "1.0.0",
  "description": "Triage and Analysis Agent for security alerts with A2A capabilities",
  "agent_id": "taa-agent-20250114-123456",
  "capabilities": [
    {
      "name": "analyze_security_alert",
      "description": "Analyze security alerts using LLM and threat intelligence",
      "input_schema": {
        "type": "object",
        "properties": {
          "alert_data": {"type": "object"},
          "enrichment_level": {"type": "string", "enum": ["basic", "full"]}
        },
        "required": ["alert_data"]
      }
    }
  ],
  "authentication": {
    "type": "oauth2",
    "scopes": ["https://www.googleapis.com/auth/cloud-platform"]
  },
  "endpoints": {
    "base_url": "https://taa-agent.agentspace.googleapis.com",
    "a2a_endpoint": "/a2a/tasks",
    "health_check": "/health",
    "agent_card": "/.well-known/agent.json"
  }
}
```

#### Task Structure
```json
{
  "task_id": "task-20250114-123456",
  "skill": "analyze_security_alert",
  "parameters": {
    "alert_data": {...},
    "enrichment_level": "full"
  },
  "callback_url": "https://callback-agent.com/a2a/callbacks",
  "priority": "high",
  "timestamp": "2025-01-14T12:34:56Z"
}
```

#### Response Structure
```json
{
  "task_id": "task-20250114-123456",
  "status": "completed",
  "result": {
    "status": "success",
    "analysis": {...},
    "agent_id": "taa-agent-20250114-123456"
  },
  "agent_id": "taa-agent-20250114-123456",
  "timestamp": "2025-01-14T12:35:01Z"
}
```

## Setup Instructions

### Prerequisites

1. **Google Cloud Project**
   - Active Google Cloud project with billing enabled
   - Vertex AI API enabled
   - Pub/Sub API enabled
   - Storage API enabled

2. **Google Cloud SDK**
   - gcloud CLI installed and configured
   - Authenticated with appropriate permissions

3. **Python Environment**
   - Python 3.8+ installed
   - Virtual environment support

### Installation

1. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd ai-driven-soc
   ```

2. **Run Deployment Script**
   ```bash
   chmod +x deploy_taa_a2a.sh
   ./deploy_taa_a2a.sh
   ```

   The deployment script will:
   - Authenticate with Google Cloud
   - Enable required APIs
   - Create service accounts and permissions
   - Set up Pub/Sub topics and subscriptions
   - Install Python dependencies
   - Deploy to Vertex AI Agent Engine

3. **Manual Setup (Alternative)**
   ```bash
   # Create virtual environment
   python3 -m venv venv
   source venv/bin/activate
   
   # Install dependencies
   pip install -r requirements_a2a.txt
   
   # Set environment variables
   export GOOGLE_CLOUD_PROJECT="your-project-id"
   export GOOGLE_CLOUD_LOCATION="us-central1"
   export STAGING_BUCKET="gs://your-staging-bucket"
   ```

### Configuration

The A2A agent uses a centralized configuration file (`config/taa_a2a_config.json`):

```json
{
  "project_id": "ai-driven-soc",
  "location": "us-central1",
  "a2a": {
    "enabled": true,
    "port": 8080,
    "discovery_interval": 300,
    "max_agents": 100
  },
  "vertex_ai": {
    "model": "gemini-2.0-flash-exp",
    "staging_bucket": "gs://ai-driven-soc-staging"
  },
  "agentspace": {
    "enabled": true,
    "registration_url": "https://agentspace.googleapis.com/v1/agents",
    "discovery_url": "https://agentspace.googleapis.com/v1/agents/discover"
  }
}
```

## Usage

### Starting the A2A Server

1. **Start the FastAPI Server**
   ```bash
   python3 taa_a2a_server.py
   ```

   The server will start on `http://localhost:8080` with the following endpoints:
   - `GET /` - Root endpoint
   - `GET /health` - Health check
   - `GET /.well-known/agent.json` - Agent card for discovery
   - `POST /a2a/tasks` - A2A task execution
   - `GET /a2a/capabilities` - Agent capabilities
   - `POST /a2a/callbacks` - Callback handling
   - `GET /metrics` - Agent metrics

2. **Verify Server Status**
   ```bash
   curl http://localhost:8080/health
   curl http://localhost:8080/.well-known/agent.json
   ```

### Testing A2A Communication

1. **Run Comprehensive Test**
   ```bash
   python3 taa_a2a_test_client.py
   ```

   The test client will:
   - Discover the TAA agent
   - Check agent health and capabilities
   - Test all A2A capabilities:
     - Security alert analysis
     - Threat intelligence sharing
     - Containment requests
     - Collaborative analysis

2. **Manual Testing**
   ```bash
   # Test alert analysis
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

### Integration with Google Agentspace

1. **Agent Registration**
   The TAA agent automatically registers with Google Agentspace when started:
   ```python
   await agent.register_with_agentspace()
   ```

2. **Agent Discovery**
   The agent discovers other agents in the ecosystem:
   ```python
   await agent.discover_agents()
   ```

3. **Multi-Agent Communication**
   ```python
   # Send task to another agent
   task = A2ATask(
       task_id="collab-001",
       skill="malware_analysis",
       parameters={"file_hash": "abc123"}
   )
   response = await agent.send_a2a_task(other_agent_url, task)
   ```

## Capabilities

### 1. Security Alert Analysis

**Skill**: `analyze_security_alert`

Analyzes security alerts using LLM and threat intelligence:

```python
task = A2ATask(
    task_id="analysis-001",
    skill="analyze_security_alert",
    parameters={
        "alert_data": {
            "alert_id": "ALERT-001",
            "timestamp": "2025-01-14T12:00:00Z",
            "source": "firewall",
            "severity": "high",
            "log_data": {...}
        },
        "enrichment_level": "full"
    }
)
```

**Response**:
```json
{
  "status": "completed",
  "result": {
    "status": "success",
    "analysis": {
      "is_true_positive": true,
      "severity": "high",
      "confidence": 0.92,
      "attack_type": "malware",
      "indicators": ["suspicious_activity", "malicious_ip"],
      "recommended_actions": ["isolate_host", "block_ip"]
    }
  }
}
```

### 2. Threat Intelligence Sharing

**Skill**: `share_threat_intelligence`

Shares threat intelligence with other agents:

```python
task = A2ATask(
    task_id="ti-001",
    skill="share_threat_intelligence",
    parameters={
        "indicators": ["malicious_ip_1", "suspicious_domain_1"],
        "confidence": 0.85,
        "source": "taa_agent"
    }
)
```

### 3. Containment Requests

**Skill**: `request_containment`

Requests containment actions from other agents:

```python
task = A2ATask(
    task_id="containment-001",
    skill="request_containment",
    parameters={
        "target_host": "192.168.1.100",
        "action_type": "isolate",
        "priority": "high"
    }
)
```

### 4. Collaborative Analysis

**Skill**: `collaborative_analysis`

Collaborates with other agents for complex analysis:

```python
task = A2ATask(
    task_id="collab-001",
    skill="collaborative_analysis",
    parameters={
        "analysis_type": "malware_analysis",
        "data": {"file_hash": "abc123"},
        "collaboration_mode": "parallel"
    }
)
```

## Security Features

### Authentication
- OAuth2 authentication with Google Cloud
- Service account-based authentication
- Token refresh and management

### Encryption
- AES-256-GCM encryption for sensitive data
- TLS/SSL for all communications
- Secure key management

### Rate Limiting
- Configurable rate limiting (default: 100 requests/minute)
- Burst limit protection
- DDoS mitigation

### Access Control
- IAM-based permissions
- Role-based access control
- Network-level security with VPC

## Monitoring and Observability

### Metrics
- Request/response metrics
- Performance monitoring
- Error rate tracking
- Queue size monitoring

### Logging
- Structured logging with JSON format
- Log rotation and archival
- Integration with Google Cloud Logging

### Health Checks
- Endpoint health monitoring
- Dependency health checks
- Automated alerting

### Tracing
- Distributed tracing support
- Request flow visualization
- Performance bottleneck identification

## Deployment Options

### 1. Local Development
```bash
python3 taa_a2a_server.py
```

### 2. Google Cloud Run
```bash
gcloud run deploy taa-a2a-agent \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### 3. Vertex AI Agent Engine
```bash
# Deploy using the provided script
./deploy_taa_a2a.sh
```

### 4. Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: taa-a2a-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: taa-a2a-agent
  template:
    metadata:
      labels:
        app: taa-a2a-agent
    spec:
      containers:
      - name: taa-a2a-agent
        image: gcr.io/your-project/taa-a2a-agent:latest
        ports:
        - containerPort: 8080
        env:
        - name: GOOGLE_CLOUD_PROJECT
          value: "your-project-id"
```

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   ```bash
   # Verify authentication
   gcloud auth list
   gcloud config get-value account
   
   # Re-authenticate if needed
   gcloud auth login
   ```

2. **API Not Enabled**
   ```bash
   # Enable required APIs
   gcloud services enable aiplatform.googleapis.com
   gcloud services enable pubsub.googleapis.com
   ```

3. **Permission Errors**
   ```bash
   # Check service account permissions
   gcloud projects get-iam-policy your-project-id \
     --flatten="bindings[].members" \
     --format="table(bindings.role)" \
     --filter="bindings.members:taa-a2a-agent"
   ```

4. **Connection Issues**
   ```bash
   # Test connectivity
   curl -v http://localhost:8080/health
   
   # Check logs
   tail -f logs/taa_a2a.log
   ```

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
python3 taa_a2a_server.py
```

### Performance Tuning

1. **Increase Concurrency**
   ```bash
   uvicorn taa_a2a_server:app --workers 4 --host 0.0.0.0 --port 8080
   ```

2. **Optimize Memory**
   ```bash
   export PYTHONOPTIMIZE=1
   python3 taa_a2a_server.py
   ```

## Best Practices

### 1. Security
- Use service accounts with minimal required permissions
- Enable VPC Service Controls for data security
- Implement proper authentication and authorization
- Regular security audits and updates

### 2. Performance
- Implement connection pooling for external services
- Use async/await for I/O operations
- Implement caching for frequently accessed data
- Monitor and optimize resource usage

### 3. Reliability
- Implement retry logic with exponential backoff
- Use circuit breakers for external dependencies
- Implement health checks and monitoring
- Regular backup and disaster recovery testing

### 4. Scalability
- Design for horizontal scaling
- Use load balancing for high availability
- Implement proper resource management
- Monitor and optimize performance bottlenecks

## Future Enhancements

### 1. Advanced A2A Features
- Support for streaming responses
- Real-time collaboration capabilities
- Advanced routing and load balancing
- Support for custom protocols

### 2. Integration Enhancements
- Integration with more AI/ML frameworks
- Support for additional cloud providers
- Enhanced threat intelligence integration
- Advanced analytics and reporting

### 3. Security Enhancements
- Zero-trust security model
- Advanced encryption and key management
- Threat detection and response
- Compliance and audit capabilities

### 4. Performance Optimizations
- Edge computing support
- Advanced caching strategies
- Machine learning-based optimization
- Real-time performance monitoring

## Conclusion

The TAA A2A implementation provides a robust, scalable, and secure foundation for agent-to-agent communication in the Google Cloud ecosystem. It enables the TAA agent to participate in complex multi-agent workflows, share threat intelligence, and collaborate with other agents for enhanced security operations.

The implementation follows Google's A2A protocol standards and integrates seamlessly with Vertex AI Agent Engine and Google Agentspace, providing enterprise-grade capabilities for modern SOC operations.

For support and questions, please refer to the troubleshooting section or contact the development team. 