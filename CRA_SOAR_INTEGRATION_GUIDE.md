# CRA & SOAR Integration Guide with MCP
=========================================

## Overview

This guide explains how the **Containment Response Agent (CRA)** and **SOAR (Security Orchestration, Automation and Response)** capabilities have been enhanced with **MCP (Model Context Protocol)** integration to create a comprehensive security automation platform.

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ADA Agent     │    │   TAA Agent     │    │  CRA SOAR MCP   │
│  (Detection)    │───▶│  (Analysis)     │───▶│     Agent       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │ Threat Intel    │    │ SOAR Workflow   │
                       │ (MCP Server)    │    │    Engine       │
                       └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │ Containment     │    │ Response        │
                       │    Actions      │    │ Automation      │
                       └─────────────────┘    └─────────────────┘
```

## Enhanced Capabilities

### 1. **MCP-Enhanced Threat Intelligence**
- **Multi-Source Enrichment**: Automatically enrich incidents with threat intelligence from 5+ sources
- **Risk Scoring**: Calculate comprehensive risk scores based on multiple TI sources
- **Real-time Context**: Provide real-time threat intelligence context for containment decisions
- **Confidence Assessment**: Assess confidence levels for automated actions

### 2. **SOAR Workflow Orchestration**
- **Intelligent Playbook Selection**: Automatically select appropriate playbooks based on incident characteristics
- **Conditional Logic**: Execute complex workflows with conditional branching
- **Parallel Execution**: Run multiple containment actions in parallel where safe
- **Approval Workflows**: Integrate human approval for high-risk actions

### 3. **Advanced Containment Actions**
- **Asset Isolation**: Automatically isolate compromised assets
- **Network Blocking**: Block malicious IPs, domains, and URLs
- **File Quarantine**: Quarantine suspicious files for analysis
- **Account Management**: Disable compromised accounts
- **System Patching**: Apply emergency security patches

### 4. **Real-time Monitoring & Response**
- **Workflow Monitoring**: Real-time monitoring of SOAR workflow execution
- **Status Tracking**: Track containment actions and their outcomes
- **Escalation Management**: Automatic escalation for time-sensitive incidents
- **Performance Metrics**: Comprehensive metrics and reporting

## File Structure

```
ai-driven-soc/
├── cra_soar_mcp_agent.py           # Enhanced CRA with SOAR and MCP
├── soar_workflow_server.py         # SOAR workflow management server
├── test_cra_soar_mcp.py            # Integration tests
├── config/
│   └── cra_soar_mcp_config.json   # CRA SOAR MCP configuration
├── threat_intelligence_mcp_server.py  # MCP TI server
├── simulated_chronicle_server.py   # Simulated Chronicle server
└── docs/
    └── CRA_SOAR_INTEGRATION_GUIDE.md  # This guide
```

## Configuration

### **CRA SOAR MCP Configuration**

The configuration file `config/cra_soar_mcp_config.json` includes:

```json
{
  "project_id": "ai-driven-soc",
  "location": "us-central1",
  
  "mcp": {
    "enabled": true,
    "env": {
      "VT_API_KEY": "",
      "AIPDB_API_KEY": "",
      "SHODAN_API_KEY": ""
    }
  },
  
  "playbooks": {
    "malware_containment": {
      "name": "Malware Containment",
      "trigger_conditions": {
        "severity": ["high", "critical"],
        "threat_types": ["malware", "ransomware"],
        "min_risk_score": 70
      },
      "actions": [...],
      "approval_workflow": {...}
    }
  }
}
```

### **Key Configuration Sections**

1. **MCP Integration**: Configure threat intelligence sources
2. **SOAR Playbooks**: Define automated response workflows
3. **Security Tools**: Configure integrations with security tools
4. **Notification Systems**: Set up alerting and notifications
5. **Monitoring**: Configure metrics and health checks

## SOAR Playbooks

### **Available Playbooks**

1. **Malware Containment**
   - Isolates infected hosts
   - Blocks malicious IPs
   - Quarantines suspicious files
   - Escalates to security team

2. **Phishing Response**
   - Blocks phishing URLs
   - Monitors affected accounts
   - Escalates for user notification

3. **DDoS Mitigation**
   - Enables DDoS protection
   - Escalates to network team
   - Implements rate limiting

4. **Insider Threat Response**
   - Disables suspicious accounts
   - Monitors user activity
   - Escalates to legal team

5. **Critical Vulnerability Response**
   - Applies emergency patches
   - Isolates unpatched systems
   - Monitors for exploit attempts

### **Playbook Structure**

Each playbook includes:
- **Trigger Conditions**: When to execute the playbook
- **Actions**: Steps to take during execution
- **Approval Workflows**: Human approval requirements
- **Timeouts**: Maximum execution time
- **Retry Logic**: Error handling and retries

## API Endpoints

### **SOAR Workflow Server (Port 8081)**

- `GET /health` - Health check
- `GET /playbooks` - List all playbooks
- `GET /playbooks/{id}` - Get specific playbook
- `POST /incidents` - Create and process incident
- `POST /playbooks/{id}/execute` - Execute playbook directly
- `GET /workflows` - List running workflows
- `GET /workflows/{id}` - Get workflow status
- `POST /workflows/{id}/approve` - Approve pending action
- `POST /workflows/{id}/cancel` - Cancel workflow
- `GET /metrics` - Get system metrics

### **Example API Usage**

```python
# Create incident
incident_data = {
    "alert_id": "malware_001",
    "severity": "high",
    "title": "Malware Detection",
    "description": "Suspicious executable detected",
    "affected_assets": ["host-001"],
    "indicators": ["192.168.1.100", "malware.com"]
}

response = requests.post("http://localhost:8081/incidents", json=incident_data)
```

## MCP Integration

### **Threat Intelligence Enrichment**

The CRA automatically enriches incidents with threat intelligence:

```python
# MCP client queries multiple TI sources
async with MCPThreatIntelligenceClient(config) as mcp:
    ti_data = await mcp.enrich_indicators(incident.indicators)
    
    # Results include:
    # - Reputation scores
    # - Threat types
    # - Risk assessments
    # - Source confidence
```

### **Enhanced Decision Making**

MCP integration enables:
- **Informed Containment**: Decisions based on comprehensive TI data
- **Risk-Based Actions**: Actions prioritized by threat risk scores
- **Source Validation**: Multiple TI sources for confidence
- **Real-time Updates**: Latest threat intelligence data

## Workflow Execution

### **Automated Workflow Process**

1. **Incident Creation**: Incident received from TAA
2. **Threat Intelligence Enrichment**: MCP queries TI sources
3. **Playbook Selection**: SOAR engine selects appropriate playbook
4. **Action Execution**: Automated containment actions
5. **Approval Workflow**: Human approval for high-risk actions
6. **Status Monitoring**: Real-time workflow tracking
7. **Reporting**: Comprehensive incident reports

### **Approval Workflows**

High-risk actions require approval:
- **Isolation Actions**: Network quarantine
- **Account Actions**: User account disabling
- **System Actions**: Emergency patching
- **Legal Actions**: Insider threat escalation

## Testing

### **Test Scenarios**

Run the test suite to validate integration:

```bash
# Start the SOAR server
python3 soar_workflow_server.py

# Run tests (in another terminal)
python3 test_cra_soar_mcp.py
```

### **Test Cases**

1. **Malware Incident**: Test malware containment playbook
2. **Phishing Incident**: Test phishing response playbook
3. **Direct Execution**: Test direct playbook execution
4. **Threat Intelligence**: Test MCP TI integration
5. **Approval Workflows**: Test human approval processes

## Deployment

### **Prerequisites**

1. **Python 3.12+** with MCP dependencies
2. **Google Cloud** project with Vertex AI
3. **Pub/Sub** topics configured
4. **Firestore** database
5. **API Keys** for threat intelligence sources (optional)

### **Installation Steps**

1. **Install Dependencies**:
   ```bash
   pip install -r requirements_mcp.txt
   ```

2. **Configure Environment**:
   ```bash
   cp mcp_env_template.txt .env
   # Edit .env with your API keys
   ```

3. **Start SOAR Server**:
   ```bash
   python3 soar_workflow_server.py
   ```

4. **Test Integration**:
   ```bash
   python3 test_cra_soar_mcp.py
   ```

### **Production Deployment**

1. **Docker Containerization**:
   ```dockerfile
   FROM python:3.12-slim
   COPY requirements_mcp.txt .
   RUN pip install -r requirements_mcp.txt
   COPY . .
   CMD ["python3", "soar_workflow_server.py"]
   ```

2. **Load Balancing**: Deploy multiple SOAR server instances
3. **Monitoring**: Set up comprehensive monitoring and alerting
4. **Backup**: Implement data backup and recovery procedures

## Monitoring & Observability

### **Metrics Available**

- **Playbook Execution**: Success/failure rates
- **Response Times**: Average response times
- **Threat Intelligence**: TI source usage and accuracy
- **Approval Workflows**: Approval times and rates
- **System Health**: Server performance and availability

### **Logging**

Structured JSON logging includes:
- **Incident Processing**: Complete incident lifecycle
- **Action Execution**: Detailed action results
- **Error Handling**: Comprehensive error logging
- **Performance**: Timing and performance metrics

### **Alerting**

Configure alerts for:
- **Failed Playbooks**: Playbook execution failures
- **Approval Timeouts**: Delayed approvals
- **High Error Rates**: System error thresholds
- **Performance Issues**: Response time degradation

## Security Considerations

### **Authentication & Authorization**

- **API Keys**: Secure storage of TI API keys
- **Service Accounts**: Google Cloud service accounts
- **Role-Based Access**: IAM roles for different functions
- **Audit Logging**: Complete audit trail

### **Data Protection**

- **Encryption**: TLS for all communications
- **Secure Storage**: Encrypted storage of sensitive data
- **Access Controls**: Restrict access to incident data
- **Data Retention**: Configurable data retention policies

### **Approval Security**

- **Multi-Factor Authentication**: For approval workflows
- **Approval Chains**: Multiple approver requirements
- **Time Limits**: Automatic escalation for delays
- **Audit Trails**: Complete approval history

## Best Practices

### **Playbook Design**

1. **Start Simple**: Begin with basic playbooks
2. **Test Thoroughly**: Validate in staging environment
3. **Monitor Closely**: Watch execution metrics
4. **Iterate Continuously**: Improve based on results

### **Approval Workflows**

1. **Define Clear Criteria**: When approval is required
2. **Set Time Limits**: Prevent approval delays
3. **Escalation Paths**: Define escalation procedures
4. **Document Decisions**: Record approval rationale

### **Threat Intelligence**

1. **Multiple Sources**: Use diverse TI sources
2. **Validate Results**: Cross-reference TI data
3. **Update Regularly**: Keep TI sources current
4. **Monitor Quality**: Track TI source accuracy

## Troubleshooting

### **Common Issues**

1. **MCP Connection Failures**:
   - Check MCP server status
   - Verify configuration
   - Test API connectivity

2. **Playbook Execution Failures**:
   - Check trigger conditions
   - Validate action parameters
   - Review error logs

3. **Approval Workflow Issues**:
   - Verify approver configuration
   - Check timeout settings
   - Review escalation paths

### **Debug Commands**

```bash
# Check MCP server status
curl http://localhost:8081/health

# List running workflows
curl http://localhost:8081/workflows

# Get system metrics
curl http://localhost:8081/metrics
```

## Future Enhancements

### **Planned Features**

1. **Machine Learning**: ML-driven playbook selection
2. **Advanced Analytics**: Predictive threat analysis
3. **Integration Expansion**: More security tool integrations
4. **Mobile Interface**: Mobile approval workflows
5. **Compliance Reporting**: Automated compliance reports

### **Integration Opportunities**

1. **SIEM Integration**: Connect with SIEM platforms
2. **Ticketing Systems**: Integrate with ITSM tools
3. **Communication Platforms**: Slack, Teams integration
4. **Asset Management**: CMDB integration
5. **Vulnerability Scanners**: Automated vulnerability response

## Conclusion

The CRA and SOAR integration with MCP provides a powerful, automated security response platform that:

- **Enriches incidents** with comprehensive threat intelligence
- **Automates response** through intelligent playbook execution
- **Ensures security** through approval workflows
- **Provides visibility** through real-time monitoring
- **Scales effectively** through cloud-native architecture

This integration significantly enhances your AI-driven SOC's ability to detect, analyze, and respond to security threats with speed, accuracy, and consistency.

---

*For technical support or questions, refer to the individual component documentation or the test suite for examples.*


