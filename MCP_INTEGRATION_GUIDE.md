# TAA A2A MCP Integration Guide

## Overview

This guide explains how your TAA (Triage & Analysis Agent) with A2A (Agent-to-Agent) capabilities has been enhanced with **MCP (Model Context Protocol)** integration to connect with **Google Security Operations**, **Chronicle**, and **Security Command Center**.

## What is MCP Integration?

**Model Context Protocol (MCP)** is an open standard that enables AI agents to seamlessly connect with external data sources and tools. This implementation uses **open-source MCP servers** for threat intelligence, allowing your ADA to:

- Query multiple threat intelligence sources (VirusTotal, AbuseIPDB, Shodan, ThreatFox, MalwareBazaar)
- Access free and commercial threat intelligence feeds
- Integrate with open-source security tools
- Share threat intelligence across the ecosystem
- Use filesystem access for data storage and retrieval

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ADA Agent     │    │   TAA A2A MCP   │    │   VirusTotal    │
│  (Detection)    │───▶│     Agent       │◄───│   (via MCP)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   AbuseIPDB     │    │ Threat Intel    │    │   MalwareBazaar │
│   (via MCP)     │◄───│    Server       │───▶│   (via MCP)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   ThreatFox     │    │   Filesystem    │
                       │   (via MCP)     │    │   (via MCP)     │
                       └─────────────────┘    └─────────────────┘
```

## Enhanced Capabilities

### 1. Open-Source Threat Intelligence
- **Multi-Source IOC Lookups**: Query IPs, domains, and file hashes across multiple sources
- **VirusTotal Integration**: Access VirusTotal's comprehensive threat database
- **AbuseIPDB Integration**: Check IP reputation and abuse reports
- **Shodan Integration**: Discover exposed services and vulnerabilities
- **ThreatFox Integration**: Access IOCs from ThreatFox database
- **MalwareBazaar Integration**: Query malware samples and families

### 2. Simulated Chronicle Integration
- **Query Detection Rules**: Search simulated detection rules and alerts
- **Historical Analysis**: Analyze past incidents and patterns
- **Threat Hunting**: Perform advanced threat hunting queries
- **Incident Investigation**: Automated investigation workflows

### 3. Filesystem Access
- **Data Storage**: Store and retrieve threat intelligence data
- **File Management**: Manage security artifacts and reports
- **Log Analysis**: Access and analyze security logs
- **Evidence Collection**: Collect and store forensic evidence

### 4. Enhanced A2A Communication
- **MCP-Enhanced Tasks**: A2A tasks now include threat intelligence context
- **Collaborative Analysis**: Multi-agent analysis with TI data
- **Threat Intelligence Sharing**: Share TI across the ecosystem
- **Automated Response**: Enhanced containment and response capabilities

## File Structure

```
ai-driven-soc/
├── taa_a2a_mcp_agent.py          # Main MCP-enhanced agent
├── taa_a2a_mcp_server.py         # FastAPI server with MCP endpoints
├── test_mcp_integration.py       # Integration test client
├── requirements_mcp.txt          # MCP dependencies
├── setup_mcp_integration.sh     # Setup script
├── MCP_INTEGRATION_GUIDE.md     # This guide
└── config/
    └── taa_a2a_config.json      # Updated config with MCP settings
```

## Setup Instructions

### 1. Prerequisites
- Python 3.9+
- Node.js and npm
- Google Cloud SDK
- Service account with appropriate permissions

### 2. Install Dependencies
```bash
# Make setup script executable
chmod +x setup_mcp_integration.sh

# Run setup script
./setup_mcp_integration.sh
```

### 3. Configure Chronicle
Update `config/taa_a2a_config.json`:
```json
{
  "chronicle": {
    "customer_id": "your-chronicle-customer-id",
    "region": "us-central1",
    "max_results": 1000,
    "timeout": 60
  }
}
```

### 4. Start the MCP Server
```bash
# Start the enhanced TAA A2A MCP server
python3 taa_a2a_mcp_server.py
```

### 5. Test Integration
```bash
# Run integration tests
python3 test_mcp_integration.py
```

## API Endpoints

### MCP Endpoints

#### Chronicle Query
```http
POST /mcp/chronicle/query
Content-Type: application/json

{
  "query_text": "metadata.event_type='NETWORK_CONNECTION' AND network.destination.ip_address='8.8.8.8'",
  "time_range": {
    "start_time": "2024-01-01T00:00:00Z",
    "end_time": "2024-01-02T00:00:00Z"
  },
  "max_results": 1000
}
```

#### Threat Intelligence Query
```http
POST /mcp/threat-intelligence/query
Content-Type: application/json

{
  "indicators": ["8.8.8.8", "malware.example.com"],
  "intelligence_type": "ioc"
}
```

#### Incident Investigation
```http
POST /mcp/investigate
Content-Type: application/json

{
  "incident_id": "incident-123",
  "investigation_scope": "deep"
}
```

### Enhanced A2A Endpoints

#### Enhanced Alert Analysis
```http
POST /a2a/tasks
Content-Type: application/json

{
  "task_id": "analysis-456",
  "skill": "analyze_security_alert",
  "parameters": {
    "alert_data": {
      "alert_id": "alert-789",
      "log_data": {
        "src_ip": "192.168.1.100",
        "dst_ip": "8.8.8.8"
      }
    },
    "enrichment_level": "chronicle"
  }
}
```

#### Threat Intelligence Sharing
```http
POST /share/intelligence
Content-Type: application/json

{
  "indicators": ["malicious.ip", "suspicious.domain"],
  "confidence": 0.95,
  "source": "taa-agent",
  "share_with_chronicle": true
}
```

## Configuration

### MCP Server Configuration
```json
{
  "mcp": {
    "enabled": true,
    "servers": {
      "google_security_operations": {
        "name": "google-security-operations",
        "command": "npx",
        "args": ["@google/mcp-server-security-operations"],
        "env": {
          "GOOGLE_CLOUD_PROJECT": "ai-driven-soc",
          "GOOGLE_APPLICATION_CREDENTIALS": "./Service Account BigQuery/sa-gatra-bigquery.json"
        }
      }
    },
    "timeout": 30,
    "retry_attempts": 3
  }
}
```

## Usage Examples

### 1. Enhanced Alert Analysis
```python
import asyncio
import aiohttp

async def analyze_alert_with_chronicle():
    async with aiohttp.ClientSession() as session:
        alert_data = {
            "alert_id": "alert-123",
            "log_data": {
                "src_ip": "192.168.1.100",
                "dst_ip": "8.8.8.8",
                "domain": "suspicious.example.com"
            },
            "enrichment_level": "chronicle"
        }
        
        async with session.post(
            "http://localhost:8080/analyze/enhanced",
            json=alert_data
        ) as response:
            result = await response.json()
            print(f"Analysis result: {result}")

asyncio.run(analyze_alert_with_chronicle())
```

### 2. Chronicle Threat Hunting
```python
async def hunt_threats():
    async with aiohttp.ClientSession() as session:
        query = {
            "query_text": "metadata.event_type='PROCESS_LAUNCH' AND target.process.command_line CONTAINS 'powershell'",
            "time_range": {
                "start_time": "2024-01-01T00:00:00Z",
                "end_time": "2024-01-02T00:00:00Z"
            }
        }
        
        async with session.post(
            "http://localhost:8080/mcp/chronicle/query",
            json=query
        ) as response:
            results = await response.json()
            print(f"Threat hunting results: {results}")

asyncio.run(hunt_threats())
```

### 3. Multi-Agent Collaboration
```python
async def collaborative_analysis():
    async with aiohttp.ClientSession() as session:
        task = {
            "task_id": "collab-789",
            "skill": "collaborative_analysis",
            "parameters": {
                "analysis_type": "threat_hunting",
                "data": {"indicators": ["malicious.ip", "suspicious.domain"]},
                "collaboration_mode": "parallel",
                "include_chronicle_context": True
            }
        }
        
        async with session.post(
            "http://localhost:8080/a2a/tasks",
            json=task
        ) as response:
            result = await response.json()
            print(f"Collaborative analysis: {result}")

asyncio.run(collaborative_analysis())
```

## Monitoring and Troubleshooting

### Health Checks
```bash
# Check server health
curl http://localhost:8080/health

# Check MCP servers status
curl http://localhost:8080/mcp/servers

# Get agent capabilities
curl http://localhost:8080/a2a/capabilities
```

### Logs
- Server logs: Check console output
- MCP logs: Check logs in `logs/` directory
- Chronicle logs: Check Chronicle console for query logs

### Common Issues

#### 1. MCP Servers Not Starting
```bash
# Check if MCP servers are installed
npm list -g | grep mcp-server

# Reinstall if needed
npm install -g @google/mcp-server-security-operations
```

#### 2. Authentication Errors
```bash
# Check service account key
ls -la "Service Account BigQuery/sa-gatra-bigquery.json"

# Set environment variable
export GOOGLE_APPLICATION_CREDENTIALS="Service Account BigQuery/sa-gatra-bigquery.json"
```

#### 3. Chronicle Connection Issues
- Verify Chronicle customer ID in config
- Check network connectivity to Chronicle API
- Verify service account has Chronicle permissions

## Security Considerations

### 1. Authentication
- Use service accounts with minimal required permissions
- Rotate service account keys regularly
- Monitor access logs

### 2. Data Privacy
- MCP communication is encrypted in transit
- Chronicle data is not stored locally
- Threat intelligence is shared responsibly

### 3. Rate Limiting
- MCP servers implement rate limiting
- Chronicle API has rate limits
- Monitor usage to avoid throttling

## Performance Optimization

### 1. Caching
- Chronicle query results are cached
- Threat intelligence responses are cached
- Cache TTL is configurable

### 2. Parallel Processing
- Multiple MCP servers can run in parallel
- A2A tasks support parallel execution
- Chronicle queries can be batched

### 3. Timeout Configuration
```json
{
  "mcp": {
    "timeout": 30,
    "retry_attempts": 3
  },
  "chronicle": {
    "timeout": 60,
    "max_results": 1000
  }
}
```

## Next Steps

### 1. Advanced Threat Hunting
- Implement custom Chronicle detection rules
- Create automated threat hunting workflows
- Build threat intelligence correlation

### 2. Incident Response Automation
- Automated containment actions
- Incident escalation workflows
- Evidence collection automation

### 3. Machine Learning Integration
- Train models on Chronicle data
- Implement anomaly detection
- Build predictive analytics

### 4. Integration Expansion
- Add more MCP servers
- Integrate with SIEM platforms
- Connect to external threat feeds

## Support and Resources

- **MCP Documentation**: https://github.com/google/mcp-security
- **Chronicle Documentation**: https://cloud.google.com/chronicle/docs
- **Google Security Operations**: https://cloud.google.com/security-operations
- **A2A Protocol**: https://cloud.google.com/agentspace

## Conclusion

The MCP integration transforms your TAA agent into a powerful, Chronicle-connected security analysis platform. With access to Google's security ecosystem, your agent can now provide enhanced threat intelligence, automated investigations, and collaborative security operations.

The integration maintains backward compatibility with your existing A2A setup while adding powerful new capabilities through MCP. This positions your SOC for advanced threat hunting, automated incident response, and intelligent security operations.
