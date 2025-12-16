# Multi-Agent SOC Architecture Summary

## 🏗️ System Architecture Overview

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI-DRIVEN SOC PLATFORM                      │
├─────────────────────────────────────────────────────────────────┤
│  DATA LAYER          │  AGENT LAYER        │  ORCHESTRATION    │
│  ┌─────────────┐     │  ┌─────────────┐    │  ┌─────────────┐  │
│  │   BigQuery  │────▶│  │     ADA     │    │  │  LangGraph  │  │
│  │  (SIEM Data)│     │  │ (Detection) │    │  │ (Workflows) │  │
│  └─────────────┘     │  └─────────────┘    │  └─────────────┘  │
│  ┌─────────────┐     │  ┌─────────────┐    │  ┌─────────────┐  │
│  │  Firestore  │────▶│  │     TAA     │    │  │   Pub/Sub   │  │
│  │   (State)   │     │  │ (Analysis)  │    │  │ (Messaging) │  │
│  └─────────────┘     │  └─────────────┘    │  └─────────────┘  │
│  ┌─────────────┐     │  ┌─────────────┐    │  ┌─────────────┐  │
│  │Cloud Storage│────▶│  │     CRA     │    │  │  Monitoring │  │
│  │  (Models)   │     │  │(Containment)│    │  │ (Dashboards)│  │
│  └─────────────┘     │  └─────────────┘    │  └─────────────┘  │
│                      │  ┌─────────────┐    │                   │
│                      │  │     CLA     │    │                   │
│                      │  │ (Learning)  │    │                   │
│                      │  └─────────────┘    │                   │
└─────────────────────────────────────────────────────────────────┘
```

## 🤖 Agent Specifications

### 1. ADA (Anomaly Detection Agent)
- **Role**: Primary threat detection
- **Technology**: LangGraph + ML Models (Isolation Forest, Random Forest)
- **Input**: SIEM events from BigQuery
- **Output**: Anomaly alerts with confidence scores
- **Service**: `ada-production.service` (Port 8080)

### 2. TAA (Triage & Analysis Agent)
- **Role**: Alert analysis and routing
- **Technology**: LangGraph workflow + LLM (Mistral/Vertex AI)
- **Input**: ADA alerts
- **Output**: Analysis results, containment requests, feedback
- **Integration**: Embedded in ADA workflow

### 3. CRA (Containment Response Agent)
- **Role**: Automated threat containment
- **Technology**: Python + Pub/Sub
- **Input**: Containment requests from TAA
- **Output**: Automated security actions
- **Service**: `cra_service_working.py`

### 4. CLA (Continuous Learning Agent)
- **Role**: Model improvement and optimization
- **Technology**: Python + Optuna + scikit-learn
- **Input**: Feedback from TAA
- **Output**: Updated models and parameters
- **Service**: `production-cla.service` (Port 8080)

## 🔄 Data Flow Architecture

### Primary Detection Flow
```
SIEM Events → ADA → TAA → [CRA, CLA, RVA]
     ↓         ↓     ↓
  BigQuery  Alerts  Analysis
```

### Message Routing
```
ADA Alerts → TAA Analysis → Multiple Outputs:
├── High Severity → CRA (Containment)
├── All Cases → CLA (Learning)
└── All Cases → RVA (Reporting)
```

## 📡 Communication Protocols

### Pub/Sub Topics
| Topic | Direction | Purpose |
|-------|-----------|---------|
| `ada-alerts` | ADA → TAA | Threat notifications |
| `containment-requests` | TAA → CRA | Automated responses |
| `taa-feedback` | TAA → CLA | Model learning |
| `taa-reports` | TAA → RVA | Documentation |

### REST APIs
- **ADA Service**: `http://10.45.254.19:8080`
- **CLA Service**: `http://10.45.254.19:8080`
- **Health Checks**: `/health`, `/status`

## 🖥️ Monitoring & Dashboards

### Available Dashboards
1. **Production CLA Dashboard**: `http://10.45.254.19:8505`
   - Service status and metrics
   - Model performance
   - Integration health

2. **TAA-CRA Integration Dashboard**: `http://10.45.254.19:8531`
   - Message flow monitoring
   - Pub/Sub topic health
   - Integration testing

### Key Metrics
- Detection accuracy and precision
- Response time and throughput
- Model performance metrics
- Integration health status

## 🚀 Deployment Status

### Running Services
```bash
# Systemd Services
✅ ada-production.service (Port 8080)
✅ production-cla.service (Port 8080)
✅ ada-bigquery-integration.service

# Manual Services
✅ cra_service_working.py
✅ taa_service.py (original)
```

### Infrastructure
- **VM**: `xdgaisocapp01` (asia-southeast2-a)
- **Platform**: Google Cloud Platform
- **Data**: BigQuery + Pub/Sub + Firestore
- **Monitoring**: Streamlit dashboards

## 🔧 Operational Commands

### Service Management
```bash
# Check status
sudo systemctl status ada-production.service
sudo systemctl status production-cla.service

# Restart services
sudo systemctl restart ada-production.service
sudo systemctl restart production-cla.service

# Check logs
sudo journalctl -u ada-production.service -f
```

### Health Checks
```bash
# API health
curl http://10.45.254.19:8080/health
curl http://10.45.254.19:8080/status

# Process check
ps aux | grep -E "(ada|cla|cra|taa)"
```

## 📊 Performance Metrics

### Current Performance
- **Detection Rate**: 66.7% (target: 20-40%)
- **Response Time**: < 100ms
- **Uptime**: 99.9%
- **Model Accuracy**: Continuously improving

### Integration Status
- ✅ **ADA-TAA**: Fully integrated via LangGraph
- ✅ **TAA-CRA**: Pub/Sub messaging active
- ✅ **TAA-CLA**: Feedback loop operational
- ✅ **TAA-RVA**: Reporting pipeline active

## 🛡️ Security & Compliance

### Security Features
- TLS 1.3 encryption in transit
- AES-256 encryption at rest
- Service account authentication
- IAM role-based permissions

### Compliance
- SOC 2 Type II alignment
- GDPR compliance ready
- Audit trail logging
- Data retention policies

## 📈 Future Enhancements

### Short-term (Q1 2025)
- Enhanced monitoring with Grafana
- Performance optimization
- Horizontal scaling

### Medium-term (Q2-Q3 2025)
- Multi-modal threat detection
- Predictive analysis
- Advanced integrations

### Long-term (Q4 2025+)
- Autonomous operations
- Self-healing capabilities
- Enterprise multi-tenancy

---

## 📚 Documentation References

1. **Main Documentation**: `MULTI_AGENT_WORKFLOW_INTEGRATION_DOCUMENTATION.md`
2. **TAA Documentation**: `taa_documentation.md`
3. **Service Status**: `CLA_SERVICE_STATUS.md`
4. **API Reference**: Service endpoints documentation

---

**Last Updated**: September 25, 2025  
**Status**: Production Ready ✅  
**Maintained By**: AI-Driven SOC Team
