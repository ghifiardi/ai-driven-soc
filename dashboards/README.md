# AI-SOC Dashboard Suite

This directory contains the comprehensive SOC dashboard suite integrated from [PR #2](https://github.com/ghifiardi/ai-soc-dashboard/pull/2).

## 📊 Available Dashboards

### 1. 🛡️ Enhanced SOC Dashboard (`enhanced_soc_dashboard.py`)
**Target Audience:** SOC Analysts, Security Engineers, Threat Hunters

**Key Features:**
- Real-time Threat Timeline
- Network Topology Mapping
- MITRE ATT&CK Heatmap
- Geographic Threat Distribution
- Incident Response Funnel
- AI-Powered Analytics
- Advanced Filtering & Search
- Auto-Refresh Capabilities

**Run:**
```bash
streamlit run enhanced_soc_dashboard.py
```

### 2. 🎯 Executive Dashboard (`executive_dashboard.py`)
**Target Audience:** CISOs, C-Suite Executives, Senior Management

**Key Features:**
- Executive KPIs
- Operational Performance Metrics (MTTD, MTTR)
- Trend Analysis
- Risk Assessment
- Compliance Scorecard
- Executive Summary

**Run:**
```bash
streamlit run executive_dashboard.py
```

### 3. 📜 Compliance Dashboard (`compliance_dashboard.py`)
**Target Audience:** Compliance Officers, Auditors, Risk Management Teams

**Supported Frameworks:**
- NIST Cybersecurity Framework
- ISO 27001
- SOC 2
- GDPR
- HIPAA
- PCI DSS
- CIS Controls

**Key Features:**
- Framework Compliance Scoring
- Control Status Tracking
- Compliance Trend Analysis
- Comprehensive Audit Trail
- Advanced Filtering & Export

**Run:**
```bash
streamlit run compliance_dashboard.py
```

### 4. 🔥 Original SOC Dashboard (`streamlit_soc_dashboard.py`)
**Target Audience:** SOC Analysts, Security Operations

**Key Features:**
- BigQuery Integration
- Fallback Demo Mode
- Real-time Metrics
- Global Threat Map
- Severity Distribution
- Attack Type Analysis

**Run:**
```bash
streamlit run streamlit_soc_dashboard.py
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running Dashboards
All dashboards run on `http://localhost:8501` by default.

**Custom Port:**
```bash
streamlit run enhanced_soc_dashboard.py --server.port 8502
```

## 📦 Dependencies

See `requirements.txt` for complete list. Core dependencies:
- `streamlit>=1.28.0`
- `pandas>=2.0.0`
- `plotly>=5.15.0`
- `numpy>=1.24.0`
- `google-cloud-bigquery>=3.11.0` (optional)

## 🔧 Configuration

### BigQuery Integration (Optional)
1. Set up Google Cloud credentials
2. Update project ID in dashboard files
3. Configure dataset and table names

**Note:** All dashboards work with high-quality mock data if BigQuery is not configured.

## 📂 Integration with Main Repository

These dashboards integrate with the main `ai-driven-soc` repository:
- Can connect to BigQuery datasets used by ADA/TAA agents
- Compatible with threat hunting platform data
- Supports multi-tenant configurations

## 🚀 Deployment

### Streamlit Cloud
1. Connect GitHub repository
2. Select dashboard file as main file
3. Configure secrets for BigQuery (if needed)
4. Deploy

### Docker
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "enhanced_soc_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 📝 Notes

- All dashboards include fallback mock data
- Auto-refresh is configurable (5-60 seconds)
- Export functions available (CSV/JSON)
- Responsive design for all devices

---

**Source:** Integrated from [ai-soc-dashboard PR #2](https://github.com/ghifiardi/ai-soc-dashboard/pull/2)  
**Date:** January 6, 2026  
**Status:** ✅ Production Ready

