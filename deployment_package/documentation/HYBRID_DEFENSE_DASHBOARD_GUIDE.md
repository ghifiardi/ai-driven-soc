# 🛡️ Hybrid Cyber Defense Agent Dashboard Guide

## 🎯 Overview

The **Hybrid Cyber Defense Agent Dashboard** provides real-time monitoring and analytics for the DQN + Gemini hybrid cybersecurity agent, fully integrated with your existing SOC infrastructure.

### 🚀 Quick Access
```
Dashboard URL: http://10.45.254.19:8529
Service Name: hybrid-defense-dashboard.service
```

---

## 📊 Dashboard Features

### **🏠 Overview Tab**
- **📈 Key Metrics**: Alerts processed, actions taken, AI explanations, system errors
- **🎯 Action Distribution**: Visual breakdown of defensive actions recommended
- **📈 Performance Trends**: 12-hour alert processing trends
- **Real-time Status**: Live agent health and activity monitoring

### **🤖 DQN Performance Tab**  
- **⚡ Inference Speed**: Real-time performance gauge (target: <100ms)
- **📊 Time Distribution**: Histogram of DQN inference times
- **🎯 Model Status**: Circuit breaker status, model configuration
- **Performance Benchmarks**: Automatic performance validation

### **🧠 Gemini Analytics Tab**
- **📞 API Usage Statistics**: Gemini usage rate and call volume
- **⚡ Response Performance**: 24-hour API response time trends
- **💬 Recent Explanations**: Sample AI-generated explanations with confidence scores
- **Cost Monitoring**: API usage tracking and optimization insights

### **🚨 Live Alerts Tab**
- **📋 Alert Activity**: Real-time alert processing table
- **🔥 Attack Heatmap**: Visual attack type frequency analysis
- **📡 Live Feed**: Streaming alert notifications
- **🎯 Filtering & Search**: Advanced alert filtering capabilities

### **⚙️ System Health Tab**
- **🔌 Integration Status**: Health check for all connected agents (ADA/TAA/CRA/CLA)
- **🔧 Circuit Breakers**: Real-time circuit breaker monitoring
- **📊 Error Analysis**: Detailed error tracking and categorization
- **💻 Resource Usage**: System performance metrics and health scores

---

## 🏗️ Architecture Integration

### **SOC Infrastructure Integration**
```
┌─────────────────────────────────────────────────────┐
│                SOC DASHBOARD ECOSYSTEM               │
├─────────────────────────────────────────────────────┤
│  Port 8511: 🛡️ Main Gatra Analytics Platform       │
│  Port 8503: 🤖 AI Model Training Dashboard          │
│  Port 8527: 📊 Comprehensive Real Data Dashboard    │
│  Port 8528: 🔄 Production Dashboard Correlator      │
│  Port 8529: 🛡️ Hybrid Defense Dashboard (NEW)      │
│  Port 99:   📱 Mobile Feedback Dashboard            │
└─────────────────────────────────────────────────────┘
```

### **Data Flow Architecture**
```
ADA Alerts → Hybrid Agent → Dashboard Metrics
     ↓             ↓              ↓
Pub/Sub → DQN + Gemini → Real-time Updates
```

---

## 🚀 Deployment & Management

### **1. Deploy Dashboard**
```bash
# Run deployment script
./deploy_hybrid_dashboard.sh

# Manual deployment
python3 -m streamlit run hybrid_defense_dashboard.py --server.port=8529 --server.address=0.0.0.0
```

### **2. Service Management**
```bash
# Start dashboard service
sudo systemctl start hybrid-defense-dashboard

# Stop dashboard service  
sudo systemctl stop hybrid-defense-dashboard

# Restart dashboard service
sudo systemctl restart hybrid-defense-dashboard

# Check service status
sudo systemctl status hybrid-defense-dashboard

# View live logs
sudo journalctl -u hybrid-defense-dashboard -f
```

### **3. Health Monitoring**
```bash
# Quick health check
curl http://localhost:8529/_stcore/health

# Dashboard accessibility test
curl -s http://localhost:8529 | grep "Hybrid Cyber Defense"
```

---

## 🔧 Configuration & Customization

### **Dashboard Configuration**
Edit `hybrid_defense_dashboard.py` to customize:

```python
# Agent connection settings
AGENT_HOST = "localhost"
AGENT_PORT = 8083
AGENT_URL = f"http://{AGENT_HOST}:{AGENT_PORT}"

# Dashboard styling
st.set_page_config(
    page_title="🛡️ Hybrid Cyber Defense Agent Dashboard",
    page_icon="🛡️", 
    layout="wide"
)
```

### **Port Configuration**
Default port: `8529`
- Available ports: `8530-8540` (automatically selected if 8529 is busy)
- Configure in deployment script: `DASHBOARD_PORT="8529"`

### **Auto-Refresh Settings**
- Default: 30-second auto-refresh (optional)
- Manual refresh available via sidebar button
- Real-time feed updates automatically

---

## 📈 Monitoring & Metrics

### **Key Performance Indicators (KPIs)**
1. **📊 Alerts Processed**: Total security alerts handled
2. **⚡ Actions Recommended**: Defensive actions suggested
3. **🧠 AI Explanations Generated**: Gemini explanation count
4. **⚠️ System Errors**: Error rate and classification

### **Performance Benchmarks**
- **DQN Inference**: Target <100ms average
- **Gemini Response**: Target <3s average  
- **End-to-End Processing**: Target <500ms
- **System Uptime**: Target >99.9%

### **Alert Categories Tracked**
- `malware`, `ddos`, `intrusion`, `phishing`, `ransomware`
- Actions: `isolate_node`, `block_traffic`, `patch_system`, `monitor`, `no_action`

---

## 🔐 Security & Access Control

### **Network Security**
- **Internal Access Only**: Bound to `0.0.0.0` but accessible via VPN/internal network
- **No Authentication**: Relies on network-level security (VPN, firewall)
- **HTTPS**: Can be configured with reverse proxy (nginx/apache)

### **Data Privacy**
- **No Persistent Storage**: All data in memory/session state
- **Mock Mode**: Automatic fallback when agent unavailable
- **Secure APIs**: Uses existing SOC security infrastructure

---

## 🐛 Troubleshooting

### **Common Issues & Solutions**

#### **Dashboard Won't Start**
```bash
# Check port availability
lsof -i :8529

# Check service logs
sudo journalctl -u hybrid-defense-dashboard -n 50

# Restart service
sudo systemctl restart hybrid-defense-dashboard
```

#### **Agent Connection Failed**
- ✅ **Solution**: Dashboard automatically switches to demo mode
- 📊 **Mock Data**: Full functionality with simulated data
- 🔄 **Auto-Recovery**: Reconnects when agent becomes available

#### **Missing Dependencies**
```bash
# Install required packages
pip3 install streamlit plotly requests pandas numpy

# Or use requirements file
pip3 install -r requirements.txt
```

#### **Permission Issues**
```bash
# Fix service file permissions
sudo chmod 644 /etc/systemd/system/hybrid-defense-dashboard.service

# Reload systemd
sudo systemctl daemon-reload
```

### **Debug Mode**
Enable debug logging in the dashboard:
```python
# Add to dashboard code
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## 🔗 Integration with Existing Systems

### **SOC Dashboard Links**
- **Main Dashboard**: [http://10.45.254.19:8511](http://10.45.254.19:8511) - Gatra Analytics Platform
- **AI Training**: [http://10.45.254.19:8503](http://10.45.254.19:8503) - Model Training Dashboard  
- **Real Data**: [http://10.45.254.19:8527](http://10.45.254.19:8527) - Comprehensive Data Dashboard
- **Correlator**: [http://10.45.254.19:8528](http://10.45.254.19:8528) - Production Correlator
- **Mobile**: [http://10.45.254.19:99](http://10.45.254.19:99) - Feedback Dashboard

### **API Integration**
The dashboard connects to:
- **Hybrid Agent API**: `http://localhost:8083` (health, status, metrics)
- **ADA Agent**: `http://localhost:8081` (integration status)
- **TAA Agent**: `http://localhost:8080` (integration status)
- **CRA Agent**: `http://localhost:8084` (integration status)
- **CLA Agent**: `http://localhost:8082` (integration status)

---

## 📞 Support & Maintenance

### **Maintenance Schedule**
- **Daily**: Automatic log rotation via systemd
- **Weekly**: Performance metrics review
- **Monthly**: Dashboard updates and feature enhancements

### **Backup & Recovery**
```bash
# Backup dashboard configuration
cp hybrid_defense_dashboard.py hybrid_defense_dashboard.backup

# Backup service configuration  
sudo cp /etc/systemd/system/hybrid-defense-dashboard.service /etc/systemd/system/hybrid-defense-dashboard.service.backup
```

### **Updates & Upgrades**
```bash
# Update dashboard
git pull origin main
sudo systemctl restart hybrid-defense-dashboard

# Update dependencies
pip3 install -r requirements.txt --upgrade
```

---

## 🎉 Success Metrics

### **Dashboard Effectiveness KPIs**
- ✅ **Uptime**: >99% availability
- ✅ **Response Time**: <2s page load time  
- ✅ **Data Accuracy**: Real-time sync with agent metrics
- ✅ **User Experience**: Intuitive navigation and insights

### **SOC Integration Success**
- ✅ **Unified Monitoring**: Single pane of glass for hybrid agent
- ✅ **Alert Correlation**: Cross-reference with other SOC dashboards
- ✅ **Performance Visibility**: Clear DQN + Gemini performance metrics
- ✅ **Operational Efficiency**: Reduced MTTR for security incidents

---

🛡️ **The Hybrid Defense Dashboard is now fully integrated with your SOC infrastructure, providing comprehensive monitoring for the DQN + Gemini hybrid cybersecurity agent!** 🎉
