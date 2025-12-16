# 🚀 Quick Deployment Guide - Revamped LangGraph ADA Dashboard

## ⚡ Fast Start (5 minutes)

### **1. Check Current Status**
```bash
# Check if dashboard is running
gcloud compute ssh app@xdgaisocapp01 --zone=asia-southeast2-a --command="ps aux | grep revamped_langgraph_dashboard"

# Check port status
gcloud compute ssh app@xdgaisocapp01 --zone=asia-southeast2-a --command="sudo netstat -tulnp | grep 8509"
```

### **2. Start Dashboard (if not running)**
```bash
gcloud compute ssh app@xdgaisocapp01 --zone=asia-southeast2-a --command="cd /home/app/langgraph-ada-dashboard && /home/app/.local/bin/streamlit run revamped_langgraph_dashboard.py --server.port 8509 --server.address 0.0.0.0 --server.headless true &"
```

### **3. Access Dashboard**
```
🌐 URL: http://10.45.254.19:8509
```

## 🔧 Quick Troubleshooting

### **Dashboard Not Loading?**
```bash
# Kill old processes
gcloud compute ssh app@xdgaisocapp01 --zone=asia-southeast2-a --command="pkill -f langgraph_dashboard"

# Start fresh
gcloud compute ssh app@xdgaisocapp01 --zone=asia-southeast2-a --command="cd /home/app/langgraph-ada-dashboard && /home/app/.local/bin/streamlit run revamped_langgraph_dashboard.py --server.port 8509 --server.address 0.0.0.0 --server.headless true &"
```

### **Port Already in Use?**
```bash
# Find process using port
gcloud compute ssh app@xdgaisocapp01 --zone=asia-southeast2-a --command="sudo netstat -tulnp | grep 8509"

# Kill process
gcloud compute ssh app@xdgaisocapp01 --zone=asia-southeast2-a --command="sudo kill -9 <PID>"
```

## 📊 What You'll See

✅ **Enhanced Dark Theme** with better contrast  
✅ **Real-time BigQuery Data** from your tables  
✅ **Interactive Charts** with hover effects  
✅ **Professional UI** optimized for security operations  
✅ **Responsive Design** for all screen sizes  

## 🎯 Key Features

- **📊 SIEM Events Count**: Real-time security event monitoring
- **🚨 ADA Alerts**: Anomaly detection results
- **🤖 Active Agents**: AI agent status
- **📈 Performance Gauges**: Response time and success rate
- **🔄 Activity Timeline**: 48-hour trend analysis
- **🏥 System Health**: Integration status monitoring

## 🔄 Update Dashboard

```bash
# Copy new version
gcloud compute scp revamped_langgraph_dashboard.py app@xdgaisocapp01:/home/app/langgraph-ada-dashboard/ --zone=asia-southeast2-a

# Restart
gcloud compute ssh app@xdgaisocapp01 --zone=asia-southeast2-a --command="pkill -f revamped_langgraph_dashboard && cd /home/app/langgraph-ada-dashboard && /home/app/.local/bin/streamlit run revamped_langgraph_dashboard.py --server.port 8509 --server.address 0.0.0.0 --server.headless true &"
```

---

**Dashboard**: `http://10.45.254.19:8509`  
**Status**: Production Ready  
**Theme**: Enhanced Dark Mode  
**Data**: Real-time BigQuery Integration
