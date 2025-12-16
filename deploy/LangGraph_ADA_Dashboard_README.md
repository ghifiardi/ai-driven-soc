# 🧠 LangGraph ADA Agent Dashboard

## Overview
A specialized, real-time dashboard for monitoring the **LangGraph ADA (Anomaly Detection Agent)** integration, workflow orchestration, and ML model performance. This dashboard provides comprehensive visibility into the AI-powered security anomaly detection system.

## 🎯 Purpose
Monitor and analyze the performance of:
- **LangGraph Workflow States** - Real-time workflow orchestration
- **ML Model Performance** - Accuracy, precision, recall, inference time
- **ADA Agent Metrics** - Anomaly detection, processing times, confidence scores
- **System Integration Health** - BigQuery, LangGraph, ML models, ADA agent status

## 🚀 Features

### 📊 Real-time Monitoring
- **Active Workflows**: Count of currently running LangGraph workflows
- **Completed Workflows**: Successfully processed workflows in last 24h
- **Failed Workflows**: Failed workflows with error tracking
- **Pending Workflows**: Workflows waiting to be processed

### 🤖 ML Model Analytics
- **Performance Metrics**: Accuracy, precision, recall, F1-score trends
- **Inference Time**: Real-time model response time monitoring
- **Performance Thresholds**: Visual alerts for optimal/warning/critical performance
- **Historical Trends**: 7-day performance analysis

### 🔄 Workflow Orchestration
- **State Transitions**: Sankey diagram showing workflow flow
- **State Distribution**: Visual breakdown of workflow states
- **Success Rates**: Workflow completion success percentages
- **Real-time Execution**: Live workflow status monitoring

### 🏥 System Health
- **Integration Status**: BigQuery, LangGraph, ML models, ADA agent
- **Performance Metrics**: Response time, error rate, throughput
- **Resource Usage**: System resource utilization monitoring
- **Alert System**: Automated health status alerts

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph ADA Dashboard                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  📊 Real-time Metrics Collection                          │
│     ├─ LangGraph Workflow States                          │
│     ├─ ML Model Performance                               │
│     ├─ ADA Agent Metrics                                  │
│     └─ System Health Status                               │
│                                                             │
│  🔄 Data Sources                                          │
│     ├─ BigQuery (Production Data)                         │
│     ├─ Mock Data (Demo Mode)                              │
│     └─ Real-time Streams                                  │
│                                                             │
│  📈 Visualization Engine                                  │
│     ├─ Plotly Charts                                      │
│     ├─ Interactive Tables                                 │
│     ├─ Real-time Updates                                  │
│     └─ Responsive Design                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 🛠️ Technology Stack

### Frontend
- **Streamlit**: Interactive web application framework
- **Plotly**: Advanced data visualization library
- **Custom CSS**: LangGraph-themed dark mode interface

### Backend
- **Python 3.8+**: Core application logic
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations

### Data Sources
- **BigQuery**: Production data integration
- **Mock Data**: Demo and testing capabilities
- **Real-time Streams**: Live workflow monitoring

### Deployment
- **Docker**: Containerized deployment
- **Systemd**: Service management
- **Nginx**: Reverse proxy and load balancing

## 📁 File Structure

```
langgraph_ada_dashboard/
├── langgraph_ada_dashboard.py          # Main dashboard application
├── langgraph_dashboard_requirements.txt # Python dependencies
├── deploy_langgraph_dashboard.sh       # Deployment script
├── langgraph-dashboard.service         # Systemd service file
├── start_langgraph_dashboard.sh        # Startup script
├── langgraph-dashboard-nginx.conf      # Nginx configuration
└── README.md                           # This file
```

## 🚀 Quick Start

### 1. Local Development
```bash
# Install dependencies
pip install -r langgraph_dashboard_requirements.txt

# Run dashboard
streamlit run langgraph_ada_dashboard.py --server.port 8507
```

### 2. Production Deployment
```bash
# Make deployment script executable
chmod +x deploy_langgraph_dashboard.sh

# Deploy to VM
./deploy_langgraph_dashboard.sh
```

### 3. Manual Deployment
```bash
# Copy files to VM
gcloud compute scp --recurse langgraph_dashboard_deploy_* app@xdgaisocapp01:/home/app/ --zone=asia-southeast2-a

# Install and start on VM
gcloud compute ssh app@xdgaisocapp01 --zone=asia-southeast2-a
cd /home/app/langgraph_dashboard_deploy_*
./deploy.sh
```

## 🌐 Access URLs

### Direct Access
- **Dashboard**: `http://10.45.254.19:8507`
- **Local Development**: `http://localhost:8507`

### Nginx Proxy (if configured)
- **Proxied Access**: `http://10.45.254.19/langgraph-dashboard/`

## 📊 Dashboard Sections

### 1. Header Section
- **Status Indicator**: Real-time system status
- **Last Update**: Timestamp of last data refresh
- **LangGraph Theme**: Branded visual design

### 2. Key Metrics Cards
- **Active Workflows**: Currently running workflows
- **Completed Workflows**: Successfully processed workflows
- **Model Accuracy**: Current ML model performance
- **Anomalies Detected**: Security threats identified

### 3. Workflow State Monitoring
- **Workflow Distribution**: Active, pending, failed counts
- **Success Rate**: Workflow completion percentage
- **Performance Metrics**: Inference time, processing time, confidence

### 4. ML Model Performance
- **Performance Trends**: 7-day accuracy, precision, recall trends
- **Inference Time**: Response time monitoring with thresholds
- **Performance Alerts**: Visual indicators for optimal/warning states

### 5. Workflow State Transitions
- **Sankey Diagram**: Visual workflow flow representation
- **State Flow**: Data preprocessing → Feature extraction → Model inference → Post processing → Result storage

### 6. Real-time Workflow Monitoring
- **Execution Table**: Recent workflow executions with status
- **Success Statistics**: Total, successful, failed counts
- **Performance Metrics**: Average duration, error tracking

### 7. System Health
- **Integration Status**: All system components health
- **Performance Metrics**: Response time, error rate, throughput
- **Resource Usage**: System resource utilization

## ⚙️ Configuration

### Environment Variables
```bash
# Dashboard Configuration
DASHBOARD_PORT=8507                    # Dashboard port number
BIGQUERY_PROJECT_ID=chronicle-dev-2be9 # BigQuery project ID
BIGQUERY_DATASET_ID=gatra_database     # BigQuery dataset ID
```

### Data Source Selection
- **Mock Data (Demo)**: For testing and demonstration
- **Real BigQuery Data**: For production monitoring

### Time Range Options
- **Last 24 Hours**: Real-time monitoring
- **Last 7 Days**: Weekly trends
- **Last 30 Days**: Monthly analysis
- **Custom**: User-defined time ranges

## 🔧 Service Management

### Systemd Commands
```bash
# Start dashboard
sudo systemctl start langgraph-dashboard

# Stop dashboard
sudo systemctl stop langgraph-dashboard

# Restart dashboard
sudo systemctl restart langgraph-dashboard

# Check status
sudo systemctl status langgraph-dashboard

# View logs
sudo journalctl -u langgraph-dashboard -f
```

### Manual Control
```bash
# Start manually
cd /home/app/langgraph-ada-dashboard
./start_langgraph_dashboard.sh

# Check process
ps aux | grep langgraph_ada_dashboard
```

## 📈 Data Sources

### BigQuery Tables
- **`ada_workflow_states`**: Workflow execution states
- **`ada_model_performance`**: ML model performance metrics
- **`ada_anomaly_results`**: Anomaly detection results

### Mock Data Structure
```python
{
    'workflow_states': {
        'active_workflows': 12,
        'completed_workflows': 156,
        'failed_workflows': 3,
        'pending_workflows': 8
    },
    'ml_model_performance': {
        'accuracy': 0.94,
        'precision': 0.91,
        'recall': 0.89,
        'f1_score': 0.90,
        'inference_time_ms': 45.2
    },
    'ada_agent_metrics': {
        'alerts_processed': 1247,
        'anomalies_detected': 89,
        'false_positives': 12,
        'avg_processing_time': 2.3,
        'model_confidence': 0.87
    }
}
```

## 🎨 Customization

### Theme Colors
```css
:root {
    --langgraph-primary: #6366f1;    /* Primary blue */
    --langgraph-secondary: #8b5cf6;  /* Secondary purple */
    --langgraph-accent: #06b6d4;     /* Accent cyan */
    --langgraph-success: #10b981;    /* Success green */
    --langgraph-warning: #f59e0b;    /* Warning orange */
    --langgraph-error: #ef4444;      /* Error red */
}
```

### Adding New Metrics
1. **Extend mock data functions** in `get_mock_langgraph_metrics()`
2. **Add new metric cards** in the main dashboard layout
3. **Create visualization components** for new data types
4. **Update BigQuery queries** for real data integration

## 🐛 Troubleshooting

### Common Issues

#### Dashboard Not Starting
```bash
# Check service status
sudo systemctl status langgraph-dashboard

# Check logs
sudo journalctl -u langgraph-dashboard -f

# Verify port availability
sudo netstat -tulnp | grep 8507
```

#### BigQuery Connection Issues
```bash
# Verify service account
ls -la /home/app/ai-driven-soc/gatra-user-gemini.json

# Test BigQuery connection
python3 -c "from google.cloud import bigquery; print('BigQuery OK')"
```

#### Port Conflicts
```bash
# Find process using port
sudo ss -tulnp | grep 8507

# Kill conflicting process
sudo kill -9 <PID>

# Restart dashboard
sudo systemctl restart langgraph-dashboard
```

### Performance Issues
- **Enable static charts**: Reduces JavaScript overhead
- **Limit data queries**: Use appropriate time ranges
- **Monitor resource usage**: Check CPU/memory consumption

## 🔒 Security Considerations

### Access Control
- **Port restrictions**: Only necessary ports exposed
- **Service account**: Limited BigQuery permissions
- **Network isolation**: Internal network access only

### Data Privacy
- **No sensitive data**: Only metadata and metrics
- **Anonymized logs**: No personal information displayed
- **Audit logging**: All access attempts logged

## 📚 API Reference

### Dashboard Functions
- **`get_mock_langgraph_metrics()`**: Generate mock metrics
- **`get_mock_ml_performance()`**: Generate ML performance data
- **`get_mock_workflow_states()`**: Generate workflow state data

### Chart Components
- **Performance Trends**: Line charts with multiple metrics
- **Inference Time**: Time series with threshold lines
- **Workflow Transitions**: Sankey diagram for state flow
- **Status Tables**: Interactive data tables with styling

## 🤝 Contributing

### Development Workflow
1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/new-metric`
3. **Make changes**: Add new functionality
4. **Test locally**: Verify dashboard works
5. **Submit pull request**: Include description of changes

### Code Standards
- **Python**: PEP 8 compliance
- **Documentation**: Comprehensive docstrings
- **Error handling**: Graceful fallbacks
- **Testing**: Mock data validation

## 📞 Support

### Getting Help
- **Documentation**: This README and inline code comments
- **Logs**: Systemd service logs and application logs
- **Monitoring**: Dashboard health indicators
- **Community**: Team collaboration and knowledge sharing

### Contact Information
- **Team**: AI-Driven SOC Development Team
- **Repository**: Internal Git repository
- **Documentation**: Team knowledge base

## 📄 License

This dashboard is part of the **GATRA Security Platform** and is proprietary software developed for internal use.

---

**🧠 LangGraph ADA Agent Dashboard** - Empowering AI-powered security monitoring with real-time workflow orchestration insights.
