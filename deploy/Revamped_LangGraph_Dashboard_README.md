# 🧠 Revamped LangGraph ADA Dashboard

## 📋 Overview

The **Revamped LangGraph ADA Dashboard** is a comprehensive, enterprise-grade monitoring solution designed specifically for monitoring your LangGraph ADA (Automated Decision Analysis) agent integration. This dashboard provides enhanced visibility, better theme support, and real-time BigQuery data integration with significantly improved display clarity.

## ✨ Key Features & Improvements

### 🎨 **Enhanced Visual Design**
- **High Contrast Display**: Significantly improved readability with enhanced contrast ratios
- **Professional Dark Theme**: Optimized for low-light environments and extended viewing sessions
- **Gradient Backgrounds**: Modern gradient designs for better visual hierarchy
- **Enhanced Typography**: Improved fonts, weights, and text shadows for better readability
- **Interactive Elements**: Hover effects, smooth transitions, and visual feedback

### 🔧 **Theme Support & Customization**
- **CSS Variables**: Centralized color management for easy theme switching
- **Dark Mode Optimized**: Primary theme optimized for security operations centers
- **Responsive Design**: Adapts to different screen sizes and resolutions
- **Custom Styling**: Enhanced styling for all Streamlit components

### 📊 **Real-Time Data Integration**
- **BigQuery Connection**: Direct integration with your `chronicle-dev-2be9.gatra_database`
- **Live Metrics**: Real-time monitoring of SIEM events, ADA alerts, and agent status
- **Fallback Support**: Graceful degradation to mock data if BigQuery is unavailable
- **Data Validation**: Proper column name handling for your existing schema

### 🚀 **Performance & Reliability**
- **Optimized Queries**: Efficient BigQuery queries with proper error handling
- **Fast Loading**: Streamlined data processing and visualization
- **Error Recovery**: Automatic fallback mechanisms for system resilience
- **Resource Efficient**: Minimal resource consumption for long-running operations

## 🏗️ Architecture & Technology Stack

### **Frontend Framework**
- **Streamlit**: Modern Python web framework for data applications
- **Custom CSS**: Advanced styling with CSS variables and modern design patterns
- **Plotly**: Interactive charts and visualizations with dark theme support

### **Backend Integration**
- **Google BigQuery**: Enterprise data warehouse for real-time metrics
- **Python 3.8+**: Core runtime environment
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing and data generation

### **Deployment & Infrastructure**
- **Google Cloud VM**: Hosted on `xdgaisocapp01` in `asia-southeast2-a`
- **Systemd Service**: Automatic startup and process management
- **Port Management**: Dedicated port allocation (8509) for isolation
- **IAP Tunneling**: Secure access without public IP exposure

## 📁 File Structure

```
langgraph-ada-dashboard/
├── revamped_langgraph_dashboard.py    # Main dashboard application
├── langgraph_dashboard_requirements.txt # Python dependencies
├── deploy_langgraph_dashboard.sh      # Deployment automation script
└── README files                       # Documentation
```

## 🚀 Quick Start Guide

### **1. Access the Dashboard**
```
URL: http://10.45.254.19:8509
```

### **2. Dashboard Controls (Sidebar)**
- **📊 Use Real BigQuery Data**: Toggle between real and mock data
- **🎨 Theme Status**: Current theme information and optimization details

### **3. Main Dashboard Sections**
- **📊 Real-Time Metrics Overview**: Key performance indicators
- **📈 Performance Analytics**: Response time and success rate gauges
- **🔄 Activity Timeline**: 48-hour activity monitoring
- **📋 Sample SIEM Data**: Real data structure analysis
- **🏥 System Health**: Integration status and health monitoring

## 📊 Dashboard Components

### **1. Real-Time Metrics Overview**
Four enhanced metric cards displaying:
- **📊 SIEM Events**: Total security events processed
- **🚨 ADA Alerts**: Anomaly detection alerts generated
- **🤖 Active Agents**: Number of running AI agents
- **🎯 Success Rate**: Overall system performance percentage

### **2. Performance Analytics**
Two interactive gauge charts:
- **⚡ Response Time Performance**: Average response time with color-coded zones
- **🎯 Success Rate Analysis**: Success rate with performance thresholds

### **3. Activity Timeline**
Interactive line chart showing:
- **SIEM Events**: Blue line with circular markers
- **ADA Alerts**: Red line with circular markers
- **48-hour window**: Hourly data points for trend analysis

### **4. Sample SIEM Data Analysis**
Real-time data display showing:
- **Alarm ID**: Security alert identifiers
- **Events Count**: Number of events per alarm
- **Data Type**: Structure information for analysis

### **5. System Health & Integration Status**
Three status cards:
- **🔗 BigQuery Connection**: Database connectivity status
- **🧠 LangGraph Status**: Workflow orchestration status
- **🚨 ADA Agent**: Agent processing and performance status

## 🎨 Theme System

### **Dark Theme (Default)**
- **Background**: Deep blue gradient (`#0f172a` to `#1e293b`)
- **Surface**: Dark slate (`#1e293b`) with transparency
- **Text**: Light gray (`#f8fafc`) with enhanced contrast
- **Accents**: Purple (`#6366f1`) and blue (`#06b6d4`) highlights

### **Color Palette**
- **Primary**: `#6366f1` (Indigo)
- **Secondary**: `#8b5cf6` (Purple)
- **Success**: `#10b981` (Emerald)
- **Warning**: `#f59e0b` (Amber)
- **Error**: `#ef4444` (Red)
- **Info**: `#06b6d4` (Cyan)

### **Enhanced Components**
- **Metric Cards**: Glassmorphism effect with backdrop blur
- **Section Headers**: Gradient backgrounds with shadows
- **Charts**: Dark theme optimized with proper contrast
- **Tables**: Enhanced readability with custom styling

## 🔌 Data Integration

### **BigQuery Tables**
- **`siem_events`**: Security information and event management data
- **`processed_ada_alerts`**: Anomaly detection results
- **`agent_state`**: AI agent status and performance metrics

### **Data Flow**
1. **Connection**: Establishes BigQuery client connection
2. **Query Execution**: Runs optimized COUNT queries
3. **Data Processing**: Processes results for dashboard display
4. **Fallback**: Uses mock data if queries fail
5. **Visualization**: Renders data in interactive charts

### **Error Handling**
- **Connection Errors**: Graceful fallback to mock data
- **Query Errors**: Detailed error messages with troubleshooting
- **Data Validation**: Proper column name handling
- **Performance Monitoring**: Query execution time tracking

## 🚀 Deployment & Operations

### **Current Deployment**
- **VM**: `xdgaisocapp01` in `asia-southeast2-a`
- **Port**: 8509
- **User**: `app`
- **Service**: Running via Streamlit

### **Service Management**
```bash
# Check dashboard status
ps aux | grep revamped_langgraph_dashboard

# Check port status
sudo netstat -tulnp | grep 8509

# Restart dashboard
pkill -f revamped_langgraph_dashboard
cd /home/app/langgraph-ada-dashboard
/home/app/.local/bin/streamlit run revamped_langgraph_dashboard.py --server.port 8509 --server.address 0.0.0.0 --server.headless true &
```

### **Monitoring & Maintenance**
- **Process Monitoring**: Check for dashboard process health
- **Port Management**: Ensure port 8509 is available
- **Log Analysis**: Monitor Streamlit logs for errors
- **Performance Tracking**: Monitor response times and data freshness

## 🔧 Configuration & Customization

### **Environment Variables**
- **BigQuery Project**: `chronicle-dev-2be9`
- **Dataset**: `gatra_database`
- **Port**: 8509
- **Host**: 0.0.0.0

### **Customization Options**
- **Color Schemes**: Modify CSS variables for different themes
- **Layout**: Adjust column widths and component positioning
- **Data Sources**: Add new BigQuery tables or data sources
- **Charts**: Customize chart types and visualizations

### **CSS Customization**
The dashboard uses advanced CSS with:
- **CSS Variables**: Easy theme switching
- **Flexbox Layout**: Responsive design
- **CSS Grid**: Advanced layout management
- **Custom Animations**: Smooth transitions and effects

## 📈 Performance Metrics

### **Dashboard Performance**
- **Load Time**: Optimized for fast initial rendering
- **Data Refresh**: Real-time updates with minimal latency
- **Memory Usage**: Efficient resource utilization
- **CPU Usage**: Minimal processing overhead

### **BigQuery Performance**
- **Query Optimization**: Efficient COUNT queries
- **Connection Pooling**: Reusable client connections
- **Error Handling**: Fast fallback mechanisms
- **Data Caching**: Intelligent data refresh strategies

## 🛡️ Security & Access Control

### **Access Control**
- **VM Access**: Google Cloud IAP tunneling
- **Port Isolation**: Dedicated port for dashboard
- **User Permissions**: Limited to `app` user
- **Network Security**: Internal network access only

### **Data Security**
- **BigQuery Access**: Service account authentication
- **Data Encryption**: In-transit and at-rest encryption
- **Audit Logging**: Access and query logging
- **Privacy Compliance**: No sensitive data exposure

## 🔍 Troubleshooting Guide

### **Common Issues**

#### **Dashboard Not Loading**
```bash
# Check if process is running
ps aux | grep revamped_langgraph_dashboard

# Check port availability
sudo netstat -tulnp | grep 8509

# Restart dashboard
pkill -f revamped_langgraph_dashboard
# Then restart using deployment command
```

#### **BigQuery Connection Errors**
- Verify service account credentials
- Check network connectivity
- Validate table names and permissions
- Review error logs for specific issues

#### **Display Issues**
- Clear browser cache
- Check CSS compatibility
- Verify browser supports modern CSS features
- Test in different browsers

### **Performance Issues**
- Monitor VM resource usage
- Check BigQuery query performance
- Review data refresh frequency
- Optimize chart rendering

## 📚 API Reference

### **Main Functions**
- **`get_real_ada_data()`**: Fetches real-time BigQuery data
- **`get_mock_data()`**: Provides fallback mock data
- **`create_charts()`**: Generates interactive visualizations

### **Data Structures**
```python
{
    'siem_events': int,      # Total SIEM events count
    'ada_alerts': int,       # Total ADA alerts count
    'active_agents': int,    # Active agent count
    'success_rate': float,   # Success rate percentage
    'avg_response_time': float,  # Average response time
    'anomalies_detected': int,   # Detected anomalies count
    'sample_data': list      # Sample data for analysis
}
```

## 🚀 Future Enhancements

### **Planned Features**
- **Light Mode Toggle**: Automatic theme switching
- **Real-time Updates**: WebSocket-based live updates
- **Advanced Analytics**: Machine learning insights
- **Mobile Optimization**: Enhanced mobile experience
- **Export Functionality**: Data export capabilities

### **Integration Opportunities**
- **Slack Notifications**: Alert integration
- **Email Reports**: Automated reporting
- **API Endpoints**: REST API for external access
- **Webhook Support**: Event-driven updates

## 📞 Support & Maintenance

### **Technical Support**
- **Dashboard Issues**: Check process status and logs
- **Data Issues**: Verify BigQuery connectivity and permissions
- **Performance Issues**: Monitor resource usage and optimize queries
- **Display Issues**: Review CSS and browser compatibility

### **Maintenance Schedule**
- **Daily**: Check dashboard availability and data freshness
- **Weekly**: Review performance metrics and error logs
- **Monthly**: Update dependencies and security patches
- **Quarterly**: Performance optimization and feature updates

## 📄 License & Attribution

This dashboard is part of your AI-driven SOC (Security Operations Center) infrastructure. It integrates with your existing LangGraph ADA agent workflow and BigQuery data warehouse.

---

**Last Updated**: August 23, 2025  
**Version**: 2.0 (Revamped)  
**Status**: Production Ready  
**Access URL**: `http://10.45.254.19:8509`
