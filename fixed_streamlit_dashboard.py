import streamlit as st
import time
import random
from datetime import datetime

# Configure the page
st.set_page_config(
    page_title="GATRA AI Security Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid;
        margin-bottom: 1rem;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
    }
    
    .metric-label {
        color: #666;
        margin: 0;
    }
    
    .status-online {
        color: #10b981;
    }
    
    .status-offline {
        color: #ef4444;
    }
    
    .anomaly-item {
        padding: 1rem;
        margin: 0.5rem 0;
        background: #f8fafc;
        border-radius: 8px;
        border-left: 4px solid;
    }
    
    .anomaly-high {
        border-left-color: #ef4444;
    }
    
    .anomaly-medium {
        border-left-color: #f59e0b;
    }
    
    .anomaly-low {
        border-left-color: #10b981;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🛡️ GATRA AI Security Dashboard</h1>
    <p>AI-Driven SOC Command Center • Real-time Threat Intelligence</p>
    <p>🟢 System Online • Last Update: """ + datetime.now().strftime("%H:%M:%S") + """</p>
</div>
""", unsafe_allow_html=True)

# Key Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card" style="border-left-color: #10b981;">
        <p class="metric-value" style="color: #10b981;">1,247</p>
        <p class="metric-label">📊 Total Alerts</p>
        <p style="color: #10b981; font-size: 0.8rem;">↑ 12% from yesterday</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card" style="border-left-color: #3b82f6;">
        <p class="metric-value" style="color: #3b82f6;">89.2%</p>
        <p class="metric-label">🎯 ADA Accuracy</p>
        <p style="color: #3b82f6; font-size: 0.8rem;">↑ 3.2% improvement</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card" style="border-left-color: #8b5cf6;">
        <p class="metric-value" style="color: #8b5cf6;">94.1%</p>
        <p class="metric-label">🧠 TAA Confidence</p>
        <p style="color: #8b5cf6; font-size: 0.8rem;">↑ 1.8% this week</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card" style="border-left-color: #10b981;">
        <p class="metric-value" style="color: #10b981;">96.8%</p>
        <p class="metric-label">💚 System Health</p>
        <p style="color: #10b981; font-size: 0.8rem;">Optimal performance</p>
    </div>
    """, unsafe_allow_html=True)

# Main Content
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔍 ADA - Anomaly Detection")
    
    # Recent Anomalies
    anomalies = [
        {"severity": "High", "description": "Suspicious network traffic from 192.168.1.100", "time": "2 min ago"},
        {"severity": "Medium", "description": "Unusual login pattern detected", "time": "5 min ago"},
        {"severity": "Low", "description": "New device connected to network", "time": "12 min ago"},
        {"severity": "High", "description": "Potential data exfiltration attempt", "time": "18 min ago"},
        {"severity": "Medium", "description": "Multiple failed authentication attempts", "time": "25 min ago"}
    ]
    
    for anomaly in anomalies:
        severity_class = f"anomaly-{anomaly['severity'].lower()}"
        severity_icon = "🔴" if anomaly['severity'] == 'High' else "🟡" if anomaly['severity'] == 'Medium' else "🟢"
        
        st.markdown(f"""
        <div class="anomaly-item {severity_class}">
            <strong>{severity_icon} {anomaly['description']}</strong><br>
            <small style="color: #666;">{anomaly['time']}</small>
        </div>
        """, unsafe_allow_html=True)
    
    if st.button("🔍 FORCE RESCAN", type="primary"):
        st.success("Rescan initiated...")

with col2:
    st.markdown("### 🧠 TAA - Threat Analysis")
    
    # Analysis Results
    st.markdown("""
    <div style="background: #f8fafc; padding: 1.5rem; border-radius: 8px; margin: 1rem 0;">
        <p><strong>🎯 Threat Score:</strong> 8.5/10</p>
        <p><strong>🔗 Correlated Events:</strong> 12</p>
        <p><strong>⏱️ Analysis Time:</strong> 2.3s</p>
        <p><strong>✅ Confidence:</strong> 94%</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔍 ANALYZE NEW THREAT", type="secondary"):
        st.success("Starting threat analysis...")

# System Status
st.markdown("### 📊 System Status")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Component Status:**")
    
    components = [
        {"name": "ADA", "status": "Running", "port": 8081, "health": 95},
        {"name": "TAA", "status": "Running", "port": 8080, "health": 92},
        {"name": "CRA", "status": "Running", "port": 8082, "health": 88},
        {"name": "CLA", "status": "Running", "port": 8083, "health": 90},
        {"name": "BigQuery", "status": "Connected", "port": "N/A", "health": 98},
        {"name": "Orchestrator", "status": "Offline", "port": 8000, "health": 0}
    ]
    
    for component in components:
        status_icon = "✅" if component['health'] > 80 else "⚠️" if component['health'] > 50 else "❌"
        status_class = "status-online" if component['health'] > 80 else "status-offline"
        
        st.markdown(f"""
        <p class="{status_class}">
            {status_icon} {component['name']}: {component['status']} (Port {component['port']}) - {component['health']}%
        </p>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("**Performance:**")
    
    # Progress bar for system health
    system_health = 96.8
    st.progress(system_health / 100)
    st.markdown(f"**System Health: {system_health}%**")
    
    st.markdown("""
    <div style="margin-top: 1rem;">
        <p>📈 Uptime: 99.97% (30 days)</p>
        <p>⚡ Response Time: 1.2s avg</p>
        <p>🔄 Processing Rate: 847 alerts/min</p>
        <p>💾 Memory Usage: 68%</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔄 REFRESH STATUS"):
        st.success("Status refreshed!")

# Auto-refresh every 30 seconds
if st.button("🔄 Auto-refresh (30s)"):
    time.sleep(30)
    st.rerun()

# Footer
st.markdown("---")
st.markdown("**GATRA AI Security Dashboard** • Powered by AI-Driven SOC • Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


