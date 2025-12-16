import streamlit as st
import plotly.express as px
import pandas as pd
from streamlit_echarts import st_echarts
import time
from datetime import datetime, timedelta
import random

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
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-left: 20px;
        padding-right: 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1e3a8a;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ===========================================
# SECTION 1: INCIDENT TIMELINE VISUALIZATION
# ===========================================
def render_incident_timeline(data: pd.DataFrame = None):
    st.markdown("## 🕒 Agent Response Timeline")

    # 🧠 Example dataset (replace with live BigQuery data later)
    if data is None:
        data = pd.DataFrame([
            dict(Task="ADA - Anomaly Detected", Start='2025-10-16 02:05:00', Finish='2025-10-16 02:07:00', Resource='ADA'),
            dict(Task="TAA - Threat Correlation", Start='2025-10-16 02:07:00', Finish='2025-10-16 02:10:00', Resource='TAA'),
            dict(Task="CRA - Containment", Start='2025-10-16 02:10:00', Finish='2025-10-16 02:15:00', Resource='CRA'),
            dict(Task="CLA - Compliance Review", Start='2025-10-16 02:15:00', Finish='2025-10-16 02:20:00', Resource='CLA'),
        ])

    fig = px.timeline(
        data, 
        x_start="Start", 
        x_end="Finish", 
        y="Task", 
        color="Resource",
        color_discrete_sequence=["#4CAF50", "#03A9F4", "#FFC107", "#E91E63"],
        title="Incident Response Flow (ADA ➜ TAA ➜ CRA ➜ CLA)"
    )

    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        template="plotly_white",
        height=380,
        xaxis_title="Time (UTC)",
        yaxis_title="Agent Workflow",
        font=dict(size=12),
        showlegend=True,
        legend_title_text="Agent"
    )

    st.plotly_chart(fig, use_container_width=True)

# ===========================================
# SECTION 2: MITRE ATT&CK COVERAGE HEATMAP
# ===========================================
def render_mitre_heatmap(heatmap_data=None):
    st.markdown("## 🧩 MITRE ATT&CK Coverage Heatmap")

    # 🎯 Example dataset (tactic × agent × confidence)
    if heatmap_data is None:
        # format: [x_index, y_index, confidence]
        heatmap_data = [
            [0, 0, 0.95], [1, 0, 0.80], [2, 0, 0.65], [3, 0, 0.88], [4, 0, 0.72],
            [0, 1, 0.60], [1, 1, 0.75], [2, 1, 0.88], [3, 1, 0.65], [4, 1, 0.78],
            [0, 2, 0.70], [1, 2, 0.50], [2, 2, 0.40], [3, 2, 0.85], [4, 2, 0.90],
            [0, 3, 0.85], [1, 3, 0.90], [2, 3, 0.75], [3, 3, 0.68], [4, 3, 0.82]
        ]

    options = {
        "title": {"text": "MITRE ATT&CK Coverage by Agent", "left": "center"},
        "tooltip": {"position": "top"},
        "xAxis": {
            "type": "category",
            "data": ["Initial Access", "Execution", "Persistence", "Privilege Escalation", "Defense Evasion"],
            "splitArea": {"show": True},
        },
        "yAxis": {
            "type": "category",
            "data": ["ADA", "TAA", "CRA", "CLA"],
            "splitArea": {"show": True},
        },
        "visualMap": {
            "min": 0,
            "max": 1,
            "calculable": True,
            "orient": "horizontal",
            "left": "center",
            "bottom": "10%",
            "inRange": {"color": ["#d6e685", "#8cc665", "#44a340", "#1e6823"]},
        },
        "series": [{
            "name": "Detection Confidence",
            "type": "heatmap",
            "data": heatmap_data,
            "label": {"show": True, "formatter": "{@[2]:.0%}"},
            "emphasis": {"itemStyle": {"shadowBlur": 10, "shadowColor": "rgba(0, 0, 0, 0.5)"}},
        }]
    }

    st_echarts(options=options, height="420px")

# ===========================================
# SECTION 3: MAIN DASHBOARD FUNCTIONS
# ===========================================

def render_overview_tab():
    """Render the main overview tab"""
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

def render_visualization_tab():
    """Render the MITRE ATT&CK and Timeline visualization tab"""
    st.markdown("""
    <div class="main-header">
        <h1>🔐 GATRA AI Visualization Module</h1>
        <p>MITRE ATT&CK Coverage & Agent Timeline Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    render_incident_timeline()
    st.markdown("---")
    render_mitre_heatmap()

# ===========================================
# MAIN EXECUTION
# ===========================================

# Create tabs
tab1, tab2 = st.tabs(["🏠 Overview", "📊 Visualizations"])

with tab1:
    render_overview_tab()

with tab2:
    render_visualization_tab()

# Auto-refresh every 30 seconds
if st.button("🔄 Auto-refresh (30s)"):
    time.sleep(30)
    st.rerun()

# Footer
st.markdown("---")
st.markdown("**GATRA AI Security Dashboard** • Powered by AI-Driven SOC • Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


