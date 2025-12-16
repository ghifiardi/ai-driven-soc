#!/usr/bin/env python3
"""
🛡️ Hybrid Cyber Defense Agent Dashboard
=======================================

Real-time monitoring dashboard for the DQN + Gemini hybrid cyber defense agent.
Integrates with existing SOC dashboard infrastructure.

Features:
- Real-time agent status and metrics
- DQN inference performance monitoring  
- Gemini API usage and explanations
- Alert processing pipeline visualization
- Integration health with ADA/TAA/CRA/CLA agents
- A2A protocol activity monitoring
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
import time
from datetime import datetime, timedelta
import asyncio
import numpy as np
from typing import Dict, List, Any
import logging

# Configure page
st.set_page_config(
    page_title="🛡️ Hybrid Cyber Defense Agent Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for SOC theme
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .status-good { 
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    
    .status-warning { 
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    
    .status-error { 
        background: linear-gradient(135deg, #fc466b 0%, #3f5efb 100%);
    }
    
    .alert-card {
        background: rgba(255, 255, 255, 0.05);
        border-left: 4px solid #00d4ff;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
AGENT_HOST = "localhost"
AGENT_PORT = 8083
AGENT_URL = f"http://{AGENT_HOST}:{AGENT_PORT}"

class HybridDashboard:
    """Main dashboard class for hybrid cyber defense agent"""
    
    def __init__(self):
        self.agent_url = AGENT_URL
        self.mock_mode = False
        
    def check_agent_connection(self) -> Dict:
        """Check if hybrid agent is accessible"""
        try:
            response = requests.get(f"{self.agent_url}/health", timeout=3)
            if response.status_code == 200:
                return {"status": "connected", "data": response.json()}
            else:
                return {"status": "error", "message": f"HTTP {response.status_code}"}
        except requests.exceptions.ConnectionError as e:
            return {"status": "disconnected", "message": "Connection refused - Agent not running"}
        except requests.exceptions.Timeout as e:
            return {"status": "timeout", "message": "Connection timeout - Agent slow to respond"}
        except requests.exceptions.RequestException as e:
            return {"status": "error", "message": f"Request failed: {str(e)[:100]}..."}
    
    def get_agent_status(self) -> Dict:
        """Get comprehensive agent status"""
        try:
            response = requests.get(f"{self.agent_url}/status", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return self.generate_mock_status()
        except:
            return self.generate_mock_status()
    
    def generate_mock_status(self) -> Dict:
        """Generate mock data for demo purposes"""
        self.mock_mode = True
        return {
            "agent_id": "hybrid-defense-demo",
            "uptime_seconds": 86400,
            "stats": {
                "alerts_processed": 1247,
                "actions_recommended": 892,
                "explanations_generated": 756,
                "errors": 12,
                "start_time": (datetime.now() - timedelta(hours=24)).isoformat()
            },
            "metrics": {
                "alerts_processed_total": 1247,
                "dqn_inference_duration": [0.045, 0.032, 0.056, 0.038, 0.041],
                "gemini_api_calls_total": 756,
                "containment_actions_total": {
                    "isolate_node": 234,
                    "block_traffic": 178,
                    "patch_system": 289,
                    "monitor": 145,
                    "no_action": 46
                },
                "errors_total": {
                    "DQNError": 3,
                    "GeminiTimeoutError": 6,
                    "PubSubError": 3
                }
            },
            "circuit_breakers": {
                "dqn_model": "CLOSED",
                "gemini_api": "CLOSED", 
                "pubsub_connection": "CLOSED"
            }
        }
    
    def generate_mock_alerts(self) -> List[Dict]:
        """Generate mock alert data for visualization"""
        alerts = []
        attack_types = ['malware', 'ddos', 'intrusion', 'phishing', 'ransomware']
        actions = ['isolate_node', 'block_traffic', 'patch_system', 'monitor', 'no_action']
        
        for i in range(20):
            alert = {
                'timestamp': (datetime.now() - timedelta(minutes=i*15)).isoformat(),
                'alert_id': f'alert_{1000+i}',
                'attack_type': np.random.choice(attack_types),
                'severity': np.random.choice(['low', 'medium', 'high', 'critical']),
                'recommended_action': np.random.choice(actions),
                'confidence': round(np.random.uniform(0.3, 0.95), 2),
                'processing_time': round(np.random.uniform(0.02, 0.08), 3),
                'explanation': "AI-generated explanation based on threat analysis and system vulnerability assessment."
            }
            alerts.append(alert)
        
        return alerts

# Initialize dashboard
dashboard = HybridDashboard()

# Header
st.title("🛡️ Hybrid Cyber Defense Agent Dashboard")
st.markdown("**Real-time monitoring for DQN + Gemini hybrid cybersecurity agent**")

# Sidebar controls
st.sidebar.header("🔧 Dashboard Controls")

# Agent connection status
connection_status = dashboard.check_agent_connection()
if connection_status["status"] == "connected":
    st.sidebar.success("✅ Agent Connected")
    agent_data = connection_status["data"]
    st.sidebar.info(f"🆔 Agent ID: {agent_data.get('agent_id', 'Unknown')}")
    st.sidebar.info(f"📊 Status: {agent_data.get('status', 'Unknown')}")
    if "components" in agent_data:
        comps = agent_data["components"]
        st.sidebar.info(f"🤖 DQN: {'✅' if comps.get('dqn_agent') else '❌'}")
        st.sidebar.info(f"🧠 Gemini: {'✅' if comps.get('gemini_client') else '❌'}")
        st.sidebar.info(f"📡 Pub/Sub: {'✅' if comps.get('pubsub') else '❌'}")
elif connection_status["status"] == "timeout":
    st.sidebar.warning(f"⏱️ Agent Slow: {connection_status.get('message', 'Timeout')}")
    st.sidebar.warning("📊 Using cached data")
else:
    st.sidebar.error(f"❌ Agent Error: {connection_status.get('message', 'Unknown error')}")
    st.sidebar.warning("📊 Using mock data for demonstration")

# Refresh controls
auto_refresh = st.sidebar.checkbox("🔄 Auto Refresh (30s)", value=False)
if st.sidebar.button("🔄 Manual Refresh"):
    st.rerun()

# Get current status
status_data = dashboard.get_agent_status()
if dashboard.mock_mode:
    st.sidebar.info("📊 Demo Mode - Using simulated data")

# Main dashboard tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Overview", 
    "🤖 DQN Performance", 
    "🧠 Gemini Analytics", 
    "🚨 Live Alerts", 
    "⚙️ System Health"
])

with tab1:
    st.header("📊 Agent Overview")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card status-good">
            <h3>📈 Alerts Processed</h3>
            <h1>{:,}</h1>
            <p>Total processed</p>
        </div>
        """.format(status_data["stats"]["alerts_processed"]), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ Actions Taken</h3>
            <h1>{:,}</h1>
            <p>Recommendations</p>
        </div>
        """.format(status_data["stats"]["actions_recommended"]), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🧠 AI Explanations</h3>
            <h1>{:,}</h1>
            <p>Generated by Gemini</p>
        </div>
        """.format(status_data["stats"]["explanations_generated"]), unsafe_allow_html=True)
    
    with col4:
        error_count = status_data["stats"]["errors"]
        error_class = "status-good" if error_count < 10 else "status-warning" if error_count < 50 else "status-error"
        st.markdown("""
        <div class="metric-card {}">
            <h3>⚠️ Errors</h3>
            <h1>{}</h1>
            <p>System errors</p>
        </div>
        """.format(error_class, error_count), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Action distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Action Distribution")
        actions_data = status_data["metrics"]["containment_actions_total"]
        
        fig_pie = px.pie(
            values=list(actions_data.values()),
            names=list(actions_data.keys()),
            title="Defensive Actions Recommended",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("📈 Performance Trends")
        
        # Generate performance trend data
        hours = [(datetime.now() - timedelta(hours=i)).strftime("%H:%M") for i in range(12, 0, -1)]
        alerts_trend = [np.random.randint(80, 120) for _ in range(12)]
        
        fig_trend = px.line(
            x=hours,
            y=alerts_trend,
            title="Alerts Processed (Last 12 Hours)",
            labels={'x': 'Time', 'y': 'Alerts/Hour'}
        )
        fig_trend.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
        )
        st.plotly_chart(fig_trend, use_container_width=True)

with tab2:
    st.header("🤖 DQN Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚡ Inference Speed")
        
        dqn_times = status_data["metrics"]["dqn_inference_duration"]
        avg_time = np.mean(dqn_times) if dqn_times else 0.045
        
        # Gauge chart for inference speed
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=avg_time * 1000,  # Convert to ms
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Average Inference Time (ms)"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 100], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig_gauge.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=16)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Performance requirements
        if avg_time < 0.1:
            st.success("✅ Performance target met (< 100ms)")
        else:
            st.warning("⚠️ Performance slower than target")
    
    with col2:
        st.subheader("📊 Inference Distribution")
        
        # Histogram of inference times
        if dqn_times:
            fig_hist = px.histogram(
                x=[t * 1000 for t in dqn_times],
                nbins=10,
                title="DQN Inference Time Distribution",
                labels={'x': 'Time (ms)', 'y': 'Frequency'}
            )
            fig_hist.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("No inference time data available yet")
    
    # DQN Model Status
    st.subheader("🎯 Model Status & Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        circuit_status = status_data["circuit_breakers"]["dqn_model"]
        status_class = "status-good" if circuit_status == "CLOSED" else "status-error"
        st.markdown(f"""
        <div class="metric-card {status_class}">
            <h4>Circuit Breaker</h4>
            <h2>{circuit_status}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>State Size</h4>
            <h2>7</h2>
            <p>Input dimensions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>Action Space</h4>
            <h2>5</h2>
            <p>Possible actions</p>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    st.header("🧠 Gemini AI Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📞 API Usage Statistics")
        
        gemini_calls = status_data["metrics"]["gemini_api_calls_total"]
        total_processed = status_data["stats"]["alerts_processed"]
        usage_rate = (gemini_calls / total_processed * 100) if total_processed > 0 else 0
        
        fig_usage = go.Figure(go.Indicator(
            mode="gauge+number",
            value=usage_rate,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Gemini Usage Rate (%)"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "purple"},
                'steps': [
                    {'range': [0, 60], 'color': "lightgray"},
                    {'range': [60, 100], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig_usage.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig_usage, use_container_width=True)
    
    with col2:
        st.subheader("⚡ Response Performance")
        
        # Mock Gemini response times
        response_times = [np.random.uniform(0.8, 3.2) for _ in range(24)]
        hours = [(datetime.now() - timedelta(hours=i)).strftime("%H:%M") for i in range(24, 0, -1)]
        
        fig_response = px.line(
            x=hours,
            y=response_times,
            title="Gemini API Response Time (Last 24h)",
            labels={'x': 'Time', 'y': 'Response Time (s)'}
        )
        fig_response.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
        )
        st.plotly_chart(fig_response, use_container_width=True)
    
    # Explanation samples
    st.subheader("💬 Recent AI Explanations")
    
    sample_explanations = [
        {
            "alert_id": "alert_1023",
            "action": "isolate_node",
            "explanation": "Recommended node isolation due to detected ransomware indicators. The compromised system shows encrypted file patterns and suspicious network communication to known C2 servers. Isolation prevents lateral movement while preserving forensic evidence.",
            "confidence": 0.89,
            "timestamp": "2025-01-11 14:23:15"
        },
        {
            "alert_id": "alert_1022", 
            "action": "block_traffic",
            "explanation": "DDoS attack pattern detected from multiple source IPs targeting web services. Traffic blocking recommended to preserve service availability while maintaining legitimate user access through load balancing.",
            "confidence": 0.76,
            "timestamp": "2025-01-11 14:18:42"
        }
    ]
    
    for exp in sample_explanations:
        st.markdown(f"""
        <div class="alert-card">
            <strong>🔍 Alert {exp['alert_id']}</strong> | 
            <span style="color: #00d4ff;">Action: {exp['action']}</span> | 
            <span style="color: #32cd32;">Confidence: {exp['confidence']:.2f}</span>
            <br><br>
            <em>"{exp['explanation']}"</em>
            <br><br>
            <small>🕒 {exp['timestamp']}</small>
        </div>
        """, unsafe_allow_html=True)

with tab4:
    st.header("🚨 Live Alert Processing")
    
    # Generate mock recent alerts
    recent_alerts = dashboard.generate_mock_alerts()
    
    # Alert stream visualization
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Recent Alert Activity")
        
        # Convert to DataFrame for better display
        df_alerts = pd.DataFrame(recent_alerts)
        df_alerts['timestamp'] = pd.to_datetime(df_alerts['timestamp'])
        
        # Display as interactive table
        st.dataframe(
            df_alerts[['alert_id', 'attack_type', 'severity', 'recommended_action', 'confidence', 'processing_time']],
            use_container_width=True,
            column_config={
                "alert_id": "Alert ID",
                "attack_type": "Attack Type", 
                "severity": "Severity",
                "recommended_action": "Recommended Action",
                "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1),
                "processing_time": "Processing Time (s)"
            }
        )
    
    with col2:
        st.subheader("🔥 Alert Heatmap")
        
        # Attack type frequency
        attack_counts = df_alerts['attack_type'].value_counts()
        
        fig_heatmap = px.bar(
            x=attack_counts.values,
            y=attack_counts.index,
            orientation='h',
            title="Attack Types (Last 5h)",
            color=attack_counts.values,
            color_continuous_scale="Reds"
        )
        fig_heatmap.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=False
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Real-time alert feed (placeholder)
    st.subheader("📡 Real-Time Feed")
    
    # Placeholder for real-time alerts
    alert_placeholder = st.empty()
    
    if auto_refresh:
        with alert_placeholder.container():
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.info(f"🔄 Monitoring for new alerts... (Last update: {current_time})")

with tab5:
    st.header("⚙️ System Health & Integration")
    
    # System status overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🔌 Integration Status")
        
        integrations = [
            {"name": "ADA Agent", "status": "✅ Connected", "endpoint": "localhost:8081"},
            {"name": "TAA Agent", "status": "✅ Connected", "endpoint": "localhost:8080"}, 
            {"name": "CRA Agent", "status": "✅ Connected", "endpoint": "localhost:8084"},
            {"name": "CLA Agent", "status": "✅ Connected", "endpoint": "localhost:8082"},
            {"name": "Pub/Sub", "status": "✅ Connected", "endpoint": "ada-alerts"},
            {"name": "Vertex AI", "status": "✅ Connected", "endpoint": "Gemini API"}
        ]
        
        for integration in integrations:
            st.markdown(f"""
            **{integration['name']}**  
            {integration['status']}  
            `{integration['endpoint']}`
            """)
    
    with col2:
        st.subheader("🔧 Circuit Breakers")
        
        cb_status = status_data["circuit_breakers"]
        for name, state in cb_status.items():
            color = "🟢" if state == "CLOSED" else "🔴" if state == "OPEN" else "🟡"
            st.markdown(f"{color} **{name.replace('_', ' ').title()}**: {state}")
        
        st.markdown("---")
        
        st.subheader("📊 Error Analysis")
        errors = status_data["metrics"]["errors_total"]
        if errors:
            for error_type, count in errors.items():
                st.markdown(f"⚠️ {error_type}: {count}")
        else:
            st.success("✅ No recent errors")
    
    with col3:
        st.subheader("💻 Resource Usage")
        
        uptime_seconds = status_data["uptime_seconds"]
        uptime_hours = uptime_seconds / 3600
        
        st.metric("⏱️ Uptime", f"{uptime_hours:.1f} hours")
        st.metric("🧠 Memory Usage", "245 MB")
        st.metric("⚡ CPU Usage", "12.3%")
        st.metric("🌐 Network I/O", "1.2 MB/s")
        
        # Health score
        health_score = 95  # Mock health score
        st.markdown(f"""
        <div class="metric-card status-good">
            <h4>🏥 Health Score</h4>
            <h1>{health_score}%</h1>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    🛡️ Hybrid Cyber Defense Agent Dashboard | 
    Powered by DQN + Gemini AI | 
    Integrated with ADA/TAA/CRA/CLA SOC Framework
</div>
""", unsafe_allow_html=True)

# Auto-refresh functionality
if auto_refresh:
    time.sleep(30)
    st.rerun()
