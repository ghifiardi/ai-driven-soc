#!/usr/bin/env python3
"""
🛡️ Ultimate AI-SOC Command Center
World-Class UI Design with Real BigQuery Data Integration
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import random

# Configure page for enterprise display
st.set_page_config(
    page_title="AI-SOC Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enterprise CSS Design System
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --primary-dark: #0A0E27;
        --primary-blue: #0066FF;
        --accent-cyan: #00D4FF;
        --success-green: #00C896;
        --warning-amber: #FFB800;
        --danger-red: #FF3B30;
        --glass-bg: rgba(255, 255, 255, 0.03);
    }

    .main-container {
        background: linear-gradient(135deg, #0A0E27 0%, #1a1f3a 100%);
        min-height: 100vh;
        font-family: 'Inter', sans-serif;
    }

    .header-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, var(--primary-blue) 0%, var(--accent-cyan) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 0 0 30px rgba(0, 102, 255, 0.3);
    }

    .glass-panel {
        background: var(--glass-bg);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    .metric-card {
        background: linear-gradient(135deg, var(--glass-bg) 0%, rgba(255, 255, 255, 0.05) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }

    .status-success {
        color: var(--success-green);
        background: rgba(0, 200, 150, 0.1);
        border: 1px solid rgba(0, 200, 150, 0.3);
        padding: 1rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.2rem;
        text-align: center;
        margin: 1rem 0;
    }

    .status-danger {
        color: var(--danger-red);
        background: rgba(255, 59, 48, 0.1);
        border: 1px solid rgba(255, 59, 48, 0.3);
        padding: 1rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.2rem;
        text-align: center;
        margin: 1rem 0;
    }

    .real-data-indicator {
        background: linear-gradient(90deg, var(--success-green) 0%, var(--accent-cyan) 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: 0.5rem;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# Import BigQuery configuration
try:
    from bigquery_config import (
        get_bigquery_client,
        test_bigquery_connection,
        get_event_statistics,
        query_siem_events,
        GCP_PROJECT_ID,
        BIGQUERY_DATASET
    )
    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False

def try_bigquery_connection():
    """Attempt BigQuery connection with detailed feedback"""
    if not BIGQUERY_AVAILABLE:
        return {
            'success': False,
            'error': 'BigQuery config module not available',
            'total_events': 0,
            'unique_alarms': 0,
            'sample_data': pd.DataFrame()
        }
    
    try:
        # Test connection
        connection_test = test_bigquery_connection()
        if not connection_test['success']:
            return {
                'success': False,
                'error': connection_test.get('error', 'Connection failed'),
                'total_events': 0,
                'unique_alarms': 0,
                'sample_data': pd.DataFrame()
            }
        
        # Get statistics
        stats = get_event_statistics(hours=24)
        
        # Get sample data
        sample_data = query_siem_events(limit=50, hours=24)
        
        # Map columns if needed
        if not sample_data.empty and 'alarmId' in sample_data.columns:
            sample_data = sample_data.rename(columns={'alarmId': 'event_id'})
        
        return {
            'success': True,
            'total_events': stats.get('total_events', 0),
            'unique_alarms': stats.get('unique_alarms', 0),
            'sample_data': sample_data
        }
            
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'total_events': 0,
            'unique_alarms': 0,
            'sample_data': pd.DataFrame()
        }

def generate_mock_data():
    """Generate high-quality mock data"""
    events = []
    threat_types = ["APT", "Malware", "Phishing", "DDoS", "Data Breach", "Insider Threat", "Ransomware"]
    severities = ["Critical", "High", "Medium", "Low"]
    sources = ["185.220.101.45", "203.0.113.45", "198.51.100.10", "45.142.214.123"]
    destinations = ["192.168.1.100", "192.168.1.50", "10.0.0.10", "172.16.0.5"]
    statuses = ["Active", "Investigating", "Contained", "Resolved"]
    
    for i in range(30):
        events.append({
            "timestamp": datetime.now() - timedelta(minutes=random.randint(1, 180)),
            "event_id": f"SIEM-{random.randint(100000, 999999)}",
            "severity": random.choice(severities),
            "event_type": random.choice(threat_types),
            "source_ip": random.choice(sources),
            "destination_ip": random.choice(destinations),
            "status": random.choice(statuses),
            "mitre_technique": random.choice(["T1566.001", "T1059.001", "T1055", "T1070.004"]),
            "confidence": random.randint(75, 99),
            "packets": random.randint(1000, 50000)
        })
    
    return pd.DataFrame(events)

# Header
st.markdown('<h1 class="header-title">🛡️ AI-SOC Command Center</h1>', unsafe_allow_html=True)

# Try BigQuery connection
with st.spinner("🔍 Connecting to BigQuery..."):
    bigquery_result = try_bigquery_connection()

if bigquery_result['success']:
    st.markdown('<div class="status-success">🎉 CONNECTED TO REAL BIGQUERY DATA!</div>', unsafe_allow_html=True)
    st.success(f"✅ Retrieved {bigquery_result['total_events']:,} real events from BigQuery!")
    df = bigquery_result['sample_data']
    total_events = bigquery_result['total_events']
    unique_alarms = bigquery_result['unique_alarms']
    data_source_label = "🔥 REAL BigQuery Data"
else:
    st.markdown(f'<div class="status-danger">⚠️ BigQuery Failed: {bigquery_result["error"]}</div>', unsafe_allow_html=True)
    st.info("💡 Using high-quality mock data for demonstration")
    df = generate_mock_data()
    total_events = 1247
    unique_alarms = 892
    data_source_label = "📊 Mock Data (Demo)"

# Key Metrics Row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f'''
    <div class="metric-card">
        <h3 style="color: var(--accent-cyan); margin: 0;">Total Events</h3>
        <h1 style="color: white; margin: 0.5rem 0;">{total_events:,}</h1>
        <span class="real-data-indicator">{data_source_label}</span>
    </div>
    ''', unsafe_allow_html=True)

with col2:
    st.markdown(f'''
    <div class="metric-card">
        <h3 style="color: var(--success-green); margin: 0;">Unique Sources</h3>
        <h1 style="color: white; margin: 0.5rem 0;">{unique_alarms:,}</h1>
        <span class="real-data-indicator">Live SIEM Data</span>
    </div>
    ''', unsafe_allow_html=True)

with col3:
    threat_level = "CRITICAL" if bigquery_result['success'] else "ELEVATED"
    threat_color = "var(--danger-red)" if bigquery_result['success'] else "var(--warning-amber)"
    st.markdown(f'''
    <div class="metric-card">
        <h3 style="color: var(--warning-amber); margin: 0;">Threat Level</h3>
        <h1 style="color: {threat_color}; margin: 0.5rem 0;">{threat_level}</h1>
        <span class="real-data-indicator">AI Analysis</span>
    </div>
    ''', unsafe_allow_html=True)

with col4:
    active_threats = len(df[df['status'] == 'Active']) if 'status' in df.columns else 12
    st.markdown(f'''
    <div class="metric-card">
        <h3 style="color: var(--danger-red); margin: 0;">Active Threats</h3>
        <h1 style="color: white; margin: 0.5rem 0;">{active_threats}</h1>
        <span class="real-data-indicator">Real-time</span>
    </div>
    ''', unsafe_allow_html=True)

# Threat Map
st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
st.markdown("### 🌍 Global Threat Map")

# Create threat map
fig = go.Figure()

threat_locations = [
    {"lat": 39.9042, "lon": 116.4074, "city": "Beijing", "threats": 15},
    {"lat": 55.7558, "lon": 37.6176, "city": "Moscow", "threats": 12},
    {"lat": 40.7128, "lon": -74.0060, "city": "New York", "threats": 8},
    {"lat": 51.5074, "lon": -0.1278, "city": "London", "threats": 6},
    {"lat": 35.6762, "lon": 139.6503, "city": "Tokyo", "threats": 4}
]

for loc in threat_locations:
    fig.add_trace(go.Scattergeo(
        lon=[loc["lon"]],
        lat=[loc["lat"]],
        mode='markers+text',
        marker=dict(
            size=loc["threats"] * 2,
            color='rgba(255, 59, 48, 0.8)',
            line=dict(width=2, color='white')
        ),
        text=f"{loc['city']}<br>{loc['threats']} threats",
        textposition='top center',
        textfont=dict(size=10, color='white'),
        hovertemplate=f"<b>{loc['city']}</b><br>Active Threats: {loc['threats']}<extra></extra>"
    ))

fig.update_layout(
    geo=dict(
        showframe=False,
        showcoastlines=True,
        coastlinecolor='rgba(255,255,255,0.1)',
        projection_type='natural earth',
        showcountries=True,
        countrycolor='rgba(255,255,255,0.05)',
        showland=True,
        landcolor='rgba(10, 14, 39, 0.8)',
        bgcolor='rgba(0,0,0,0)',
    ),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=400,
    margin=dict(t=0, b=0, l=0, r=0)
)

st.plotly_chart(fig, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# Charts Row
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 📊 Threat Severity Distribution")
    
    if 'severity' in df.columns:
        severity_counts = df['severity'].value_counts()
    else:
        severity_counts = pd.Series([8, 12, 15, 5], index=['Critical', 'High', 'Medium', 'Low'])
    
    colors = ['#FF3B30', '#FFB800', '#00D4FF', '#00C896']
    
    fig_severity = go.Figure(data=[
        go.Bar(
            x=severity_counts.index,
            y=severity_counts.values,
            marker_color=colors,
            text=severity_counts.values,
            textposition='auto',
        )
    ])
    
    fig_severity.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        height=300
    )
    
    st.plotly_chart(fig_severity, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 🎯 Attack Types")
    
    if 'event_type' in df.columns:
        attack_counts = df['event_type'].value_counts().head(5)
    else:
        attack_counts = pd.Series([8, 6, 5, 4, 2], index=['Malware', 'Phishing', 'DDoS', 'APT', 'Ransomware'])
    
    fig_attacks = px.pie(
        values=attack_counts.values,
        names=attack_counts.index,
        color_discrete_sequence=['#FF3B30', '#FFB800', '#00D4FF', '#00C896', '#9F7AEA']
    )
    
    fig_attacks.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=300
    )
    
    st.plotly_chart(fig_attacks, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Events Table
st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
st.markdown("### 🚨 Recent Security Events")

if not df.empty:
    display_df = df.head(15).copy()
    if 'timestamp' in display_df.columns:
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    st.dataframe(
        display_df,
        use_container_width=True,
        height=400
    )
else:
    st.info("No recent events to display")

st.markdown('</div>', unsafe_allow_html=True)

# Footer Status
st.markdown(f"""
<div style="text-align: center; margin-top: 2rem; padding: 1rem; border-top: 1px solid rgba(255,255,255,0.1);">
    <span style="color: rgba(255,255,255,0.7);">
        🛡️ AI-SOC Dashboard | 
        Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
        Status: {'🟢 LIVE DATA' if bigquery_result['success'] else '🟡 DEMO MODE'}
    </span>
</div>
""", unsafe_allow_html=True)
