#!/usr/bin/env python3
"""
🛡️ Ultimate AI-SOC Command Center - Enhanced Edition
World-Class Enterprise Security Operations Center Dashboard
with Advanced Analytics, Real-time Monitoring, and Interactive Visualizations
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import random
import json
from io import StringIO

# Configure page for enterprise display
st.set_page_config(
    page_title="AI-SOC Command Center | Enhanced",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Enterprise CSS Design System
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --primary-dark: #0A0E27;
        --secondary-dark: #1a1f3a;
        --tertiary-dark: #252b48;
        --primary-blue: #0066FF;
        --accent-cyan: #00D4FF;
        --success-green: #00C896;
        --warning-amber: #FFB800;
        --danger-red: #FF3B30;
        --purple: #9F7AEA;
        --glass-bg: rgba(255, 255, 255, 0.03);
        --glass-border: rgba(255, 255, 255, 0.1);
    }

    .main-container {
        background: linear-gradient(135deg, #0A0E27 0%, #1a1f3a 100%);
        min-height: 100vh;
        font-family: 'Inter', sans-serif;
    }

    .header-title {
        font-size: 3.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--primary-blue) 0%, var(--accent-cyan) 50%, var(--purple) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 0 0 40px rgba(0, 102, 255, 0.4);
        animation: glow 3s ease-in-out infinite alternate;
    }

    @keyframes glow {
        from { filter: drop-shadow(0 0 20px rgba(0, 212, 255, 0.4)); }
        to { filter: drop-shadow(0 0 30px rgba(159, 122, 234, 0.6)); }
    }

    .subtitle {
        text-align: center;
        color: rgba(255, 255, 255, 0.7);
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    .glass-panel {
        background: var(--glass-bg);
        backdrop-filter: blur(20px);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }

    .glass-panel:hover {
        border-color: rgba(0, 212, 255, 0.3);
        box-shadow: 0 12px 48px rgba(0, 102, 255, 0.2);
        transform: translateY(-2px);
    }

    .metric-card {
        background: linear-gradient(135deg, var(--glass-bg) 0%, rgba(255, 255, 255, 0.05) 100%);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: left 0.5s;
    }

    .metric-card:hover::before {
        left: 100%;
    }

    .metric-card:hover {
        border-color: var(--accent-cyan);
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0, 212, 255, 0.3);
    }

    .status-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .status-success {
        background: linear-gradient(135deg, rgba(0, 200, 150, 0.2), rgba(0, 200, 150, 0.1));
        border: 2px solid var(--success-green);
        color: var(--success-green);
    }

    .status-danger {
        background: linear-gradient(135deg, rgba(255, 59, 48, 0.2), rgba(255, 59, 48, 0.1));
        border: 2px solid var(--danger-red);
        color: var(--danger-red);
    }

    .status-warning {
        background: linear-gradient(135deg, rgba(255, 184, 0, 0.2), rgba(255, 184, 0, 0.1));
        border: 2px solid var(--warning-amber);
        color: var(--warning-amber);
    }

    .data-indicator {
        background: linear-gradient(90deg, var(--success-green) 0%, var(--accent-cyan) 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-left: 0.5rem;
        animation: pulse 2s infinite;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.8; transform: scale(1.05); }
    }

    .threat-tag {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 0.2rem;
    }

    .tag-critical { background: rgba(255, 59, 48, 0.2); color: var(--danger-red); border: 1px solid var(--danger-red); }
    .tag-high { background: rgba(255, 184, 0, 0.2); color: var(--warning-amber); border: 1px solid var(--warning-amber); }
    .tag-medium { background: rgba(0, 212, 255, 0.2); color: var(--accent-cyan); border: 1px solid var(--accent-cyan); }
    .tag-low { background: rgba(0, 200, 150, 0.2); color: var(--success-green); border: 1px solid var(--success-green); }

    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: white;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }

    .section-header::before {
        content: '';
        width: 4px;
        height: 2rem;
        background: linear-gradient(180deg, var(--primary-blue), var(--accent-cyan));
        border-radius: 2px;
    }

    .stat-number {
        font-size: 2.5rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        background: linear-gradient(135deg, white, rgba(255, 255, 255, 0.8));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .code-block {
        background: rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: var(--accent-cyan);
    }

    .timeline-item {
        border-left: 3px solid var(--primary-blue);
        padding-left: 1.5rem;
        margin-bottom: 1.5rem;
        position: relative;
    }

    .timeline-item::before {
        content: '';
        position: absolute;
        left: -7px;
        top: 0;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: var(--accent-cyan);
        box-shadow: 0 0 10px var(--accent-cyan);
    }

    .footer-status {
        text-align: center;
        margin-top: 3rem;
        padding: 2rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.9rem;
    }

    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, var(--primary-dark), var(--secondary-dark));
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }

    ::-webkit-scrollbar-track {
        background: var(--primary-dark);
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--primary-blue), var(--accent-cyan));
        border-radius: 5px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--accent-cyan);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False
if 'selected_severity' not in st.session_state:
    st.session_state.selected_severity = 'All'
if 'date_range' not in st.session_state:
    st.session_state.date_range = 24  # hours

# Import BigQuery configuration
try:
    from bigquery_config import (
        get_bigquery_client,
        test_bigquery_connection,
        get_event_statistics,
        query_siem_events,
        get_available_tables,
        BIGQUERY_TABLES,
        GCP_PROJECT_ID,
        BIGQUERY_DATASET
    )
    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False

# Helper Functions
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
        sample_data = query_siem_events(limit=100, hours=24)
        
        # Transform data if needed (handle different table structures)
        if not sample_data.empty:
            # Map BigQuery columns to dashboard expected columns
            column_mapping = {
                'alarmId': 'event_id',
                'timestamp': 'timestamp',
                'events': 'event_type',
                'processed_by_ada': 'analyst_assigned'
            }
            
            # Rename columns if they exist
            for old_col, new_col in column_mapping.items():
                if old_col in sample_data.columns and new_col not in sample_data.columns:
                    sample_data = sample_data.rename(columns={old_col: new_col})
        
        return {
            'success': True,
            'total_events': stats.get('total_events', 0),
            'unique_alarms': stats.get('unique_alarms', 0),
            'unique_sources': stats.get('unique_sources', 0),
            'unique_destinations': stats.get('unique_destinations', 0),
            'sample_data': sample_data,
            'tables_available': get_available_tables(),
            'project_id': GCP_PROJECT_ID,
            'dataset': BIGQUERY_DATASET
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'total_events': 0,
            'unique_alarms': 0,
            'sample_data': pd.DataFrame()
        }

def generate_enhanced_mock_data(num_records=100):
    """Generate high-quality enhanced mock data with realistic patterns"""
    events = []

    threat_types = [
        "Advanced Persistent Threat", "Malware Infection", "Phishing Attack",
        "DDoS Attack", "Data Exfiltration", "Insider Threat", "Ransomware",
        "SQL Injection", "XSS Attack", "Zero-Day Exploit", "Brute Force",
        "Man-in-the-Middle", "Credential Stuffing", "Cryptojacking"
    ]

    severities = ["Critical", "High", "Medium", "Low"]
    severity_weights = [0.15, 0.25, 0.35, 0.25]  # Realistic distribution

    sources = [
        "185.220.101.45", "203.0.113.45", "198.51.100.10", "45.142.214.123",
        "91.205.189.15", "103.251.167.20", "176.123.4.67", "89.248.165.13"
    ]

    destinations = [
        "192.168.1.100", "192.168.1.50", "10.0.0.10", "172.16.0.5",
        "192.168.2.75", "10.0.1.20", "172.16.1.100", "192.168.3.200"
    ]

    statuses = ["Active", "Investigating", "Contained", "Resolved", "False Positive"]
    status_weights = [0.20, 0.25, 0.20, 0.30, 0.05]

    mitre_techniques = [
        "T1566.001 - Spearphishing Attachment",
        "T1059.001 - PowerShell",
        "T1055 - Process Injection",
        "T1070.004 - File Deletion",
        "T1003.001 - LSASS Memory",
        "T1569.002 - Service Execution",
        "T1071.001 - Web Protocols",
        "T1027 - Obfuscated Files"
    ]

    countries = ["China", "Russia", "North Korea", "Iran", "Unknown", "Brazil", "Nigeria", "Ukraine"]

    threat_actors = [
        "APT28", "APT29", "Lazarus Group", "FIN7", "Carbanak",
        "Unknown", "Script Kiddie", "Insider", "Automated Bot"
    ]

    protocols = ["HTTP", "HTTPS", "SSH", "FTP", "SMB", "DNS", "SMTP", "RDP"]

    assets = [
        "Web Server 01", "Database Server", "File Server", "Domain Controller",
        "Email Server", "Application Server", "Backup Server", "Test Environment"
    ]

    for i in range(num_records):
        severity = random.choices(severities, weights=severity_weights)[0]
        status = random.choices(statuses, weights=status_weights)[0]

        # More critical events are more recent
        if severity == "Critical":
            time_range = 30
        elif severity == "High":
            time_range = 120
        elif severity == "Medium":
            time_range = 360
        else:
            time_range = 720

        timestamp = datetime.now() - timedelta(minutes=random.randint(1, time_range))

        events.append({
            "timestamp": timestamp,
            "event_id": f"EVT-{random.randint(100000, 999999)}",
            "severity": severity,
            "event_type": random.choice(threat_types),
            "source_ip": random.choice(sources),
            "destination_ip": random.choice(destinations),
            "source_country": random.choice(countries),
            "status": status,
            "mitre_technique": random.choice(mitre_techniques),
            "confidence": random.randint(65, 99),
            "packets": random.randint(100, 100000),
            "bytes_transferred": random.randint(1024, 10485760),
            "protocol": random.choice(protocols),
            "threat_actor": random.choice(threat_actors),
            "affected_asset": random.choice(assets),
            "analyst_assigned": f"Analyst {random.randint(1, 8)}" if status == "Investigating" else "Unassigned",
            "response_time_min": random.randint(1, 120) if status != "Active" else None,
            "false_positive_score": random.uniform(0.1, 0.95)
        })

    return pd.DataFrame(events)

def create_threat_timeline(df):
    """Create interactive threat timeline"""
    if df.empty or 'timestamp' not in df.columns:
        return go.Figure()

    # Group by hour
    df_copy = df.copy()
    
    # Ensure timestamp is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df_copy['timestamp']):
        try:
            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'], errors='coerce')
        except Exception:
            # If conversion fails, return empty figure
            return go.Figure()
    
    # Remove rows with invalid timestamps
    df_copy = df_copy.dropna(subset=['timestamp'])
    
    if df_copy.empty:
        return go.Figure()
    
    # Group by hour
    df_copy['hour'] = df_copy['timestamp'].dt.floor('H')
    timeline_data = df_copy.groupby(['hour', 'severity']).size().reset_index(name='count')

    fig = px.line(
        timeline_data,
        x='hour',
        y='count',
        color='severity',
        title='Threat Activity Timeline',
        color_discrete_map={
            'Critical': '#FF3B30',
            'High': '#FFB800',
            'Medium': '#00D4FF',
            'Low': '#00C896'
        }
    )

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            title='Time'
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            title='Event Count'
        ),
        height=350,
        hovermode='x unified'
    )

    return fig

def create_network_topology(df):
    """Create network topology graph"""
    if df.empty or 'source_ip' not in df.columns:
        return go.Figure()

    # Get top source and destination IPs
    top_sources = df['source_ip'].value_counts().head(10)
    top_dests = df['destination_ip'].value_counts().head(10)

    # Create network graph
    edge_trace = []
    node_trace = []

    # Nodes
    all_ips = list(set(list(top_sources.index) + list(top_dests.index)))

    # Create positions in a circle
    n = len(all_ips)
    node_x = [np.cos(2 * np.pi * i / n) for i in range(n)]
    node_y = [np.sin(2 * np.pi * i / n) for i in range(n)]

    # Create edges
    for idx, row in df.head(50).iterrows():
        if row['source_ip'] in all_ips and row['destination_ip'] in all_ips:
            src_idx = all_ips.index(row['source_ip'])
            dst_idx = all_ips.index(row['destination_ip'])

            edge_trace.append(
                go.Scatter(
                    x=[node_x[src_idx], node_x[dst_idx]],
                    y=[node_y[src_idx], node_y[dst_idx]],
                    mode='lines',
                    line=dict(width=0.5, color='rgba(0, 212, 255, 0.3)'),
                    hoverinfo='none',
                    showlegend=False
                )
            )

    # Node colors based on threat count
    node_colors = []
    node_sizes = []
    node_text = []

    for ip in all_ips:
        threat_count = len(df[(df['source_ip'] == ip) | (df['destination_ip'] == ip)])
        node_colors.append(threat_count)
        node_sizes.append(min(10 + threat_count * 2, 40))
        node_text.append(f"{ip}<br>Threats: {threat_count}")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        hovertext=node_text,
        hoverinfo='text',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            colorscale='Reds',
            showscale=True,
            colorbar=dict(
                title="Threat<br>Count",
                thickness=15,
                len=0.7,
                bgcolor='rgba(0,0,0,0)',
                tickfont=dict(color='white')
            ),
            line=dict(width=2, color='white')
        ),
        text=[ip.split('.')[-1] for ip in all_ips],
        textposition='top center',
        textfont=dict(size=8, color='white'),
        showlegend=False
    )

    fig = go.Figure(data=edge_trace + [node_trace])

    fig.update_layout(
        title='Network Topology & Threat Connections',
        showlegend=False,
        hovermode='closest',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=450
    )

    return fig

def create_mitre_heatmap(df):
    """Create MITRE ATT&CK technique heatmap"""
    if df.empty or 'mitre_technique' not in df.columns:
        return go.Figure()

    # Extract technique categories
    technique_counts = df['mitre_technique'].value_counts().head(15)

    # Create matrix for heatmap
    techniques = [t.split(' - ')[0] if ' - ' in t else t for t in technique_counts.index]
    descriptions = [t.split(' - ')[1] if ' - ' in t else 'Unknown' for t in technique_counts.index]

    fig = go.Figure(data=go.Heatmap(
        z=[technique_counts.values],
        x=techniques,
        y=['Frequency'],
        colorscale='Reds',
        text=[[f"{desc}<br>Count: {count}" for desc, count in zip(descriptions, technique_counts.values)]],
        hovertemplate='%{text}<extra></extra>',
        colorbar=dict(
            title="Events",
            thickness=15,
            len=0.7,
            bgcolor='rgba(0,0,0,0)',
            tickfont=dict(color='white')
        )
    ))

    fig.update_layout(
        title='MITRE ATT&CK Technique Distribution',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=300,
        xaxis=dict(title='', tickangle=-45),
        yaxis=dict(title='')
    )

    return fig

def create_response_funnel(df):
    """Create incident response funnel"""
    if df.empty or 'status' not in df.columns:
        return go.Figure()

    status_order = ['Active', 'Investigating', 'Contained', 'Resolved']
    status_counts = df['status'].value_counts()

    values = [status_counts.get(status, 0) for status in status_order]

    fig = go.Figure(go.Funnel(
        y=status_order,
        x=values,
        textposition="inside",
        textinfo="value+percent initial",
        marker=dict(
            color=['#FF3B30', '#FFB800', '#00D4FF', '#00C896']
        ),
        connector=dict(line=dict(color="rgba(255,255,255,0.3)", width=2))
    ))

    fig.update_layout(
        title='Incident Response Funnel',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=400
    )

    return fig

def create_geo_distribution(df):
    """Create geographic threat distribution"""
    if df.empty or 'source_country' not in df.columns:
        return go.Figure()

    country_counts = df['source_country'].value_counts()

    fig = go.Figure(data=go.Choropleth(
        locations=country_counts.index,
        locationmode='country names',
        z=country_counts.values,
        colorscale='Reds',
        marker_line_color='rgba(255,255,255,0.2)',
        colorbar=dict(
            title="Threats",
            thickness=15,
            len=0.7,
            bgcolor='rgba(0,0,0,0)',
            tickfont=dict(color='white')
        )
    ))

    fig.update_layout(
        title='Global Threat Origin Distribution',
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor='rgba(255,255,255,0.2)',
            projection_type='natural earth',
            bgcolor='rgba(0,0,0,0)',
            landcolor='rgba(26, 31, 58, 0.5)',
            showland=True
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=450
    )

    return fig

# Sidebar Configuration
with st.sidebar:
    st.markdown('<h2 style="color: var(--accent-cyan);">⚙️ Dashboard Controls</h2>', unsafe_allow_html=True)

    st.markdown("---")

    # Auto-refresh
    st.markdown("### 🔄 Auto-Refresh")
    auto_refresh = st.checkbox("Enable Auto-Refresh", value=st.session_state.auto_refresh)
    st.session_state.auto_refresh = auto_refresh

    if auto_refresh:
        refresh_interval = st.slider("Refresh Interval (seconds)", 5, 60, 30)

    # Date Range
    st.markdown("### 📅 Time Range")
    date_range = st.selectbox(
        "Select Time Range",
        options=[1, 6, 12, 24, 48, 72, 168],
        format_func=lambda x: f"Last {x} hours" if x < 24 else f"Last {x//24} days",
        index=3
    )
    st.session_state.date_range = date_range

    # Severity Filter
    st.markdown("### 🎯 Severity Filter")
    severity_filter = st.multiselect(
        "Select Severities",
        options=['Critical', 'High', 'Medium', 'Low'],
        default=['Critical', 'High', 'Medium', 'Low']
    )

    # Dashboard Sections
    st.markdown("### 📊 Dashboard Sections")
    show_overview = st.checkbox("Overview Metrics", value=True)
    show_timeline = st.checkbox("Threat Timeline", value=True)
    show_network = st.checkbox("Network Topology", value=True)
    show_mitre = st.checkbox("MITRE ATT&CK", value=True)
    show_geo = st.checkbox("Geographic Distribution", value=True)
    show_analytics = st.checkbox("Advanced Analytics", value=True)
    show_events = st.checkbox("Recent Events Table", value=True)

    st.markdown("---")

    # Export Options
    st.markdown("### 💾 Export Data")

    st.markdown("---")

    # System Status
    st.markdown("### 🖥️ System Status")
    st.metric("CPU Usage", "34%", delta="-5%")
    st.metric("Memory", "2.1 GB", delta="0.3 GB")
    st.metric("Uptime", "15d 7h 23m")

    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: rgba(255,255,255,0.5); font-size: 0.8rem;">'
        '🛡️ AI-SOC v2.0<br>Enhanced Edition</div>',
        unsafe_allow_html=True
    )

# Main Content
st.markdown('<h1 class="header-title">🛡️ AI-SOC COMMAND CENTER</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enterprise Security Operations | Real-time Threat Intelligence & Advanced Analytics</p>', unsafe_allow_html=True)

# Try BigQuery connection
with st.spinner("🔍 Establishing secure connection to BigQuery..."):
    bigquery_result = try_bigquery_connection()

# Connection Status
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if bigquery_result['success']:
        st.markdown(
            '<div class="status-badge status-success">🎉 CONNECTED TO LIVE BIGQUERY DATA</div>',
            unsafe_allow_html=True
        )
        df = bigquery_result['sample_data']
        total_events = bigquery_result['total_events']
        unique_alarms = bigquery_result['unique_alarms']
        data_source = "BigQuery"
    else:
        st.markdown(
            f'<div class="status-badge status-warning">⚠️ BigQuery Offline - Using Enhanced Demo Data</div>',
            unsafe_allow_html=True
        )
        df = generate_enhanced_mock_data(150)
        total_events = len(df)
        unique_alarms = df['event_id'].nunique()
        data_source = "Demo"

# Apply severity filter
if severity_filter:
    df_filtered = df[df['severity'].isin(severity_filter)] if 'severity' in df.columns else df
else:
    df_filtered = df

# Ensure timestamp is datetime type for all visualizations
if 'timestamp' in df_filtered.columns:
    if not pd.api.types.is_datetime64_any_dtype(df_filtered['timestamp']):
        try:
            df_filtered['timestamp'] = pd.to_datetime(df_filtered['timestamp'], errors='coerce')
        except Exception:
            pass  # Keep as-is if conversion fails

# Overview Metrics
if show_overview:
    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(f'''
        <div class="metric-card">
            <h3 style="color: var(--accent-cyan); margin: 0; font-size: 1rem;">Total Events</h3>
            <div class="stat-number">{total_events:,}</div>
            <span class="data-indicator">{data_source}</span>
        </div>
        ''', unsafe_allow_html=True)

    with col2:
        critical_count = len(df_filtered[df_filtered['severity'] == 'Critical']) if 'severity' in df_filtered.columns else 0
        st.markdown(f'''
        <div class="metric-card">
            <h3 style="color: var(--danger-red); margin: 0; font-size: 1rem;">Critical Alerts</h3>
            <div class="stat-number">{critical_count}</div>
            <span class="data-indicator">High Priority</span>
        </div>
        ''', unsafe_allow_html=True)

    with col3:
        active_threats = len(df_filtered[df_filtered['status'] == 'Active']) if 'status' in df_filtered.columns else 0
        st.markdown(f'''
        <div class="metric-card">
            <h3 style="color: var(--warning-amber); margin: 0; font-size: 1rem;">Active Threats</h3>
            <div class="stat-number">{active_threats}</div>
            <span class="data-indicator">Real-time</span>
        </div>
        ''', unsafe_allow_html=True)

    with col4:
        resolved_count = len(df_filtered[df_filtered['status'] == 'Resolved']) if 'status' in df_filtered.columns else 0
        st.markdown(f'''
        <div class="metric-card">
            <h3 style="color: var(--success-green); margin: 0; font-size: 1rem;">Resolved</h3>
            <div class="stat-number">{resolved_count}</div>
            <span class="data-indicator">Success Rate</span>
        </div>
        ''', unsafe_allow_html=True)

    with col5:
        avg_response = df_filtered['response_time_min'].mean() if 'response_time_min' in df_filtered.columns else 0
        st.markdown(f'''
        <div class="metric-card">
            <h3 style="color: var(--purple); margin: 0; font-size: 1rem;">Avg Response</h3>
            <div class="stat-number">{avg_response:.0f}m</div>
            <span class="data-indicator">MTTR</span>
        </div>
        ''', unsafe_allow_html=True)

# Threat Timeline
if show_timeline:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">📈 Threat Activity Timeline</h2>', unsafe_allow_html=True)
    fig_timeline = create_threat_timeline(df_filtered)
    st.plotly_chart(fig_timeline, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Two Column Layout
col_left, col_right = st.columns(2)

with col_left:
    # Network Topology
    if show_network:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-header">🌐 Network Topology</h2>', unsafe_allow_html=True)
        fig_network = create_network_topology(df_filtered)
        st.plotly_chart(fig_network, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # MITRE ATT&CK
    if show_mitre:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-header">⚔️ MITRE ATT&CK Distribution</h2>', unsafe_allow_html=True)
        fig_mitre = create_mitre_heatmap(df_filtered)
        st.plotly_chart(fig_mitre, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    # Response Funnel
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">🔄 Incident Response Pipeline</h2>', unsafe_allow_html=True)
    fig_funnel = create_response_funnel(df_filtered)
    st.plotly_chart(fig_funnel, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Geographic Distribution
if show_geo:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">🌍 Global Threat Origin Map</h2>', unsafe_allow_html=True)
    fig_geo = create_geo_distribution(df_filtered)
    st.plotly_chart(fig_geo, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Advanced Analytics
if show_analytics:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">🤖 AI-Powered Analytics</h2>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🎯 Threat Actor Profiling")
        if 'threat_actor' in df_filtered.columns:
            actor_counts = df_filtered['threat_actor'].value_counts().head(5)
            for actor, count in actor_counts.items():
                st.markdown(f'<div class="code-block">{actor}: {count} attacks</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("#### 🔍 Top Targeted Assets")
        if 'affected_asset' in df_filtered.columns:
            asset_counts = df_filtered['affected_asset'].value_counts().head(5)
            for asset, count in asset_counts.items():
                risk_level = "🔴" if count > 10 else "🟡" if count > 5 else "🟢"
                st.markdown(f'{risk_level} **{asset}**: {count} incidents')

    with col3:
        st.markdown("#### 📊 Protocol Analysis")
        if 'protocol' in df_filtered.columns:
            protocol_counts = df_filtered['protocol'].value_counts().head(5)
            fig_protocol = px.bar(
                x=protocol_counts.values,
                y=protocol_counts.index,
                orientation='h',
                color=protocol_counts.values,
                color_continuous_scale='Reds'
            )
            fig_protocol.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', size=10),
                height=200,
                showlegend=False,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            st.plotly_chart(fig_protocol, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Recent Events Table
if show_events:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">🚨 Recent Security Events</h2>', unsafe_allow_html=True)

    # Search functionality
    search_term = st.text_input("🔍 Search events (IP, Event ID, Type...)", "")

    if not df_filtered.empty:
        display_df = df_filtered.copy()

        # Apply search filter
        if search_term:
            mask = display_df.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
            display_df = display_df[mask]

        # Sort by timestamp
        if 'timestamp' in display_df.columns:
            # Ensure timestamp is datetime for sorting
            if not pd.api.types.is_datetime64_any_dtype(display_df['timestamp']):
                display_df['timestamp'] = pd.to_datetime(display_df['timestamp'], errors='coerce')
            display_df = display_df.sort_values('timestamp', ascending=False)
            # Format for display
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

        # Select columns to display
        columns_to_show = ['timestamp', 'event_id', 'severity', 'event_type',
                          'source_ip', 'destination_ip', 'status', 'confidence']
        available_columns = [col for col in columns_to_show if col in display_df.columns]

        st.dataframe(
            display_df[available_columns].head(20),
            use_container_width=True,
            height=400
        )

        # Export buttons
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Export CSV",
                data=csv,
                file_name=f'soc_events_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv'
            )

        with col2:
            json_str = display_df.to_json(orient='records', indent=2)
            st.download_button(
                label="📥 Export JSON",
                data=json_str,
                file_name=f'soc_events_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
                mime='application/json'
            )
    else:
        st.info("No events match the current filters")

    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown(f"""
<div class="footer-status">
    <div style="display: flex; justify-content: center; align-items: center; gap: 2rem; flex-wrap: wrap;">
        <span>🛡️ AI-SOC Command Center Enhanced</span>
        <span>|</span>
        <span>⏰ Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>
        <span>|</span>
        <span>{'🟢 LIVE DATA' if bigquery_result['success'] else '🟡 DEMO MODE'}</span>
        <span>|</span>
        <span>📊 {len(df_filtered)} Events Displayed</span>
        <span>|</span>
        <span>⚡ {len(severity_filter)} Severity Filters Active</span>
    </div>
    <div style="margin-top: 1rem; font-size: 0.8rem; color: rgba(255,255,255,0.5);">
        Powered by Streamlit • Plotly • BigQuery | v2.0.0 Enhanced Edition
    </div>
</div>
""", unsafe_allow_html=True)

# Auto-refresh logic
if auto_refresh:
    st.rerun()
