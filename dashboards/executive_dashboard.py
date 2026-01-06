#!/usr/bin/env python3
"""
🎯 Executive SOC Dashboard
High-level security overview for C-suite and management
Focus on KPIs, trends, and strategic metrics
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import random

# Configure page
st.set_page_config(
    page_title="Executive SOC Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Executive CSS Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    :root {
        --executive-navy: #1E293B;
        --executive-blue: #3B82F6;
        --executive-green: #10B981;
        --executive-red: #EF4444;
        --executive-amber: #F59E0B;
        --executive-gray: #64748B;
    }

    .main {
        background: linear-gradient(135deg, #F8FAFC 0%, #E2E8F0 100%);
    }

    .executive-title {
        font-size: 3rem;
        font-weight: 800;
        color: var(--executive-navy);
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }

    .executive-subtitle {
        text-align: center;
        color: var(--executive-gray);
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    .kpi-card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        text-align: center;
        transition: all 0.3s ease;
        border-left: 4px solid var(--executive-blue);
    }

    .kpi-card:hover {
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.12);
        transform: translateY(-4px);
    }

    .kpi-value {
        font-size: 3rem;
        font-weight: 800;
        color: var(--executive-navy);
        margin: 0.5rem 0;
    }

    .kpi-label {
        font-size: 0.95rem;
        color: var(--executive-gray);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .kpi-delta {
        font-size: 0.9rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }

    .delta-positive { color: var(--executive-green); }
    .delta-negative { color: var(--executive-red); }

    .panel {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        margin: 1.5rem 0;
    }

    .panel-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--executive-navy);
        margin-bottom: 1.5rem;
    }

    .risk-badge {
        display: inline-block;
        padding: 0.5rem 1.2rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .risk-critical {
        background: #FEE2E2;
        color: var(--executive-red);
    }

    .risk-high {
        background: #FEF3C7;
        color: var(--executive-amber);
    }

    .risk-medium {
        background: #DBEAFE;
        color: var(--executive-blue);
    }

    .risk-low {
        background: #D1FAE5;
        color: var(--executive-green);
    }

    .stat-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 0;
        border-bottom: 1px solid #E2E8F0;
    }

    .stat-row:last-child {
        border-bottom: none;
    }

    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 2rem;
        color: var(--executive-gray);
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

def generate_executive_data():
    """Generate executive-level metrics and KPIs"""
    current_period = {
        'total_incidents': random.randint(80, 150),
        'critical_incidents': random.randint(5, 15),
        'avg_response_time': random.uniform(15, 45),
        'resolved_rate': random.uniform(85, 98),
        'false_positive_rate': random.uniform(3, 12),
        'security_score': random.randint(75, 95),
        'compliance_score': random.randint(88, 99),
        'mttr': random.uniform(30, 120),  # Mean Time To Respond
        'mttd': random.uniform(10, 60),   # Mean Time To Detect
        'mttr_resolve': random.uniform(240, 720),  # Mean Time To Resolve
    }

    previous_period = {k: v * random.uniform(0.85, 1.15) for k, v in current_period.items()}

    return current_period, previous_period

def calculate_delta(current, previous):
    """Calculate percentage change"""
    if previous == 0:
        return 0
    return ((current - previous) / previous) * 100

def format_delta(delta, inverse=False):
    """Format delta with color coding"""
    # inverse=True means lower is better (e.g., response time)
    if delta == 0:
        return "0%", "neutral"

    is_positive = delta > 0
    if inverse:
        is_positive = not is_positive

    sign = "+" if delta > 0 else ""
    color_class = "delta-positive" if is_positive else "delta-negative"
    icon = "▲" if delta > 0 else "▼"

    return f'{icon} {sign}{delta:.1f}%', color_class

# Generate data
current, previous = generate_executive_data()

# Header
st.markdown('<h1 class="executive-title">🎯 Executive Security Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="executive-subtitle">Strategic Overview | Security Posture | Business Risk Assessment</p>', unsafe_allow_html=True)

# Date range selector
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    period = st.selectbox(
        "Reporting Period",
        options=["Last 7 Days", "Last 30 Days", "Last Quarter", "Year to Date"],
        index=1
    )

st.markdown("---")

# Key Performance Indicators
st.markdown("## 📊 Key Security Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    delta, color = format_delta(calculate_delta(current['total_incidents'], previous['total_incidents']), inverse=True)
    st.markdown(f'''
    <div class="kpi-card">
        <div class="kpi-label">Total Incidents</div>
        <div class="kpi-value">{current["total_incidents"]}</div>
        <div class="kpi-delta {color}">{delta} vs previous</div>
    </div>
    ''', unsafe_allow_html=True)

with col2:
    delta, color = format_delta(calculate_delta(current['critical_incidents'], previous['critical_incidents']), inverse=True)
    st.markdown(f'''
    <div class="kpi-card" style="border-left-color: var(--executive-red);">
        <div class="kpi-label">Critical Incidents</div>
        <div class="kpi-value">{current["critical_incidents"]}</div>
        <div class="kpi-delta {color}">{delta} vs previous</div>
    </div>
    ''', unsafe_allow_html=True)

with col3:
    delta, color = format_delta(calculate_delta(current['resolved_rate'], previous['resolved_rate']))
    st.markdown(f'''
    <div class="kpi-card" style="border-left-color: var(--executive-green);">
        <div class="kpi-label">Resolution Rate</div>
        <div class="kpi-value">{current["resolved_rate"]:.1f}%</div>
        <div class="kpi-delta {color}">{delta} vs previous</div>
    </div>
    ''', unsafe_allow_html=True)

with col4:
    delta, color = format_delta(calculate_delta(current['security_score'], previous['security_score']))
    st.markdown(f'''
    <div class="kpi-card" style="border-left-color: var(--executive-amber);">
        <div class="kpi-label">Security Score</div>
        <div class="kpi-value">{current["security_score"]}/100</div>
        <div class="kpi-delta {color}">{delta} vs previous</div>
    </div>
    ''', unsafe_allow_html=True)

# Operational Metrics
st.markdown("---")
st.markdown("## ⏱️ Operational Performance")

col1, col2, col3 = st.columns(3)

with col1:
    delta, color = format_delta(calculate_delta(current['mttd'], previous['mttd']), inverse=True)
    st.markdown(f'''
    <div class="kpi-card">
        <div class="kpi-label">Mean Time to Detect</div>
        <div class="kpi-value">{current["mttd"]:.0f}m</div>
        <div class="kpi-delta {color}">{delta} vs previous</div>
    </div>
    ''', unsafe_allow_html=True)

with col2:
    delta, color = format_delta(calculate_delta(current['mttr'], previous['mttr']), inverse=True)
    st.markdown(f'''
    <div class="kpi-card">
        <div class="kpi-label">Mean Time to Respond</div>
        <div class="kpi-value">{current["mttr"]:.0f}m</div>
        <div class="kpi-delta {color}">{delta} vs previous</div>
    </div>
    ''', unsafe_allow_html=True)

with col3:
    delta, color = format_delta(calculate_delta(current['mttr_resolve'], previous['mttr_resolve']), inverse=True)
    st.markdown(f'''
    <div class="kpi-card">
        <div class="kpi-label">Mean Time to Resolve</div>
        <div class="kpi-value">{current["mttr_resolve"]/60:.1f}h</div>
        <div class="kpi-delta {color}">{delta} vs previous</div>
    </div>
    ''', unsafe_allow_html=True)

# Charts Row
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">📈 Incident Trend (30 Days)</div>', unsafe_allow_html=True)

    # Generate trend data
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    incidents = [random.randint(2, 8) for _ in range(30)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=incidents,
        mode='lines+markers',
        name='Incidents',
        line=dict(color='#3B82F6', width=3),
        marker=dict(size=6),
        fill='tozeroy',
        fillcolor='rgba(59, 130, 246, 0.1)'
    ))

    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='#1E293B'),
        xaxis=dict(
            gridcolor='#E2E8F0',
            showgrid=True,
            title='Date'
        ),
        yaxis=dict(
            gridcolor='#E2E8F0',
            showgrid=True,
            title='Incidents'
        ),
        height=300,
        margin=dict(l=0, r=0, t=0, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">🎯 Incident by Severity</div>', unsafe_allow_html=True)

    severities = ['Critical', 'High', 'Medium', 'Low']
    counts = [random.randint(2, 15), random.randint(10, 25), random.randint(20, 40), random.randint(15, 30)]
    colors_severity = ['#EF4444', '#F59E0B', '#3B82F6', '#10B981']

    fig = go.Figure(data=[
        go.Pie(
            labels=severities,
            values=counts,
            marker=dict(colors=colors_severity),
            hole=0.4,
            textinfo='label+percent',
            textfont=dict(size=14, color='white'),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )
    ])

    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='#1E293B'),
        height=300,
        margin=dict(l=0, r=0, t=20, b=0),
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Risk Assessment
st.markdown("---")
st.markdown("## 🛡️ Current Risk Posture")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Top Security Risks</div>', unsafe_allow_html=True)

    risks = [
        {"risk": "Unpatched Systems", "severity": "Critical", "affected": 24, "trend": "▲"},
        {"risk": "Phishing Attempts", "severity": "High", "affected": 156, "trend": "▼"},
        {"risk": "Unauthorized Access Attempts", "severity": "High", "affected": 89, "trend": "▲"},
        {"risk": "Data Exfiltration Alerts", "severity": "Medium", "affected": 12, "trend": "▬"},
        {"risk": "Malware Detections", "severity": "Medium", "affected": 34, "trend": "▼"},
    ]

    for risk in risks:
        severity_class = f"risk-{risk['severity'].lower()}"
        st.markdown(f'''
        <div class="stat-row">
            <div>
                <strong style="color: var(--executive-navy); font-size: 1.05rem;">{risk["risk"]}</strong>
                <div style="margin-top: 0.3rem;">
                    <span class="risk-badge {severity_class}">{risk["severity"]}</span>
                    <span style="color: var(--executive-gray); margin-left: 1rem;">
                        {risk["affected"]} affected assets
                    </span>
                </div>
            </div>
            <div style="font-size: 1.5rem; color: var(--executive-blue);">
                {risk["trend"]}
            </div>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Compliance Status</div>', unsafe_allow_html=True)

    frameworks = [
        {"name": "NIST CSF", "score": 94},
        {"name": "ISO 27001", "score": 91},
        {"name": "SOC 2", "score": 96},
        {"name": "GDPR", "score": 88},
        {"name": "HIPAA", "score": 92},
    ]

    for fw in frameworks:
        color = "#10B981" if fw["score"] >= 90 else "#F59E0B" if fw["score"] >= 75 else "#EF4444"
        st.markdown(f'''
        <div style="margin-bottom: 1.5rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="font-weight: 600; color: var(--executive-navy);">{fw["name"]}</span>
                <span style="font-weight: 700; color: {color};">{fw["score"]}%</span>
            </div>
            <div style="background: #E2E8F0; height: 8px; border-radius: 4px; overflow: hidden;">
                <div style="background: {color}; height: 100%; width: {fw["score"]}%; border-radius: 4px;"></div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Executive Summary
st.markdown("---")
st.markdown("## 📋 Executive Summary")

st.markdown('<div class="panel">', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### Key Findings

    **Security Posture:** The organization's overall security posture remains **strong** with a security
    score of **{}/100**. Critical incident count has **decreased by 12%** compared to the previous period,
    indicating effective threat mitigation strategies.

    **Operational Excellence:** Mean Time to Detect (MTTD) improved by **18%** to **{:.0f} minutes**,
    while Mean Time to Respond (MTTR) decreased to **{:.0f} minutes**, demonstrating enhanced
    SOC operational efficiency.

    **Compliance:** All major compliance frameworks maintain scores above **88%**, with SOC 2
    achieving **96%** compliance. Continuous monitoring and quarterly audits ensure regulatory alignment.

    **Recommendations:**
    - Prioritize patching of **24 critical systems** identified as high-risk
    - Enhance phishing awareness training to reduce social engineering success rate
    - Invest in advanced threat intelligence to improve proactive threat hunting
    """.format(current['security_score'], current['mttd'], current['mttr']))

with col2:
    st.markdown("### Quick Stats")

    st.metric("Active Analysts", "12", delta="2")
    st.metric("Security Tools", "24", delta="1")
    st.metric("Automated Playbooks", "47", delta="5")
    st.metric("Threat Intel Feeds", "15", delta="0")

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown(f'''
<div class="footer">
    <div style="font-weight: 600; color: var(--executive-navy); margin-bottom: 0.5rem;">
        🎯 Executive Security Dashboard | Report Generated: {datetime.now().strftime("%B %d, %Y at %H:%M")}
    </div>
    <div>
        For detailed operational metrics, please refer to the full SOC dashboard
    </div>
</div>
''', unsafe_allow_html=True)
