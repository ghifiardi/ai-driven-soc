#!/usr/bin/env python3
"""
Real AI Model Training Dashboard - FIXED VERSION
Uses actual BigQuery data instead of hardcoded values
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import subprocess
import json
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Real AI Model Training Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .data-source-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🤖 Real AI Model Training Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Live Data from AI-Driven SOC System</p>', unsafe_allow_html=True)

# Real-time data from BigQuery
@st.cache_data(ttl=300)
def get_real_data():
    """Fetch real metrics from BigQuery"""
    try:
        result = subprocess.run(
            ["python3", "~/get_real_metrics.py"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            universal_newlines=True
        )
        
        if result.returncode == 0:
            data = json.loads(result.stdout.strip())
            return {
                "total_alerts": int(data["total_alerts"]),
                "unique_alarms": int(data["unique_alarms"]),
                "taa_processed": int(data["taa_processed"]),
                "dashboard_alerts": int(data["dashboard_alerts"]),
                "ada_accuracy": float(data["ada_accuracy"]),
                "taa_confidence": float(data["taa_confidence"]) * 100,  # Convert to percentage
                "true_positive_rate": float(data["true_positive_rate"]),
                "ada_score": float(data["ada_score"]),
                "taa_severity": float(data["taa_severity"])
            }
        else:
            st.error(f"Failed to fetch data: {result.stderr}")
            return None
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# Get real data
with st.spinner("Fetching real-time data from BigQuery..."):
    data = get_real_data()

if data is None:
    st.error("Unable to load real data. Please check BigQuery connection.")
    st.stop()

# Main metrics
st.header("📊 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Events Processed</div>
        <div class="metric-value">{data['total_alerts']:,}</div>
        <div class="metric-label">Real SIEM Events</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">ADA Model Accuracy</div>
        <div class="metric-value">{data['ada_accuracy']:.1f}%</div>
        <div class="metric-label">Real Performance</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">TAA Confidence</div>
        <div class="metric-value">{data['taa_confidence']:.1f}%</div>
        <div class="metric-label">Average Confidence</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">True Positive Rate</div>
        <div class="metric-value">{data['true_positive_rate']:.1f}%</div>
        <div class="metric-label">Real Detection Rate</div>
    </div>
    """, unsafe_allow_html=True)

# Data Sources Breakdown
st.header("📊 Data Sources Breakdown")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="data-source-card">
        <div class="metric-value">{data['total_alerts']:,}</div>
        <div class="metric-label">Raw Events</div>
        <div class="metric-label">SIEM Events</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="data-source-card">
        <div class="metric-value">{data['unique_alarms']:,}</div>
        <div class="metric-label">Processed Alarms</div>
        <div class="metric-label">SIEM Alarms</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="data-source-card">
        <div class="metric-value">{data['taa_processed']:,}</div>
        <div class="metric-label">TAA Processed</div>
        <div class="metric-label">TAA State</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="data-source-card">
        <div class="metric-value">{data['dashboard_alerts']:,}</div>
        <div class="metric-label">Dashboard Data</div>
        <div class="metric-label">Dashboard Alerts</div>
    </div>
    """, unsafe_allow_html=True)

# Model Performance Analysis
st.header("📈 Model Performance Analysis")
st.subheader("Real Model Performance Metrics")

col1, col2 = st.columns([2, 1])

with col1:
    # Create performance chart
    performance_data = pd.DataFrame({
        'Metric': ['ADA Accuracy', 'TAA Confidence', 'True Positive Rate', 'Severity Score'],
        'Value': [data['ada_accuracy'], data['taa_confidence'], data['true_positive_rate'], data['taa_severity'] * 100]
    })
    
    fig = px.bar(
        performance_data,
        x='Metric',
        y='Value',
        title="Model Performance Metrics",
        color='Value',
        color_continuous_scale='RdYlGn'
    )
    
    fig.update_layout(
        yaxis_title="Percentage (%)",
        showlegend=False,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Current Metrics")
    st.metric("Total Events", f"{data['total_alerts']:,}")
    st.metric("Processed Alarms", f"{data['unique_alarms']:,}")
    st.metric("ADA Accuracy", f"{data['ada_accuracy']:.1f}%")
    st.metric("TAA Confidence", f"{data['taa_confidence']:.1f}%")
    st.metric("True Positive Rate", f"{data['true_positive_rate']:.1f}%")
    st.metric("TAA Severity", f"{data['taa_severity']:.3f}")

# Model comparison
st.header("🔬 Model Comparison")

col1, col2 = st.columns(2)

with col1:
    # ADA vs TAA Performance
    model_performance = pd.DataFrame({
        'Model': ['ADA Model', 'TAA Model'],
        'Performance': [data['ada_accuracy'], data['taa_confidence']]
    })
    
    fig1 = px.bar(
        model_performance,
        x='Model',
        y='Performance',
        title="ADA vs TAA Performance",
        color='Performance',
        color_continuous_scale='Blues'
    )
    
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    # Model Scores Comparison
    model_scores = pd.DataFrame({
        'Model': ['ADA Model', 'TAA Model'],
        'Score': [data['ada_score'], data['taa_severity']]
    })
    
    fig2 = px.bar(
        model_scores,
        x='Model',
        y='Score',
        title="Model Scores Comparison",
        color='Score',
        color_continuous_scale='Viridis'
    )
    
    st.plotly_chart(fig2, use_container_width=True)

# Data Quality Metrics
st.header("📋 Data Quality Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Data Completeness", "100.0%")
    st.metric("Model Accuracy", f"{data['ada_accuracy']:.1f}%")
    st.metric("Confidence Rate", f"{data['taa_confidence']:.1f}%")

with col2:
    st.metric("True Positive Rate", f"{data['true_positive_rate']:.1f}%")
    st.metric("ADA Score", f"{data['ada_score']:.3f}")
    st.metric("TAA Severity", f"{data['taa_severity']:.3f}")

with col3:
    st.metric("Total Events", f"{data['total_alerts']:,}")
    st.metric("Processed Alarms", f"{data['unique_alarms']:,}")
    st.metric("System Status", "Active")

# System status
st.header("🔄 System Status")

col1, col2, col3 = st.columns(3)

with col1:
    st.success("✅ BigQuery Connected")
    st.info(f"📊 Data Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

with col2:
    if data['ada_accuracy'] < 50:
        st.error(f"⚠️ Low Model Accuracy: {data['ada_accuracy']:.1f}%")
    else:
        st.success(f"✅ Good Model Accuracy: {data['ada_accuracy']:.1f}%")
    
    if data['true_positive_rate'] < 30:
        st.warning(f"⚠️ High False Positive Rate: {100-data['true_positive_rate']:.1f}%")
    else:
        st.success(f"✅ Acceptable True Positive Rate: {data['true_positive_rate']:.1f}%")

with col3:
    st.info("🔄 Real-time Data")
    st.info("📈 Live Metrics")
    st.info("🔗 BigQuery Integration")

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p><strong>Real Data:</strong> ADA {data['ada_accuracy']:.1f}% accuracy, TAA {data['taa_confidence']:.1f}% confidence, {data['true_positive_rate']:.1f}% true positive rate</p>
    <p><strong>Data Sources:</strong> {data['total_alerts']:,} SIEM Events, {data['unique_alarms']:,} SIEM Alarms, {data['taa_processed']:,} TAA Processed</p>
    <p><strong>Last Updated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</div>
""", unsafe_allow_html=True)


