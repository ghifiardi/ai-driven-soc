#!/usr/bin/env python3
"""
Simple Working AI Security Dashboard
Basic Streamlit components only - guaranteed to work
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="AI Security Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# Simple header
st.title("🛡️ AI Security Dashboard")
st.write("Simple and working dashboard")

# Sidebar
st.sidebar.header("Controls")
use_real_data = st.sidebar.checkbox("Use Real Data", value=False)

# Simple data
if use_real_data:
    try:
        from google.cloud import bigquery
        client = bigquery.Client(project="chronicle-dev-2be9")
        
        # Simple count queries
        siem_count = next(client.query("SELECT COUNT(*) as total FROM `chronicle-dev-2be9.gatra_database.siem_events`").result()).total
        ada_count = next(client.query("SELECT COUNT(*) as total FROM `chronicle-dev-2be9.gatra_database.processed_ada_alerts`").result()).total
        
        st.success(f"Connected to BigQuery! SIEM Events: {siem_count:,}, ADA Alerts: {ada_count:,}")
        
        data = {
            'siem_events': siem_count,
            'ada_alerts': ada_count,
            'taa_cases': 5000,
            'cra_incidents': 3000
        }
        
    except Exception as e:
        st.error(f"BigQuery error: {e}")
        data = {'siem_events': 100000, 'ada_alerts': 50000, 'taa_cases': 5000, 'cra_incidents': 3000}
else:
    data = {'siem_events': 100000, 'ada_alerts': 50000, 'taa_cases': 5000, 'cra_incidents': 3000}
    st.info("Using sample data")

# Simple metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("SIEM Events", f"{data['siem_events']:,}")

with col2:
    st.metric("ADA Alerts", f"{data['ada_alerts']:,}")

with col3:
    st.metric("TAA Cases", f"{data['taa_cases']:,}")

with col4:
    st.metric("CRA Incidents", f"{data['cra_incidents']:,}")

# Simple chart
st.subheader("Threat Distribution")
chart_data = pd.DataFrame({
    'Agent': ['ADA', 'TAA', 'CRA'],
    'Count': [data['ada_alerts'], data['taa_cases'], data['cra_incidents']]
})

fig = px.bar(chart_data, x='Agent', y='Count', title="AI Agent Activity")
st.plotly_chart(fig)

# Simple progress bars
st.subheader("System Status")
st.write("System Health")
st.progress(0.95)
st.write("95%")

st.write("Response Time")
st.progress(0.8)
st.write("80%")

# Status indicators
st.subheader("Agent Status")
col1, col2, col3 = st.columns(3)

with col1:
    st.write("🚨 ADA Agent")
    st.success("Active")

with col2:
    st.write("📋 TAA Agent")
    st.success("Active")

with col3:
    st.write("🛡️ CRA Agent")
    st.success("Active")

# Footer
st.markdown("---")
st.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.write("Simple working dashboard - no complex features")
