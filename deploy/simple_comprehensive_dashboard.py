import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="AI Models Dashboard",
    page_icon="🤖",
    layout="wide"
)

# Main header
st.title("🤖 Comprehensive AI Models Dashboard")
st.subheader("CRA + ADA + TAA - All in One View")

# Sidebar
st.sidebar.header("Controls")
model_focus = st.sidebar.selectbox("Focus Model", ["All", "CRA", "ADA", "TAA"])

# Mock data for quick display
def get_cra_data():
    return {
        'version': 'v5.2.1',
        'accuracy': 89.7,
        'automation': 41.6,
        'samples': 127842,
        'status': 'Active Learning'
    }

def get_ada_data():
    return {
        'alerts': 156,
        'anomalies': 23,
        'confidence': 67,
        'accuracy': 89.3,
        'status': 'Operational'
    }

def get_taa_data():
    return {
        'processed': 1273,
        'processing_rate': 90.7,
        'accuracy': 94.1,
        'efficiency': 87.3,
        'status': 'Active'
    }

# Get data
cra = get_cra_data()
ada = get_ada_data()
taa = get_taa_data()

# Executive Summary
st.header("📊 Executive Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("CRA Version", cra['version'], "🔄 " + cra['status'])
    st.metric("Accuracy", f"{cra['accuracy']}%", "+2.1%")
    st.metric("Automation", f"{cra['automation']}%", "+5.3%")

with col2:
    st.metric("ADA Alerts", ada['alerts'], "🧠 " + ada['status'])
    st.metric("Anomalies", ada['anomalies'], "+3 today")
    st.metric("Confidence", f"{ada['confidence']}%", "+2.8%")

with col3:
    st.metric("TAA Processed", taa['processed'], "🔍 " + taa['status'])
    st.metric("Processing Rate", f"{taa['processing_rate']}%", "+1.2%")
    st.metric("Accuracy", f"{taa['accuracy']}%", "+1.8%")

# Model Performance Comparison
st.header("🔬 Model Performance Comparison")

# Create comparison chart
comparison_data = pd.DataFrame({
    'Model': ['CRA', 'ADA', 'TAA'],
    'Accuracy': [cra['accuracy'], ada['accuracy'], taa['accuracy']],
    'Efficiency': [cra['automation'], ada['confidence'], taa['efficiency']]
})

fig = px.bar(
    comparison_data,
    x='Model',
    y=['Accuracy', 'Efficiency'],
    title="Model Performance Comparison",
    barmode='group'
)

fig.update_layout(
    height=400,
    plot_bgcolor='#1e1e1e',
    paper_bgcolor='#1e1e1e',
    font=dict(color='white')
)

st.plotly_chart(fig, use_container_width=True)

# Quick Status Overview
st.header("📋 Quick Status Overview")

status_data = pd.DataFrame({
    'Model': ['CRA', 'ADA', 'TAA'],
    'Status': [cra['status'], ada['status'], taa['status']],
    'Key Metric': [f"{cra['accuracy']}% Accuracy", f"{ada['anomalies']} Anomalies", f"{taa['processing_rate']}% Rate"],
    'Last Update': ['2 min ago', '5 min ago', '1 min ago']
})

st.dataframe(status_data, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

st.success("✅ Comprehensive AI Models Dashboard loaded successfully!")
