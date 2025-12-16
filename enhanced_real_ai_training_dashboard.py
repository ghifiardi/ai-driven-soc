"""
Enhanced Real AI Model Training Dashboard
Uses actual BigQuery data from the AI-driven SOC system
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from enhanced_real_data_connector import enhanced_real_data_connector

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
    .training-status {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
        margin: 0.5rem 0;
    }
    .status-active {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .status-completed {
        background-color: #d1ecf1;
        color: #0c5460;
        border: 1px solid #bee5eb;
    }
    .status-high-anomaly {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    .status-low-confidence {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .status-no-data {
        background-color: #e2e3e5;
        color: #383d41;
        border: 1px solid #d6d8db;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🤖 Real AI Model Training Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Live Data from AI-Driven SOC System</p>', unsafe_allow_html=True)

# Sidebar for controls
st.sidebar.header("📊 Dashboard Controls")
time_range = st.sidebar.selectbox(
    "Time Range",
    ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "Last 90 Days", "All Time"]
)

model_type = st.sidebar.selectbox(
    "Model Type",
    ["All Models", "ADA Model", "TAA Model", "Combined Analysis"]
)

# Data refresh button
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# Load real data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_real_data():
    """Load real data from BigQuery"""
    try:
        # Get real training metrics
        metrics, _ = enhanced_real_data_connector.get_real_training_metrics()
        
        # Get real performance data
        performance_df = enhanced_real_data_connector.get_real_model_performance()
        
        # Get real training sessions
        sessions_df = enhanced_real_data_connector.get_real_training_sessions()
        
        # Get real resource utilization
        resource_metrics = enhanced_real_data_connector.get_real_resource_utilization()
        
        # Get real model comparison
        comparison_df = enhanced_real_data_connector.get_real_model_comparison()
        
        # Get real data quality metrics
        quality_metrics = enhanced_real_data_connector.get_real_data_quality_metrics()
        
        return {
            'metrics': metrics,
            'performance_df': performance_df,
            'sessions_df': sessions_df,
            'resource_metrics': resource_metrics,
            'comparison_df': comparison_df,
            'quality_metrics': quality_metrics
        }
    except Exception as e:
        st.error(f"Error loading real data: {e}")
        return None

# Load data
data = load_real_data()

if data is None:
    st.error("❌ Unable to load real data. Please check BigQuery connection.")
    st.stop()

metrics = data['metrics']
performance_df = data['performance_df']
sessions_df = data['sessions_df']
resource_metrics = data['resource_metrics']
comparison_df = data['comparison_df']
quality_metrics = data['quality_metrics']

# Main dashboard layout
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Alerts Processed</div>
        <div class="metric-value">{metrics['total_alerts']:,}</div>
        <div class="metric-label">Real SOC Data</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">ADA Model Accuracy</div>
        <div class="metric-value">{metrics['model_accuracy']:.1f}%</div>
        <div class="metric-label">Current Performance</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">High Confidence Rate</div>
        <div class="metric-value">{metrics['high_confidence_rate']:.1f}%</div>
        <div class="metric-label">Reliable Predictions</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Automation Rate</div>
        <div class="metric-value">{metrics['automation_rate']:.1f}%</div>
        <div class="metric-label">Auto-Processed Alerts</div>
    </div>
    """, unsafe_allow_html=True)

# Real Performance Section
st.header("📊 Real Model Performance Analysis")

col1, col2 = st.columns([2, 1])

with col1:
    if not performance_df.empty:
        # Real performance over time
        fig_performance = px.line(
            performance_df, 
            x='date', 
            y=['accuracy', 'automation_rate', 'true_positive_rate'],
            title='Real Model Performance Over Time',
            labels={'value': 'Performance %', 'variable': 'Metric'},
            color_discrete_map={
                'accuracy': '#1f77b4', 
                'automation_rate': '#ff7f0e',
                'true_positive_rate': '#2ca02c'
            }
        )
        fig_performance.update_layout(height=400)
        st.plotly_chart(fig_performance, use_container_width=True)
    else:
        st.info("📊 No performance data available for the selected time range")

with col2:
    # Current real metrics
    st.subheader("Current Real Metrics")
    
    st.metric("Total Alerts", f"{metrics['total_alerts']:,}")
    st.metric("Unique Alarms", f"{metrics['unique_alarms']:,}")
    st.metric("ADA Confidence", f"{metrics['avg_ada_confidence']:.3f}")
    st.metric("TAA Confidence", f"{metrics['avg_taa_confidence']:.3f}")
    st.metric("Malicious Rate", f"{metrics['malicious_rate']:.1f}%")
    st.metric("False Positive Rate", f"{metrics['false_positive_rate']:.1f}%")

# Real Resource Utilization
st.header("⚡ Real Resource Utilization")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("CPU Usage", f"{resource_metrics['cpu_usage']:.1f}%")
    st.metric("Memory Usage", f"{resource_metrics['memory_usage']:.1f}%")

with col2:
    st.metric("GPU Usage", f"{resource_metrics['gpu_usage']:.1f}%")
    st.metric("Data Volume", f"{resource_metrics['data_volume']:,} alerts")

with col3:
    # Resource utilization chart
    resource_data = pd.DataFrame({
        'Resource': ['CPU', 'Memory', 'GPU'],
        'Usage %': [resource_metrics['cpu_usage'], resource_metrics['memory_usage'], resource_metrics['gpu_usage']]
    })
    
    fig_resources = px.bar(
        resource_data,
        x='Resource',
        y='Usage %',
        title='Current Resource Utilization',
        color='Usage %',
        color_continuous_scale='RdYlGn_r'
    )
    fig_resources.update_layout(height=300)
    st.plotly_chart(fig_resources, use_container_width=True)

# Real Model Comparison
st.header("🔬 Real Model Comparison (ADA vs TAA)")

if not comparison_df.empty:
    col1, col2 = st.columns(2)
    
    with col1:
        # Model accuracy comparison
        fig_accuracy = px.bar(
            comparison_df,
            x='Model Version',
            y='Accuracy',
            title='Real Model Accuracy Comparison',
            color='Model Version',
            color_discrete_map={'ADA Model': '#1f77b4', 'TAA Model': '#ff7f0e'}
        )
        fig_accuracy.update_layout(height=350)
        st.plotly_chart(fig_accuracy, use_container_width=True)
    
    with col2:
        # Model automation rate comparison
        fig_automation = px.bar(
            comparison_df,
            x='Model Version',
            y='Automation Rate',
            title='Real Model Automation Rate',
            color='Model Version',
            color_discrete_map={'ADA Model': '#1f77b4', 'TAA Model': '#ff7f0e'}
        )
        fig_automation.update_layout(height=350)
        st.plotly_chart(fig_automation, use_container_width=True)
    
    # Detailed comparison table
    st.subheader("📋 Detailed Model Performance Comparison")
    display_comparison = comparison_df[['Model Version', 'Accuracy', 'Automation Rate', 'False Positive Rate', 'Response Time (min)', 'Score']].copy()
    display_comparison['Accuracy'] = display_comparison['Accuracy'].round(2)
    display_comparison['Automation Rate'] = display_comparison['Automation Rate'].round(2)
    display_comparison['False Positive Rate'] = display_comparison['False Positive Rate'].round(2)
    display_comparison['Response Time (min)'] = display_comparison['Response Time (min)'].round(2)
    display_comparison['Score'] = display_comparison['Score'].round(3)
    
    st.dataframe(display_comparison, use_container_width=True)
else:
    st.info("📊 No model comparison data available")

# Real Data Quality Metrics
st.header("📈 Real Data Quality Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Data Quality Score", f"{quality_metrics['data_quality_score']:.1f}%")
    st.metric("Label Accuracy", f"{quality_metrics['label_accuracy']:.1f}%")
    st.metric("Data Completeness", f"{quality_metrics['data_completeness']:.1f}%")

with col2:
    st.metric("Feature Coverage", f"{quality_metrics['feature_coverage']:.1f}%")
    st.metric("Data Freshness", f"{quality_metrics['data_freshness']:.1f} days")
    st.metric("Outlier Detection", f"{quality_metrics['outlier_detection']:.1f}%")

with col3:
    st.metric("Data Diversity", f"{quality_metrics['data_diversity']:.1f}%")
    st.metric("Class Balance", f"{quality_metrics['class_balance']:.1f}%")
    st.metric("Validation Split", quality_metrics['validation_split'])

# Real Training Sessions
st.header("📋 Real Training Sessions (Last 7 Days)")

if not sessions_df.empty:
    # Color code the status
    def color_status(val):
        if val == 'completed':
            return 'background-color: #d1ecf1'
        elif val == 'high_anomaly':
            return 'background-color: #fff3cd'
        elif val == 'low_confidence':
            return 'background-color: #f8d7da'
        else:
            return 'background-color: #e2e3e5'
    
    st.dataframe(
        sessions_df.style.applymap(color_status, subset=['Status']), 
        use_container_width=True
    )
    
    # Training sessions summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_samples = sessions_df['Samples'].sum()
        st.metric("Total Samples Processed", f"{total_samples:,}")
    
    with col2:
        total_patterns = sessions_df['Patterns'].sum()
        st.metric("Unique Patterns Detected", f"{total_patterns:,}")
    
    with col3:
        avg_confidence = sessions_df['Confidence'].str.rstrip('%').astype(float).mean()
        st.metric("Average Confidence", f"{avg_confidence:.1f}%")
        
else:
    st.info("📊 No training session data available for the selected time range")

# Real-time Data Status
st.header("🔄 Real-time Data Status")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.success("✅ BigQuery Connected")
    st.caption("Real data source")

with col2:
    if metrics['total_alerts'] > 0:
        st.success("✅ Data Available")
        st.caption(f"{metrics['total_alerts']:,} records")
    else:
        st.warning("⚠️ No Data")
        st.caption("Check data pipeline")

with col3:
    if metrics['model_accuracy'] > 0:
        st.success("✅ Models Active")
        st.caption("Processing alerts")
    else:
        st.warning("⚠️ Models Inactive")
        st.caption("No processing")

with col4:
    st.info("📊 Live Dashboard")
    st.caption("Auto-refresh: 5min")

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🤖 Real AI Model Training Dashboard | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p>This dashboard displays real-time data from the AI-driven SOC system BigQuery tables.</p>
    <p><strong>Data Sources:</strong> dashboard_alerts, taa_state, siem_events tables</p>
</div>
""", unsafe_allow_html=True)

