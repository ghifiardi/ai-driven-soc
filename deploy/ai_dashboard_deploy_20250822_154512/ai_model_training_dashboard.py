import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from datetime import timedelta
import random

# Page configuration
st.set_page_config(
    page_title="AI Model Training Dashboard",
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
    .status-failed {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">AI Model Training Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Continuous Learning & Model Performance Monitoring</p>', unsafe_allow_html=True)

# Sidebar for controls
st.sidebar.header("Dashboard Controls")
time_range = st.sidebar.selectbox(
    "Time Range",
    ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "Last 90 Days", "All Time"]
)

model_type = st.sidebar.selectbox(
    "Model Type",
    ["All Models", "Threat Detection", "False Positive Filter", "Incident Classification", "Response Automation"]
)

# Generate sample data for demonstration
def generate_training_data():
    # Model training history
    dates = pd.date_range(start=datetime.datetime.now() - timedelta(days=90), end=datetime.datetime.now(), freq='D')
    
    training_data = []
    for date in dates:
        # Simulate realistic training metrics
        base_accuracy = 0.75 + random.uniform(-0.05, 0.05)
        training_data.append({
            'date': date,
            'model_version': f"v{random.randint(1, 5)}.{random.randint(0, 9)}.{random.randint(0, 9)}",
            'training_samples': random.randint(5000, 15000),
            'training_accuracy': base_accuracy,
            'validation_accuracy': base_accuracy + random.uniform(-0.02, 0.02),
            'training_loss': random.uniform(0.1, 0.3),
            'validation_loss': random.uniform(0.12, 0.35),
            'training_time_minutes': random.randint(45, 180),
            'gpu_utilization': random.uniform(0.7, 0.95),
            'memory_usage_gb': random.uniform(8, 16),
            'status': random.choices(['completed', 'active', 'failed'], weights=[0.8, 0.15, 0.05])[0]
        })
    
    return pd.DataFrame(training_data)

# Generate performance improvement data
def generate_performance_data():
    dates = pd.date_range(start=datetime.datetime.now() - timedelta(days=90), end=datetime.datetime.now(), freq='D')
    
    performance_data = []
    base_automation = 0.35
    base_accuracy = 0.68
    
    for i, date in enumerate(dates):
        # Simulate gradual improvement over time
        improvement_factor = i / len(dates)
        performance_data.append({
            'date': date,
            'automation_rate': base_automation + (improvement_factor * 0.15) + random.uniform(-0.02, 0.02),
            'detection_accuracy': base_accuracy + (improvement_factor * 0.12) + random.uniform(-0.01, 0.01),
            'false_positive_rate': 0.85 - (improvement_factor * 0.15) + random.uniform(-0.02, 0.02),
            'response_time_minutes': 12 - (improvement_factor * 4) + random.uniform(-0.5, 0.5)
        })
    
    return pd.DataFrame(performance_data)

# Load data
training_df = generate_training_data()
performance_df = generate_performance_data()

# Filter data based on sidebar selections
if model_type != "All Models":
    # In a real scenario, you'd filter by actual model type
    pass

# Main dashboard layout
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Current Model Version</div>
        <div class="metric-value">v5.2.1</div>
        <div class="metric-label">Latest Stable Release</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Training Status</div>
        <div class="metric-value">🔄 Active</div>
        <div class="metric-label">Incremental Learning</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Total Training Samples</div>
        <div class="metric-value">127,842</div>
        <div class="metric-label">Last 30 Days</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Model Accuracy</div>
        <div class="metric-value">89.7%</div>
        <div class="metric-label">Current Best</div>
    </div>
    """, unsafe_allow_html=True)

# Training Progress Section
st.header("📊 Training Progress & Performance")

col1, col2 = st.columns([2, 1])

with col1:
    # Training accuracy over time
    fig_accuracy = px.line(
        training_df, 
        x='date', 
        y=['training_accuracy', 'validation_accuracy'],
        title='Model Accuracy Over Time',
        labels={'value': 'Accuracy', 'variable': 'Dataset'},
        color_discrete_map={'training_accuracy': '#1f77b4', 'validation_accuracy': '#ff7f0e'}
    )
    fig_accuracy.update_layout(height=400)
    st.plotly_chart(fig_accuracy, use_container_width=True)

with col2:
    # Current training metrics
    st.subheader("Current Training Metrics")
    
    latest = training_df.iloc[-1]
    
    st.metric("Training Accuracy", f"{latest['training_accuracy']:.1%}")
    st.metric("Validation Accuracy", f"{latest['validation_accuracy']:.1%}")
    st.metric("Training Loss", f"{latest['training_loss']:.3f}")
    st.metric("Validation Loss", f"{latest['validation_loss']:.3f}")
    
    # Training status indicator
    status_color = {
        'completed': 'status-completed',
        'active': 'status-active',
        'failed': 'status-failed'
    }
    
    st.markdown(f"""
    <div class="training-status {status_color[latest['status']]}">
        Status: {latest['status'].title()}
    </div>
    """, unsafe_allow_html=True)

# Performance Improvement Section
st.header("🚀 Performance Improvements Over Time")

col1, col2 = st.columns(2)

with col1:
    # Automation rate improvement
    fig_automation = px.line(
        performance_df,
        x='date',
        y='automation_rate',
        title='AI Automation Rate Improvement',
        labels={'automation_rate': 'Automation Rate', 'date': 'Date'}
    )
    fig_automation.update_layout(height=350)
    st.plotly_chart(fig_automation, use_container_width=True)

with col2:
    # False positive rate reduction
    fig_fp = px.line(
        performance_df,
        x='date',
        y='false_positive_rate',
        title='False Positive Rate Reduction',
        labels={'false_positive_rate': 'False Positive Rate', 'date': 'Date'}
    )
    fig_fp.update_layout(height=350)
    st.plotly_chart(fig_fp, use_container_width=True)

# Training Resource Utilization
st.header("⚡ Training Resource Utilization")

col1, col2 = st.columns(2)

with col1:
    # GPU utilization over time
    fig_gpu = px.line(
        training_df,
        x='date',
        y='gpu_utilization',
        title='GPU Utilization During Training',
        labels={'gpu_utilization': 'GPU Utilization %', 'date': 'Date'}
    )
    fig_gpu.update_layout(height=350)
    st.plotly_chart(fig_gpu, use_container_width=True)

with col2:
    # Memory usage over time
    fig_memory = px.line(
        training_df,
        x='date',
        y='memory_usage_gb',
        title='Memory Usage During Training',
        labels={'memory_usage_gb': 'Memory Usage (GB)', 'date': 'Date'}
    )
    fig_memory.update_layout(height=350)
    st.plotly_chart(fig_memory, use_container_width=True)

# Model Comparison & A/B Testing
st.header("🔬 Model Comparison & A/B Testing")

# Generate A/B testing data
ab_test_data = pd.DataFrame({
    'Model Version': ['v4.1.0', 'v4.2.0', 'v5.0.0', 'v5.1.0', 'v5.2.1'],
    'Accuracy': [0.82, 0.85, 0.87, 0.89, 0.90],
    'Automation Rate': [0.28, 0.32, 0.38, 0.41, 0.44],
    'False Positive Rate': [0.78, 0.75, 0.72, 0.69, 0.66],
    'Response Time (min)': [9.2, 8.8, 8.1, 7.6, 7.2],
    'Training Time (hrs)': [2.1, 2.3, 2.8, 3.1, 3.4]
})

# A/B testing results
fig_ab = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Accuracy Comparison', 'Automation Rate', 'False Positive Rate', 'Response Time'),
    specs=[[{"secondary_y": False}, {"secondary_y": False}],
           [{"secondary_y": False}, {"secondary_y": False}]]
)

# Accuracy
fig_ab.add_trace(
    go.Bar(x=ab_test_data['Model Version'], y=ab_test_data['Accuracy'], name='Accuracy', marker_color='#1f77b4'),
    row=1, col=1
)

# Automation Rate
fig_ab.add_trace(
    go.Bar(x=ab_test_data['Model Version'], y=ab_test_data['Automation Rate'], name='Automation Rate', marker_color='#ff7f0e'),
    row=1, col=2
)

# False Positive Rate
fig_ab.add_trace(
    go.Bar(x=ab_test_data['Model Version'], y=ab_test_data['False Positive Rate'], name='False Positive Rate', marker_color='#d62728'),
    row=2, col=1
)

# Response Time
fig_ab.add_trace(
    go.Bar(x=ab_test_data['Model Version'], y=ab_test_data['Response Time (min)'], name='Response Time', marker_color='#2ca02c'),
    row=2, col=2
)

fig_ab.update_layout(height=600, title_text="Model Version Performance Comparison")
st.plotly_chart(fig_ab, use_container_width=True)

# Training Data Quality Metrics
st.header("📈 Training Data Quality Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Data Quality Score", "94.2%", "+2.1%")
    st.metric("Label Accuracy", "96.8%", "+1.5%")
    st.metric("Data Completeness", "91.3%", "+3.2%")

with col2:
    st.metric("Feature Coverage", "89.7%", "+2.8%")
    st.metric("Data Freshness", "2.3 days", "-0.5 days")
    st.metric("Outlier Detection", "87.4%", "+4.1%")

with col3:
    st.metric("Data Diversity", "92.1%", "+1.8%")
    st.metric("Class Balance", "88.9%", "+2.3%")
    st.metric("Validation Split", "80/20", "Optimal")

# Recent Training Sessions
st.header("📋 Recent Training Sessions")

# Filter recent training sessions
recent_sessions = training_df.tail(10)[['date', 'model_version', 'training_samples', 'training_accuracy', 'training_time_minutes', 'status']]

# Format the data for display
recent_sessions['date'] = recent_sessions['date'].dt.strftime('%Y-%m-%d %H:%M')
recent_sessions['training_accuracy'] = recent_sessions['training_accuracy'].apply(lambda x: f"{x:.1%}")
recent_sessions['training_time_minutes'] = recent_sessions['training_time_minutes'].apply(lambda x: f"{x} min")

# Rename columns for display
recent_sessions.columns = ['Date', 'Model Version', 'Samples', 'Accuracy', 'Duration', 'Status']

# Color code the status
def color_status(val):
    if val == 'completed':
        return 'background-color: #d4edda'
    elif val == 'active':
        return 'background-color: #d1ecf1'
    else:
        return 'background-color: #f8d7da'

st.dataframe(recent_sessions.style.map(color_status, subset=['Status']), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🤖 AI Model Training Dashboard | Last Updated: {}</p>
    <p>This dashboard monitors continuous learning and model performance metrics for the AI-driven SOC system.</p>
</div>
""".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)
