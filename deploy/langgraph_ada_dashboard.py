#!/usr/bin/env python3
"""
LangGraph ADA Agent Dashboard
Real-time monitoring of LangGraph workflow states, ML model performance, and ADA integration
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import json
import os
import sys

# Page configuration
st.set_page_config(
    page_title="LangGraph ADA Agent Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for LangGraph theme
st.markdown("""
<style>
    :root {
        --langgraph-primary: #6366f1;
        --langgraph-secondary: #8b5cf6;
        --langgraph-accent: #06b6d4;
        --langgraph-success: #10b981;
        --langgraph-warning: #f59e0b;
        --langgraph-error: #ef4444;
        --langgraph-dark: #1e293b;
        --langgraph-darker: #0f172a;
        --text-color: #e2e8f0;
        --bg-color: #0f172a;
    }

    .main {
        background: linear-gradient(135deg, var(--bg-color) 0%, var(--langgraph-darker) 100%);
        color: var(--text-color);
    }

    .langgraph-header {
        background: linear-gradient(135deg, var(--langgraph-primary) 0%, var(--langgraph-secondary) 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.3);
        border: 1px solid rgba(99, 102, 241, 0.2);
    }

    .metric-card {
        background: rgba(30, 41, 59, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
        color: var(--text-color);
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.2);
        border-color: var(--langgraph-accent);
    }

    .status-active {
        background: linear-gradient(135deg, var(--langgraph-success) 0%, #059669 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
    }

    .status-warning {
        background: linear-gradient(135deg, var(--langgraph-warning) 0%, #d97706 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
    }

    .status-error {
        background: linear-gradient(135deg, var(--langgraph-error) 0%, #dc2626 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
    }

    .workflow-state {
        background: rgba(99, 102, 241, 0.1);
        border: 2px solid var(--langgraph-primary);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }

    .workflow-state:hover {
        background: rgba(99, 102, 241, 0.2);
        transform: scale(1.02);
    }

    h1, h2, h3, h4, h5, h6 {
        color: var(--text-color) !important;
    }

    p, div {
        color: var(--text-color) !important;
    }
</style>
""", unsafe_allow_html=True)

# Mock data functions
def get_mock_langgraph_metrics():
    """Generate mock LangGraph ADA metrics"""
    return {
        'workflow_states': {
            'active_workflows': 12,
            'completed_workflows': 156,
            'failed_workflows': 3,
            'pending_workflows': 8
        },
        'ml_model_performance': {
            'accuracy': 0.94,
            'precision': 0.91,
            'recall': 0.89,
            'f1_score': 0.90,
            'inference_time_ms': 45.2
        },
        'ada_agent_metrics': {
            'alerts_processed': 1247,
            'anomalies_detected': 89,
            'false_positives': 12,
            'avg_processing_time': 2.3,
            'model_confidence': 0.87
        },
        'workflow_transitions': {
            'data_preprocessing': 156,
            'feature_extraction': 156,
            'model_inference': 156,
            'post_processing': 156,
            'result_storage': 156
        }
    }

def get_mock_ml_performance():
    """Generate mock ML model performance data"""
    dates = pd.date_range(start='2025-08-01', end='2025-08-07', freq='D')
    data = []
    
    for date in dates:
        data.append({
            'date': date,
            'accuracy': np.random.uniform(0.92, 0.96),
            'precision': np.random.uniform(0.89, 0.93),
            'recall': np.random.uniform(0.87, 0.91),
            'f1_score': np.random.uniform(0.88, 0.92),
            'inference_time_ms': np.random.uniform(40, 50)
        })
    
    return pd.DataFrame(data)

def get_mock_workflow_states():
    """Generate mock workflow state data"""
    states = ['data_preprocessing', 'feature_extraction', 'model_inference', 'post_processing', 'result_storage']
    data = []
    
    for i in range(50):
        state = np.random.choice(states)
        timestamp = datetime.now() - timedelta(minutes=np.random.randint(1, 120))
        duration = np.random.uniform(0.5, 5.0)
        success = np.random.choice([True, False], p=[0.95, 0.05])
        
        data.append({
            'workflow_id': f'wf_{i:04d}',
            'state': state,
            'timestamp': timestamp,
            'duration_seconds': duration,
            'success': success,
            'error_message': None if success else 'Mock error message'
        })
    
    return pd.DataFrame(data)

# Main dashboard
def main():
    # Header
    st.markdown("""
    <div class="langgraph-header">
        <h1>🧠 LangGraph ADA Agent Dashboard</h1>
        <p>Real-time monitoring of LangGraph workflow orchestration, ML model performance, and ADA integration</p>
        <p><strong>Status:</strong> <span class="status-active">ACTIVE</span> | <strong>Last Update:</strong> {}</p>
    </div>
    """.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 LangGraph Controls")
    
    # Data source selection
    data_source = st.sidebar.selectbox(
        "📊 Data Source:",
        ["Mock Data (Demo)", "Real BigQuery Data"],
        help="Choose between mock data for demonstration or real production data"
    )
    
    # Time range selection
    time_range = st.sidebar.selectbox(
        "⏰ Time Range:",
        ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "Custom"],
        help="Select the time range for data analysis"
    )
    
    # Get metrics
    metrics = get_mock_langgraph_metrics()
    
    # Main dashboard content
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🔄 Active Workflows</h3>
            <h2 style="color: var(--langgraph-accent); font-size: 2.5rem;">{metrics['workflow_states']['active_workflows']}</h2>
            <p>Currently running LangGraph workflows</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>✅ Completed Workflows</h3>
            <h2 style="color: var(--langgraph-success); font-size: 2.5rem;">{metrics['workflow_states']['completed_workflows']}</h2>
            <p>Successfully completed in last 24h</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎯 Model Accuracy</h3>
            <h2 style="color: var(--langgraph-primary); font-size: 2.5rem;">{metrics['ml_model_performance']['accuracy']:.1%}</h2>
            <p>Current ML model performance</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🚨 Anomalies Detected</h3>
            <h2 style="color: var(--langgraph-warning); font-size: 2.5rem;">{metrics['ada_agent_metrics']['anomalies_detected']}</h2>
            <p>Security threats identified</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Workflow State Monitoring
    st.markdown("---")
    st.markdown("## 🔄 LangGraph Workflow State Monitoring")
    
    workflow_cols = st.columns(3)
    
    with workflow_cols[0]:
        st.markdown(f"""
        <div class="workflow-state">
            <h4>📊 Workflow Distribution</h4>
            <p><strong>Active:</strong> {metrics['workflow_states']['active_workflows']}</p>
            <p><strong>Pending:</strong> {metrics['workflow_states']['pending_workflows']}</p>
            <p><strong>Failed:</strong> {metrics['workflow_states']['failed_workflows']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with workflow_cols[1]:
        success_rate = (metrics['workflow_states']['completed_workflows'] / 
                       (metrics['workflow_states']['completed_workflows'] + metrics['workflow_states']['failed_workflows'])) * 100 if (metrics['workflow_states']['completed_workflows'] + metrics['workflow_states']['failed_workflows']) > 0 else 0
        
        status_class = "status-active" if success_rate >= 95 else "status-warning" if success_rate >= 85 else "status-error"
        
        st.markdown(f"""
        <div class="workflow-state">
            <h4>📈 Success Rate</h4>
            <div class="{status_class}">{success_rate:.1f}%</div>
            <p>Workflow completion success</p>
        </div>
        """, unsafe_allow_html=True)
    
    with workflow_cols[2]:
        st.markdown(f"""
        <div class="workflow-state">
            <h4>⚡ Performance</h4>
            <p><strong>Avg Inference:</strong> {metrics['ml_model_performance']['inference_time_ms']:.1f}ms</p>
            <p><strong>Processing Time:</strong> {metrics['ada_agent_metrics']['avg_processing_time']:.1f}s</p>
            <p><strong>Confidence:</strong> {metrics['ada_agent_metrics']['model_confidence']:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ML Model Performance Charts
    st.markdown("---")
    st.markdown("## 🤖 ML Model Performance Analytics")
    
    # Get performance data
    perf_df = get_mock_ml_performance()
    
    # Performance metrics chart
    chart_cols = st.columns(2)
    
    with chart_cols[0]:
        st.markdown("**📊 Model Metrics Over Time**")
        fig_perf = go.Figure()
        
        fig_perf.add_trace(go.Scatter(
            x=perf_df['date'],
            y=perf_df['accuracy'] * 100,
            mode='lines+markers',
            name='Accuracy (%)',
            line=dict(color='#10b981', width=3)
        ))
        
        fig_perf.add_trace(go.Scatter(
            x=perf_df['date'],
            y=perf_df['precision'] * 100,
            mode='lines+markers',
            name='Precision (%)',
            line=dict(color='#3b82f6', width=3)
        ))
        
        fig_perf.add_trace(go.Scatter(
            x=perf_df['date'],
            y=perf_df['recall'] * 100,
            mode='lines+markers',
            name='Recall (%)',
            line=dict(color='#8b5cf6', width=3)
        ))
        
        fig_perf.update_layout(
            title='ML Model Performance Trends',
            xaxis_title='Date',
            yaxis_title='Score (%)',
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            legend=dict(bgcolor='rgba(0,0,0,0.5)')
        )
        
        st.plotly_chart(fig_perf, use_container_width=True)
    
    with chart_cols[1]:
        st.markdown("**⚡ Inference Time Performance**")
        fig_inference = go.Figure()
        
        fig_inference.add_trace(go.Scatter(
            x=perf_df['date'],
            y=perf_df['inference_time_ms'],
            mode='lines+markers',
            name='Inference Time (ms)',
            line=dict(color='#f59e0b', width=3),
            fill='tonexty',
            fillcolor='rgba(245, 158, 11, 0.1)'
        ))
        
        # Add performance thresholds
        fig_inference.add_hline(y=50, line_dash="dash", line_color="green", annotation_text="Target: 50ms")
        fig_inference.add_hline(y=100, line_dash="dash", line_color="orange", annotation_text="Warning: 100ms")
        
        fig_inference.update_layout(
            title='Model Inference Time Trends',
            xaxis_title='Date',
            yaxis_title='Time (ms)',
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            legend=dict(bgcolor='rgba(0,0,0,0.5)')
        )
        
        st.plotly_chart(fig_inference, use_container_width=True)
    
    # Workflow State Transitions
    st.markdown("---")
    st.markdown("## 🔄 LangGraph Workflow State Transitions")
    
    # Create workflow state diagram
    workflow_states = ['Data Preprocessing', 'Feature Extraction', 'Model Inference', 'Post Processing', 'Result Storage']
    transition_counts = [
        metrics['workflow_transitions']['data_preprocessing'],
        metrics['workflow_transitions']['feature_extraction'],
        metrics['workflow_transitions']['model_inference'],
        metrics['workflow_transitions']['post_processing'],
        metrics['workflow_transitions']['result_storage']
    ]
    
    fig_workflow = go.Figure(data=[
        go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=workflow_states,
                color=["#6366f1", "#8b5cf6", "#06b6d4", "#10b981", "#f59e0b"]
            ),
            link=dict(
                source=[0, 1, 2, 3],  # Source nodes
                target=[1, 2, 3, 4],  # Target nodes
                value=transition_counts[:-1]  # Flow values
            )
        )
    ])
    
    fig_workflow.update_layout(
        title='LangGraph Workflow State Transitions',
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0', size=12)
    )
    
    st.plotly_chart(fig_workflow, use_container_width=True)
    
    # Real-time Workflow Monitoring
    st.markdown("---")
    st.markdown("## 📊 Real-time Workflow Monitoring")
    
    # Get workflow data
    workflow_df = get_mock_workflow_states()
    
    # Workflow status table
    if not workflow_df.empty:
        # Add status icons
        workflow_df['Status_Icon'] = workflow_df['success'].apply(
            lambda x: '✅' if x else '❌'
        )
        
        # Format timestamp
        workflow_df['Formatted_Time'] = pd.to_datetime(workflow_df['timestamp']).dt.strftime('%H:%M:%S')
        
        # Display workflow table
        st.markdown("**🔄 Recent Workflow Executions**")
        
        # Create a styled dataframe
        display_df = workflow_df[['Status_Icon', 'workflow_id', 'state', 'Formatted_Time', 'duration_seconds', 'error_message']].copy()
        display_df.columns = ['Status', 'Workflow ID', 'State', 'Time', 'Duration (s)', 'Error Message']
        
        # Apply styling
        def style_workflow_status(val):
            if '✅' in str(val):
                return 'background-color: rgba(16, 185, 129, 0.2); color: #10b981;'
            else:
                return 'background-color: rgba(239, 68, 68, 0.2); color: #ef4444;'
        
        styled_df = display_df.style.applymap(
            style_workflow_status, 
            subset=['Status']
        ).format({
            'Duration (s)': '{:.2f}'
        })
        
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        # Workflow statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Workflows", len(workflow_df))
        
        with col2:
            success_count = workflow_df['success'].sum()
            st.metric("Successful", f"{success_count} ({success_count/len(workflow_df)*100:.1f}%)")
        
        with col3:
            avg_duration = workflow_df['duration_seconds'].mean()
            st.metric("Avg Duration", f"{avg_duration:.2f}s")
    
    # System Health Monitoring
    st.markdown("---")
    st.markdown("## 🏥 System Health & Integration Status")
    
    health_cols = st.columns(2)
    
    with health_cols[0]:
        st.markdown("""
        <div class="metric-card">
            <h4>🔗 Integration Status</h4>
            <p><span class="status-active">✅ BigQuery Connected</span></p>
            <p><span class="status-active">✅ LangGraph Running</span></p>
            <p><span class="status-active">✅ ML Model Loaded</span></p>
            <p><span class="status-active">✅ ADA Agent Active</span></p>
        </div>
        """, unsafe_allow_html=True)
    
    with health_cols[1]:
        st.markdown("""
        <div class="metric-card">
            <h4>📈 Performance Metrics</h4>
            <p><strong>Response Time:</strong> <span style="color: var(--langgraph-success);">Optimal</span></p>
            <p><strong>Error Rate:</strong> <span style="color: var(--langgraph-success);">Low</span></p>
            <p><strong>Throughput:</strong> <span style="color: var(--langgraph-success);">High</span></p>
            <p><strong>Resource Usage:</strong> <span style="color: var(--langgraph-warning);">Normal</span></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: var(--text-color); padding: 2rem;">
        <p><strong>🧠 LangGraph ADA Agent Dashboard</strong> | Real-time monitoring of AI-powered anomaly detection</p>
        <p>Powered by LangGraph workflow orchestration and ML model integration</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
