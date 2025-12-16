#!/usr/bin/env python3
"""
Hybrid ADA Dashboard - Shows ML + AI Performance with Hybrid Decision Engine
Real-time monitoring of hybrid anomaly detection system
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import asyncio
import time

# Import the hybrid decision engine
from hybrid_ada_decision_engine import HybridADAWorkflow, HybridDecisionResult

# Page config
st.set_page_config(
    page_title="Hybrid ADA Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme toggle in sidebar
st.sidebar.markdown("## 🎨 Theme Selection")
theme_mode = st.sidebar.selectbox(
    "Choose Theme:",
    ["🌙 Dark Mode", "☀️ Light Mode"],
    index=0
)

# Dynamic CSS based on theme selection
if "🌙 Dark Mode" in theme_mode:
    st.markdown("""
    <style>
        .main { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); }
        .metric-card { background: rgba(30, 41, 59, 0.95); border: 2px solid #6366f1; color: #f8fafc; }
        .section-header { background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); color: white; }
        h1, h2, h3, h4, h5, h6 { color: #f8fafc !important; }
        p, div, span { color: #f8fafc !important; }
        .stMetric { background: rgba(30, 41, 59, 0.9) !important; color: #f8fafc !important; }
        .hybrid-card { background: rgba(30, 41, 59, 0.9); border: 2px solid #10b981; color: #f8fafc; }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        .main { background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%); }
        .metric-card { background: rgba(255, 255, 255, 0.95); border: 2px solid #6366f1; color: #1e293b; }
        .section-header { background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); color: white; }
        h1, h2, h3, h4, h5, h6 { color: #1e293b !important; }
        p, div, span { color: #1e293b !important; }
        .stMetric { background: rgba(255, 255, 255, 0.9) !important; color: #1e293b !important; }
        .hybrid-card { background: rgba(255, 255, 255, 0.9); border: 2px solid #10b981; color: #1e293b; }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown("""
<div class="section-header" style="padding: 20px; border-radius: 15px; margin: 20px 0; text-align: center;">
    <h1>🧠 Hybrid ADA Dashboard</h1>
    <p style="font-size: 18px; margin: 10px 0 0 0;">ML (Isolation Forest) + AI (Gemini) = Optimal Threat Detection</p>
</div>
""", unsafe_allow_html=True)

# Initialize hybrid workflow
@st.cache_resource
def get_hybrid_workflow():
    return HybridADAWorkflow()

workflow = get_hybrid_workflow()

# Sidebar controls
st.sidebar.markdown("## 🔧 Hybrid Controls")
st.sidebar.markdown("""
<div style="background: rgba(30, 41, 59, 0.9); padding: 15px; border-radius: 10px; border: 1px solid #6366f1;">
    <h4 style="color: #f8fafc; margin-bottom: 10px;">🧠 Hybrid Engine Status</h4>
    <p style="color: #10b981; font-weight: bold;">✅ Active</p>
    <p style="color: #94a3b8; font-size: 12px;">ML + AI Integration</p>
</div>
""", unsafe_allow_html=True)

# Test hybrid engine button
if st.sidebar.button("🚀 Test Hybrid Engine", type="primary"):
    st.sidebar.success("Testing hybrid engine...")
    
    # Sample test event
    test_event = {
        'event_id': f'evt_{int(time.time())}',
        'timestamp': datetime.now().isoformat(),
        'source_ip': '192.168.1.100',
        'destination_ip': '10.0.0.50',
        'event_type': 'network_scan',
        'severity': 'medium',
        'context': {
            'user_id': 'admin',
            'session_duration': 3600,
            'previous_events': 5,
            'geolocation': 'internal'
        }
    }
    
    # Process through hybrid engine
    with st.spinner("Processing through hybrid engine..."):
        try:
            result = asyncio.run(workflow.process_security_event(test_event))
            st.session_state.last_hybrid_result = result
            st.sidebar.success("✅ Hybrid test completed!")
        except Exception as e:
            st.sidebar.error(f"❌ Test failed: {e}")

# Data source toggle
use_real_data = st.sidebar.checkbox("📊 Use Real BigQuery Data", value=True)

# Performance metrics
performance_metrics = workflow.get_performance_metrics()

# Overall System Overview
st.markdown("---")
st.markdown("""
<div class="section-header" style="padding: 20px; border-radius: 15px; margin: 20px 0; text-align: center;">
    <h2>📊 Hybrid System Performance Overview</h2>
</div>
""", unsafe_allow_html=True)

overview_cols = st.columns(4)

with overview_cols[0]:
    st.markdown(f"""
    <div class="metric-card" style="padding: 25px; border-radius: 15px; margin: 15px 0; text-align: center;">
        <h3 style="color: #06b6d4; margin-bottom: 10px;">🤖 ML Accuracy</h3>
        <h1 style="font-size: 2.5rem; margin: 10px 0;">{performance_metrics['ml_accuracy']:.1%}</h1>
        <p style="margin: 0;">Isolation Forest</p>
    </div>
    """, unsafe_allow_html=True)

with overview_cols[1]:
    st.markdown(f"""
    <div class="metric-card" style="padding: 25px; border-radius: 15px; margin: 15px 0; text-align: center;">
        <h3 style="color: #8b5cf6; margin-bottom: 10px;">🧠 AI Accuracy</h3>
        <h1 style="font-size: 2.5rem; margin: 10px 0;">{performance_metrics['ai_accuracy']:.1%}</h1>
        <p style="margin: 0;">Gemini 2.0 Flash</p>
    </div>
    """, unsafe_allow_html=True)

with overview_cols[2]:
    st.markdown(f"""
    <div class="hybrid-card" style="padding: 25px; border-radius: 15px; margin: 15px 0; text-align: center;">
        <h3 style="color: #10b981; margin-bottom: 10px;">🚀 Hybrid Accuracy</h3>
        <h1 style="font-size: 2.5rem; margin: 10px 0;">{performance_metrics['hybrid_accuracy']:.1%}</h1>
        <p style="margin: 0;">ML + AI Combined</p>
    </div>
    """, unsafe_allow_html=True)

with overview_cols[3]:
    st.markdown(f"""
    <div class="metric-card" style="padding: 25px; border-radius: 15px; margin: 15px 0; text-align: center;">
        <h3 style="color: #f59e0b; margin-bottom: 10px;">⚡ Speed Improvement</h3>
        <h1 style="font-size: 2.5rem; margin: 10px 0;">{performance_metrics['detection_speed_improvement']:.1%}</h1>
        <p style="margin: 0;">vs. AI Only</p>
    </div>
    """, unsafe_allow_html=True)

# Processing Time Comparison
st.markdown("---")
st.markdown("""
<div class="section-header" style="padding: 20px; border-radius: 15px; margin: 20px 0; text-align: center;">
    <h2>⚡ Processing Time Comparison</h2>
</div>
""", unsafe_allow_html=True)

time_cols = st.columns(3)

with time_cols[0]:
    st.markdown(f"""
    <div class="metric-card" style="padding: 20px; border-radius: 15px; margin: 15px 0; text-align: center;">
        <h4 style="color: #06b6d4; margin-bottom: 15px;">🤖 ML Processing</h4>
        <h2 style="font-size: 2rem; margin: 10px 0;">{performance_metrics['ml_avg_processing_time']:.3f}s</h2>
        <p style="margin: 0;">Ultra-fast detection</p>
        <p style="color: #10b981; font-weight: bold; margin: 5px 0;">⚡ FASTEST</p>
    </div>
    """, unsafe_allow_html=True)

with time_cols[1]:
    st.markdown(f"""
    <div class="metric-card" style="padding: 20px; border-radius: 15px; margin: 15px 0; text-align: center;">
        <h4 style="color: #8b5cf6; margin-bottom: 15px;">🧠 AI Processing</h4>
        <h2 style="font-size: 2rem; margin: 10px 0;">{performance_metrics['ai_avg_processing_time']:.1f}s</h2>
        <p style="margin: 0;">Intelligent analysis</p>
        <p style="color: #f59e0b; font-weight: bold; margin: 5px 0;">🐌 SLOWER</p>
    </div>
    """, unsafe_allow_html=True)

with time_cols[2]:
    st.markdown(f"""
    <div class="hybrid-card" style="padding: 20px; border-radius: 15px; margin: 15px 0; text-align: center;">
        <h4 style="color: #10b981; margin-bottom: 15px;">🚀 Hybrid Processing</h4>
        <h2 style="font-size: 2rem; margin: 10px 0;">{performance_metrics['hybrid_avg_processing_time']:.1f}s</h2>
        <p style="margin: 0;">Best of both worlds</p>
        <p style="color: #10b981; font-weight: bold; margin: 5px 0;">⚡ OPTIMAL</p>
    </div>
    """, unsafe_allow_html=True)

# Hybrid Decision Results
if 'last_hybrid_result' in st.session_state:
    st.markdown("---")
    st.markdown("""
    <div class="section-header" style="padding: 20px; border-radius: 15px; margin: 20px 0; text-align: center;">
        <h2>🧠 Latest Hybrid Decision Result</h2>
    </div>
    """, unsafe_allow_html=True)
    
    result = st.session_state.last_hybrid_result
    
    result_cols = st.columns(2)
    
    with result_cols[0]:
        st.markdown(f"""
        <div class="hybrid-card" style="padding: 20px; border-radius: 15px; margin: 15px 0;">
            <h4 style="color: #10b981; margin-bottom: 15px;">🎯 Decision Summary</h4>
            <p><strong>Decision:</strong> {result.decision.replace('_', ' ').title()}</p>
            <p><strong>Threat Score:</strong> {result.final_threat_score:.3f}</p>
            <p><strong>Priority:</strong> {result.priority.title()}</p>
            <p><strong>Confidence:</strong> {result.confidence:.1%}</p>
            <p><strong>Processing Time:</strong> {result.processing_time:.3f}s</p>
        </div>
        """, unsafe_allow_html=True)
    
    with result_cols[1]:
        st.markdown(f"""
        <div class="hybrid-card" style="padding: 20px; border-radius: 15px; margin: 15px 0;">
            <h4 style="color: #10b981; margin-bottom: 15px;">⚖️ Contribution Analysis</h4>
            <p><strong>ML Contribution:</strong> {result.ml_contribution:.1%}</p>
            <p><strong>AI Contribution:</strong> {result.ai_contribution:.1%}</p>
            <p><strong>Combined Score:</strong> {result.final_threat_score:.3f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Reasoning and recommendations
    st.markdown("""
    <div class="hybrid-card" style="padding: 20px; border-radius: 15px; margin: 15px 0;">
        <h4 style="color: #10b981; margin-bottom: 15px;">💭 AI Reasoning</h4>
        <p>{}</p>
        
        <h4 style="color: #10b981; margin-bottom: 15px; margin-top: 20px;">📋 Recommendations</h4>
        <ul>
        {}
        </ul>
    </div>
    """.format(
        result.reasoning,
        ''.join([f'<li>{rec}</li>' for rec in result.recommendations])
    ), unsafe_allow_html=True)

# Performance Charts
st.markdown("---")
st.markdown("""
<div class="section-header" style="padding: 20px; border-radius: 15px; margin: 20px 0; text-align: center;">
    <h2>📈 Performance Analytics</h2>
</div>
""", unsafe_allow_html=True)

chart_cols = st.columns(2)

with chart_cols[0]:
    # Accuracy comparison chart
    methods = ['ML Only', 'AI Only', 'Hybrid']
    accuracies = [
        performance_metrics['ml_accuracy'],
        performance_metrics['ai_accuracy'],
        performance_metrics['hybrid_accuracy']
    ]
    
    fig_accuracy = go.Figure(data=[
        go.Bar(
            x=methods,
            y=accuracies,
            marker_color=['#06b6d4', '#8b5cf6', '#10b981'],
            text=[f'{acc:.1%}' for acc in accuracies],
            textposition='auto'
        )
    ])
    
    fig_accuracy.update_layout(
        title='Detection Accuracy Comparison',
        yaxis_title='Accuracy Rate',
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig_accuracy, use_container_width=True)

with chart_cols[1]:
    # Processing time comparison chart
    fig_time = go.Figure(data=[
        go.Bar(
            x=methods,
            y=[
                performance_metrics['ml_avg_processing_time'],
                performance_metrics['ai_avg_processing_time'],
                performance_metrics['hybrid_avg_processing_time']
            ],
            marker_color=['#06b6d4', '#8b5cf6', '#10b981'],
            text=[
                f"{performance_metrics['ml_avg_processing_time']:.3f}s",
                f"{performance_metrics['ai_avg_processing_time']:.1f}s",
                f"{performance_metrics['hybrid_avg_processing_time']:.1f}s"
            ],
            textposition='auto'
        )
    ])
    
    fig_time.update_layout(
        title='Processing Time Comparison',
        yaxis_title='Time (seconds)',
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig_time, use_container_width=True)

# Hybrid Workflow Diagram
st.markdown("---")
st.markdown("""
<div class="section-header" style="padding: 20px; border-radius: 15px; margin: 20px 0; text-align: center;">
    <h2>🔄 Hybrid Workflow Architecture</h2>
</div>
""", unsafe_allow_html=True)

workflow_cols = st.columns(3)

with workflow_cols[0]:
    st.markdown("""
    <div class="metric-card" style="padding: 20px; border-radius: 15px; margin: 15px 0; text-align: center;">
        <h4 style="color: #06b6d4; margin-bottom: 15px;">🤖 ML Layer</h4>
        <p style="margin: 5px 0;"><strong>Technology:</strong> Isolation Forest</p>
        <p style="margin: 5px 0;"><strong>Speed:</strong> 2ms</p>
        <p style="margin: 5px 0;"><strong>Role:</strong> Fast Detection</p>
        <p style="margin: 5px 0;"><strong>Output:</strong> Anomaly Score</p>
    </div>
    """, unsafe_allow_html=True)

with workflow_cols[1]:
    st.markdown("""
    <div class="metric-card" style="padding: 20px; border-radius: 15px; margin: 15px 0; text-align: center;">
        <h4 style="color: #8b5cf6; margin-bottom: 15px;">🧠 AI Layer</h4>
        <p style="margin: 5px 0;"><strong>Technology:</strong> Gemini 2.0 Flash</p>
        <p style="margin: 5px 0;"><strong>Speed:</strong> 1.5s</p>
        <p style="margin: 5px 0;"><strong>Role:</strong> Intelligent Analysis</p>
        <p style="margin: 5px 0;"><strong>Output:</strong> Context & Reasoning</p>
    </div>
    """, unsafe_allow_html=True)

with workflow_cols[2]:
    st.markdown("""
    <div class="hybrid-card" style="padding: 20px; border-radius: 15px; margin: 15px 0; text-align: center;">
        <h4 style="color: #10b981; margin-bottom: 15px;">🚀 Hybrid Engine</h4>
        <p style="margin: 5px 0;"><strong>Technology:</strong> Decision Engine</p>
        <p style="margin: 5px 0;"><strong>Speed:</strong> 1.8s</p>
        <p style="margin: 5px 0;"><strong>Role:</strong> Optimal Decision</p>
        <p style="margin: 5px 0;"><strong>Output:</strong> Final Threat Assessment</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(f"""
<div style="padding: 30px; border-radius: 15px; text-align: center; border: 2px solid #10b981;">
    <h3>🧠 Hybrid ADA Decision Engine Dashboard</h3>
    <p>ML (Isolation Forest) + AI (Gemini) = Optimal Threat Detection</p>
    <p>Current Theme: {theme_mode} | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    pass
