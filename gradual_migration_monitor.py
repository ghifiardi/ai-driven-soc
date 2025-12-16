#!/usr/bin/env python3
"""
Gradual Migration Monitor Dashboard
==================================

Real-time monitoring dashboard for gradual migration of TAA services.
Shows A/B comparison results, performance metrics, and migration progress.

Author: AI-Driven SOC Team
Date: 2025-09-18
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import time
from google.cloud import pubsub_v1
from google.cloud import bigquery
import logging

# Configuration
PROJECT_ID = "chronicle-dev-2be9"
COMPARISON_TOPIC_ID = "taa-comparison"
SUBSCRIPTION_ID = "taa-comparison-subscription"

# Page config
st.set_page_config(
    page_title="Gradual Migration Monitor",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
}
.success-card {
    border-left-color: #2ca02c;
}
.warning-card {
    border-left-color: #ff7f0e;
}
.error-card {
    border-left-color: #d62728;
}
</style>
""", unsafe_allow_html=True)

class GradualMigrationMonitor:
    """Monitor for gradual migration process"""
    
    def __init__(self):
        self.project_id = PROJECT_ID
        self.subscriber = pubsub_v1.SubscriberClient()
        self.bq_client = bigquery.Client(project=PROJECT_ID)
        
    def get_sample_comparison_data(self):
        """Get sample comparison data for demonstration"""
        return pd.DataFrame([
            {
                'timestamp': datetime.now() - timedelta(minutes=i),
                'alarm_id': f'ALERT_{1000+i}',
                'original_classification': 'benign',
                'enhanced_classification': 'suspicious' if i % 3 == 0 else 'critical' if i % 5 == 0 else 'low_risk',
                'original_confidence': 0.95,
                'enhanced_confidence': 0.85 if i % 3 == 0 else 0.92 if i % 5 == 0 else 0.78,
                'classification_changed': True if i % 3 == 0 else False,
                'confidence_difference': -0.1 if i % 3 == 0 else -0.03 if i % 5 == 0 else -0.17,
                'migration_phase': 1
            }
            for i in range(50)
        ])
    
    def calculate_migration_metrics(self, df):
        """Calculate migration metrics"""
        if df.empty:
            return {}
        
        total_alerts = len(df)
        changed_classifications = len(df[df['classification_changed'] == True])
        improvement_rate = (changed_classifications / total_alerts * 100) if total_alerts > 0 else 0
        
        # Classification distribution
        enhanced_dist = df['enhanced_classification'].value_counts()
        original_dist = df['original_classification'].value_counts()
        
        return {
            'total_alerts': total_alerts,
            'changed_classifications': changed_classifications,
            'improvement_rate': improvement_rate,
            'enhanced_distribution': enhanced_dist,
            'original_distribution': original_dist,
            'avg_confidence_diff': df['confidence_difference'].mean()
        }
    
    def create_classification_comparison_chart(self, df):
        """Create classification comparison chart"""
        if df.empty:
            return go.Figure()
        
        # Prepare data for comparison
        original_counts = df['original_classification'].value_counts()
        enhanced_counts = df['enhanced_classification'].value_counts()
        
        fig = go.Figure()
        
        # Original classification
        fig.add_trace(go.Bar(
            name='Original TAA',
            x=original_counts.index,
            y=original_counts.values,
            marker_color='lightblue'
        ))
        
        # Enhanced classification
        fig.add_trace(go.Bar(
            name='Enhanced TAA',
            x=enhanced_counts.index,
            y=enhanced_counts.values,
            marker_color='lightgreen'
        ))
        
        fig.update_layout(
            title='Classification Comparison: Original vs Enhanced',
            xaxis_title='Classification',
            yaxis_title='Number of Alerts',
            barmode='group'
        )
        
        return fig
    
    def create_confidence_comparison_chart(self, df):
        """Create confidence comparison chart"""
        if df.empty:
            return go.Figure()
        
        fig = go.Figure()
        
        # Original confidence
        fig.add_trace(go.Scatter(
            name='Original Confidence',
            x=df.index,
            y=df['original_confidence'],
            mode='markers',
            marker=dict(color='blue', size=8)
        ))
        
        # Enhanced confidence
        fig.add_trace(go.Scatter(
            name='Enhanced Confidence',
            x=df.index,
            y=df['enhanced_confidence'],
            mode='markers',
            marker=dict(color='green', size=8)
        ))
        
        fig.update_layout(
            title='Confidence Comparison',
            xaxis_title='Alert Index',
            yaxis_title='Confidence Score',
            yaxis=dict(range=[0, 1])
        )
        
        return fig

def main():
    """Main dashboard"""
    st.title("🔄 TAA Gradual Migration Monitor")
    st.markdown("**Real-time monitoring of Enhanced TAA vs Original TAA classification results**")
    
    # Initialize monitor
    monitor = GradualMigrationMonitor()
    
    # Sidebar
    st.sidebar.header("Migration Settings")
    migration_phase = st.sidebar.selectbox(
        "Current Migration Phase",
        [1, 2, 3, 4],
        index=0,
        format_func=lambda x: f"Phase {x}: {[10, 25, 50, 100][x-1]}% traffic"
    )
    
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 5, 60, 30)
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    
    # Main content
    col1, col2, col3, col4 = st.columns(4)
    
    # Get sample data (in real implementation, this would come from Pub/Sub)
    df = monitor.get_sample_comparison_data()
    metrics = monitor.calculate_migration_metrics(df)
    
    # Metrics cards
    with col1:
        st.metric(
            label="Total Alerts Processed",
            value=metrics.get('total_alerts', 0),
            delta=None
        )
    
    with col2:
        st.metric(
            label="Classification Changes",
            value=metrics.get('changed_classifications', 0),
            delta=f"{metrics.get('improvement_rate', 0):.1f}%"
        )
    
    with col3:
        st.metric(
            label="Average Confidence Change",
            value=f"{metrics.get('avg_confidence_diff', 0):.3f}",
            delta=None
        )
    
    with col4:
        st.metric(
            label="Migration Phase",
            value=f"Phase {migration_phase}",
            delta=f"{[10, 25, 50, 100][migration_phase-1]}%"
        )
    
    # Charts section
    st.header("📊 Classification Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(
            monitor.create_classification_comparison_chart(df),
            use_container_width=True
        )
    
    with col2:
        st.plotly_chart(
            monitor.create_confidence_comparison_chart(df),
            use_container_width=True
        )
    
    # Detailed results table
    st.header("📋 Detailed Results")
    
    # Filter options
    col1, col2 = st.columns(2)
    
    with col1:
        show_changes_only = st.checkbox("Show only changed classifications", value=False)
    
    with col2:
        classification_filter = st.selectbox(
            "Filter by Enhanced Classification",
            ["All"] + list(df['enhanced_classification'].unique()),
            index=0
        )
    
    # Apply filters
    filtered_df = df.copy()
    
    if show_changes_only:
        filtered_df = filtered_df[filtered_df['classification_changed'] == True]
    
    if classification_filter != "All":
        filtered_df = filtered_df[filtered_df['enhanced_classification'] == classification_filter]
    
    # Display table
    st.dataframe(
        filtered_df[['timestamp', 'alarm_id', 'original_classification', 'enhanced_classification', 
                    'original_confidence', 'enhanced_confidence', 'classification_changed']],
        use_container_width=True
    )
    
    # Migration progress
    st.header("🎯 Migration Progress")
    
    progress = migration_phase / 4
    st.progress(progress)
    st.write(f"**Current Phase:** {migration_phase}/4 - {[10, 25, 50, 100][migration_phase-1]}% of traffic routed to Enhanced TAA")
    
    # Recommendations
    st.header("💡 Recommendations")
    
    if migration_phase == 1:
        st.info("""
        **Phase 1 Status:** 10% traffic routing
        - Monitor for 24-48 hours
        - Validate enhanced classification accuracy
        - Check for any performance issues
        - Ready to proceed to Phase 2 when confident
        """)
    elif migration_phase == 2:
        st.info("""
        **Phase 2 Status:** 25% traffic routing
        - Continue monitoring
        - Analyze classification improvements
        - Prepare for Phase 3 (50% traffic)
        """)
    elif migration_phase == 3:
        st.info("""
        **Phase 3 Status:** 50% traffic routing
        - Half of traffic now using Enhanced TAA
        - Monitor performance closely
        - Prepare for full migration
        """)
    else:
        st.success("""
        **Phase 4 Status:** 100% traffic routing
        - Full migration complete
        - Enhanced TAA processing all alerts
        - Consider decommissioning original TAA
        """)
    
    # Auto refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()


