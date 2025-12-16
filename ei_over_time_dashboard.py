#!/usr/bin/env python3
"""
EI-over-time Dashboard
Shows pre- vs post-entropy spike and learning progression
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import json
import logging
from google.cloud import bigquery
from google.oauth2 import service_account

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="EI-over-time Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class EIDashboard:
    """Entropy Index over time dashboard"""
    
    def __init__(self):
        self.project_id = "chronicle-dev-2be9"
        self.client = self._get_bigquery_client()
        
    def _get_bigquery_client(self):
        """Get BigQuery client with fallback authentication"""
        try:
            service_account_paths = [
                'Service Account BigQuery/chronicle-dev-2be-a-driven-soc.json',
                'Service Account BigQuery/sa-gatra-bigquery.json',
                'Service Account BigQuery/gatra-user-bigquery.json'
            ]
            
            for sa_path in service_account_paths:
                if os.path.exists(sa_path):
                    credentials = service_account.Credentials.from_service_account_file(sa_path)
                    return bigquery.Client(credentials=credentials, project=self.project_id)
            
            return bigquery.Client(project=self.project_id)
        except Exception as e:
            logger.error(f"BigQuery client initialization failed: {e}")
            return None
    
    def get_entropy_timeline_data(self, days: int = 21):
        """Get entropy timeline data from BigQuery"""
        if not self.client:
            return self._get_sample_entropy_data()
        
        try:
            query = f"""
            WITH daily_entropy AS (
                SELECT 
                    DATE(embedding_timestamp) as analysis_date,
                    COUNT(*) as total_alerts,
                    COUNT(DISTINCT ROUND(embedding_similarity, 2)) as unique_clusters,
                    ROUND(1 - (COUNT(DISTINCT ROUND(embedding_similarity, 2)) / COUNT(*)), 4) as entropy_index,
                    AVG(embedding_similarity) as avg_similarity,
                    AVG(rl_reward_score) as avg_reward,
                    COUNT(CASE WHEN embedding_similarity > 0.8 THEN 1 END) as high_similarity_alerts,
                    COUNT(CASE WHEN rl_reward_score > 0.7 THEN 1 END) as high_reward_alerts
                FROM `{self.project_id}.soc_data.processed_alerts`
                WHERE embedding_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days} DAY)
                GROUP BY DATE(embedding_timestamp)
                ORDER BY analysis_date ASC
            ),
            phase_analysis AS (
                SELECT 
                    analysis_date,
                    entropy_index,
                    total_alerts,
                    unique_clusters,
                    avg_similarity,
                    avg_reward,
                    high_similarity_alerts,
                    high_reward_alerts,
                    -- Identify phases
                    CASE 
                        WHEN analysis_date < DATE_SUB(CURRENT_DATE(), INTERVAL 14 DAY) THEN 'Chaotic Baseline'
                        WHEN analysis_date < DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY) THEN 'Learning Phase'
                        ELSE 'Stabilization Phase'
                    END as phase,
                    -- Calculate moving averages
                    AVG(entropy_index) OVER (
                        ORDER BY analysis_date 
                        ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING
                    ) as entropy_ma_5d,
                    AVG(avg_similarity) OVER (
                        ORDER BY analysis_date 
                        ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING
                    ) as similarity_ma_5d,
                    AVG(avg_reward) OVER (
                        ORDER BY analysis_date 
                        ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING
                    ) as reward_ma_5d
                FROM daily_entropy
            )
            SELECT 
                analysis_date,
                entropy_index,
                total_alerts,
                unique_clusters,
                avg_similarity,
                avg_reward,
                high_similarity_alerts,
                high_reward_alerts,
                phase,
                entropy_ma_5d,
                similarity_ma_5d,
                reward_ma_5d,
                -- Calculate daily changes
                LAG(entropy_index, 1) OVER (ORDER BY analysis_date) as prev_entropy,
                entropy_index - LAG(entropy_index, 1) OVER (ORDER BY analysis_date) as entropy_change,
                LAG(avg_similarity, 1) OVER (ORDER BY analysis_date) as prev_similarity,
                avg_similarity - LAG(avg_similarity, 1) OVER (ORDER BY analysis_date) as similarity_change,
                LAG(avg_reward, 1) OVER (ORDER BY analysis_date) as prev_reward,
                avg_reward - LAG(avg_reward, 1) OVER (ORDER BY analysis_date) as reward_change
            FROM phase_analysis
            ORDER BY analysis_date ASC
            """
            
            result = self.client.query(query).to_dataframe()
            return result if not result.empty else self._get_sample_entropy_data()
            
        except Exception as e:
            logger.error(f"Failed to get entropy timeline data: {e}")
            return self._get_sample_entropy_data()
    
    def _get_sample_entropy_data(self):
        """Generate sample entropy data for demo"""
        dates = [(datetime.now() - timedelta(days=i)).date() for i in range(21, 0, -1)]
        data = []
        
        for i, date in enumerate(dates):
            # Simulate learning progression
            if i < 7:  # Chaotic baseline
                entropy = 0.9 - (i * 0.05) + random.uniform(-0.1, 0.1)
                phase = "Chaotic Baseline"
            elif i < 14:  # Learning phase
                learning_factor = (i - 7) / 7
                entropy = 0.9 - (learning_factor * 0.4) + random.uniform(-0.05, 0.05)
                phase = "Learning Phase"
            else:  # Stabilization
                entropy = 0.5 - ((i - 14) * 0.02) + random.uniform(-0.03, 0.03)
                phase = "Stabilization Phase"
            
            entropy = max(0.1, min(1.0, entropy))
            
            data.append({
                'analysis_date': date,
                'entropy_index': round(entropy, 3),
                'total_alerts': 50 + random.randint(-10, 20),
                'unique_clusters': int(50 * entropy) + random.randint(-5, 5),
                'avg_similarity': round(1 - entropy + random.uniform(-0.1, 0.1), 3),
                'avg_reward': round(0.2 + (1 - entropy) * 0.6 + random.uniform(-0.1, 0.1), 3),
                'high_similarity_alerts': int(50 * (1 - entropy) * 0.3),
                'high_reward_alerts': int(50 * (1 - entropy) * 0.4),
                'phase': phase,
                'entropy_ma_5d': round(entropy + random.uniform(-0.05, 0.05), 3),
                'similarity_ma_5d': round(1 - entropy + random.uniform(-0.05, 0.05), 3),
                'reward_ma_5d': round(0.2 + (1 - entropy) * 0.6 + random.uniform(-0.05, 0.05), 3),
                'prev_entropy': data[i-1]['entropy_index'] if i > 0 else entropy,
                'entropy_change': 0 if i == 0 else round(entropy - data[i-1]['entropy_index'], 3),
                'prev_similarity': data[i-1]['avg_similarity'] if i > 0 else 1 - entropy,
                'similarity_change': 0 if i == 0 else round((1 - entropy) - data[i-1]['avg_similarity'], 3),
                'prev_reward': data[i-1]['avg_reward'] if i > 0 else 0.2 + (1 - entropy) * 0.6,
                'reward_change': 0 if i == 0 else round((0.2 + (1 - entropy) * 0.6) - data[i-1]['avg_reward'], 3)
            })
        
        return pd.DataFrame(data)
    
    def create_entropy_timeline_chart(self, data):
        """Create entropy timeline chart with phases"""
        fig = go.Figure()
        
        # Add entropy index line
        fig.add_trace(go.Scatter(
            x=data['analysis_date'],
            y=data['entropy_index'],
            mode='lines+markers',
            name='Entropy Index (EI)',
            line=dict(color='red', width=3),
            hovertemplate='<b>%{x}</b><br>EI: %{y:.3f}<br>Phase: %{customdata}<extra></extra>',
            customdata=data['phase']
        ))
        
        # Add moving average
        fig.add_trace(go.Scatter(
            x=data['analysis_date'],
            y=data['entropy_ma_5d'],
            mode='lines',
            name='5-Day Moving Average',
            line=dict(color='blue', width=2, dash='dash'),
            hovertemplate='<b>%{x}</b><br>MA: %{y:.3f}<extra></extra>'
        ))
        
        # Add phase regions
        phases = data['phase'].unique()
        colors = {'Chaotic Baseline': 'rgba(255,0,0,0.1)', 
                 'Learning Phase': 'rgba(255,165,0,0.1)', 
                 'Stabilization Phase': 'rgba(0,255,0,0.1)'}
        
        for phase in phases:
            phase_data = data[data['phase'] == phase]
            if not phase_data.empty:
                fig.add_vrect(
                    x0=phase_data['analysis_date'].min(),
                    x1=phase_data['analysis_date'].max(),
                    fillcolor=colors.get(phase, 'rgba(128,128,128,0.1)'),
                    layer="below",
                    line_width=0,
                    annotation_text=phase,
                    annotation_position="top left"
                )
        
        # Add target lines
        fig.add_hline(y=0.7, line_dash="dash", line_color="orange", 
                      annotation_text="Target: EI < 0.7", annotation_position="top right")
        fig.add_hline(y=0.3, line_dash="dash", line_color="green", 
                      annotation_text="Excellent: EI < 0.3", annotation_position="top right")
        
        fig.update_layout(
            title="Entropy Index (EI) Over Time - Learning Progression",
            xaxis_title="Date",
            yaxis_title="Entropy Index (EI = 1 - Unique Clusters / Total Alerts)",
            hovermode='x unified',
            height=500
        )
        
        return fig
    
    def create_learning_metrics_chart(self, data):
        """Create learning metrics comparison chart"""
        fig = go.Figure()
        
        # Add similarity trend
        fig.add_trace(go.Scatter(
            x=data['analysis_date'],
            y=data['avg_similarity'],
            mode='lines+markers',
            name='Average Similarity',
            line=dict(color='green', width=3),
            yaxis='y1'
        ))
        
        # Add reward trend
        fig.add_trace(go.Scatter(
            x=data['analysis_date'],
            y=data['avg_reward'],
            mode='lines+markers',
            name='Average Reward',
            line=dict(color='blue', width=3),
            yaxis='y2'
        ))
        
        # Add entropy trend (inverted)
        fig.add_trace(go.Scatter(
            x=data['analysis_date'],
            y=1 - data['entropy_index'],  # Invert entropy for coherence
            mode='lines+markers',
            name='Coherence (1-EI)',
            line=dict(color='purple', width=3),
            yaxis='y1'
        ))
        
        fig.update_layout(
            title="Learning Metrics Progression",
            xaxis_title="Date",
            yaxis=dict(title="Similarity & Coherence", side="left"),
            yaxis2=dict(title="Reward Score", side="right", overlaying="y"),
            hovermode='x unified',
            height=400
        )
        
        return fig
    
    def create_phase_comparison_chart(self, data):
        """Create phase comparison chart"""
        phase_summary = data.groupby('phase').agg({
            'entropy_index': ['mean', 'std', 'min', 'max'],
            'avg_similarity': 'mean',
            'avg_reward': 'mean',
            'total_alerts': 'sum'
        }).round(3)
        
        phase_summary.columns = ['_'.join(col).strip() for col in phase_summary.columns]
        phase_summary = phase_summary.reset_index()
        
        fig = go.Figure()
        
        # Add entropy bars
        fig.add_trace(go.Bar(
            name='Average Entropy',
            x=phase_summary['phase'],
            y=phase_summary['entropy_index_mean'],
            error_y=dict(type='data', array=phase_summary['entropy_index_std']),
            marker_color='red'
        ))
        
        # Add similarity bars
        fig.add_trace(go.Bar(
            name='Average Similarity',
            x=phase_summary['phase'],
            y=phase_summary['avg_similarity_mean'],
            marker_color='green',
            yaxis='y2'
        ))
        
        # Add reward bars
        fig.add_trace(go.Bar(
            name='Average Reward',
            x=phase_summary['phase'],
            y=phase_summary['avg_reward_mean'],
            marker_color='blue',
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="Phase Comparison - Learning Progression",
            xaxis_title="Phase",
            yaxis=dict(title="Entropy Index", side="left"),
            yaxis2=dict(title="Similarity & Reward", side="right", overlaying="y"),
            barmode='group',
            height=400
        )
        
        return fig

def main():
    """Main dashboard function"""
    st.title("📊 EI-over-time Dashboard")
    st.markdown("**Pre- vs Post-Entropy Spike Analysis & Learning Progression**")
    
    # Initialize dashboard
    dashboard = EIDashboard()
    
    # Sidebar controls
    st.sidebar.header("📊 Dashboard Controls")
    days = st.sidebar.slider("Analysis Period (days)", 7, 30, 21)
    refresh = st.sidebar.button("🔄 Refresh Data")
    
    # Data source indicator
    if dashboard.client:
        st.sidebar.success("✅ Connected to BigQuery")
    else:
        st.sidebar.warning("⚠️ Using sample data (BigQuery not available)")
    
    # Load data
    with st.spinner("Loading entropy timeline data..."):
        data = dashboard.get_entropy_timeline_data(days)
    
    # Summary metrics
    st.header("🎯 Learning Progression Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        latest_entropy = data.iloc[-1]['entropy_index']
        st.metric(
            label="Current Entropy Index",
            value=f"{latest_entropy:.3f}",
            delta=f"{data.iloc[-1]['entropy_change']:+.3f}" if 'entropy_change' in data.columns else None
        )
    
    with col2:
        latest_similarity = data.iloc[-1]['avg_similarity']
        st.metric(
            label="Average Similarity",
            value=f"{latest_similarity:.3f}",
            delta=f"{data.iloc[-1]['similarity_change']:+.3f}" if 'similarity_change' in data.columns else None
        )
    
    with col3:
        latest_reward = data.iloc[-1]['avg_reward']
        st.metric(
            label="Average Reward",
            value=f"{latest_reward:.3f}",
            delta=f"{data.iloc[-1]['reward_change']:+.3f}" if 'reward_change' in data.columns else None
        )
    
    with col4:
        coherence_level = "Highly Coherent" if latest_entropy < 0.3 else "Moderately Coherent" if latest_entropy < 0.5 else "Somewhat Chaotic" if latest_entropy < 0.7 else "Highly Chaotic"
        st.metric(
            label="Coherence Level",
            value=coherence_level,
            delta=None
        )
    
    # Charts
    st.header("📈 Entropy Index Timeline")
    fig1 = dashboard.create_entropy_timeline_chart(data)
    st.plotly_chart(fig1, use_container_width=True)
    
    st.header("🧠 Learning Metrics Progression")
    fig2 = dashboard.create_learning_metrics_chart(data)
    st.plotly_chart(fig2, use_container_width=True)
    
    st.header("📊 Phase Comparison")
    fig3 = dashboard.create_phase_comparison_chart(data)
    st.plotly_chart(fig3, use_container_width=True)
    
    # Detailed data table
    st.header("📋 Detailed Timeline Data")
    
    # Filter by phase
    selected_phases = st.multiselect(
        "Filter by Phase",
        options=data['phase'].unique(),
        default=data['phase'].unique()
    )
    
    filtered_data = data[data['phase'].isin(selected_phases)]
    st.dataframe(filtered_data, use_container_width=True)
    
    # Learning assessment
    st.header("🎯 Learning Assessment")
    
    # Calculate learning metrics
    chaotic_phase = data[data['phase'] == 'Chaotic Baseline']
    learning_phase = data[data['phase'] == 'Learning Phase']
    stabilization_phase = data[data['phase'] == 'Stabilization Phase']
    
    if not chaotic_phase.empty and not stabilization_phase.empty:
        chaotic_avg = chaotic_phase['entropy_index'].mean()
        stabilization_avg = stabilization_phase['entropy_index'].mean()
        improvement = chaotic_avg - stabilization_avg
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Learning Progress:**")
            st.write(f"• Chaotic Baseline: {chaotic_avg:.3f}")
            st.write(f"• Stabilization: {stabilization_avg:.3f}")
            st.write(f"• Improvement: {improvement:+.3f}")
            
            if improvement > 0.3:
                st.success("🎉 Excellent learning progress!")
            elif improvement > 0.1:
                st.info("✅ Good learning progress")
            else:
                st.warning("⚠️ Minimal learning progress")
        
        with col2:
            st.markdown("**Success Criteria:**")
            st.write(f"• Target EI < 0.7: {'✅' if stabilization_avg < 0.7 else '❌'}")
            st.write(f"• Significant Improvement: {'✅' if improvement > 0.1 else '❌'}")
            st.write(f"• Stable Trend: {'✅' if stabilization_phase['entropy_index'].std() < 0.1 else '❌'}")

if __name__ == "__main__":
    main()












