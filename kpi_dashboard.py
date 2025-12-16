#!/usr/bin/env python3
"""
KPI Dashboard for AI-Driven SOC Phase 1
Visualizes the follow-up KPIs and measurement infrastructure
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
    page_title="AI-Driven SOC KPI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class KPIDashboard:
    """KPI Dashboard for Phase 1 metrics"""
    
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
    
    def get_alert_quality_data(self, days: int = 7):
        """Get alert quality metrics from BigQuery"""
        if not self.client:
            return self._get_sample_alert_quality_data()
        
        try:
            query = f"""
            SELECT 
                analysis_date,
                total_alerts,
                unique_clusters,
                entropy_index,
                high_redundancy_alerts,
                unique_alerts,
                redundancy_percentage,
                uniqueness_percentage,
                coherence_rating,
                processing_speed_rating
            FROM `{self.project_id}.soc_data.kpi_summary`
            WHERE analysis_date >= DATE_SUB(CURRENT_DATE(), INTERVAL {days} DAY)
            ORDER BY analysis_date DESC
            """
            
            result = self.client.query(query).to_dataframe()
            return result if not result.empty else self._get_sample_alert_quality_data()
            
        except Exception as e:
            logger.error(f"Failed to get alert quality data: {e}")
            return self._get_sample_alert_quality_data()
    
    def get_entropy_index_trend(self, days: int = 30):
        """Get Entropy Index trend data"""
        if not self.client:
            return self._get_sample_entropy_data()
        
        try:
            query = f"""
            SELECT 
                analysis_date,
                entropy_index,
                total_alerts,
                unique_clusters,
                coherence_level,
                entropy_change
            FROM `{self.project_id}.soc_data.entropy_index_view`
            WHERE analysis_date >= DATE_SUB(CURRENT_DATE(), INTERVAL {days} DAY)
            ORDER BY analysis_date ASC
            """
            
            result = self.client.query(query).to_dataframe()
            return result if not result.empty else self._get_sample_entropy_data()
            
        except Exception as e:
            logger.error(f"Failed to get entropy index trend: {e}")
            return self._get_sample_entropy_data()
    
    def get_triage_efficiency_data(self, days: int = 7):
        """Get triage efficiency metrics"""
        if not self.client:
            return self._get_sample_triage_data()
        
        try:
            query = f"""
            SELECT 
                analysis_date,
                similarity_group,
                alert_count,
                avg_processing_time_seconds,
                fast_processing_ratio,
                slow_processing_ratio,
                estimated_time_savings
            FROM `{self.project_id}.soc_data.triage_time_analysis`
            WHERE analysis_date >= DATE_SUB(CURRENT_DATE(), INTERVAL {days} DAY)
            ORDER BY analysis_date DESC, similarity_group
            """
            
            result = self.client.query(query).to_dataframe()
            return result if not result.empty else self._get_sample_triage_data()
            
        except Exception as e:
            logger.error(f"Failed to get triage efficiency data: {e}")
            return self._get_sample_triage_data()
    
    def _get_sample_alert_quality_data(self):
        """Generate sample alert quality data for demo"""
        dates = [(datetime.now() - timedelta(days=i)).date() for i in range(7, 0, -1)]
        data = []
        
        for i, date in enumerate(dates):
            data.append({
                'analysis_date': date,
                'total_alerts': 150 + i * 10,
                'unique_clusters': 45 + i * 2,
                'entropy_index': round(0.7 - i * 0.02, 3),
                'high_redundancy_alerts': 60 + i * 5,
                'unique_alerts': 30 + i * 3,
                'redundancy_percentage': round(40 - i * 1.5, 2),
                'uniqueness_percentage': round(20 + i * 1.2, 2),
                'coherence_rating': ['Poor', 'Fair', 'Good', 'Good', 'Good', 'Excellent', 'Excellent'][i],
                'processing_speed_rating': ['Slow', 'Normal', 'Normal', 'Fast', 'Fast', 'Fast', 'Fast'][i]
            })
        
        return pd.DataFrame(data)
    
    def _get_sample_entropy_data(self):
        """Generate sample entropy index trend data"""
        dates = [(datetime.now() - timedelta(days=i)).date() for i in range(30, 0, -1)]
        data = []
        
        for i, date in enumerate(dates):
            entropy = 0.8 - (i * 0.01) + (i % 7) * 0.05  # Trending down with weekly variation
            data.append({
                'analysis_date': date,
                'entropy_index': round(max(0.1, min(0.9, entropy)), 3),
                'total_alerts': 120 + i * 2,
                'unique_clusters': 40 + i,
                'coherence_level': 'highly_coherent' if entropy < 0.3 else 'moderately_coherent' if entropy < 0.5 else 'somewhat_chaotic',
                'entropy_change': round((i % 3 - 1) * 0.02, 3)
            })
        
        return pd.DataFrame(data)
    
    def _get_sample_triage_data(self):
        """Generate sample triage efficiency data"""
        dates = [(datetime.now() - timedelta(days=i)).date() for i in range(7, 0, -1)]
        groups = ['high_similarity', 'medium_similarity', 'low_similarity', 'unique']
        data = []
        
        for date in dates:
            for group in groups:
                base_time = {'high_similarity': 15, 'medium_similarity': 25, 'low_similarity': 35, 'unique': 45}[group]
                data.append({
                    'analysis_date': date,
                    'similarity_group': group,
                    'alert_count': 20 + hash(str(date) + group) % 15,
                    'avg_processing_time_seconds': base_time + (hash(str(date)) % 10),
                    'fast_processing_ratio': 0.8 if group == 'high_similarity' else 0.6 if group == 'medium_similarity' else 0.4,
                    'slow_processing_ratio': 0.1 if group == 'high_similarity' else 0.2 if group == 'medium_similarity' else 0.3,
                    'estimated_time_savings': 30 if group == 'high_similarity' else 20 if group == 'medium_similarity' else 10
                })
        
        return pd.DataFrame(data)

def create_alert_quality_chart(data):
    """Create alert quality visualization"""
    fig = go.Figure()
    
    # Add redundancy percentage line
    fig.add_trace(go.Scatter(
        x=data['analysis_date'],
        y=data['redundancy_percentage'],
        mode='lines+markers',
        name='Redundancy %',
        line=dict(color='red', width=3),
        hovertemplate='<b>%{x}</b><br>Redundancy: %{y:.1f}%<extra></extra>'
    ))
    
    # Add uniqueness percentage line
    fig.add_trace(go.Scatter(
        x=data['analysis_date'],
        y=data['uniqueness_percentage'],
        mode='lines+markers',
        name='Uniqueness %',
        line=dict(color='green', width=3),
        hovertemplate='<b>%{x}</b><br>Uniqueness: %{y:.1f}%<extra></extra>'
    ))
    
    # Add target line (15-20% reduction target)
    fig.add_hline(y=20, line_dash="dash", line_color="orange", 
                  annotation_text="Target: <20% Redundancy", annotation_position="top right")
    
    fig.update_layout(
        title="Alert Quality Metrics - Redundancy vs Uniqueness",
        xaxis_title="Date",
        yaxis_title="Percentage (%)",
        hovermode='x unified',
        height=400
    )
    
    return fig

def create_entropy_index_chart(data):
    """Create Entropy Index trend chart"""
    fig = go.Figure()
    
    # Add entropy index line
    fig.add_trace(go.Scatter(
        x=data['analysis_date'],
        y=data['entropy_index'],
        mode='lines+markers',
        name='Entropy Index (EI)',
        line=dict(color='purple', width=3),
        hovertemplate='<b>%{x}</b><br>EI: %{y:.3f}<br>Coherence: %{customdata}<extra></extra>',
        customdata=data['coherence_level']
    ))
    
    # Add coherence zones
    fig.add_hrect(y0=0, y1=0.3, fillcolor="green", opacity=0.1, 
                  annotation_text="Highly Coherent", annotation_position="top left")
    fig.add_hrect(y0=0.3, y1=0.5, fillcolor="yellow", opacity=0.1, 
                  annotation_text="Moderately Coherent", annotation_position="top left")
    fig.add_hrect(y0=0.5, y1=0.7, fillcolor="orange", opacity=0.1, 
                  annotation_text="Somewhat Chaotic", annotation_position="top left")
    fig.add_hrect(y0=0.7, y1=1.0, fillcolor="red", opacity=0.1, 
                  annotation_text="Highly Chaotic", annotation_position="top left")
    
    fig.update_layout(
        title="SOC Noise Coherence - Entropy Index Trend",
        xaxis_title="Date",
        yaxis_title="Entropy Index (EI = 1 - Unique Clusters / Total Alerts)",
        hovermode='x unified',
        height=400
    )
    
    return fig

def create_triage_efficiency_chart(data):
    """Create triage efficiency chart"""
    # Pivot data for grouped bar chart
    pivot_data = data.pivot(index='analysis_date', columns='similarity_group', values='avg_processing_time_seconds')
    
    fig = go.Figure()
    
    colors = {'high_similarity': 'green', 'medium_similarity': 'yellow', 
              'low_similarity': 'orange', 'unique': 'red'}
    
    for group in pivot_data.columns:
        fig.add_trace(go.Bar(
            name=group.replace('_', ' ').title(),
            x=pivot_data.index,
            y=pivot_data[group],
            marker_color=colors.get(group, 'blue'),
            hovertemplate=f'<b>%{{x}}</b><br>{group}: %{{y:.1f}}s<extra></extra>'
        ))
    
    fig.update_layout(
        title="Triage Processing Time by Similarity Group",
        xaxis_title="Date",
        yaxis_title="Average Processing Time (seconds)",
        barmode='group',
        height=400
    )
    
    return fig

def create_kpi_summary_cards(data):
    """Create KPI summary cards"""
    latest = data.iloc[0] if not data.empty else {}
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Total Alerts",
            value=f"{latest.get('total_alerts', 0):,}",
            delta=f"+{latest.get('total_alerts', 0) - data.iloc[1].get('total_alerts', 0) if len(data) > 1 else 0}" if not data.empty else None
        )
    
    with col2:
        st.metric(
            label="🎯 Redundancy %",
            value=f"{latest.get('redundancy_percentage', 0):.1f}%",
            delta=f"{latest.get('redundancy_percentage', 0) - data.iloc[1].get('redundancy_percentage', 0):.1f}%" if len(data) > 1 else None,
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="🧠 Entropy Index",
            value=f"{latest.get('entropy_index', 0):.3f}",
            delta=f"{latest.get('entropy_index', 0) - data.iloc[1].get('entropy_index', 0):.3f}" if len(data) > 1 else None,
            delta_color="inverse"
        )
    
    with col4:
        coherence = latest.get('coherence_rating', 'Unknown')
        color = {'Excellent': '🟢', 'Good': '🟡', 'Fair': '🟠', 'Poor': '🔴'}.get(coherence, '⚪')
        st.metric(
            label="📈 Coherence Rating",
            value=f"{color} {coherence}",
            delta=None
        )

def main():
    """Main dashboard function"""
    st.title("📊 AI-Driven SOC KPI Dashboard")
    st.markdown("**Phase 1: Cognitive Telemetry + RL Policy Feedback Metrics**")
    
    # Initialize dashboard
    dashboard = KPIDashboard()
    
    # Sidebar controls
    st.sidebar.header("📊 Dashboard Controls")
    days = st.sidebar.slider("Analysis Period (days)", 1, 30, 7)
    refresh = st.sidebar.button("🔄 Refresh Data")
    
    # Data source indicator
    if dashboard.client:
        st.sidebar.success("✅ Connected to BigQuery")
    else:
        st.sidebar.warning("⚠️ Using sample data (BigQuery not available)")
    
    # Load data
    with st.spinner("Loading KPI data..."):
        alert_data = dashboard.get_alert_quality_data(days)
        entropy_data = dashboard.get_entropy_index_trend(days)
        triage_data = dashboard.get_triage_efficiency_data(days)
    
    # KPI Summary Cards
    st.header("🎯 Key Performance Indicators")
    create_kpi_summary_cards(alert_data)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Alert Quality Trends")
        fig1 = create_alert_quality_chart(alert_data)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("🧠 Entropy Index (SOC Coherence)")
        fig2 = create_entropy_index_chart(entropy_data)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Triage Efficiency
    st.subheader("⏱️ Triage Processing Efficiency")
    fig3 = create_triage_efficiency_chart(triage_data)
    st.plotly_chart(fig3, use_container_width=True)
    
    # Detailed Metrics Table
    st.subheader("📋 Detailed Metrics")
    
    tab1, tab2, tab3 = st.tabs(["Alert Quality", "Entropy Analysis", "Triage Efficiency"])
    
    with tab1:
        st.dataframe(alert_data, use_container_width=True)
    
    with tab2:
        st.dataframe(entropy_data, use_container_width=True)
    
    with tab3:
        st.dataframe(triage_data, use_container_width=True)
    
    # Interpretation Guide
    st.subheader("📚 KPI Interpretation Guide")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🎯 Alert Quality Metrics:**
        - **Redundancy %**: Lower is better (target <20%)
        - **Uniqueness %**: Higher indicates more diverse alerts
        - **Coherence Rating**: Excellent = organized, Poor = chaotic
        
        **🧠 Entropy Index (EI):**
        - **EI = 1 - (Unique Clusters / Total Alerts)**
        - **Lower EI** = More organized, less chaotic
        - **Higher EI** = More chaotic, needs attention
        """)
    
    with col2:
        st.markdown("""
        **⏱️ Triage Efficiency:**
        - **High Similarity**: Fast processing (15-25s)
        - **Medium Similarity**: Normal processing (25-35s)
        - **Low Similarity**: Slower processing (35-45s)
        - **Unique**: Slowest processing (45s+)
        
        **📊 Success Indicators:**
        - ✅ Redundancy reduction 15-20%
        - ✅ Entropy Index trending down
        - ✅ Processing time decreasing
        - ✅ Coherence rating improving
        """)

if __name__ == "__main__":
    main()












