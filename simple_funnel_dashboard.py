#!/usr/bin/env python3
"""
Simple working dashboard with Security Operations Funnel
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import time

# Page config
st.set_page_config(
    page_title="SOC Dashboard with Security Operations Funnel",
    page_icon="🛡️",
    layout="wide"
)

def create_security_operations_funnel(data):
    """Create a vertical Security Operations Funnel"""
    stages = [
        {
            "name": "Raw Events",
            "count": data.get('raw_events', 14622),
            "percentage": 100,
            "color": "#1f77b4",
            "icon": "📊"
        },
        {
            "name": "ADA Analysis", 
            "count": data.get('ada_analyzed', 12428),
            "percentage": round((data.get('ada_analyzed', 12428) / data.get('raw_events', 14622)) * 100, 1),
            "color": "#ff7f0e",
            "icon": "🔍"
        },
        {
            "name": "TAA Triage",
            "count": data.get('taa_triaged', 10195),
            "percentage": round((data.get('taa_triaged', 10195) / data.get('raw_events', 14622)) * 100, 1),
            "color": "#2ca02c",
            "icon": "⚡"
        },
        {
            "name": "CRA Containment",
            "count": data.get('cra_contained', 3396),
            "percentage": round((data.get('cra_contained', 3396) / data.get('raw_events', 14622)) * 100, 1),
            "color": "#d62728",
            "icon": "🛡️"
        }
    ]
    
    fig = go.Figure()
    
    for stage in stages:
        bar_width = stage['percentage'] / 100
        
        fig.add_trace(go.Bar(
            x=[stage['count']],
            y=[stage['name']],
            orientation='h',
            width=bar_width,
            marker=dict(
                color=stage['color'],
                line=dict(width=2, color='white')
            ),
            text=f"{stage['icon']} {stage['count']:,} ({stage['percentage']}%)",
            textposition='inside',
            textfont=dict(size=14, color='white'),
            name=stage['name']
        ))
    
    fig.update_layout(
        title="🛡️ Security Operations Funnel (Vertical - Best Practice)",
        xaxis=dict(title="Number of Events", showgrid=True),
        yaxis=dict(title="Processing Stage", showgrid=False),
        barmode='overlay',
        height=500,
        showlegend=False,
        plot_bgcolor='white'
    )
    
    return fig

def main():
    st.title("🛡️ SOC Dashboard with Security Operations Funnel")
    st.markdown("**AI-driven Security Operations Center with Real-time Pipeline Visualization**")
    
    # Sidebar
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["Security Operations Funnel", "Overview", "Analytics"]
    )
    
    # Sample data
    funnel_data = {
        'raw_events': 14622,
        'ada_analyzed': 12428,
        'taa_triaged': 10195,
        'cra_contained': 3396
    }
    
    if page == "Security Operations Funnel":
        st.header("🛡️ Security Operations Funnel")
        st.markdown("**Vertical orientation following SOC industry best practices**")
        
        # Metrics cards
        col1, col2, col3, col4 = st.columns(4)
        
        stages = [
            {"name": "Raw Events", "count": funnel_data['raw_events'], "icon": "📊", "color": "blue"},
            {"name": "ADA Analysis", "count": funnel_data['ada_analyzed'], "icon": "🔍", "color": "orange"},
            {"name": "TAA Triage", "count": funnel_data['taa_triaged'], "icon": "⚡", "color": "green"},
            {"name": "CRA Containment", "count": funnel_data['cra_contained'], "icon": "🛡️", "color": "red"}
        ]
        
        for i, stage in enumerate(stages):
            with [col1, col2, col3, col4][i]:
                st.metric(
                    label=f"{stage['icon']} {stage['name']}",
                    value=f"{stage['count']:,}",
                    delta=f"{round((stage['count'] / funnel_data['raw_events']) * 100, 1)}% of total"
                )
        
        st.markdown("---")
        
        # Funnel chart
        funnel_fig = create_security_operations_funnel(funnel_data)
        st.plotly_chart(funnel_fig, use_container_width=True)
        
        # Agent explanations
        st.header("🤖 Agent Pipeline Explanation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🔍 ADA (Anomaly Detection Agent)**
            - **Input:** Raw security events from SIEM
            - **Process:** ML models identify potential anomalies
            - **Output:** Flagged events requiring analysis
            - **Success Rate:** 85% of events processed
            """)
            
            st.markdown("""
            **⚡ TAA (Triage & Analysis Agent)**
            - **Input:** Flagged events from ADA
            - **Process:** Contextual intelligence validation
            - **Output:** Prioritized genuine threats
            - **Success Rate:** 70% of flagged events
            """)
        
        with col2:
            st.markdown("""
            **🛡️ CRA (Containment Response Agent)**
            - **Input:** Validated threats from TAA
            - **Process:** Automated response execution
            - **Output:** Contained security incidents
            - **Success Rate:** 23% end-to-end containment
            """)
            
            st.markdown("""
            **📊 Overall Pipeline Efficiency**
            - **Total Events:** 14,622
            - **Contained Threats:** 3,396
            - **False Positive Rate:** 77%
            - **Average Processing Time:** 2.3 minutes
            """)
        
        # Analytics
        st.header("📈 Funnel Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            conversion_data = {
                'Stage': ['Raw→ADA', 'ADA→TAA', 'TAA→CRA'],
                'Conversion Rate': [85.0, 82.0, 33.3]
            }
            
            fig = px.bar(
                conversion_data, 
                x='Stage', 
                y='Conversion Rate',
                title="Stage-to-Stage Conversion Rates",
                color='Conversion Rate',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            time_data = {
                'Stage': ['Raw Events', 'ADA Analysis', 'TAA Triage', 'CRA Containment'],
                'Avg Time (min)': [0.1, 1.2, 0.8, 0.2],
                'Volume': [14622, 12428, 10195, 3396]
            }
            
            fig = px.scatter(
                time_data, 
                x='Avg Time (min)', 
                y='Volume',
                size='Volume',
                color='Stage',
                title="Processing Time vs Volume by Stage"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Overview":
        st.header("📊 System Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Overall Accuracy", "90.0%", "+2.1%")
        with col2:
            st.metric("Predictions Count", "20.0", "+1,234")
        with col3:
            st.metric("Model Version", "CLA v2.1.3", "↑ Latest")
        
        st.success("✅ All systems operational")
    
    elif page == "Analytics":
        st.header("📈 Performance Analytics")
        
        # Sample performance data
        dates = pd.date_range(start='2024-01-01', end='2024-01-30', freq='D')
        performance_data = pd.DataFrame({
            'Date': dates,
            'Accuracy': [85 + i*0.2 + (i%7)*2 for i in range(len(dates))],
            'Precision': [82 + i*0.15 + (i%5)*1.5 for i in range(len(dates))],
            'Recall': [88 + i*0.18 + (i%6)*1.8 for i in range(len(dates))]
        })
        
        fig = px.line(
            performance_data, 
            x='Date', 
            y=['Accuracy', 'Precision', 'Recall'],
            title="Performance Metrics Over Time"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Refresh button
    if st.button("🔄 Refresh Data"):
        st.rerun()

if __name__ == "__main__":
    main()



