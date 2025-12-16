import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

def create_security_operations_funnel(data):
    """
    Create a vertical Security Operations Funnel
    Best practice: Vertical orientation for SOC workflows
    """
    
    # Define funnel stages (top to bottom)
    stages = [
        {
            "name": "Raw Events",
            "count": data.get('raw_events', 14622),
            "percentage": 100,
            "color": "#1f77b4",
            "icon": "📊",
            "description": "Total security events from SIEM"
        },
        {
            "name": "ADA Analysis", 
            "count": data.get('ada_analyzed', 12428),
            "percentage": round((data.get('ada_analyzed', 12428) / data.get('raw_events', 14622)) * 100, 1),
            "color": "#ff7f0e",
            "icon": "🔍",
            "description": "ML-analyzed anomalies"
        },
        {
            "name": "TAA Triage",
            "count": data.get('taa_triaged', 10195),
            "percentage": round((data.get('taa_triaged', 10195) / data.get('raw_events', 14622)) * 100, 1),
            "color": "#2ca02c",
            "icon": "⚡",
            "description": "Contextually validated threats"
        },
        {
            "name": "CRA Containment",
            "count": data.get('cra_contained', 3396),
            "percentage": round((data.get('cra_contained', 3396) / data.get('raw_events', 14622)) * 100, 1),
            "color": "#d62728",
            "icon": "🛡️",
            "description": "Confirmed threats requiring response"
        }
    ]
    
    # Create vertical funnel chart
    fig = go.Figure()
    
    # Add funnel bars (vertical)
    for i, stage in enumerate(stages):
        # Calculate bar width based on percentage
        bar_width = stage['percentage'] / 100
        
        # Add horizontal bar for each stage
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
            name=stage['name'],
            hovertemplate=f"<b>{stage['name']}</b><br>" +
                         f"Count: {stage['count']:,}<br>" +
                         f"Percentage: {stage['percentage']}%<br>" +
                         f"Description: {stage['description']}<extra></extra>"
        ))
    
    # Update layout for vertical funnel
    fig.update_layout(
        title={
            'text': "🛡️ Security Operations Funnel",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis=dict(
            title="Number of Events",
            showgrid=True,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title="Processing Stage",
            showgrid=False
        ),
        barmode='overlay',
        height=500,
        showlegend=False,
        plot_bgcolor='white',
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

def create_funnel_metrics_cards(data):
    """
    Create metric cards for the funnel
    """
    col1, col2, col3, col4 = st.columns(4)
    
    stages = [
        {
            "name": "Raw Events",
            "count": data.get('raw_events', 14622),
            "color": "blue",
            "icon": "📊"
        },
        {
            "name": "ADA Analysis",
            "count": data.get('ada_analyzed', 12428),
            "color": "orange", 
            "icon": "🔍"
        },
        {
            "name": "TAA Triage",
            "count": data.get('taa_triaged', 10195),
            "color": "green",
            "icon": "⚡"
        },
        {
            "name": "CRA Containment",
            "count": data.get('cra_contained', 3396),
            "color": "red",
            "icon": "🛡️"
        }
    ]
    
    for i, stage in enumerate(stages):
        with [col1, col2, col3, col4][i]:
            st.metric(
                label=f"{stage['icon']} {stage['name']}",
                value=f"{stage['count']:,}",
                delta=f"{round((stage['count'] / data.get('raw_events', 14622)) * 100, 1)}% of total"
            )

def create_agent_pipeline_explanation():
    """
    Create detailed explanation of each agent's role
    """
    st.subheader("🤖 Agent Pipeline Explanation")
    
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

def create_funnel_analytics(data):
    """
    Create additional analytics for the funnel
    """
    st.subheader("📈 Funnel Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Conversion rates between stages
        conversion_data = {
            'Stage': ['Raw→ADA', 'ADA→TAA', 'TAA→CRA'],
            'Conversion Rate': [
                round((data.get('ada_analyzed', 12428) / data.get('raw_events', 14622)) * 100, 1),
                round((data.get('taa_triaged', 10195) / data.get('ada_analyzed', 12428)) * 100, 1),
                round((data.get('cra_contained', 3396) / data.get('taa_triaged', 10195)) * 100, 1)
            ]
        }
        
        fig = px.bar(
            conversion_data, 
            x='Stage', 
            y='Conversion Rate',
            title="Stage-to-Stage Conversion Rates",
            color='Conversion Rate',
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Processing time by stage
        time_data = {
            'Stage': ['Raw Events', 'ADA Analysis', 'TAA Triage', 'CRA Containment'],
            'Avg Time (min)': [0.1, 1.2, 0.8, 0.2],
            'Volume': [data.get('raw_events', 14622), data.get('ada_analyzed', 12428), 
                      data.get('taa_triaged', 10195), data.get('cra_contained', 3396)]
        }
        
        fig = px.scatter(
            time_data, 
            x='Avg Time (min)', 
            y='Volume',
            size='Volume',
            color='Stage',
            title="Processing Time vs Volume by Stage",
            hover_data=['Stage', 'Avg Time (min)', 'Volume']
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

def security_operations_funnel_page():
    """
    Main page for Security Operations Funnel
    """
    st.title("🛡️ Security Operations Funnel")
    st.markdown("**Real-time SOC pipeline visualization with vertical flow**")
    
    # Sample data - replace with real BigQuery data
    funnel_data = {
        'raw_events': 14622,
        'ada_analyzed': 12428,
        'taa_triaged': 10195,
        'cra_contained': 3396
    }
    
    # Create funnel metrics cards
    create_funnel_metrics_cards(funnel_data)
    
    st.markdown("---")
    
    # Create vertical funnel chart
    funnel_fig = create_security_operations_funnel(funnel_data)
    st.plotly_chart(funnel_fig, use_container_width=True)
    
    # Create agent pipeline explanation
    create_agent_pipeline_explanation()
    
    # Create funnel analytics
    create_funnel_analytics(funnel_data)
    
    # Add refresh button
    if st.button("🔄 Refresh Funnel Data"):
        st.rerun()

if __name__ == "__main__":
    security_operations_funnel_page()



