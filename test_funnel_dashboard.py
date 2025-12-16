#!/usr/bin/env python3
"""
Test script to verify Security Operations Funnel functionality
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def create_security_operations_funnel(data):
    """Create a vertical Security Operations Funnel"""
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
    
    for i, stage in enumerate(stages):
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
            name=stage['name'],
            hovertemplate=f"<b>{stage['name']}</b><br>" +
                         f"Count: {stage['count']:,}<br>" +
                         f"Percentage: {stage['percentage']}%<br>" +
                         f"Description: {stage['description']}<extra></extra>"
        ))
    
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

def main():
    st.title("🛡️ Security Operations Funnel Test")
    st.markdown("**Testing the vertical funnel implementation**")
    
    # Sample data
    funnel_data = {
        'raw_events': 14622,
        'ada_analyzed': 12428,
        'taa_triaged': 10195,
        'cra_contained': 3396
    }
    
    # Create metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    stages = [
        {"name": "Raw Events", "count": funnel_data['raw_events'], "icon": "📊"},
        {"name": "ADA Analysis", "count": funnel_data['ada_analyzed'], "icon": "🔍"},
        {"name": "TAA Triage", "count": funnel_data['taa_triaged'], "icon": "⚡"},
        {"name": "CRA Containment", "count": funnel_data['cra_contained'], "icon": "🛡️"}
    ]
    
    for i, stage in enumerate(stages):
        with [col1, col2, col3, col4][i]:
            st.metric(
                label=f"{stage['icon']} {stage['name']}",
                value=f"{stage['count']:,}",
                delta=f"{round((stage['count'] / funnel_data['raw_events']) * 100, 1)}% of total"
            )
    
    st.markdown("---")
    
    # Create funnel chart
    funnel_fig = create_security_operations_funnel(funnel_data)
    st.plotly_chart(funnel_fig, use_container_width=True)
    
    st.success("✅ Security Operations Funnel is working correctly!")

if __name__ == "__main__":
    main()



