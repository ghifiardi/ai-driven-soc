#!/usr/bin/env python3
"""
Enhanced Security Operations Funnel with False Positive Analysis
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta

def create_improved_funnel_analysis():
    """Create funnel with false positive analysis and improvement recommendations"""
    
    # Current vs Improved metrics
    current_metrics = {
        'raw_events': 14622,
        'ada_analyzed': 12428,
        'taa_triaged': 10195,
        'cra_contained': 3396,
        'false_positive_rate': 77.0
    }
    
    improved_metrics = {
        'raw_events': 14622,
        'ada_analyzed': 11000,  # Better filtering
        'taa_triaged': 7500,    # Better context
        'cra_contained': 4500,  # More accurate containment
        'false_positive_rate': 35.0  # Target improvement
    }
    
    # Create comparison chart
    stages = ['Raw Events', 'ADA Analysis', 'TAA Triage', 'CRA Containment']
    current_values = [current_metrics['raw_events'], current_metrics['ada_analyzed'], 
                     current_metrics['taa_triaged'], current_metrics['cra_contained']]
    improved_values = [improved_metrics['raw_events'], improved_metrics['ada_analyzed'], 
                      improved_metrics['taa_triaged'], improved_metrics['cra_contained']]
    
    fig = go.Figure()
    
    # Current funnel
    fig.add_trace(go.Bar(
        name='Current Performance',
        x=current_values,
        y=stages,
        orientation='h',
        marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
        text=[f"{val:,}" for val in current_values],
        textposition='inside'
    ))
    
    # Improved funnel
    fig.add_trace(go.Bar(
        name='Target Performance',
        x=improved_values,
        y=stages,
        orientation='h',
        marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
        opacity=0.6,
        text=[f"{val:,}" for val in improved_values],
        textposition='inside'
    ))
    
    fig.update_layout(
        title="🛡️ Current vs Target Security Operations Funnel",
        xaxis_title="Number of Events",
        yaxis_title="Processing Stage",
        barmode='group',
        height=500
    )
    
    return fig, current_metrics, improved_metrics

def create_false_positive_breakdown(current_metrics):
    """Create detailed false positive analysis"""
    
    # Calculate false positives at each stage
    ada_false_positives = current_metrics['ada_analyzed'] - current_metrics['taa_triaged']
    taa_false_positives = current_metrics['taa_triaged'] - current_metrics['cra_contained']
    
    breakdown_data = {
        'Stage': ['ADA Analysis', 'TAA Triage', 'CRA Containment'],
        'Events_Processed': [current_metrics['ada_analyzed'], current_metrics['taa_triaged'], current_metrics['cra_contained']],
        'False_Positives': [ada_false_positives, taa_false_positives, 0],
        'True_Positives': [current_metrics['taa_triaged'], current_metrics['cra_contained'], current_metrics['cra_contained']]
    }
    
    fig = px.bar(
        breakdown_data,
        x='Stage',
        y=['False_Positives', 'True_Positives'],
        title="False Positive Breakdown by Stage",
        color_discrete_map={'False_Positives': '#ff6b6b', 'True_Positives': '#51cf66'},
        labels={'value': 'Number of Events', 'variable': 'Type'}
    )
    
    return fig

def create_improvement_recommendations():
    """Create improvement recommendations"""
    
    recommendations = {
        'Area': [
            'ADA Model Training',
            'Feature Engineering', 
            'Threat Intelligence',
            'Context Enrichment',
            'Rule Optimization',
            'Feedback Loop'
        ],
        'Current_FPR': [85, 82, 78, 75, 72, 77],
        'Target_FPR': [70, 65, 55, 50, 45, 35],
        'Impact': ['High', 'High', 'Medium', 'Medium', 'Low', 'High'],
        'Effort': ['High', 'Medium', 'Low', 'Medium', 'Low', 'Medium']
    }
    
    df = pd.DataFrame(recommendations)
    
    fig = px.scatter(
        df,
        x='Effort',
        y='Impact',
        size='Current_FPR',
        color='Target_FPR',
        hover_data=['Area', 'Current_FPR', 'Target_FPR'],
        title="Improvement Recommendations Matrix",
        labels={'Current_FPR': 'Current False Positive Rate', 'Target_FPR': 'Target False Positive Rate'}
    )
    
    return fig, df

def main():
    st.title("🔍 False Positive Rate Analysis & Improvement")
    st.markdown("**Understanding and reducing the 77% False Positive Rate in SOC Pipeline**")
    
    # Create analysis
    funnel_fig, current_metrics, improved_metrics = create_improved_funnel_analysis()
    breakdown_fig = create_false_positive_breakdown(current_metrics)
    recommendations_fig, recommendations_df = create_improvement_recommendations()
    
    # Display current vs target metrics
    st.header("📊 Current vs Target Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current False Positive Rate", f"{current_metrics['false_positive_rate']:.1f}%", 
                 f"{current_metrics['false_positive_rate'] - improved_metrics['false_positive_rate']:.1f}%", 
                 delta_color="inverse")
    
    with col2:
        st.metric("Target False Positive Rate", f"{improved_metrics['false_positive_rate']:.1f}%", 
                 f"Target", delta_color="normal")
    
    with col3:
        improvement = current_metrics['false_positive_rate'] - improved_metrics['false_positive_rate']
        st.metric("Improvement Potential", f"{improvement:.1f}%", f"Reduction")
    
    with col4:
        efficiency_gain = (improved_metrics['cra_contained'] / improved_metrics['raw_events']) * 100
        current_efficiency = (current_metrics['cra_contained'] / current_metrics['raw_events']) * 100
        st.metric("Efficiency Gain", f"{efficiency_gain - current_efficiency:.1f}%", f"Improvement")
    
    # Display funnel comparison
    st.plotly_chart(funnel_fig, use_container_width=True)
    
    # False positive breakdown
    st.header("🔍 False Positive Breakdown by Stage")
    st.plotly_chart(breakdown_fig, use_container_width=True)
    
    # Root cause analysis
    st.header("🔍 Root Cause Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Current Issues")
        st.markdown("""
        **ADA (Anomaly Detection):**
        - Over-sensitive ML models
        - Insufficient training data
        - Poor feature engineering
        
        **TAA (Triage & Analysis):**
        - Limited context information
        - Missing threat intelligence
        - Incomplete rule coverage
        
        **CRA (Containment):**
        - Conservative containment rules
        - Manual verification bottlenecks
        - Limited automation
        """)
    
    with col2:
        st.subheader("Impact on Operations")
        st.markdown("""
        **High False Positive Rate (77%) means:**
        - **Analyst Fatigue:** 77% of alerts are false alarms
        - **Resource Waste:** Time spent on non-threats
        - **Alert Fatigue:** Real threats may be missed
        - **Cost Impact:** Unnecessary containment actions
        - **Trust Issues:** Analysts lose confidence in system
        """)
    
    # Improvement recommendations
    st.header("🚀 Improvement Recommendations")
    st.plotly_chart(recommendations_fig, use_container_width=True)
    
    # Detailed recommendations table
    st.subheader("📋 Detailed Action Plan")
    st.dataframe(recommendations_df, use_container_width=True)
    
    # Implementation timeline
    st.header("⏱️ Implementation Timeline")
    
    timeline_data = {
        'Phase': ['Phase 1 (Weeks 1-2)', 'Phase 2 (Weeks 3-4)', 'Phase 3 (Weeks 5-6)', 'Phase 4 (Weeks 7-8)'],
        'Actions': [
            'Enhance ADA model training with more data',
            'Implement threat intelligence integration',
            'Optimize TAA context enrichment',
            'Deploy feedback loop and continuous learning'
        ],
        'Expected_FPR_Reduction': ['77% → 70%', '70% → 60%', '60% → 50%', '50% → 35%'],
        'Priority': ['High', 'High', 'Medium', 'High']
    }
    
    timeline_df = pd.DataFrame(timeline_data)
    st.dataframe(timeline_df, use_container_width=True)
    
    # Success metrics
    st.header("📈 Success Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Efficiency", "23.2%", "Events to Containment")
    
    with col2:
        st.metric("Target Efficiency", "30.8%", "Events to Containment")
    
    with col3:
        st.metric("Efficiency Gain", "+7.6%", "Improvement")

if __name__ == "__main__":
    main()



