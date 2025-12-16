#!/usr/bin/env python3
"""
Real Twitter SOC Dashboard
Displays live Twitter data for telecom security monitoring
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from google.cloud import bigquery
import time

# Page configuration
st.set_page_config(
    page_title="Real Twitter SOC Monitor",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_bigquery_client():
    """Get BigQuery client"""
    try:
        service_account_paths = [
            'Service Account BigQuery/gatra-user-bigquery.json',
            'Service Account BigQuery/gatra-user-gemini.json',
            'Service Account BigQuery/sa-gatra-bigquery.json'
        ]
        
        for path in service_account_paths:
            if os.path.exists(path):
                return bigquery.Client.from_service_account_json(path)
        
        return bigquery.Client()
    except Exception as e:
        st.error(f"BigQuery client initialization failed: {e}")
        return None

def get_twitter_alerts(client, limit=200):
    """Get real Twitter alerts from BigQuery"""
    if not client:
        return pd.DataFrame()
    
    try:
        query = f"""
        SELECT 
            tweet_id,
            text,
            created_at,
            username,
            user_name,
            user_followers,
            user_verified,
            retweet_count,
            like_count,
            reply_count,
            quote_count,
            language,
            query_used,
            threat_score,
            severity,
            category,
            risk_factors,
            keywords_found,
            analysis_timestamp
        FROM `chronicle-dev-2be9.soc_data.twitter_real_alerts`
        ORDER BY created_at DESC
        LIMIT {limit}
        """
        
        result = client.query(query).result()
        
        data = []
        for row in result:
            data.append({
                'tweet_id': row.tweet_id,
                'text': row.text,
                'created_at': row.created_at,
                'username': row.username,
                'user_name': row.user_name,
                'user_followers': row.user_followers,
                'user_verified': row.user_verified,
                'retweet_count': row.retweet_count,
                'like_count': row.like_count,
                'reply_count': row.reply_count,
                'quote_count': row.quote_count,
                'language': row.language,
                'query_used': row.query_used,
                'threat_score': row.threat_score,
                'severity': row.severity,
                'category': row.category,
                'risk_factors': row.risk_factors,
                'keywords_found': row.keywords_found,
                'analysis_timestamp': row.analysis_timestamp
            })
        
        return pd.DataFrame(data)
        
    except Exception as e:
        st.error(f"Failed to fetch Twitter alerts: {e}")
        return pd.DataFrame()

def display_twitter_metrics(alerts_df):
    """Display Twitter monitoring metrics"""
    if alerts_df.empty:
        st.warning("No Twitter data available. Run the Twitter monitor first.")
        return
    
    # Calculate metrics
    total_alerts = len(alerts_df)
    high_threat = len(alerts_df[alerts_df['severity'] == 'High'])
    cyber_threats = len(alerts_df[alerts_df['category'] == 'Cyber Threat'])
    complaints = len(alerts_df[alerts_df['category'] == 'Customer Complaint'])
    verified_users = len(alerts_df[alerts_df['user_verified'] == True])
    avg_threat_score = alerts_df['threat_score'].mean()
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Total Alerts",
            value=total_alerts,
            delta=f"{len(alerts_df[alerts_df['created_at'] > datetime.now() - timedelta(hours=1)])} in last hour"
        )
    
    with col2:
        st.metric(
            label="🔴 High Threat",
            value=high_threat,
            delta=f"{len(alerts_df[(alerts_df['severity'] == 'High') & (alerts_df['created_at'] > datetime.now() - timedelta(hours=1))])} recent"
        )
    
    with col3:
        st.metric(
            label="🛡️ Cyber Threats",
            value=cyber_threats,
            delta=f"{len(alerts_df[(alerts_df['category'] == 'Cyber Threat') & (alerts_df['created_at'] > datetime.now() - timedelta(hours=1))])} recent"
        )
    
    with col4:
        st.metric(
            label="📈 Avg Threat Score",
            value=f"{avg_threat_score:.2f}",
            delta=f"{verified_users} verified users"
        )

def display_twitter_charts(alerts_df):
    """Display Twitter monitoring charts"""
    if alerts_df.empty:
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Severity distribution
        severity_counts = alerts_df['severity'].value_counts()
        fig_severity = px.pie(
            values=severity_counts.values,
            names=severity_counts.index,
            title="Alert Severity Distribution",
            color_discrete_map={
                'High': '#FF6B6B',
                'Medium': '#FFE66D',
                'Low': '#4ECDC4'
            }
        )
        st.plotly_chart(fig_severity, use_container_width=True)
    
    with col2:
        # Category distribution
        category_counts = alerts_df['category'].value_counts()
        fig_category = px.bar(
            x=category_counts.index,
            y=category_counts.values,
            title="Alert Categories",
            color=category_counts.values,
            color_continuous_scale='Viridis'
        )
        fig_category.update_layout(showlegend=False)
        st.plotly_chart(fig_category, use_container_width=True)
    
    # Timeline chart
    if len(alerts_df) > 1:
        alerts_df['hour'] = pd.to_datetime(alerts_df['created_at']).dt.floor('H')
        timeline_data = alerts_df.groupby(['hour', 'severity']).size().reset_index(name='count')
        
        fig_timeline = px.line(
            timeline_data,
            x='hour',
            y='count',
            color='severity',
            title="Alerts Timeline (Last 24 Hours)",
            color_discrete_map={
                'High': '#FF6B6B',
                'Medium': '#FFE66D',
                'Low': '#4ECDC4'
            }
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

def display_twitter_alert_card(alert, index):
    """Display individual Twitter alert card"""
    # Determine color based on severity
    if alert['severity'] == 'High':
        border_color = '#FF6B6B'
        severity_icon = '🔴'
    elif alert['severity'] == 'Medium':
        border_color = '#FFE66D'
        severity_icon = '🟡'
    else:
        border_color = '#4ECDC4'
        severity_icon = '🟢'
    
    # Format created time
    created_time = pd.to_datetime(alert['created_at']).strftime('%Y-%m-%d %H:%M:%S')
    
    with st.container():
        st.markdown(f"""
        <div style="
            border-left: 4px solid {border_color};
            padding: 15px;
            margin: 10px 0;
            background-color: #f8f9fa;
            border-radius: 5px;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h4 style="margin: 0; color: #2c3e50;">
                        {severity_icon} {alert['severity']} Threat - {alert['category']}
                    </h4>
                    <p style="margin: 5px 0; color: #7f8c8d; font-size: 12px;">
                        @{alert['username']} • {created_time} • Threat Score: {alert['threat_score']:.2f}
                    </p>
                </div>
                <div style="text-align: right;">
                    <span style="background-color: {border_color}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 11px;">
                        {alert['user_followers']:,} followers
                    </span>
                </div>
            </div>
            <p style="margin: 10px 0; color: #34495e;">
                {alert['text'][:200]}{'...' if len(alert['text']) > 200 else ''}
            </p>
            <div style="display: flex; gap: 15px; font-size: 12px; color: #7f8c8d;">
                <span>❤️ {alert['like_count']}</span>
                <span>🔄 {alert['retweet_count']}</span>
                <span>💬 {alert['reply_count']}</span>
                <span>📊 {alert['quote_count']}</span>
                {f"<span>✅ Verified</span>" if alert['user_verified'] else ""}
            </div>
            {f"<p style='margin: 5px 0; font-size: 11px; color: #e74c3c;'><strong>Risk Factors:</strong> {alert['risk_factors']}</p>" if alert['risk_factors'] else ""}
            {f"<p style='margin: 5px 0; font-size: 11px; color: #3498db;'><strong>Keywords:</strong> {alert['keywords_found']}</p>" if alert['keywords_found'] else ""}
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main dashboard function"""
    st.title("🐦 Real Twitter SOC Monitor")
    st.markdown("**Live Twitter monitoring for Indosat, IOH, IM3, and Tri security threats**")
    
    # Sidebar controls
    st.sidebar.header("📊 Dashboard Controls")
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Twitter Data"):
        st.rerun()
    
    # Data limit
    limit = st.sidebar.slider("Number of alerts to display", 50, 500, 200)
    
    # Time filter
    time_filter = st.sidebar.selectbox(
        "Time Filter",
        ["Last 24 hours", "Last 7 days", "All time"],
        index=0
    )
    
    # Get BigQuery client
    client = get_bigquery_client()
    
    if not client:
        st.error("❌ Cannot connect to BigQuery. Please check your credentials.")
        return
    
    # Load Twitter data
    with st.spinner("Loading Twitter data..."):
        alerts_df = get_twitter_alerts(client, limit)
    
    if alerts_df.empty:
        st.warning("⚠️ No Twitter data found. Please run the Twitter monitor first:")
        st.code("python twitter_real_monitor.py")
        
        st.info("📋 Setup Instructions:")
        st.markdown("""
        1. **Get Twitter API Access**: Visit [Twitter Developer Portal](https://developer.twitter.com/en/portal/dashboard)
        2. **Create App**: Create a new Twitter App with API v2 access
        3. **Get Bearer Token**: Generate Bearer Token from Keys & Tokens tab
        4. **Set Environment Variable**: `export TWITTER_BEARER_TOKEN=your_token_here`
        5. **Run Monitor**: `python twitter_real_monitor.py`
        """)
        return
    
    # Apply time filter
    if time_filter == "Last 24 hours":
        cutoff_time = datetime.now() - timedelta(hours=24)
        alerts_df = alerts_df[pd.to_datetime(alerts_df['created_at']) > cutoff_time]
    elif time_filter == "Last 7 days":
        cutoff_time = datetime.now() - timedelta(days=7)
        alerts_df = alerts_df[pd.to_datetime(alerts_df['created_at']) > cutoff_time]
    
    # Display metrics
    st.header("📈 Real-Time Metrics")
    display_twitter_metrics(alerts_df)
    
    # Display charts
    st.header("📊 Analytics")
    display_twitter_charts(alerts_df)
    
    # Display alerts
    st.header("🚨 Recent Twitter Alerts")
    
    if alerts_df.empty:
        st.info("No alerts found for the selected time period.")
        return
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        severity_filter = st.selectbox(
            "Filter by Severity",
            ["All", "High", "Medium", "Low"],
            index=0
        )
    
    with col2:
        category_filter = st.selectbox(
            "Filter by Category",
            ["All"] + list(alerts_df['category'].unique()),
            index=0
        )
    
    with col3:
        min_threat_score = st.slider(
            "Minimum Threat Score",
            0.0, 1.0, 0.0, 0.1
        )
    
    # Apply filters
    filtered_df = alerts_df.copy()
    
    if severity_filter != "All":
        filtered_df = filtered_df[filtered_df['severity'] == severity_filter]
    
    if category_filter != "All":
        filtered_df = filtered_df[filtered_df['category'] == category_filter]
    
    filtered_df = filtered_df[filtered_df['threat_score'] >= min_threat_score]
    
    # Display filtered alerts
    st.write(f"Showing {len(filtered_df)} alerts (filtered from {len(alerts_df)} total)")
    
    for index, (_, alert) in enumerate(filtered_df.iterrows()):
        display_twitter_alert_card(alert, index)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d; font-size: 12px;'>
        🐦 Real Twitter SOC Monitor | Powered by Twitter API v2 & BigQuery<br>
        Monitoring: Indosat, IOH, IM3, Tri, and related telecom keywords
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
























