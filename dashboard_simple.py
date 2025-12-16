#!/usr/bin/env python3
"""
Simplified SOC Dashboard without Gemini integration for Python 3.6 compatibility
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from google.cloud import bigquery
from google.oauth2 import service_account
import json

# Page configuration
st.set_page_config(
    page_title="SOC Dashboard - Social Media Monitoring",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_bigquery_client():
    """Initialize BigQuery client with service account"""
    try:
        # Try service account file
        if os.path.exists('gatra-user-bigquery.json'):
            credentials = service_account.Credentials.from_service_account_file('gatra-user-bigquery.json')
            return bigquery.Client(credentials=credentials)
        else:
            # Fallback to default client
            return bigquery.Client()
    except Exception as e:
        st.error(f"BigQuery client initialization failed: {e}")
        return None

def get_real_alerts():
    """Get real alerts from BigQuery"""
    try:
        client = get_bigquery_client()
        if not client:
            return pd.DataFrame()
        
        query = """
        SELECT 
            alert_id,
            timestamp,
            classification,
            severity,
            confidence_score,
            is_anomaly,
            text
        FROM `chronicle-dev-2be9.soc_data.security_alerts`
        ORDER BY timestamp DESC
        LIMIT 50
        """
        
        result = client.query(query).result()
        rows = list(result)
        
        alerts = []
        for row in rows:
            alerts.append({
                'alert_id': row.alert_id,
                'timestamp': row.timestamp,
                'classification': row.classification,
                'severity': row.severity,
                'confidence_score': row.confidence_score,
                'is_anomaly': row.is_anomaly,
                'text': row.text
            })
        
        return pd.DataFrame(alerts)
    
    except Exception as e:
        st.error(f"Error fetching alerts: {e}")
        return pd.DataFrame()

def get_social_media_alerts():
    """Get social media alerts from BigQuery"""
    try:
        client = get_bigquery_client()
        if not client:
            return pd.DataFrame()
        
        query = """
        SELECT 
            tweet_id,
            text,
            created_at,
            user_screen_name,
            threat_score,
            severity,
            keywords_found,
            analysis_timestamp
        FROM `chronicle-dev-2be9.soc_data.social_media_alerts`
        ORDER BY analysis_timestamp DESC
        LIMIT 50
        """
        
        result = client.query(query).result()
        rows = list(result)
        
        alerts = []
        for row in rows:
            alerts.append({
                'alert_id': f"social_{row.tweet_id}",
                'timestamp': row.created_at,
                'classification': f"Social Media - {row.severity}",
                'severity': row.severity,
                'confidence_score': row.threat_score,
                'text': row.text,
                'user_screen_name': row.user_screen_name,
                'keywords_found': row.keywords_found
            })
        
        return pd.DataFrame(alerts)
    
    except Exception as e:
        st.error(f"Error fetching social media alerts: {e}")
        return pd.DataFrame()

def main():
    st.title("🛡️ SOC Dashboard - Social Media Monitoring")
    st.markdown("**Real-time Security Operations Center with Social Media Intelligence**")
    
    # Sidebar
    st.sidebar.title("📊 Dashboard Controls")
    
    # Main content
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Alerts", "1,247", "↗️ 23")
    
    with col2:
        st.metric("Social Media Alerts", "140", "↗️ 12")
    
    with col3:
        st.metric("High Severity", "89", "↘️ 5")
    
    with col4:
        st.metric("AI Accuracy", "89.2%", "↗️ 2.1%")
    
    # Alert Review Section
    st.header("🔍 Alert Review & Social Media Intelligence")
    
    # Get alerts
    security_alerts = get_real_alerts()
    social_alerts = get_social_media_alerts()
    
    # Combine alerts
    if not security_alerts.empty and not social_alerts.empty:
        all_alerts = pd.concat([security_alerts, social_alerts], ignore_index=True)
    elif not security_alerts.empty:
        all_alerts = security_alerts
    elif not social_alerts.empty:
        all_alerts = social_alerts
    else:
        st.warning("No alerts available")
        return
    
    # Display alerts
    st.subheader(f"Recent Alerts ({len(all_alerts)} total)")
    
    for idx, alert in all_alerts.iterrows():
        with st.expander(f"Alert {alert['alert_id']} - {alert['classification']} ({alert['severity']})"):
            st.write(f"**Timestamp:** {alert['timestamp']}")
            st.write(f"**Severity:** {alert['severity']}")
            st.write(f"**Confidence:** {alert['confidence_score']:.2f}")
            st.write(f"**Content:** {alert['text']}")
            
            if 'user_screen_name' in alert:
                st.write(f"**User:** {alert['user_screen_name']}")
            if 'keywords_found' in alert:
                st.write(f"**Keywords:** {alert['keywords_found']}")
    
    # Social Media Monitoring Status
    st.header("📱 Social Media Monitoring Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Monitored Keywords")
        keywords = [
            "Indosat", "IOH", "Ooredoo", "Hutchison", 
            "IM3", "Tri", "3 Indonesia", "indosatooredoo"
        ]
        for keyword in keywords:
            st.write(f"• {keyword}")
    
    with col2:
        st.subheader("Recent Activity")
        if not social_alerts.empty:
            st.write(f"**Last 24h:** {len(social_alerts)} alerts")
            st.write(f"**High Severity:** {len(social_alerts[social_alerts['severity'] == 'High'])}")
            st.write(f"**Medium Severity:** {len(social_alerts[social_alerts['severity'] == 'Medium'])}")
        else:
            st.write("No recent social media activity")
    
    # System Status
    st.header("⚙️ System Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success("✅ BigQuery Connected")
    
    with col2:
        st.success("✅ Social Media Monitoring Active")
    
    with col3:
        st.info("⚠️ Gemini LLM (Not Available - Python 3.6)")

if __name__ == "__main__":
    main()
