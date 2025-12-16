#!/usr/bin/env python3
"""
Fixed SOC Dashboard - Social Media Monitoring Only
Focuses on social media alerts with categorization
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

# Import Gemini REST integration (Python 3.6 compatible)
try:
    from gemini_rest_integration import get_gemini_client, GeminiRESTClient
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="SOC Dashboard - Social Media Intelligence",
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
            user_followers,
            retweet_count,
            favorite_count,
            threat_score,
            severity,
            risk_factors,
            keywords_found,
            analysis_timestamp,
            language
        FROM `chronicle-dev-2be9.soc_data.social_media_alerts`
        ORDER BY analysis_timestamp DESC
        LIMIT 100
        """
        
        result = client.query(query).result()
        rows = list(result)
        
        alerts = []
        for row in rows:
            alerts.append({
                'alert_id': f"SM_{row.tweet_id}",
                'timestamp': row.created_at,
                'classification': f"Social Media - {row.severity}",
                'severity': row.severity,
                'confidence_score': row.threat_score,
                'text': row.text,
                'user_screen_name': row.user_screen_name,
                'user_followers': row.user_followers,
                'retweet_count': row.retweet_count,
                'favorite_count': row.favorite_count,
                'keywords_found': row.keywords_found,
                'risk_factors': row.risk_factors,
                'language': row.language
            })
        
        return pd.DataFrame(alerts)
    
    except Exception as e:
        st.error(f"Error fetching social media alerts: {e}")
        return pd.DataFrame()

def categorize_alerts(alerts_df):
    """Categorize alerts by severity and type"""
    if alerts_df.empty:
        return {}
    
    # Handle timezone-aware datetime comparison
    try:
        # Convert to timezone-naive for comparison
        now_naive = datetime.now()
        recent_threshold = now_naive - timedelta(hours=1)
        
        # Convert timestamp column to timezone-naive if needed
        if hasattr(alerts_df['timestamp'].iloc[0], 'tz') and alerts_df['timestamp'].iloc[0].tz is not None:
            timestamps_naive = alerts_df['timestamp'].dt.tz_localize(None)
        else:
            timestamps_naive = alerts_df['timestamp']
        
        recent_mask = timestamps_naive >= recent_threshold
    except:
        # Fallback: no recent filtering if timezone handling fails
        recent_mask = pd.Series([False] * len(alerts_df))
    
    categories = {
        'High Severity': alerts_df[alerts_df['severity'] == 'High'],
        'Medium Severity': alerts_df[alerts_df['severity'] == 'Medium'],
        'Low Severity': alerts_df[alerts_df['severity'] == 'Low'],
        'Recent (Last Hour)': alerts_df[recent_mask],
        'High Engagement': alerts_df[(alerts_df['retweet_count'] > 10) | (alerts_df['favorite_count'] > 50)],
        'Influential Users': alerts_df[alerts_df['user_followers'] > 10000]
    }
    
    return categories

def display_alert_card(alert, index, category_name=""):
    """Display a single alert as a card"""
    severity_colors = {
        'High': '🔴',
        'Medium': '🟡', 
        'Low': '🟢'
    }
    
    with st.container():
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.write(f"**{severity_colors.get(alert['severity'], '⚪')} Alert {alert['alert_id']}**")
            st.write(f"📝 {alert['text'][:150]}{'...' if len(alert['text']) > 150 else ''}")
            
        with col2:
            st.write(f"**👤** {alert['user_screen_name']}")
            st.write(f"**👥** {alert['user_followers']:,} followers")
            
        with col3:
            st.write(f"**📊** RT: {alert['retweet_count']} | ❤️: {alert['favorite_count']}")
            st.write(f"**🏷️** {alert['keywords_found']}")
        
        # Risk factors if available
        if alert.get('risk_factors') and alert['risk_factors'] != '':
            st.write(f"⚠️ **Risk Factors:** {alert['risk_factors']}")
        
        # AI Analysis section (if Gemini is available)
        if GEMINI_AVAILABLE:
            # Create truly unique key by combining alert_id, category, and index
            unique_key = f"ai_btn_{alert['alert_id']}_{category_name}_{index}_{hash(alert['text'])}"
            if st.button(f"🤖 AI Analysis", key=unique_key):
                with st.spinner("Analyzing with Gemini AI..."):
                    client = get_gemini_client()
                    if client:
                        user_info = {
                            'user_screen_name': alert['user_screen_name'],
                            'user_followers': alert['user_followers']
                        }
                        result = client.enrich_social_media_alert(alert['text'], user_info)
                        
                        if result['success']:
                            enrichment = result['enrichment']
                            st.success("✅ AI Analysis Complete")
                            st.write(f"**Threat Level:** {enrichment.get('threat_level', 'Unknown')}")
                            st.write(f"**Brand Impact:** {enrichment.get('brand_impact', 'Unknown')}")
                            st.write(f"**Key Concerns:** {', '.join(enrichment.get('key_concerns', []))}")
                            st.write(f"**Urgency:** {enrichment.get('urgency_level', 'Unknown')}")
                            st.write(f"**Recommendation:** {enrichment.get('recommended_response', 'Manual review')}")
                        else:
                            st.error(f"❌ AI Analysis Failed: {result.get('error', 'Unknown error')}")
                    else:
                        st.error("❌ Gemini AI not available. Check API key.")
        
        st.write("---")  # Simple horizontal line instead of st.divider()

def main():
    st.title("🛡️ SOC Dashboard - Social Media Intelligence")
    st.markdown("**Real-time Social Media Threat Monitoring for Telecom Sector**")
    
    # Sidebar controls
    st.sidebar.title("📊 Dashboard Controls")
    
    # Get alerts
    alerts_df = get_social_media_alerts()
    
    if alerts_df.empty:
        st.warning("No social media alerts available. Check BigQuery connection.")
        return
    
    # Categorize alerts
    categories = categorize_alerts(alerts_df)
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Alerts", len(alerts_df), f"↗️ {len(alerts_df)}")
    
    with col2:
        high_severity = len(categories['High Severity'])
        st.metric("High Severity", high_severity, f"🔴 {high_severity}")
    
    with col3:
        recent_alerts = len(categories['Recent (Last Hour)'])
        st.metric("Recent (1h)", recent_alerts, f"⏰ {recent_alerts}")
    
    with col4:
        influential = len(categories['Influential Users'])
        st.metric("Influential Users", influential, f"👑 {influential}")
    
    # Alert categorization using expanders (compatible with Streamlit 1.10.0)
    
    # High Severity Alerts
    with st.expander(f"🔴 High Severity Alerts ({len(categories['High Severity'])} alerts)", expanded=True):
        high_alerts = categories['High Severity']
        if not high_alerts.empty:
            st.write(f"**{len(high_alerts)} high severity alerts found**")
            for idx, alert in high_alerts.iterrows():
                display_alert_card(alert, idx, "High")
        else:
            st.info("No high severity alerts")
    
    # Medium Severity Alerts
    with st.expander(f"🟡 Medium Severity Alerts ({len(categories['Medium Severity'])} alerts)", expanded=False):
        medium_alerts = categories['Medium Severity']
        if not medium_alerts.empty:
            st.write(f"**{len(medium_alerts)} medium severity alerts found**")
            for idx, alert in medium_alerts.iterrows():
                display_alert_card(alert, idx, "Medium")
        else:
            st.info("No medium severity alerts")
    
    # Low Severity Alerts
    with st.expander(f"🟢 Low Severity Alerts ({len(categories['Low Severity'])} alerts)", expanded=False):
        low_alerts = categories['Low Severity']
        if not low_alerts.empty:
            st.write(f"**{len(low_alerts)} low severity alerts found**")
            for idx, alert in low_alerts.iterrows():
                display_alert_card(alert, idx, "Low")
        else:
            st.info("No low severity alerts")
    
    # All Alerts
    with st.expander(f"📊 All Social Media Alerts ({len(alerts_df)} total)", expanded=False):
        st.write(f"**{len(alerts_df)} total alerts**")
        
        # Show first 20 alerts to avoid overwhelming
        display_count = min(20, len(alerts_df))
        st.write(f"Showing first {display_count} alerts:")
        
        for idx, alert in alerts_df.head(display_count).iterrows():
            display_alert_card(alert, idx, "All")
        
        if len(alerts_df) > 20:
            st.info(f"Showing first 20 of {len(alerts_df)} total alerts. Use expanders above to see specific categories.")
    
    # Monitoring status
    st.header("📱 Social Media Monitoring Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Monitored Keywords")
        keywords = [
            "Indosat", "IOH", "Ooredoo", "Hutchison", 
            "IM3", "Tri", "3 Indonesia", "indosatooredoo"
        ]
        for keyword in keywords:
            st.write(f"• {keyword}")
    
    with col2:
        st.subheader("📈 Activity Summary")
        st.write(f"**Total Alerts:** {len(alerts_df)}")
        st.write(f"**High Severity:** {len(categories['High Severity'])}")
        st.write(f"**Medium Severity:** {len(categories['Medium Severity'])}")
        st.write(f"**Low Severity:** {len(categories['Low Severity'])}")
        st.write(f"**Recent (1h):** {len(categories['Recent (Last Hour)'])}")
        st.write(f"**High Engagement:** {len(categories['High Engagement'])}")
    
    # System status
    st.header("⚙️ System Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success("✅ BigQuery Connected")
    
    with col2:
        st.success("✅ Social Media Monitoring Active")
    
    with col3:
        if GEMINI_AVAILABLE:
            st.success("✅ Gemini LLM (REST API - Python 3.6 Compatible)")
        else:
            st.warning("⚠️ Gemini LLM (REST integration not loaded)")

if __name__ == "__main__":
    main()
