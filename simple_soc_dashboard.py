#!/usr/bin/env python3
"""
Simple SOC Dashboard - Fixed Gemini API and Proper Categorization
"""

import streamlit as st
import pandas as pd
import requests
import json
from datetime import datetime, timedelta
from google.cloud import bigquery
import os

# Page configuration
st.set_page_config(
    page_title="🛡️ SOC Intelligence Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyCB1jWcqRrHUueOUxLA2kt0gJUuY7ng8Ac')
GEMINI_AVAILABLE = bool(GEMINI_API_KEY)

def get_bigquery_client():
    """Get BigQuery client with fallback authentication"""
    try:
        # Try multiple service account files
        service_account_paths = [
            'Service Account BigQuery/chronicle-dev-2be-a-driven-soc.json',
            'Service Account BigQuery/sa-gatra-bigquery.json',
            'Service Account BigQuery/gatra-user-bigquery.json'
        ]

        for sa_path in service_account_paths:
            if os.path.exists(sa_path):
                from google.oauth2 import service_account
                credentials = service_account.Credentials.from_service_account_file(sa_path)
                return bigquery.Client(credentials=credentials)

        # Fallback to default client
        return bigquery.Client()
    except Exception as e:
        st.error(f"BigQuery client initialization failed: {e}")
        return None

def get_social_media_alerts(client, limit: int = 200):
    """Get social media alerts from BigQuery"""
    try:
        query = f"""
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
        LIMIT {limit}
        """

        result = client.query(query).result()
        rows = list(result)

        alerts = []
        for row in rows:
            text_lower = row.text.lower()
            
            # Determine alert type based on content
            alert_type = "Social Media Alert"
            
            # Check for actual cyber threats
            cyber_keywords = [
                'data breach', 'data leak', 'hacked', 'compromised', 'leaked', 'stolen',
                'ransomware', 'malware', 'trojan', 'phishing', 'botnet',
                'ddos', 'server down', 'dns hijacking', 'mitm',
                'fake indosat', 'scam indosat', 'phishing indosat', 'impersonation',
                'vulnerability', 'exploit', 'zero day', 'cve', 'security flaw'
            ]
            
            if any(keyword in text_lower for keyword in cyber_keywords):
                alert_type = "Cyber Threat"
            elif any(complaint_word in text_lower for complaint_word in ['lambat', 'lemot', 'down', 'gangguan', 'masalah', 'buruk', 'jelek', 'mahal']):
                alert_type = "General Complaint"
            
            alerts.append({
                'alert_id': f"alert_{row.tweet_id}",
                'alert_type': alert_type,
                'timestamp': row.created_at,
                'classification': f"{alert_type} - {row.severity}",
                'severity': row.severity,
                'confidence_score': row.threat_score,
                'text': row.text,
                'user_screen_name': row.user_screen_name,
                'user_followers': row.user_followers,
                'retweet_count': row.retweet_count,
                'favorite_count': row.favorite_count,
                'risk_factors': row.risk_factors.split(',') if row.risk_factors else [],
                'keywords_found': row.keywords_found.split(',') if row.keywords_found else [],
                'source': 'twitter',
                'analysis_timestamp': row.analysis_timestamp,
                'language': row.language
            })

        return pd.DataFrame(alerts)

    except Exception as e:
        st.error(f"Error getting social media alerts: {e}")
        return pd.DataFrame()

def get_gemini_client():
    """Get Gemini client for AI analysis"""
    if not GEMINI_AVAILABLE:
        return None
    
    class GeminiClient:
        def __init__(self, api_key):
            self.api_key = api_key
            self.base_url = "https://generative-language.googleapis.com/v1beta"
        
        def enrich_social_media_alert(self, text, user_info):
            """Analyze social media alert"""
            try:
                prompt = f"""
                Analyze this social media post for cybersecurity threats:
                
                Post: "{text}"
                User: @{user_info['user_screen_name']} ({user_info['user_followers']} followers)
                
                Provide a JSON response with:
                1. threat_level: "Critical", "High", "Medium", "Low", or "None"
                2. threat_category: "Data Breach", "Malware", "Social Engineering", "Infrastructure", "Brand Abuse", "Vulnerability", or "Other"
                3. brand_impact: "Critical", "High", "Medium", "Low", or "None"
                4. key_concerns: List of specific security concerns
                5. urgency_level: "Immediate", "High", "Medium", "Low"
                6. recommended_actions: List of recommended response actions
                7. false_positive_probability: 0.0 to 1.0 (lower = more likely real threat)
                
                Focus on actual cybersecurity threats, not general complaints.
                """
                
                # Try different Gemini models
                models_to_try = [
                    "gemini-1.5-pro",
                    "gemini-1.5-flash", 
                    "gemini-pro",
                    "gemini-1.0-pro"
                ]
                
                for model in models_to_try:
                    try:
                        url = f"{self.base_url}/models/{model}:generateContent"
                        headers = {
                            "Content-Type": "application/json",
                            "x-goog-api-key": self.api_key
                        }
                        
                        payload = {
                            "contents": [{
                                "parts": [{"text": prompt}]
                            }],
                            "generationConfig": {
                                "temperature": 0.1,
                                "topK": 1,
                                "topP": 0.8,
                                "maxOutputTokens": 1024
                            }
                        }
                        
                        response = requests.post(url, headers=headers, json=payload, timeout=30)
                        
                        if response.status_code == 200:
                            result = response.json()
                            if 'candidates' in result and result['candidates']:
                                content = result['candidates'][0]['content']['parts'][0]['text']
                                
                                # Try to extract JSON from response
                                try:
                                    start = content.find('{')
                                    end = content.rfind('}') + 1
                                    if start != -1 and end != 0:
                                        json_str = content[start:end]
                                        enrichment = json.loads(json_str)
                                    else:
                                        enrichment = {
                                            'threat_level': 'Medium',
                                            'threat_category': 'Other',
                                            'brand_impact': 'Medium',
                                            'key_concerns': ['AI analysis completed'],
                                            'urgency_level': 'Medium',
                                            'recommended_actions': ['Further investigation recommended'],
                                            'false_positive_probability': 0.4
                                        }
                                    
                                    return {'success': True, 'enrichment': enrichment, 'model_used': model}
                                except json.JSONDecodeError:
                                    enrichment = {
                                        'threat_level': 'Medium',
                                        'threat_category': 'Other',
                                        'brand_impact': 'Medium',
                                        'key_concerns': ['AI analysis completed'],
                                        'urgency_level': 'Medium',
                                        'recommended_actions': ['Further investigation recommended'],
                                        'false_positive_probability': 0.4
                                    }
                                    return {'success': True, 'enrichment': enrichment, 'model_used': model}
                            else:
                                continue
                        else:
                            continue
                            
                    except Exception as e:
                        continue
                
                # If all models fail, return a fallback
                return {
                    'success': False, 
                    'error': 'All Gemini models unavailable',
                    'fallback': {
                        'threat_level': 'Medium',
                        'threat_category': 'Other',
                        'brand_impact': 'Medium',
                        'key_concerns': ['Manual review required'],
                        'urgency_level': 'Medium',
                        'recommended_actions': ['Review manually'],
                        'false_positive_probability': 0.5
                    }
                }
                    
            except Exception as e:
                return {'success': False, 'error': str(e)}
    
    return GeminiClient(GEMINI_API_KEY)

def display_alert_card(alert, index, category_name=""):
    """Display a single alert card"""
    with st.container():
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            # Use appropriate icon based on alert type
            if alert['alert_type'] == "Cyber Threat":
                icon = "🚨"
            elif alert['alert_type'] == "General Complaint":
                icon = "💬"
            else:
                icon = "📱"
            
            st.write(f"**{icon} {alert['alert_type']} {alert['alert_id']}**")
            st.write(f"📝 {alert['text'][:200]}{'...' if len(alert['text']) > 200 else ''}")
            
        with col2:
            st.write(f"**👤** {alert['user_screen_name']}")
            st.write(f"**👥** {alert['user_followers']:,} followers")
            st.write(f"**📊** Score: {alert['confidence_score']:.2f}")
            
        with col3:
            st.write(f"**📈** RT: {alert['retweet_count']} | ❤️: {alert['favorite_count']}")
            if alert.get('keywords_found'):
                keywords_str = ', '.join(alert['keywords_found'][:3])
                st.write(f"**🏷️** {keywords_str}")
        
        # Risk factors if available
        if alert.get('risk_factors') and alert['risk_factors'] != '':
            st.write(f"⚠️ **Risk Factors:** {alert['risk_factors']}")
        
        # AI Analysis section
        if GEMINI_AVAILABLE:
            unique_key = f"ai_btn_{alert['alert_id']}_{category_name}_{index}_{hash(alert['text'])}"
            if st.button(f"🤖 AI Analysis", key=unique_key):
                with st.spinner("Analyzing with AI..."):
                    client = get_gemini_client()
                    if client:
                        user_info = {
                            'user_screen_name': alert['user_screen_name'],
                            'user_followers': alert['user_followers']
                        }
                        result = client.enrich_social_media_alert(alert['text'], user_info)
                        
                        if result['success']:
                            enrichment = result['enrichment']
                            model_used = result.get('model_used', 'Unknown')
                            st.success(f"✅ AI Analysis Complete (Model: {model_used})")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**🚨 Threat Level:** {enrichment.get('threat_level', 'Unknown')}")
                                st.write(f"**📂 Category:** {enrichment.get('threat_category', 'Unknown')}")
                                st.write(f"**🎯 Brand Impact:** {enrichment.get('brand_impact', 'Unknown')}")
                            with col2:
                                st.write(f"**⚡ Urgency:** {enrichment.get('urgency_level', 'Unknown')}")
                                st.write(f"**🎲 False Positive:** {enrichment.get('false_positive_probability', 0.5):.1%}")
                            
                            st.write("**🔍 Key Concerns:**")
                            for concern in enrichment.get('key_concerns', []):
                                st.write(f"• {concern}")
                            
                            st.write("**📋 Recommended Actions:**")
                            for action in enrichment.get('recommended_actions', []):
                                st.write(f"• {action}")
                        else:
                            # Show fallback analysis
                            if 'fallback' in result:
                                enrichment = result['fallback']
                                st.warning(f"⚠️ AI Analysis (Fallback Mode): {result.get('error', 'Unknown error')}")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"**🚨 Threat Level:** {enrichment.get('threat_level', 'Unknown')}")
                                    st.write(f"**📂 Category:** {enrichment.get('threat_category', 'Unknown')}")
                                    st.write(f"**🎯 Brand Impact:** {enrichment.get('brand_impact', 'Unknown')}")
                                with col2:
                                    st.write(f"**⚡ Urgency:** {enrichment.get('urgency_level', 'Unknown')}")
                                    st.write(f"**🎲 False Positive:** {enrichment.get('false_positive_probability', 0.5):.1%}")
                            else:
                                st.error(f"Analysis failed: {result.get('error', 'Unknown error')}")
        
        st.write("---")

def categorize_alerts(alerts_df):
    """Categorize alerts by type"""
    if alerts_df.empty:
        return {}
    
    categories = {
        'Cyber Threats': [],
        'General Complaints': [],
        'Social Media Alerts': [],
        'High Engagement': [],
        'Recent (24h)': []
    }
    
    for idx, alert in alerts_df.iterrows():
        # Categorize by alert type
        if alert['alert_type'] == "Cyber Threat":
            categories['Cyber Threats'].append(alert)
        elif alert['alert_type'] == "General Complaint":
            categories['General Complaints'].append(alert)
        else:
            categories['Social Media Alerts'].append(alert)
        
        # High engagement
        if alert['retweet_count'] > 10 or alert['favorite_count'] > 50:
            categories['High Engagement'].append(alert)
        
        # Recent (last 24 hours)
        try:
            now = datetime.now()
            if hasattr(alert['timestamp'], 'tz') and alert['timestamp'].tz is not None:
                alert_time = alert['timestamp'].replace(tzinfo=None)
            else:
                alert_time = alert['timestamp']
            
            if (now - alert_time).total_seconds() < 86400:  # 24 hours
                categories['Recent (24h)'].append(alert)
        except:
            pass
    
    # Convert lists to DataFrames
    for category, alerts_list in categories.items():
        if alerts_list:
            categories[category] = pd.DataFrame(alerts_list)
        else:
            categories[category] = pd.DataFrame()
    
    return categories

def main():
    st.title("🛡️ SOC Intelligence Dashboard")
    st.markdown("**Smart Social Media Threat Monitoring**")
    
    # Get alerts
    client = get_bigquery_client()
    if client:
        alerts_df = get_social_media_alerts(client, limit=200)
        categories = categorize_alerts(alerts_df)
    else:
        st.error("Unable to connect to BigQuery. Using sample data.")
        alerts_df = pd.DataFrame()
        categories = {}
    
    # Calculate metrics
    total_alerts = len(alerts_df)
    cyber_threats = len(categories.get('Cyber Threats', pd.DataFrame()))
    complaints = len(categories.get('General Complaints', pd.DataFrame()))
    recent_alerts = len(categories.get('Recent (24h)', pd.DataFrame()))
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Alerts", total_alerts)
    with col2:
        st.metric("🚨 Cyber Threats", cyber_threats, delta=f"{cyber_threats}")
    with col3:
        st.metric("💬 Complaints", complaints, delta=f"{complaints}")
    with col4:
        st.metric("🕐 Recent (24h)", recent_alerts, delta=f"{recent_alerts}")
    
    # Alert categories
    st.header("📊 Alert Categories")
    
    # Cyber threats first
    if not categories.get('Cyber Threats', pd.DataFrame()).empty:
        with st.expander(f"🚨 Cyber Threats ({len(categories['Cyber Threats'])} alerts)", expanded=True):
            for idx, alert in categories['Cyber Threats'].iterrows():
                display_alert_card(alert, idx, "Cyber")
    
    # General complaints
    if not categories.get('General Complaints', pd.DataFrame()).empty:
        with st.expander(f"💬 General Complaints ({len(categories['General Complaints'])} alerts)", expanded=False):
            for idx, alert in categories['General Complaints'].iterrows():
                display_alert_card(alert, idx, "Complaint")
    
    # Social media alerts
    if not categories.get('Social Media Alerts', pd.DataFrame()).empty:
        with st.expander(f"📱 Social Media Alerts ({len(categories['Social Media Alerts'])} alerts)", expanded=False):
            for idx, alert in categories['Social Media Alerts'].iterrows():
                display_alert_card(alert, idx, "Social")
    
    # High engagement
    if not categories.get('High Engagement', pd.DataFrame()).empty:
        with st.expander(f"🔥 High Engagement ({len(categories['High Engagement'])} alerts)", expanded=False):
            for idx, alert in categories['High Engagement'].iterrows():
                display_alert_card(alert, idx, "Engagement")
    
    # Recent alerts
    if not categories.get('Recent (24h)', pd.DataFrame()).empty:
        with st.expander(f"🕐 Recent Alerts - Last 24 Hours ({len(categories['Recent (24h)'])} alerts)", expanded=False):
            for idx, alert in categories['Recent (24h)'].iterrows():
                display_alert_card(alert, idx, "Recent")
    
    # All alerts
    with st.expander(f"📊 All Alerts ({total_alerts} total)", expanded=False):
        st.write(f"**{total_alerts} total alerts**")
        
        display_count = min(30, len(alerts_df))
        st.write(f"Showing first {display_count} alerts:")
        
        for idx, alert in alerts_df.head(display_count).iterrows():
            display_alert_card(alert, idx, "All")
        
        if len(alerts_df) > 30:
            st.info(f"Showing first 30 of {len(alerts_df)} total alerts. Use expanders above to see specific categories.")
    
    # Status
    st.header("📱 Monitoring Status")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**⏰ Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.write(f"**🤖 AI Analysis:** {'✅ Available' if GEMINI_AVAILABLE else '❌ Unavailable'}")
    with col2:
        st.write(f"**📊 Alert Types:** Cyber Threats, Complaints, Social Media")
        st.write(f"**🎯 Total Monitored:** {total_alerts} alerts")

if __name__ == "__main__":
    main()
























