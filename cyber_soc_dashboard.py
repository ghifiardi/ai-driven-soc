#!/usr/bin/env python3
"""
Cybersecurity-focused Social Media Intelligence Dashboard
Monitors social media for cyber threats, security incidents, and brand protection
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
    page_title="🛡️ Cyber SOC Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants - API keys must be provided via environment variables
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    import logging
    logging.warning("GEMINI_API_KEY not set. Gemini features will be disabled.")
GEMINI_AVAILABLE = bool(GEMINI_API_KEY)

# Cyber Security Keywords
CYBER_KEYWORDS = {
    'data_breach': [
        'data breach', 'data leak', 'data stolen', 'hacked', 'compromised',
        'password leaked', 'credentials stolen', 'personal data', 'PII exposed'
    ],
    'malware_threats': [
        'ransomware', 'malware', 'virus', 'trojan', 'phishing', 'spam',
        'botnet', 'keylogger', 'backdoor', 'rootkit', 'cryptocurrency mining'
    ],
    'social_engineering': [
        'phishing', 'vishing', 'smishing', 'pretexting', 'baiting',
        'quid pro quo', 'tailgating', 'impersonation', 'fake account'
    ],
    'infrastructure_attacks': [
        'DDoS', 'denial of service', 'server down', 'website hacked',
        'DNS hijacking', 'man in the middle', 'MITM', 'SSL certificate'
    ],
    'brand_abuse': [
        'fake indosat', 'fake im3', 'scam indosat', 'phishing indosat',
        'fake customer service', 'impersonation', 'brand abuse'
    ],
    'vulnerability_disclosure': [
        'vulnerability', 'exploit', 'zero day', 'CVE', 'security flaw',
        'bug bounty', 'responsible disclosure', 'patch available'
    ]
}

# Risk scoring weights
RISK_WEIGHTS = {
    'data_breach': 0.9,
    'malware_threats': 0.8,
    'social_engineering': 0.7,
    'infrastructure_attacks': 0.8,
    'brand_abuse': 0.6,
    'vulnerability_disclosure': 0.5
}

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
    """Get social media alerts from BigQuery for cyber threat analysis"""
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
            # Determine if this is actually a cyber threat or just a complaint
            text_lower = row.text.lower()
            is_cyber_threat = False
            
            # Check for actual cyber threat indicators
            cyber_indicators = [
                'data breach', 'data leak', 'hacked', 'compromised', 'leaked', 'stolen',
                'ransomware', 'malware', 'trojan', 'phishing', 'botnet',
                'ddos', 'server down', 'dns hijacking', 'mitm',
                'fake indosat', 'scam indosat', 'phishing indosat', 'impersonation',
                'vulnerability', 'exploit', 'zero day', 'cve', 'security flaw'
            ]
            
            if any(indicator in text_lower for indicator in cyber_indicators):
                is_cyber_threat = True
            
            # Determine classification
            if is_cyber_threat:
                classification = f"Cyber Threat - {row.severity}"
            else:
                # Check if it's a complaint
                complaint_words = ['lambat', 'lemot', 'down', 'gangguan', 'masalah', 'buruk', 'jelek', 'mahal']
                if any(word in text_lower for word in complaint_words):
                    classification = f"General Complaint - {row.severity}"
                else:
                    classification = f"Social Media Alert - {row.severity}"
            
            alerts.append({
                'alert_id': f"cyber_{row.tweet_id}",
                'timestamp': row.created_at,
                'classification': classification,
                'severity': row.severity,
                'confidence_score': row.threat_score,
                'is_anomaly': row.threat_score > 0.5,
                'text': row.text,
                'user_screen_name': row.user_screen_name,
                'user_followers': row.user_followers,
                'retweet_count': row.retweet_count,
                'favorite_count': row.favorite_count,
                'risk_factors': row.risk_factors.split(',') if row.risk_factors else [],
                'keywords_found': row.keywords_found.split(',') if row.keywords_found else [],
                'source': 'twitter',
                'analysis_timestamp': row.analysis_timestamp,
                'language': row.language,
                'is_cyber_threat': is_cyber_threat
            })

        return pd.DataFrame(alerts)

    except Exception as e:
        st.error(f"Error getting social media alerts: {e}")
        return pd.DataFrame()

def analyze_cyber_threats(alerts_df):
    """Analyze alerts for cybersecurity threats with improved categorization"""
    if alerts_df.empty:
        return {}
    
    threat_categories = {
        'Data Breach Incidents': [],
        'Malware & Ransomware': [],
        'Social Engineering': [],
        'Infrastructure Attacks': [],
        'Brand Abuse': [],
        'Vulnerability Disclosures': [],
        'General Complaints': [],
        'High-Risk Users': [],
        'Recent Threats (24h)': []
    }
    
    for idx, alert in alerts_df.iterrows():
        text_lower = alert['text'].lower()
        
        # More strict cyber threat detection - require multiple indicators
        is_cyber_threat = False
        threat_type = None
        
        # Check for actual cyber threats (not just complaints)
        if any(keyword in text_lower for keyword in CYBER_KEYWORDS['data_breach']):
            # Additional check for actual breach mentions, not just complaints
            if any(indicator in text_lower for indicator in ['leaked', 'stolen', 'hacked', 'breach', 'compromised']):
                threat_categories['Data Breach Incidents'].append(alert)
                is_cyber_threat = True
        
        elif any(keyword in text_lower for keyword in CYBER_KEYWORDS['malware_threats']):
            # Check for actual malware mentions, not just "virus" in context of slow internet
            if any(indicator in text_lower for indicator in ['ransomware', 'malware', 'trojan', 'phishing', 'botnet']):
                threat_categories['Malware & Ransomware'].append(alert)
                is_cyber_threat = True
        
        elif any(keyword in text_lower for keyword in CYBER_KEYWORDS['social_engineering']):
            # Check for actual social engineering attempts
            if any(indicator in text_lower for indicator in ['phishing', 'scam', 'fake', 'impersonation', 'vishing']):
                threat_categories['Social Engineering'].append(alert)
                is_cyber_threat = True
        
        elif any(keyword in text_lower for keyword in CYBER_KEYWORDS['infrastructure_attacks']):
            # Check for actual infrastructure attacks
            if any(indicator in text_lower for indicator in ['ddos', 'hacked', 'server down', 'dns hijacking', 'mitm']):
                threat_categories['Infrastructure Attacks'].append(alert)
                is_cyber_threat = True
        
        elif any(keyword in text_lower for keyword in CYBER_KEYWORDS['brand_abuse']):
            # Check for actual brand abuse
            if any(indicator in text_lower for indicator in ['fake indosat', 'scam indosat', 'phishing indosat', 'impersonation']):
                threat_categories['Brand Abuse'].append(alert)
                is_cyber_threat = True
        
        elif any(keyword in text_lower for keyword in CYBER_KEYWORDS['vulnerability_disclosure']):
            # Check for actual vulnerability mentions
            if any(indicator in text_lower for indicator in ['vulnerability', 'exploit', 'zero day', 'cve', 'security flaw']):
                threat_categories['Vulnerability Disclosures'].append(alert)
                is_cyber_threat = True
        
        # If not a cyber threat, categorize as general complaint
        if not is_cyber_threat:
            # Use the is_cyber_threat field from the alert data
            if hasattr(alert, 'is_cyber_threat') and not alert['is_cyber_threat']:
                threat_categories['General Complaints'].append(alert)
            elif not hasattr(alert, 'is_cyber_threat'):
                # Fallback: check if it's a general complaint about service
                if any(complaint_word in text_lower for complaint_word in ['lambat', 'lemot', 'down', 'gangguan', 'masalah', 'buruk', 'jelek', 'mahal']):
                    threat_categories['General Complaints'].append(alert)
        
        # High-risk users (high followers + high engagement)
        if alert['user_followers'] > 50000 and (alert['retweet_count'] > 100 or alert['favorite_count'] > 200):
            threat_categories['High-Risk Users'].append(alert)
        
        # Recent threats (last 24 hours)
        try:
            now = datetime.now()
            if hasattr(alert['timestamp'], 'tz') and alert['timestamp'].tz is not None:
                alert_time = alert['timestamp'].replace(tzinfo=None)
            else:
                alert_time = alert['timestamp']
            
            if (now - alert_time).total_seconds() < 86400:  # 24 hours
                threat_categories['Recent Threats (24h)'].append(alert)
        except:
            pass
    
    # Convert lists to DataFrames
    for category, alerts_list in threat_categories.items():
        if alerts_list:
            threat_categories[category] = pd.DataFrame(alerts_list)
        else:
            threat_categories[category] = pd.DataFrame()
    
    return threat_categories

def calculate_threat_score(alert):
    """Calculate enhanced threat score for cybersecurity"""
    # Handle both dictionary and Pandas tuple objects
    if hasattr(alert, 'get'):
        # Dictionary-like object
        base_score = alert.get('threat_score', 0)
        text = alert.get('text', '')
        retweet_count = alert.get('retweet_count', 0)
        favorite_count = alert.get('favorite_count', 0)
        user_followers = alert.get('user_followers', 0)
    else:
        # Pandas tuple object
        base_score = getattr(alert, 'threat_score', 0)
        text = getattr(alert, 'text', '')
        retweet_count = getattr(alert, 'retweet_count', 0)
        favorite_count = getattr(alert, 'favorite_count', 0)
        user_followers = getattr(alert, 'user_followers', 0)
    
    # Boost score based on threat category
    text_lower = text.lower() if text else ''
    category_boost = 0
    
    for category, keywords in CYBER_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            category_boost += RISK_WEIGHTS[category]
    
    # Boost for high engagement
    engagement_boost = min(0.3, (retweet_count + favorite_count) / 1000)
    
    # Boost for influential users
    influence_boost = min(0.2, user_followers / 100000)
    
    # Calculate final score
    final_score = min(1.0, base_score + (category_boost * 0.3) + engagement_boost + influence_boost)
    
    return final_score

def get_gemini_client():
    """Get Gemini client for AI analysis"""
    if not GEMINI_AVAILABLE:
        return None
    
    class GeminiClient:
        def __init__(self, api_key):
            self.api_key = api_key
            self.base_url = "https://generative-language.googleapis.com/v1beta"
        
        def enrich_social_media_alert(self, text, user_info):
            """Analyze social media alert for cybersecurity threats"""
            try:
                prompt = f"""
                Analyze this social media post for cybersecurity threats to telecom companies like Indosat:
                
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
                
                url = f"{self.base_url}/models/gemini-1.5-pro:generateContent"
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
                response.raise_for_status()
                
                result = response.json()
                if 'candidates' in result and result['candidates']:
                    content = result['candidates'][0]['content']['parts'][0]['text']
                    
                    # Try to extract JSON from response
                    try:
                        # Find JSON in the response
                        start = content.find('{')
                        end = content.rfind('}') + 1
                        if start != -1 and end != 0:
                            json_str = content[start:end]
                            enrichment = json.loads(json_str)
                        else:
                            # Fallback parsing
                            enrichment = {
                                'threat_level': 'Medium',
                                'threat_category': 'Other',
                                'brand_impact': 'Medium',
                                'key_concerns': ['Social media monitoring detected potential security concern'],
                                'urgency_level': 'Medium',
                                'recommended_actions': ['Monitor for additional indicators', 'Verify threat authenticity'],
                                'false_positive_probability': 0.3
                            }
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
                    
                    return {'success': True, 'enrichment': enrichment}
                else:
                    return {'success': False, 'error': 'No response from Gemini'}
                    
            except Exception as e:
                return {'success': False, 'error': str(e)}
    
    return GeminiClient(GEMINI_API_KEY)

def display_cyber_threat_card(alert, index, category_name=""):
    """Display a cybersecurity threat alert card"""
    threat_score = calculate_threat_score(alert)
    
    with st.container():
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            threat_icon = "🚨" if threat_score > 0.8 else "⚠️" if threat_score > 0.6 else "🔍"
            st.write(f"**{threat_icon} Cyber Threat {alert['alert_id']}**")
            st.write(f"📝 {alert['text'][:200]}{'...' if len(alert['text']) > 200 else ''}")
            
        with col2:
            st.write(f"**👤** {alert['user_screen_name']}")
            st.write(f"**👥** {alert['user_followers']:,} followers")
            st.write(f"**🎯** Threat Score: {threat_score:.2f}")
            
        with col3:
            st.write(f"**📊** RT: {alert['retweet_count']} | ❤️: {alert['favorite_count']}")
            if alert.get('keywords_found'):
                keywords_str = ', '.join(alert['keywords_found'][:3])
                st.write(f"**🏷️** {keywords_str}")
        
        # Risk factors if available
        if alert.get('risk_factors') and alert['risk_factors'] != '':
            st.write(f"⚠️ **Risk Factors:** {alert['risk_factors']}")
        
        # AI Analysis section
        if GEMINI_AVAILABLE:
            unique_key = f"cyber_ai_btn_{alert['alert_id']}_{category_name}_{index}_{hash(alert['text'])}"
            if st.button(f"🤖 Cyber Threat Analysis", key=unique_key):
                with st.spinner("Analyzing cybersecurity threat..."):
                    client = get_gemini_client()
                    if client:
                        user_info = {
                            'user_screen_name': alert['user_screen_name'],
                            'user_followers': alert['user_followers']
                        }
                        result = client.enrich_social_media_alert(alert['text'], user_info)
                        
                        if result['success']:
                            enrichment = result['enrichment']
                            st.success("✅ Cyber Threat Analysis Complete")
                            
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
                            st.error(f"Analysis failed: {result.get('error', 'Unknown error')}")
        
        st.write("---")

def main():
    st.title("🛡️ Cyber SOC Intelligence Dashboard")
    st.markdown("**Advanced Cybersecurity Threat Monitoring from Social Media**")
    
    # Get alerts
    client = get_bigquery_client()
    if client:
        alerts_df = get_social_media_alerts(client, limit=300)
        threat_categories = analyze_cyber_threats(alerts_df)
    else:
        st.error("Unable to connect to BigQuery. Using sample data.")
        alerts_df = pd.DataFrame()
        threat_categories = {}
    
    # Calculate metrics
    total_alerts = len(alerts_df)
    critical_threats = len([a for a in alerts_df.itertuples() if calculate_threat_score(a) > 0.8])
    high_threats = len([a for a in alerts_df.itertuples() if 0.6 < calculate_threat_score(a) <= 0.8])
    recent_threats = len(threat_categories.get('Recent Threats (24h)', pd.DataFrame()))
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Cyber Alerts", total_alerts)
    with col2:
        st.metric("🚨 Critical Threats", critical_threats, delta=f"{critical_threats}")
    with col3:
        st.metric("⚠️ High Risk Threats", high_threats, delta=f"{high_threats}")
    with col4:
        st.metric("🕐 Recent (24h)", recent_threats, delta=f"{recent_threats}")
    
    # Threat categories
    st.header("🎯 Cybersecurity Threat Categories")
    
    # Critical threats first
    if not threat_categories.get('Data Breach Incidents', pd.DataFrame()).empty:
        with st.expander(f"🚨 Data Breach Incidents ({len(threat_categories['Data Breach Incidents'])} alerts)", expanded=True):
            for idx, alert in threat_categories['Data Breach Incidents'].iterrows():
                display_cyber_threat_card(alert, idx, "DataBreach")
    
    if not threat_categories.get('Malware & Ransomware', pd.DataFrame()).empty:
        with st.expander(f"🦠 Malware & Ransomware ({len(threat_categories['Malware & Ransomware'])} alerts)", expanded=False):
            for idx, alert in threat_categories['Malware & Ransomware'].iterrows():
                display_cyber_threat_card(alert, idx, "Malware")
    
    if not threat_categories.get('Social Engineering', pd.DataFrame()).empty:
        with st.expander(f"🎭 Social Engineering ({len(threat_categories['Social Engineering'])} alerts)", expanded=False):
            for idx, alert in threat_categories['Social Engineering'].iterrows():
                display_cyber_threat_card(alert, idx, "SocialEng")
    
    if not threat_categories.get('Infrastructure Attacks', pd.DataFrame()).empty:
        with st.expander(f"🏗️ Infrastructure Attacks ({len(threat_categories['Infrastructure Attacks'])} alerts)", expanded=False):
            for idx, alert in threat_categories['Infrastructure Attacks'].iterrows():
                display_cyber_threat_card(alert, idx, "Infrastructure")
    
    if not threat_categories.get('Brand Abuse', pd.DataFrame()).empty:
        with st.expander(f"🎭 Brand Abuse ({len(threat_categories['Brand Abuse'])} alerts)", expanded=False):
            for idx, alert in threat_categories['Brand Abuse'].iterrows():
                display_cyber_threat_card(alert, idx, "BrandAbuse")
    
    if not threat_categories.get('General Complaints', pd.DataFrame()).empty:
        with st.expander(f"💬 General Complaints ({len(threat_categories['General Complaints'])} alerts)", expanded=False):
            for idx, alert in threat_categories['General Complaints'].iterrows():
                display_cyber_threat_card(alert, idx, "General")
    
    if not threat_categories.get('High-Risk Users', pd.DataFrame()).empty:
        with st.expander(f"👑 High-Risk Users ({len(threat_categories['High-Risk Users'])} alerts)", expanded=False):
            for idx, alert in threat_categories['High-Risk Users'].iterrows():
                display_cyber_threat_card(alert, idx, "HighRisk")
    
    # Recent threats
    if not threat_categories.get('Recent Threats (24h)', pd.DataFrame()).empty:
        with st.expander(f"🕐 Recent Threats - Last 24 Hours ({len(threat_categories['Recent Threats (24h)'])} alerts)", expanded=False):
            for idx, alert in threat_categories['Recent Threats (24h)'].iterrows():
                display_cyber_threat_card(alert, idx, "Recent")
    
    # All alerts
    with st.expander(f"📊 All Cyber Threats ({total_alerts} total)", expanded=False):
        st.write(f"**{total_alerts} total cybersecurity alerts**")
        
        display_count = min(30, len(alerts_df))
        st.write(f"Showing first {display_count} alerts:")
        
        for idx, alert in alerts_df.head(display_count).iterrows():
            display_cyber_threat_card(alert, idx, "All")
        
        if len(alerts_df) > 30:
            st.info(f"Showing first 30 of {len(alerts_df)} total alerts. Use expanders above to see specific threat categories.")
    
    # Monitoring status
    st.header("📱 Cybersecurity Monitoring Status")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**🔍 Monitored Keywords:**")
        for category, keywords in CYBER_KEYWORDS.items():
            st.write(f"• **{category.replace('_', ' ').title()}:** {', '.join(keywords[:3])}...")
    
    with col2:
        st.write(f"**⏰ Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.write(f"**🤖 AI Analysis:** {'✅ Available' if GEMINI_AVAILABLE else '❌ Unavailable'}")
        st.write(f"**📊 Total Keywords:** {sum(len(keywords) for keywords in CYBER_KEYWORDS.values())}")

if __name__ == "__main__":
    main()
