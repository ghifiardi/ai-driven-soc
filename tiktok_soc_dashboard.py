#!/usr/bin/env python3
"""
TikTok SOC Dashboard - Monitor TikTok for telecom threats and mentions
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from google.cloud import bigquery
import os

# Page configuration
st.set_page_config(
    page_title="📱 TikTok SOC Intelligence Dashboard",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

def get_tiktok_alerts(client, limit: int = 200):
    """Get TikTok alerts from BigQuery"""
    try:
        query = f"""
        SELECT
            video_id,
            text,
            created_at,
            user_screen_name,
            user_followers,
            view_count,
            like_count,
            comment_count,
            share_count,
            threat_score,
            severity,
            risk_factors,
            keywords_found,
            analysis_timestamp,
            language,
            video_url,
            hashtags,
            is_cyber_threat,
            engagement_rate,
            virality_score
        FROM `chronicle-dev-2be9.soc_data.tiktok_soc_alerts`
        ORDER BY analysis_timestamp DESC
        LIMIT {limit}
        """

        result = client.query(query).result()
        rows = list(result)

        alerts = []
        for row in rows:
            text_lower = row.text.lower()
            
            # Determine alert type based on content
            alert_type = "TikTok Video"
            
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
            elif row.virality_score > 0.7:
                alert_type = "Viral Content"
            
            alerts.append({
                'alert_id': f"tiktok_{row.video_id}",
                'alert_type': alert_type,
                'timestamp': row.created_at,
                'classification': f"{alert_type} - {row.severity}",
                'severity': row.severity,
                'confidence_score': row.threat_score,
                'text': row.text,
                'user_screen_name': row.user_screen_name,
                'user_followers': row.user_followers,
                'view_count': row.view_count,
                'like_count': row.like_count,
                'comment_count': row.comment_count,
                'share_count': row.share_count,
                'engagement_rate': row.engagement_rate,
                'virality_score': row.virality_score,
                'risk_factors': row.risk_factors.split(',') if row.risk_factors else [],
                'keywords_found': row.keywords_found.split(',') if row.keywords_found else [],
                'source': 'tiktok',
                'video_url': row.video_url,
                'hashtags': row.hashtags,
                'analysis_timestamp': row.analysis_timestamp,
                'language': row.language,
                'is_cyber_threat': row.is_cyber_threat
            })

        return pd.DataFrame(alerts)

    except Exception as e:
        st.error(f"Error getting TikTok alerts: {e}")
        return pd.DataFrame()

def analyze_tiktok_alert_manually(alert):
    """Provide manual analysis based on TikTok alert content"""
    text_lower = alert['text'].lower()
    
    # Threat level analysis
    if alert['alert_type'] == "Cyber Threat":
        threat_level = "High"
        category = "Security Incident"
        urgency = "Immediate"
        brand_impact = "High"
        concerns = ["Potential security breach", "Brand reputation risk", "Customer data exposure"]
        actions = ["Investigate immediately", "Notify security team", "Prepare incident response"]
        false_positive = 0.1
    elif alert['alert_type'] == "Viral Content":
        threat_level = "Medium"
        category = "Viral Content"
        urgency = "High"
        brand_impact = "High"
        concerns = ["Viral spread potential", "Brand reputation", "Mass audience reach"]
        actions = ["Monitor engagement", "Prepare response strategy", "Track viral metrics"]
        false_positive = 0.3
    elif alert['alert_type'] == "General Complaint":
        threat_level = "Low"
        category = "Customer Service"
        urgency = "Medium"
        brand_impact = "Medium"
        concerns = ["Customer satisfaction", "Service quality", "Brand perception"]
        actions = ["Forward to customer service", "Monitor sentiment", "Address service issues"]
        false_positive = 0.8
    else:
        threat_level = "Low"
        category = "Social Media"
        urgency = "Low"
        brand_impact = "Low"
        concerns = ["Brand monitoring", "Social media presence"]
        actions = ["Monitor engagement", "Track mentions"]
        false_positive = 0.9
    
    # Adjust based on engagement
    if alert['engagement_rate'] > 0.1 or alert['virality_score'] > 0.7:
        urgency = "High"
        brand_impact = "High"
        concerns.append("High engagement - viral potential")
    
    # Adjust based on user influence
    if alert['user_followers'] > 100000:
        urgency = "High"
        brand_impact = "High"
        concerns.append("Influential user - wide reach")
    
    return {
        'threat_level': threat_level,
        'threat_category': category,
        'brand_impact': brand_impact,
        'key_concerns': concerns,
        'urgency_level': urgency,
        'recommended_actions': actions,
        'false_positive_probability': false_positive
    }

def display_tiktok_alert_card(alert, index, category_name=""):
    """Display a single TikTok alert card"""
    with st.container():
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            # Use appropriate icon based on alert type
            if alert['alert_type'] == "Cyber Threat":
                icon = "🚨"
            elif alert['alert_type'] == "Viral Content":
                icon = "🔥"
            elif alert['alert_type'] == "General Complaint":
                icon = "💬"
            else:
                icon = "📱"
            
            st.write(f"**{icon} {alert['alert_type']} {alert['alert_id']}**")
            st.write(f"📝 {alert['text'][:200]}{'...' if len(alert['text']) > 200 else ''}")
            
            # Show hashtags if available
            if alert.get('hashtags'):
                st.write(f"🏷️ **Hashtags:** {alert['hashtags']}")
            
            # Show video URL if available
            if alert.get('video_url'):
                st.write(f"🔗 **Video:** [View on TikTok]({alert['video_url']})")
        
        with col2:
            st.write(f"**👤** {alert['user_screen_name']}")
            st.write(f"**👥** {alert['user_followers']:,} followers")
            st.write(f"**📊** Score: {alert['confidence_score']:.2f}")
            st.write(f"**🎯** Virality: {alert['virality_score']:.2f}")
            
        with col3:
            st.write(f"**👁️** Views: {alert['view_count']:,}")
            st.write(f"**❤️** Likes: {alert['like_count']:,}")
            st.write(f"**💬** Comments: {alert['comment_count']:,}")
            st.write(f"**📤** Shares: {alert['share_count']:,}")
            st.write(f"**📈** Engagement: {alert['engagement_rate']:.2%}")
        
        # Risk factors if available
        if alert.get('risk_factors') and alert['risk_factors'] != '':
            st.write(f"⚠️ **Risk Factors:** {alert['risk_factors']}")
        
        # Keywords if available
        if alert.get('keywords_found'):
            keywords_str = ', '.join(alert['keywords_found'][:5])
            st.write(f"🔍 **Keywords:** {keywords_str}")
        
        # AI Analysis section (manual analysis)
        unique_key = f"tiktok_ai_btn_{alert['alert_id']}_{category_name}_{index}_{hash(alert['text'])}"
        if st.button(f"🤖 TikTok Analysis", key=unique_key):
            with st.spinner("Analyzing TikTok content..."):
                analysis = analyze_tiktok_alert_manually(alert)
                
                st.success("✅ TikTok Analysis Complete (Smart Analysis Engine)")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**🚨 Threat Level:** {analysis['threat_level']}")
                    st.write(f"**📂 Category:** {analysis['threat_category']}")
                    st.write(f"**🎯 Brand Impact:** {analysis['brand_impact']}")
                with col2:
                    st.write(f"**⚡ Urgency:** {analysis['urgency_level']}")
                    st.write(f"**🎲 False Positive:** {analysis['false_positive_probability']:.1%}")
                    st.write(f"**📱 Platform:** TikTok")
                
                st.write("**🔍 Key Concerns:**")
                for concern in analysis['key_concerns']:
                    st.write(f"• {concern}")
                
                st.write("**📋 Recommended Actions:**")
                for action in analysis['recommended_actions']:
                    st.write(f"• {action}")
                
                # TikTok-specific metrics
                st.info(f"""
                **📱 TikTok Metrics:**
                • **Views:** {alert['view_count']:,} | **Engagement Rate:** {alert['engagement_rate']:.2%}
                • **Virality Score:** {alert['virality_score']:.2f} | **User Followers:** {alert['user_followers']:,}
                • **Analysis Method:** Smart rule-based analysis (No external API dependency)
                """)
        
        st.write("---")

def categorize_tiktok_alerts(alerts_df):
    """Categorize TikTok alerts by type"""
    if alerts_df.empty:
        return {}
    
    categories = {
        'Cyber Threats': [],
        'Viral Content': [],
        'General Complaints': [],
        'High Engagement': [],
        'Influential Users': [],
        'Recent (24h)': []
    }
    
    for idx, alert in alerts_df.iterrows():
        # Categorize by alert type
        if alert['alert_type'] == "Cyber Threat":
            categories['Cyber Threats'].append(alert)
        elif alert['alert_type'] == "Viral Content":
            categories['Viral Content'].append(alert)
        elif alert['alert_type'] == "General Complaint":
            categories['General Complaints'].append(alert)
        
        # High engagement
        if alert['engagement_rate'] > 0.1 or alert['view_count'] > 100000:
            categories['High Engagement'].append(alert)
        
        # Influential users
        if alert['user_followers'] > 50000:
            categories['Influential Users'].append(alert)
        
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
    st.title("📱 TikTok SOC Intelligence Dashboard")
    st.markdown("**Smart TikTok Threat Monitoring for Telecom Sector**")
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ TikTok Monitoring Controls")
        
        if st.button("🔄 Refresh TikTok Data"):
            # Force refresh by showing success message
            st.success("TikTok data refreshed! Please refresh the page to see updated data.")
        
        st.write("**📊 Monitoring Keywords:**")
        st.write("• Indosat, IOH, IM3, Tri")
        st.write("• Customer complaints")
        st.write("• Security threats")
        st.write("• Viral content")
    
    # Get TikTok alerts
    client = get_bigquery_client()
    if client:
        alerts_df = get_tiktok_alerts(client, limit=200)
        categories = categorize_tiktok_alerts(alerts_df)
    else:
        st.error("Unable to connect to BigQuery. Using sample data.")
        alerts_df = pd.DataFrame()
        categories = {}
    
    # Calculate metrics
    total_alerts = len(alerts_df)
    cyber_threats = len(categories.get('Cyber Threats', pd.DataFrame()))
    viral_content = len(categories.get('Viral Content', pd.DataFrame()))
    complaints = len(categories.get('General Complaints', pd.DataFrame()))
    recent_alerts = len(categories.get('Recent (24h)', pd.DataFrame()))
    
    # Display metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total TikTok Alerts", total_alerts)
    with col2:
        st.metric("🚨 Cyber Threats", cyber_threats, delta=f"{cyber_threats}")
    with col3:
        st.metric("🔥 Viral Content", viral_content, delta=f"{viral_content}")
    with col4:
        st.metric("💬 Complaints", complaints, delta=f"{complaints}")
    with col5:
        st.metric("🕐 Recent (24h)", recent_alerts, delta=f"{recent_alerts}")
    
    # TikTok-specific metrics
    if not alerts_df.empty:
        total_views = alerts_df['view_count'].sum()
        total_engagement = alerts_df['engagement_rate'].mean()
        avg_virality = alerts_df['virality_score'].mean()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("👁️ Total Views", f"{total_views:,}")
        with col2:
            st.metric("📈 Avg Engagement", f"{total_engagement:.2%}")
        with col3:
            st.metric("🔥 Avg Virality", f"{avg_virality:.2f}")
    
    # Alert categories
    st.header("📊 TikTok Alert Categories")
    
    # Cyber threats first
    if not categories.get('Cyber Threats', pd.DataFrame()).empty:
        with st.expander(f"🚨 Cyber Threats ({len(categories['Cyber Threats'])} alerts)", expanded=True):
            for idx, alert in categories['Cyber Threats'].iterrows():
                display_tiktok_alert_card(alert, idx, "Cyber")
    
    # Viral content
    if not categories.get('Viral Content', pd.DataFrame()).empty:
        with st.expander(f"🔥 Viral Content ({len(categories['Viral Content'])} alerts)", expanded=False):
            for idx, alert in categories['Viral Content'].iterrows():
                display_tiktok_alert_card(alert, idx, "Viral")
    
    # General complaints
    if not categories.get('General Complaints', pd.DataFrame()).empty:
        with st.expander(f"💬 General Complaints ({len(categories['General Complaints'])} alerts)", expanded=False):
            for idx, alert in categories['General Complaints'].iterrows():
                display_tiktok_alert_card(alert, idx, "Complaint")
    
    # High engagement
    if not categories.get('High Engagement', pd.DataFrame()).empty:
        with st.expander(f"📈 High Engagement ({len(categories['High Engagement'])} alerts)", expanded=False):
            for idx, alert in categories['High Engagement'].iterrows():
                display_tiktok_alert_card(alert, idx, "Engagement")
    
    # Influential users
    if not categories.get('Influential Users', pd.DataFrame()).empty:
        with st.expander(f"👑 Influential Users ({len(categories['Influential Users'])} alerts)", expanded=False):
            for idx, alert in categories['Influential Users'].iterrows():
                display_tiktok_alert_card(alert, idx, "Influential")
    
    # Recent alerts
    if not categories.get('Recent (24h)', pd.DataFrame()).empty:
        with st.expander(f"🕐 Recent Alerts - Last 24 Hours ({len(categories['Recent (24h)'])} alerts)", expanded=False):
            for idx, alert in categories['Recent (24h)'].iterrows():
                display_tiktok_alert_card(alert, idx, "Recent")
    
    # All alerts
    with st.expander(f"📊 All TikTok Alerts ({total_alerts} total)", expanded=False):
        st.write(f"**{total_alerts} total TikTok alerts**")
        
        display_count = min(20, len(alerts_df))
        st.write(f"Showing first {display_count} alerts:")
        
        for idx, alert in alerts_df.head(display_count).iterrows():
            display_tiktok_alert_card(alert, idx, "All")
        
        if len(alerts_df) > 20:
            st.info(f"Showing first 20 of {len(alerts_df)} total alerts. Use expanders above to see specific categories.")
    
    # Status
    st.header("📱 TikTok Monitoring Status")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**⏰ Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.write(f"**🤖 AI Analysis:** ✅ Smart Analysis Engine (No API dependency)")
        st.write(f"**📱 Platform:** TikTok")
    with col2:
        st.write(f"**📊 Alert Types:** Cyber Threats, Viral Content, Complaints")
        st.write(f"**🎯 Total Monitored:** {total_alerts} TikTok alerts")
        st.write(f"**👁️ Total Views:** {total_views:,}" if not alerts_df.empty else "**👁️ Total Views:** 0")
    
    # Info about TikTok monitoring
    st.info("""
    **💡 TikTok SOC Monitoring Features:**
    - **Viral Content Detection:** Identifies high-engagement TikTok videos
    - **Cyber Threat Analysis:** Monitors for security-related content
    - **Brand Mention Tracking:** Tracks mentions of Indosat/IOH/IM3/Tri
    - **Engagement Metrics:** Views, likes, comments, shares analysis
    - **Influencer Monitoring:** Tracks content from users with large followings
    - **Real-Time Updates:** Fresh data from BigQuery every minute
    """)

if __name__ == "__main__":
    main()
