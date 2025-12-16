import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import random

# Set page config
st.set_page_config(
    page_title="AI-Driven SOC Dashboard",
    page_icon="🛡️",
    layout="wide"
)

def get_sample_alerts():
    """Generate sample alerts for demonstration"""
    alerts = []
    for i in range(20):
        alert = {
            'alert_id': f'ALERT-{i+1:03d}',
            'timestamp': datetime.now() - timedelta(days=random.randint(1, 30)),
            'classification': random.choice(['anomaly', 'threat', 'benign', 'suspicious']),
            'confidence_score': round(random.uniform(0.3, 0.95), 3),
            'is_anomaly': random.choice([True, False]),
            'source_ip': f'192.168.1.{random.randint(1, 254)}',
            'dest_ip': f'10.0.{random.randint(1, 10)}.{random.randint(1, 254)}'
        }
        alerts.append(alert)
    
    return pd.DataFrame(alerts)

def alert_review_page():
    """Alert Review page with sample data"""
    st.header("🚨 Alert Review & Feedback")
    st.markdown("**Real-time alert review with feedback integration**")
    
    if st.button("🔄 Refresh Alerts"):
        st.rerun()
    
    # Get sample alerts
    alerts_df = get_sample_alerts()
    
    # Show summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Alerts", len(alerts_df))
    
    with col2:
        anomalies = len(alerts_df[alerts_df['is_anomaly'] == True])
        st.metric("Anomalies", anomalies)
    
    with col3:
        high_confidence = len(alerts_df[alerts_df['confidence_score'] > 0.8])
        st.metric("High Confidence", high_confidence)
    
    with col4:
        avg_confidence = alerts_df['confidence_score'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
    
    st.markdown("---")
    
    # Clean alerts table
    st.subheader("📊 Alert Summary")
    
    # Add severity column
    alerts_df['severity'] = alerts_df['confidence_score'].apply(
        lambda x: 'High' if x > 0.8 else 'Medium' if x > 0.6 else 'Low'
    )
    
    # Format timestamp
    alerts_df['timestamp_short'] = alerts_df['timestamp'].dt.strftime('%m-%d %H:%M')
    
    # Display table
    display_columns = ['alert_id', 'timestamp_short', 'classification', 'severity', 'confidence_score', 'is_anomaly']
    clean_df = alerts_df[display_columns].rename(columns={
        'timestamp_short': 'Time',
        'classification': 'Type',
        'confidence_score': 'Confidence'
    })
    
    st.dataframe(clean_df, use_container_width=True, height=400)
    
    st.markdown("---")
    
    # Alert selection for feedback
    st.subheader("📝 Provide Feedback")
    
    # Create dropdown options
    alert_options = {}
    for _, row in alerts_df.iterrows():
        option_key = f"{row['alert_id']} | {row['timestamp_short']} | {row['severity']} | {row['classification']}"
        alert_options[option_key] = row
    
    selected_option = st.selectbox(
        "Select Alert to Review:",
        options=list(alert_options.keys())
    )
    
    if selected_option:
        selected_alert = alert_options[selected_option]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**📋 Alert Details**")
            st.markdown(f"**Alert ID:** `{selected_alert['alert_id']}`")
            st.markdown(f"**Timestamp:** {selected_alert['timestamp']}")
            st.markdown(f"**Classification:** {selected_alert['classification']}")
            st.markdown(f"**Confidence Score:** {selected_alert['confidence_score']:.3f}")
            st.markdown(f"**Is Anomaly:** {'Yes' if selected_alert['is_anomaly'] else 'No'}")
            st.markdown(f"**Source IP:** {selected_alert['source_ip']}")
            st.markdown(f"**Dest IP:** {selected_alert['dest_ip']}")
        
        with col2:
            st.markdown("**⚡ Quick Actions**")
            if st.button("✅ True Positive", use_container_width=True):
                st.success("✅ Marked as True Positive")
            if st.button("❌ False Positive", use_container_width=True):
                st.warning("❌ Marked as False Positive")
            if st.button("📋 Manual Review", use_container_width=True):
                st.info("📋 Manual review requested")

def overview_page():
    """Overview page"""
    st.header("🛡️ AI-Driven SOC Dashboard")
    st.markdown("**Comprehensive Security Operations Center Overview**")
    
    # System Status
    st.subheader("🔧 System Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("CLA Agent", "✅ Active", "Running")
    
    with col2:
        st.metric("ADA Agent", "❌ Inactive", "Stopped")
    
    with col3:
        st.metric("TAA Agent", "❌ Inactive", "Stopped")
    
    with col4:
        st.metric("CRA Agent", "❌ Inactive", "Stopped")
    
    # Quick Stats
    st.subheader("📊 Quick Statistics")
    
    alerts_df = get_sample_alerts()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Alerts (30d)", len(alerts_df))
    
    with col2:
        anomalies = len(alerts_df[alerts_df['is_anomaly'] == True])
        st.metric("Anomalies Detected", anomalies)
    
    with col3:
        avg_confidence = alerts_df['confidence_score'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")

def main():
    """Main dashboard application"""
    
    # Sidebar navigation
    st.sidebar.title("Dashboard Navigation")
    
    pages = {
        "Overview": overview_page,
        "Alert Review": alert_review_page
    }
    
    selected_page = st.sidebar.radio("Select Page", list(pages.keys()))
    
    # System Status in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 System Status")
    
    status_items = [
        ("ADA Agent", "❌ Inactive"),
        ("CLA Agent", "✅ Active"),
        ("TAA Agent", "❌ Inactive"),
        ("CRA Agent", "❌ Inactive")
    ]
    
    for name, status in status_items:
        st.sidebar.markdown(f"{status} {name}")
    
    # Render selected page
    pages[selected_page]()

if __name__ == "__main__":
    main()