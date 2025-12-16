import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os

# Set page config
st.set_page_config(
    page_title="AI-Driven SOC Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-high { color: #d62728; }
    .alert-medium { color: #ff7f0e; }
    .alert-low { color: #2ca02c; }
</style>
""", unsafe_allow_html=True)

def get_bigquery_client():
    """Initialize BigQuery client"""
    try:
        from google.cloud import bigquery
        client = bigquery.Client()
        return client
    except Exception as e:
        st.error(f"Failed to initialize BigQuery client: {e}")
        return None

def get_real_alerts():
    """Fetch real alerts from BigQuery"""
    client = get_bigquery_client()
    if not client:
        return pd.DataFrame()
    
    try:
        query = """
        SELECT 
            alert_id,
            timestamp,
            is_anomaly,
            confidence_score,
            classification
        FROM `chronicle-dev-2be9.soc_data.processed_alerts`
        WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
        ORDER BY timestamp DESC
        LIMIT 100
        """
        
        df = client.query(query).to_dataframe(create_bqstorage_client=False)
        
        if not df.empty:
            # Add severity column based on confidence score
            df['severity'] = df['confidence_score'].apply(
                lambda x: 'High' if x > 0.8 else 'Medium' if x > 0.6 else 'Low'
            )
            
            # Remove duplicates
            df = df.drop_duplicates(subset='alert_id', keep='first')
            
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching alerts: {e}")
        return pd.DataFrame()

def alert_review_page():
    """Clean Alert Review page with better structure"""
    st.header("🚨 Alert Review & Feedback")
    st.markdown("**Historical alert review with feedback integration**")
    
    if st.button("🔄 Refresh Alerts"):
        st.rerun()
    
    # Get alerts data
    with st.spinner("Loading alerts..."):
        alerts_df = get_real_alerts()
    
    if not alerts_df.empty:
        # Show summary metrics first
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_alerts = len(alerts_df)
            st.metric("Total Alerts", total_alerts)
        
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
        
        # Clean alerts table with only essential columns
        st.subheader("📊 Alert Summary")
        
        # Prepare clean display data
        display_df = alerts_df.copy()
        display_df['timestamp_short'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%m-%d %H:%M')
        
        # Select only essential columns for display
        clean_columns = ['alert_id', 'timestamp_short', 'classification', 'severity', 'confidence_score', 'is_anomaly']
        clean_df = display_df[clean_columns].rename(columns={
            'timestamp_short': 'Time',
            'classification': 'Type',
            'confidence_score': 'Confidence'
        })
        
        st.dataframe(
            clean_df,
            use_container_width=True,
            height=300,
            hide_index=True
        )
        
        st.markdown("---")
        
        # Alert selection for feedback - simplified
        st.subheader("📝 Provide Feedback")
        
        # Create a dropdown instead of radio buttons
        alert_options = {}
        for _, row in display_df.iterrows():
            option_key = f"{row['alert_id'][:12]}... | {row['timestamp_short']} | {row['severity']} | {row['classification']}"
            alert_options[option_key] = row
        
        selected_option = st.selectbox(
            "Select Alert to Review:",
            options=list(alert_options.keys()),
            key="alert_dropdown"
        )
        
        if selected_option:
            selected_alert = alert_options[selected_option]
            
            # Display selected alert details in a clean format
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("**📋 Alert Details**")
                st.markdown(f"**Alert ID:** `{selected_alert['alert_id']}`")
                st.markdown(f"**Timestamp:** {selected_alert['timestamp']}")
                st.markdown(f"**Classification:** {selected_alert['classification']}")
                st.markdown(f"**Confidence Score:** {selected_alert['confidence_score']:.3f}")
                st.markdown(f"**Is Anomaly:** {'Yes' if selected_alert['is_anomaly'] else 'No'}")
            
            with col2:
                st.markdown("**⚡ Quick Actions**")
                if st.button("✅ True Positive", key="tp_btn", use_container_width=True):
                    st.success("✅ Marked as True Positive")
                if st.button("❌ False Positive", key="fp_btn", use_container_width=True):
                    st.warning("❌ Marked as False Positive")
                if st.button("📋 Manual Review", key="mr_btn", use_container_width=True):
                    st.info("📋 Manual review requested")
        
        # Show additional insights
        if len(alerts_df) > 0:
            st.markdown("---")
            st.subheader("📈 Alert Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Classification distribution
                classification_counts = alerts_df['classification'].value_counts()
                st.markdown("**Alert Types:**")
                for classification, count in classification_counts.items():
                    percentage = (count / len(alerts_df)) * 100
                    st.markdown(f"• {classification}: {count} ({percentage:.1f}%)")
            
            with col2:
                # Time distribution
                alerts_df['hour'] = pd.to_datetime(alerts_df['timestamp']).dt.hour
                hour_counts = alerts_df['hour'].value_counts().sort_index()
                st.markdown("**Peak Hours:**")
                for hour, count in hour_counts.head(3).items():
                    st.markdown(f"• {hour:02d}:00 - {count} alerts")
    
    else:
        st.info("No alerts available for review")

def overview_page():
    """Overview page with system status"""
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
    
    # Get alerts for stats
    alerts_df = get_real_alerts()
    
    if not alerts_df.empty:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Alerts (30d)", len(alerts_df))
        
        with col2:
            anomalies = len(alerts_df[alerts_df['is_anomaly'] == True])
            st.metric("Anomalies Detected", anomalies)
        
        with col3:
            avg_confidence = alerts_df['confidence_score'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")
    else:
        st.info("No recent alert data available")

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
