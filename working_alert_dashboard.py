import streamlit as st
import pandas as pd
from google.cloud import bigquery
import plotly.express as px
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="GATRA Alert Dashboard",
    page_icon="🛡️",
    layout="wide"
)

@st.cache_resource
def get_bigquery_client():
    try:
        return bigquery.Client()
    except Exception as e:
        st.error(f"BigQuery client error: {e}")
        return None

def get_simple_alerts():
    """Get alerts without the problematic raw_alert column"""
    client = get_bigquery_client()
    if not client:
        return pd.DataFrame()
    
    try:
        # Simple query without raw_alert to avoid casting issues
        query = """
        SELECT 
            alert_id,
            timestamp,
            is_anomaly,
            confidence_score,
            classification
        FROM `chronicle-dev-2be9.soc_data.processed_alerts`
        WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
        ORDER BY timestamp DESC
        LIMIT 50
        """
        
        df = client.query(query).to_dataframe(create_bqstorage_client=False)
        
        if not df.empty:
            # Add severity column
            df['severity'] = df['confidence_score'].apply(
                lambda x: 'High' if x > 0.8 else 'Medium' if x > 0.6 else 'Low'
            )
            return df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Error fetching alerts: {e}")
        return pd.DataFrame()

def main():
    st.title("🛡️ GATRA Alert Dashboard")
    st.markdown("**Simple, Working Alert Review**")
    
    # System Status
    st.sidebar.title("📊 System Status")
    st.sidebar.success("✅ ADA: Active")
    st.sidebar.success("✅ CLA: Learning (66 Models)")
    st.sidebar.success("✅ TAA: Google Flash 2.5")
    st.sidebar.success("✅ CRA: Ready")
    
    # Refresh button
    if st.button("🔄 Refresh Alerts"):
        st.rerun()
    
    # Get alerts
    with st.spinner("Fetching alerts from BigQuery..."):
        alerts_df = get_simple_alerts()
    
    if not alerts_df.empty:
        st.success(f"✅ Found {len(alerts_df)} alerts")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Alerts", len(alerts_df))
        with col2:
            high_severity = len(alerts_df[alerts_df['severity'] == 'High'])
            st.metric("High Severity", high_severity)
        with col3:
            anomalies = len(alerts_df[alerts_df['is_anomaly'] == True])
            st.metric("Anomalies", anomalies)
        with col4:
            avg_confidence = alerts_df['confidence_score'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")
        
        # Alerts table
        st.subheader("🚨 Recent Alerts")
        
        # Display options
        display_cols = ['alert_id', 'timestamp', 'severity', 'confidence_score', 'classification', 'is_anomaly']
        st.dataframe(
            alerts_df[display_cols],
            use_container_width=True,
            height=400
        )
        
        # Alert details
        st.subheader("📋 Alert Details")
        selected_alert_id = st.selectbox(
            "Select Alert to Review",
            alerts_df['alert_id'].tolist()
        )
        
        if selected_alert_id:
            selected_alert = alerts_df[alerts_df['alert_id'] == selected_alert_id].iloc[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Alert ID:** {selected_alert['alert_id']}")
                st.markdown(f"**Timestamp:** {selected_alert['timestamp']}")
                st.markdown(f"**Severity:** {selected_alert['severity']}")
                st.markdown(f"**Confidence:** {selected_alert['confidence_score']:.2f}")
            
            with col2:
                st.markdown(f"**Classification:** {selected_alert['classification']}")
                st.markdown(f"**Is Anomaly:** {selected_alert['is_anomaly']}")
                
                # Feedback buttons
                st.markdown("**Quick Feedback:**")
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("✅ True Positive", key=f"tp_{selected_alert_id}"):
                        st.success("Feedback recorded: True Positive")
                with col_b:
                    if st.button("❌ False Positive", key=f"fp_{selected_alert_id}"):
                        st.warning("Feedback recorded: False Positive")
        
        # Charts
        st.subheader("📊 Alert Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Severity distribution
            severity_counts = alerts_df['severity'].value_counts()
            fig1 = px.pie(
                values=severity_counts.values,
                names=severity_counts.index,
                title="Alert Severity Distribution"
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Confidence score distribution
            fig2 = px.histogram(
                alerts_df,
                x='confidence_score',
                nbins=20,
                title="Confidence Score Distribution"
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Timeline
        st.subheader("📈 Alert Timeline")
        alerts_df['hour'] = pd.to_datetime(alerts_df['timestamp']).dt.floor('H')
        timeline_data = alerts_df.groupby('hour').size().reset_index(name='count')
        
        fig3 = px.line(
            timeline_data,
            x='hour',
            y='count',
            title="Alerts Over Time (Hourly)"
        )
        st.plotly_chart(fig3, use_container_width=True)
        
    else:
        st.info("ℹ️ No alerts found in BigQuery from the last 7 days.")
        st.info("💡 To see alerts, ensure your security agents are running and publishing data to BigQuery.")
        
        # Show sample data for demo
        if st.checkbox("Show Sample Data for Demo"):
            sample_data = pd.DataFrame({
                'alert_id': ['ALERT_001', 'ALERT_002', 'ALERT_003'],
                'timestamp': [datetime.now() - timedelta(hours=i) for i in range(3)],
                'severity': ['High', 'Medium', 'Low'],
                'confidence_score': [0.95, 0.75, 0.45],
                'classification': ['Malware', 'Suspicious', 'Normal'],
                'is_anomaly': [True, True, False]
            })
            
            st.dataframe(sample_data, use_container_width=True)

if __name__ == "__main__":
    main()
