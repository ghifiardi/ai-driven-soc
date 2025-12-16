#!/usr/bin/env python3
"""
FIXED ALERT REVIEW DASHBOARD - Based on Foundation Documentation
This restores the Alert Review tab to the stable foundation version
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz
import json
import os
from google.cloud import bigquery
from google.oauth2 import service_account

# Page config
st.set_page_config(
    page_title="AI-Driven SOC Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_bigquery_client():
    """Initialize BigQuery client"""
    try:
        return bigquery.Client()
    except Exception as e:
        st.error(f"BigQuery client initialization failed: {e}")
        return None

def get_real_alerts():
    """Get real alerts from BigQuery or use sample data"""
    client = get_bigquery_client()
    if client:
        try:
            query = """
            SELECT 
                alert_id,
                timestamp,
                severity,
                classification,
                confidence_score,
                is_anomaly,
                raw_alert
            FROM `chronicle-dev-2be9.soc_data.alerts`
            ORDER BY timestamp DESC
            LIMIT 50
            """
            
            df = client.query(query).to_dataframe()
            return df
        except Exception as e:
            st.error(f"Error fetching alerts: {e}")
    
    # Return sample data as fallback
    return pd.DataFrame({
        'alert_id': ['2a4e195b-70b...', '2a4e4071-4ec...', '2a4e5766-c66...', '2a4e9081-f51...', '2a4eb967-c3a...'],
        'timestamp': [datetime.now() - timedelta(hours=i) for i in range(5)],
        'severity': ['High', 'High', 'High', 'High', 'High'],
        'classification': ['anomaly', 'anomaly', 'anomaly', 'anomaly', 'anomaly'],
        'confidence_score': [0.9, 0.7, 0.85, 0.95, 0.8],
        'is_anomaly': [True, True, True, True, True],
        'raw_alert': [None] * 5
    })

def submit_real_feedback(alert_id: str, is_true_positive: bool, confidence: float, comments: str) -> bool:
    """Persist analyst feedback to BigQuery"""
    client = get_bigquery_client()
    if not client:
        st.error("BigQuery client is not available; cannot submit feedback.")
        return False
    
    try:
        feedback_data = {
            'alert_id': alert_id,
            'is_true_positive': is_true_positive,
            'confidence': confidence,
            'comments': comments,
            'timestamp': datetime.now().isoformat(),
            'analyst_id': 'dashboard_user'
        }
        
        table_id = "chronicle-dev-2be9.soc_data.feedback"
        errors = client.insert_rows_json(table_id, [feedback_data])
        
        if errors == []:
            return True
        else:
            st.error(f"Error inserting feedback: {errors}")
            return False
    except Exception as e:
        st.error(f"Error submitting feedback: {e}")
        return False

def extract_alert_parameters(alert_data):
    """Extract parameters from alert data"""
    params = {}
    
    if isinstance(alert_data, pd.Series):
        alert_dict = alert_data.to_dict()
    else:
        alert_dict = alert_data
    
    # Extract IP addresses
    if 'raw_alert' in alert_dict and pd.notna(alert_dict.get('raw_alert')):
        try:
            raw_data = alert_dict['raw_alert']
            if isinstance(raw_data, str):
                raw_data = json.loads(raw_data)
            if isinstance(raw_data, dict):
                ips = []
                for field in ['source_ip', 'src_ip', 'destination_ip', 'dst_ip', 'ip']:
                    if field in raw_data:
                        ips.append(str(raw_data[field]))
                if ips:
                    params['ip_addresses'] = list(set(ips))
        except:
            pass
    
    return params

def alert_review_page():
    """Alert Review page - Foundation Version"""
    st.title("🔍 Alert Review & Feedback")
    st.markdown("**Real-time alert review with feedback integration**")
    
    # Keep a stable alerts snapshot during the session
    if 'alerts_df' not in st.session_state:
        st.session_state.alerts_df = get_real_alerts()

    # Manual refresh
    refresh_col1, refresh_col2 = st.columns([1,6])
    with refresh_col1:
        if st.button('Refresh Alerts'):
            st.session_state.alerts_df = get_real_alerts()
            st.rerun()
    
    alerts_df = st.session_state.alerts_df.copy()
    
    # Remove duplicates
    if not alerts_df.empty and 'alert_id' in alerts_df.columns:
        if 'timestamp' in alerts_df.columns:
            alerts_df = alerts_df.sort_values('timestamp', ascending=False)
        alerts_df = alerts_df.drop_duplicates(subset='alert_id', keep='first')
        st.session_state.alerts_df = alerts_df.copy()
    
    if not alerts_df.empty:
        # Recent Alerts Table
        st.header("Recent Alerts")
        
        # Display alerts table
        display_columns = ['alert_id', 'timestamp', 'severity', 'classification', 'confidence_score', 'is_anomaly']
        available_columns = [col for col in display_columns if col in alerts_df.columns]
        
        st.dataframe(
            alerts_df[available_columns].head(10),
            use_container_width=True
        )
        
        # Submit Feedback Section
        st.header("Submit Feedback")
        
        # Get alert IDs for selection
        alert_ids = alerts_df['alert_id'].dropna().astype(str).tolist()
        
        if not alert_ids:
            st.warning("No alerts available for feedback")
            return
        
        # Simple selectbox for alert selection
        selected_alert = st.selectbox(
            "Select Alert for Feedback",
            options=alert_ids,
            index=0,
            key='alert_selector'
        )
        
        # Get selected alert data
        if selected_alert:
            selected_alert_data = alerts_df[alerts_df['alert_id'].astype(str) == str(selected_alert)]
            
            if not selected_alert_data.empty:
                selected_alert_data = selected_alert_data.iloc[0]
                
                # Display alert details
                st.subheader("📋 Alert Details for Review")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Alert ID", selected_alert)
                    st.metric("Severity", selected_alert_data.get('severity', 'Unknown'))
                    st.metric("Classification", selected_alert_data.get('classification', 'Unknown'))
                
                with col2:
                    # Safe timestamp handling
                    try:
                        timestamp = selected_alert_data.get('timestamp')
                        if pd.notna(timestamp):
                            if hasattr(timestamp, 'strftime'):
                                timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                            else:
                                timestamp_str = str(timestamp)[:19]
                        else:
                            timestamp_str = "N/A"
                    except:
                        timestamp_str = "Invalid timestamp"
                    
                    st.metric("Timestamp", timestamp_str)
                    st.metric("Confidence Score", f"{selected_alert_data.get('confidence_score', 0):.2f}")
                    
                    # Safe anomaly handling
                    is_anomaly = selected_alert_data.get('is_anomaly')
                    if pd.isna(is_anomaly):
                        anomaly_display = "Unknown"
                    else:
                        anomaly_display = "Yes" if is_anomaly else "No"
                    st.metric("Is Anomaly", anomaly_display)
                
                with col3:
                    st.subheader("🤖 Model Analysis")
                    
                    # Extract parameters
                    extracted_params = extract_alert_parameters(selected_alert_data)
                    
                    if extracted_params.get('ip_addresses'):
                        st.markdown(f"**IP Addresses:** {', '.join(extracted_params['ip_addresses'])}")
                    
                    # Simple recommendations
                    confidence = selected_alert_data.get('confidence_score', 0)
                    if confidence > 0.8:
                        st.info("High confidence - prioritize investigation")
                    elif confidence > 0.6:
                        st.info("Medium confidence - review with context")
                    else:
                        st.info("Low confidence - may be false positive")
                
                st.markdown("---")
                
                # Feedback form
                st.subheader("📝 Provide Feedback")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    is_true_positive = st.radio(
                        "Is this a true positive?",
                        ["Yes", "No"],
                        key="feedback_tp"
                    )
                
                with col2:
                    confidence = st.slider(
                        "Confidence in your assessment",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.8,
                        step=0.1
                    )
                
                comments = st.text_area(
                    "Additional Comments (Optional)",
                    placeholder="Provide additional context..."
                )
                
                # Submit button
                if st.button("Submit Feedback", type="primary"):
                    if selected_alert:
                        ok = submit_real_feedback(
                            alert_id=selected_alert,
                            is_true_positive=(is_true_positive == "Yes"),
                            confidence=confidence,
                            comments=comments
                        )
                        if ok:
                            st.success(f"✅ Feedback submitted for alert {selected_alert}")
                            st.info("Feedback is stored in BigQuery table `soc_data.feedback`.")
                        else:
                            st.error("Feedback submission failed. See error above.")
                    else:
                        st.error("Please select an alert to provide feedback on")
    else:
        st.warning("No alerts available for review")

def main():
    """Main dashboard function"""
    
    # Sidebar
    with st.sidebar:
        st.title("🛡️ AI-Driven SOC")
        
        # Navigation
        page = st.selectbox(
            "Select Dashboard Section",
            ["Overview & Funnel", "Alert Review", "System Status"],
            key="page_selector"
        )
        
        # System Status
        st.subheader("System Status")
        st.success("✅ ADA: Active")
        st.success("✅ CLA: Learning (66 Models)")
        st.success("✅ TAA: Google Flash 2.5")
        st.success("✅ CRA: Ready")
    
    # Main content
    if page == "Overview & Funnel":
        overview_page()
    elif page == "Alert Review":
        alert_review_page()
    elif page == "System Status":
        system_status_page()

def overview_page():
    """Overview page"""
    st.title("📊 SOC Overview & Operations Funnel")
    
    # Executive Summary
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;">
        <h2 style="color: white; margin: 0; text-align: center;">
            🎯 AI-Powered Cybersecurity Operations Center
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Accuracy", "89.2%", delta="+22.5%")
    with col2:
        st.metric("False Positive Rate", "14.8%", delta="-18.5%")
    with col3:
        st.metric("Processing Rate", "99.2%")
    with col4:
        st.metric("Model Version", "CLA v3.0.0")
    
    # Funnel
    st.subheader("📈 Security Operations Funnel")
    
    funnel_data = {
        'Stage': ['Raw Events', 'Alerts Analyzed', 'Threats Triaged', 'Incidents Contained'],
        'Count': [13203, 12428, 10670, 3396]
    }
    
    fig = go.Figure(go.Funnel(
        y=funnel_data['Stage'],
        x=funnel_data['Count'],
        textinfo="value+percent initial"
    ))
    
    st.plotly_chart(fig, use_container_width=True)

def system_status_page():
    """System Status page"""
    st.title("⚙️ System Status")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Uptime", "99.9%")
    with col2:
        st.metric("Response Time", "0.8s")
    with col3:
        st.metric("Active Alerts", "1,247")

if __name__ == "__main__":
    main()
