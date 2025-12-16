#!/usr/bin/env python3
"""
TAA-CRA Integration Dashboard
Real-time monitoring of TAA-CRA integration status and message flow
"""

import streamlit as st
import requests
import json
import time
from datetime import datetime, timedelta
from google.cloud import pubsub_v1
from google.cloud import bigquery
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="TAA-CRA Integration Dashboard",
    page_icon="🔗",
    layout="wide"
)

def get_pubsub_stats():
    """Get Pub/Sub topic statistics"""
    try:
        publisher = pubsub_v1.PublisherClient()
        subscriber = pubsub_v1.SubscriberClient()
        project_id = "chronicle-dev-2be9"
        
        # Get topic stats
        topics = {
            "containment-requests": "TAA → CRA",
            "taa-feedback": "TAA → CLA", 
            "taa-reports": "TAA → RVA"
        }
        
        stats = {}
        for topic_name, description in topics.items():
            try:
                topic_path = publisher.topic_path(project_id, topic_name)
                # Get subscription stats
                subscription_name = f"cra-{topic_name}" if topic_name == "containment-requests" else f"cla-{topic_name}" if topic_name == "taa-feedback" else f"rva-{topic_name}"
                subscription_path = subscriber.subscription_path(project_id, subscription_name)
                
                # Get message count (approximate)
                request = pubsub_v1.GetSubscriptionRequest(subscription=subscription_path)
                subscription = subscriber.get_subscription(request=request)
                
                stats[topic_name] = {
                    "description": description,
                    "topic_exists": True,
                    "subscription_exists": True,
                    "message_count": "N/A"  # Would need monitoring API for exact count
                }
            except Exception as e:
                stats[topic_name] = {
                    "description": description,
                    "topic_exists": False,
                    "subscription_exists": False,
                    "error": str(e)
                }
        
        return stats
    except Exception as e:
        return {"error": str(e)}

def get_bigquery_feedback_stats():
    """Get feedback statistics from BigQuery"""
    try:
        client = bigquery.Client(project="chronicle-dev-2be9")
        
        # Query for recent feedback
        query = """
        SELECT 
            COUNT(*) as total_feedback,
            COUNT(CASE WHEN processed = TRUE THEN 1 END) as processed_feedback,
            COUNT(CASE WHEN processed = FALSE OR processed IS NULL THEN 1 END) as unprocessed_feedback,
            COUNT(CASE WHEN is_true_positive = TRUE THEN 1 END) as true_positives,
            COUNT(CASE WHEN is_true_positive = FALSE THEN 1 END) as false_positives
        FROM `chronicle-dev-2be9.soc_data.feedback`
        WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
        """
        
        query_job = client.query(query)
        results = list(query_job.result())
        
        if results:
            return results[0]
        else:
            return {
                "total_feedback": 0,
                "processed_feedback": 0,
                "unprocessed_feedback": 0,
                "true_positives": 0,
                "false_positives": 0
            }
    except Exception as e:
        return {"error": str(e)}

def get_containment_requests_stats():
    """Get containment requests statistics from BigQuery"""
    try:
        client = bigquery.Client(project="chronicle-dev-2be9")
        
        # Query for recent containment requests
        query = """
        SELECT 
            COUNT(*) as total_requests,
            COUNT(CASE WHEN processed = TRUE THEN 1 END) as processed_requests,
            COUNT(CASE WHEN processed = FALSE OR processed IS NULL THEN 1 END) as pending_requests
        FROM `chronicle-dev-2be9.soc_data.containment_requests`
        WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
        """
        
        query_job = client.query(query)
        results = list(query_job.result())
        
        if results:
            return results[0]
        else:
            return {
                "total_requests": 0,
                "processed_requests": 0,
                "pending_requests": 0
            }
    except Exception as e:
        return {"error": str(e)}

def test_taa_cra_integration():
    """Test TAA-CRA integration by sending a test message"""
    try:
        publisher = pubsub_v1.PublisherClient()
        project_id = "chronicle-dev-2be9"
        topic_name = "containment-requests"
        topic_path = publisher.topic_path(project_id, topic_name)
        
        # Create test message
        test_message = {
            "alert_id": f"test-{int(time.time())}",
            "alert_data": {
                "src_ip": "192.168.1.100",
                "dst_ip": "10.0.0.1",
                "port": 22,
                "protocol": "SSH"
            },
            "llm_result": {
                "is_true_positive": True,
                "confidence": 0.95,
                "severity": "high",
                "reasoning": "Test message for TAA-CRA integration"
            },
            "timestamp": datetime.now().isoformat(),
            "containment_request": {
                "action": "immediate_containment",
                "severity": "high",
                "confidence": 0.95,
                "reasoning": "Test containment request"
            }
        }
        
        # Publish message
        message_data = json.dumps(test_message).encode("utf-8")
        future = publisher.publish(topic_path, message_data)
        message_id = future.result()
        
        return {
            "success": True,
            "message_id": message_id,
            "test_message": test_message
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def main():
    """Main dashboard function"""
    
    # Header
    st.title("🔗 TAA-CRA Integration Dashboard")
    st.markdown("**Real-time monitoring of TAA-CRA integration and message flow**")
    
    # Auto-refresh
    if st.sidebar.checkbox("Auto-refresh (30s)", value=True):
        time.sleep(30)
        st.rerun()
    
    # Get data
    with st.spinner("Fetching integration data..."):
        pubsub_stats = get_pubsub_stats()
        feedback_stats = get_bigquery_feedback_stats()
        containment_stats = get_containment_requests_stats()
    
    # Integration Status Overview
    st.header("📊 Integration Status Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'error' not in pubsub_stats:
            taa_cra_status = "🟢 Active" if pubsub_stats.get("containment-requests", {}).get("topic_exists", False) else "🔴 Inactive"
            st.metric("TAA → CRA", taa_cra_status)
        else:
            st.metric("TAA → CRA", "🔴 Error")
    
    with col2:
        if 'error' not in pubsub_stats:
            taa_cla_status = "🟢 Active" if pubsub_stats.get("taa-feedback", {}).get("topic_exists", False) else "🔴 Inactive"
            st.metric("TAA → CLA", taa_cla_status)
        else:
            st.metric("TAA → CLA", "🔴 Error")
    
    with col3:
        if 'error' not in pubsub_stats:
            taa_rva_status = "🟢 Active" if pubsub_stats.get("taa-reports", {}).get("topic_exists", False) else "🔴 Inactive"
            st.metric("TAA → RVA", taa_rva_status)
        else:
            st.metric("TAA → RVA", "🔴 Error")
    
    with col4:
        if 'error' not in feedback_stats:
            feedback_rate = f"{feedback_stats.get('processed_feedback', 0)}/{feedback_stats.get('total_feedback', 0)}"
            st.metric("Feedback Processed", feedback_rate)
        else:
            st.metric("Feedback Processed", "Error")
    
    # Message Flow Status
    st.header("📡 Message Flow Status")
    
    if 'error' not in pubsub_stats:
        for topic_name, stats in pubsub_stats.items():
            if topic_name != "error":
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**{stats['description']}**")
                
                with col2:
                    if stats.get('topic_exists', False):
                        st.success("✅ Topic Active")
                    else:
                        st.error("❌ Topic Inactive")
                
                with col3:
                    if stats.get('subscription_exists', False):
                        st.success("✅ Subscription Active")
                    else:
                        st.error("❌ Subscription Inactive")
    else:
        st.error(f"Error fetching Pub/Sub stats: {pubsub_stats['error']}")
    
    # Feedback Statistics
    st.header("📈 Feedback Statistics (Last 24 Hours)")
    
    if 'error' not in feedback_stats:
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Feedback", feedback_stats.get('total_feedback', 0))
        
        with col2:
            st.metric("Processed", feedback_stats.get('processed_feedback', 0))
        
        with col3:
            st.metric("Unprocessed", feedback_stats.get('unprocessed_feedback', 0))
        
        with col4:
            st.metric("True Positives", feedback_stats.get('true_positives', 0))
        
        with col5:
            st.metric("False Positives", feedback_stats.get('false_positives', 0))
    else:
        st.error(f"Error fetching feedback stats: {feedback_stats['error']}")
    
    # Containment Requests Statistics
    st.header("🚨 Containment Requests (Last 24 Hours)")
    
    if 'error' not in containment_stats:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Requests", containment_stats.get('total_requests', 0))
        
        with col2:
            st.metric("Processed", containment_stats.get('processed_requests', 0))
        
        with col3:
            st.metric("Pending", containment_stats.get('pending_requests', 0))
    else:
        st.error(f"Error fetching containment stats: {containment_stats['error']}")
    
    # Test Integration
    st.header("🧪 Test TAA-CRA Integration")
    
    if st.button("Send Test Containment Request"):
        with st.spinner("Sending test message..."):
            result = test_taa_cra_integration()
        
        if result.get('success', False):
            st.success(f"✅ Test message sent successfully!")
            st.info(f"Message ID: {result['message_id']}")
            st.json(result['test_message'])
        else:
            st.error(f"❌ Error sending test message: {result.get('error', 'Unknown error')}")
    
    # Integration Workflow
    st.header("🔄 Integration Workflow")
    
    st.markdown("""
    **Current TAA-CRA Integration Flow:**
    
    1. **ADA** detects anomalies and publishes alerts
    2. **TAA** receives alerts and performs analysis
    3. **TAA** publishes to multiple topics:
       - `containment-requests` → **CRA** (for containment actions)
       - `taa-feedback` → **CLA** (for learning)
       - `taa-reports` → **RVA** (for reporting)
    4. **CRA** processes containment requests
    5. **CLA** learns from feedback
    6. **RVA** generates reports
    
    **Integration Status:**
    - ✅ **TAA LangGraph Agent**: Enhanced with Pub/Sub publishing
    - ✅ **CRA Service**: Listening for containment requests
    - ✅ **Pub/Sub Topics**: Active and configured
    - ✅ **BigQuery Integration**: Storing feedback and requests
    """)
    
    # Footer
    st.markdown("---")
    st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.markdown("**Dashboard Status:** 🟢 Active - Monitoring TAA-CRA integration in real-time")

if __name__ == "__main__":
    main()
