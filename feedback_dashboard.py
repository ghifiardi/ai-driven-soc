#!/usr/bin/env python3
"""
Feedback Dashboard - Analyst Interface for Providing Feedback
"""

import streamlit as st
import pandas as pd
from google.cloud import bigquery
import json
from datetime import datetime

st.set_page_config(
    page_title="Feedback Dashboard",
    page_icon="📝",
    layout="wide"
)

def get_unreviewed_alerts():
    """Get alerts that need feedback"""
    try:
        client = bigquery.Client(project="chronicle-dev-2be9")

        query = """
        SELECT
            alert_id,
            confidence_score,
            classification,
            timestamp,
            is_anomaly,
            raw_alert
        FROM `chronicle-dev-2be9.soc_data.processed_alerts`
        WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
        ORDER BY timestamp DESC
        LIMIT 50
        """

        df = client.query(query).to_dataframe()
        return df
    except Exception as e:
        st.error(f"Error fetching alerts: {e}")
        return pd.DataFrame()

def submit_feedback(alert_id, is_true_positive, confidence, comments):
    """Submit feedback to BigQuery"""
    try:
        client = bigquery.Client(project="chronicle-dev-2be9")

        feedback_data = {
            "alert_id": alert_id,
            "is_true_positive": is_true_positive,
            "analyst_comments": comments,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "processed": False
        }

        errors = client.insert_rows_json(
            "chronicle-dev-2be9.soc_data.feedback",
            [feedback_data]
        )

        if errors:
            return False, f"Error: {errors}"
        else:
            return True, "Feedback submitted successfully!"

    except Exception as e:
        return False, f"Error: {e}"

def main():
    st.title("📝 SOC Analyst Feedback Dashboard")
    st.markdown("Review alerts and provide feedback for continuous learning")

    # Get alerts
    alerts_df = get_unreviewed_alerts()

    if alerts_df.empty:
        st.info("No alerts available for review")
        return

    st.subheader("📋 Alerts for Review")

    # Display alerts in a table
    for i, (_, alert) in enumerate(alerts_df.iterrows()):
        with st.expander(f"Alert {i+1}: {alert['alert_id'][:20]}..."):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Classification:** {alert['classification']}")
                st.write(f"**Confidence:** {alert['confidence_score']:.3f}")
                st.write(f"**Timestamp:** {alert['timestamp']}")

            with col2:
                st.write("**Your Assessment:**")

                # Feedback form
                with st.form(f"feedback_{i}"):
                    is_true_positive = st.radio(
                        "Is this a true positive?",
                        ["True Positive", "False Positive"],
                        key=f"tp_{i}"
                    )

                    analyst_confidence = st.slider(
                        "Your confidence in this assessment",
                        0.0, 1.0, 0.8, 0.1,
                        key=f"conf_{i}"
                    )

                    comments = st.text_area(
                        "Comments (optional)",
                        placeholder="Add any additional context or reasoning...",
                        key=f"comments_{i}"
                    )

                    submitted = st.form_submit_button("Submit Feedback")

                    if submitted:
                        success, message = submit_feedback(
                            alert['alert_id'],
                            is_true_positive == "True Positive",
                            analyst_confidence,
                            comments
                        )

                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)

    # Show recent feedback submissions
    st.subheader("📝 Recent Feedback Submissions")

    try:
        client = bigquery.Client(project="chronicle-dev-2be9")
        feedback_query = """
        SELECT
            alert_id,
            is_true_positive,
            confidence,
            analyst_comments,
            timestamp
        FROM `chronicle-dev-2be9.soc_data.feedback`
        WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)
        ORDER BY timestamp DESC
        LIMIT 10
        """

        feedback_df = client.query(feedback_query).to_dataframe()

        if not feedback_df.empty:
            st.dataframe(feedback_df)
        else:
            st.info("No recent feedback submissions")

    except Exception as e:
        st.error(f"Error fetching feedback: {e}")

if __name__ == "__main__":
    main()
