import streamlit as st
import re
from datetime import datetime

LOG_FILE = "taa_service.log"

st.title("TAA Service Dashboard")

# Read log file
try:
    with open(LOG_FILE, "r") as f:
        log_lines = f.readlines()
except FileNotFoundError:
    st.error(f"Log file {LOG_FILE} not found.")
    st.stop()

# Parse log lines
alert_received = []
feedback_published = []
containment_published = []

for line in log_lines:
    if "Received alert:" in line:
        m = re.search(r"Received alert: (\{.*\})", line)
        if m:
            alert = m.group(1)
            alert_id = re.search(r"'alert_id': '([^']+)'", alert)
            alert_id = alert_id.group(1) if alert_id else "unknown"
            timestamp = line[:19]
            alert_received.append((timestamp, alert_id, alert))
    elif "Published feedback for alert" in line:
        m = re.search(r"Published feedback for alert ([^\s]+)", line)
        if m:
            timestamp = line[:19]
            feedback_published.append((timestamp, m.group(1)))
    elif "Published containment request for alert" in line:
        m = re.search(r"Published containment request for alert ([^\s]+)", line)
        if m:
            timestamp = line[:19]
            containment_published.append((timestamp, m.group(1)))

# Dashboard metrics
st.metric("Alerts Processed", len(alert_received))
st.metric("Feedback Published", len(feedback_published))
st.metric("Containment Requests", len(containment_published))

st.subheader("Recent Alerts Processed")
view_mode = st.radio("View mode", ["Compact (10)", "Full (all)"])
if view_mode == "Compact (10)":
    display_alerts = alert_received[-10:][::-1]
else:
    display_alerts = alert_received[::-1]
for ts, alert_id, alert in display_alerts:
    st.write(f"[{ts}] {alert_id}")
    try:
        alert_obj = eval(alert)
        # Try to extract analysis_summary if present
        analysis_summary = None
        if isinstance(alert_obj, dict):
            # If the alert is a dict with 'analysis' or 'decision' keys
            if 'analysis' in alert_obj and isinstance(alert_obj['analysis'], dict):
                analysis_summary = alert_obj['analysis'].get('analysis_summary')
            elif 'decision' in alert_obj and isinstance(alert_obj['decision'], dict):
                analysis_summary = alert_obj['decision'].get('analysis_summary')
            elif 'analysis_summary' in alert_obj:
                analysis_summary = alert_obj.get('analysis_summary')
        if analysis_summary:
            st.markdown(f"**LLM Reasoning:** {analysis_summary}")
        with st.expander("Show full alert JSON"):
            st.json(alert_obj)
    except Exception:
        st.code(alert, language="json")

st.subheader("Recent Feedback Published")
for ts, alert_id in feedback_published[-10:][::-1]:
    st.write(f"[{ts}] Feedback for {alert_id}")

st.subheader("Recent Containment Requests")
for ts, alert_id in containment_published[-10:][::-1]:
    st.write(f"[{ts}] Containment for {alert_id}") 