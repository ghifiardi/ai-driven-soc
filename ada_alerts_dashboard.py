import streamlit as st
from google.cloud import pubsub_v1
import json
import threading

PROJECT_ID = "chronicle-dev-2be9"
SUBSCRIPTION_ID = "dashboard-ada-alerts-sub"  # Use your dedicated dashboard subscription

st.title("Live ADA Alerts Dashboard (GCP Pub/Sub)")

# Shared list to store alerts
alerts = []

def listen_for_alerts():
    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_ID)

    def callback(message):
        alert_data = json.loads(message.data.decode("utf-8"))
        alerts.append(alert_data)
        message.ack()

    streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
    try:
        streaming_pull_future.result()
    except Exception as e:
        st.error(f"Error in Pub/Sub subscription: {e}")

# Start the Pub/Sub listener in a background thread
if 'listener_started' not in st.session_state:
    thread = threading.Thread(target=listen_for_alerts, daemon=True)
    thread.start()
    st.session_state['listener_started'] = True

st.info("Listening for ADA alerts from Pub/Sub...")

# Display the most recent alerts
if alerts:
    st.subheader("Recent ADA Alerts")
    for alert in alerts[-10:][::-1]:
        st.json(alert)
else:
    st.write("No alerts received yet. Waiting for messages...") 