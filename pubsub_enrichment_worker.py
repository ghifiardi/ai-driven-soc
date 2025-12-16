import os
import json
from google.cloud import pubsub_v1
from concurrent.futures import TimeoutError
from taa_langchain_enrichment import enrich_alert

# Use environment variables for configuration
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "your-gcp-project-id")
ALERT_SUBSCRIPTION = os.environ.get("ALERT_SUBSCRIPTION", "your-alert-subscription")
ENRICHED_TOPIC = os.environ.get("ENRICHED_TOPIC", "your-enriched-alerts-topic")

subscriber = pubsub_v1.SubscriberClient()
publisher = pubsub_v1.PublisherClient()
subscription_path = subscriber.subscription_path(PROJECT_ID, ALERT_SUBSCRIPTION)
enriched_topic_path = publisher.topic_path(PROJECT_ID, ENRICHED_TOPIC)

def callback(message):
    try:
        alert = json.loads(message.data.decode("utf-8"))
        print("Received alert:", alert)
        enriched = enrich_alert(alert)
        print("Enriched alert:", json.dumps(enriched, indent=2))
        # Publish to enriched topic
        publisher.publish(enriched_topic_path, json.dumps(enriched).encode("utf-8"))
        message.ack()
    except Exception as e:
        print("Error processing alert:", e)
        message.nack()

print(f"Listening for alerts on {subscription_path} ...")
streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)

try:
    streaming_pull_future.result()
except KeyboardInterrupt:
    streaming_pull_future.cancel() 