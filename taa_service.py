from triage_analysis_agent import TriageAnalysisAgent
from google.cloud import pubsub_v1
import json
import logging
import os

PROJECT_ID = "chronicle-dev-2be9"
SUBSCRIPTION_ID = "taa-alerts-sub"

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("TAAService")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath("gatra-user-gemini.json")

def main():
    agent = TriageAnalysisAgent()
    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_ID)

    def callback(message):
        alert_data = json.loads(message.data.decode("utf-8"))
        logger.info(f"Received alert: {alert_data}")
        result = agent.process_alert(alert_data)
        logger.info(f"TAA result: {result}")
        message.ack()

    streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
    logger.info(f"TAA Service listening for ADA alerts on {subscription_path}...")

    try:
        streaming_pull_future.result()
    except KeyboardInterrupt:
        streaming_pull_future.cancel()
        logger.info("TAA Service stopped.")

if __name__ == "__main__":
    main() 