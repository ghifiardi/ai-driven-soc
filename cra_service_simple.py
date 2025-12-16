#!/usr/bin/env python3
"""
Simple CRA Service for testing TAA-CRA integration
"""

import json
import logging
import time
from google.cloud import pubsub_v1

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CRAService")

def main():
    """Main CRA service loop"""
    logger.info("Starting CRA Service...")
    
    # Initialize Pub/Sub
    subscriber = pubsub_v1.SubscriberClient()
    project_id = "chronicle-dev-2be9"
    subscription_name = "cra-containment-requests"
    subscription_path = subscriber.subscription_path(project_id, subscription_name)
    
    logger.info(f"Listening for containment requests on: {subscription_path}")
    
    # Start listening using streaming pull
    streaming_pull_future = subscriber.pull(
        request={"subscription": subscription_path, "max_messages": 10}
    )
    
    try:
        # Process messages
        for message in streaming_pull_future:
            try:
                containment_data = json.loads(message.data.decode("utf-8"))
                logger.info(f"Received containment request: {containment_data}")
                
                # Simulate containment action
                alert_id = containment_data.get('alert_id', 'unknown')
                action = containment_data.get('containment_request', {}).get('action', 'unknown')
                severity = containment_data.get('containment_request', {}).get('severity', 'unknown')
                
                logger.info(f"🚨 CONTAINMENT ACTION: {action} for alert {alert_id} (severity: {severity})")
                
                # Simulate processing time
                time.sleep(1)
                
                logger.info(f"✅ Containment action completed for alert {alert_id}")
                
                # Acknowledge the message
                message.ack()
                
            except Exception as e:
                logger.error(f"Error processing containment request: {e}")
                message.nack()
                
    except KeyboardInterrupt:
        logger.info("Shutting down CRA Service...")
        streaming_pull_future.cancel()
        streaming_pull_future.result()

if __name__ == "__main__":
    main()
