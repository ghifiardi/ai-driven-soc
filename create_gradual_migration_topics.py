#!/usr/bin/env python3
"""
Create Pub/Sub Topics for Gradual Migration
==========================================

This script creates the necessary Pub/Sub topics for gradual migration:
- enhanced-taa-results: Results from enhanced TAA
- taa-comparison: A/B comparison data

Author: AI-Driven SOC Team
Date: 2025-09-18
"""

from google.cloud import pubsub_v1
import logging

# Configuration
PROJECT_ID = "chronicle-dev-2be9"
TOPICS = [
    "enhanced-taa-results",
    "taa-comparison"
]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CreateTopics")

def create_topic(topic_id: str):
    """Create a Pub/Sub topic"""
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(PROJECT_ID, topic_id)
    
    try:
        topic = publisher.create_topic(request={"name": topic_path})
        logger.info(f"✅ Created topic: {topic.name}")
        return True
    except Exception as e:
        if "already exists" in str(e).lower():
            logger.info(f"ℹ️ Topic already exists: {topic_id}")
            return True
        else:
            logger.error(f"❌ Failed to create topic {topic_id}: {e}")
            return False

def create_subscription(topic_id: str, subscription_id: str):
    """Create a subscription for the topic"""
    publisher = pubsub_v1.PublisherClient()
    subscriber = pubsub_v1.SubscriberClient()
    
    topic_path = publisher.topic_path(PROJECT_ID, topic_id)
    subscription_path = subscriber.subscription_path(PROJECT_ID, subscription_id)
    
    try:
        subscription = subscriber.create_subscription(
            request={"name": subscription_path, "topic": topic_path}
        )
        logger.info(f"✅ Created subscription: {subscription.name}")
        return True
    except Exception as e:
        if "already exists" in str(e).lower():
            logger.info(f"ℹ️ Subscription already exists: {subscription_id}")
            return True
        else:
            logger.error(f"❌ Failed to create subscription {subscription_id}: {e}")
            return False

def main():
    """Create all topics and subscriptions"""
    logger.info(f"🚀 Creating Pub/Sub topics for gradual migration in project: {PROJECT_ID}")
    
    success_count = 0
    total_count = len(TOPICS)
    
    for topic_id in TOPICS:
        logger.info(f"📡 Creating topic: {topic_id}")
        if create_topic(topic_id):
            # Create subscription for each topic
            subscription_id = f"{topic_id}-subscription"
            if create_subscription(topic_id, subscription_id):
                success_count += 1
    
    logger.info(f"✅ Created {success_count}/{total_count} topics successfully")
    
    if success_count == total_count:
        logger.info("🎉 All topics created successfully!")
        logger.info("📋 Available topics:")
        for topic_id in TOPICS:
            logger.info(f"   - {topic_id} (subscription: {topic_id}-subscription)")
    else:
        logger.warning(f"⚠️ {total_count - success_count} topics failed to create")

if __name__ == "__main__":
    main()


