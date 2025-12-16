from google.cloud import pubsub_v1

# Set your GCP project ID
PROJECT_ID = "chronicle-dev-2be9"

# Define topic and subscription names
TOPICS = {
    "ada_alerts": "ada-alerts",           # ADA publishes, TAA subscribes
    "taa_feedback": "taa-feedback",        # TAA publishes, CLA/ADA subscribes
}

# Add new topics for manual review and containment
TOPICS["manual_review"] = "taa-manual-review"
TOPICS["containment_requests"] = "containment-requests"

SUBSCRIPTIONS = {
    "taa_alerts_sub": ("ada-alerts", "taa-alerts-sub"),   # TAA subscribes to ADA alerts
    "cla_feedback_sub": ("taa-feedback", "cla-feedback-sub"), # CLA subscribes to TAA feedback
}

def create_topic(publisher, topic_name):
    topic_path = publisher.topic_path(PROJECT_ID, topic_name)
    try:
        publisher.create_topic(request={"name": topic_path})
        print(f"Created topic: {topic_path}")
    except Exception as e:
        print(f"Topic {topic_path} may already exist: {e}")

def create_subscription(subscriber, topic_name, subscription_name):
    topic_path = subscriber.topic_path(PROJECT_ID, topic_name).replace("subscriptions", "topics")
    subscription_path = subscriber.subscription_path(PROJECT_ID, subscription_name)
    try:
        subscriber.create_subscription(
            request={"name": subscription_path, "topic": topic_path}
        )
        print(f"Created subscription: {subscription_path} for topic: {topic_path}")
    except Exception as e:
        print(f"Subscription {subscription_path} may already exist: {e}")

def main():
    publisher = pubsub_v1.PublisherClient()
    subscriber = pubsub_v1.SubscriberClient()

    # Create topics
    for topic in TOPICS.values():
        create_topic(publisher, topic)

    # Create subscriptions
    for sub_name, (topic, sub) in SUBSCRIPTIONS.items():
        create_subscription(subscriber, topic, sub)

if __name__ == "__main__":
    main() 