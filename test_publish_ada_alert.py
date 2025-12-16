from google.cloud import pubsub_v1
import json

PROJECT_ID = "chronicle-dev-2be9"
TOPIC_ID = "ada-alerts"

publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(PROJECT_ID, TOPIC_ID)

# Example ADA alert data
alert_data = {
    "alert_id": "ADA-20240708123456-0001",
    "timestamp": "2024-07-08T12:34:56Z",
    "anomaly_score": 0.92,
    "confidence": 0.91,
    "alert_type": "ANOMALY_DETECTION",
    "description": "Anomalous login activity detected with high confidence.",
    "raw_data_reference": "log-20240708-0001",
    "log_data": {
        "log_id": "log-20240708-0001",
        "timestamp": "2024-07-08T12:34:56Z",
        "duration": 45,
        "protocol_type": "tcp",
        "service": "ssh",
        "flag": "SF",
        "src_bytes": 500,
        "dst_bytes": 2000,
        "src_ip": "10.1.2.3",
        "dst_ip": "203.0.113.42",
        "domain": "testdomain.com"
    }
}

future = publisher.publish(topic_path, json.dumps(alert_data).encode("utf-8"))
print(f"Published test ADA alert to {topic_path}, message ID: {future.result()}") 