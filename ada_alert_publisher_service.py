import time
import json
import random
from google.cloud import pubsub_v1
import logging

PROJECT_ID = "chronicle-dev-2be9"
TOPIC_ID = "ada-alerts"
PUBLISH_INTERVAL = 30  # seconds

publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(PROJECT_ID, TOPIC_ID)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("ADAAlertPublisherService")

def generate_random_alert(alert_num):
    return {
        "alert_id": f"ADA-20240708-{alert_num:04d}",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "anomaly_score": round(random.uniform(0.7, 0.99), 2),
        "confidence": round(random.uniform(0.7, 0.99), 2),
        "alert_type": "ANOMALY_DETECTION",
        "description": "Simulated anomalous activity detected.",
        "raw_data_reference": f"log-20240708-{alert_num:04d}",
        "log_data": {
            "log_id": f"log-20240708-{alert_num:04d}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "duration": random.randint(10, 300),
            "protocol_type": random.choice(["tcp", "udp"]),
            "service": random.choice(["ssh", "http", "ftp", "smtp"]),
            "flag": random.choice(["SF", "REJ", "S0"]),
            "src_bytes": random.randint(100, 5000),
            "dst_bytes": random.randint(100, 20000),
            "src_ip": f"10.1.2.{random.randint(1, 254)}",
            "dst_ip": f"203.0.113.{random.randint(1, 254)}",
            "domain": random.choice(["testdomain.com", "example.com", "corp.local"])
        }
    }

def main():
    alert_num = 1
    logger.info(f"ADA Alert Publisher Service started. Publishing to {topic_path} every {PUBLISH_INTERVAL} seconds.")
    while True:
        alert_data = generate_random_alert(alert_num)
        future = publisher.publish(topic_path, json.dumps(alert_data).encode("utf-8"))
        logger.info(f"Published ADA alert: {alert_data['alert_id']} (message ID: {future.result()})")
        alert_num += 1
        time.sleep(PUBLISH_INTERVAL)

if __name__ == "__main__":
    main() 