import logging
from google.cloud import firestore
from datetime import datetime, timezone

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ID = "chronicle-dev-2be9"
ANALYST_ID = "sandra.c" # The analyst ID to use for the batch-created labels

# The starting number for the batch of alerts to create
START_ALERT_NUM = 4
# The total number of alerts to create in this batch
NUM_ALERTS_TO_CREATE = 4

def get_sample_raw_alert_data(alert_id: str) -> dict:
    """Generates a sample raw alert with a unique timestamp."""
    return {
        "principal_email": "compromised-account@example.com",
        "service_name": "compute.googleapis.com",
        "method_name": "v1.compute.instances.insert",
        "source_ip": "203.0.113.42",
        "details": f"A new VM instance was created from an unusual IP address (Batch {alert_id}).",
        "anomaly_score": 0.85,
        "status": "unclassified",
        "timestamp": datetime.now(timezone.utc)
    }

def create_batch_alerts(project_id: str):
    """
    Creates and labels a batch of test alerts to quickly populate the dataset.
    """
    try:
        db = firestore.Client(project=project_id)
        logger.info(f"Successfully initialized Firestore client for project '{project_id}'.")
    except Exception as e:
        logger.critical(f"Failed to initialize Firestore client: {e}")
        return

    end_alert_num = START_ALERT_NUM + NUM_ALERTS_TO_CREATE
    logger.info(f"Starting batch creation of {NUM_ALERTS_TO_CREATE} alerts, from alert-{START_ALERT_NUM:03} to alert-{end_alert_num - 1:03}...")

    for i in range(START_ALERT_NUM, end_alert_num):
        alert_id = f"test-alert-{i:03}"
        is_true_positive = i % 2 == 0  # Alternate between True Positive and False Positive

        # --- Step 1: Create the raw alert in 'security_alerts' ---
        raw_alert_data = get_sample_raw_alert_data(alert_id)
        try:
            db.collection("security_alerts").document(alert_id).set(raw_alert_data)
            logger.info(f"[{i}/{end_alert_num-1}] Successfully created raw alert: {alert_id}")
        except Exception as e:
            logger.error(f"Failed to create raw alert {alert_id}: {e}")
            continue # Skip to the next alert if this one fails

        # --- Step 2: Create the labeled record in 'ground_truth_alerts' ---
        classification = "true_positive" if is_true_positive else "false_positive"
        ground_truth_record = {
            "alert_id": alert_id,
            "analyst_id": ANALYST_ID,
            "classification": classification,
            "classification_time": firestore.SERVER_TIMESTAMP,
            "confidence": 0.95,
            "incident_type": f"Batch-Generated {classification}",
            "notes": "This record was created automatically by the batch_labeler.py script.",
            "raw_alert_data": raw_alert_data
        }

        try:
            db.collection("ground_truth_alerts").document(alert_id).set(ground_truth_record)
            logger.info(f"[{i}/{end_alert_num-1}] Successfully created ground truth label for {alert_id} as '{classification}'.")
        except Exception as e:
            logger.error(f"Failed to create ground truth for {alert_id}: {e}")

    logger.info("Batch alert creation process complete.")

if __name__ == "__main__":
    create_batch_alerts(project_id=PROJECT_ID)
