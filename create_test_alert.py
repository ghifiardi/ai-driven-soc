import logging
from google.cloud import firestore

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_alert(project_id: str, alert_id: str):
    """
    Creates a single sample alert in the 'security_alerts' collection for testing purposes.
    """
    try:
        db = firestore.Client(project=project_id)
        collection_ref = db.collection("security_alerts")
        
        # Check if the alert already exists
        if collection_ref.document(alert_id).get().exists:
            logger.warning(f"Test alert with ID '{alert_id}' already exists. No action taken.")
            return

        # Sample data for the alert
        alert_data = {
            "service_name": "compute.googleapis.com",
            "method_name": "v1.compute.instances.insert",
            "source_ip": "203.0.113.42",
            "principal_email": "compromised-account@example.com",
            "timestamp": firestore.SERVER_TIMESTAMP,
            "anomaly_score": 0.85,
            "details": "A new VM instance was created from an unusual IP address.",
            "status": "unclassified"
        }

        # Create the document
        collection_ref.document(alert_id).set(alert_data)
        logger.info(f"Successfully created test alert with ID: {alert_id} in 'security_alerts' collection.")

    except Exception as e:
        logger.critical(f"Failed to create test alert: {e}")

if __name__ == "__main__":
    # Configuration
    PROJECT_ID = "chronicle-dev-2be9"
    # This is the ID you will use to test the label_alert.py script
    TEST_ALERT_ID = "test-alert-003"
    
    logger.info(f"Attempting to create a test alert with ID '{TEST_ALERT_ID}'...")
    create_sample_alert(project_id=PROJECT_ID, alert_id=TEST_ALERT_ID)
