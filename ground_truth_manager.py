import datetime
import logging
import json
from typing import Optional

from google.cloud import firestore

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GroundTruthManager:
    """Manages the creation and retrieval of ground truth data in Firestore."""

    def __init__(self, project_id: str, collection_name: str = "ground_truth_alerts"):
        """
        Initializes the GroundTruthManager with a Firestore client.

        Args:
            project_id (str): The Google Cloud project ID.
            collection_name (str): The name of the Firestore collection for ground truth data.
        """
        try:
            self.db = firestore.Client(project=project_id)
            self.collection_ref = self.db.collection(collection_name)
            logger.info(f"GroundTruthManager initialized for project '{project_id}' and collection '{collection_name}'")
        except Exception as e:
            logger.error(f"Failed to initialize Firestore client: {e}")
            self.db = None
            self.collection_ref = None

    def add_ground_truth_record(
        self,
        alert_id: str,
        raw_alert_data: dict,
        classification: str,
        analyst_id: str,
        incident_type: str,
        confidence: float,
        notes: str = ""
    ) -> tuple[bool, str]:
        """
        Adds a new ground truth record to the Firestore collection.

        Args:
            alert_id (str): The unique identifier for the alert.
            raw_alert_data (dict): The original raw alert data.
            classification (str): The classification ('true_positive', 'false_positive', 'benign').
            analyst_id (str): The ID of the analyst providing the classification.
            incident_type (str): The type of incident (e.g., 'brute_force').
            confidence (float): The analyst's confidence in the classification (0.0 to 1.0).
            notes (str): Additional notes from the analyst.

        Returns:
            tuple[bool, str]: A tuple containing a success flag and a message (document ID or error).
        """
        if not self.collection_ref:
            return False, "Firestore collection not initialized."

        try:
            document_id = alert_id
            record = {
                "alert_id": alert_id,
                "raw_alert_data": raw_alert_data,
                "classification": classification,
                "analyst_id": analyst_id,
                "classification_time": datetime.datetime.now(datetime.timezone.utc),
                "incident_type": incident_type,
                "confidence": confidence,
                "notes": notes,
            }

            self.collection_ref.document(document_id).set(record)
            logger.info(f"Successfully added ground truth record with ID: {document_id}")
            return True, document_id
        except Exception as e:
            logger.error(f"Failed to add ground truth record for alert_id '{alert_id}': {e}")
            return False, str(e)

    def get_ground_truth_record(self, alert_id: str) -> Optional[dict]:
        """
        Retrieves a ground truth record from Firestore.

        Args:
            alert_id (str): The ID of the alert to retrieve.

        Returns:
            Optional[dict]: The record data as a dictionary, or None if not found.
        """
        if not self.collection_ref:
            logger.error("Firestore collection not initialized.")
            return None
        
        try:
            doc_ref = self.collection_ref.document(alert_id)
            doc = doc_ref.get()
            if doc.exists:
                logger.info(f"Successfully retrieved ground truth record for alert_id: {alert_id}")
                return doc.to_dict()
            else:
                logger.warning(f"No ground truth record found for alert_id: {alert_id}")
                return None
        except Exception as e:
            logger.error(f"Failed to retrieve ground truth record for alert_id '{alert_id}': {e}")
            return None

if __name__ == '__main__':
    # This is an example of how to use the GroundTruthManager.
    # Replace with your actual project ID and alert data.
    
    # Configuration
    PROJECT_ID = "chronicle-dev-2be9"  # <--- IMPORTANT: Replace with your GCP project ID
    
    # Initialize the manager
    gt_manager = GroundTruthManager(project_id=PROJECT_ID)

    if gt_manager.db:
        # Example Data
        example_alert_id = f"gt-example-{int(datetime.datetime.now().timestamp())}"
        example_raw_data = {
            "service_name": "storage.googleapis.com",
            "method_name": "storage.buckets.getIamPolicy",
            "source_ip": "198.51.100.77",
            "principal_email": "user@example.com",
            "timestamp": "2025-07-05T12:40:00Z"
        }

        # 1. Add a new ground truth record
        print("--- Adding a new ground truth record ---")
        success, message = gt_manager.add_ground_truth_record(
            alert_id=example_alert_id,
            raw_alert_data=example_raw_data,
            classification="false_positive",
            analyst_id="analyst-01",
            incident_type="Normal administrative activity",
            confidence=0.95,
            notes="This was a scheduled check by an authorized administrator."
        )

        if success:
            print(f"Record added successfully. Document ID: {message}")
        else:
            print(f"Failed to add record: {message}")

        # 2. Retrieve the record we just added
        print("\n--- Retrieving the ground truth record ---")
        record = gt_manager.get_ground_truth_record(alert_id=example_alert_id)

        if record:
            print("Retrieved record:")
            print(json.dumps(record, indent=2, default=str))
        else:
            print("Could not retrieve the record.")
            
    else:
        print("Could not initialize GroundTruthManager. Please check your GCP authentication and project ID.")
