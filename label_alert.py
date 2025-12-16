import argparse
import logging
import json
from typing import Optional
from google.cloud import firestore
from ground_truth_manager import GroundTruthManager
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def json_datetime_serializer(obj):
    """Custom JSON serializer for datetime objects."""
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

# --- Live function to get raw alert data ---
def get_raw_alert_data(db: firestore.Client, alert_id: str) -> Optional[dict]:
    """
    Fetches raw alert data for a given alert ID from the 'security_alerts' collection.
    """
    logger.info(f"Fetching raw data for alert_id: {alert_id} from 'security_alerts' collection...")
    try:
        # NOTE: This assumes your live, unclassified alerts are in a collection named 'security_alerts'
        doc_ref = db.collection("security_alerts").document(alert_id)
        doc = doc_ref.get()
        if doc.exists:
            logger.info(f"Successfully retrieved raw alert data for alert_id: {alert_id}")
            return doc.to_dict()
        else:
            logger.warning(f"No alert found in 'security_alerts' for alert_id: {alert_id}")
            return None
    except Exception as e:
        logger.error(f"Failed to retrieve alert data for alert_id '{alert_id}': {e}")
        return None
# --- End of live function ---

def get_analyst_input(prompt: str, validation_fn=None, valid_options=None):
    """Helper function to get and validate analyst input."""
    while True:
        if valid_options:
            print(f"\n{prompt} ({'/'.join(valid_options)}):")
            for i, option in enumerate(valid_options, 1):
                print(f"  {i}. {option}")
        else:
            print(f"\n{prompt}:")
        
        user_input = input("> ").strip()

        if valid_options:
            try:
                # Allow user to enter number or the string itself
                if user_input.isdigit() and 1 <= int(user_input) <= len(valid_options):
                    user_input = valid_options[int(user_input) - 1]
                elif user_input.lower() not in [opt.lower() for opt in valid_options]:
                    logger.error(f"Invalid option. Please choose from: {', '.join(valid_options)}")
                    continue
            except (ValueError, IndexError):
                 logger.error(f"Invalid input. Please enter a number from 1 to {len(valid_options)} or one of the options.")
                 continue

        if validation_fn:
            is_valid, error_message = validation_fn(user_input)
            if not is_valid:
                logger.error(error_message)
                continue
        
        return user_input

def validate_confidence(value: str) -> tuple[bool, str]:
    """Validates that confidence is a float between 0.0 and 1.0."""
    try:
        score = float(value)
        if 0.0 <= score <= 1.0:
            return True, ""
        else:
            return False, "Confidence must be between 0.0 and 1.0."
    except ValueError:
        return False, "Invalid input. Please enter a number."

def main():
    parser = argparse.ArgumentParser(description="A command-line tool for security analysts to label alerts.")
    parser.add_argument("alert_id", type=str, help="The unique ID of the alert to be labeled.")
    parser.add_argument("--project_id", type=str, default="chronicle-dev-2be9", help="The Google Cloud project ID.")
    parser.add_argument("--analyst_id", type=str, required=True, help="The ID of the analyst performing the labeling.")
    
    args = parser.parse_args()

    # Initialize the manager
    gt_manager = GroundTruthManager(project_id=args.project_id)
    if not gt_manager.db:
        logger.critical("Could not initialize GroundTruthManager. Exiting.")
        return

    # 1. Fetch and display the raw alert data
    raw_alert = get_raw_alert_data(gt_manager.db, args.alert_id)
    if not raw_alert:
        logger.critical("Could not fetch the alert data. Please check the alert ID and try again. Exiting.")
        return
        
    print("--- Alert Details ---")
    print(json.dumps(raw_alert, indent=2, default=json_datetime_serializer))
    print("---------------------")

    # 2. Get input from the analyst
    print("\nPlease provide the classification for this alert:")
    
    classification = get_analyst_input(
        prompt="Classification",
        valid_options=["true_positive", "false_positive", "benign"]
    )
    
    incident_type = get_analyst_input(
        prompt="Enter a brief incident type (e.g., 'Brute Force', 'Data Exfiltration', 'Admin Activity')"
    )

    confidence_str = get_analyst_input(
        prompt="Enter your confidence in this classification (0.0 - 1.0)",
        validation_fn=validate_confidence
    )
    confidence = float(confidence_str)

    notes = get_analyst_input(
        prompt="Enter any additional notes (optional, press Enter to skip)"
    )

    # 3. Confirm and save the record
    print("\n--- Summary ---")
    print(f"Alert ID:         {args.alert_id}")
    print(f"Analyst ID:       {args.analyst_id}")
    print(f"Classification:   {classification}")
    print(f"Incident Type:    {incident_type}")
    print(f"Confidence:       {confidence}")
    print(f"Notes:            {notes}")
    print("---------------")

    confirm = input("\nDo you want to save this ground truth record? (y/n): ").lower()
    if confirm == 'y':
        success, message = gt_manager.add_ground_truth_record(
            alert_id=args.alert_id,
            raw_alert_data=raw_alert,
            classification=classification,
            analyst_id=args.analyst_id,
            incident_type=incident_type,
            confidence=confidence,
            notes=notes
        )
        if success:
            logger.info(f"Successfully saved ground truth record. Document ID: {message}")
        else:
            logger.error(f"Failed to save record: {message}")
    else:
        logger.info("Operation cancelled. Record not saved.")

if __name__ == "__main__":
    main()
