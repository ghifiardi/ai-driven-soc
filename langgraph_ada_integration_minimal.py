"""
Minimal test script for LangGraph ADA BigQuery integration.
"""

import logging
import os
from bigquery_client import BigQueryClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("LangGraphADA")

if __name__ == "__main__":
    # Get configuration from environment variables
    project_id = os.getenv("BIGQUERY_PROJECT_ID")
    dataset_id = os.getenv("BIGQUERY_DATASET_ID")
    table_id = os.getenv("BIGQUERY_TABLE_ID")
    
    logger.info("Starting minimal test version of LangGraph ADA")
    logger.info("Testing BigQuery connection...")
    
    # Create BigQuery client
    try:
        bq_client = BigQueryClient(
            project_id=project_id,
            dataset_id=dataset_id,
            table_id=table_id
        )
        
        # Test fetching alerts
        alerts = bq_client.fetch_new_alerts(limit=5)
        logger.info(f"Successfully fetched {len(alerts)} alerts from BigQuery.")
        
        # Test updating alert status (if alerts were found)
        if alerts:
            test_alert_id = alerts[0].get("alarmId")
            logger.info(f"Testing status update for alarm ID: {test_alert_id}")
            bq_client.update_alert_status(alarm_ids=[test_alert_id], new_status="TEST")
        
        logger.info("Test script completed successfully.")

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
