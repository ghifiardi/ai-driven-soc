#!/usr/bin/env python3
"""
Script to fix remaining Firestore references in the Continuous Learning Agent.
This script will patch the CLA code to remove Firestore references and update to BigQuery.
"""

import sys
import re

def patch_cla_code(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # This is the new, complete main execution block with polling logic.
    # It will replace the entire original main block.
    main_replacement = '''if __name__ == "__main__":
    import argparse
    import os
    import time
    from datetime import datetime, timezone
    import logging

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("ContinuousLearningAgent")

    parser = argparse.ArgumentParser(description='Continuous Learning Agent')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    agent = ContinuousLearningAgent(config_path=args.config)

    logger.info("Starting agent with BigQuery polling for feedback.")

    # Main loop for polling and scheduled tasks
    while True:
        try:
            # Run scheduled tasks like model evaluation first
            agent.run_scheduled_tasks()

            # Now, poll for new feedback
            logger.info("Polling for new feedback...")
            feedback_table_id = f"{agent.config['project_id']}.{agent.config['bigquery_dataset']}.{agent.config['bigquery_feedback_table']}"
            query = f"""
                SELECT * FROM `{feedback_table_id}`
                WHERE processed = FALSE OR processed IS NULL
                LIMIT 100
            """
            query_job = agent.bq_client.query(query)
            feedback_rows = list(query_job.result())

            if feedback_rows:
                logger.info(f"Found {len(feedback_rows)} new feedback entries to process.")
                for row in feedback_rows:
                    feedback_data = dict(row)
                    agent.process_feedback(feedback_data)
                    # In the original feedback logic, the alert is updated. Here we mark it processed.
                    update_query = f"""
                        UPDATE `{feedback_table_id}`
                        SET processed = TRUE
                        WHERE alert_id = '{row['alert_id']}' AND timestamp = TIMESTAMP('{row['timestamp'].isoformat()}')
                    """
                    agent.bq_client.query(update_query).result()
                    logger.info(f"Marked feedback for alert {row['alert_id']} as processed.")
            else:
                logger.info("No new feedback found.")

        except Exception as e:
            logger.error(f"An error occurred in the main loop: {e}", exc_info=True)
        
        logger.info("Cycle complete. Waiting for 60 seconds...")
        time.sleep(60)
    '''

    # Find and replace the entire main block in the agent's code
    # This regex finds the 'if __name__ == "__main__":' block and replaces it entirely.
    content = re.sub(r'if __name__ == "__main__":.*', main_replacement, content, flags=re.DOTALL)

    # Also, ensure all Pub/Sub related code is removed to prevent conflicts.
    content = re.sub(r'from google.cloud import pubsub_v1', '# Pub/Sub disabled for polling prototype', content)
    content = re.sub(r'self\.publisher =.*', 'self.pubsub_available = False', content, flags=re.DOTALL)
    content = re.sub(r'def start_listeners.*?(?=\n    def|$)', '# Listeners disabled for polling prototype', content, flags=re.DOTALL)
    content = re.sub(r'def _feedback_listener_thread.*?(?=\n    def|$)', '', content, flags=re.DOTALL)
    content = re.sub(r'def _process_feedback_message.*?(?=\n    def|$)', '', content, flags=re.DOTALL)
    content = re.sub(r'def _metrics_listener_thread.*?(?=\n    def|$)', '', content, flags=re.DOTALL)
    content = re.sub(r'def _process_metrics_message.*?(?=\n    def|$)', '', content, flags=re.DOTALL)

    with open(file_path, 'w') as file:
        file.write(content)
    
    print(f"Successfully patched {file_path} with BigQuery polling logic.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fix_firestore_refs.py <path_to_cla_py>")
        sys.exit(1)
    
    patch_cla_code(sys.argv[1])
