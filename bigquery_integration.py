import asyncio
import pprint
import os
import logging
import sys
import subprocess
import json
from langgraph_ada_integration import LangGraphAnomalyDetectionAgent, ADAConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# === CONFIGURATION ===
# Ensure this matches your GCP setup
PROJECT_ID = "chronicle-dev-2be9"
DATASET_NAME = "gatra_database"
TABLE_NAME = "siem_events"
BQ_LOCATION = "asia-southeast2"
BQ_LIMIT = 100  # Let's start with a smaller batch for the first test

# === QUERY SETUP ===
BQ_TABLE = f"{PROJECT_ID}.{DATASET_NAME}.{TABLE_NAME}"
BQ_QUERY = f"SELECT * FROM `{BQ_TABLE}` LIMIT {BQ_LIMIT}"

def fetch_bigquery_data(query, location, max_rows=1000, logger=logging.getLogger()):
    """
    Fetches data from BigQuery by executing a shell script ('fetch_data.sh')
    that calls the 'bq' command-line tool. This is a robust workaround for
    Python subprocess and SSL issues on the VM.
    """
    try:
        # Correct, hardcoded paths for the VM environment
        vm_base_dir = '/home/raditio.ghifiardigmail.com/ai-driven-soc'
        shell_script_path = os.path.join(vm_base_dir, 'fetch_data.sh')
        output_file = os.path.join(vm_base_dir, 'results.json')
        
        logger.info(f"Executing shell script: {shell_script_path}")

        # Execute the script, passing the query as an argument
        command = [shell_script_path, query]
        process = subprocess.run(command, capture_output=True, text=True, check=True, timeout=60)
        
        logger.info(f"Shell script executed successfully. Reading {output_file}...")
        
        # Read the data from the output file
        with open(output_file, 'r') as f:
            data = json.load(f)
            
        logger.info(f"Successfully loaded {len(data)} rows from {output_file}.")
        
        # Clean up the JSON file
        os.remove(output_file)
        logger.info(f"Cleaned up {output_file}.")
        
        return data
        
    except subprocess.TimeoutExpired as e:
        logger.critical("The fetch_data.sh script timed out after 60 seconds.")
        logger.critical(f"Stdout: {e.stdout}")
        logger.critical(f"Stderr: {e.stderr}")
        return None
    except subprocess.CalledProcessError as e:
        logger.critical(f"The fetch_data.sh script failed with exit code {e.returncode}.")
        logger.critical(f"Stderr from script: {e.stderr}")
        return None
    except FileNotFoundError:
        logger.critical(f"Could not find the output file '{output_file}'. The shell script might have failed silently.")
        return None
    except json.JSONDecodeError as e:
        logger.critical(f"Failed to parse JSON from '{output_file}'.", exc_info=True)
        return None
    except Exception as e:
        logger.critical(f"An unexpected error occurred in fetch_bigquery_data: {e}", exc_info=True)
        return None

def process_bigquery_rows(rows):
    """
    Processes raw BigQuery rows, unpacking nested events into a flat structure
    that the ADA agent can process.
    """
    processed = []
    for row in rows:  # Each row is an alarm with an 'events' list
        alarm_id = row.get("alarmId")
        events = row.get("events", [])
        if not isinstance(events, list):
            continue  # Skip if events is not a list

        for i, event in enumerate(events):
            if not isinstance(event, dict):
                continue  # Skip if an event is not a dictionary

            # Use a unique ID for each event within the alarm
            log_id = event.get("commonEventId") or f"{alarm_id}_{i}"

            alert_data = {
                "log_id": log_id,
                "timestamp": event.get("logDate"),
                "source_ip": event.get("originIP"),
                "dest_ip": event.get("impactedIP"),
                "protocol": event.get("protocolName"),
                "port": event.get("impactedPort"),
                "bytes_sent": event.get("bytesOut"),
                "bytes_received": event.get("bytesIn"),
                "duration": event.get("duration"),
                "raw_log": event,  # The entire nested event is the raw log
                "label": event.get("threatName") or event.get("classificationName", "Unknown"),
                "kill_chain_stage": event.get("kill_chain_stage", "None"),
            }
            # Filter out records with None values for critical fields
            if all(v is not None for v in [alert_data["source_ip"], alert_data["dest_ip"], alert_data["port"]]):
                 processed.append(alert_data)
            
    return processed

async def main():
    """Main function to fetch data and process it with the agent."""
    logger.info(f"--- Starting BigQuery Live Test ---")
    logger.info(f"Fetching up to {BQ_LIMIT} latest rows from: {BQ_TABLE}")
    
    try:
        bq_rows = fetch_bigquery_data(BQ_QUERY, BQ_LOCATION, max_rows=BQ_LIMIT, logger=logger)
        logger.info(f"Successfully fetched {len(bq_rows)} rows from BigQuery.")
    except Exception as e:
        logger.critical(f"FATAL: Could not fetch data from BigQuery. Error: {e}", exc_info=True)
        return

    if not bq_rows:
        logger.warning("No data returned. Please check your dataset, table name, and permissions.")
        return

    alert_data_list = process_bigquery_rows(bq_rows)
    if not alert_data_list:
        logger.warning("No processable events were found in the fetched BigQuery rows.")
        return
        
    logger.info(f"Extracted {len(alert_data_list)} processable events to be sent to the agent.")

    # Ensure we are in live mode, not local testing mode
    config = ADAConfig(local_mode=False) 
    agent = LangGraphAnomalyDetectionAgent(config)

    logger.info("Processing alerts with ADA agent concurrently...")
    tasks = [agent.process_alert(alert_data) for alert_data in alert_data_list]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    logger.info(f"\n--- Processing Complete ---")
    for i, result in enumerate(results):
        logger.info(f"\n--- Result for Alert {i+1} ---")
        if isinstance(result, Exception):
            logger.error(f"An error occurred: {result}", exc_info=True)
        else:
            pprint.pprint(result)
        logger.info("-" * 25)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.critical("An uncaught exception occurred in the main execution block.", exc_info=True)
        sys.exit(1)
