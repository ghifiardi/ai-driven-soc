from google.cloud import bigquery
from langgraph_ada_integration import LangGraphAnomalyDetectionAgent, ADAConfig
import pprint
import os

# === CONFIGURATION ===
PROJECT_ID = "chronicle-dev-2be9"
DATASET_NAME = "gatra_database"
TABLE_NAME = "siem_events"
BQ_LOCATION = "asia-southeast2"
BQ_LIMIT = 1000  # Adjust as needed

# === QUERY SETUP ===
BQ_TABLE = f"{PROJECT_ID}.{DATASET_NAME}.{TABLE_NAME}"
BQ_QUERY = f"SELECT * FROM `{BQ_TABLE}` LIMIT {BQ_LIMIT}"

def fetch_bigquery_data(query, location, max_rows=1000):
    client = bigquery.Client(location=location)
    query_job = client.query(query)
    results = query_job.result()
    data = [dict(row) for i, row in enumerate(results) if i < max_rows]
    return data

def process_bigquery_rows(rows):
    processed = []
    for row in rows:
        alert_data = {
            "log_id": row.get("log_id"),
            "timestamp": row.get("timestamp"),
            "source_ip": row.get("source_ip"),
            "dest_ip": row.get("dest_ip"),
            "protocol": row.get("protocol"),
            "port": row.get("port"),
            "bytes_sent": row.get("bytes_sent"),
            "bytes_received": row.get("bytes_received"),
            "duration": row.get("duration"),
            "raw_log": row,
            "label": row.get("label", "Unknown"),
            "kill_chain_stage": row.get("kill_chain_stage", "None"),
            # Add more fields as needed for your ADA agent
        }
        processed.append(alert_data)
    return processed

if __name__ == "__main__":
    print(f"Fetching data from BigQuery table: {BQ_TABLE} in location: {BQ_LOCATION} ...")
    bq_rows = fetch_bigquery_data(BQ_QUERY, BQ_LOCATION, max_rows=BQ_LIMIT)
    print(f"Fetched {len(bq_rows)} rows from BigQuery.")
    if not bq_rows:
        print("No data returned. Double-check your dataset, table, and query.")
        exit(1)

    alert_data_list = process_bigquery_rows(bq_rows)

    config = ADAConfig()
    agent = LangGraphAnomalyDetectionAgent(config)

    print("Processing rows with ADA agent...")
    for i, alert_data in enumerate(alert_data_list):
        import asyncio
        result = asyncio.run(agent.process_alert(alert_data))
        print(f"Result for row {i}:")
        pprint.pprint(result)
        # Optionally: break after N rows for testing
        # if i > 10: break
