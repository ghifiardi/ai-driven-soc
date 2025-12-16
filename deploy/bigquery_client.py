import logging
from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPIError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("bigquery_client")

class BigQueryClient:
    """Client for interacting with Google BigQuery."""
    
    def __init__(self, project_id: str, dataset_id: str, table_id: str, results_table_id: str):
        """Initialize BigQuery client and set table reference."""
        if not all([project_id, dataset_id, table_id, results_table_id]):
            raise ValueError("BigQuery project, dataset, table ID, and results table ID must be provided.")
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.results_table_id = results_table_id
        self.client = bigquery.Client(project=project_id, location="asia-southeast2")
        self.table_fqn = f"{project_id}.{dataset_id}.{table_id}"
        self.results_table_fqn = f"{project_id}.{dataset_id}.{results_table_id}"
        logger.info(f"BigQuery client initialized for source table: {self.table_fqn}")
        logger.info(f"BigQuery client initialized for results table: {self.results_table_fqn}")

    def fetch_new_alerts(self, limit: int = 10) -> list[dict]:
        """Fetches a batch of alerts from BigQuery."""
        logger.info(f"Fetching up to {limit} new alerts from {self.table_fqn}")
        # Fetch alerts that are new or have not been processed yet
        query = f"""
            SELECT
                alarmId, events
            FROM
                `{self.table_fqn}`
            LIMIT {limit}
        """
        try:
            query_job = self.client.query(query)
            results = query_job.result()
            alerts = [dict(row) for row in results]
            logger.info(f"Successfully fetched {len(alerts)} alerts.")
            return alerts
        except GoogleAPIError as e:
            logger.error(f"A BigQuery API error occurred while fetching new alerts: {e}", exc_info=False)
        except Exception as e:
            logger.error(f"An unexpected error occurred while fetching alerts: {e}", exc_info=True)
        return []

    def update_alert_status(self, alarm_ids: list[str], new_status: str):
        """Simulates updating the status of alerts in BigQuery.
        
        This is a stub method because the table doesn't have a 'status' column yet.
        Just logs the intended action without executing any SQL.
        """
        if not alarm_ids:
            return

        logger.info(f"Updating status to '{new_status}' for {len(alarm_ids)} alerts.")
        logger.info(f"NOTE: Status updates are currently logged only. No database update is performed.")
        
        # If we actually had a status column, we would run a query like this:
        # formatted_ids = ", ".join([f"'{id}'" for id in alarm_ids])
        # update_query = f"""
        #     UPDATE `{self.table_fqn}`
        #     SET status = '{new_status}'
        #     WHERE alarmId IN ({formatted_ids})
        # """
        # query_job = self.client.query(update_query)
        # query_job.result()  # Wait for the job to complete

    def insert_rows_json(self, rows_to_insert: list[dict]):
        """Inserts rows into the results BigQuery table from a list of dictionaries."""
        if not rows_to_insert:
            logger.info("No rows to insert.")
            return

        try:
            errors = self.client.insert_rows_json(self.results_table_fqn, rows_to_insert)
            if not errors:
                logger.info(f"Successfully inserted {len(rows_to_insert)} rows into {self.results_table_fqn}")
            else:
                logger.error(f"Encountered errors while inserting rows into {self.results_table_fqn}: {errors}")
        except GoogleAPIError as e:
            logger.error(f"BigQuery error while inserting rows into {self.results_table_fqn}: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while inserting rows into {self.results_table_fqn}: {e}", exc_info=True)
            raise
