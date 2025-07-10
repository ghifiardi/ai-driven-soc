import logging
from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPIError

logger = logging.getLogger(__name__)

class BigQueryClient:
    """Client to interact with Google BigQuery for fetching and updating alerts."""

    def __init__(self, project_id: str, dataset_id: str, table_id: str):
        """Initializes the BigQuery client."""
        if not all([project_id, dataset_id, table_id]):
            raise ValueError("BigQuery project, dataset, and table ID must be provided.")
        self.client = bigquery.Client(project=project_id, location="asia-southeast2")
        self.table_fqn = f"{project_id}.{dataset_id}.{table_id}"
        logger.info(f"BigQuery client initialized for table: {self.table_fqn}")

    def fetch_new_alerts(self, limit: int = 10) -> list[dict]:
        """Fetches a batch of alerts from BigQuery."""
        logger.info(f"Fetching up to {limit} new alerts from {self.table_fqn}")
        # Fetch alerts that are new or have not been processed yet
        query = f"""
            SELECT alarmId, events
            FROM `{self.table_fqn}`
            WHERE status IS NULL OR status = 'NEW'
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
        """Updates the status of alerts in BigQuery."""
        if not alarm_ids:
            return

        logger.info(f"Updating status to '{new_status}' for {len(alarm_ids)} alerts.")

        # Format alarm_ids for the IN clause
        formatted_ids = ", ".join([f"'{_id}'" for _id in alarm_ids])

        query = f"""
            UPDATE `{self.table_fqn}`
            SET status = '{new_status}'
            WHERE alarmId IN ({formatted_ids})
        """

        try:
            query_job = self.client.query(query)
            query_job.result()  # Wait for the job to complete
            logger.info(f"Successfully updated status for {len(alarm_ids)} alerts to '{new_status}'.")
        except Exception as e:
            logger.error(f"Error updating BigQuery table: {e}", exc_info=True)

    def insert_rows_json(self, table_id: str, rows: list[dict]):
        """Inserts rows into the specified BigQuery table from a list of dictionaries."""
        if not rows:
            logger.info("No rows to insert.")
            return

        try:
            table_ref = self.client.dataset(self.dataset_id).table(table_id)
            errors = self.client.insert_rows_json(table_ref, rows)
            if not errors:
                logger.info(f"Successfully inserted {len(rows)} rows into {table_id}.")
            else:
                logger.error(f"Encountered errors while inserting rows into {table_id}: {errors}")
        except GoogleAPIError as e:
            logger.error(f"BigQuery error while inserting rows into {table_id}: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while inserting rows into {table_id}: {e}", exc_info=True)
            raise
