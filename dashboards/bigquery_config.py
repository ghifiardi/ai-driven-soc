"""
BigQuery Configuration for SOC Dashboards
Connects to GCP BigQuery and provides data access functions
"""

import os
from typing import Dict, Optional, List
import pandas as pd

# GCP Configuration
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "chronicle-dev-2be9")
BIGQUERY_DATASET = os.getenv("BIGQUERY_DATASET", "gatra_database")

# BigQuery Table Mappings
BIGQUERY_TABLES = {
    "siem_events": f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.siem_events",
    "activity_logs": f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.activity_logs",
    "ada_agent_metri": f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.ada_agent_metri",
    "ada_features": f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.ada_features",
    "ada_feedback": f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.ada_feedback",
    "ada_ml_results": f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.ada_ml_results",
    "ada_state": f"{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.ada_state",
    # Threat hunting tables
    "thor_scan_results": f"{GCP_PROJECT_ID}.soc_data.thor_scan_results",
    "threat_intel": f"{GCP_PROJECT_ID}.soc_data.threat_intel",
}

# Service Account Key Path (optional, can use Application Default Credentials)
SERVICE_ACCOUNT_KEY = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", None)

def get_bigquery_client():
    """Get BigQuery client with proper authentication"""
    try:
        from google.cloud import bigquery
        
        if SERVICE_ACCOUNT_KEY and os.path.exists(SERVICE_ACCOUNT_KEY):
            client = bigquery.Client.from_service_account_json(
                SERVICE_ACCOUNT_KEY,
                project=GCP_PROJECT_ID
            )
        else:
            # Use Application Default Credentials (ADC)
            client = bigquery.Client(project=GCP_PROJECT_ID)
        
        return client
    except Exception as e:
        raise Exception(f"Failed to initialize BigQuery client: {e}")

def test_bigquery_connection() -> Dict:
    """Test BigQuery connection and return status"""
    try:
        client = get_bigquery_client()
        
        # Test query
        test_query = f"""
        SELECT COUNT(*) as table_count
        FROM `{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.__TABLES__`
        """
        
        result = client.query(test_query).result()
        for row in result:
            return {
                'success': True,
                'project_id': GCP_PROJECT_ID,
                'dataset': BIGQUERY_DATASET,
                'tables_found': row.table_count,
                'message': 'BigQuery connection successful'
            }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'message': f'BigQuery connection failed: {e}'
        }

def get_available_tables() -> List[str]:
    """Get list of available tables in the dataset"""
    try:
        client = get_bigquery_client()
        
        query = f"""
        SELECT table_name
        FROM `{GCP_PROJECT_ID}.{BIGQUERY_DATASET}.__TABLES__`
        ORDER BY table_name
        """
        
        result = client.query(query).to_dataframe()
        return result['table_name'].tolist()
    except Exception as e:
        return []

def query_siem_events(limit: int = 100, hours: int = 24) -> pd.DataFrame:
    """Query SIEM events from BigQuery"""
    try:
        client = get_bigquery_client()
        
        query = f"""
        SELECT *
        FROM `{BIGQUERY_TABLES['siem_events']}`
        WHERE TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), timestamp, HOUR) <= {hours}
        ORDER BY timestamp DESC
        LIMIT {limit}
        """
        
        return client.query(query).to_dataframe()
    except Exception as e:
        print(f"Error querying SIEM events: {e}")
        return pd.DataFrame()

def query_ada_metrics(limit: int = 100) -> pd.DataFrame:
    """Query ADA agent metrics from BigQuery"""
    try:
        client = get_bigquery_client()
        
        query = f"""
        SELECT *
        FROM `{BIGQUERY_TABLES['ada_agent_metri']}`
        ORDER BY timestamp DESC
        LIMIT {limit}
        """
        
        return client.query(query).to_dataframe()
    except Exception as e:
        print(f"Error querying ADA metrics: {e}")
        return pd.DataFrame()

def query_threat_hunting_results(limit: int = 100) -> pd.DataFrame:
    """Query threat hunting scan results from BigQuery"""
    try:
        client = get_bigquery_client()
        
        query = f"""
        SELECT *
        FROM `{BIGQUERY_TABLES['thor_scan_results']}`
        ORDER BY scan_timestamp DESC
        LIMIT {limit}
        """
        
        return client.query(query).to_dataframe()
    except Exception as e:
        print(f"Error querying threat hunting results: {e}")
        return pd.DataFrame()

def get_event_statistics(hours: int = 24) -> Dict:
    """Get event statistics from BigQuery"""
    try:
        client = get_bigquery_client()
        
        query = f"""
        SELECT 
            COUNT(*) as total_events,
            COUNT(DISTINCT alarmId) as unique_alarms,
            COUNT(DISTINCT source_ip) as unique_sources,
            COUNT(DISTINCT destination_ip) as unique_destinations
        FROM `{BIGQUERY_TABLES['siem_events']}`
        WHERE TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), timestamp, HOUR) <= {hours}
        """
        
        result = client.query(query).result()
        for row in result:
            return {
                'total_events': row.total_events or 0,
                'unique_alarms': row.unique_alarms or 0,
                'unique_sources': row.unique_sources or 0,
                'unique_destinations': row.unique_destinations or 0
            }
    except Exception as e:
        return {
            'total_events': 0,
            'unique_alarms': 0,
            'unique_sources': 0,
            'unique_destinations': 0,
            'error': str(e)
        }

