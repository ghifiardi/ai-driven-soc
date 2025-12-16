#!/usr/bin/env python3
"""
BigQuery Direct Data Extraction
===============================

Extract comprehensive feedback data directly from BigQuery.
"""

from google.cloud import bigquery
import pandas as pd
import json
import os
from datetime import datetime, timedelta

def extract_from_bigquery():
    """Extract feedback data from BigQuery"""
    
    # Set up BigQuery client
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "Service Account BigQuery/sa-gatra-bigquery.json"
    client = bigquery.Client()
    
    # Common table names to try
    possible_tables = [
        "ai-driven-soc.feedback.alerts",
        "ai-driven-soc.soc.feedback", 
        "ai-driven-soc.alerts.feedback",
        "ai-driven-soc.main.feedback",
        "gatra-project.feedback.alerts",
        "gatra-project.soc.alerts"
    ]
    
    print("🔍 Searching for feedback tables in BigQuery...")
    
    # Try to list datasets first
    try:
        datasets = list(client.list_datasets())
        print(f"📊 Found {len(datasets)} datasets:")
        for dataset in datasets:
            print(f"  • {dataset.dataset_id}")
            
            # List tables in each dataset
            tables = list(client.list_tables(dataset.dataset_id))
            for table in tables:
                print(f"    - {table.table_id}")
                
                # Try to query tables that might contain feedback
                if any(keyword in table.table_id.lower() for keyword in ['feedback', 'alert', 'review']):
                    query = f"""
                    SELECT *
                    FROM `{dataset.dataset_id}.{table.table_id}`
                    LIMIT 1000
                    """
                    
                    try:
                        result = client.query(query).to_dataframe()
                        print(f"    ✅ Successfully queried {len(result)} rows from {table.table_id}")
                        
                        # Save to CSV
                        filename = f"bigquery_{dataset.dataset_id}_{table.table_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        result.to_csv(filename, index=False)
                        print(f"    💾 Saved to {filename}")
                        
                    except Exception as e:
                        print(f"    ❌ Error querying {table.table_id}: {e}")
                        
    except Exception as e:
        print(f"❌ Error accessing BigQuery: {e}")
        print("💡 Make sure the service account key is valid and has proper permissions")

if __name__ == "__main__":
    extract_from_bigquery()
