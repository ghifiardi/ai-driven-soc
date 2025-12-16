#!/usr/bin/env python3
"""
Startup script for ADA BigQuery Dashboard
"""

import os
import sys
from ada_bigquery_dashboard import app

if __name__ == '__main__':
    print("=" * 60)
    print("ADA BigQuery Dashboard - Starting Up")
    print("=" * 60)
    
    # Check environment variables
    project_id = os.getenv("BIGQUERY_PROJECT_ID", "chronicle-dev-2be9")
    dataset_id = os.getenv("BIGQUERY_DATASET_ID", "soc_data")
    table_id = os.getenv("BIGQUERY_PROCESSED_ALERTS_TABLE_ID", "processed_alerts")
    
    print(f"BigQuery Configuration:")
    print(f"  Project ID: {project_id}")
    print(f"  Dataset ID: {dataset_id}")
    print(f"  Table ID: {table_id}")
    print(f"  Full Table: {project_id}.{dataset_id}.{table_id}")
    print()
    
    print("Dashboard Features:")
    print("  ✓ Filters out NULL/invalid alert records")
    print("  ✓ Real-time metrics from BigQuery")
    print("  ✓ Service status monitoring")
    print("  ✓ Recent alerts display")
    print("  ✓ Auto-refresh every 30 seconds")
    print()
    
    print("Starting dashboard on http://localhost:3002")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    try:
        app.run(host='0.0.0.0', port=3002, debug=False)
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
    except Exception as e:
        print(f"Error starting dashboard: {e}")
        sys.exit(1)
