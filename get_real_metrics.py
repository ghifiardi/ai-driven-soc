#!/usr/bin/env python3
"""
Real Metrics Fetcher for AI Model Training Dashboard
Fetches actual metrics from BigQuery instead of using hardcoded values
"""

import subprocess
import json
import sys

def get_real_metrics():
    """Fetch real metrics from BigQuery"""
    try:
        # Query to get real data counts
        query = """
        SELECT 
            (SELECT COUNT(*) FROM `chronicle-dev-2be9.gatra_database.siem_events`) as total_events,
            (SELECT COUNT(*) FROM `chronicle-dev-2be9.gatra_database.siem_alarms`) as total_alarms,
            (SELECT COUNT(*) FROM `chronicle-dev-2be9.gatra_database.taa_state`) as taa_processed,
            (SELECT COUNT(*) FROM `chronicle-dev-2be9.gatra_database.dashboard_alerts`) as dashboard_alerts,
            (SELECT ROUND(COUNT(CASE WHEN valid = true THEN 1 END) * 100.0 / COUNT(*), 1) FROM `chronicle-dev-2be9.gatra_database.taa_state` WHERE confidence IS NOT NULL) as real_accuracy,
            (SELECT ROUND(AVG(confidence), 1) FROM `chronicle-dev-2be9.gatra_database.taa_state` WHERE confidence IS NOT NULL) as avg_confidence,
            (SELECT ROUND(AVG(severity), 3) FROM `chronicle-dev-2be9.gatra_database.taa_state` WHERE severity IS NOT NULL) as avg_severity,
            (SELECT ROUND(AVG(ada_score), 3) FROM `chronicle-dev-2be9.gatra_database.dashboard_alerts` WHERE ada_score IS NOT NULL) as avg_ada_score
        """
        
        # Execute BigQuery command
        result = subprocess.run([
            "bq", "query", "--use_legacy_sql=false", "--max_rows=1", "--format=json", query
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            data = json.loads(result.stdout.strip())[0]
            
            # Return formatted metrics
            metrics = {
                "total_alerts": data["total_events"],
                "unique_alarms": data["total_alarms"], 
                "taa_processed": data["taa_processed"],
                "dashboard_alerts": data["dashboard_alerts"],
                "ada_accuracy": data["real_accuracy"],
                "taa_confidence": data["avg_confidence"],
                "true_positive_rate": data["real_accuracy"],  # Same as accuracy for TAA
                "ada_score": data["avg_ada_score"],
                "taa_severity": data["avg_severity"]
            }
            
            return metrics
        else:
            print(f"Error: {result.stderr}", file=sys.stderr)
            return None
            
    except Exception as e:
        print(f"Error fetching metrics: {str(e)}", file=sys.stderr)
        return None

if __name__ == "__main__":
    metrics = get_real_metrics()
    if metrics:
        print(json.dumps(metrics, indent=2))
    else:
        sys.exit(1)


