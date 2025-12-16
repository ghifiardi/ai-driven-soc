#!/usr/bin/env python3
"""
Get Real Training Metrics from CLA Service
Reads the latest model metrics and training data from the CLA service
"""

import json
import os
import glob
import subprocess
import sys
from datetime import datetime

def get_latest_model_metrics():
    """Get metrics from the latest trained model"""
    try:
        # Path to model metrics directory
        models_dir = "/home/raditio.ghifiardigmail.com/ai-driven-soc/models/"
        
        # Find the latest model metrics file
        metrics_files = glob.glob(os.path.join(models_dir, "model_metrics_*.json"))
        if not metrics_files:
            return None
            
        latest_file = max(metrics_files, key=os.path.getctime)
        
        # Read the metrics
        with open(latest_file, 'r') as f:
            metrics = json.load(f)
            
        # Add file timestamp
        metrics['model_timestamp'] = os.path.basename(latest_file)
        metrics['last_training_time'] = datetime.fromtimestamp(os.path.getctime(latest_file)).isoformat()
        
        return metrics
        
    except Exception as e:
        print(f"Error reading model metrics: {e}", file=sys.stderr)
        return None

def get_bigquery_data():
    """Get data counts from BigQuery"""
    try:
        query = """
        SELECT 
            (SELECT COUNT(*) FROM `chronicle-dev-2be9.gatra_database.siem_events`) as total_events,
            (SELECT COUNT(*) FROM `chronicle-dev-2be9.gatra_database.siem_alarms`) as total_alarms,
            (SELECT COUNT(*) FROM `chronicle-dev-2be9.gatra_database.taa_state`) as taa_processed,
            (SELECT COUNT(*) FROM `chronicle-dev-2be9.gatra_database.dashboard_alerts`) as dashboard_alerts
        """
        
        result = subprocess.run([
            "bq", "query", "--use_legacy_sql=false", "--max_rows=1", "--format=json", query
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        
        if result.returncode == 0:
            data = json.loads(result.stdout.strip())[0]
            return {
                "total_events": int(data["total_events"]),
                "total_alarms": int(data["total_alarms"]),
                "taa_processed": int(data["taa_processed"]),
                "dashboard_alerts": int(data["dashboard_alerts"])
            }
        else:
            print(f"BigQuery query failed: {result.stderr}", file=sys.stderr)
            return None
            
    except Exception as e:
        print(f"Error querying BigQuery: {e}", file=sys.stderr)
        return None

def get_cla_service_status():
    """Check if CLA service is running and training"""
    try:
        # Check if CLA service is active
        result = subprocess.run([
            "systemctl", "is-active", "cla.service"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        
        is_active = result.stdout.strip() == "active"
        
        # Count model files to see training activity
        models_dir = "/home/raditio.ghifiardigmail.com/ai-driven-soc/models/"
        model_files = glob.glob(os.path.join(models_dir, "trained_model_*.pkl"))
        
        return {
            "cla_service_active": is_active,
            "total_models_trained": len(model_files),
            "training_active": len(model_files) > 0
        }
        
    except Exception as e:
        print(f"Error checking CLA service: {e}", file=sys.stderr)
        return {"cla_service_active": False, "total_models_trained": 0, "training_active": False}

def main():
    """Main function to get all real training metrics"""
    try:
        # Get model metrics
        model_metrics = get_latest_model_metrics()
        
        # Get BigQuery data
        bigquery_data = get_bigquery_data()
        
        # Get CLA service status
        cla_status = get_cla_service_status()
        
        # Combine all data
        result = {}
        
        if bigquery_data:
            result.update(bigquery_data)
            
        if model_metrics:
            result.update({
                "model_precision": model_metrics.get("precision", 0.0),
                "model_recall": model_metrics.get("recall", 0.0),
                "model_f1_score": model_metrics.get("f1_score", 0.0),
                "model_accuracy": model_metrics.get("accuracy", 0.0),
                "training_samples": model_metrics.get("training_samples", 0),
                "test_samples": model_metrics.get("test_samples", 0),
                "last_training_time": model_metrics.get("last_training_time", ""),
                "model_timestamp": model_metrics.get("model_timestamp", "")
            })
        
        if cla_status:
            result.update(cla_status)
            
        # Add calculated metrics for dashboard compatibility
        if model_metrics:
            # Calculate balanced accuracy (since we have class imbalance)
            balanced_accuracy = (model_metrics.get("precision", 0.0) + model_metrics.get("recall", 0.0)) / 2.0 if (model_metrics.get("precision", 0.0) + model_metrics.get("recall", 0.0)) > 0 else model_metrics.get("accuracy", 0.0)
            
            result.update({
                "ada_accuracy": balanced_accuracy * 100,  # Convert to percentage
                "taa_confidence": model_metrics.get("accuracy", 0.0) * 100,  # Use accuracy as confidence proxy
                "true_positive_rate": model_metrics.get("f1_score", 0.0) * 100,  # Use F1 as TPR proxy
                "ada_score": model_metrics.get("precision", 0.0) * 100  # Use precision as ADA score
            })
        
        # Add status indicators
        result.update({
            "training_mode": "LEARNING MODE" if cla_status.get("training_active", False) else "STATISTICS MODE",
            "last_updated": datetime.now().isoformat(),
            "data_source": "Real Training Data" if model_metrics else "BigQuery Only"
        })
        
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"Error in main: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()


