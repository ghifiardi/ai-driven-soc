# firestore_to_bq.py
import json
import datetime
import subprocess
from google.cloud import firestore

# Connect to Firestore
db = firestore.Client()

def export_ground_truth():
    """Export ground_truth_alerts from Firestore to BigQuery"""
    collection = db.collection('ground_truth_alerts')
    docs = collection.stream()
    
    # Process in batches of 100
    batch_size = 100
    batch = []
    
    print("Exporting ground truth alerts...")
    count = 0
    
    for doc in docs:
        data = doc.to_dict()
        # Flatten nested data structure
        row = {
            "alert_id": doc.id,
            "analyst_id": data.get("analyst_id", ""),
            "label": data.get("label", ""),
            "timestamp": data.get("timestamp", datetime.datetime.now()).isoformat(),
            "raw_alert_data_method_name": data.get("raw_alert_data", {}).get("method_name", ""),
            "raw_alert_data_service_name": data.get("raw_alert_data", {}).get("service_name", ""),
            "raw_alert_data_anomaly_score": float(data.get("raw_alert_data", {}).get("anomaly_score", 0)),
            "bytes_sent": int(data.get("bytes_sent", 0)),
            "bytes_received": int(data.get("bytes_received", 0)),
            "duration": float(data.get("duration", 0))
        }
        batch.append(row)
        count += 1
        
        # Process batch
        if len(batch) >= batch_size:
            _insert_batch_to_bq('ground_truth_alerts', batch)
            batch = []
            print(f"Processed {count} alerts")
    
    # Process remaining items
    if batch:
        _insert_batch_to_bq('ground_truth_alerts', batch)
    
    print(f"Completed export of {count} ground truth alerts")

def export_model_performance():
    """Export model_performance from Firestore to BigQuery"""
    collection = db.collection('model_performance')
    docs = collection.stream()
    
    batch = []
    count = 0
    
    for doc in docs:
        data = doc.to_dict()
        row = {
            "timestamp": data.get("timestamp", datetime.datetime.now()).isoformat(),
            "precision": float(data.get("precision", 0)),
            "recall": float(data.get("recall", 0)),
            "f1_score": float(data.get("f1_score", 0)),
            "true_positives": int(data.get("true_positives", 0)),
            "false_positives": int(data.get("false_positives", 0))
        }
        batch.append(row)
        count += 1
        
    if batch:
        _insert_batch_to_bq('model_performance', batch)
    
    print(f"Completed export of {count} model performance records")

def export_model_metadata():
    """Export model_metadata from Firestore to BigQuery"""
    collection = db.collection('model_metadata')
    docs = collection.stream()
    
    batch = []
    count = 0
    
    for doc in docs:
        data = doc.to_dict()
        row = {
            "model_id": doc.id,
            "training_timestamp": data.get("training_timestamp", datetime.datetime.now()).isoformat(),
            "training_samples": int(data.get("training_samples", 0)),
            "gcs_path": data.get("gcs_path", ""),
            "precision": float(data.get("metrics", {}).get("precision", 0)),
            "recall": float(data.get("metrics", {}).get("recall", 0)),
            "f1_score": float(data.get("metrics", {}).get("f1_score", 0))
        }
        batch.append(row)
        count += 1
        
    if batch:
        _insert_batch_to_bq('model_metadata', batch)
    
    print(f"Completed export of {count} model metadata records")

def _insert_batch_to_bq(table_name, rows):
    """Insert a batch of rows into BigQuery using bq command"""
    # Write rows to temporary file
    with open(f'/tmp/{table_name}_batch.json', 'w') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')
    
    # Load into BigQuery using bq command
    cmd = [
        '/usr/bin/bq', 'load', 
        '--source_format=NEWLINE_DELIMITED_JSON',
        f'chronicle-dev-2be9:gatra_database.{table_name}', 
        f'/tmp/{table_name}_batch.json'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error loading to BigQuery: {result.stderr}")
    else:
        print(f"Successfully loaded batch to {table_name}")

if __name__ == "__main__":
    export_ground_truth()
    export_model_performance()
    export_model_metadata()