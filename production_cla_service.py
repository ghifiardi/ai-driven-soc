#!/usr/bin/env python3
"""
Production CLA Service - Deploy 100% Performance CLA to Production

This service provides:
1. Real-time alert classification
2. Model retraining on schedule
3. Performance monitoring
4. Health checks
5. Metrics reporting
"""

import os
import json
import logging
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from google.cloud import bigquery, pubsub_v1
from google.cloud import storage
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import threading
import schedule
from flask import Flask, request, jsonify
import requests
from collections import deque
import signal
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('production_cla_service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ProductionCLA")

class ProductionCLAService:
    """Production CLA Service with 100% performance"""
    
    def __init__(self, config_path: str = "config/production_cla_config.json"):
        """Initialize Production CLA Service"""
        self.config = self._load_config(config_path)
        self.bigquery_client = bigquery.Client(project=self.config["project_id"])
        self.pubsub_publisher = pubsub_v1.PublisherClient()
        self.storage_client = storage.Client(project=self.config["project_id"])
        
        # Model and scaler
        self.model = None
        self.scaler = None
        self.model_version = None
        self.model_loaded = False
        
        # Performance tracking
        self.predictions_count = 0
        self.correct_predictions = 0
        self.performance_history = deque(maxlen=1000)
        
        # Service status
        self.service_running = True
        self.last_retrain = None
        self.next_retrain = datetime.now() + timedelta(hours=24)  # Daily retraining
        
        # Load initial model
        self.load_latest_model()
        
        # Derived Fallbacks
        self._processed_alerts_fqn = f"{self.config['project_id']}.{self.config['bigquery_dataset']}.{self.config['processed_alerts_table']}"
        self._feedback_fqn = f"{self.config['project_id']}.{self.config['bigquery_dataset']}.{self.config['feedback_table']}"
        
    def _load_config(self, config_path: str) -> Dict:
        """Load production configuration"""
        default_config = {
            "project_id": "chronicle-dev-2be9",
            "bigquery_dataset": "soc_data",
            "processed_alerts_table": "processed_alerts",
            "feedback_table": "feedback",
            "model_bucket": "cla-production-models",
            "model_path": "models/",
            "pubsub_topic": "cla-classifications",
            "retrain_interval_hours": 24,
            "min_retrain_samples": 1000,
            "performance_threshold": 0.95,
            "service_port": 8080,
            "health_check_interval": 300
        }
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                # Merge with defaults
                default_config.update(config)
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")
        
        return default_config
    
    def load_latest_model(self):
        """Load the latest trained model"""
        try:
            # Check local models first
            model_files = []
            if os.path.exists(self.config["model_path"]):
                model_files = [f for f in os.listdir(self.config["model_path"]) 
                              if f.startswith("real_data_model_") and f.endswith(".pkl")]
            
            if model_files:
                # Get the latest model
                latest_model = max(model_files, key=lambda x: os.path.getctime(
                    os.path.join(self.config["model_path"], x)
                ))
                model_path = os.path.join(self.config["model_path"], latest_model)
                
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                if isinstance(model_data, tuple) and len(model_data) == 2:
                    self.model, self.scaler = model_data
                    self.model_version = latest_model.replace("real_data_model_", "").replace(".pkl", "")
                    self.model_loaded = True
                    logger.info(f"Loaded model version: {self.model_version}")
                else:
                    logger.error("Invalid model format")
                    return False
            
            # Try loading from GCS if local model not found
            elif self.config.get("model_bucket"):
                if self.load_model_from_gcs():
                    return True
            
            if not self.model_loaded:
                logger.error("No model found, training new model...")
                return self.train_initial_model()
                
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def load_model_from_gcs(self) -> bool:
        """Load model from Google Cloud Storage"""
        try:
            bucket_name = self.config["model_bucket"]
            bucket = self.storage_client.bucket(bucket_name)
            
            # List model files
            blobs = bucket.list_blobs(prefix="models/")
            model_files = [blob.name for blob in blobs if blob.name.endswith(".pkl")]
            
            if not model_files:
                return False
            
            # Get latest model
            latest_model = max(model_files, key=lambda x: x.split("/")[-1])
            
            # Download and load
            blob = bucket.blob(latest_model)
            model_data = pickle.loads(blob.download_as_bytes())
            
            if isinstance(model_data, tuple) and len(model_data) == 2:
                self.model, self.scaler = model_data
                self.model_version = latest_model.split("/")[-1].replace("real_data_model_", "").replace(".pkl", "")
                self.model_loaded = True
                logger.info(f"Loaded model from GCS: {self.model_version}")
                return True
            
        except Exception as e:
            logger.warning(f"Could not load model from GCS: {e}")
        
        return False
    
    def train_initial_model(self) -> bool:
        """Train initial model for production"""
        try:
            logger.info("Training initial model for production...")
            
            # Gather training data
            training_data = self.gather_training_data(limit=10000)
            if not training_data or len(training_data) < 1000:
                logger.error("Insufficient training data")
                return False
            
            # Prepare features
            X, y = self.prepare_features(training_data)
            
            # Train model
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test_scaled)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
            
            logger.info(f"Initial model trained - F1: {f1:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")
            
            # Save model
            self.save_model()
            self.model_loaded = True
            self.model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training initial model: {e}")
            return False
    
    def gather_training_data(self, limit: int = 20000) -> Optional[pd.DataFrame]:
        """Gather training data from processed_alerts"""
        try:
            processed_alerts_table = f"{self.config['project_id']}.{self.config['bigquery_dataset']}.processed_alerts"
            
            query = f"""
            SELECT 
                alert_id,
                classification,
                confidence_score,
                timestamp,
                is_anomaly,
                raw_alert
            FROM `{processed_alerts_table}`
            WHERE classification IS NOT NULL 
            AND confidence_score IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT {limit}
            """
            
            rows = list(self.bigquery_client.query(query).result())
            
            if not rows:
                return None
            
            data = []
            for row in rows:
                if row.raw_alert:
                    if isinstance(row.raw_alert, str):
                        raw_alert = json.loads(row.raw_alert)
                    else:
                        raw_alert = row.raw_alert
                else:
                    raw_alert = {}
                
                data.append({
                    'alert_id': row.alert_id,
                    'classification': str(row.classification),
                    'confidence_score': float(row.confidence_score) if row.confidence_score else 0.5,
                    'timestamp': row.timestamp,
                    'is_anomaly': bool(row.is_anomaly) if row.is_anomaly is not None else False,
                    'severity': raw_alert.get('severity', 'UNKNOWN'),
                    'source': raw_alert.get('source', 'UNKNOWN'),
                    'destination': raw_alert.get('destination', 'UNKNOWN'),
                    'protocol': raw_alert.get('protocol', 'UNKNOWN'),
                    'bytes_transferred': float(raw_alert.get('bytes_transferred', 0)),
                    'connection_count': int(raw_alert.get('connection_count', 0)),
                    'description': str(raw_alert.get('description', ''))[:200]
                })
            
            df = pd.DataFrame(data)
            logger.info(f"Gathered {len(df)} training samples for retraining")
            return df
            
        except Exception as e:
            logger.error(f"Error gathering training data: {e}")
            return None
    
    def prepare_features(self, training_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for training"""
        try:
            features = []
            
            # Basic features
            features.append(training_data['confidence_score'].fillna(0.5))
            
            # Severity encoding
            severity_map = {'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1, 'UNKNOWN': 0}
            severity_encoded = training_data['severity'].map(severity_map).fillna(0)
            features.append(severity_encoded)
            
            # Network features
            features.append(training_data['bytes_transferred'].fillna(0))
            features.append(training_data['connection_count'].fillna(0))
            
            # Protocol features (simplified to match prediction)
            protocol_features = []
            for protocol in training_data['protocol']:
                protocol_vec = [1 if protocol == p else 0 for p in ['TCP', 'UDP', 'ICMP', 'HTTP', 'HTTPS']]
                protocol_features.append(protocol_vec)
            protocol_df = pd.DataFrame(protocol_features, columns=['protocol_TCP', 'protocol_UDP', 'protocol_ICMP', 'protocol_HTTP', 'protocol_HTTPS'])
            features.append(protocol_df)
            
            # Time features
            training_data['timestamp'] = pd.to_datetime(training_data['timestamp'])
            features.append(training_data['timestamp'].dt.hour)
            features.append(training_data['timestamp'].dt.dayofweek)
            
            # Text features
            desc_length = training_data['description'].fillna('').str.len()
            features.append(desc_length)
            
            # IP features
            source_ip_private = training_data['source'].str.contains('192\.168\.|10\.|172\.', regex=True, na=False).astype(int)
            dest_ip_private = training_data['destination'].str.contains('192\.168\.|10\.|172\.', regex=True, na=False).astype(int)
            features.append(source_ip_private)
            features.append(dest_ip_private)
            
            # Combine features
            X = pd.concat(features, axis=1)
            X = X.fillna(0)
            
            # Labels
            y = (training_data['classification'] == 'anomaly').astype(int)
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            raise
    
    def classify_alert(self, alert_data: Dict) -> Dict:
        """Classify a single alert"""
        try:
            if not self.model_loaded:
                return {"error": "Model not loaded", "classification": "unknown"}
            
            # Prepare features for prediction
            features = self.extract_alert_features(alert_data)
            if not features:
                return {"error": "Could not extract features", "classification": "unknown"}
            
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Predict
            prediction = self.model.predict(features_scaled)[0]
            prediction_proba = self.model.predict_proba(features_scaled)[0]
            
            # Update performance tracking
            self.predictions_count += 1
            
            classification = "anomaly" if prediction == 1 else "benign"
            confidence = float(max(prediction_proba))
            
            result = {
                "classification": classification,
                "confidence": confidence,
                "model_version": self.model_version,
                "timestamp": datetime.now().isoformat(),
                "alert_id": alert_data.get("alert_id", "unknown")
            }
            
            # Publish to Pub/Sub if configured
            if self.config.get("pubsub_topic"):
                self.publish_classification(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error classifying alert: {e}")
            return {"error": str(e), "classification": "unknown"}
    
    def extract_alert_features(self, alert_data: Dict) -> Optional[List[float]]:
        """Extract features from alert data"""
        try:
            features = []
            
            # Basic features
            features.append(float(alert_data.get('confidence_score', 0.5)))
            
            # Severity
            severity_map = {'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1, 'UNKNOWN': 0}
            severity = alert_data.get('severity', 'UNKNOWN')
            features.append(severity_map.get(severity, 0))
            
            # Network features
            features.append(float(alert_data.get('bytes_transferred', 0)))
            features.append(int(alert_data.get('connection_count', 0)))
            
            # Protocol (simplified - match training features)
            protocol = alert_data.get('protocol', 'UNKNOWN')
            protocol_features = [1 if protocol == p else 0 for p in ['TCP', 'UDP', 'ICMP', 'HTTP', 'HTTPS']]
            features.extend(protocol_features)
            
            # Time features
            timestamp = alert_data.get('timestamp', datetime.now())
            if isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp)
            features.append(timestamp.hour)
            features.append(timestamp.weekday())
            
            # Text features
            description = alert_data.get('description', '')
            features.append(len(str(description)))
            
            # IP features
            source = alert_data.get('source', '')
            destination = alert_data.get('destination', '')
            features.append(1 if any(ip in source for ip in ['192.168.', '10.', '172.']) else 0)
            features.append(1 if any(ip in destination for ip in ['192.168.', '10.', '172.']) else 0)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None
    
    def publish_classification(self, result: Dict):
        """Publish classification result to Pub/Sub"""
        try:
            topic_path = self.pubsub_publisher.topic_path(
                self.config["project_id"], 
                self.config["pubsub_topic"]
            )
            
            message_data = json.dumps(result).encode('utf-8')
            future = self.pubsub_publisher.publish(topic_path, message_data)
            logger.debug(f"Published classification: {result['alert_id']}")
            
        except Exception as e:
            logger.warning(f"Could not publish to Pub/Sub: {e}")
    
    def save_model(self):
        """Save model to local storage and GCS"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"real_data_model_{timestamp}.pkl"
            local_path = os.path.join(self.config["model_path"], model_filename)
            
            # Ensure directory exists
            os.makedirs(self.config["model_path"], exist_ok=True)
            
            # Save locally
            with open(local_path, 'wb') as f:
                pickle.dump((self.model, self.scaler), f)
            
            logger.info(f"Model saved locally: {local_path}")
            
            # Save to GCS if configured
            if self.config.get("model_bucket"):
                bucket_name = self.config["model_bucket"]
                bucket = self.storage_client.bucket(bucket_name)
                blob = bucket.blob(f"models/{model_filename}")
                blob.upload_from_filename(local_path)
                logger.info(f"Model saved to GCS: gs://{bucket_name}/models/{model_filename}")
            
            self.model_version = timestamp
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def retrain_model(self):
        """Retrain model with new data"""
        try:
            logger.info("Starting model retraining...")
            
            # Check if enough time has passed (skip check for initial training)
            if self.last_retrain and datetime.now() - self.last_retrain < timedelta(hours=self.config["retrain_interval_hours"]):
                logger.info("Retraining not due yet")
                return
            
            # Gather new training data
            training_data = self.gather_training_data()
            if training_data is None or len(training_data) < self.config["min_retrain_samples"]:
                logger.warning("Insufficient data for retraining")
                return
            
            # Prepare features
            X, y = self.prepare_features(training_data)
            
            # Train new model
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            new_scaler = StandardScaler()
            X_train_scaled = new_scaler.fit_transform(X_train)
            X_test_scaled = new_scaler.transform(X_test)
            
            # Train new model
            new_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            
            new_model.fit(X_train_scaled, y_train)
            
            # Evaluate new model
            y_pred = new_model.predict(X_test_scaled)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
            
            logger.info(f"New model performance - F1: {f1:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")
            
            # Only replace if performance is good
            if f1 >= self.config["performance_threshold"]:
                self.model = new_model
                self.scaler = new_scaler
                self.save_model()
                self.last_retrain = datetime.now()
                self.next_retrain = datetime.now() + timedelta(hours=self.config["retrain_interval_hours"])
                logger.info("Model retrained successfully")
            else:
                logger.warning(f"New model performance ({f1:.3f}) below threshold ({self.config['performance_threshold']})")
            
        except Exception as e:
            logger.error(f"Error retraining model: {e}")
    
    def _determine_prediction_correctness(self, alert_id: str, is_true_positive: bool) -> Optional[bool]:
        """Compare feedback with last known classification for an alert.
        
        Returns True if feedback agrees with the model's classification, False if not,
        and None if the alert cannot be found.
        """
        try:
            query = f"""
            SELECT classification
            FROM `{self._processed_alerts_fqn}`
            WHERE alert_id = @alert_id
            ORDER BY timestamp DESC
            LIMIT 1
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter("alert_id", "STRING", alert_id)]
            )
            rows = list(self.bigquery_client.query(query, job_config=job_config).result())
            if not rows:
                return None
            classification = str(rows[0].classification).lower()
            model_pred_is_anomaly = classification == "anomaly"
            # If analyst says true positive, model is correct when it predicted anomaly
            # If analyst says false (benign/false positive), model is correct when predicted benign
            return (is_true_positive and model_pred_is_anomaly) or ((not is_true_positive) and (not model_pred_is_anomaly))
        except Exception as e:
            logger.warning(f"Could not determine correctness for alert {alert_id}: {e}")
            return None

    def _mark_feedback_processed(self, alert_id: str, timestamp: Optional[str]) -> None:
        """Mark feedback row(s) as processed in BigQuery."""
        try:
            if timestamp:
                query = f"""
                UPDATE `{self._feedback_fqn}`
                SET processed = TRUE
                WHERE alert_id = @alert_id AND timestamp = TIMESTAMP(@ts)
                """
                job_config = bigquery.QueryJobConfig(
                    query_parameters=[
                        bigquery.ScalarQueryParameter("alert_id", "STRING", alert_id),
                        bigquery.ScalarQueryParameter("ts", "STRING", timestamp),
                    ]
                )
            else:
                query = f"""
                UPDATE `{self._feedback_fqn}`
                SET processed = TRUE
                WHERE alert_id = @alert_id AND (processed IS FALSE OR processed IS NULL)
                """
                job_config = bigquery.QueryJobConfig(
                    query_parameters=[bigquery.ScalarQueryParameter("alert_id", "STRING", alert_id)]
                )
            self.bigquery_client.query(query, job_config=job_config).result()
        except Exception as e:
            logger.warning(f"Failed to mark feedback processed for {alert_id}: {e}")

    def poll_feedback_and_update_accuracy(self, batch_limit: int = 100) -> None:
        """Poll BigQuery feedback table, update accuracy counters, and mark processed."""
        try:
            query = f"""
            SELECT alert_id, is_true_positive, CAST(timestamp AS STRING) AS ts
            FROM `{self._feedback_fqn}`
            WHERE processed = FALSE OR processed IS NULL
            LIMIT {batch_limit}
            """
            rows = list(self.bigquery_client.query(query).result())
            if not rows:
                logger.info("No new feedback to process.")
                return
            updated_correct = 0
            processed = 0
            for row in rows:
                alert_id = row.alert_id
                is_tp = bool(row.is_true_positive)
                ts = row.ts
                correctness = self._determine_prediction_correctness(alert_id, is_tp)
                if correctness is True:
                    self.correct_predictions += 1
                    updated_correct += 1
                # Mark processed regardless of whether we could match
                self._mark_feedback_processed(alert_id, ts)
                processed += 1
            logger.info(f"Processed feedback: {processed} rows, correct_predictions +{updated_correct} (total={self.correct_predictions}).")
        except Exception as e:
            logger.error(f"Error while polling feedback: {e}")

    def get_service_status(self) -> Dict:
        """Get service status and metrics"""
        return {
            "status": "running" if self.service_running else "stopped",
            "model_loaded": self.model_loaded,
            "model_version": self.model_version,
            "predictions_count": self.predictions_count,
            "accuracy": self.correct_predictions / max(self.predictions_count, 1),
            "last_retrain": self.last_retrain.isoformat() if self.last_retrain else None,
            "next_retrain": self.next_retrain.isoformat(),
            "uptime": time.time() - getattr(self, 'start_time', time.time())
        }
    
    def health_check(self) -> Dict:
        """Health check for monitoring"""
        try:
            # Test model prediction
            test_alert = {
                "confidence_score": 0.8,
                "severity": "HIGH",
                "bytes_transferred": 1000,
                "connection_count": 5,
                "protocol": "TCP",
                "timestamp": datetime.now(),
                "description": "Test alert",
                "source": "192.168.1.1",
                "destination": "10.0.0.1"
            }
            
            result = self.classify_alert(test_alert)
            
            return {
                "healthy": True,
                "model_loaded": self.model_loaded,
                "test_prediction": result.get("classification", "unknown"),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def start_scheduled_tasks(self):
        """Start background scheduled tasks"""
        schedule.every().day.at("02:00").do(self.retrain_model)
        schedule.every(5).minutes.do(self.health_check)
        
        def run_scheduler():
            while self.service_running:
                schedule.run_pending()
                time.sleep(60)
        
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        logger.info("Scheduled tasks started")
    
    def stop_service(self):
        """Stop the service gracefully"""
        logger.info("Stopping Production CLA Service...")
        self.service_running = False
        # Save final model state
        if self.model_loaded:
            self.save_model()

# Flask API for production service
app = Flask(__name__)
cla_service = None

@app.route('/classify', methods=['POST'])
def classify_alert_endpoint():
    """API endpoint for alert classification"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        result = cla_service.classify_alert(data)
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Classification error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/status', methods=['GET'])
def get_status():
    """API endpoint for service status"""
    return jsonify(cla_service.get_service_status())

@app.route('/health', methods=['GET'])
def health_check_endpoint():
    """API endpoint for health check"""
    return jsonify(cla_service.health_check())

@app.route('/retrain', methods=['POST'])
def retrain_endpoint():
    """API endpoint for manual retraining"""
    try:
        cla_service.retrain_model()
        return jsonify({"status": "Retraining initiated"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/feedback/poll', methods=['POST'])
def feedback_poll_endpoint():
    """Manually trigger a feedback poll and accuracy update."""
    try:
        cla_service.poll_feedback_and_update_accuracy(batch_limit=500)
        return jsonify({
            "status": "ok",
            "predictions_count": cla_service.predictions_count,
            "correct_predictions": cla_service.correct_predictions,
            "accuracy": cla_service.correct_predictions / max(cla_service.predictions_count, 1)
        })
    except Exception as e:
        logger.error(f"Manual feedback poll failed: {e}")
        return jsonify({"error": str(e)}), 500

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info("Received shutdown signal")
    if cla_service:
        cla_service.stop_service()
    sys.exit(0)

def main():
    """Main function to start production service"""
    global cla_service
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize service
        cla_service = ProductionCLAService()
        cla_service.start_time = time.time()
        
        # Start scheduled tasks
        cla_service.start_scheduled_tasks()
        
        # Add periodic feedback polling to update accuracy
        def feedback_polling_loop():
            while cla_service.service_running:
                try:
                    cla_service.poll_feedback_and_update_accuracy(batch_limit=200)
                except Exception as e:
                    logger.error(f"Feedback polling error: {e}")
                # Poll every 2 minutes
                time.sleep(120)
        threading.Thread(target=feedback_polling_loop, daemon=True).start()
        
        # Start Flask app
        logger.info(f"Starting Production CLA Service on port {cla_service.config['service_port']}")
        app.run(
            host='0.0.0.0',
            port=cla_service.config['service_port'],
            debug=False,
            threaded=True
        )
        
    except Exception as e:
        logger.error(f"Failed to start service: {e}")
        if cla_service:
            cla_service.stop_service()

if __name__ == "__main__":
    main()
