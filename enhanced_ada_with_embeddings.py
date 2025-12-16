#!/usr/bin/env python3
"""
Enhanced Anomaly Detection Agent (ADA) with Embedding Support
Integrates contextual embeddings and RL feedback for improved SOC intelligence
"""

import os
import json
import logging
import base64
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime
from google.cloud import storage, pubsub_v1, bigquery
from google.cloud import aiplatform
from sklearn.ensemble import IsolationForest

# Import our embedding service
from embedding_service import EmbeddingService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("EnhancedADA")

class EnhancedADAWithEmbeddings:
    """Enhanced Anomaly Detection Agent with embedding support"""
    
    def __init__(self, config_path: str = "config/ada_config.json"):
        """Initialize the Enhanced ADA with embedding support
        
        Args:
            config_path: Path to the agent configuration file
        """
        self.config = self._load_config(config_path)
        
        # Initialize Google Cloud clients
        self.publisher = pubsub_v1.PublisherClient()
        self.subscriber = pubsub_v1.SubscriberClient()
        self.storage_client = storage.Client()
        self.bigquery_client = self._get_bigquery_client()
        
        # Initialize embedding service
        self.embedding_service = EmbeddingService(
            project_id=self.config["project_id"],
            location=self.config.get("location", "us-central1")
        )
        
        # Initialize Pub/Sub topics
        self.alert_topic_path = self.publisher.topic_path(
            self.config["project_id"], 
            self.config["alert_topic"]
        )
        self.embedding_topic_path = self.publisher.topic_path(
            self.config["project_id"], 
            "ada-embeddings"
        )
        
        # Initialize Vertex AI
        aiplatform.init(project=self.config["project_id"])
        
        # Load or initialize the model
        self.model = self._load_model()
        
        logger.info("Enhanced ADA with embeddings initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            # Return default configuration
            return {
                "project_id": "chronicle-dev-2be9",
                "alert_topic": "ada-alerts",
                "location": "us-central1",
                "endpoint_id": None
            }
    
    def _get_bigquery_client(self):
        """Get BigQuery client with fallback authentication"""
        try:
            service_account_paths = [
                'Service Account BigQuery/chronicle-dev-2be-a-driven-soc.json',
                'Service Account BigQuery/sa-gatra-bigquery.json',
                'Service Account BigQuery/gatra-user-bigquery.json'
            ]
            
            for sa_path in service_account_paths:
                if os.path.exists(sa_path):
                    from google.oauth2 import service_account
                    credentials = service_account.Credentials.from_service_account_file(sa_path)
                    return bigquery.Client(credentials=credentials)
            
            return bigquery.Client()
        except Exception as e:
            logger.error(f"BigQuery client initialization failed: {e}")
            return None
    
    def _load_model(self):
        """Load or initialize the anomaly detection model"""
        try:
            if self.config.get("endpoint_id"):
                # Use Vertex AI endpoint
                return aiplatform.Endpoint(self.config["endpoint_id"])
            else:
                # Use local Isolation Forest model
                return IsolationForest(contamination=0.01, random_state=42)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    
    def preprocess_log(self, log_data: Dict[str, Any]) -> pd.DataFrame:
        """Preprocess log data for anomaly detection
        
        Args:
            log_data: Raw log data dictionary
            
        Returns:
            Preprocessed features dataframe
        """
        try:
            # Extract features from log data
            features = {}
            
            # Basic features
            features['timestamp'] = log_data.get('timestamp', datetime.now().isoformat())
            features['source_ip'] = log_data.get('source_ip', '0.0.0.0')
            features['dest_ip'] = log_data.get('dest_ip', '0.0.0.0')
            features['protocol'] = log_data.get('protocol', 'unknown')
            features['port'] = log_data.get('port', 0)
            features['bytes_sent'] = log_data.get('bytes_sent', 0)
            features['bytes_received'] = log_data.get('bytes_received', 0)
            features['duration'] = log_data.get('duration', 0.0)
            
            # Convert to DataFrame
            df = pd.DataFrame([features])
            
            # Handle categorical features
            categorical_cols = ["protocol", "source_ip", "dest_ip"]
            for col in categorical_cols:
                if col in df.columns:
                    df_encoded = pd.get_dummies(df[col], prefix=col)
                    df = pd.concat([df.drop(col, axis=1), df_encoded], axis=1)
            
            return df
            
        except Exception as e:
            logger.error(f"Error preprocessing log: {e}")
            return pd.DataFrame()
    
    def detect_anomalies(self, logs_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect anomalies in a batch of logs with embedding generation
        
        Args:
            logs_batch: List of log data dictionaries
            
        Returns:
            List of detected alerts with embeddings
        """
        alerts = []
        
        for log_data in logs_batch:
            try:
                # Preprocess log data
                features_df = self.preprocess_log(log_data)
                
                if features_df.empty:
                    continue
                
                # Detect anomaly
                if self.config.get("endpoint_id"):
                    # Use Vertex AI endpoint
                    prediction = self._predict_with_endpoint(features_df)
                    anomaly_score = prediction.get("anomaly_score", 0.0)
                else:
                    # Use local model
                    numeric_features = features_df.select_dtypes(include=[np.number]).columns
                    if len(numeric_features) > 0:
                        anomaly_score = self.model.decision_function(features_df[numeric_features])[0]
                    else:
                        anomaly_score = 0.0
                
                # Determine if anomaly based on threshold
                is_anomaly = anomaly_score < -0.1  # Threshold for Isolation Forest
                confidence = min(max(abs(anomaly_score), 0.0), 1.0)
                
                # Create alert data
                alert_data = {
                    "alert_id": f"ada_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                    "timestamp": datetime.now().isoformat(),
                    "log_id": log_data.get("log_id", "unknown"),
                    "anomaly_score": float(anomaly_score),
                    "confidence": float(confidence),
                    "is_anomaly": bool(is_anomaly),
                    "classification": "anomaly" if is_anomaly else "benign",
                    "raw_log": log_data,
                    "source_ip": log_data.get("source_ip", "unknown"),
                    "dest_ip": log_data.get("dest_ip", "unknown"),
                    "protocol": log_data.get("protocol", "unknown"),
                    "bytes_sent": log_data.get("bytes_sent", 0),
                    "bytes_received": log_data.get("bytes_received", 0)
                }
                
                # Generate embedding for the alert
                if os.getenv("ENABLE_EMBEDDINGS", "true").lower() == "true":
                    try:
                        alert_data = self.embedding_service.process_alert_for_embedding(alert_data)
                        logger.info(f"Generated embedding for alert {alert_data['alert_id']}")
                    except Exception as e:
                        logger.warning(f"Failed to generate embedding for alert {alert_data['alert_id']}: {e}")
                        # Continue without embedding
                
                # Only add to alerts if it's actually an anomaly
                if is_anomaly:
                    alerts.append(alert_data)
                
            except Exception as e:
                logger.error(f"Error processing log: {e}")
                continue
        
        return alerts
    
    def _predict_with_endpoint(self, features_df: pd.DataFrame) -> Dict:
        """Predict using Vertex AI endpoint
        
        Args:
            features_df: Preprocessed features dataframe
            
        Returns:
            Prediction results
        """
        try:
            endpoint = aiplatform.Endpoint(self.config["endpoint_id"])
            instances = features_df.to_dict(orient="records")
            prediction = endpoint.predict(instances=instances)
            
            result = {
                "anomaly_score": prediction.predictions[0][0]
            }
            return result
        except Exception as e:
            logger.error(f"Error with endpoint prediction: {e}")
            return {"anomaly_score": 0.0}
    
    def publish_alerts(self, alerts: List[Dict]) -> None:
        """Publish detected alerts to Pub/Sub with embedding support
        
        Args:
            alerts: List of alert dictionaries with embeddings
        """
        for alert in alerts:
            try:
                # Publish to main alert topic
                message_data = json.dumps(alert).encode("utf-8")
                future = self.publisher.publish(self.alert_topic_path, message_data)
                message_id = future.result()
                logger.info(f"Alert published with message ID: {message_id}")
                
                # Publish to embedding topic if embedding exists
                if "embedding" in alert and alert["embedding"]:
                    embedding_data = {
                        "alert_id": alert["alert_id"],
                        "embedding": alert["embedding"],
                        "embedding_timestamp": alert.get("embedding_timestamp"),
                        "embedding_model": alert.get("embedding_model"),
                        "anomaly_score": alert["anomaly_score"],
                        "confidence": alert["confidence"]
                    }
                    
                    embedding_message = json.dumps(embedding_data).encode("utf-8")
                    embedding_future = self.publisher.publish(self.embedding_topic_path, embedding_message)
                    embedding_message_id = embedding_future.result()
                    logger.info(f"Embedding published with message ID: {embedding_message_id}")
                
            except Exception as e:
                logger.error(f"Error publishing alert: {e}")
    
    def store_alerts_in_bigquery(self, alerts: List[Dict]) -> None:
        """Store alerts with embeddings in BigQuery
        
        Args:
            alerts: List of alert dictionaries with embeddings
        """
        if not self.bigquery_client or not alerts:
            return
        
        try:
            table_id = f"{self.config['project_id']}.soc_data.processed_alerts"
            table = self.bigquery_client.get_table(table_id)
            
            # Prepare data for BigQuery
            rows_to_insert = []
            for alert in alerts:
                row = {
                    "alert_id": alert["alert_id"],
                    "timestamp": alert["timestamp"],
                    "classification": alert["classification"],
                    "confidence_score": alert["confidence"],
                    "is_anomaly": alert["is_anomaly"],
                    "raw_alert": json.dumps(alert["raw_log"]),
                    "source_ip": alert.get("source_ip"),
                    "dest_ip": alert.get("dest_ip"),
                    "protocol": alert.get("protocol"),
                    "bytes_sent": alert.get("bytes_sent", 0),
                    "bytes_received": alert.get("bytes_received", 0),
                    "duration": alert.get("duration", 0.0)
                }
                
                # Add embedding data if available
                if "embedding" in alert and alert["embedding"]:
                    row["embedding"] = alert["embedding"]
                    row["embedding_timestamp"] = alert.get("embedding_timestamp")
                    row["embedding_model"] = alert.get("embedding_model")
                    row["embedding_similarity"] = alert.get("embedding_similarity")
                    row["rl_reward_score"] = alert.get("rl_reward_score")
                
                rows_to_insert.append(row)
            
            # Insert rows
            errors = self.bigquery_client.insert_rows_json(table, rows_to_insert)
            if errors:
                logger.error(f"BigQuery insert errors: {errors}")
            else:
                logger.info(f"Successfully stored {len(rows_to_insert)} alerts in BigQuery")
                
        except Exception as e:
            logger.error(f"Error storing alerts in BigQuery: {e}")
    
    def process_logs(self, event, context):
        """Enhanced Cloud Function entry point for processing new logs with embeddings
        
        Args:
            event: Pub/Sub event
            context: Event context
        """
        try:
            # Parse the incoming logs
            log_data = json.loads(base64.b64decode(event["data"]).decode("utf-8"))
            
            # Handle both single log and batch formats
            logs_batch = log_data if isinstance(log_data, list) else [log_data]
            
            # Detect anomalies with embedding generation
            alerts = self.detect_anomalies(logs_batch)
            
            # Store alerts in BigQuery
            if alerts:
                self.store_alerts_in_bigquery(alerts)
                self.publish_alerts(alerts)
                
            return f"Log processing completed successfully. Generated {len(alerts)} alerts with embeddings."
        except Exception as e:
            logger.error(f"Error in process_logs: {e}")
            raise

# Test function
def test_enhanced_ada():
    """Test the enhanced ADA with embedding support"""
    try:
        # Initialize enhanced ADA
        ada = EnhancedADAWithEmbeddings()
        
        # Create sample log data
        sample_logs = [
            {
                "log_id": "test_001",
                "timestamp": "2024-01-01T10:00:00Z",
                "source_ip": "192.168.1.100",
                "dest_ip": "10.0.0.1",
                "protocol": "tcp",
                "port": 80,
                "bytes_sent": 1024,
                "bytes_received": 2048,
                "duration": 1.5
            },
            {
                "log_id": "test_002",
                "timestamp": "2024-01-01T10:01:00Z",
                "source_ip": "192.168.1.101",
                "dest_ip": "10.0.0.2",
                "protocol": "udp",
                "port": 53,
                "bytes_sent": 512,
                "bytes_received": 1024,
                "duration": 0.5
            }
        ]
        
        # Test anomaly detection with embeddings
        alerts = ada.detect_anomalies(sample_logs)
        
        print(f"✅ Enhanced ADA test completed")
        print(f"📊 Processed {len(sample_logs)} logs")
        print(f"🚨 Generated {len(alerts)} alerts")
        
        for alert in alerts:
            has_embedding = "embedding" in alert and alert["embedding"]
            print(f"  - Alert {alert['alert_id']}: {alert['classification']} (embedding: {'✅' if has_embedding else '❌'})")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced ADA test failed: {e}")
        return False

if __name__ == "__main__":
    test_enhanced_ada()












