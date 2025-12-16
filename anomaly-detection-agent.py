"""
Anomaly Detection Agent (ADA) for AI-driven SOC

Core Responsibilities:
- Continuous monitoring of network traffic and security logs
- Application of machine learning models to detect anomalous patterns
- Real-time scoring and classification of potential security events
- Generation of alerts with confidence scores
- Adaptation based on feedback from other agents
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from google.cloud import storage, pubsub_v1
from google.cloud import aiplatform
from sklearn.ensemble import IsolationForest
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AnomalyDetectionAgent")

class AnomalyDetectionAgent:
    """Anomaly Detection Agent for security log analysis"""
    
    def __init__(self, config_path: str = "config/ada_config.json"):
        """Initialize the Anomaly Detection Agent
        
        Args:
            config_path: Path to the agent configuration file
        """
        self.config = self._load_config(config_path)
        self.publisher = pubsub_v1.PublisherClient()
        self.alert_topic_path = self.publisher.topic_path(
            self.config["project_id"], 
            self.config["alert_topic"]
        )
        self.storage_client = storage.Client()
        
        # Initialize Vertex AI
        aiplatform.init(project=self.config["project_id"])
        
        # Load or initialize the model
        self.model = self._initialize_model()
        
        logger.info("Anomaly Detection Agent initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load the agent configuration from a JSON file
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Dict containing the configuration
        """
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            # Use default configuration if file not found
            return {
                "project_id": "ai-driven-soc",
                "alert_topic": "security-alerts",
                "model_bucket": "security-models",
                "model_path": "anomaly-detection/isolation_forest.pkl",
                "threshold": 0.8,
                "features": ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes"],
                "endpoint_id": None  # Will use local model if None
            }
    
    def _initialize_model(self):
        """Initialize the anomaly detection model
        
        Returns:
            The loaded model or a new instance if no saved model exists
        """
        # Check if we should use a deployed Vertex AI endpoint
        if self.config.get("endpoint_id"):
            logger.info(f"Using Vertex AI endpoint: {self.config['endpoint_id']}")
            return None  # We'll use the endpoint directly when needed
        
        # Otherwise, load or create a local model
        try:
            bucket = self.storage_client.bucket(self.config["model_bucket"])
            blob = bucket.blob(self.config["model_path"])
            
            # If the model exists in GCS, download and load it
            if blob.exists():
                model_path = "/tmp/model.pkl"
                blob.download_to_filename(model_path)
                import pickle
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                logger.info("Model loaded from GCS")
                return model
        except Exception as e:
            logger.warning(f"Could not load model from GCS: {e}")
        
        # If no model exists or error occurred, create a new one
        logger.info("Creating new Isolation Forest model")
        return IsolationForest(
            contamination=0.01,  # Assuming 1% of traffic is anomalous
            random_state=42
        )
    
    def preprocess_log(self, log_data: Dict) -> pd.DataFrame:
        """Preprocess the raw log data for model input
        
        Args:
            log_data: Raw log data dictionary
            
        Returns:
            DataFrame with preprocessed features
        """
        # Extract the relevant features from the log
        features = {}
        for feature in self.config["features"]:
            if feature in log_data:
                features[feature] = log_data[feature]
            else:
                features[feature] = 0  # Default value for missing features
        
        # Convert categorical features using one-hot encoding
        df = pd.DataFrame([features])
        categorical_cols = ["protocol_type", "service", "flag"]
        for col in categorical_cols:
            if col in df.columns:
                df_encoded = pd.get_dummies(df[col], prefix=col)
                df = pd.concat([df.drop(col, axis=1), df_encoded], axis=1)
        
        return df
    
    def detect_anomalies(self, logs_batch: List[Dict]) -> List[Dict]:
        """Detect anomalies in a batch of logs
        
        Args:
            logs_batch: List of log dictionaries
            
        Returns:
            List of alerts for anomalous logs with confidence scores
        """
        if not logs_batch:
            return []
        
        alerts = []
        
        # Process each log separately to maintain context
        for log in logs_batch:
            try:
                # Preprocess the log data
                features_df = self.preprocess_log(log)
                
                # Use Vertex AI endpoint if configured
                if self.config.get("endpoint_id"):
                    prediction = self._predict_with_endpoint(features_df)
                    score = prediction["anomaly_score"]
                    is_anomaly = score > self.config["threshold"]
                else:
                    # Use local model
                    # Isolation Forest returns negative scores for anomalies
                    raw_score = self.model.decision_function(features_df)[0]
                    # Convert to a confidence score (0-1 where higher is more anomalous)
                    score = 1 - (raw_score + 0.5)  # Normalize to 0-1 range
                    is_anomaly = score > self.config["threshold"]
                
                # Create an alert if anomalous
                if is_anomaly:
                    alert = {
                        "alert_id": f"ADA-{datetime.now().strftime('%Y%m%d%H%M%S')}-{hash(str(log)) % 10000}",
                        "timestamp": datetime.now().isoformat(),
                        "log_data": log,
                        "anomaly_score": float(score),
                        "confidence": float(score),
                        "alert_type": "ANOMALY_DETECTION",
                        "description": f"Anomalous network activity detected with confidence {score:.2f}",
                        "raw_data_reference": log.get("log_id", "unknown")
                    }
                    alerts.append(alert)
                    logger.info(f"Anomaly detected: {alert['alert_id']} with score {score:.2f}")
            except Exception as e:
                logger.error(f"Error processing log: {e}")
                continue
        
        return alerts
    
    def _predict_with_endpoint(self, features_df: pd.DataFrame) -> Dict:
        """Use Vertex AI endpoint for prediction
        
        Args:
            features_df: Preprocessed features dataframe
            
        Returns:
            Prediction results
        """
        endpoint = aiplatform.Endpoint(self.config["endpoint_id"])
        instances = features_df.to_dict(orient="records")
        prediction = endpoint.predict(instances=instances)
        
        # Process the prediction response
        result = {
            "anomaly_score": prediction.predictions[0][0]
        }
        return result
    
    def publish_alerts(self, alerts: List[Dict]) -> None:
        """Publish detected alerts to Pub/Sub
        
        Args:
            alerts: List of alert dictionaries
        """
        for alert in alerts:
            try:
                message_data = json.dumps(alert).encode("utf-8")
                future = self.publisher.publish(self.alert_topic_path, message_data)
                message_id = future.result()  # Wait for message to be published
                logger.info(f"Alert published with message ID: {message_id}")
            except Exception as e:
                logger.error(f"Error publishing alert: {e}")
    
    def process_logs(self, event, context):
        """Cloud Function entry point for processing new logs
        
        Args:
            event: Pub/Sub event
            context: Event context
        """
        try:
            # Parse the incoming logs
            log_data = json.loads(base64.b64decode(event["data"]).decode("utf-8"))
            
            # Handle both single log and batch formats
            logs_batch = log_data if isinstance(log_data, list) else [log_data]
            
            # Detect anomalies
            alerts = self.detect_anomalies(logs_batch)
            
            # Publish alerts if any were found
            if alerts:
                self.publish_alerts(alerts)
                
            return "Log processing completed successfully"
        except Exception as e:
            logger.error(f"Error in process_logs: {e}")
            raise
    
    def train_model(self, training_data_uri: str) -> None:
        """Train or update the anomaly detection model with new data
        
        Args:
            training_data_uri: GCS URI for training data
        """
        try:
            # Download training data from GCS
            local_path = "/tmp/training_data.csv"
            bucket_name, blob_name = self._parse_gcs_uri(training_data_uri)
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.download_to_filename(local_path)
            
            # Load and preprocess training data
            df = pd.read_csv(local_path)
            
            # Preprocess the training data (similar to preprocess_log but for batch)
            # Handle categorical features
            categorical_cols = ["protocol_type", "service", "flag"]
            for col in categorical_cols:
                if col in df.columns:
                    df_encoded = pd.get_dummies(df[col], prefix=col)
                    df = pd.concat([df.drop(col, axis=1), df_encoded], axis=1)
            
            # Train the model
            if self.config.get("endpoint_id"):
                # Use Vertex AI Pipeline for training
                self._train_with_vertex_ai(df)
            else:
                # Train local model
                numeric_features = df.select_dtypes(include=[np.number]).columns
                self.model = IsolationForest(
                    contamination=0.01,
                    random_state=42
                )
                self.model.fit(df[numeric_features])
                
                # Save the model
                import pickle
                model_path = "/tmp/model.pkl"
                with open(model_path, "wb") as f:
                    pickle.dump(self.model, f)
                
                # Upload to GCS
                bucket = self.storage_client.bucket(self.config["model_bucket"])
                blob = bucket.blob(self.config["model_path"])
                blob.upload_from_filename(model_path)
                
            logger.info("Model training completed successfully")
        except Exception as e:
            logger.error(f"Error in train_model: {e}")
            raise
    
    def _parse_gcs_uri(self, uri: str) -> Tuple[str, str]:
        """Parse a GCS URI into bucket and blob names
        
        Args:
            uri: GCS URI in the format gs://bucket-name/path/to/file
            
        Returns:
            Tuple of (bucket_name, blob_name)
        """
        if not uri.startswith("gs://"):
            raise ValueError(f"Invalid GCS URI: {uri}")
        
        path = uri[5:]  # Remove 'gs://'
        bucket_name = path.split("/")[0]
        blob_name = "/".join(path.split("/")[1:])
        
        return bucket_name, blob_name
    
    def _train_with_vertex_ai(self, training_data: pd.DataFrame) -> None:
        """Train a model using Vertex AI Pipeline
        
        Args:
            training_data: Training data dataframe
        """
        # Save training data to a temporary file
        tmp_path = "/tmp/vertex_training_data.csv"
        training_data.to_csv(tmp_path, index=False)
        
        # Upload to GCS
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        gcs_uri = f"gs://{self.config['model_bucket']}/training/data_{timestamp}.csv"
        bucket_name, blob_name = self._parse_gcs_uri(gcs_uri)
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(tmp_path)
        
        # Define and run the training pipeline
        pipeline_job = aiplatform.PipelineJob(
            display_name="anomaly-detection-training",
            template_path=self.config.get("pipeline_template_path"),
            parameter_values={
                "data_path": gcs_uri,
                "model_display_name": f"anomaly-detection-{timestamp}",
                "output_dir": f"gs://{self.config['model_bucket']}/models/{timestamp}/"
            },
            enable_caching=True
        )
        
        pipeline_job.run()
        logger.info(f"Vertex AI training pipeline started: {pipeline_job.name}")

# Example usage as a Cloud Function
def process_logs_function(event, context):
    """Cloud Function entry point
    
    Args:
        event: Pub/Sub event
        context: Event context
    """
    import base64
    agent = AnomalyDetectionAgent()
    return agent.process_logs(event, context)

# Example of scheduled model training
def train_model_function(event, context):
    """Cloud Function for scheduled model training
    
    Args:
        event: Cloud Scheduler event
        context: Event context
    """
    agent = AnomalyDetectionAgent()
    training_data_uri = os.environ.get("TRAINING_DATA_URI", 
                                      "gs://security-data/training/latest.csv")
    agent.train_model(training_data_uri)
    return "Model training initiated"

if __name__ == "__main__":
    # For local testing
    agent = AnomalyDetectionAgent()
    
    # Test with sample log data
    sample_logs = [
        {
            "log_id": "log-123456",
            "timestamp": "2023-05-03T12:34:56Z",
            "duration": 120,
            "protocol_type": "tcp",
            "service": "http",
            "flag": "SF",
            "src_bytes": 1200,
            "dst_bytes": 15000
        }
    ]
    
    # Detect anomalies
    alerts = agent.detect_anomalies(sample_logs)
    
    # Print results
    for alert in alerts:
        print(f"Alert ID: {alert['alert_id']}")
        print(f"Confidence: {alert['confidence']}")
        print(f"Description: {alert['description']}")
