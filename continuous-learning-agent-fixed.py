"""
Continuous Learning Agent (CLA) for AI-driven SOC - Fixed Version

This version fixes the training trigger issue and ensures continuous training
even when all feedback is already processed.

Core Responsibilities:
- Collection and processing of feedback on system performance
- Periodic retraining of detection models
- Identification of emerging threat patterns
- Optimization of agent parameters and thresholds
"""

import os
import json
import logging
import time
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from google.cloud import bigquery
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
import pickle
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("ContinuousLearningAgent")

class ContinuousLearningAgent:
    """Continuous Learning Agent for improving SOC performance"""
    
    def __init__(self, config_path: str = "config/cla_config.json"):
        """Initialize the Continuous Learning Agent
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        
        # Initialize BigQuery client
        self.bigquery_client = bigquery.Client(
            project=self.config["project_id"], 
            location=self.config["location"]
        )
        
        # Initialize model storage
        self.model_path = f"/home/raditio.ghifiardigmail.com/ai-driven-soc/models"
        os.makedirs(self.model_path, exist_ok=True)
        
        # Training state
        self.last_training_time = 0
        self.current_model = None
        self.training_enabled = self.config.get("training_enabled", True)
        
        logger.info("Continuous Learning Agent initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load the agent configuration from a JSON file"""
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            # Use default configuration
            return {
                "project_id": "chronicle-dev-2be9",
                "location": "asia-southeast2",
                "bigquery_dataset": "soc_data",
                "bigquery_feedback_table": "feedback",
                "bigquery_evaluations_table": "evaluations",
                "bigquery_training_table": "model_training",
                "min_training_samples": 100,
                "training_enabled": True,
                "retraining_interval_hours": 0.1,  # 6 minutes
                "feedback_threshold": 50
            }
    
    def poll_for_feedback(self):
        """Poll for new feedback and trigger training if needed"""
        logger.info("Polling for new feedback...")
        
        try:
            # Get unprocessed feedback
            feedback_table_id = f"{self.config['project_id']}.{self.config['bigquery_dataset']}.{self.config['bigquery_feedback_table']}"
            query = f"""
            SELECT * FROM `{feedback_table_id}` 
            WHERE processed = FALSE OR processed IS NULL 
            LIMIT 1000
            """
            
            rows = list(self.bigquery_client.query(query).result())
            
            if not rows:
                logger.info("No new feedback found, checking retraining conditions...")
                # Check retraining even without new feedback
                if self.should_retrain():
                    logger.info("Retraining conditions met. Starting model training...")
                    self.retrain_detection_model()
                return
            
            logger.info(f"Found {len(rows)} new feedback entries.")
            
            # Process feedback
            for row in rows:
                self.process_feedback(dict(row))
            
            # Check if retraining is needed
            if self.should_retrain():
                logger.info("Retraining conditions met. Starting model training...")
                self.retrain_detection_model()
            
            # Mark feedback as processed
            alert_ids = [str(row.alert_id) for row in rows if hasattr(row, 'alert_id')]
            if alert_ids:
                alert_ids_str = ', '.join([f"'{_id}'" for _id in alert_ids])
                update_query = f"""
                UPDATE `{feedback_table_id}` 
                SET processed = TRUE 
                WHERE alert_id IN ({alert_ids_str})
                """
                self.bigquery_client.query(update_query).result()
                logger.info(f"Marked {len(alert_ids)} feedback items as processed.")
                
        except Exception as e:
            logger.error(f"Error during feedback polling: {e}")
    
    def process_feedback(self, feedback_data: Dict):
        """Process individual feedback entry"""
        alert_id = feedback_data.get('alert_id', 'unknown')
        logger.info(f"Processing feedback for alert {alert_id}")
        
        # Store feedback processing logic here
        # This is where you would extract features, labels, etc.
        
    def should_retrain(self) -> bool:
        """Check if model retraining is needed"""
        if not self.training_enabled:
            return False
            
        # Check time-based retraining
        current_time = time.time()
        retraining_interval = self.config.get("retraining_interval_hours", 0.1) * 3600  # 6 minutes
        
        if current_time - self.last_training_time > retraining_interval:
            logger.info("Time-based retraining trigger activated")
            return True
            
        # Check feedback threshold
        feedback_threshold = self.config.get("feedback_threshold", 50)
        try:
            feedback_table_id = f"{self.config['project_id']}.{self.config['bigquery_dataset']}.{self.config['bigquery_feedback_table']}"
            query = f"""
            SELECT COUNT(*) as count FROM `{feedback_table_id}` 
            WHERE processed = FALSE OR processed IS NULL
            """
            result = list(self.bigquery_client.query(query).result())
            unprocessed_count = result[0].count if result else 0
            
            if unprocessed_count >= feedback_threshold:
                logger.info(f"Feedback threshold retraining trigger activated ({unprocessed_count} >= {feedback_threshold})")
                return True
                
        except Exception as e:
            logger.error(f"Error checking feedback threshold: {e}")
            
        return False
    
    def retrain_detection_model(self) -> Dict:
        """Retrain the anomaly detection model with new data"""
        logger.info("Starting model retraining...")
        
        try:
            # Gather training data
            training_data = self._gather_training_data()
            
            if training_data is None or len(training_data) < self.config.get("min_training_samples", 100):
                logger.warning(f"Not enough training data: {len(training_data) if training_data is not None else 0} samples")
                return {"status": "insufficient_data", "message": "Not enough training data"}
            
            # Prepare features and labels
            X, y = self._prepare_training_features(training_data)
            
            # Train model
            model, metrics = self._train_model(X, y)
            
            # Save the model
            model_path = self._save_model(model, metrics)
            
            # Store training results
            self._store_training_results(model_path, metrics, len(training_data))
            
            # Update training time
            self.last_training_time = time.time()
            
            logger.info(f"Model retraining completed successfully. Metrics: {metrics}")
            
            return {
                "status": "success",
                "metrics": metrics,
                "model_path": model_path,
                "training_samples": len(training_data)
            }
            
        except Exception as e:
            logger.error(f"Error during model retraining: {e}")
            return {"status": "error", "message": str(e)}
    
    def _gather_training_data(self) -> Optional[pd.DataFrame]:
        """Gather training data from BigQuery"""
        try:
            # Get feedback data with labels - use processed data for training
            feedback_table_id = f"{self.config['project_id']}.{self.config['bigquery_dataset']}.{self.config['bigquery_feedback_table']}"
            query = f"""
            SELECT 
                alert_id,
                is_true_positive,
                confidence,
                analyst_comments,
                timestamp
            FROM `{feedback_table_id}`
            WHERE processed IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT 5000
            """
            
            rows = list(self.bigquery_client.query(query).result())
            
            if not rows:
                logger.warning("No training data found")
                return None
            
            # Convert to DataFrame
            data = []
            for row in rows:
                data.append({
                    'alert_id': row.alert_id,
                    'is_true_positive': row.is_true_positive,
                    'confidence': row.confidence,
                    'analyst_comments': row.analyst_comments,
                    'timestamp': row.timestamp
                })
            
            df = pd.DataFrame(data)
            logger.info(f"Gathered {len(df)} training samples")
            return df
            
        except Exception as e:
            logger.error(f"Error gathering training data: {e}")
            return None
    
    def _prepare_training_features(self, training_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and labels for training"""
        try:
            # Create simple features
            features = []
            
            # Confidence feature
            features.append(training_data['confidence'].fillna(0.5))
            
            # Comment length feature
            comment_length = training_data['analyst_comments'].fillna('').str.len()
            features.append(comment_length)
            
            # Time-based features
            training_data['timestamp'] = pd.to_datetime(training_data['timestamp'])
            training_data['hour'] = training_data['timestamp'].dt.hour
            training_data['day_of_week'] = training_data['timestamp'].dt.dayofweek
            
            features.append(training_data['hour'])
            features.append(training_data['day_of_week'])
            
            # Combine features
            X = pd.concat(features, axis=1)
            X.columns = ['confidence', 'comment_length', 'hour', 'day_of_week']
            
            # Labels
            y = training_data['is_true_positive'].astype(int)
            
            logger.info(f"Prepared features: {X.shape}, labels: {y.shape}")
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing training features: {e}")
            raise
    
    def _train_model(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, Dict]:
        """Train the anomaly detection model"""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train Random Forest model
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='binary', zero_division=0
            )
            
            cm = confusion_matrix(y_test, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                tn = fp = fn = tp = 0
            
            metrics = {
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "accuracy": float(model.score(X_test, y_test)),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
                "training_samples": len(X_train),
                "test_samples": len(X_test)
            }
            
            logger.info(f"Model training completed. F1 Score: {f1:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")
            
            return model, metrics
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def _save_model(self, model: Any, metrics: Dict) -> str:
        """Save the trained model"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"trained_model_{timestamp}.pkl"
            model_path = os.path.join(self.model_path, model_filename)
            
            # Save model
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Save metrics
            metrics_filename = f"model_metrics_{timestamp}.json"
            metrics_path = os.path.join(self.model_path, metrics_filename)
            
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"Model saved to {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def _store_training_results(self, model_path: str, metrics: Dict, sample_count: int):
        """Store training results in BigQuery"""
        try:
            training_table_id = f"{self.config['project_id']}.{self.config['bigquery_dataset']}.{self.config['bigquery_training_table']}"
            
            training_record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model_path": model_path,
                "metrics": json.dumps(metrics),
                "training_samples": sample_count,
                "status": "completed"
            }
            
            # Insert into BigQuery
            self.bigquery_client.insert_rows_json(training_table_id, [training_record])
            logger.info("Training results stored in BigQuery")
            
        except Exception as e:
            logger.error(f"Error storing training results: {e}")
    
    def run_continuous_learning(self):
        """Main loop for continuous learning"""
        logger.info("Starting continuous learning loop...")
        
        while True:
            try:
                # Poll for feedback and potentially retrain
                self.poll_for_feedback()
                
                # Wait before next cycle
                time.sleep(60)  # Check every minute
                
            except KeyboardInterrupt:
                logger.info("Continuous learning stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in continuous learning loop: {e}")
                time.sleep(60)  # Wait before retrying

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Continuous Learning Agent')
    parser.add_argument('--config', type=str, default='config/cla_config.json',
                        help='Path to configuration file')
    args = parser.parse_args()
    
    # Initialize agent
    agent = ContinuousLearningAgent(args.config)
    
    # Start continuous learning
    agent.run_continuous_learning()

if __name__ == "__main__":
    main()


