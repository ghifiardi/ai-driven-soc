"""
Continuous Learning Agent (CLA) for AI-driven SOC

Core Responsibilities:
- Collection and processing of feedback on system performance
- Periodic retraining of detection models
- Identification of emerging threat patterns
- Optimization of agent parameters and thresholds
- Suggestion of improvements to the overall workflow
"""

import os
import json
import base64
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from google.cloud import pubsub_v1, storage, bigquery, aiplatform
from google.cloud import firestore
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import optuna
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ContinuousLearningAgent")

class ContinuousLearningAgent:
    """Continuous Learning Agent for improving SOC performance"""
    
    def __init__(self, config_path: str = "config/cla_config.json"):
        """Initialize the Continuous Learning Agent
        
        Args:
            config_path: Path to the agent configuration file
        """
        self.config = self._load_config(config_path)
        
        # Initialize clients
        self.storage_client = storage.Client(project=self.config["project_id"])
        self.bigquery_client = bigquery.Client(project=self.config["project_id"])
        self.db = firestore.Client(project=self.config["project_id"])
        
        # Initialize Pub/Sub
        self.publisher = pubsub_v1.PublisherClient()
        self.subscriber = pubsub_v1.SubscriberClient()
        
        # Model update topics
        self.ada_model_topic_path = self.publisher.topic_path(
            self.config["project_id"], 
            self.config["ada_model_topic"]
        )
        
        self.taa_parameters_topic_path = self.publisher.topic_path(
            self.config["project_id"], 
            self.config["taa_parameters_topic"]
        )
        
        # Initialize Vertex AI if needed
        if self.config.get("use_vertex_ai", True):
            aiplatform.init(project=self.config["project_id"], 
                         location=self.config["location"])
        
        logger.info("Continuous Learning Agent initialized successfully")
    
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
                "location": "us-central1",
                "feedback_subscription": "analysis-feedback-sub",
                "metrics_subscription": "performance-metrics-sub",
                "ada_model_topic": "ada-model-updates",
                "taa_parameters_topic": "taa-parameters-updates",
                "model_bucket": "security-models",
                "bigquery_dataset": "security_analytics",
                "min_training_samples": 1000,
                "optimization_trials": 20,
                "retraining_interval_hours": 24,
                "metrics_collection_days": 30,
                "feedback_threshold": 100,
                "use_vertex_ai": True,
                "vertex_endpoints": {
                    "isolation_forest": None,  # Will be created if None
                    "attack_classifier": None  # Will be created if None
                }
            }
    
    def process_feedback(self, feedback_data: Dict) -> Dict:
        """Process feedback from the Triage and Analysis Agent
        
        Args:
            feedback_data: Feedback data
            
        Returns:
            Dict containing processing results
        """
        # Log feedback
        alert_id = feedback_data.get("alert_id", "unknown")
        is_true_positive = feedback_data.get("is_true_positive", False)
        confidence = feedback_data.get("confidence", 0.0)
        
        logger.info(f"Processing feedback for alert {alert_id}: TP={is_true_positive}, confidence={confidence}")
        
        # Store the feedback in Firestore
        feedback_doc = {
            "alert_id": alert_id,
            "timestamp": datetime.now(),
            "is_true_positive": is_true_positive,
            "confidence": confidence,
            "feedback_type": feedback_data.get("feedback_type", "unknown"),
            "processed": False
        }
        
        self.db.collection("feedback").document(alert_id).set(feedback_doc)
        
        # Check if we have enough feedback to trigger model evaluation
        self._check_feedback_threshold()
        
        return {
            "alert_id": alert_id,
            "processed": True,
            "timestamp": datetime.now().isoformat()
        }
    
    def process_metrics(self, metrics_data: Dict) -> Dict:
        """Process performance metrics from the Reporting Agent
        
        Args:
            metrics_data: Performance metrics data
            
        Returns:
            Dict containing processing results
        """
        incident_id = metrics_data.get("incident_id", "unknown")
        logger.info(f"Processing metrics for incident {incident_id}")
        
        # Store metrics in BigQuery for long-term analysis
        self._store_metrics_in_bigquery(metrics_data)
        
        # Store metrics in Firestore for quick access
        metrics_doc = {
            "incident_id": incident_id,
            "timestamp": datetime.now(),
            "metrics": metrics_data,
            "processed": False
        }
        
        self.db.collection("performance_metrics").document(incident_id).set(metrics_doc)
        
        # Check if we should run the periodic model evaluation
        self._check_retraining_schedule()
        
        return {
            "incident_id": incident_id,
            "processed": True,
            "timestamp": datetime.now().isoformat()
        }
    
    def _store_metrics_in_bigquery(self, metrics_data: Dict) -> None:
        """Store metrics in BigQuery for long-term analysis
        
        Args:
            metrics_data: Performance metrics data
        """
        # Flatten nested metrics for BigQuery
        flat_metrics = self._flatten_metrics(metrics_data)
        
        # Insert into BigQuery
        table_id = f"{self.config['project_id']}.{self.config['bigquery_dataset']}.performance_metrics"
        
        try:
            errors = self.bigquery_client.insert_rows_json(table_id, [flat_metrics])
            if errors:
                logger.error(f"Error inserting metrics into BigQuery: {errors}")
        except Exception as e:
            logger.error(f"Error inserting metrics into BigQuery: {e}")
    
    def _flatten_metrics(self, metrics_data: Dict) -> Dict:
        """Flatten nested metrics data for BigQuery
        
        Args:
            metrics_data: Nested metrics data
            
        Returns:
            Flattened metrics suitable for BigQuery
        """
        # Extract fields from nested structure
        flat = {
            "incident_id": metrics_data.get("incident_id"),
            "timestamp": metrics_data.get("timestamp"),
            "original_alert_id": metrics_data.get("detection", {}).get("original_alert_id"),
            "detection_confidence": metrics_data.get("detection", {}).get("confidence"),
            "severity": metrics_data.get("detection", {}).get("severity"),
            "attack_type": metrics_data.get("analysis", {}).get("attack_type"),
            "is_true_positive": metrics_data.get("analysis", {}).get("is_true_positive"),
            "containment_status": metrics_data.get("response", {}).get("containment_status"),
            "actions_executed": metrics_data.get("response", {}).get("actions_executed"),
            "actions_pending": metrics_data.get("response", {}).get("actions_pending"),
            "response_time": metrics_data.get("response", {}).get("response_time"),
            "processed_at": datetime.now().isoformat()
        }
        
        return flat
    
    def _check_feedback_threshold(self) -> None:
        """Check if we have enough feedback to trigger model evaluation"""
        # Count unprocessed feedback
        query = self.db.collection("feedback").where("processed", "==", False)
        docs = list(query.stream())
        unprocessed_count = len(docs)
        
        logger.info(f"Unprocessed feedback count: {unprocessed_count}")
        
        # If we have enough feedback, trigger model evaluation
        if unprocessed_count >= self.config["feedback_threshold"]:
            logger.info("Feedback threshold reached, triggering model evaluation")
            self.evaluate_detection_model()
    
    def _check_retraining_schedule(self) -> None:
        """Check if we should run the periodic model retraining"""
        # Get last retraining time
        last_training_doc = self.db.collection("agent_state").document("cla").get()
        
        current_time = datetime.now()
        should_retrain = True
        
        if last_training_doc.exists:
            last_training_time = last_training_doc.get("last_model_training")
            if last_training_time:
                # Convert Firestore timestamp to datetime
                if isinstance(last_training_time, datetime):
                    last_training = last_training_time
                else:
                    last_training = last_training_time.timestamp()
                    
                hours_since_last_training = (current_time - last_training).total_seconds() / 3600
                
                if hours_since_last_training < self.config["retraining_interval_hours"]:
                    should_retrain = False
        
        if should_retrain:
            logger.info("Retraining interval reached, triggering model retraining")
            self.retrain_detection_model()
            
            # Update last training time
            self.db.collection("agent_state").document("cla").set({
                "last_model_training": current_time
            }, merge=True)
    
    def evaluate_detection_model(self) -> Dict:
        """Evaluate the current detection model using collected feedback
        
        Returns:
            Dict containing evaluation results
        """
        logger.info("Evaluating detection model")
        
        # Get all unprocessed feedback
        query = self.db.collection("feedback").where("processed", "==", False)
        feedback_docs = list(query.stream())
        
        if not feedback_docs:
            return {"status": "no_data", "message": "No unprocessed feedback found"}
        
        # Collect alert IDs to fetch the original alerts
        alert_ids = [doc.get("alert_id") for doc in [doc.to_dict() for doc in feedback_docs]]
        
        # Fetch the original alerts from BigQuery
        alerts_data = self._fetch_alerts_data(alert_ids)
        
        if not alerts_data:
            return {"status": "no_data", "message": "Could not fetch alert data"}
        
        # Prepare the evaluation dataset
        evaluation_data = self._prepare_evaluation_data(alerts_data, feedback_docs)
        
        # Calculate evaluation metrics
        evaluation_results = self._calculate_model_metrics(evaluation_data)
        
        # Determine if model needs retraining
        needs_retraining = self._check_if_retraining_needed(evaluation_results)
        
        # Mark feedback as processed
        for doc in feedback_docs:
            doc.reference.update({"processed": True})
        
        # Store evaluation results
        eval_doc = {
            "timestamp": datetime.now(),
            "metrics": evaluation_results,
            "feedback_count": len(feedback_docs),
            "needs_retraining": needs_retraining
        }
        
        self.db.collection("model_evaluations").document(f"eval_{int(time.time())}").set(eval_doc)
        
        # If model needs retraining, trigger it
        if needs_retraining:
            logger.info("Model needs retraining based on evaluation")
            self.retrain_detection_model()
        
        return {
            "status": "success",
            "metrics": evaluation_results,
            "feedback_processed": len(feedback_docs),
            "needs_retraining": needs_retraining
        }
    
    def _fetch_alerts_data(self, alert_ids: List[str]) -> Optional[pd.DataFrame]:
        """Fetch original alert data from BigQuery
        
        Args:
            alert_ids: List of alert IDs
            
        Returns:
            DataFrame with alert data or None if no data found
        """
        if not alert_ids:
            return None
        
        # Construct a query that safely handles a list of IDs
        placeholders = ", ".join([f"'{aid}'" for aid in alert_ids])
        query = f"""
        SELECT * FROM `{self.config['project_id']}.{self.config['bigquery_dataset']}.alerts`
        WHERE alert_id IN ({placeholders})
        """
        
        try:
            # Run the query
            query_job = self.bigquery_client.query(query)
            results = query_job.result()
            
            # Convert to DataFrame
            df = results.to_dataframe()
            
            if df.empty:
                logger.warning("No alert data found in BigQuery")
                return None
            
            return df
        
        except Exception as e:
            logger.error(f"Error fetching alert data from BigQuery: {e}")
            return None
    
    def _prepare_evaluation_data(self, alerts_data: pd.DataFrame, 
                               feedback_docs: List[firestore.DocumentSnapshot]) -> pd.DataFrame:
        """Prepare evaluation dataset by combining alerts with feedback
        
        Args:
            alerts_data: DataFrame with alert data
            feedback_docs: List of feedback document snapshots
            
        Returns:
            DataFrame with combined data for evaluation
        """
        # Convert feedback to DataFrame
        feedback_data = [doc.to_dict() for doc in feedback_docs]
        feedback_df = pd.DataFrame(feedback_data)
        
        # Merge based on alert_id
        merged_df = pd.merge(alerts_data, feedback_df, on="alert_id", how="inner")
        
        # This merged dataset now contains the original alert features plus the ground truth labels
        return merged_df
    
    def _calculate_model_metrics(self, evaluation_data: pd.DataFrame) -> Dict:
        """Calculate model performance metrics
        
        Args:
            evaluation_data: DataFrame with alerts and ground truth
            
        Returns:
            Dict with metrics
        """
        # Calculate precision, recall, F1 score
        y_true = evaluation_data["is_true_positive"]
        y_pred = evaluation_data["anomaly_score"] > 0.5  # Assuming 0.5 threshold
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary"
        )
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate false positive rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # Correlation between confidence and true positives
        confidence_correlation = evaluation_data["anomaly_score"].corr(
            evaluation_data["is_true_positive"].astype(float)
        )
        
        # Analyze by attack type if available
        attack_type_metrics = {}
        if "attack_type" in evaluation_data.columns:
            for attack_type in evaluation_data["attack_type"].unique():
                if pd.isna(attack_type):
                    continue
                    
                type_data = evaluation_data[evaluation_data["attack_type"] == attack_type]
                if len(type_data) > 5:  # Only include if we have enough samples
                    type_metrics = self._calculate_metrics_for_subset(type_data)
                    attack_type_metrics[attack_type] = type_metrics
        
        # Return metrics
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "fpr": float(fpr),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "confidence_correlation": float(confidence_correlation),
            "sample_size": len(evaluation_data),
            "attack_type_metrics": attack_type_metrics
        }
    
    def _calculate_metrics_for_subset(self, subset_data: pd.DataFrame) -> Dict:
        """Calculate metrics for a subset of data (e.g., by attack type)
        
        Args:
            subset_data: DataFrame with subset of evaluation data
            
        Returns:
            Dict with metrics for the subset
        """
        y_true = subset_data["is_true_positive"]
        y_pred = subset_data["anomaly_score"] > 0.5
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        
        # Get confusion matrix if possible
        cm = np.zeros((2, 2))
        try:
            cm = confusion_matrix(y_true, y_pred)
        except:
            pass
            
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "sample_size": len(subset_data)
        }
    
    def _check_if_retraining_needed(self, evaluation_results: Dict) -> bool:
        """Determine if model retraining is needed based on metrics
        
        Args:
            evaluation_results: Dict with evaluation metrics
            
        Returns:
            True if retraining is needed, False otherwise
        """
        # The logic here can be customized based on specific requirements
        # Some simple heuristics:
        
        # 1. If false positive rate is too high
        if evaluation_results.get("fpr", 0) > 0.3:
            return True
        
        # 2. If precision is too low
        if evaluation_results.get("precision", 1.0) < 0.6:
            return True
        
        # 3. If recall is too low (missing too many true positives)
        if evaluation_results.get("recall", 1.0) < 0.6:
            return True
        
        # 4. If F1 score is below threshold
        if evaluation_results.get("f1", 1.0) < 0.65:
            return True
        
        # 5. If there's a significant number of samples and poor correlation
        sample_size = evaluation_results.get("sample_size", 0)
        correlation = evaluation_results.get("confidence_correlation", 1.0)
        if sample_size > 100 and correlation < 0.4:
            return True
        
        return False
    
    def retrain_detection_model(self) -> Dict:
        """Retrain the anomaly detection model with new data
        
        Returns:
            Dict containing retraining results
        """
        logger.info("Retraining detection model")
        
        # Gather training data
        training_data = self._gather_training_data()
        
        if training_data is None or len(training_data) < self.config["min_training_samples"]:
            logger.warning(f"Not enough training data: {len(training_data) if training_data is not None else 0} samples")
            return {"status": "insufficient_data", "message": "Not enough training data"}
        
        # Prepare features and labels
        X, y = self._prepare_training_features(training_data)
        
        # Optimize model hyperparameters
        best_params = self._optimize_model_hyperparameters(X, y)
        
        # Train model with best parameters
        model, metrics = self._train_model_with_params(X, y, best_params)
        
        # Save the model
        model_path = self._save_model(model, best_params, metrics)
        
        # Deploy model to Vertex AI if configured
        endpoint_id = None
        if self.config.get("use_vertex_ai", True):
            endpoint_id = self._deploy_to_vertex_ai(model_path)
        
        # Send model update notification
        self._send_model_update_notification(model_path, endpoint_id, metrics)
        
        # Update parameter recommendations for TAA
        taa_params = self._generate_taa_parameters(metrics)
        self._send_taa_parameters(taa_params)
        
        # Store retraining results
        training_doc = {
            "timestamp": datetime.now(),
            "metrics": metrics,
            "parameters": best_params,
            "samples": len(training_data),
            "model_path": model_path,
            "endpoint_id": endpoint_id
        }
        
        self.db.collection("model_training").document(f"training_{int(time.time())}").set(training_doc)
        
        return {
            "status": "success",
            "metrics": metrics,
            "parameters": best_params,
            "samples": len(training_data),
            "model_path": model_path,
            "endpoint_id": endpoint_id
        }
    
    def _gather_training_data(self) -> Optional[pd.DataFrame]:
        """Gather training data from BigQuery
        
        Returns:
            DataFrame with training data or None if no data found
        """
        # Define the query to get alerts with feedback
        query = f"""
        SELECT a.*, f.is_true_positive
        FROM `{self.config['project_id']}.{self.config['bigquery_dataset']}.alerts` a
        JOIN `{self.config['project_id']}.{self.config['bigquery_dataset']}.feedback` f
        ON a.alert_id = f.alert_id
        WHERE f._PARTITIONTIME >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
        """
        
        try:
            # Run the query
            query_job = self.bigquery_client.query(query)
            results = query_job.result()
            
            # Convert to DataFrame
            df = results.to_dataframe()
            
            if df.empty:
                logger.warning("No training data found in BigQuery")
                return None
            
            return df
        
        except Exception as e:
            logger.error(f"Error gathering training data: {e}")
            
            # Fall back to local data if BigQuery fails
            return self._gather_local_training_data()
    
    def _gather_local_training_data(self) -> Optional[pd.DataFrame]:
        """Gather training data from Firestore as a fallback
        
        Returns:
            DataFrame with training data or None if no data found
        """
        # Get feedback
        feedback_query = self.db.collection("feedback").limit(1000)
        feedback_docs = list(feedback_query.stream())
        feedback_data = [doc.to_dict() for doc in feedback_docs]
        
        if not feedback_data:
            return None
        
        # Get corresponding alerts (this is simplified - in a real system 
        # you would store the alert data alongside the feedback or use a join)
        alert_ids = [data.get("alert_id") for data in feedback_data]
        
        # Construct a simulated DataFrame with some features
        # This is just for testing - in a real system, you'd have actual alert data
        features = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes"]
        df = pd.DataFrame({
            "alert_id": alert_ids,
            "anomaly_score": [data.get("confidence", 0.5) for data in feedback_data],
            "is_true_positive": [data.get("is_true_positive", False) for data in feedback_data]
        })
        
        # Add some random features for testing
        for feature in features:
            df[feature] = np.random.rand(len(df))
        
        return df
    
    def _prepare_training_features(self, training_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and labels for model training
        
        Args:
            training_data: DataFrame with training data
            
        Returns:
            Tuple of (features_df, labels_series)
        """
        # Define features to use
        # This would need customization based on available data
        numeric_features = training_data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove non-feature columns
        exclude_cols = ["alert_id", "is_true_positive", "timestamp", "anomaly_score"]
        feature_cols = [col for col in numeric_features if col not in exclude_cols]
        
        # Handle missing values
        X = training_data[feature_cols].fillna(0)
        
        # Extract labels
        y = training_data["is_true_positive"]
        
        return X, y
    
    def _optimize_model_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Optimize model hyperparameters using Optuna
        
        Args:
            X: Feature DataFrame
            y: Labels Series
            
        Returns:
            Dict with optimized hyperparameters
        """
        logger.info("Optimizing model hyperparameters")
        
        # Create a study
        study = optuna.create_study(direction="maximize")
        
        # Define the objective function
        def objective(trial):
            # Define hyperparameters to tune
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_samples": trial.suggest_float("max_samples", 0.5, 1.0),
                "contamination": trial.suggest_float("contamination", 0.01, 0.3),
                "max_features": trial.suggest_float("max_features", 0.5, 1.0),
                "bootstrap": True,
                "random_state": 42
            }
            
            # Create and train the model
            model = IsolationForest(**params)
            model.fit(X)
            
            # Get anomaly scores
            # Isolation Forest returns negative scores; lower values are more anomalous
            # We need to invert for consistent interpretation (higher = more anomalous)
            scores = -model.decision_function(X)
            
            # Normalize to 0-1 range
            min_score, max_score = scores.min(), scores.max()
            normalized_scores = (scores - min_score) / (max_score - min_score)
            
            # Use a threshold of 0.5 for binary classification
            predictions = normalized_scores > 0.5
            
            # Calculate metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                y, predictions, average="binary", zero_division=0
            )
            
            # Return the metric to optimize
            return f1
        
        # Run the optimization
        study.optimize(objective, n_trials=self.config["optimization_trials"])
        
        # Get the best parameters
        best_params = study.best_params
        logger.info(f"Best parameters: {best_params}")
        
        return best_params
    
    def _train_model_with_params(self, X: pd.DataFrame, y: pd.Series, 
                               params: Dict) -> Tuple[Any, Dict]:
        """Train model with optimized parameters
        
        Args:
            X: Feature DataFrame
            y: Labels Series
            params: Dict with hyperparameters
            
        Returns:
            Tuple of (trained_model, metrics_dict)
        """
        logger.info(f"Training model with parameters: {params}")
        
        # Create and train the model
        model = IsolationForest(**params)
        model.fit(X)
        
        # Get anomaly scores and normalize
        scores = -model.decision_function(X)
        min_score, max_score = scores.min(), scores.max()
        normalized_scores = (scores - min_score) / (max_score - min_score)
        
        # Use a threshold of 0.5 for binary classification
        predictions = normalized_scores > 0.5
        
        # Calculate final metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, predictions, average="binary", zero_division=0
        )
        
        tn, fp, fn, tp = confusion_matrix(y, predictions).ravel()
        
        # Calculate false positive rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # Store metrics
        metrics = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "fpr": float(fpr),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "sample_size": len(X)
        }
        
        return model, metrics
    
    def _save_model(self, model: Any, params: Dict, metrics: Dict) -> str:
        """Save the trained model to Cloud Storage
        
        Args:
            model: Trained model
            params: Model parameters
            metrics: Training metrics
            
        Returns:
            Path to the saved model
        """
        # Create a timestamp for versioning
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"isolation_forest_{timestamp}.pkl"
        model_path = f"models/anomaly_detection/{model_filename}"
        
        # Create a temporary local file
        local_path = f"/tmp/{model_filename}"
        
        # Save the model to the local file
        with open(local_path, "wb") as f:
            pickle.dump(model, f)
        
        # Upload to Cloud Storage
        bucket = self.storage_client.bucket(self.config["model_bucket"])
        blob = bucket.blob(model_path)
        blob.upload_from_filename(local_path)
        
        # Also save model metadata
        metadata = {
            "model_type": "isolation_forest",
            "timestamp": timestamp,
            "parameters": params,
            "metrics": metrics,
            "features": {},  # Would include feature names in a real system
            "version": "1.0"
        }
        
        metadata_path = f"models/anomaly_detection/{model_filename.replace('.pkl', '_metadata.json')}"
        metadata_blob = bucket.blob(metadata_path)
        metadata_blob.upload_from_string(json.dumps(metadata, indent=2))
        
        return f"gs://{self.config['model_bucket']}/{model_path}"
    
    def _deploy_to_vertex_ai(self, model_path: str) -> Optional[str]:
        """Deploy the model to Vertex AI
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Endpoint ID or None if deployment failed
        """
        logger.info(f"Deploying model to Vertex AI: {model_path}")
        
        try:
            # Check if endpoint exists
            endpoint_id = self.config["vertex_endpoints"].get("isolation_forest")
            
            if not endpoint_id:
                # Create a new endpoint
                endpoint = aiplatform.Endpoint.create(display_name="isolation_forest_endpoint")
                endpoint_id = endpoint.name
                
                # Update config with the new endpoint ID
                self.config["vertex_endpoints"]["isolation_forest"] = endpoint_id
                
                # Save updated config
                self.db.collection("agent_state").document("cla").set({
                    "vertex_endpoints": self.config["vertex_endpoints"]
                }, merge=True)
            else:
                # Get existing endpoint
                endpoint = aiplatform.Endpoint(endpoint_id)
            
            # Deploy model to endpoint
            deployed_model = endpoint.deploy(
                model=model_path,
                machine_type="n1-standard-2",
                min_replica_count=1,
                max_replica_count=2,
                accelerator_type=None,
                accelerator_count=0
            )
            
            logger.info(f"Model deployed to endpoint: {endpoint_id}")
            return endpoint_id
        
        except Exception as e:
            logger.error(f"Error deploying model to Vertex AI: {e}")
            return None
    
    def _send_model_update_notification(self, model_path: str, 
                                     endpoint_id: Optional[str], 
                                     metrics: Dict) -> None:
        """Send model update notification to the Anomaly Detection Agent
        
        Args:
            model_path: Path to the saved model
            endpoint_id: Vertex AI endpoint ID or None
            metrics: Training metrics
        """
        update_message = {
            "timestamp": datetime.now().isoformat(),
            "model_path": model_path,
            "endpoint_id": endpoint_id,
            "metrics": metrics,
            "model_type": "isolation_forest"
        }
        
        try:
            message_data = json.dumps(update_message).encode("utf-8")
            future = self.publisher.publish(self.ada_model_topic_path, message_data)
            message_id = future.result()
            logger.info(f"Model update notification published with message ID: {message_id}")
        except Exception as e:
            logger.error(f"Error sending model update notification: {e}")
    
    def _generate_taa_parameters(self, metrics: Dict) -> Dict:
        """Generate parameter recommendations for the Triage and Analysis Agent
        
        Args:
            metrics: Model metrics
            
        Returns:
            Dict with parameter recommendations
        """
        # This is a simplified implementation
        # In a real system, you would have more sophisticated logic
        
        # Adjust confidence threshold based on precision/recall tradeoff
        precision = metrics.get("precision", 0.8)
        recall = metrics.get("recall", 0.8)
        f1 = metrics.get("f1", 0.8)
        
        # Balance precision and recall
        if precision < 0.6 and recall > 0.8:
            # Too many false positives, increase threshold
            confidence_threshold = 0.7
        elif precision > 0.8 and recall < 0.6:
            # Too many false negatives, decrease threshold
            confidence_threshold = 0.3
        else:
            # Good balance, use moderate threshold
            confidence_threshold = 0.5
        
        # Generate threat intelligence weight based on metrics
        ti_weight = min(1.0, max(0.1, f1))  # Scale with model performance
        
        parameters = {
            "timestamp": datetime.now().isoformat(),
            "confidence_threshold": confidence_threshold,
            "threat_intelligence_weight": ti_weight,
            "context_importance_factor": 0.8,
            "severity_thresholds": {
                "high": 0.8,
                "medium": 0.5,
                "low": 0.3
            },
            "based_on_metrics": metrics
        }
        
        return parameters
    
    def _send_taa_parameters(self, parameters: Dict) -> None:
        """Send parameter recommendations to the Triage and Analysis Agent
        
        Args:
            parameters: Parameter recommendations
        """
        try:
            message_data = json.dumps(parameters).encode("utf-8")
            future = self.publisher.publish(self.taa_parameters_topic_path, message_data)
            message_id = future.result()
            logger.info(f"TAA parameters published with message ID: {message_id}")
        except Exception as e:
            logger.error(f"Error sending TAA parameters: {e}")
    
    def identify_emerging_patterns(self) -> Dict:
        """Identify emerging threat patterns from recent incidents
        
        Returns:
            Dict containing identified patterns
        """
        logger.info("Analyzing incidents for emerging patterns")
        
        # Get recent incidents from the last 30 days
        days = self.config.get("pattern_analysis_days", 30)
        start_time = datetime.now() - timedelta(days=days)
        
        query = self.db.collection("incident_reports").where(
            "created_at", ">=", start_time
        ).limit(1000)
        
        docs = query.stream()
        incidents = [doc.to_dict() for doc in docs]
        
        if not incidents:
            logger.info("No recent incidents found for pattern analysis")
            return {"status": "no_data", "message": "No recent incidents found"}
        
        # Convert to DataFrame for analysis
        df = self._prepare_incident_dataframe(incidents)
        
        # Identify emerging patterns
        patterns = self._analyze_incident_patterns(df)
        
        # Store the analysis results
        analysis_doc = {
            "timestamp": datetime.now(),
            "analysis_period_days": days,
            "incident_count": len(incidents),
            "patterns": patterns
        }
        
        self.db.collection("pattern_analysis").document(f"analysis_{int(time.time())}").set(analysis_doc)
        
        return {
            "status": "success",
            "patterns": patterns,
            "incident_count": len(incidents),
            "analysis_period_days": days
        }
    
    def _prepare_incident_dataframe(self, incidents: List[Dict]) -> pd.DataFrame:
        """Prepare a DataFrame from incident data for pattern analysis
        
        Args:
            incidents: List of incident dictionaries
            
        Returns:
            DataFrame with incident data
        """
        # Extract key fields from incidents
        data = []
        for incident in incidents:
            incident_data = incident.get("data", {})
            
            # Skip if no incident data
            if not incident_data:
                continue
            
            # Extract timestamp
            timestamp = None
            if "timestamp" in incident_data:
                try:
                    timestamp = datetime.fromisoformat(incident_data["timestamp"].replace('Z', '+00:00'))
                except:
                    # Fall back to created_at from the document
                    timestamp = incident.get("created_at")
            else:
                timestamp = incident.get("created_at")
            
            # Skip if no timestamp
            if not timestamp:
                continue
            
            # Extract other fields
            original_incident = incident_data.get("incident", {})
            analysis = original_incident.get("analysis", {})
            log_data = original_incident.get("log_data", {})
            
            # Extract features from log data
            # This would be customized based on available data
            src_ip = log_data.get("src_ip")
            dst_ip = log_data.get("dst_ip")
            protocol = log_data.get("protocol_type")
            service = log_data.get("service")
            
            data.append({
                "incident_id": incident_data.get("incident_id"),
                "timestamp": timestamp,
                "severity": original_incident.get("severity"),
                "attack_type": analysis.get("attack_type"),
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "protocol": protocol,
                "service": service,
                "containment_status": incident_data.get("containment_status")
            })
        
        # Create DataFrame
        return pd.DataFrame(data)
    
    def _analyze_incident_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze incident data to identify emerging patterns
        
        Args:
            df: DataFrame with incident data
            
        Returns:
            Dict with identified patterns
        """
        # This is a simplified implementation
        # In a real system, you would use more sophisticated pattern recognition algorithms
        
        patterns = {}
        
        # 1. Trend analysis - are incidents increasing?
        if "timestamp" in df.columns:
            df["date"] = pd.to_datetime(df["timestamp"]).dt.date
            daily_counts = df.groupby("date").size()
            
            # Calculate 7-day moving average
            if len(daily_counts) >= 7:
                daily_counts = daily_counts.sort_index()
                moving_avg = daily_counts.rolling(window=7).mean()
                
                # Check if recent average is higher than earlier average
                recent_avg = moving_avg.iloc[-7:].mean() if len(moving_avg) >= 7 else None
                earlier_avg = moving_avg.iloc[-14:-7].mean() if len(moving_avg) >= 14 else None
                
                if recent_avg is not None and earlier_avg is not None:
                    percent_change = (recent_avg - earlier_avg) / earlier_avg * 100 if earlier_avg > 0 else 0
                    
                    patterns["incident_trend"] = {
                        "recent_daily_avg": float(recent_avg),
                        "earlier_daily_avg": float(earlier_avg),
                        "percent_change": float(percent_change),
                        "is_increasing": percent_change > 20,  # 20% threshold
                        "is_decreasing": percent_change < -20
                    }
        
        # 2. Attack type distribution changes
        if "attack_type" in df.columns:
            attack_counts = df["attack_type"].value_counts(normalize=True)
            
            # Check if there are emerging attack types
            emerging_types = []
            for attack_type, proportion in attack_counts.items():
                if proportion > 0.1:  # At least 10% of incidents
                    # Check if this is a new or increasing attack type
                    # Would compare with historical data in a real system
                    emerging_types.append({
                        "attack_type": attack_type,
                        "proportion": float(proportion)
                    })
            
            if emerging_types:
                patterns["emerging_attack_types"] = emerging_types
        
        # 3. Source IP clusters
        if "src_ip" in df.columns:
            src_ip_counts = df["src_ip"].value_counts()
            repeat_sources = [
                {"ip": ip, "count": int(count)}
                for ip, count in src_ip_counts.items()
                if count > 1 and ip is not None  # Filter out None values
            ]
            
            if repeat_sources:
                patterns["repeat_source_ips"] = repeat_sources[:10]  # Top 10
        
        # 4. Protocol and service patterns
        for field in ["protocol", "service"]:
            if field in df.columns:
                field_counts = df[field].value_counts(normalize=True)
                unusual_values = [
                    {"value": val, "proportion": float(prop)}
                    for val, prop in field_counts.items()
                    if prop > 0.2 and val is not None  # Filter out None values
                ]
                
                if unusual_values:
                    patterns[f"unusual_{field}_distribution"] = unusual_values
        
        # 5. Severity distribution
        if "severity" in df.columns:
            severity_counts = df["severity"].value_counts(normalize=True)
            high_severity_prop = severity_counts.get("high", 0)
            
            patterns["severity_distribution"] = {
                "high": float(high_severity_prop),
                "medium": float(severity_counts.get("medium", 0)),
                "low": float(severity_counts.get("low", 0)),
                "high_severity_alert": high_severity_prop > 0.3  # Alert if >30% high severity
            }
        
        return patterns
    
    def generate_system_improvements(self) -> Dict:
        """Generate system improvement recommendations based on collected data
        
        Returns:
            Dict containing improvement recommendations
        """
        logger.info("Generating system improvement recommendations")
        
        # Get recent model evaluations
        eval_query = self.db.collection("model_evaluations").order_by(
            "timestamp", direction=firestore.Query.DESCENDING
        ).limit(10)
        
        eval_docs = eval_query.stream()
        evaluations = [doc.to_dict() for doc in eval_docs]
        
        # Get recent pattern analyses
        pattern_query = self.db.collection("pattern_analysis").order_by(
            "timestamp", direction=firestore.Query.DESCENDING
        ).limit(5)
        
        pattern_docs = pattern_query.stream()
        pattern_analyses = [doc.to_dict() for doc in pattern_docs]
        
        # Generate recommendations based on the data
        recommendations = self._generate_recommendations(evaluations, pattern_analyses)
        
        # Store the recommendations
        recommendation_doc = {
            "timestamp": datetime.now(),
            "recommendations": recommendations,
            "based_on_evaluations": len(evaluations),
            "based_on_patterns": len(pattern_analyses)
        }
        
        self.db.collection("system_improvements").document(f"recommendations_{int(time.time())}").set(recommendation_doc)
        
        return {
            "status": "success",
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }
    
    def _generate_recommendations(self, evaluations: List[Dict], 
                               pattern_analyses: List[Dict]) -> Dict:
        """Generate improvement recommendations based on evaluations and patterns
        
        Args:
            evaluations: List of model evaluation results
            pattern_analyses: List of pattern analysis results
            
        Returns:
            Dict with recommendations by category
        """
        recommendations = {
            "detection_models": [],
            "analysis_parameters": [],
            "response_playbooks": [],
            "data_collection": [],
            "workflow_optimizations": []
        }
        
        # Analyze model evaluations
        if evaluations:
            latest_eval = evaluations[0]
            metrics = latest_eval.get("metrics", {})
            
            # Detection model recommendations
            precision = metrics.get("precision", 1.0)
            recall = metrics.get("recall", 1.0)
            f1 = metrics.get("f1", 1.0)
            fpr = metrics.get("fpr", 0.0)
            
            if precision < 0.7:
                recommendations["detection_models"].append({
                    "title": "Improve model precision",
                    "description": "Current detection models have high false positive rates. Consider feature engineering to better distinguish between normal and anomalous traffic.",
                    "priority": "high",
                    "current_value": float(precision),
                    "target_value": 0.8
                })
            
            if recall < 0.7:
                recommendations["detection_models"].append({
                    "title": "Improve model recall",
                    "description": "Current detection models are missing many true positives. Consider lowering detection thresholds or adding ensemble methods.",
                    "priority": "high",
                    "current_value": float(recall),
                    "target_value": 0.8
                })
            
            # Analysis parameter recommendations
            if "attack_type_metrics" in metrics:
                for attack_type, type_metrics in metrics["attack_type_metrics"].items():
                    type_f1 = type_metrics.get("f1", 1.0)
                    
                    if type_f1 < 0.6:
                        recommendations["analysis_parameters"].append({
                            "title": f"Improve {attack_type} detection",
                            "description": f"Performance for {attack_type} attacks is below target. Consider specialized detection rules or models for this attack type.",
                            "priority": "medium",
                            "current_value": float(type_f1),
                            "target_value": 0.8,
                            "attack_type": attack_type
                        })
        
        # Analyze pattern analyses
        if pattern_analyses:
            latest_pattern = pattern_analyses[0]
            patterns = latest_pattern.get("patterns", {})
            
            # Response playbook recommendations
            if "emerging_attack_types" in patterns:
                emerging_types = patterns["emerging_attack_types"]
                
                for attack_type in emerging_types:
                    attack_name = attack_type.get("attack_type")
                    proportion = attack_type.get("proportion", 0)
                    
                    if proportion > 0.2:  # Significant proportion
                        recommendations["response_playbooks"].append({
                            "title": f"Create playbook for {attack_name}",
                            "description": f"{attack_name} attacks represent {proportion:.1%} of recent incidents. Develop or update automated response playbooks for this attack type.",
                            "priority": "high" if proportion > 0.3 else "medium",
                            "attack_type": attack_name
                        })
            
            # Data collection recommendations
            if "severity_distribution" in patterns:
                severity_dist = patterns["severity_distribution"]
                
                if severity_dist.get("high_severity_alert", False):
                    recommendations["data_collection"].append({
                        "title": "Enhance high-severity data collection",
                        "description": "High proportion of high-severity incidents detected. Consider collecting additional context data for these incidents to improve analysis.",
                        "priority": "high"
                    })
            
            # Workflow optimizations
            if "incident_trend" in patterns:
                trend = patterns["incident_trend"]
                
                if trend.get("is_increasing", False):
                    recommendations["workflow_optimizations"].append({
                        "title": "Scale detection resources",
                        "description": f"Incident volume increasing by {trend.get('percent_change', 0):.1f}%. Consider scaling detection resources to handle increased load.",
                        "priority": "high",
                        "percent_increase": float(trend.get("percent_change", 0))
                    })
        
        # Additional general recommendations
        recommendations["workflow_optimizations"].append({
            "title": "Implement A/B testing for detection models",
            "description": "Set up an A/B testing framework to compare performance of different detection models on subsets of traffic.",
            "priority": "medium"
        })
        
        recommendations["data_collection"].append({
            "title": "Expand threat intelligence integration",
            "description": "Integrate additional threat intelligence feeds to improve context for alert triage.",
            "priority": "medium"
        })
        
        return recommendations
    
    def process_message(self, message) -> None:
        """Process a message from the feedback or metrics subscription
        
        Args:
            message: Pub/Sub message
        """
        try:
            # Decode the message data
            data = json.loads(message.data.decode("utf-8"))
            
            # Determine message type and process accordingly
            if "feedback_type" in data:
                result = self.process_feedback(data)
                logger.info(f"Processed feedback message: {result}")
            elif "detection" in data and "analysis" in data:
                result = self.process_metrics(data)
                logger.info(f"Processed metrics message: {result}")
            else:
                logger.warning(f"Unknown message type: {data}")
            
            # Acknowledge the message
            message.ack()
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            # Don't acknowledge to allow retry
    
    def listen_for_feedback(self):
        """Listen for feedback messages"""
        subscription_path = self.subscriber.subscription_path(
            self.config["project_id"],
            self.config["feedback_subscription"]
        )
        
        def callback(message):
            self.process_message(message)
        
        streaming_pull_future = self.subscriber.subscribe(
            subscription_path, callback=callback
        )
        logger.info(f"Listening for feedback on {subscription_path}")
        
        try:
            # Keep the main thread alive
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            streaming_pull_future.cancel()
            logger.info("Listener stopped")
    
    def listen_for_metrics(self):
        """Listen for metrics messages"""
        subscription_path = self.subscriber.subscription_path(
            self.config["project_id"],
            self.config["metrics_subscription"]
        )
        
        def callback(message):
            self.process_message(message)
        
        streaming_pull_future = self.subscriber.subscribe(
            subscription_path, callback=callback
        )
        logger.info(f"Listening for metrics on {subscription_path}")
        
        try:
            # Keep the main thread alive
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            streaming_pull_future.cancel()
            logger.info("Listener stopped")
    
    def run_scheduled_tasks(self):
        """Run all scheduled tasks
        
        This would be triggered by a scheduled Cloud Function or similar
        """
        logger.info("Running scheduled CLA tasks")
        
        # Evaluate detection model
        try:
            eval_result = self.evaluate_detection_model()
            logger.info(f"Model evaluation result: {eval_result}")
        except Exception as e:
            logger.error(f"Error in model evaluation: {e}")
        
        # Identify emerging patterns
        try:
            pattern_result = self.identify_emerging_patterns()
            logger.info(f"Pattern analysis result: {pattern_result}")
        except Exception as e:
            logger.error(f"Error in pattern analysis: {e}")
        
        # Generate improvement recommendations
        try:
            improve_result = self.generate_system_improvements()
            logger.info(f"Improvement recommendations: {improve_result}")
        except Exception as e:
            logger.error(f"Error generating improvements: {e}")
        
        return {
            "status": "completed",
            "timestamp": datetime.now().isoformat()
        }

# Cloud Function entry point for processing feedback
def process_feedback_function(event, context):
    """Cloud Function entry point for processing feedback
    
    Args:
        event: Pub/Sub event
        context: Event context
    """
    agent = ContinuousLearningAgent()
    
    # Decode the feedback data
    feedback_data = json.loads(base64.b64decode(event["data"]).decode("utf-8"))
    
    # Process the feedback
    result = agent.process_feedback(feedback_data)
    
    return "Feedback processed successfully"

# Cloud Function entry point for processing metrics
def process_metrics_function(event, context):
    """Cloud Function entry point for processing metrics
    
    Args:
        event: Pub/Sub event
        context: Event context
    """
    agent = ContinuousLearningAgent()
    
    # Decode the metrics data
    metrics_data = json.loads(base64.b64decode(event["data"]).decode("utf-8"))
    
    # Process the metrics
    result = agent.process_metrics(metrics_data)
    
    return "Metrics processed successfully"

# Cloud Function entry point for scheduled tasks
def run_scheduled_tasks_function(event, context):
    """Cloud Function entry point for scheduled tasks
    
    Args:
        event: Cloud Scheduler event
        context: Event context
    """
    agent = ContinuousLearningAgent()
    result = agent.run_scheduled_tasks()
    return json.dumps(result)

if __name__ == "__main__":
    # For testing
    agent = ContinuousLearningAgent()
    
    # Test feedback
    test_feedback = {
        "alert_id": "ADA-20230503123456-7890",
        "timestamp": "2023-05-03T12:34:56Z",
        "is_true_positive": True,
        "confidence": 0.85,
        "feedback_type": "triage_analysis"
    }
    
    # Process the test feedback
    feedback_result = agent.process_feedback(test_feedback)
    print(f"Feedback result: {feedback_result}")
    
    # Test metrics
    test_metrics = {
        "incident_id": "INC-20230503123456",
        "timestamp": "2023-05-03T12:34:56Z",
        "detection": {
            "original_alert_id": "ADA-20230503123456-7890",
            "confidence": 0.85,
            "severity": "high"
        },
        "analysis": {
            "attack_type": "brute_force",
            "is_true_positive": True
        },
        "response": {
            "containment_status": "partial",
            "actions_executed": 2,
            "actions_pending": 1,
            "response_time": 120  # seconds
        }
    }
    
    # Process the test metrics
    metrics_result = agent.process_metrics(test_metrics)
    print(f"Metrics result: {metrics_result}")
    
    # Generate system improvements
    improvements = agent.generate_system_improvements()
    print(json.dumps(improvements, indent=2))
