#!/usr/bin/env python3
"""
Improved CLA Trainer with Imbalance Handling
Addresses the zero precision/recall issue by implementing proper class balancing
"""

import os
import json
import time
import logging
import argparse
from datetime import datetime, timezone, timedelta
from google.cloud import bigquery
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd
import numpy as np
import pickle
import glob

# Optional imports for advanced techniques
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("SMOTE not available. Install imbalanced-learn for synthetic oversampling.")

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not available. Install xgboost for gradient boosting.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("ImprovedCLATrainer")

class ImprovedCLATrainer:
    """Improved Continuous Learning Agent Trainer with imbalance handling."""
    
    def __init__(self, config_path="config/cla_config.json"):
        self.config = self._load_config(config_path)
        
        self.project_id = self.config.get("project_id", "chronicle-dev-2be9")
        self.location = self.config.get("location", "asia-southeast2")
        self.bigquery_dataset = self.config.get("bigquery_dataset", "soc_data")
        self.bigquery_feedback_table = self.config.get("bigquery_feedback_table", "feedback")
        self.bigquery_evaluations_table = self.config.get("bigquery_evaluations_table", "evaluations")
        self.bigquery_training_table = self.config.get("bigquery_training_table", "model_training")
        self.model_bucket = self.config.get("model_bucket", "gs://ai-driven-soc-models")
        self.min_training_samples = self.config.get("min_training_samples", 1000)
        self.retraining_interval_hours = self.config.get("retraining_interval_hours", 0.1)
        self.use_vertex_ai = self.config.get("use_vertex_ai", False)

        self.bq_client = bigquery.Client(project=self.project_id, location=self.location)
        
        self.last_training_time = time.time()
        self.training_enabled = True

        # Ensure models directory exists
        os.makedirs("models", exist_ok=True)
        logger.info("Improved CLA Trainer initialized successfully")

    def _load_config(self, config_path: str) -> dict:
        """Load the agent configuration from a JSON file."""
        try:
            with open(config_path, 'r') as config_file:
                config = json.load(config_file)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return {}

    def _get_feedback_data(self):
        """Fetches feedback data from BigQuery with enhanced data quality checks."""
        feedback_table_id = f'{self.project_id}.{self.bigquery_dataset}.{self.bigquery_feedback_table}'
        
        # Enhanced query with data quality checks
        query = f"""
        SELECT 
            alert_id, 
            is_true_positive, 
            confidence, 
            analyst_comments, 
            timestamp,
            CASE 
                WHEN is_true_positive IS NULL THEN 'missing'
                WHEN is_true_positive = TRUE THEN 'true_positive'
                WHEN is_true_positive = FALSE THEN 'false_positive'
                ELSE 'unknown'
            END as label_status
        FROM `{feedback_table_id}`
        WHERE processed = TRUE OR processed IS NULL
        ORDER BY timestamp DESC
        """
        
        try:
            df = self.bq_client.query(query).to_dataframe()
            
            # Data quality analysis
            total_samples = len(df)
            true_positives = len(df[df['is_true_positive'] == True])
            false_positives = len(df[df['is_true_positive'] == False])
            missing_labels = len(df[df['is_true_positive'].isnull()])
            
            logger.info(f"Data Quality Analysis:")
            logger.info(f"  Total samples: {total_samples}")
            logger.info(f"  True positives: {true_positives} ({true_positives/total_samples*100:.1f}%)")
            logger.info(f"  False positives: {false_positives} ({false_positives/total_samples*100:.1f}%)")
            logger.info(f"  Missing labels: {missing_labels} ({missing_labels/total_samples*100:.1f}%)")
            
            # Filter out missing labels for training
            df_clean = df[df['is_true_positive'].notnull()].copy()
            logger.info(f"Clean training samples: {len(df_clean)}")
            
            return df_clean
            
        except Exception as e:
            logger.error(f"Error fetching feedback data from BigQuery: {e}")
            return pd.DataFrame()

    def _prepare_training_data(self, df: pd.DataFrame):
        """Prepares features and labels with enhanced feature engineering."""
        if df.empty:
            return None, None

        # Enhanced Feature Engineering
        df['comment_length'] = df['analyst_comments'].apply(lambda x: len(str(x)) if pd.notnull(x) else 0)
        df['has_comments'] = df['analyst_comments'].apply(lambda x: 1 if pd.notnull(x) and str(x).strip() else 0)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['confidence_normalized'] = df['confidence'].fillna(0.5)  # Fill missing confidence
        
        # Additional features
        df['confidence_high'] = (df['confidence_normalized'] > 0.8).astype(int)
        df['confidence_low'] = (df['confidence_normalized'] < 0.3).astype(int)
        
        # Feature selection
        feature_columns = [
            'confidence_normalized', 'comment_length', 'has_comments',
            'hour', 'day_of_week', 'is_weekend', 'confidence_high', 'confidence_low'
        ]
        
        features = df[feature_columns].fillna(0)
        labels = df['is_true_positive'].astype(int)
        
        logger.info(f"Prepared features: {features.shape}, labels: {labels.shape}")
        logger.info(f"Feature columns: {feature_columns}")
        
        return features, labels

    def _train_with_imbalance_handling(self, X_train, X_test, y_train, y_test):
        """Train multiple models with different imbalance handling techniques."""
        models = {}
        results = {}
        
        # 1. Baseline Random Forest with class weights
        logger.info("Training Random Forest with class weights...")
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_map = {c: w for c, w in zip(classes, weights)}
        
        rf_weighted = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=42,
            class_weight=class_weight_map
        )
        rf_weighted.fit(X_train, y_train)
        
        y_pred_rf = rf_weighted.predict(X_test)
        models['rf_weighted'] = rf_weighted
        results['rf_weighted'] = self._evaluate_model(y_test, y_pred_rf, "Random Forest (Weighted)")
        
        # 2. SMOTE + Random Forest (if available)
        if SMOTE_AVAILABLE:
            logger.info("Training with SMOTE oversampling...")
            try:
                sm = SMOTE(random_state=42, k_neighbors=min(3, min(np.bincount(y_train))-1))
                X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train)
                
                rf_smote = RandomForestClassifier(
                    n_estimators=100, 
                    max_depth=10, 
                    random_state=42
                )
                rf_smote.fit(X_train_smote, y_train_smote)
                
                y_pred_smote = rf_smote.predict(X_test)
                models['rf_smote'] = rf_smote
                results['rf_smote'] = self._evaluate_model(y_test, y_pred_smote, "Random Forest (SMOTE)")
                
            except Exception as e:
                logger.warning(f"SMOTE training failed: {e}")
        
        # 3. XGBoost with scale_pos_weight (if available)
        if XGB_AVAILABLE:
            logger.info("Training XGBoost with scale_pos_weight...")
            try:
                # Calculate scale_pos_weight
                neg_count = (y_train == 0).sum()
                pos_count = (y_train == 1).sum()
                scale_pos_weight = max(1, neg_count / max(1, pos_count))
                
                xgb = XGBClassifier(
                    n_estimators=500,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=42,
                    tree_method="hist",
                    scale_pos_weight=scale_pos_weight
                )
                xgb.fit(X_train, y_train)
                
                y_pred_xgb = xgb.predict(X_test)
                models['xgb_weighted'] = xgb
                results['xgb_weighted'] = self._evaluate_model(y_test, y_pred_xgb, "XGBoost (Weighted)")
                
            except Exception as e:
                logger.warning(f"XGBoost training failed: {e}")
        
        # Select best model based on F1-score
        best_model_name = max(results.keys(), key=lambda k: results[k]['f1_score'])
        best_model = models[best_model_name]
        best_results = results[best_model_name]
        
        logger.info(f"Best model: {best_model_name} (F1: {best_results['f1_score']:.3f})")
        
        return best_model, best_results, best_model_name

    def _evaluate_model(self, y_true, y_pred, model_name):
        """Evaluate model performance with comprehensive metrics."""
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        metrics = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "model_name": model_name
        }
        
        logger.info(f"{model_name} - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, Accuracy: {accuracy:.3f}")
        
        return metrics

    def retrain_detection_model(self):
        """Retrains the detection model with improved imbalance handling."""
        logger.info("Starting improved model retraining...")
        feedback_df = self._get_feedback_data()

        if feedback_df.empty or len(feedback_df) < self.min_training_samples:
            logger.warning(f"Not enough training data: {len(feedback_df)} samples. Minimum required: {self.min_training_samples}")
            return

        features, labels = self._prepare_training_data(feedback_df)
        if features is None or labels is None:
            logger.error("Failed to prepare training data.")
            return

        # Check for class imbalance
        class_counts = labels.value_counts()
        logger.info(f"Class distribution: {class_counts.to_dict()}")
        
        if len(class_counts) < 2:
            logger.error("Only one class present in training data. Cannot train binary classifier.")
            return
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Train with imbalance handling
        best_model, best_metrics, best_model_name = self._train_with_imbalance_handling(
            X_train, X_test, y_train, y_test
        )
        
        # Add training metadata
        best_metrics.update({
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "total_samples": len(feedback_df),
            "class_distribution": class_counts.to_dict(),
            "training_timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Save model and metrics
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        model_filename = f"models/improved_model_{timestamp}.pkl"
        metrics_filename = f"models/improved_metrics_{timestamp}.json"
        
        with open(model_filename, 'wb') as f:
            pickle.dump(best_model, f)
        logger.info(f"Model saved to {model_filename}")
        
        with open(metrics_filename, 'w') as f:
            json.dump(best_metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_filename}")
        
        self.last_training_time = time.time()
        self._store_training_results(best_metrics)
        
        logger.info(f"Improved model retraining completed successfully. Best model: {best_model_name}")
        logger.info(f"Final metrics - Precision: {best_metrics['precision']:.3f}, Recall: {best_metrics['recall']:.3f}, F1: {best_metrics['f1_score']:.3f}")

    def _store_training_results(self, metrics: dict):
        """Stores training results in BigQuery."""
        table_id = f'{self.project_id}.{self.bigquery_dataset}.{self.bigquery_training_table}'
        rows_to_insert = [{
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_name": metrics.get('model_name', 'improved_model'),
            "precision": metrics['precision'],
            "recall": metrics['recall'],
            "f1_score": metrics['f1_score'],
            "accuracy": metrics['accuracy'],
            "true_positives": metrics['true_positives'],
            "false_positives": metrics['false_positives'],
            "true_negatives": metrics['true_negatives'],
            "false_negatives": metrics['false_negatives'],
            "training_samples": metrics['training_samples'],
            "test_samples": metrics['test_samples']
        }]
        try:
            errors = self.bq_client.insert_rows_json(table_id, rows_to_insert)
            if errors:
                logger.error(f"Error storing training results: {errors}")
            else:
                logger.info("Training results stored in BigQuery.")
        except Exception as e:
            logger.error(f"Error storing training results: {e}")

    def run(self):
        """Main loop for the Improved CLA Trainer."""
        logger.info("Starting improved continuous learning loop...")
        while True:
            logger.info("Polling for new feedback...")
            
            # Always check retraining conditions
            if self.should_retrain():
                logger.info("Retraining conditions met. Starting improved model training...")
                self.retrain_detection_model()
            
            time.sleep(60)  # Poll every 60 seconds

    def should_retrain(self) -> bool:
        """Check if model retraining is needed based on time interval."""
        current_time = time.time()
        retraining_interval = self.retraining_interval_hours * 3600  # Convert hours to seconds
        
        if current_time - self.last_training_time > retraining_interval:
            logger.info("Time-based retraining trigger activated")
            return True
        return False

def main():
    parser = argparse.ArgumentParser(description="Improved Continuous Learning Agent Trainer")
    parser.add_argument("--config", type=str, default="config/cla_config.json",
                        help="Path to the configuration file")
    parser.add_argument("--run-once", action="store_true",
                        help="Run training once instead of continuous loop")
    args = parser.parse_args()
    
    trainer = ImprovedCLATrainer(config_path=args.config)
    
    if args.run_once:
        logger.info("Running single training cycle...")
        trainer.retrain_detection_model()
    else:
        trainer.run()

if __name__ == "__main__":
    main()


