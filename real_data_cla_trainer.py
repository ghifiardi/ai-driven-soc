#!/usr/bin/env python3
"""
Real Data CLA Trainer - Uses actual processed alerts from ADA, TAA, and CRA

This trainer uses the processed_alerts table which contains real classifications:
- Anomaly: 239,032 samples (81.26%)  
- Benign: 55,131 samples (18.74%)
"""

import os
import json
import logging
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from google.cloud import bigquery
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
import xgboost as xgb
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("RealDataCLA")

class RealDataCLATrainer:
    """CLA Trainer using real processed alerts data"""
    
    def __init__(self, config_path: str = "config/cla_config.json"):
        """Initialize the Real Data CLA Trainer"""
        self.config = self._load_config(config_path)
        self.bigquery_client = bigquery.Client(project=self.config["project_id"])
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            # Return default config
            return {
                "project_id": "chronicle-dev-2be9",
                "bigquery_dataset": "soc_data",
                "bigquery_processed_alerts_table": "processed_alerts",
                "min_training_samples": 1000
            }
    
    def gather_real_training_data(self) -> Optional[pd.DataFrame]:
        """Gather real training data from processed_alerts table"""
        try:
            processed_alerts_table = f"{self.config['project_id']}.{self.config['bigquery_dataset']}.processed_alerts"
            
            # Get real processed alerts data
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
            LIMIT 50000
            """
            
            rows = list(self.bigquery_client.query(query).result())
            
            if not rows:
                logger.warning("No real training data found")
                return None
            
            # Convert to DataFrame
            data = []
            for row in rows:
                # Parse raw_alert JSON to extract features
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
                    'description': str(raw_alert.get('description', ''))[:200]  # Truncate long descriptions
                })
            
            df = pd.DataFrame(data)
            logger.info(f"Gathered {len(df)} real training samples")
            
            # Log class distribution
            class_counts = df['classification'].value_counts()
            logger.info(f"Real data class distribution: {dict(class_counts)}")
            for class_name, count in class_counts.items():
                percentage = count / len(df) * 100
                logger.info(f"  {class_name}: {count} samples ({percentage:.1f}%)")
            
            return df
            
        except Exception as e:
            logger.error(f"Error gathering real training data: {e}")
            return None
    
    def prepare_real_features(self, training_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features from real alert data"""
        try:
            features = []
            
            # Confidence score feature
            features.append(training_data['confidence_score'].fillna(0.5))
            
            # Severity encoding (HIGH=3, MEDIUM=2, LOW=1, UNKNOWN=0)
            severity_map = {'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1, 'UNKNOWN': 0}
            severity_encoded = training_data['severity'].map(severity_map).fillna(0)
            features.append(severity_encoded)
            
            # Network features
            features.append(training_data['bytes_transferred'].fillna(0))
            features.append(training_data['connection_count'].fillna(0))
            
            # Protocol features (one-hot encoded)
            protocol_dummies = pd.get_dummies(training_data['protocol'], prefix='protocol')
            features.append(protocol_dummies)
            
            # Time features
            training_data['timestamp'] = pd.to_datetime(training_data['timestamp'])
            features.append(training_data['timestamp'].dt.hour)
            features.append(training_data['timestamp'].dt.dayofweek)
            features.append(training_data['timestamp'].dt.day)
            
            # Description features
            desc_length = training_data['description'].fillna('').str.len()
            features.append(desc_length)
            
            # Has description (binary)
            has_desc = (training_data['description'].fillna('').str.len() > 0).astype(int)
            features.append(has_desc)
            
            # IP address features (simplified)
            source_ip_private = training_data['source'].str.contains('192\.168\.|10\.|172\.', regex=True, na=False).astype(int)
            dest_ip_private = training_data['destination'].str.contains('192\.168\.|10\.|172\.', regex=True, na=False).astype(int)
            
            features.append(source_ip_private)
            features.append(dest_ip_private)
            
            # Combine all features
            X = pd.concat(features, axis=1)
            
            # Handle any remaining NaN values
            X = X.fillna(0)
            
            # Labels - convert classification to binary (anomaly=1, benign=0)
            y = (training_data['classification'] == 'anomaly').astype(int)
            
            logger.info(f"Prepared real features: {X.shape}, labels: {y.shape}")
            logger.info(f"Feature columns: {list(X.columns)}")
            
            # Log label distribution
            label_counts = y.value_counts()
            logger.info(f"Label distribution: {dict(label_counts)}")
            for label, count in label_counts.items():
                percentage = count / len(y) * 100
                label_name = 'anomaly' if label == 1 else 'benign'
                logger.info(f"  {label_name}: {count} samples ({percentage:.1f}%)")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing real features: {e}")
            raise
    
    def train_real_model(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, Dict]:
        """Train model with real data using advanced techniques"""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            logger.info(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
            logger.info(f"Training class distribution: {y_train.value_counts().to_dict()}")
            
            # Calculate class weights
            classes = np.unique(y_train)
            weights = compute_class_weight('balanced', classes=classes, y=y_train)
            class_weight_map = dict(zip(classes, weights))
            logger.info(f"Class weights: {class_weight_map}")
            
            # Apply SMOTE to balance training data
            try:
                smote = SMOTE(random_state=42, k_neighbors=min(5, min(y_train.value_counts()) - 1))
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
                logger.info(f"After SMOTE: {len(X_train_balanced)} samples")
                logger.info(f"SMOTE class distribution: {pd.Series(y_train_balanced).value_counts().to_dict()}")
            except Exception as e:
                logger.warning(f"SMOTE failed: {e}, using original data")
                X_train_balanced, y_train_balanced = X_train, y_train
            
            # Train multiple models
            models = {}
            
            # 1. Balanced Random Forest
            try:
                rf_balanced = BalancedRandomForestClassifier(
                    n_estimators=300,
                    max_depth=15,
                    random_state=42,
                    n_jobs=-1
                )
                rf_balanced.fit(X_train_balanced, y_train_balanced)
                models['balanced_rf'] = rf_balanced
                logger.info("Trained Balanced Random Forest")
            except Exception as e:
                logger.warning(f"Balanced RF failed: {e}")
            
            # 2. XGBoost with scale_pos_weight
            try:
                scale_pos_weight = max(1, int((y_train == 0).sum() / max(1, (y_train == 1).sum())))
                xgb_model = xgb.XGBClassifier(
                    n_estimators=500,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=42,
                    scale_pos_weight=scale_pos_weight,
                    tree_method='hist'
                )
                xgb_model.fit(X_train_balanced, y_train_balanced)
                models['xgboost'] = xgb_model
                logger.info(f"Trained XGBoost with scale_pos_weight={scale_pos_weight}")
            except Exception as e:
                logger.warning(f"XGBoost failed: {e}")
            
            # 3. Regular Random Forest with class weights
            try:
                rf_weighted = RandomForestClassifier(
                    n_estimators=400,
                    max_depth=12,
                    random_state=42,
                    class_weight=class_weight_map,
                    n_jobs=-1
                )
                rf_weighted.fit(X_train_balanced, y_train_balanced)
                models['weighted_rf'] = rf_weighted
                logger.info("Trained Weighted Random Forest")
            except Exception as e:
                logger.warning(f"Weighted RF failed: {e}")
            
            # 4. Logistic Regression with class weights
            try:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_balanced)
                X_test_scaled = scaler.transform(X_test)
                
                lr_weighted = LogisticRegression(
                    random_state=42,
                    class_weight=class_weight_map,
                    max_iter=1000
                )
                lr_weighted.fit(X_train_scaled, y_train_balanced)
                models['weighted_lr'] = (lr_weighted, scaler)
                logger.info("Trained Weighted Logistic Regression")
            except Exception as e:
                logger.warning(f"Weighted LR failed: {e}")
            
            # Select best model based on F1 score
            best_model = None
            best_f1 = 0
            best_metrics = None
            
            for name, model_data in models.items():
                try:
                    if name == 'weighted_lr':
                        model, scaler = model_data
                        X_test_processed = scaler.transform(X_test)
                        y_pred = model.predict(X_test_processed)
                    else:
                        model = model_data
                        y_pred = model.predict(X_test)
                    
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        y_test, y_pred, average='binary', zero_division=0
                    )
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_model = model_data
                        best_metrics = {
                            "model_name": name,
                            "precision": float(precision),
                            "recall": float(recall),
                            "f1_score": float(f1),
                            "accuracy": float((y_pred == y_test).mean()),
                            "true_positives": int((y_pred & y_test).sum()),
                            "false_positives": int((y_pred & ~y_test).sum()),
                            "true_negatives": int((~y_pred & ~y_test).sum()),
                            "false_negatives": int((~y_pred & y_test).sum()),
                            "training_samples": len(X_train_balanced),
                            "test_samples": len(X_test),
                            "class_distribution": {str(k): int(v) for k, v in y_train.value_counts().to_dict().items()},
                            "class_weights": {str(k): float(v) for k, v in class_weight_map.items()}
                        }
                    
                    logger.info(f"{name} - F1: {f1:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")
                    
                except Exception as e:
                    logger.warning(f"Error evaluating {name}: {e}")
            
            if best_model is None:
                raise Exception("No models trained successfully")
            
            logger.info(f"Best model: {best_metrics['model_name']} with F1: {best_metrics['f1_score']:.3f}")
            
            return best_model, best_metrics
            
        except Exception as e:
            logger.error(f"Error training real model: {e}")
            raise
    
    def save_real_model(self, model, metrics: Dict) -> str:
        """Save the real data model and metrics"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"models/real_data_model_{timestamp}.pkl"
            metrics_filename = f"models/real_data_metrics_{timestamp}.json"
            
            os.makedirs("models", exist_ok=True)
            
            # Save model
            with open(model_filename, 'wb') as f:
                pickle.dump(model, f)
            
            # Save metrics
            with open(metrics_filename, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"Real data model saved to {model_filename}")
            logger.info(f"Metrics saved to {metrics_filename}")
            
            # Sync to dashboard
            try:
                import subprocess
                subprocess.run(["python3", "sync_models_to_dashboard.py"], check=True)
                logger.info("Models synchronized to dashboard")
            except Exception as e:
                logger.warning(f"Could not sync to dashboard: {e}")
            
            return model_filename
            
        except Exception as e:
            logger.error(f"Error saving real model: {e}")
            raise
    
    def run_real_training(self) -> Dict:
        """Run the complete real data training pipeline"""
        try:
            logger.info("Starting real data CLA training...")
            
            # Gather real data
            training_data = self.gather_real_training_data()
            if training_data is None or len(training_data) < 1000:
                return {"status": "insufficient_data", "message": f"Only {len(training_data) if training_data is not None else 0} samples"}
            
            # Prepare features
            X, y = self.prepare_real_features(training_data)
            
            # Train model
            model, metrics = self.train_real_model(X, y)
            
            # Save model
            model_path = self.save_real_model(model, metrics)
            
            logger.info("Real data CLA training completed successfully!")
            return {
                "status": "success",
                "model_path": model_path,
                "metrics": metrics
            }
            
        except Exception as e:
            logger.error(f"Real training failed: {e}")
            return {"status": "error", "message": str(e)}

def main():
    """Main function to run real data training"""
    try:
        trainer = RealDataCLATrainer()
        result = trainer.run_real_training()
        
        print("\n" + "="*60)
        print("REAL DATA CLA TRAINING RESULTS")
        print("="*60)
        print(f"Status: {result['status']}")
        
        if result['status'] == 'success':
            metrics = result['metrics']
            print(f"Model: {metrics['model_name']}")
            print(f"Precision: {metrics['precision']:.1%}")
            print(f"Recall: {metrics['recall']:.1%}")
            print(f"F1-Score: {metrics['f1_score']:.1%}")
            print(f"Accuracy: {metrics['accuracy']:.1%}")
            print(f"True Positives: {metrics['true_positives']}")
            print(f"False Positives: {metrics['false_positives']}")
            print(f"False Negatives: {metrics['false_negatives']}")
            print(f"Training Samples: {metrics['training_samples']}")
            print(f"Test Samples: {metrics['test_samples']}")
        else:
            print(f"Error: {result['message']}")
        
        print("="*60)
        
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        print(f"ERROR: {e}")

if __name__ == "__main__":
    main()
