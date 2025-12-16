#!/usr/bin/env python3
"""
Improved Continuous Learning Agent (CLA) with Advanced Class Imbalance Handling

This version implements multiple techniques to handle severe class imbalance:
1. SMOTE oversampling
2. Advanced class weighting
3. XGBoost with scale_pos_weight
4. Ensemble methods
5. Better feature engineering
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
from sklearn.preprocessing import StandardScaler
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
logger = logging.getLogger("ImprovedCLA")

class ImprovedCLATrainer:
    """Improved CLA with advanced imbalance handling"""
    
    def __init__(self, config_path: str = "config/cla_config.json"):
        """Initialize the Improved CLA"""
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
                "bigquery_feedback_table": "feedback",
                "min_training_samples": 100
            }
    
    def gather_training_data(self) -> Optional[pd.DataFrame]:
        """Gather training data from BigQuery with better filtering"""
        try:
            feedback_table_id = f"{self.config['project_id']}.{self.config['bigquery_dataset']}.{self.config['bigquery_feedback_table']}"
            
            # Get training data with available columns
            query = f"""
            SELECT 
                alert_id,
                is_true_positive,
                confidence,
                analyst_comments,
                timestamp
            FROM `{feedback_table_id}`
            WHERE processed IS NOT NULL 
            AND is_true_positive IS NOT NULL
            AND confidence IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT 10000
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
                    'is_true_positive': bool(row.is_true_positive) if row.is_true_positive is not None else False,
                    'confidence': float(row.confidence) if row.confidence is not None else 0.5,
                    'analyst_comments': str(row.analyst_comments) if row.analyst_comments else '',
                    'timestamp': row.timestamp
                })
            
            df = pd.DataFrame(data)
            logger.info(f"Gathered {len(df)} training samples")
            
            # Log class distribution
            class_counts = df['is_true_positive'].value_counts()
            logger.info(f"Class distribution: {dict(class_counts)}")
            logger.info(f"Positive class ratio: {class_counts.get(True, 0) / len(df):.3f}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error gathering training data: {e}")
            return None
    
    def prepare_enhanced_features(self, training_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare enhanced features for training"""
        try:
            features = []
            
            # Basic features
            features.append(training_data['confidence'].fillna(0.5))
            
            # Text features
            comment_length = training_data['analyst_comments'].fillna('').str.len()
            features.append(comment_length)
            
            # Has comment (binary feature)
            has_comment = (training_data['analyst_comments'].fillna('').str.len() > 0).astype(int)
            features.append(has_comment)
            
            # Time features
            training_data['timestamp'] = pd.to_datetime(training_data['timestamp'])
            features.append(training_data['timestamp'].dt.hour)
            features.append(training_data['timestamp'].dt.dayofweek)
            features.append(training_data['timestamp'].dt.day)
            
            # Additional text features
            comment_words = training_data['analyst_comments'].fillna('').str.split().str.len()
            features.append(comment_words)
            
            # Confidence categories
            confidence_high = (training_data['confidence'] > 0.8).astype(int)
            confidence_medium = ((training_data['confidence'] > 0.5) & (training_data['confidence'] <= 0.8)).astype(int)
            confidence_low = (training_data['confidence'] <= 0.5).astype(int)
            
            features.extend([confidence_high, confidence_medium, confidence_low])
            
            # Combine all features
            X = pd.concat(features, axis=1)
            
            # Handle any remaining NaN values
            X = X.fillna(0)
            
            # Labels
            y = training_data['is_true_positive'].astype(int)
            
            logger.info(f"Prepared enhanced features: {X.shape}, labels: {y.shape}")
            logger.info(f"Feature columns: {list(X.columns)}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            raise
    
    def train_improved_model(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, Dict]:
        """Train improved model with multiple techniques"""
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
                smote = SMOTE(random_state=42, k_neighbors=1)  # Use k_neighbors=1 for small minority class
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
                    n_estimators=200,
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
                    n_estimators=300,
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
                            "class_distribution": y_train.value_counts().to_dict(),
                            "class_weights": class_weight_map
                        }
                    
                    logger.info(f"{name} - F1: {f1:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")
                    
                except Exception as e:
                    logger.warning(f"Error evaluating {name}: {e}")
            
            if best_model is None:
                raise Exception("No models trained successfully")
            
            logger.info(f"Best model: {best_metrics['model_name']} with F1: {best_metrics['f1_score']:.3f}")
            
            return best_model, best_metrics
            
        except Exception as e:
            logger.error(f"Error training improved model: {e}")
            raise
    
    def save_improved_model(self, model, metrics: Dict) -> str:
        """Save the improved model and metrics"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"models/improved_model_{timestamp}.pkl"
            metrics_filename = f"models/improved_metrics_{timestamp}.json"
            
            os.makedirs("models", exist_ok=True)
            
            # Save model
            with open(model_filename, 'wb') as f:
                pickle.dump(model, f)
            
            # Save metrics
            with open(metrics_filename, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"Improved model saved to {model_filename}")
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
            logger.error(f"Error saving improved model: {e}")
            raise
    
    def run_improved_training(self) -> Dict:
        """Run the complete improved training pipeline"""
        try:
            logger.info("Starting improved CLA training...")
            
            # Gather data
            training_data = self.gather_training_data()
            if training_data is None or len(training_data) < 50:
                return {"status": "insufficient_data", "message": f"Only {len(training_data) if training_data is not None else 0} samples"}
            
            # Prepare features
            X, y = self.prepare_enhanced_features(training_data)
            
            # Train model
            model, metrics = self.train_improved_model(X, y)
            
            # Save model
            model_path = self.save_improved_model(model, metrics)
            
            logger.info("Improved CLA training completed successfully!")
            return {
                "status": "success",
                "model_path": model_path,
                "metrics": metrics
            }
            
        except Exception as e:
            logger.error(f"Improved training failed: {e}")
            return {"status": "error", "message": str(e)}

def main():
    """Main function to run improved training"""
    try:
        trainer = ImprovedCLATrainer()
        result = trainer.run_improved_training()
        
        print("\n" + "="*50)
        print("IMPROVED CLA TRAINING RESULTS")
        print("="*50)
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
        else:
            print(f"Error: {result['message']}")
        
        print("="*50)
        
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        print(f"ERROR: {e}")

if __name__ == "__main__":
    main()
