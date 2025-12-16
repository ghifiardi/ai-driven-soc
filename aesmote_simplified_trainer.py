#!/usr/bin/env python3
"""
Simplified AESMOTE CLA Trainer - Adversarial Reinforcement Learning with SMOTE

This simplified version uses scikit-learn instead of PyTorch for easier deployment.
Implements the core AESMOTE concepts:
1. Adversarial sample selection (hard examples)
2. SMOTE for synthetic minority oversampling
3. Ensemble methods for robust training
4. Dynamic difficulty adjustment

Based on the AESMOTE framework for improving CLA AI Agent performance.
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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier
import pickle
from collections import deque
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("AESMOTESimple")

class AdversarialSampleSelector:
    """Adversarial sample selector that chooses difficult examples"""
    
    def __init__(self, difficulty_threshold=0.7):
        self.difficulty_threshold = difficulty_threshold
        self.difficulty_history = deque(maxlen=1000)
        self.sample_difficulties = {}
        
    def calculate_sample_difficulty(self, predictions, true_labels):
        """Calculate difficulty score for each sample"""
        difficulties = []
        for pred, true in zip(predictions, true_labels):
            # Difficulty based on prediction confidence and correctness
            if pred == true:
                difficulty = 0.1  # Easy sample
            else:
                difficulty = 0.9  # Hard sample (misclassified)
            difficulties.append(difficulty)
        return np.array(difficulties)
    
    def select_difficult_samples(self, X, y, model, selection_ratio=0.3):
        """Select most difficult samples for training"""
        try:
            # Convert to numpy arrays if needed
            if hasattr(X, 'values'):
                X_array = X.values
            else:
                X_array = X
                
            if hasattr(y, 'values'):
                y_array = y.values
            else:
                y_array = y
            
            # Get predictions and probabilities
            predictions = model.predict(X_array)
            
            # Calculate difficulty scores
            difficulties = self.calculate_sample_difficulty(predictions, y_array)
            
            # Select hardest samples
            num_select = int(len(X_array) * selection_ratio)
            if num_select == 0:
                num_select = min(10, len(X_array))
            
            # Get indices of most difficult samples
            difficult_indices = np.argsort(difficulties)[-num_select:]
            
            # Update difficulty history
            for idx, diff in zip(difficult_indices, difficulties[difficult_indices]):
                self.difficulty_history.append(diff)
            
            logger.info(f"Selected {len(difficult_indices)} difficult samples (avg difficulty: {np.mean(difficulties[difficult_indices]):.3f})")
            
            return X_array[difficult_indices], y_array[difficult_indices], difficult_indices
            
        except Exception as e:
            logger.warning(f"Adversarial selection failed: {e}, using random samples")
            
            # Convert to numpy arrays if needed
            if hasattr(X, 'values'):
                X_array = X.values
            else:
                X_array = X
                
            if hasattr(y, 'values'):
                y_array = y.values
            else:
                y_array = y
            
            num_select = int(len(X_array) * selection_ratio)
            if num_select == 0:
                num_select = min(10, len(X_array))
            indices = np.random.choice(len(X_array), num_select, replace=False)
            return X_array[indices], y_array[indices], indices

class AESMOTESimpleTrainer:
    """Simplified AESMOTE CLA Trainer using scikit-learn"""
    
    def __init__(self, config_path: str = "config/cla_config.json"):
        """Initialize the Simplified AESMOTE CLA Trainer"""
        self.config = self._load_config(config_path)
        self.bigquery_client = bigquery.Client(project=self.config["project_id"])
        
        # Training parameters
        self.episodes = 50  # Reduced for faster training
        self.adversarial_ratio = 0.3  # Percentage of difficult samples to select
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {
                "project_id": "chronicle-dev-2be9",
                "bigquery_dataset": "soc_data",
                "bigquery_processed_alerts_table": "processed_alerts",
                "min_training_samples": 1000
            }
    
    def gather_training_data(self) -> Optional[pd.DataFrame]:
        """Gather training data from processed_alerts table"""
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
            LIMIT 15000
            """
            
            rows = list(self.bigquery_client.query(query).result())
            
            if not rows:
                logger.warning("No training data found")
                return None
            
            # Convert to DataFrame
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
            logger.info(f"Gathered {len(df)} training samples")
            
            # Log class distribution
            class_counts = df['classification'].value_counts()
            logger.info(f"Class distribution: {dict(class_counts)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error gathering training data: {e}")
            return None
    
    def prepare_features(self, training_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for AESMOTE training"""
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
            
            # Protocol features
            protocol_dummies = pd.get_dummies(training_data['protocol'], prefix='protocol')
            features.append(protocol_dummies)
            
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
            
            logger.info(f"Prepared features: {X.shape}, labels: {y.shape}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            raise
    
    def apply_advanced_smote(self, X_train, y_train, k_neighbors=3):
        """Apply advanced SMOTE with difficulty-aware sampling"""
        try:
            # Calculate class distribution
            minority_ratio = (y_train == 1).sum() / len(y_train)
            majority_ratio = (y_train == 0).sum() / len(y_train)
            
            logger.info(f"Before SMOTE - Minority: {minority_ratio:.3f}, Majority: {majority_ratio:.3f}")
            
            # Apply SMOTE with different strategies
            smote_strategies = []
            
            # Standard SMOTE
            try:
                smote_std = SMOTE(random_state=42, k_neighbors=k_neighbors)
                X_std, y_std = smote_std.fit_resample(X_train, y_train)
                smote_strategies.append(("Standard SMOTE", X_std, y_std))
            except Exception as e:
                logger.warning(f"Standard SMOTE failed: {e}")
            
            # Borderline SMOTE (if available)
            try:
                from imblearn.over_sampling import BorderlineSMOTE
                smote_border = BorderlineSMOTE(random_state=42, k_neighbors=k_neighbors)
                X_border, y_border = smote_border.fit_resample(X_train, y_train)
                smote_strategies.append(("Borderline SMOTE", X_border, y_border))
            except Exception as e:
                logger.warning(f"Borderline SMOTE failed: {e}")
            
            # Select best SMOTE strategy
            if smote_strategies:
                # Use the first successful strategy
                strategy_name, X_balanced, y_balanced = smote_strategies[0]
                logger.info(f"Using {strategy_name}")
            else:
                logger.warning("All SMOTE strategies failed, using original data")
                X_balanced, y_balanced = X_train, y_train
            
            # Log results
            new_minority_ratio = (y_balanced == 1).sum() / len(y_balanced)
            new_majority_ratio = (y_balanced == 0).sum() / len(y_balanced)
            
            logger.info(f"After SMOTE - Minority: {new_minority_ratio:.3f}, Majority: {new_majority_ratio:.3f}")
            logger.info(f"SMOTE generated {len(X_balanced) - len(X_train)} synthetic samples")
            
            return X_balanced, y_balanced
            
        except Exception as e:
            logger.warning(f"Advanced SMOTE failed: {e}, using original data")
            return X_train, y_train
    
    def train_aesmote_ensemble(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, Dict]:
        """Train AESMOTE ensemble with adversarial learning"""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            logger.info(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
            
            # Apply advanced SMOTE
            X_balanced, y_balanced = self.apply_advanced_smote(X_train, y_train)
            
            # Initialize adversarial selector
            adversarial_selector = AdversarialSampleSelector()
            
            # Create ensemble of models
            models = {
                'balanced_rf': BalancedRandomForestClassifier(n_estimators=200, random_state=42),
                'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'logistic_regression': LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
                'svm': SVC(probability=True, class_weight='balanced', random_state=42)
            }
            
            logger.info("Starting AESMOTE adversarial training...")
            
            # Train each model with adversarial learning
            trained_models = {}
            model_performances = {}
            
            for model_name, model in models.items():
                logger.info(f"Training {model_name} with adversarial learning...")
                
                # Initial training on balanced data
                model.fit(X_balanced, y_balanced)
                
                # Adversarial training episodes
                episode_accuracies = []
                
                for episode in range(self.episodes):
                    # Select difficult samples
                    X_difficult, y_difficult, _ = adversarial_selector.select_difficult_samples(
                        X_balanced, y_balanced, model, self.adversarial_ratio
                    )
                    
                    if len(X_difficult) > 0:
                        # Retrain on difficult samples
                        try:
                            if hasattr(model, 'partial_fit'):
                                model.partial_fit(X_difficult, y_difficult)
                            else:
                                # For models without partial_fit, retrain on combined data
                                X_combined = np.vstack([X_balanced, X_difficult])
                                y_combined = np.hstack([y_balanced, y_difficult])
                                model.fit(X_combined, y_combined)
                        except Exception as e:
                            logger.warning(f"Retraining failed for {model_name} episode {episode}: {e}")
                    
                    # Evaluate performance
                    if episode % 10 == 0:
                        try:
                            y_pred = model.predict(X_test)
                            accuracy = accuracy_score(y_test, y_pred)
                            episode_accuracies.append(accuracy)
                            
                            if episode % 20 == 0:
                                logger.info(f"{model_name} Episode {episode}: Accuracy={accuracy:.3f}")
                        except Exception as e:
                            logger.warning(f"Evaluation failed for {model_name} episode {episode}: {e}")
                
                trained_models[model_name] = model
                model_performances[model_name] = episode_accuracies
            
            # Create ensemble model
            ensemble_models = [(name, model) for name, model in trained_models.items()]
            ensemble = VotingClassifier(estimators=ensemble_models, voting='soft')
            
            # Final training of ensemble
            logger.info("Training final ensemble...")
            ensemble.fit(X_balanced, y_balanced)
            
            # Final evaluation
            logger.info("Evaluating AESMOTE ensemble...")
            
            y_pred = ensemble.predict(X_test)
            y_pred_proba = ensemble.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='binary', zero_division=0
            )
            
            cm = confusion_matrix(y_test, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                tn = fp = fn = tp = 0
            
            # Calculate additional metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation score
            try:
                cv_scores = cross_val_score(ensemble, X_balanced, y_balanced, cv=5, scoring='f1')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
            except Exception as e:
                logger.warning(f"Cross-validation failed: {e}")
                cv_mean = cv_std = 0.0
            
            metrics = {
                "model_name": "aesmote_ensemble",
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "accuracy": float(accuracy),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
                "training_samples": len(X_balanced),
                "test_samples": len(X_test),
                "episodes_trained": self.episodes,
                "adversarial_ratio": self.adversarial_ratio,
                "cv_f1_mean": float(cv_mean),
                "cv_f1_std": float(cv_std),
                "ensemble_models": list(models.keys()),
                "model_performances": {k: float(np.mean(v[-10:])) if v else 0.0 for k, v in model_performances.items()}
            }
            
            logger.info(f"AESMOTE ensemble training completed. F1 Score: {f1:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")
            
            return ensemble, metrics
            
        except Exception as e:
            logger.error(f"Error training AESMOTE ensemble: {e}")
            raise
    
    def save_aesmote_model(self, model, metrics: Dict) -> str:
        """Save the AESMOTE model and metrics"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"models/aesmote_simple_model_{timestamp}.pkl"
            metrics_filename = f"models/aesmote_simple_metrics_{timestamp}.json"
            
            os.makedirs("models", exist_ok=True)
            
            # Save model
            with open(model_filename, 'wb') as f:
                pickle.dump(model, f)
            
            # Save metrics
            with open(metrics_filename, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"AESMOTE simple model saved to {model_filename}")
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
            logger.error(f"Error saving AESMOTE model: {e}")
            raise
    
    def run_aesmote_training(self) -> Dict:
        """Run the complete AESMOTE training pipeline"""
        try:
            logger.info("Starting AESMOTE Simple CLA training...")
            
            # Gather data
            training_data = self.gather_training_data()
            if training_data is None or len(training_data) < 1000:
                return {"status": "insufficient_data", "message": f"Only {len(training_data) if training_data is not None else 0} samples"}
            
            # Prepare features
            X, y = self.prepare_features(training_data)
            
            # Train AESMOTE ensemble
            model, metrics = self.train_aesmote_ensemble(X, y)
            
            # Save model
            model_path = self.save_aesmote_model(model, metrics)
            
            logger.info("AESMOTE Simple CLA training completed successfully!")
            return {
                "status": "success",
                "model_path": model_path,
                "metrics": metrics
            }
            
        except Exception as e:
            logger.error(f"AESMOTE training failed: {e}")
            return {"status": "error", "message": str(e)}

def main():
    """Main function to run AESMOTE Simple training"""
    try:
        trainer = AESMOTESimpleTrainer()
        result = trainer.run_aesmote_training()
        
        print("\n" + "="*70)
        print("AESMOTE SIMPLE CLA TRAINING RESULTS")
        print("="*70)
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
            print(f"Episodes Trained: {metrics['episodes_trained']}")
            print(f"CV F1-Score: {metrics['cv_f1_mean']:.3f} ± {metrics['cv_f1_std']:.3f}")
            print(f"Ensemble Models: {', '.join(metrics['ensemble_models'])}")
        else:
            print(f"Error: {result['message']}")
        
        print("="*70)
        
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        print(f"ERROR: {e}")

if __name__ == "__main__":
    main()
