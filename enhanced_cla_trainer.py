#!/usr/bin/env python3
"""
Enhanced CLA Trainer with Advanced ML Techniques
Target: Improve accuracy from 66.7% to 94%+
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import precision_recall_curve, f1_score, classification_report
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import optuna
import joblib
from datetime import datetime
import json

class EnhancedCLATrainer:
    """Enhanced Continuous Learning Agent with advanced ML techniques"""
    
    def __init__(self, config_path="enhanced_cla_config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.models = {}
        self.ensemble_weights = {}
        self.feature_scaler = StandardScaler()
        self.optimal_threshold = 0.5
        
    def extract_advanced_features(self, df):
        """Extract advanced cybersecurity features"""
        features = pd.DataFrame()
        
        # Basic features
        if 'confidence_score' in df.columns:
            features['confidence_score'] = df['confidence_score']
        
        # Network behavior features
        if 'raw_alert' in df.columns:
            features['alert_complexity'] = df['raw_alert'].str.len().fillna(0)
            features['has_ip_address'] = df['raw_alert'].str.contains(r'\d+\.\d+\.\d+\.\d+', na=False).astype(int)
            features['has_port_scan'] = df['raw_alert'].str.contains('port|scan', case=False, na=False).astype(int)
        
        # Temporal features
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            features['hour_of_day'] = df['timestamp'].dt.hour
            features['day_of_week'] = df['timestamp'].dt.dayofweek
            features['is_weekend'] = (df['timestamp'].dt.dayofweek >= 5).astype(int)
        
        # Severity encoding
        if 'severity' in df.columns:
            severity_map = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
            features['severity_numeric'] = df['severity'].map(severity_map).fillna(1)
        
        # Classification features
        if 'classification' in df.columns:
            threat_types = ['malware', 'anomaly', 'threat', 'intrusion']
            for threat_type in threat_types:
                features[f'is_{threat_type}'] = df['classification'].str.contains(threat_type, na=False).astype(int)
        
        return features.fillna(0)
    
    def train_ensemble_models(self, X, y):
        """Train multiple models and create ensemble"""
        print(f"Training ensemble models on {len(X)} samples...")
        
        # Split data for validation
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_val_scaled = self.feature_scaler.transform(X_val)
        
        models_to_train = {
            'isolation_forest': IsolationForest(
                n_estimators=200,
                contamination=0.1,
                max_samples=0.8,
                random_state=42
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50, 25),
                max_iter=500,
                random_state=42
            )
        }
        
        # Train each model
        for name, model in models_to_train.items():
            print(f"Training {name}...")
            
            if name == 'isolation_forest':
                # Isolation Forest needs different handling
                model.fit(X_train_scaled)
                scores = -model.decision_function(X_val_scaled)
                # Normalize scores to 0-1
                min_score, max_score = scores.min(), scores.max()
                predictions = (scores - min_score) / (max_score - min_score)
            else:
                model.fit(X_train_scaled, y_train)
                predictions = model.predict_proba(X_val_scaled)[:, 1]
            
            # Store model and calculate validation performance
            self.models[name] = model
            val_f1 = f1_score(y_val, predictions > 0.5)
            print(f"{name} validation F1: {val_f1:.3f}")
        
        # Optimize ensemble weights
        self.optimize_ensemble_weights(X_val_scaled, y_val)
        
        return self.models
    
    def optimize_ensemble_weights(self, X_val, y_val):
        """Optimize ensemble weights using validation data"""
        print("Optimizing ensemble weights...")
        
        def objective(trial):
            weights = {}
            for model_name in self.models.keys():
                weights[model_name] = trial.suggest_float(f'weight_{model_name}', 0.0, 1.0)
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight == 0:
                return 0.0
            
            for key in weights:
                weights[key] /= total_weight
            
            # Get predictions from each model
            ensemble_pred = np.zeros(len(X_val))
            
            for name, model in self.models.items():
                if name == 'isolation_forest':
                    scores = -model.decision_function(X_val)
                    min_score, max_score = scores.min(), scores.max()
                    pred = (scores - min_score) / (max_score - min_score)
                else:
                    pred = model.predict_proba(X_val)[:, 1]
                
                ensemble_pred += weights[name] * pred
            
            # Calculate F1 score
            f1 = f1_score(y_val, ensemble_pred > 0.5)
            return f1
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        
        # Set optimized weights
        best_params = study.best_params
        total_weight = sum(best_params.values())
        self.ensemble_weights = {k.replace('weight_', ''): v/total_weight 
                                for k, v in best_params.items()}
        
        print(f"Optimized weights: {self.ensemble_weights}")
    
    def optimize_threshold(self, X_val, y_val):
        """Optimize classification threshold using precision-recall curve"""
        print("Optimizing classification threshold...")
        
        # Get ensemble predictions
        ensemble_pred = self.predict_proba(X_val)
        
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_val, ensemble_pred)
        
        # Find optimal threshold based on F1 score
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        self.optimal_threshold = thresholds[optimal_idx]
        
        print(f"Optimal threshold: {self.optimal_threshold:.3f}")
        print(f"Best F1 score: {f1_scores[optimal_idx]:.3f}")
        
        return self.optimal_threshold
    
    def predict_proba(self, X):
        """Get ensemble probability predictions"""
        if not self.models:
            raise ValueError("Models not trained yet")
        
        # Scale features
        X_scaled = self.feature_scaler.transform(X)
        
        ensemble_pred = np.zeros(len(X_scaled))
        
        for name, model in self.models.items():
            if name == 'isolation_forest':
                scores = -model.decision_function(X_scaled)
                min_score, max_score = scores.min(), scores.max()
                pred = (scores - min_score) / (max_score - min_score)
            else:
                pred = model.predict_proba(X_scaled)[:, 1]
            
            ensemble_pred += self.ensemble_weights[name] * pred
        
        return ensemble_pred
    
    def predict(self, X):
        """Get binary predictions using optimized threshold"""
        proba = self.predict_proba(X)
        return (proba > self.optimal_threshold).astype(int)
    
    def evaluate_model(self, X_test, y_test):
        """Comprehensive model evaluation"""
        print("Evaluating model performance...")
        
        predictions = self.predict(X_test)
        proba = self.predict_proba(X_test)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        
        print(f"\n=== MODEL PERFORMANCE ===")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1 Score: {f1:.3f}")
        
        # Check if target achieved
        if accuracy >= 0.94:
            print("🎉 TARGET ACHIEVED: 94%+ accuracy!")
        else:
            print(f"⚠️  Target gap: {0.94 - accuracy:.3f} accuracy points needed")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'threshold': self.optimal_threshold
        }
    
    def save_model(self, model_path="enhanced_cla_model.pkl"):
        """Save trained model and metadata"""
        model_data = {
            'models': self.models,
            'ensemble_weights': self.ensemble_weights,
            'feature_scaler': self.feature_scaler,
            'optimal_threshold': self.optimal_threshold,
            'config': self.config,
            'training_date': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
        
        # Also save as JSON for metadata
        metadata = {
            'ensemble_weights': self.ensemble_weights,
            'optimal_threshold': self.optimal_threshold,
            'training_date': model_data['training_date'],
            'model_types': list(self.models.keys())
        }
        
        with open(model_path.replace('.pkl', '_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

def main():
    """Main training pipeline"""
    print("🚀 Starting Enhanced CLA Training...")
    
    # Initialize trainer
    trainer = EnhancedCLATrainer()
    
    # Load training data (you'll need to implement this based on your BigQuery setup)
    # For now, we'll create sample data
    print("Loading training data...")
    
    # Sample data generation (replace with real BigQuery data)
    np.random.seed(42)
    n_samples = 10000
    
    # Generate synthetic features
    X = pd.DataFrame({
        'confidence_score': np.random.beta(2, 5, n_samples),
        'alert_complexity': np.random.exponential(100, n_samples),
        'has_ip_address': np.random.binomial(1, 0.7, n_samples),
        'has_port_scan': np.random.binomial(1, 0.2, n_samples),
        'hour_of_day': np.random.randint(0, 24, n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples),
        'is_weekend': np.random.binomial(1, 0.3, n_samples),
        'severity_numeric': np.random.choice([1, 2, 3, 4], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        'is_malware': np.random.binomial(1, 0.15, n_samples),
        'is_anomaly': np.random.binomial(1, 0.25, n_samples),
        'is_threat': np.random.binomial(1, 0.2, n_samples),
        'is_intrusion': np.random.binomial(1, 0.1, n_samples)
    })
    
    # Generate realistic labels (threats are more likely with certain features)
    threat_prob = (
        X['confidence_score'] * 0.3 +
        X['has_port_scan'] * 0.4 +
        X['severity_numeric'] * 0.1 +
        X['is_malware'] * 0.6 +
        X['is_anomaly'] * 0.3 +
        X['is_threat'] * 0.5 +
        X['is_intrusion'] * 0.7
    )
    y = (threat_prob > 0.5).astype(int)
    
    print(f"Training data: {len(X)} samples, {y.sum()} threats ({y.mean():.1%})")
    
    # Train models
    trainer.train_ensemble_models(X, y)
    
    # Split for final evaluation
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Optimize threshold
    trainer.optimize_threshold(X_train, y_train)
    
    # Final evaluation
    results = trainer.evaluate_model(X_test, y_test)
    
    # Save model
    trainer.save_model()
    
    print("\n✅ Enhanced CLA training completed!")
    return results

if __name__ == "__main__":
    main()
