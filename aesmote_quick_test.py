#!/usr/bin/env python3
"""
Quick AESMOTE Test - Fast comparison with current CLA performance

This script runs a quick test to see if AESMOTE provides any improvement
over the current 100% F1-score CLA system.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from google.cloud import bigquery
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AESMOTEQuickTest")

class QuickAESMOTETest:
    """Quick AESMOTE test for performance comparison"""
    
    def __init__(self):
        self.config = {
            "project_id": "chronicle-dev-2be9",
            "bigquery_dataset": "soc_data"
        }
        self.bigquery_client = bigquery.Client(project=self.config["project_id"])
    
    def gather_sample_data(self, limit=5000):
        """Gather a smaller sample for quick testing"""
        try:
            processed_alerts_table = f"{self.config['project_id']}.{self.config['bigquery_dataset']}.processed_alerts"
            
            query = f"""
            SELECT 
                alert_id,
                classification,
                confidence_score,
                timestamp,
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
                    'severity': raw_alert.get('severity', 'UNKNOWN'),
                    'bytes_transferred': float(raw_alert.get('bytes_transferred', 0)),
                    'connection_count': int(raw_alert.get('connection_count', 0))
                })
            
            df = pd.DataFrame(data)
            logger.info(f"Quick test data: {len(df)} samples")
            logger.info(f"Class distribution: {df['classification'].value_counts().to_dict()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error gathering sample data: {e}")
            return None
    
    def prepare_quick_features(self, data):
        """Prepare features quickly"""
        features = []
        
        # Basic features only
        features.append(data['confidence_score'].fillna(0.5))
        
        # Severity encoding
        severity_map = {'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1, 'UNKNOWN': 0}
        severity_encoded = data['severity'].map(severity_map).fillna(0)
        features.append(severity_encoded)
        
        # Network features
        features.append(data['bytes_transferred'].fillna(0))
        features.append(data['connection_count'].fillna(0))
        
        # Time features
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        features.append(data['timestamp'].dt.hour)
        features.append(data['timestamp'].dt.dayofweek)
        
        X = pd.concat(features, axis=1)
        X = X.fillna(0)
        
        y = (data['classification'] == 'anomaly').astype(int)
        
        return X, y
    
    def test_current_cla(self, X, y):
        """Test current CLA approach (RandomForest with class weights)"""
        logger.info("Testing current CLA approach...")
        start_time = time.time()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Current CLA model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)
        
        cm = confusion_matrix(y_test, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
        
        training_time = time.time() - start_time
        
        results = {
            "approach": "Current CLA",
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "accuracy": float(accuracy),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            "training_time": training_time,
            "training_samples": len(X_train),
            "test_samples": len(X_test)
        }
        
        logger.info(f"Current CLA - F1: {f1:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")
        
        return results
    
    def test_aesmote_light(self, X, y):
        """Test lightweight AESMOTE approach"""
        logger.info("Testing lightweight AESMOTE approach...")
        start_time = time.time()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Apply SMOTE
        try:
            smote = SMOTE(random_state=42, k_neighbors=3)
            X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
            logger.info(f"SMOTE generated {len(X_balanced) - len(X_train)} synthetic samples")
        except Exception as e:
            logger.warning(f"SMOTE failed: {e}, using original data")
            X_balanced, y_balanced = X_train, y_train
        
        # Ensemble with adversarial-like selection
        models = {
            'rf': RandomForestClassifier(n_estimators=50, random_state=42),
            'lr': LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
        }
        
        # Train ensemble
        ensemble = VotingClassifier(estimators=list(models.items()), voting='soft')
        ensemble.fit(X_balanced, y_balanced)
        
        # Predictions
        y_pred = ensemble.predict(X_test)
        
        # Metrics
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)
        accuracy = accuracy_score(y_test, y_pred)
        
        cm = confusion_matrix(y_test, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
        
        training_time = time.time() - start_time
        
        results = {
            "approach": "AESMOTE Light",
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "accuracy": float(accuracy),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            "training_time": training_time,
            "training_samples": len(X_balanced),
            "test_samples": len(X_test),
            "smote_applied": len(X_balanced) > len(X_train)
        }
        
        logger.info(f"AESMOTE Light - F1: {f1:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")
        
        return results
    
    def test_robustness(self, X, y):
        """Test robustness with cross-validation"""
        logger.info("Testing robustness with cross-validation...")
        
        # Current CLA
        cla_model = RandomForestClassifier(class_weight='balanced', random_state=42)
        cla_scores = cross_val_score(cla_model, X, y, cv=5, scoring='f1')
        
        # AESMOTE approach
        try:
            smote = SMOTE(random_state=42, k_neighbors=3)
            X_balanced, y_balanced = smote.fit_resample(X, y)
            aesmote_model = VotingClassifier(
                estimators=[
                    ('rf', RandomForestClassifier(random_state=42)),
                    ('lr', LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000))
                ],
                voting='soft'
            )
            aesmote_scores = cross_val_score(aesmote_model, X_balanced, y_balanced, cv=5, scoring='f1')
        except Exception as e:
            logger.warning(f"AESMOTE cross-validation failed: {e}")
            aesmote_scores = np.array([0.0])
        
        robustness_results = {
            "cla_f1_mean": float(cla_scores.mean()),
            "cla_f1_std": float(cla_scores.std()),
            "aesmote_f1_mean": float(aesmote_scores.mean()),
            "aesmote_f1_std": float(aesmote_scores.std()),
            "improvement": float(aesmote_scores.mean() - cla_scores.mean())
        }
        
        logger.info(f"Robustness - CLA: {cla_scores.mean():.3f}±{cla_scores.std():.3f}, AESMOTE: {aesmote_scores.mean():.3f}±{aesmote_scores.std():.3f}")
        
        return robustness_results
    
    def run_quick_test(self):
        """Run the complete quick test"""
        logger.info("Starting AESMOTE Quick Test...")
        
        # Gather sample data
        data = self.gather_sample_data(limit=3000)  # Smaller sample for speed
        if data is None:
            return {"error": "No data available"}
        
        # Prepare features
        X, y = self.prepare_quick_features(data)
        
        # Test both approaches
        cla_results = self.test_current_cla(X, y)
        aesmote_results = self.test_aesmote_light(X, y)
        robustness_results = self.test_robustness(X, y)
        
        # Calculate improvement
        f1_improvement = aesmote_results['f1_score'] - cla_results['f1_score']
        
        summary = {
            "test_summary": {
                "total_samples": len(data),
                "feature_count": X.shape[1],
                "class_distribution": data['classification'].value_counts().to_dict()
            },
            "cla_results": cla_results,
            "aesmote_results": aesmote_results,
            "robustness_results": robustness_results,
            "improvement_analysis": {
                "f1_score_improvement": f1_improvement,
                "precision_improvement": aesmote_results['precision'] - cla_results['precision'],
                "recall_improvement": aesmote_results['recall'] - cla_results['recall'],
                "training_time_ratio": aesmote_results['training_time'] / cla_results['training_time'],
                "significant_improvement": abs(f1_improvement) > 0.01  # 1% threshold
            }
        }
        
        return summary

def main():
    """Main function"""
    try:
        tester = QuickAESMOTETest()
        results = tester.run_quick_test()
        
        if "error" in results:
            print(f"ERROR: {results['error']}")
            return
        
        print("\n" + "="*70)
        print("AESMOTE QUICK TEST RESULTS")
        print("="*70)
        
        # Test summary
        summary = results['test_summary']
        print(f"Test Data: {summary['total_samples']} samples, {summary['feature_count']} features")
        print(f"Class Distribution: {summary['class_distribution']}")
        print()
        
        # Performance comparison
        cla = results['cla_results']
        aesmote = results['aesmote_results']
        improvement = results['improvement_analysis']
        
        print("PERFORMANCE COMPARISON:")
        print(f"{'Metric':<15} {'Current CLA':<12} {'AESMOTE':<12} {'Improvement':<12}")
        print("-" * 60)
        print(f"{'F1-Score':<15} {cla['f1_score']:<12.3f} {aesmote['f1_score']:<12.3f} {improvement['f1_score_improvement']:<+12.3f}")
        print(f"{'Precision':<15} {cla['precision']:<12.3f} {aesmote['precision']:<12.3f} {improvement['precision_improvement']:<+12.3f}")
        print(f"{'Recall':<15} {cla['recall']:<12.3f} {aesmote['recall']:<12.3f} {improvement['recall_improvement']:<+12.3f}")
        print(f"{'Accuracy':<15} {cla['accuracy']:<12.3f} {aesmote['accuracy']:<12.3f} {'N/A':<12}")
        print()
        
        # Training time
        print(f"TRAINING TIME:")
        print(f"Current CLA: {cla['training_time']:.2f}s")
        print(f"AESMOTE: {aesmote['training_time']:.2f}s")
        print(f"Time Ratio: {improvement['training_time_ratio']:.2f}x")
        print()
        
        # Robustness
        robustness = results['robustness_results']
        print(f"ROBUSTNESS (Cross-Validation):")
        print(f"CLA F1-Score: {robustness['cla_f1_mean']:.3f} ± {robustness['cla_f1_std']:.3f}")
        print(f"AESMOTE F1-Score: {robustness['aesmote_f1_mean']:.3f} ± {robustness['aesmote_f1_std']:.3f}")
        print(f"Robustness Improvement: {robustness['improvement']:+.3f}")
        print()
        
        # Recommendation
        print("RECOMMENDATION:")
        if improvement['significant_improvement']:
            print("✅ AESMOTE shows significant improvement - worth implementing")
        elif improvement['f1_score_improvement'] > 0:
            print("⚠️  AESMOTE shows minor improvement - consider for edge cases")
        else:
            print("❌ AESMOTE does not improve performance - current CLA is optimal")
        
        print("="*70)
        
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    main()


