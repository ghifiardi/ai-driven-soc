#!/usr/bin/env python3
"""
Enhanced ADA Model Training Pipeline
Improves anomaly detection by training with diverse data patterns
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from google.cloud import bigquery
import pickle
import json
import os
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedADATrainer:
    def __init__(self):
        self.project_id = "chronicle-dev-2be9"
        self.dataset_id = "soc_data"
        self.table_id = "alerts"
        
        # Initialize BigQuery client
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/raditio.ghifiardigmail.com/Downloads/ai-driven-soc/chronicle-dev-2be9-6a50b8d0b3a0.json'
        self.client = bigquery.Client(project=self.project_id)
        
        # Model components
        self.scaler = StandardScaler()
        self.text_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.label_encoders = {}
        self.model = None
        
    def extract_enhanced_training_data(self, sample_size=50000):
        """Extract diverse training data with enhanced features"""
        logger.info(f"Extracting {sample_size} training samples...")
        
        # Get diverse data across different time periods and alert types
        query = f"""
        WITH diverse_alerts AS (
            SELECT 
                alert_id,
                timestamp,
                event_type,
                severity,
                source_ip,
                destination_ip,
                event_description,
                rule_name,
                user_name,
                asset_name,
                protocol,
                port,
                bytes_sent,
                bytes_received,
                packet_count,
                session_duration,
                -- Add time-based features
                EXTRACT(HOUR FROM timestamp) as hour_of_day,
                EXTRACT(DAYOFWEEK FROM timestamp) as day_of_week,
                -- Add frequency features
                COUNT(*) OVER (PARTITION BY source_ip, EXTRACT(HOUR FROM timestamp)) as source_ip_hourly_count,
                COUNT(*) OVER (PARTITION BY event_type, EXTRACT(HOUR FROM timestamp)) as event_type_hourly_count,
                COUNT(*) OVER (PARTITION BY rule_name, EXTRACT(HOUR FROM timestamp)) as rule_hourly_count,
                -- Add behavioral features
                LAG(timestamp) OVER (PARTITION BY source_ip ORDER BY timestamp) as prev_timestamp,
                LEAD(timestamp) OVER (PARTITION BY source_ip ORDER BY timestamp) as next_timestamp
            FROM `{self.project_id}.{self.dataset_id}.{self.table_id}`
            WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
            AND event_description IS NOT NULL
            AND event_description != ''
        ),
        enhanced_features AS (
            SELECT *,
                -- Calculate time gaps for behavioral analysis
                TIMESTAMP_DIFF(timestamp, prev_timestamp, SECOND) as time_since_prev,
                TIMESTAMP_DIFF(next_timestamp, timestamp, SECOND) as time_to_next,
                -- Create anomaly indicators based on patterns
                CASE 
                    WHEN source_ip_hourly_count > 1000 THEN 1  -- High frequency from single IP
                    WHEN bytes_sent > 1000000 THEN 1  -- Large data transfer
                    WHEN session_duration > 3600 THEN 1  -- Long sessions
                    WHEN event_type IN ('malware_detected', 'intrusion_attempt', 'data_exfiltration') THEN 1
                    WHEN severity IN ('HIGH', 'CRITICAL') THEN 1
                    WHEN rule_name LIKE '%attack%' OR rule_name LIKE '%malware%' OR rule_name LIKE '%breach%' THEN 1
                    WHEN time_since_prev < 1 AND source_ip_hourly_count > 100 THEN 1  -- Rapid fire from same IP
                    ELSE 0
                END as is_potential_anomaly
            FROM diverse_alerts
        )
        SELECT *
        FROM enhanced_features
        ORDER BY RAND()
        LIMIT {sample_size}
        """
        
        try:
            df = self.client.query(query).to_dataframe()
            logger.info(f"Extracted {len(df)} training samples")
            
            # Create synthetic anomalies for better training balance
            df = self.create_synthetic_anomalies(df)
            
            return df
        except Exception as e:
            logger.error(f"Error extracting training data: {e}")
            return None
    
    def create_synthetic_anomalies(self, df):
        """Create synthetic anomalous patterns for better model training"""
        logger.info("Creating synthetic anomalies for balanced training...")
        
        synthetic_anomalies = []
        normal_samples = df[df['is_potential_anomaly'] == 0].sample(n=min(1000, len(df)//10))
        
        for _, row in normal_samples.iterrows():
            # Create different types of synthetic anomalies
            
            # Type 1: High frequency anomaly
            anomaly1 = row.copy()
            anomaly1['source_ip_hourly_count'] = np.random.randint(2000, 5000)
            anomaly1['event_type_hourly_count'] = np.random.randint(1000, 3000)
            anomaly1['is_potential_anomaly'] = 1
            anomaly1['event_description'] = f"High frequency activity detected from {row['source_ip']}"
            synthetic_anomalies.append(anomaly1)
            
            # Type 2: Data exfiltration pattern
            anomaly2 = row.copy()
            anomaly2['bytes_sent'] = np.random.randint(5000000, 50000000)  # Large data transfer
            anomaly2['session_duration'] = np.random.randint(7200, 28800)  # Long session
            anomaly2['is_potential_anomaly'] = 1
            anomaly2['event_description'] = f"Large data transfer detected: {anomaly2['bytes_sent']} bytes"
            synthetic_anomalies.append(anomaly2)
            
            # Type 3: Suspicious timing pattern
            anomaly3 = row.copy()
            anomaly3['hour_of_day'] = np.random.choice([2, 3, 4])  # Unusual hours
            anomaly3['time_since_prev'] = np.random.uniform(0.1, 0.5)  # Very rapid succession
            anomaly3['is_potential_anomaly'] = 1
            anomaly3['event_description'] = f"Suspicious off-hours activity at {anomaly3['hour_of_day']}:00"
            synthetic_anomalies.append(anomaly3)
        
        # Add synthetic anomalies to dataset
        synthetic_df = pd.DataFrame(synthetic_anomalies)
        enhanced_df = pd.concat([df, synthetic_df], ignore_index=True)
        
        logger.info(f"Added {len(synthetic_anomalies)} synthetic anomalies")
        logger.info(f"Final dataset: {len(enhanced_df)} samples, {enhanced_df['is_potential_anomaly'].sum()} anomalies")
        
        return enhanced_df
    
    def engineer_features(self, df):
        """Create comprehensive feature set for anomaly detection"""
        logger.info("Engineering features...")
        
        features = []
        feature_names = []
        
        # Numerical features
        numerical_cols = ['source_ip_hourly_count', 'event_type_hourly_count', 'rule_hourly_count',
                         'bytes_sent', 'bytes_received', 'packet_count', 'session_duration',
                         'hour_of_day', 'day_of_week', 'time_since_prev', 'time_to_next']
        
        for col in numerical_cols:
            if col in df.columns:
                # Fill NaN values with median
                values = df[col].fillna(df[col].median()).values
                features.append(values.reshape(-1, 1))
                feature_names.append(col)
        
        # Categorical features
        categorical_cols = ['event_type', 'severity', 'protocol']
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    encoded = self.label_encoders[col].fit_transform(df[col].fillna('unknown'))
                else:
                    encoded = self.label_encoders[col].transform(df[col].fillna('unknown'))
                features.append(encoded.reshape(-1, 1))
                feature_names.append(f"{col}_encoded")
        
        # Text features from event descriptions
        if 'event_description' in df.columns:
            text_features = self.text_vectorizer.fit_transform(df['event_description'].fillna(''))
            features.append(text_features.toarray())
            feature_names.extend([f"text_feature_{i}" for i in range(text_features.shape[1])])
        
        # Combine all features
        if features:
            X = np.hstack(features)
            logger.info(f"Created feature matrix: {X.shape}")
            return X, feature_names
        else:
            logger.error("No features could be created")
            return None, None
    
    def train_enhanced_model(self, X, y):
        """Train an enhanced anomaly detection model"""
        logger.info("Training enhanced anomaly detection model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest for better interpretability and performance
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',  # Handle class imbalance
            random_state=42,
            n_jobs=-1
        )
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        logger.info(f"Model training completed:")
        logger.info(f"Training accuracy: {train_score:.3f}")
        logger.info(f"Test accuracy: {test_score:.3f}")
        
        # Detailed evaluation
        y_pred = self.model.predict(X_test_scaled)
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred))
        
        # Feature importance
        feature_importance = self.model.feature_importances_
        logger.info("\nTop 10 Most Important Features:")
        for i in np.argsort(feature_importance)[-10:]:
            logger.info(f"Feature {i}: {feature_importance[i]:.3f}")
        
        return train_score, test_score
    
    def save_enhanced_model(self, model_path="enhanced_ada_model.pkl"):
        """Save the enhanced model and preprocessing components"""
        logger.info(f"Saving enhanced model to {model_path}")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'text_vectorizer': self.text_vectorizer,
            'label_encoders': self.label_encoders,
            'training_timestamp': datetime.now().isoformat(),
            'model_type': 'enhanced_random_forest'
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info("Enhanced model saved successfully!")
        return model_path
    
    def run_enhanced_training(self):
        """Run the complete enhanced training pipeline"""
        logger.info("Starting enhanced ADA model training...")
        
        # Extract training data
        df = self.extract_enhanced_training_data()
        if df is None:
            logger.error("Failed to extract training data")
            return False
        
        # Engineer features
        X, feature_names = self.engineer_features(df)
        if X is None:
            logger.error("Failed to engineer features")
            return False
        
        # Prepare target variable
        y = df['is_potential_anomaly'].values
        
        # Train model
        train_score, test_score = self.train_enhanced_model(X, y)
        
        # Save model
        model_path = self.save_enhanced_model()
        
        logger.info("Enhanced training completed successfully!")
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Training accuracy: {train_score:.3f}")
        logger.info(f"Test accuracy: {test_score:.3f}")
        
        return True

def main():
    """Main training function"""
    trainer = EnhancedADATrainer()
    success = trainer.run_enhanced_training()
    
    if success:
        print("\n🎯 Enhanced ADA Model Training Completed!")
        print("✅ Model trained with diverse anomaly patterns")
        print("✅ Synthetic anomalies created for better detection")
        print("✅ Enhanced feature engineering applied")
        print("✅ Model saved and ready for deployment")
        print("\nNext step: Deploy the enhanced model to improve anomaly detection!")
    else:
        print("\n❌ Training failed. Check logs for details.")

if __name__ == "__main__":
    main()
