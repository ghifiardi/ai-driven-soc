#!/usr/bin/env python3
"""
AESMOTE CLA Trainer - Adversarial Reinforcement Learning with SMOTE for Anomaly Detection

This implementation combines:
1. Adversarial Reinforcement Learning (Classifier Agent vs Environment Agent)
2. SMOTE (Synthetic Minority Over-sampling Technique)
3. Double Deep Q-Network (DDQN) for stable learning
4. Huber Loss for robust training

Based on the AESMOTE framework for improving CLA AI Agent performance on imbalanced datasets.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("AESMOTECLA")

class DDQN(nn.Module):
    """Double Deep Q-Network for stable reinforcement learning"""
    
    def __init__(self, input_size, hidden_sizes, output_size):
        super(DDQN, self).__init__()
        
        # Build network layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class HuberLoss(nn.Module):
    """Huber Loss for robust training"""
    
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta
        
    def forward(self, y_pred, y_true):
        diff = torch.abs(y_pred - y_true)
        loss = torch.where(
            diff < self.delta,
            0.5 * diff ** 2,
            self.delta * diff - 0.5 * self.delta ** 2
        )
        return torch.mean(loss)

class ClassifierAgent:
    """Classifier Agent - Makes final predictions and receives rewards"""
    
    def __init__(self, input_size, hidden_sizes=[128, 64], num_classes=2, learning_rate=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Main network
        self.q_network = DDQN(input_size, hidden_sizes, num_classes).to(self.device)
        self.target_network = DDQN(input_size, hidden_sizes, num_classes).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = HuberLoss()
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
        # Training parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.update_target_freq = 100
        self.step_count = 0
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and np.random.random() <= self.epsilon:
            return random.randrange(2)  # Random action
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self):
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
            
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            # Double DQN: use main network to select action, target network to evaluate
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            target_q_values = rewards.unsqueeze(1) + (0.99 * next_q_values * ~dones.unsqueeze(1))
        
        loss = self.criterion(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.update_target_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

class EnvironmentAgent:
    """Environment Agent - Selects training samples and receives opposite rewards"""
    
    def __init__(self, input_size, hidden_sizes=[128, 64], learning_rate=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Network for sample selection
        self.q_network = DDQN(input_size, hidden_sizes, 1).to(self.device)  # Binary: select or not
        self.target_network = DDQN(input_size, hidden_sizes, 1).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = HuberLoss()
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
        # Training parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.update_target_freq = 100
        self.step_count = 0
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
        
    def select_sample(self, samples, training=True):
        """Select which samples to use for training (adversarial selection)"""
        selected_indices = []
        
        for i, sample in enumerate(samples):
            if training and np.random.random() <= self.epsilon:
                # Random selection
                if np.random.random() > 0.5:
                    selected_indices.append(i)
            else:
                # Network-based selection
                state_tensor = torch.FloatTensor(sample).unsqueeze(0).to(self.device)
                q_value = self.q_network(state_tensor)
                if q_value.item() > 0.5:  # Threshold for selection
                    selected_indices.append(i)
        
        return selected_indices
    
    def replay(self):
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
            
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            target_q_values = rewards.unsqueeze(1) + (0.99 * next_q_values * ~dones.unsqueeze(1))
        
        loss = self.criterion(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.update_target_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

class AESMOTECLATrainer:
    """AESMOTE CLA Trainer - Adversarial Reinforcement Learning with SMOTE"""
    
    def __init__(self, config_path: str = "config/cla_config.json"):
        """Initialize the AESMOTE CLA Trainer"""
        self.config = self._load_config(config_path)
        self.bigquery_client = bigquery.Client(project=self.config["project_id"])
        
        # Training parameters
        self.episodes = 200
        self.max_samples_per_episode = 1000
        
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
            LIMIT 20000
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
            
            # Protocol features (simplified for AESMOTE)
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
    
    def apply_smote(self, X_train, y_train, k_neighbors=5):
        """Apply SMOTE for synthetic minority oversampling"""
        try:
            # Calculate minority class ratio
            minority_ratio = (y_train == 1).sum() / len(y_train)
            majority_ratio = (y_train == 0).sum() / len(y_train)
            
            logger.info(f"Before SMOTE - Minority: {minority_ratio:.3f}, Majority: {majority_ratio:.3f}")
            
            # Apply SMOTE
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
            
            # Log results
            new_minority_ratio = (y_balanced == 1).sum() / len(y_balanced)
            new_majority_ratio = (y_balanced == 0).sum() / len(y_balanced)
            
            logger.info(f"After SMOTE - Minority: {new_minority_ratio:.3f}, Majority: {new_majority_ratio:.3f}")
            logger.info(f"SMOTE generated {len(X_balanced) - len(X_train)} synthetic samples")
            
            return X_balanced, y_balanced
            
        except Exception as e:
            logger.warning(f"SMOTE failed: {e}, using original data")
            return X_train, y_train
    
    def train_aesmote_model(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, Dict]:
        """Train AESMOTE model with adversarial reinforcement learning"""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            logger.info(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
            
            # Apply SMOTE
            X_balanced, y_balanced = self.apply_smote(X_train, y_train)
            
            # Normalize features for neural networks
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_balanced)
            X_test_scaled = scaler.transform(X_test)
            
            # Initialize agents
            input_size = X_scaled.shape[1]
            classifier_agent = ClassifierAgent(input_size)
            environment_agent = EnvironmentAgent(input_size)
            
            logger.info("Starting AESMOTE adversarial training...")
            
            # Training episodes
            episode_rewards = []
            classifier_performance = []
            
            for episode in range(self.episodes):
                episode_reward = 0
                correct_predictions = 0
                total_predictions = 0
                
                # Sample a batch for this episode
                batch_size = min(self.max_samples_per_episode, len(X_scaled))
                batch_indices = np.random.choice(len(X_scaled), batch_size, replace=False)
                batch_X = X_scaled[batch_indices]
                batch_y = y_balanced[batch_indices]
                
                # Environment Agent selects difficult samples
                selected_indices = environment_agent.select_sample(batch_X, training=True)
                
                if not selected_indices:
                    continue
                
                selected_X = batch_X[selected_indices]
                selected_y = batch_y[selected_indices]
                
                # Train Classifier Agent on selected samples
                for i, (sample, true_label) in enumerate(zip(selected_X, selected_y)):
                    # Classifier makes prediction
                    prediction = classifier_agent.act(sample, training=True)
                    
                    # Calculate rewards
                    is_correct = (prediction == true_label)
                    classifier_reward = 1.0 if is_correct else -1.0
                    environment_reward = -classifier_reward  # Opposite reward
                    
                    # Update counters
                    if is_correct:
                        correct_predictions += 1
                    total_predictions += 1
                    episode_reward += classifier_reward
                    
                    # Store experiences
                    next_state = sample  # Simplified for this implementation
                    classifier_agent.remember(sample, prediction, classifier_reward, next_state, False)
                    environment_agent.remember(sample, 1 if i in selected_indices else 0, environment_reward, next_state, False)
                
                # Train both agents
                classifier_agent.replay()
                environment_agent.replay()
                
                # Log progress
                episode_rewards.append(episode_reward)
                accuracy = correct_predictions / max(total_predictions, 1)
                classifier_performance.append(accuracy)
                
                if episode % 50 == 0:
                    logger.info(f"Episode {episode}: Reward={episode_reward:.2f}, Accuracy={accuracy:.3f}")
            
            # Final evaluation
            logger.info("Evaluating final AESMOTE model...")
            
            # Create final classifier using the trained agent
            final_predictions = []
            for sample in X_test_scaled:
                prediction = classifier_agent.act(sample, training=False)
                final_predictions.append(prediction)
            
            final_predictions = np.array(final_predictions)
            
            # Calculate metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, final_predictions, average='binary', zero_division=0
            )
            
            cm = confusion_matrix(y_test, final_predictions)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                tn = fp = fn = tp = 0
            
            metrics = {
                "model_name": "aesmote_cla",
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "accuracy": float(accuracy_score(y_test, final_predictions)),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
                "training_samples": len(X_balanced),
                "test_samples": len(X_test),
                "episodes_trained": self.episodes,
                "final_epsilon_ca": classifier_agent.epsilon,
                "final_epsilon_ea": environment_agent.epsilon,
                "avg_episode_reward": float(np.mean(episode_rewards[-50:])) if episode_rewards else 0.0,
                "avg_accuracy": float(np.mean(classifier_performance[-50:])) if classifier_performance else 0.0
            }
            
            logger.info(f"AESMOTE training completed. F1 Score: {f1:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")
            
            return (classifier_agent, scaler), metrics
            
        except Exception as e:
            logger.error(f"Error training AESMOTE model: {e}")
            raise
    
    def save_aesmote_model(self, model_data, metrics: Dict) -> str:
        """Save the AESMOTE model and metrics"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"models/aesmote_model_{timestamp}.pkl"
            metrics_filename = f"models/aesmote_metrics_{timestamp}.json"
            
            os.makedirs("models", exist_ok=True)
            
            # Save model (classifier agent and scaler)
            classifier_agent, scaler = model_data
            with open(model_filename, 'wb') as f:
                pickle.dump((classifier_agent, scaler), f)
            
            # Save metrics
            with open(metrics_filename, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"AESMOTE model saved to {model_filename}")
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
            logger.info("Starting AESMOTE CLA training...")
            
            # Gather data
            training_data = self.gather_training_data()
            if training_data is None or len(training_data) < 1000:
                return {"status": "insufficient_data", "message": f"Only {len(training_data) if training_data is not None else 0} samples"}
            
            # Prepare features
            X, y = self.prepare_features(training_data)
            
            # Train AESMOTE model
            model_data, metrics = self.train_aesmote_model(X, y)
            
            # Save model
            model_path = self.save_aesmote_model(model_data, metrics)
            
            logger.info("AESMOTE CLA training completed successfully!")
            return {
                "status": "success",
                "model_path": model_path,
                "metrics": metrics
            }
            
        except Exception as e:
            logger.error(f"AESMOTE training failed: {e}")
            return {"status": "error", "message": str(e)}

def main():
    """Main function to run AESMOTE training"""
    try:
        trainer = AESMOTECLATrainer()
        result = trainer.run_aesmote_training()
        
        print("\n" + "="*70)
        print("AESMOTE CLA TRAINING RESULTS")
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
            print(f"Average Episode Reward: {metrics['avg_episode_reward']:.2f}")
            print(f"Average Accuracy: {metrics['avg_accuracy']:.1%}")
        else:
            print(f"Error: {result['message']}")
        
        print("="*70)
        
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        print(f"ERROR: {e}")

if __name__ == "__main__":
    main()
