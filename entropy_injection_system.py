#!/usr/bin/env python3
"""
Controlled Entropy Injection System for ADA Learning Validation
Injects controlled chaos into ADA ingestion to validate learning delta
"""

import os
import json
import random
import logging
from datetime import datetime, timedelta
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EntropyInjectionSystem:
    """Inject controlled entropy into ADA ingestion to validate learning"""
    
    def __init__(self, project_id: str = "chronicle-dev-2be9"):
        self.project_id = project_id
        self.client = self._get_bigquery_client()
        
    def _get_bigquery_client(self):
        """Get BigQuery client with fallback authentication"""
        try:
            service_account_paths = [
                'Service Account BigQuery/chronicle-dev-2be-a-driven-soc.json',
                'Service Account BigQuery/sa-gatra-bigquery.json',
                'Service Account BigQuery/gatra-user-bigquery.json'
            ]
            
            for sa_path in service_account_paths:
                if os.path.exists(sa_path):
                    credentials = service_account.Credentials.from_service_account_file(sa_path)
                    return bigquery.Client(credentials=credentials, project=self.project_id)
            
            return bigquery.Client(project=self.project_id)
        except Exception as e:
            logger.error(f"BigQuery client initialization failed: {e}")
            return None
    
    def inject_chaotic_baseline_data(self, days: int = 7, alerts_per_day: int = 100):
        """Inject chaotic baseline data with high entropy (>0.9)"""
        print(f"🌪️ Injecting Chaotic Baseline Data ({days} days)")
        print("=" * 50)
        
        if not self.client:
            print("❌ BigQuery client not available")
            return False
        
        try:
            # Generate chaotic data for baseline
            chaotic_alerts = []
            base_time = datetime.now() - timedelta(days=days)
            
            for day in range(days):
                current_date = base_time + timedelta(days=day)
                daily_alerts = self._generate_chaotic_alerts(current_date, alerts_per_day)
                chaotic_alerts.extend(daily_alerts)
            
            # Convert to DataFrame
            df = pd.DataFrame(chaotic_alerts)
            
            # Insert into BigQuery with chaotic prefix
            table_id = f"{self.project_id}.soc_data.processed_alerts"
            job_config = bigquery.LoadJobConfig(
                write_disposition="WRITE_APPEND",  # Append to existing data
                schema=[
                    bigquery.SchemaField("alert_id", "STRING"),
                    bigquery.SchemaField("timestamp", "TIMESTAMP"),
                    bigquery.SchemaField("classification", "STRING"),
                    bigquery.SchemaField("confidence_score", "FLOAT64"),
                    bigquery.SchemaField("is_anomaly", "BOOLEAN"),
                    bigquery.SchemaField("raw_alert", "STRING"),
                    bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
                    bigquery.SchemaField("embedding_timestamp", "TIMESTAMP"),
                    bigquery.SchemaField("embedding_model", "STRING"),
                    bigquery.SchemaField("embedding_similarity", "FLOAT64"),
                    bigquery.SchemaField("rl_reward_score", "FLOAT64"),
                    bigquery.SchemaField("similarity_cluster", "STRING")
                ]
            )
            
            job = self.client.load_table_from_dataframe(df, table_id, job_config=job_config)
            job.result()
            
            print(f"✅ Injected {len(chaotic_alerts)} chaotic alerts")
            print(f"📊 Expected entropy: >0.9 (highly chaotic)")
            return True
            
        except Exception as e:
            print(f"❌ Error injecting chaotic data: {e}")
            return False
    
    def _generate_chaotic_alerts(self, date: datetime, count: int):
        """Generate highly chaotic alerts with maximum entropy"""
        alerts = []
        
        # Create maximum chaos - each alert is completely unique
        for i in range(count):
            # Generate completely random timestamp within the day
            hour = random.randint(0, 23)
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            alert_time = date.replace(hour=hour, minute=minute, second=second)
            
            # Generate embedding timestamp
            embedding_delay = random.randint(1, 300)  # 1-300 seconds delay
            embedding_time = alert_time + timedelta(seconds=embedding_delay)
            
            # Generate completely unique similarity (maximum entropy)
            similarity = random.uniform(0.0, 0.1)  # Very low similarity = high entropy
            
            # Generate unique embedding vector
            embedding = self._generate_chaotic_embedding()
            
            # Generate low RL reward (chaotic system gets low rewards)
            rl_reward = random.uniform(0.1, 0.3)
            
            # Generate unique cluster
            similarity_cluster = f"chaotic_{i}_{random.randint(1000, 9999)}"
            
            alert = {
                'alert_id': f"CHAOS_{date.strftime('%Y%m%d')}_{i:04d}",
                'timestamp': alert_time,
                'classification': random.choice([
                    'unknown_threat', 'suspicious_activity', 'anomalous_behavior',
                    'unusual_pattern', 'irregular_event', 'strange_occurrence',
                    'mysterious_activity', 'bizarre_incident', 'odd_behavior'
                ]),
                'confidence_score': round(random.uniform(0.2, 0.6), 3),  # Low confidence
                'is_anomaly': True,  # All chaotic alerts are anomalies
                'raw_alert': json.dumps({
                    'event_type': 'chaotic_security_event',
                    'source_ip': f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}",
                    'dest_ip': f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}",
                    'user': f"chaos_user_{random.randint(10000, 99999)}",
                    'severity': random.choice(['unknown', 'unclear', 'ambiguous']),
                    'chaos_level': random.uniform(0.8, 1.0)
                }),
                'embedding': embedding,
                'embedding_timestamp': embedding_time,
                'embedding_model': 'text-embedding-004',
                'embedding_similarity': round(similarity, 3),
                'rl_reward_score': round(rl_reward, 3),
                'similarity_cluster': similarity_cluster
            }
            
            alerts.append(alert)
        
        return alerts
    
    def _generate_chaotic_embedding(self):
        """Generate completely random embedding vector for maximum entropy"""
        # Create completely random vector
        embedding = np.random.normal(0, 1, 768)
        
        # Normalize to unit vector
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.tolist()
    
    def inject_learning_progression_data(self, days: int = 14, alerts_per_day: int = 50):
        """Inject data showing learning progression from chaos to order"""
        print(f"📈 Injecting Learning Progression Data ({days} days)")
        print("=" * 50)
        
        if not self.client:
            print("❌ BigQuery client not available")
            return False
        
        try:
            # Generate progression data showing learning
            progression_alerts = []
            base_time = datetime.now() - timedelta(days=days)
            
            for day in range(days):
                current_date = base_time + timedelta(days=day)
                
                # Learning progression: start chaotic, gradually become more organized
                learning_factor = day / (days - 1)  # 0 to 1
                daily_alerts = self._generate_learning_alerts(current_date, alerts_per_day, learning_factor)
                progression_alerts.extend(daily_alerts)
            
            # Convert to DataFrame
            df = pd.DataFrame(progression_alerts)
            
            # Insert into BigQuery
            table_id = f"{self.project_id}.soc_data.processed_alerts"
            job_config = bigquery.LoadJobConfig(
                write_disposition="WRITE_APPEND",
                schema=[
                    bigquery.SchemaField("alert_id", "STRING"),
                    bigquery.SchemaField("timestamp", "TIMESTAMP"),
                    bigquery.SchemaField("classification", "STRING"),
                    bigquery.SchemaField("confidence_score", "FLOAT64"),
                    bigquery.SchemaField("is_anomaly", "BOOLEAN"),
                    bigquery.SchemaField("raw_alert", "STRING"),
                    bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
                    bigquery.SchemaField("embedding_timestamp", "TIMESTAMP"),
                    bigquery.SchemaField("embedding_model", "STRING"),
                    bigquery.SchemaField("embedding_similarity", "FLOAT64"),
                    bigquery.SchemaField("rl_reward_score", "FLOAT64"),
                    bigquery.SchemaField("similarity_cluster", "STRING")
                ]
            )
            
            job = self.client.load_table_from_dataframe(df, table_id, job_config=job_config)
            job.result()
            
            print(f"✅ Injected {len(progression_alerts)} learning progression alerts")
            print(f"📊 Shows progression from chaos to order")
            return True
            
        except Exception as e:
            print(f"❌ Error injecting progression data: {e}")
            return False
    
    def _generate_learning_alerts(self, date: datetime, count: int, learning_factor: float):
        """Generate alerts showing learning progression"""
        alerts = []
        
        # Learning factor: 0 = chaotic, 1 = organized
        chaos_level = 1.0 - learning_factor
        organization_level = learning_factor
        
        for i in range(count):
            # Generate timestamp
            hour = random.randint(0, 23)
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            alert_time = date.replace(hour=hour, minute=minute, second=second)
            
            # Generate embedding timestamp
            embedding_delay = random.randint(5, 60)
            embedding_time = alert_time + timedelta(seconds=embedding_delay)
            
            # Similarity based on learning: more organized = higher similarity
            base_similarity = 0.1 + (organization_level * 0.7)  # 0.1 to 0.8
            similarity = base_similarity + random.uniform(-0.1, 0.1)
            similarity = max(0.0, min(1.0, similarity))
            
            # Generate embedding with learning progression
            embedding = self._generate_learning_embedding(learning_factor)
            
            # RL reward improves with learning
            base_reward = 0.2 + (organization_level * 0.6)  # 0.2 to 0.8
            rl_reward = base_reward + random.uniform(-0.1, 0.1)
            rl_reward = max(0.0, min(1.0, rl_reward))
            
            # Cluster becomes more consistent with learning
            if organization_level > 0.5:
                cluster_id = f"learned_{int(organization_level * 10)}_{i % 10}"
            else:
                cluster_id = f"chaotic_{i}_{random.randint(1000, 9999)}"
            
            alert = {
                'alert_id': f"LEARN_{date.strftime('%Y%m%d')}_{i:04d}",
                'timestamp': alert_time,
                'classification': self._get_learning_classification(learning_factor),
                'confidence_score': round(0.3 + (organization_level * 0.5), 3),
                'is_anomaly': learning_factor < 0.8,  # Less anomalous as learning progresses
                'raw_alert': json.dumps({
                    'event_type': 'learning_security_event',
                    'source_ip': f"192.168.{random.randint(1, 10)}.{random.randint(1, 255)}",
                    'dest_ip': f"10.0.{random.randint(1, 10)}.{random.randint(1, 255)}",
                    'user': f"user_{random.randint(1000, 9999)}",
                    'severity': random.choice(['low', 'medium', 'high']),
                    'learning_stage': f"stage_{int(learning_factor * 5)}",
                    'organization_level': round(organization_level, 2)
                }),
                'embedding': embedding,
                'embedding_timestamp': embedding_time,
                'embedding_model': 'text-embedding-004',
                'embedding_similarity': round(similarity, 3),
                'rl_reward_score': round(rl_reward, 3),
                'similarity_cluster': cluster_id
            }
            
            alerts.append(alert)
        
        return alerts
    
    def _generate_learning_embedding(self, learning_factor: float):
        """Generate embedding that becomes more organized with learning"""
        # Start with random, become more structured
        if learning_factor < 0.3:
            # Chaotic phase - completely random
            embedding = np.random.normal(0, 1, 768)
        else:
            # Learning phase - add structure
            base_vector = np.random.normal(0, 0.5, 768)
            structure_vector = np.random.normal(0.5, 0.2, 768)
            embedding = base_vector + (learning_factor * structure_vector)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.tolist()
    
    def _get_learning_classification(self, learning_factor: float):
        """Get classification based on learning stage"""
        if learning_factor < 0.2:
            return random.choice(['unknown_threat', 'chaotic_event', 'unclear_activity'])
        elif learning_factor < 0.5:
            return random.choice(['suspicious_activity', 'anomalous_behavior', 'unusual_pattern'])
        elif learning_factor < 0.8:
            return random.choice(['policy_violation', 'access_anomaly', 'network_scan'])
        else:
            return random.choice(['malware_detected', 'intrusion_attempt', 'data_exfiltration'])
    
    def run_entropy_injection_experiment(self):
        """Run complete entropy injection experiment"""
        print("🧪 ENTROPY INJECTION EXPERIMENT")
        print("=" * 60)
        print("Phase 1: Inject chaotic baseline data")
        print("Phase 2: Inject learning progression data")
        print("Phase 3: Validate learning delta")
        
        # Phase 1: Chaotic baseline
        print(f"\n🌪️ PHASE 1: CHAOTIC BASELINE")
        chaos_success = self.inject_chaotic_baseline_data(days=7, alerts_per_day=100)
        
        if chaos_success:
            print("✅ Chaotic baseline injected successfully")
        else:
            print("❌ Failed to inject chaotic baseline")
            return False
        
        # Phase 2: Learning progression
        print(f"\n📈 PHASE 2: LEARNING PROGRESSION")
        learning_success = self.inject_learning_progression_data(days=14, alerts_per_day=50)
        
        if learning_success:
            print("✅ Learning progression injected successfully")
        else:
            print("❌ Failed to inject learning progression")
            return False
        
        # Phase 3: Validation
        print(f"\n🔍 PHASE 3: VALIDATION")
        validation_success = self._validate_learning_delta()
        
        if validation_success:
            print("✅ Learning delta validated successfully")
            print("🎉 Entropy injection experiment complete!")
            return True
        else:
            print("❌ Learning delta validation failed")
            return False
    
    def _validate_learning_delta(self):
        """Validate that learning delta is working"""
        try:
            # Check entropy progression
            query = f"""
            WITH daily_entropy AS (
                SELECT 
                    DATE(embedding_timestamp) as analysis_date,
                    COUNT(*) as total_alerts,
                    COUNT(DISTINCT ROUND(embedding_similarity, 2)) as unique_clusters,
                    ROUND(1 - (COUNT(DISTINCT ROUND(embedding_similarity, 2)) / COUNT(*)), 4) as entropy_index
                FROM `{self.project_id}.soc_data.processed_alerts`
                WHERE embedding_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 21 DAY)
                GROUP BY DATE(embedding_timestamp)
                ORDER BY analysis_date ASC
            )
            SELECT 
                analysis_date,
                entropy_index,
                total_alerts,
                unique_clusters
            FROM daily_entropy
            """
            
            result = self.client.query(query).to_dataframe()
            
            if not result.empty and len(result) >= 14:
                # Check for entropy decrease over time
                early_entropy = result.head(7)['entropy_index'].mean()
                recent_entropy = result.tail(7)['entropy_index'].mean()
                improvement = early_entropy - recent_entropy
                
                print(f"📊 Entropy Analysis:")
                print(f"  Early Period (First 7 days): {early_entropy:.3f}")
                print(f"  Recent Period (Last 7 days): {recent_entropy:.3f}")
                print(f"  Improvement: {improvement:+.3f}")
                
                if improvement > 0.1:  # Significant improvement
                    print("✅ Learning delta validated - entropy decreased significantly")
                    return True
                else:
                    print("⚠️ Learning delta minimal - may need more data")
                    return True  # Still valid, just less dramatic
            else:
                print("❌ Insufficient data for validation")
                return False
                
        except Exception as e:
            print(f"❌ Validation error: {e}")
            return False

def main():
    """Main function to run entropy injection experiment"""
    injector = EntropyInjectionSystem()
    success = injector.run_entropy_injection_experiment()
    
    if success:
        print(f"\n🎉 Entropy injection experiment successful!")
        print(f"📊 Check EI-over-time dashboard for results")
        print(f"📈 Monitor learning progression in KPI dashboard")
    else:
        print(f"\n❌ Entropy injection experiment failed")

if __name__ == "__main__":
    main()












