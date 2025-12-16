#!/usr/bin/env python3
"""
Generate Phase 1 Demo Data for BigQuery
Creates realistic sample data to demonstrate Phase 1 success metrics
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

class Phase1DemoDataGenerator:
    """Generate realistic demo data for Phase 1 success measurement"""
    
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
    
    def generate_processed_alerts_data(self, days: int = 30, alerts_per_day: int = 50):
        """Generate realistic processed alerts data with embeddings"""
        print(f"🔄 Generating {days} days of processed alerts data...")
        
        if not self.client:
            print("❌ BigQuery client not available")
            return False
        
        try:
            # Generate data for each day
            all_alerts = []
            base_time = datetime.now() - timedelta(days=days)
            
            for day in range(days):
                current_date = base_time + timedelta(days=day)
                
                # Generate alerts for this day
                daily_alerts = self._generate_daily_alerts(current_date, alerts_per_day)
                all_alerts.extend(daily_alerts)
            
            # Convert to DataFrame
            df = pd.DataFrame(all_alerts)
            
            # Insert into BigQuery
            table_id = f"{self.project_id}.soc_data.processed_alerts"
            job_config = bigquery.LoadJobConfig(
                write_disposition="WRITE_TRUNCATE",  # Replace existing data
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
            
            print(f"✅ Generated {len(all_alerts)} alerts across {days} days")
            return True
            
        except Exception as e:
            print(f"❌ Error generating processed alerts data: {e}")
            return False
    
    def _generate_daily_alerts(self, date: datetime, count: int):
        """Generate alerts for a specific day"""
        alerts = []
        
        # Define alert patterns for realistic data
        alert_patterns = {
            'high_similarity': {
                'count': int(count * 0.3),  # 30% high similarity
                'base_similarity': 0.85,
                'similarity_variance': 0.1,
                'processing_time': 15,
                'classifications': ['suspicious_login', 'network_scan', 'malware_detected']
            },
            'medium_similarity': {
                'count': int(count * 0.4),  # 40% medium similarity
                'base_similarity': 0.65,
                'similarity_variance': 0.15,
                'processing_time': 25,
                'classifications': ['unusual_activity', 'policy_violation', 'access_anomaly']
            },
            'low_similarity': {
                'count': int(count * 0.2),  # 20% low similarity
                'base_similarity': 0.45,
                'similarity_variance': 0.1,
                'processing_time': 35,
                'classifications': ['data_exfiltration', 'privilege_escalation', 'lateral_movement']
            },
            'unique': {
                'count': int(count * 0.1),  # 10% unique
                'base_similarity': 0.25,
                'similarity_variance': 0.1,
                'processing_time': 45,
                'classifications': ['zero_day', 'advanced_persistent_threat', 'insider_threat']
            }
        }
        
        alert_id_counter = 1
        
        for pattern_name, pattern_config in alert_patterns.items():
            for i in range(pattern_config['count']):
                # Generate timestamp within the day
                hour = random.randint(0, 23)
                minute = random.randint(0, 59)
                second = random.randint(0, 59)
                alert_time = date.replace(hour=hour, minute=minute, second=second)
                
                # Generate embedding timestamp (after alert time)
                embedding_delay = random.randint(5, 60)  # 5-60 seconds delay
                embedding_time = alert_time + timedelta(seconds=embedding_delay)
                
                # Generate similarity score
                similarity = max(0.0, min(1.0, 
                    pattern_config['base_similarity'] + 
                    random.uniform(-pattern_config['similarity_variance'], pattern_config['similarity_variance'])
                ))
                
                # Generate embedding vector (768 dimensions)
                embedding = self._generate_embedding_vector(similarity, pattern_name)
                
                # Generate RL reward score based on similarity
                rl_reward = self._calculate_rl_reward(similarity, pattern_name)
                
                # Generate similarity cluster
                similarity_cluster = f"{pattern_name}_{int(similarity * 100)}"
                
                alert = {
                    'alert_id': f"ALERT_{date.strftime('%Y%m%d')}_{alert_id_counter:04d}",
                    'timestamp': alert_time,  # Use datetime object directly
                    'classification': random.choice(pattern_config['classifications']),
                    'confidence_score': round(random.uniform(0.6, 0.95), 3),
                    'is_anomaly': random.choice([True, False]),
                    'raw_alert': json.dumps({
                        'event_type': 'security_alert',
                        'source_ip': f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}",
                        'dest_ip': f"10.0.{random.randint(1, 10)}.{random.randint(1, 255)}",
                        'user': f"user_{random.randint(1000, 9999)}",
                        'severity': random.choice(['low', 'medium', 'high', 'critical'])
                    }),
                    'embedding': embedding,
                    'embedding_timestamp': embedding_time,  # Use datetime object directly
                    'embedding_model': 'text-embedding-004',
                    'embedding_similarity': round(similarity, 3),
                    'rl_reward_score': round(rl_reward, 3),
                    'similarity_cluster': similarity_cluster
                }
                
                alerts.append(alert)
                alert_id_counter += 1
        
        return alerts
    
    def _generate_embedding_vector(self, similarity: float, pattern: str):
        """Generate a realistic 768-dimensional embedding vector"""
        # Create base vector with some structure
        base_vector = np.random.normal(0, 0.1, 768)
        
        # Add pattern-specific characteristics
        if pattern == 'high_similarity':
            # High similarity alerts have similar vectors
            base_vector += np.random.normal(0.5, 0.2, 768)
        elif pattern == 'medium_similarity':
            base_vector += np.random.normal(0.3, 0.3, 768)
        elif pattern == 'low_similarity':
            base_vector += np.random.normal(0.1, 0.4, 768)
        else:  # unique
            base_vector += np.random.normal(0, 0.5, 768)
        
        # Normalize to unit vector
        norm = np.linalg.norm(base_vector)
        if norm > 0:
            base_vector = base_vector / norm
        
        # Scale by similarity
        base_vector = base_vector * similarity
        
        return base_vector.tolist()
    
    def _calculate_rl_reward(self, similarity: float, pattern: str):
        """Calculate RL reward score based on similarity and pattern"""
        base_reward = similarity * 0.8  # Base reward from similarity
        
        # Pattern-specific adjustments
        pattern_bonus = {
            'high_similarity': 0.2,  # Bonus for clustering similar alerts
            'medium_similarity': 0.1,
            'low_similarity': 0.0,
            'unique': -0.1  # Slight penalty for unique alerts (harder to process)
        }
        
        reward = base_reward + pattern_bonus.get(pattern, 0)
        return max(0.0, min(1.0, reward))  # Clamp between 0 and 1
    
    def generate_historical_trends(self):
        """Generate historical trends showing Phase 1 improvement over time"""
        print("📈 Generating historical trends showing Phase 1 improvement...")
        
        # This would generate data showing improvement over time
        # For now, we'll use the existing data generation with some modifications
        return True
    
    def run_demo_data_generation(self):
        """Run complete demo data generation"""
        print("🚀 Starting Phase 1 Demo Data Generation")
        print("=" * 50)
        
        if not self.client:
            print("❌ BigQuery client not available")
            return False
        
        # Generate 30 days of data
        success = self.generate_processed_alerts_data(days=30, alerts_per_day=50)
        
        if success:
            print("\n✅ Demo data generation complete!")
            print("📊 You can now run the Phase 1 success measurement")
            print("🔍 The KPI views will now have data to analyze")
            return True
        else:
            print("\n❌ Demo data generation failed")
            return False

def main():
    """Main function to generate demo data"""
    generator = Phase1DemoDataGenerator()
    generator.run_demo_data_generation()

if __name__ == "__main__":
    main()
