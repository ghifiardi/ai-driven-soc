#!/usr/bin/env python3
"""
BigQuery Schema Migration for Embedding Support
Adds vector columns to support 768-dimensional embeddings
"""

import os
import json
import logging
from typing import List, Dict, Any
from google.cloud import bigquery
from google.oauth2 import service_account

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BigQuerySchemaMigration:
    """Handles BigQuery schema migration for embedding support"""
    
    def __init__(self, project_id: str = "chronicle-dev-2be9"):
        """Initialize the migration service"""
        self.project_id = project_id
        self.client = self._get_bigquery_client()
        
    def _get_bigquery_client(self):
        """Get BigQuery client with fallback authentication"""
        try:
            # Try multiple service account files
            service_account_paths = [
                'Service Account BigQuery/chronicle-dev-2be-a-driven-soc.json',
                'Service Account BigQuery/sa-gatra-bigquery.json',
                'Service Account BigQuery/gatra-user-bigquery.json'
            ]
            
            for sa_path in service_account_paths:
                if os.path.exists(sa_path):
                    credentials = service_account.Credentials.from_service_account_file(sa_path)
                    return bigquery.Client(credentials=credentials, project=self.project_id)
            
            # Fallback to default client
            return bigquery.Client(project=self.project_id)
        except Exception as e:
            logger.error(f"BigQuery client initialization failed: {e}")
            return None
    
    def add_embedding_columns(self, dataset_id: str = "soc_data", table_id: str = "processed_alerts"):
        """Add embedding columns to the processed_alerts table
        
        Args:
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID
        """
        try:
            table_ref = self.client.dataset(dataset_id).table(table_id)
            table = self.client.get_table(table_ref)
            
            # Define new columns for embeddings
            new_columns = [
                bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED", description="768-dimensional embedding vector"),
                bigquery.SchemaField("embedding_timestamp", "TIMESTAMP", mode="NULLABLE", description="When embedding was generated"),
                bigquery.SchemaField("embedding_model", "STRING", mode="NULLABLE", description="Model used for embedding generation"),
                bigquery.SchemaField("embedding_similarity", "FLOAT64", mode="NULLABLE", description="Similarity score for RL feedback"),
                bigquery.SchemaField("rl_reward_score", "FLOAT64", mode="NULLABLE", description="Reinforcement learning reward score")
            ]
            
            # Add new columns to existing schema
            original_schema = list(table.schema)
            updated_schema = original_schema + new_columns
            
            # Update table schema
            table.schema = updated_schema
            table = self.client.update_table(table, ["schema"])
            
            logger.info(f"Successfully added embedding columns to {dataset_id}.{table_id}")
            logger.info(f"New columns: {[col.name for col in new_columns]}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add embedding columns: {e}")
            return False
    
    def create_embedding_indexes(self, dataset_id: str = "soc_data", table_id: str = "processed_alerts"):
        """Create indexes for embedding similarity queries
        
        Args:
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID
        """
        try:
            # Note: BigQuery doesn't support traditional indexes, but we can create clustering
            # and partitioning for better performance with vector operations
            
            table_ref = self.client.dataset(dataset_id).table(table_id)
            table = self.client.get_table(table_ref)
            
            # Update table with clustering on embedding_timestamp for better performance
            table.clustering_fields = ["embedding_timestamp", "classification"]
            
            # Update the table
            table = self.client.update_table(table, ["clustering_fields"])
            
            logger.info(f"Successfully updated clustering for {dataset_id}.{table_id}")
            logger.info(f"Clustering fields: {table.clustering_fields}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create embedding indexes: {e}")
            return False
    
    def create_embedding_views(self, dataset_id: str = "soc_data"):
        """Create views for embedding analysis
        
        Args:
            dataset_id: BigQuery dataset ID
        """
        try:
            # View for embedding similarity analysis
            similarity_view_query = f"""
            CREATE OR REPLACE VIEW `{self.project_id}.{dataset_id}.embedding_similarity_analysis` AS
            SELECT 
                alert_id,
                timestamp,
                classification,
                confidence_score,
                embedding_timestamp,
                embedding_model,
                embedding_similarity,
                rl_reward_score,
                ARRAY_LENGTH(embedding) as embedding_dimensions
            FROM `{self.project_id}.{dataset_id}.processed_alerts`
            WHERE embedding IS NOT NULL
            ORDER BY embedding_timestamp DESC
            """
            
            # Execute the view creation
            job = self.client.query(similarity_view_query)
            job.result()  # Wait for completion
            
            logger.info(f"Successfully created embedding similarity analysis view")
            
            # View for RL feedback metrics
            rl_metrics_view_query = f"""
            CREATE OR REPLACE VIEW `{self.project_id}.{dataset_id}.rl_feedback_metrics` AS
            SELECT 
                DATE(embedding_timestamp) as date,
                classification,
                AVG(rl_reward_score) as avg_reward_score,
                AVG(embedding_similarity) as avg_similarity,
                COUNT(*) as total_alerts,
                COUNT(CASE WHEN rl_reward_score > 0.7 THEN 1 END) as high_reward_alerts
            FROM `{self.project_id}.{dataset_id}.processed_alerts`
            WHERE embedding_timestamp IS NOT NULL
            GROUP BY DATE(embedding_timestamp), classification
            ORDER BY date DESC, avg_reward_score DESC
            """
            
            # Execute the RL metrics view creation
            job = self.client.query(rl_metrics_view_query)
            job.result()  # Wait for completion
            
            logger.info(f"Successfully created RL feedback metrics view")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create embedding views: {e}")
            return False
    
    def test_embedding_insert(self, dataset_id: str = "soc_data", table_id: str = "processed_alerts"):
        """Test inserting data with embeddings
        
        Args:
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID
        """
        try:
            # Create test data with embedding
            test_data = {
                "alert_id": "test_embedding_001",
                "timestamp": "2024-01-01T10:00:00Z",
                "classification": "test",
                "confidence_score": 0.85,
                "is_anomaly": True,
                "raw_alert": '{"test": "data"}',
                "embedding": [0.1] * 768,  # Dummy 768-dimensional vector
                "embedding_timestamp": "2024-01-01T10:00:01Z",
                "embedding_model": "text-embedding-004",
                "embedding_similarity": 0.75,
                "rl_reward_score": 0.8
            }
            
            # Insert test data
            table_ref = self.client.dataset(dataset_id).table(table_id)
            errors = self.client.insert_rows_json(table_ref, [test_data])
            
            if errors:
                logger.error(f"Failed to insert test data: {errors}")
                return False
            else:
                logger.info("Successfully inserted test data with embedding")
                return True
                
        except Exception as e:
            logger.error(f"Failed to test embedding insert: {e}")
            return False
    
    def run_migration(self, dataset_id: str = "soc_data", table_id: str = "processed_alerts"):
        """Run the complete migration process
        
        Args:
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID
        """
        logger.info("Starting BigQuery schema migration for embedding support...")
        
        # Step 1: Add embedding columns
        if not self.add_embedding_columns(dataset_id, table_id):
            logger.error("Failed to add embedding columns")
            return False
        
        # Step 2: Create embedding indexes
        if not self.create_embedding_indexes(dataset_id, table_id):
            logger.error("Failed to create embedding indexes")
            return False
        
        # Step 3: Create embedding views
        if not self.create_embedding_views(dataset_id):
            logger.error("Failed to create embedding views")
            return False
        
        # Step 4: Test embedding insert
        if not self.test_embedding_insert(dataset_id, table_id):
            logger.error("Failed to test embedding insert")
            return False
        
        logger.info("✅ BigQuery schema migration completed successfully!")
        return True

# Test function
def test_schema_migration():
    """Test the schema migration"""
    try:
        migration = BigQuerySchemaMigration()
        
        if migration.client is None:
            print("❌ Failed to initialize BigQuery client")
            return False
        
        print("✅ BigQuery client initialized successfully")
        
        # Test the migration
        success = migration.run_migration()
        
        if success:
            print("🎉 Schema migration test completed successfully!")
        else:
            print("❌ Schema migration test failed")
        
        return success
        
    except Exception as e:
        print(f"❌ Schema migration test failed: {e}")
        return False

if __name__ == "__main__":
    test_schema_migration()












