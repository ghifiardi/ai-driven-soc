"""
Configuration file for AI Model Training Dashboard
Connects to the Continuous Learning Agent data sources
"""

import os
from google.cloud import bigquery
from google.cloud import storage
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIDashboardConfig:
    """Configuration class for AI Model Training Dashboard"""
    
    def __init__(self):
        self.project_id = os.getenv('GCP_PROJECT_ID', 'your-project-id')
        self.dataset_id = os.getenv('BIGQUERY_DATASET_ID', 'soc_analytics')
        
        # BigQuery table configurations
        self.tables = {
            'model_training_history': 'model_training_history',
            'model_performance': 'model_performance_metrics',
            'training_data_quality': 'training_data_quality',
            'ab_testing_results': 'model_ab_testing_results',
            'resource_utilization': 'training_resource_utilization'
        }
        
        # Initialize BigQuery client
        try:
            self.bq_client = bigquery.Client(project=self.project_id)
            logger.info(f"BigQuery client initialized for project: {self.project_id}")
        except Exception as e:
            logger.error(f"Failed to initialize BigQuery client: {e}")
            self.bq_client = None
    
    def get_training_history_query(self, days_back=90):
        """Query to get model training history"""
        return f"""
        SELECT 
            training_date as date,
            model_version,
            training_samples,
            training_accuracy,
            validation_accuracy,
            training_loss,
            validation_loss,
            training_time_minutes,
            gpu_utilization,
            memory_usage_gb,
            training_status as status
        FROM `{self.project_id}.{self.dataset_id}.{self.tables['model_training_history']}`
        WHERE training_date >= DATE_SUB(CURRENT_DATE(), INTERVAL {days_back} DAY)
        ORDER BY training_date DESC
        """
    
    def get_performance_metrics_query(self, days_back=90):
        """Query to get model performance metrics over time"""
        return f"""
        SELECT 
            metric_date as date,
            automation_rate,
            detection_accuracy,
            false_positive_rate,
            response_time_minutes
        FROM `{self.project_id}.{self.dataset_id}.{self.tables['model_performance']}`
        WHERE metric_date >= DATE_SUB(CURRENT_DATE(), INTERVAL {days_back} DAY)
        ORDER BY metric_date ASC
        """
    
    def get_data_quality_metrics_query(self):
        """Query to get training data quality metrics"""
        return f"""
        SELECT 
            data_quality_score,
            label_accuracy,
            data_completeness,
            feature_coverage,
            data_freshness_days,
            outlier_detection_rate,
            data_diversity_score,
            class_balance_score,
            validation_split_ratio
        FROM `{self.project_id}.{self.dataset_id}.{self.tables['training_data_quality']}`
        WHERE metric_date = (
            SELECT MAX(metric_date) 
            FROM `{self.project_id}.{self.dataset_id}.{self.tables['training_data_quality']}`
        )
        """
    
    def get_ab_testing_results_query(self):
        """Query to get A/B testing results"""
        return f"""
        SELECT 
            model_version,
            accuracy,
            automation_rate,
            false_positive_rate,
            response_time_minutes,
            training_time_hours
        FROM `{self.project_id}.{self.dataset_id}.{self.tables['ab_testing_results']}`
        ORDER BY model_version DESC
        LIMIT 10
        """
    
    def get_resource_utilization_query(self, days_back=30):
        """Query to get training resource utilization"""
        return f"""
        SELECT 
            training_date as date,
            gpu_utilization,
            memory_usage_gb,
            cpu_utilization,
            network_bandwidth_mbps
        FROM `{self.project_id}.{self.dataset_id}.{self.tables['resource_utilization']}`
        WHERE training_date >= DATE_SUB(CURRENT_DATE(), INTERVAL {days_back} DAY)
        ORDER BY training_date ASC
        """
    
    def execute_query(self, query):
        """Execute a BigQuery query and return results"""
        if not self.bq_client:
            logger.error("BigQuery client not initialized")
            return None
        
        try:
            query_job = self.bq_client.query(query)
            results = query_job.result()
            return results.to_dataframe()
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return None

# Environment variables for configuration
ENV_VARS = {
    'GCP_PROJECT_ID': 'your-gcp-project-id',
    'BIGQUERY_DATASET_ID': 'soc_analytics',
    'GOOGLE_APPLICATION_CREDENTIALS': 'path/to/your/service-account-key.json'
}

# Example usage:
# export GCP_PROJECT_ID="your-project-id"
# export BIGQUERY_DATASET_ID="soc_analytics"
# export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account-key.json"
