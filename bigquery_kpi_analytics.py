#!/usr/bin/env python3
"""
BigQuery KPI Analytics for AI-Driven SOC Phase 1
Implements the follow-up KPIs and measurement infrastructure
"""

import os
import json
import logging
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
from google.cloud import bigquery
from google.oauth2 import service_account

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BigQueryKPIAnalytics:
    """BigQuery analytics for Phase 1 KPI measurement"""
    
    def __init__(self, project_id: str = "chronicle-dev-2be9"):
        """Initialize the KPI analytics service"""
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
    
    def create_embedding_analysis_view(self):
        """Create the embedding_analysis_view for alert quality measurement"""
        try:
            view_query = f"""
            CREATE OR REPLACE VIEW `{self.project_id}.soc_data.embedding_analysis_view` AS
            WITH embedding_clusters AS (
                SELECT 
                    alert_id,
                    timestamp,
                    classification,
                    confidence_score,
                    is_anomaly,
                    embedding_timestamp,
                    embedding_model,
                    embedding_similarity,
                    rl_reward_score,
                    -- Calculate embedding cluster using approximate similarity
                    ROUND(embedding_similarity, 2) as similarity_cluster,
                    -- Group alerts by similarity patterns
                    CASE 
                        WHEN embedding_similarity > 0.8 THEN 'high_similarity'
                        WHEN embedding_similarity > 0.6 THEN 'medium_similarity'
                        WHEN embedding_similarity > 0.4 THEN 'low_similarity'
                        ELSE 'unique'
                    END as similarity_group
                FROM `{self.project_id}.soc_data.processed_alerts`
                WHERE embedding IS NOT NULL
                AND embedding_timestamp IS NOT NULL
            ),
            cluster_metrics AS (
                SELECT 
                    similarity_group,
                    similarity_cluster,
                    COUNT(*) as alert_count,
                    AVG(confidence_score) as avg_confidence,
                    AVG(embedding_similarity) as avg_similarity,
                    AVG(rl_reward_score) as avg_reward,
                    COUNT(DISTINCT classification) as unique_classifications,
                    MIN(timestamp) as first_alert,
                    MAX(timestamp) as last_alert
                FROM embedding_clusters
                GROUP BY similarity_group, similarity_cluster
            )
            SELECT 
                similarity_group,
                similarity_cluster,
                alert_count,
                ROUND(avg_confidence, 3) as avg_confidence,
                ROUND(avg_similarity, 3) as avg_similarity,
                ROUND(avg_reward, 3) as avg_reward,
                unique_classifications,
                first_alert,
                last_alert,
                -- Calculate redundancy score (higher = more redundant)
                CASE 
                    WHEN alert_count > 10 THEN 'high_redundancy'
                    WHEN alert_count > 5 THEN 'medium_redundancy'
                    WHEN alert_count > 2 THEN 'low_redundancy'
                    ELSE 'unique'
                END as redundancy_level
            FROM cluster_metrics
            ORDER BY alert_count DESC, avg_similarity DESC
            """
            
            job = self.client.query(view_query)
            job.result()
            logger.info("✅ Created embedding_analysis_view successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create embedding_analysis_view: {e}")
            return False
    
    def create_entropy_index_view(self):
        """Create view for Entropy Index (EI) calculation"""
        try:
            view_query = f"""
            CREATE OR REPLACE VIEW `{self.project_id}.soc_data.entropy_index_view` AS
            WITH daily_metrics AS (
                SELECT 
                    DATE(embedding_timestamp) as analysis_date,
                    COUNT(*) as total_alerts,
                    COUNT(DISTINCT ROUND(embedding_similarity, 2)) as unique_clusters,
                    -- Calculate Entropy Index: EI = 1 - (Unique Clusters / Total Alerts)
                    ROUND(1 - (COUNT(DISTINCT ROUND(embedding_similarity, 2)) / COUNT(*)), 4) as entropy_index,
                    -- Additional coherence metrics
                    AVG(embedding_similarity) as avg_similarity,
                    COUNT(CASE WHEN embedding_similarity > 0.7 THEN 1 END) as high_similarity_alerts,
                    COUNT(CASE WHEN rl_reward_score > 0.7 THEN 1 END) as high_reward_alerts
                FROM `{self.project_id}.soc_data.processed_alerts`
                WHERE embedding_timestamp IS NOT NULL
                AND embedding IS NOT NULL
                GROUP BY DATE(embedding_timestamp)
            ),
            weekly_trends AS (
                SELECT 
                    DATE_TRUNC(analysis_date, WEEK) as week_start,
                    AVG(entropy_index) as avg_entropy_index,
                    AVG(unique_clusters) as avg_unique_clusters,
                    SUM(total_alerts) as total_weekly_alerts,
                    AVG(avg_similarity) as avg_weekly_similarity,
                    SUM(high_similarity_alerts) as total_high_similarity,
                    SUM(high_reward_alerts) as total_high_reward
                FROM daily_metrics
                GROUP BY DATE_TRUNC(analysis_date, WEEK)
            )
            SELECT 
                analysis_date,
                total_alerts,
                unique_clusters,
                entropy_index,
                avg_similarity,
                high_similarity_alerts,
                high_reward_alerts,
                -- Coherence interpretation
                CASE 
                    WHEN entropy_index < 0.3 THEN 'highly_coherent'
                    WHEN entropy_index < 0.5 THEN 'moderately_coherent'
                    WHEN entropy_index < 0.7 THEN 'somewhat_chaotic'
                    ELSE 'highly_chaotic'
                END as coherence_level,
                -- Trend indicators
                LAG(entropy_index, 1) OVER (ORDER BY analysis_date) as prev_entropy_index,
                ROUND(entropy_index - LAG(entropy_index, 1) OVER (ORDER BY analysis_date), 4) as entropy_change
            FROM daily_metrics
            ORDER BY analysis_date DESC
            """
            
            job = self.client.query(view_query)
            job.result()
            logger.info("✅ Created entropy_index_view successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create entropy_index_view: {e}")
            return False
    
    def create_triage_time_view(self):
        """Create view for triage time measurement"""
        try:
            view_query = f"""
            CREATE OR REPLACE VIEW `{self.project_id}.soc_data.triage_time_analysis` AS
            WITH alert_processing_times AS (
                SELECT 
                    alert_id,
                    timestamp as alert_created,
                    embedding_timestamp,
                    -- Calculate processing time (embedding generation time)
                    DATETIME_DIFF(
                        DATETIME(embedding_timestamp), 
                        DATETIME(timestamp), 
                        SECOND
                    ) as processing_time_seconds,
                    classification,
                    confidence_score,
                    embedding_similarity,
                    rl_reward_score,
                    -- Categorize by similarity group for analysis
                    CASE 
                        WHEN embedding_similarity > 0.8 THEN 'high_similarity'
                        WHEN embedding_similarity > 0.6 THEN 'medium_similarity'
                        WHEN embedding_similarity > 0.4 THEN 'low_similarity'
                        ELSE 'unique'
                    END as similarity_group
                FROM `{self.project_id}.soc_data.processed_alerts`
                WHERE embedding_timestamp IS NOT NULL
                AND timestamp IS NOT NULL
            ),
            daily_triage_metrics AS (
                SELECT 
                    DATE(alert_created) as analysis_date,
                    similarity_group,
                    COUNT(*) as alert_count,
                    AVG(processing_time_seconds) as avg_processing_time,
                    -- Calculate percentiles using APPROX_QUANTILES
                    APPROX_QUANTILES(processing_time_seconds, 100)[OFFSET(50)] as median_processing_time,
                    APPROX_QUANTILES(processing_time_seconds, 100)[OFFSET(95)] as p95_processing_time,
                    -- Calculate efficiency metrics
                    COUNT(CASE WHEN processing_time_seconds < 30 THEN 1 END) as fast_processing_count,
                    COUNT(CASE WHEN processing_time_seconds > 120 THEN 1 END) as slow_processing_count
                FROM alert_processing_times
                GROUP BY DATE(alert_created), similarity_group
            )
            SELECT 
                analysis_date,
                similarity_group,
                alert_count,
                ROUND(avg_processing_time, 2) as avg_processing_time_seconds,
                ROUND(median_processing_time, 2) as median_processing_time_seconds,
                ROUND(p95_processing_time, 2) as p95_processing_time_seconds,
                fast_processing_count,
                slow_processing_count,
                -- Efficiency ratios
                ROUND(fast_processing_count / alert_count, 3) as fast_processing_ratio,
                ROUND(slow_processing_count / alert_count, 3) as slow_processing_ratio,
                -- Time savings estimation (compared to unique alerts)
                ROUND(
                    (avg_processing_time - 
                     (SELECT AVG(processing_time_seconds) 
                      FROM alert_processing_times 
                      WHERE similarity_group = 'unique' 
                      AND DATE(alert_created) = analysis_date)
                    ), 2
                ) as estimated_time_savings
            FROM daily_triage_metrics
            ORDER BY analysis_date DESC, similarity_group
            """
            
            job = self.client.query(view_query)
            job.result()
            logger.info("✅ Created triage_time_analysis view successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create triage_time_analysis view: {e}")
            return False
    
    def create_kpi_summary_view(self):
        """Create comprehensive KPI summary view"""
        try:
            view_query = f"""
            CREATE OR REPLACE VIEW `{self.project_id}.soc_data.kpi_summary` AS
            WITH current_period AS (
                SELECT 
                    DATE(embedding_timestamp) as analysis_date,
                    COUNT(*) as total_alerts,
                    COUNT(DISTINCT CASE WHEN embedding IS NOT NULL THEN alert_id END) as alerts_with_embeddings,
                    COUNT(DISTINCT ROUND(embedding_similarity, 2)) as unique_clusters,
                    AVG(embedding_similarity) as avg_similarity,
                    AVG(rl_reward_score) as avg_reward_score,
                    -- Calculate Entropy Index
                    ROUND(1 - (COUNT(DISTINCT ROUND(embedding_similarity, 2)) / COUNT(*)), 4) as entropy_index,
                    -- Redundancy metrics
                    COUNT(CASE WHEN embedding_similarity > 0.8 THEN 1 END) as high_redundancy_alerts,
                    COUNT(CASE WHEN embedding_similarity <= 0.4 THEN 1 END) as unique_alerts,
                    -- Processing efficiency
                    AVG(DATETIME_DIFF(DATETIME(embedding_timestamp), DATETIME(timestamp), SECOND)) as avg_processing_time
                FROM `{self.project_id}.soc_data.processed_alerts`
                WHERE embedding_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
                GROUP BY DATE(embedding_timestamp)
            ),
            baseline_period AS (
                SELECT 
                    COUNT(*) as baseline_total_alerts,
                    COUNT(DISTINCT alert_id) as baseline_unique_alerts
                FROM `{self.project_id}.soc_data.processed_alerts`
                WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 14 DAY)
                AND timestamp < TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
                AND embedding IS NULL  -- Pre-embedding period
            )
            SELECT 
                analysis_date,
                total_alerts,
                alerts_with_embeddings,
                unique_clusters,
                ROUND(avg_similarity, 3) as avg_similarity,
                ROUND(avg_reward_score, 3) as avg_reward_score,
                entropy_index,
                high_redundancy_alerts,
                unique_alerts,
                ROUND(avg_processing_time, 2) as avg_processing_time_seconds,
                -- KPI Calculations
                ROUND((high_redundancy_alerts / total_alerts) * 100, 2) as redundancy_percentage,
                ROUND((unique_alerts / total_alerts) * 100, 2) as uniqueness_percentage,
                -- Coherence assessment
                CASE 
                    WHEN entropy_index < 0.3 THEN 'Excellent'
                    WHEN entropy_index < 0.5 THEN 'Good'
                    WHEN entropy_index < 0.7 THEN 'Fair'
                    ELSE 'Poor'
                END as coherence_rating,
                -- Performance indicators
                CASE 
                    WHEN avg_processing_time < 30 THEN 'Fast'
                    WHEN avg_processing_time < 60 THEN 'Normal'
                    WHEN avg_processing_time < 120 THEN 'Slow'
                    ELSE 'Very Slow'
                END as processing_speed_rating
            FROM current_period
            ORDER BY analysis_date DESC
            """
            
            job = self.client.query(view_query)
            job.result()
            logger.info("✅ Created kpi_summary view successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create kpi_summary view: {e}")
            return False
    
    def get_alert_quality_metrics(self, days: int = 7) -> Dict[str, Any]:
        """Get alert quality metrics for the specified period"""
        try:
            query = f"""
            SELECT 
                COUNT(*) as total_alerts,
                COUNT(DISTINCT similarity_cluster) as unique_clusters,
                ROUND(1 - (COUNT(DISTINCT similarity_cluster) / COUNT(*)), 4) as entropy_index,
                COUNT(CASE WHEN similarity_group = 'high_similarity' THEN 1 END) as redundant_alerts,
                COUNT(CASE WHEN similarity_group = 'unique' THEN 1 END) as unique_alerts,
                ROUND((COUNT(CASE WHEN similarity_group = 'high_similarity' THEN 1 END) / COUNT(*)) * 100, 2) as redundancy_percentage,
                ROUND(AVG(avg_similarity), 3) as avg_similarity,
                ROUND(AVG(avg_reward), 3) as avg_reward_score
            FROM `{self.project_id}.soc_data.embedding_analysis_view`
            WHERE analysis_date >= DATE_SUB(CURRENT_DATE(), INTERVAL {days} DAY)
            """
            
            result = self.client.query(query).to_dataframe()
            if not result.empty:
                return result.iloc[0].to_dict()
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get alert quality metrics: {e}")
            return {}
    
    def get_entropy_index_trend(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get Entropy Index trend over time"""
        try:
            query = f"""
            SELECT 
                analysis_date,
                entropy_index,
                total_alerts,
                unique_clusters,
                coherence_level,
                entropy_change
            FROM `{self.project_id}.soc_data.entropy_index_view`
            WHERE analysis_date >= DATE_SUB(CURRENT_DATE(), INTERVAL {days} DAY)
            ORDER BY analysis_date DESC
            """
            
            result = self.client.query(query).to_dataframe()
            return result.to_dict('records')
            
        except Exception as e:
            logger.error(f"Failed to get entropy index trend: {e}")
            return []
    
    def get_triage_efficiency_metrics(self, days: int = 7) -> Dict[str, Any]:
        """Get triage efficiency metrics"""
        try:
            query = f"""
            SELECT 
                AVG(avg_processing_time_seconds) as avg_processing_time,
                AVG(median_processing_time_seconds) as median_processing_time,
                AVG(fast_processing_ratio) as fast_processing_ratio,
                AVG(slow_processing_ratio) as slow_processing_ratio,
                SUM(estimated_time_savings) as total_time_savings_seconds,
                COUNT(DISTINCT similarity_group) as similarity_groups_processed
            FROM `{self.project_id}.soc_data.triage_time_analysis`
            WHERE analysis_date >= DATE_SUB(CURRENT_DATE(), INTERVAL {days} DAY)
            """
            
            result = self.client.query(query).to_dataframe()
            if not result.empty:
                return result.iloc[0].to_dict()
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get triage efficiency metrics: {e}")
            return {}
    
    def run_all_analytics_setup(self):
        """Run complete analytics setup"""
        logger.info("🚀 Setting up BigQuery KPI Analytics for Phase 1...")
        
        success_count = 0
        total_operations = 4
        
        # Create all views
        if self.create_embedding_analysis_view():
            success_count += 1
        
        if self.create_entropy_index_view():
            success_count += 1
        
        if self.create_triage_time_view():
            success_count += 1
        
        if self.create_kpi_summary_view():
            success_count += 1
        
        if success_count == total_operations:
            logger.info("✅ All BigQuery KPI analytics views created successfully!")
            return True
        else:
            logger.error(f"❌ Only {success_count}/{total_operations} operations completed successfully")
            return False

# Test function
def test_kpi_analytics():
    """Test the KPI analytics setup"""
    try:
        analytics = BigQueryKPIAnalytics()
        
        if analytics.client is None:
            print("❌ Failed to initialize BigQuery client")
            return False
        
        print("✅ BigQuery client initialized successfully")
        
        # Run analytics setup
        success = analytics.run_all_analytics_setup()
        
        if success:
            print("🎉 KPI Analytics setup completed successfully!")
            
            # Test metric retrieval
            print("\n📊 Testing metric retrieval...")
            
            # Alert Quality Metrics
            quality_metrics = analytics.get_alert_quality_metrics(7)
            if quality_metrics:
                print(f"📈 Alert Quality (7 days):")
                print(f"  - Total Alerts: {quality_metrics.get('total_alerts', 0)}")
                print(f"  - Unique Clusters: {quality_metrics.get('unique_clusters', 0)}")
                print(f"  - Entropy Index: {quality_metrics.get('entropy_index', 0)}")
                print(f"  - Redundancy %: {quality_metrics.get('redundancy_percentage', 0)}%")
            
            # Entropy Index Trend
            entropy_trend = analytics.get_entropy_index_trend(7)
            if entropy_trend:
                print(f"📊 Entropy Index Trend: {len(entropy_trend)} days of data")
            
            # Triage Efficiency
            triage_metrics = analytics.get_triage_efficiency_metrics(7)
            if triage_metrics:
                print(f"⏱️ Triage Efficiency:")
                print(f"  - Avg Processing Time: {triage_metrics.get('avg_processing_time', 0):.2f}s")
                print(f"  - Fast Processing Ratio: {triage_metrics.get('fast_processing_ratio', 0):.3f}")
        
        return success
        
    except Exception as e:
        print(f"❌ KPI Analytics test failed: {e}")
        return False

if __name__ == "__main__":
    test_kpi_analytics()
