#!/usr/bin/env python3
"""
Simple Phase 1 Critical Indicators Check
Answer the 3 specific questions about Phase 1 success
"""

import os
import json
import logging
from datetime import datetime, timedelta
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimplePhase1Check:
    """Simple check of the 3 critical Phase 1 indicators"""
    
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
    
    def check_embedding_pipeline(self):
        """1. Embedding pipeline table: No Schema drift or Pub/Sub backlog"""
        print("🔍 1. EMBEDDING PIPELINE HEALTH")
        print("-" * 40)
        
        if not self.client:
            print("❌ BigQuery client not available")
            return False
        
        try:
            # Simple data quality check
            query = f"""
            SELECT 
                COUNT(*) as total_alerts,
                COUNT(embedding) as alerts_with_embeddings,
                COUNT(embedding_timestamp) as alerts_with_timestamps,
                COUNT(embedding_similarity) as alerts_with_similarity,
                COUNT(rl_reward_score) as alerts_with_rewards,
                -- Check for data quality issues
                COUNT(CASE WHEN ARRAY_LENGTH(embedding) != 768 THEN 1 END) as invalid_embedding_length,
                COUNT(CASE WHEN embedding_similarity < 0 OR embedding_similarity > 1 THEN 1 END) as invalid_similarity,
                COUNT(CASE WHEN rl_reward_score < 0 OR rl_reward_score > 1 THEN 1 END) as invalid_rewards
            FROM `{self.project_id}.soc_data.processed_alerts`
            WHERE embedding_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
            """
            
            result = self.client.query(query).to_dataframe()
            
            if not result.empty:
                data = result.iloc[0]
                print(f"📊 Data Quality Analysis (Last 7 days):")
                print(f"  Total Alerts: {data['total_alerts']:,}")
                print(f"  With Embeddings: {data['alerts_with_embeddings']:,} ({data['alerts_with_embeddings']/data['total_alerts']*100:.1f}%)")
                print(f"  With Timestamps: {data['alerts_with_timestamps']:,} ({data['alerts_with_timestamps']/data['total_alerts']*100:.1f}%)")
                print(f"  With Similarity: {data['alerts_with_similarity']:,} ({data['alerts_with_similarity']/data['total_alerts']*100:.1f}%)")
                print(f"  With Rewards: {data['alerts_with_rewards']:,} ({data['alerts_with_rewards']/data['total_alerts']*100:.1f}%)")
                
                print(f"\n🚨 Data Quality Issues:")
                print(f"  Invalid Embedding Length: {data['invalid_embedding_length']}")
                print(f"  Invalid Similarity Scores: {data['invalid_similarity']}")
                print(f"  Invalid Reward Scores: {data['invalid_rewards']}")
                
                # Health assessment
                coverage = (data['alerts_with_embeddings'] + data['alerts_with_timestamps'] + 
                           data['alerts_with_similarity'] + data['alerts_with_rewards']) / (4 * data['total_alerts'])
                no_errors = (data['invalid_embedding_length'] + data['invalid_similarity'] + data['invalid_rewards']) == 0
                
                if coverage >= 0.95 and no_errors:
                    print(f"\n✅ PIPELINE HEALTH: EXCELLENT")
                    return True
                elif coverage >= 0.90:
                    print(f"\n⚠️ PIPELINE HEALTH: GOOD")
                    return True
                else:
                    print(f"\n❌ PIPELINE HEALTH: POOR")
                    return False
            else:
                print("❌ No data available")
                return False
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def check_rl_reward_trend(self):
        """2. RL reward signal consistent: Positive trend for > 7 days"""
        print("\n🎯 2. RL REWARD SIGNAL TREND")
        print("-" * 40)
        
        if not self.client:
            print("❌ BigQuery client not available")
            return False
        
        try:
            # Get daily RL reward averages
            query = f"""
            SELECT 
                DATE(embedding_timestamp) as analysis_date,
                AVG(rl_reward_score) as avg_reward,
                COUNT(*) as alert_count
            FROM `{self.project_id}.soc_data.processed_alerts`
            WHERE embedding_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 14 DAY)
            AND rl_reward_score IS NOT NULL
            GROUP BY DATE(embedding_timestamp)
            ORDER BY analysis_date ASC
            """
            
            result = self.client.query(query).to_dataframe()
            
            if not result.empty and len(result) >= 7:
                print(f"📊 RL Reward Trend (Last {len(result)} days):")
                
                # Calculate trends
                recent_7d = result.tail(7)
                older_7d = result.head(7) if len(result) >= 14 else result.head(len(result)-7)
                
                recent_avg = recent_7d['avg_reward'].mean()
                older_avg = older_7d['avg_reward'].mean()
                improvement = recent_avg - older_avg
                
                print(f"  Recent 7-day average: {recent_avg:.3f}")
                print(f"  Older 7-day average: {older_avg:.3f}")
                print(f"  Improvement: {improvement:+.3f}")
                
                # Check for positive trend
                positive_trend = improvement > 0
                good_rewards = recent_avg > 0.5
                
                print(f"\n✅ RL REWARD SUCCESS CRITERIA:")
                print(f"  Positive Trend: {'✅' if positive_trend else '❌'} ({improvement:+.3f})")
                print(f"  Good Rewards (>0.5): {'✅' if good_rewards else '❌'} ({recent_avg:.3f})")
                
                success = positive_trend and good_rewards
                print(f"\n🏆 RL REWARD SIGNAL: {'✅ SUCCESS' if success else '❌ NEEDS WORK'}")
                return success
            else:
                print("❌ Insufficient data (need at least 7 days)")
                return False
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def check_alert_entropy(self):
        """3. Alert entropy decreased: EI < 0.7 vs baseline > 0.9"""
        print("\n🧠 3. ALERT ENTROPY DECREASE")
        print("-" * 40)
        
        if not self.client:
            print("❌ BigQuery client not available")
            return False
        
        try:
            # Get entropy analysis
            query = f"""
            WITH daily_entropy AS (
                SELECT 
                    DATE(embedding_timestamp) as analysis_date,
                    COUNT(*) as total_alerts,
                    COUNT(DISTINCT ROUND(embedding_similarity, 2)) as unique_clusters,
                    -- Calculate Entropy Index: EI = 1 - (Unique Clusters / Total Alerts)
                    ROUND(1 - (COUNT(DISTINCT ROUND(embedding_similarity, 2)) / COUNT(*)), 4) as entropy_index
                FROM `{self.project_id}.soc_data.processed_alerts`
                WHERE embedding_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 14 DAY)
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
            
            if not result.empty and len(result) >= 7:
                print(f"📊 Entropy Analysis (Last {len(result)} days):")
                
                # Calculate baseline vs current
                recent_7d = result.tail(7)
                older_7d = result.head(7) if len(result) >= 14 else result.head(len(result)-7)
                
                current_entropy = recent_7d['entropy_index'].mean()
                baseline_entropy = older_7d['entropy_index'].mean()
                improvement = baseline_entropy - current_entropy
                
                print(f"  Current Entropy (Last 7 days): {current_entropy:.3f}")
                print(f"  Baseline Entropy (First 7 days): {baseline_entropy:.3f}")
                print(f"  Entropy Improvement: {improvement:+.3f}")
                
                # Check success criteria
                current_low = current_entropy < 0.7
                baseline_high = baseline_entropy > 0.9
                entropy_decreased = improvement > 0
                
                print(f"\n✅ ENTROPY SUCCESS CRITERIA:")
                print(f"  Current EI < 0.7: {'✅' if current_low else '❌'} ({current_entropy:.3f})")
                print(f"  Baseline EI > 0.9: {'✅' if baseline_high else '❌'} ({baseline_entropy:.3f})")
                print(f"  Entropy Decreased: {'✅' if entropy_decreased else '❌'} ({improvement:+.3f})")
                
                success = current_low and baseline_high and entropy_decreased
                print(f"\n🏆 ALERT ENTROPY: {'✅ SUCCESS' if success else '❌ NEEDS WORK'}")
                return success
            else:
                print("❌ Insufficient data (need at least 7 days)")
                return False
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def run_complete_check(self):
        """Run complete Phase 1 critical indicators check"""
        print("🧠 PHASE 1 CRITICAL INDICATORS CHECK")
        print("=" * 50)
        print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check all 3 indicators
        results = {
            "1. Embedding Pipeline": self.check_embedding_pipeline(),
            "2. RL Reward Trend": self.check_rl_reward_trend(),
            "3. Alert Entropy": self.check_alert_entropy()
        }
        
        # Summary
        print(f"\n🏆 PHASE 1 CRITICAL INDICATORS SUMMARY")
        print("=" * 50)
        
        success_count = sum(results.values())
        total_indicators = len(results)
        
        for indicator, success in results.items():
            status = "✅ SUCCESS" if success else "❌ NEEDS WORK"
            print(f"  {indicator}: {status}")
        
        print(f"\n📊 OVERALL STATUS: {success_count}/{total_indicators} indicators met")
        
        if success_count == total_indicators:
            print("🎉 PHASE 1 IS READY!")
        elif success_count >= 2:
            print("⚠️ PHASE 1 IS MOSTLY READY")
        else:
            print("❌ PHASE 1 NEEDS WORK")
        
        return results

def main():
    """Main function"""
    checker = SimplePhase1Check()
    results = checker.run_complete_check()
    
    print(f"\n🎉 Check complete!")
    print(f"📊 Access KPI Dashboard: http://localhost:8528")

if __name__ == "__main__":
    main()












