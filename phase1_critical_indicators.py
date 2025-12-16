#!/usr/bin/env python3
"""
Phase 1 Critical Success Indicators Analysis
Check the 3 key metrics for Phase 1 success
"""

import os
import json
import logging
from datetime import datetime, timedelta
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Phase1CriticalIndicators:
    """Analyze the 3 critical Phase 1 success indicators"""
    
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
    
    def check_embedding_pipeline_health(self):
        """1. Check Embedding Pipeline: No Schema drift or Pub/Sub backlog"""
        print("🔍 1. EMBEDDING PIPELINE HEALTH CHECK")
        print("=" * 50)
        
        if not self.client:
            print("❌ BigQuery client not available")
            return False
        
        try:
            # Check schema consistency
            schema_query = f"""
            SELECT 
                column_name,
                data_type,
                is_nullable,
                description
            FROM `{self.project_id}.soc_data.INFORMATION_SCHEMA.COLUMNS`
            WHERE table_name = 'processed_alerts'
            AND column_name LIKE '%embedding%'
            ORDER BY ordinal_position
            """
            
            schema_result = self.client.query(schema_query).to_dataframe()
            
            print("📊 Schema Analysis:")
            if not schema_result.empty:
                for _, row in schema_result.iterrows():
                    print(f"  ✅ {row['column_name']}: {row['data_type']} ({row['description']})")
            else:
                print("  ❌ No embedding columns found")
                return False
            
            # Check data consistency
            data_consistency_query = f"""
            WITH embedding_analysis AS (
                SELECT 
                    COUNT(*) as total_alerts,
                    COUNT(embedding) as alerts_with_embeddings,
                    COUNT(embedding_timestamp) as alerts_with_timestamps,
                    COUNT(embedding_similarity) as alerts_with_similarity,
                    COUNT(rl_reward_score) as alerts_with_rewards,
                    -- Check for schema drift
                    COUNT(CASE WHEN ARRAY_LENGTH(embedding) != 768 THEN 1 END) as invalid_embedding_length,
                    COUNT(CASE WHEN embedding_similarity < 0 OR embedding_similarity > 1 THEN 1 END) as invalid_similarity,
                    COUNT(CASE WHEN rl_reward_score < 0 OR rl_reward_score > 1 THEN 1 END) as invalid_rewards
                FROM `{self.project_id}.soc_data.processed_alerts`
                WHERE embedding_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
            )
            SELECT 
                total_alerts,
                alerts_with_embeddings,
                alerts_with_timestamps,
                alerts_with_similarity,
                alerts_with_rewards,
                invalid_embedding_length,
                invalid_similarity,
                invalid_rewards,
                -- Calculate health scores
                ROUND((alerts_with_embeddings / total_alerts) * 100, 2) as embedding_coverage_pct,
                ROUND((alerts_with_timestamps / total_alerts) * 100, 2) as timestamp_coverage_pct,
                ROUND((alerts_with_similarity / total_alerts) * 100, 2) as similarity_coverage_pct,
                ROUND((alerts_with_rewards / total_alerts) * 100, 2) as reward_coverage_pct
            FROM embedding_analysis
            """
            
            consistency_result = self.client.query(data_consistency_query).to_dataframe()
            
            if not consistency_result.empty:
                data = consistency_result.iloc[0]
                print(f"\n📈 Data Consistency Analysis:")
                print(f"  📊 Total Alerts: {data['total_alerts']:,}")
                print(f"  🧠 Embedding Coverage: {data['embedding_coverage_pct']:.1f}%")
                print(f"  ⏰ Timestamp Coverage: {data['timestamp_coverage_pct']:.1f}%")
                print(f"  🔗 Similarity Coverage: {data['similarity_coverage_pct']:.1f}%")
                print(f"  🎯 Reward Coverage: {data['reward_coverage_pct']:.1f}%")
                
                print(f"\n🚨 Data Quality Issues:")
                print(f"  Invalid Embedding Length: {data['invalid_embedding_length']}")
                print(f"  Invalid Similarity Scores: {data['invalid_similarity']}")
                print(f"  Invalid Reward Scores: {data['invalid_rewards']}")
                
                # Health assessment
                health_score = (
                    data['embedding_coverage_pct'] + 
                    data['timestamp_coverage_pct'] + 
                    data['similarity_coverage_pct'] + 
                    data['reward_coverage_pct']
                ) / 4
                
                if health_score >= 95 and data['invalid_embedding_length'] == 0:
                    print(f"\n✅ PIPELINE HEALTH: EXCELLENT ({health_score:.1f}%)")
                    return True
                elif health_score >= 90:
                    print(f"\n⚠️ PIPELINE HEALTH: GOOD ({health_score:.1f}%)")
                    return True
                else:
                    print(f"\n❌ PIPELINE HEALTH: POOR ({health_score:.1f}%)")
                    return False
            else:
                print("❌ No data available for analysis")
                return False
                
        except Exception as e:
            print(f"❌ Error checking embedding pipeline: {e}")
            return False
    
    def check_rl_reward_trend(self):
        """2. Check RL Reward Signal: Positive trend for > 7 days"""
        print("\n🎯 2. RL REWARD SIGNAL TREND ANALYSIS")
        print("=" * 50)
        
        if not self.client:
            print("❌ BigQuery client not available")
            return False
        
        try:
            # Get daily RL reward trends
            reward_trend_query = f"""
            WITH daily_rewards AS (
                SELECT 
                    DATE(embedding_timestamp) as analysis_date,
                    AVG(rl_reward_score) as avg_reward,
                    COUNT(*) as alert_count,
                    STDDEV(rl_reward_score) as reward_volatility,
                    MIN(rl_reward_score) as min_reward,
                    MAX(rl_reward_score) as max_reward
                FROM `{self.project_id}.soc_data.processed_alerts`
                WHERE embedding_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 14 DAY)
                AND rl_reward_score IS NOT NULL
                GROUP BY DATE(embedding_timestamp)
                ORDER BY analysis_date ASC
            ),
            trend_analysis AS (
                SELECT 
                    analysis_date,
                    avg_reward,
                    alert_count,
                    reward_volatility,
                    min_reward,
                    max_reward,
                    -- Calculate 7-day moving average
                    AVG(avg_reward) OVER (
                        ORDER BY analysis_date 
                        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                    ) as moving_avg_7d,
                    -- Calculate trend direction
                    LAG(avg_reward, 1) OVER (ORDER BY analysis_date) as prev_reward,
                    avg_reward - LAG(avg_reward, 1) OVER (ORDER BY analysis_date) as daily_change
                FROM daily_rewards
            ),
            trend_summary AS (
                SELECT 
                    COUNT(*) as total_days,
                    AVG(avg_reward) as overall_avg_reward,
                    AVG(moving_avg_7d) as avg_7d_moving_avg,
                    COUNT(CASE WHEN daily_change > 0 THEN 1 END) as improving_days,
                    COUNT(CASE WHEN daily_change < 0 THEN 1 END) as declining_days,
                    COUNT(CASE WHEN daily_change = 0 THEN 1 END) as stable_days,
                    -- Calculate trend strength
                    CORR(analysis_date, avg_reward) as trend_correlation,
                    -- Check if recent 7 days are positive
                    AVG(CASE 
                        WHEN analysis_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY) 
                        THEN avg_reward 
                    END) as recent_7d_avg
                FROM trend_analysis
                WHERE analysis_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 14 DAY)
            )
            SELECT 
                t.*,
                s.overall_avg_reward,
                s.avg_7d_moving_avg,
                s.improving_days,
                s.declining_days,
                s.stable_days,
                s.trend_correlation,
                s.recent_7d_avg,
                -- Trend assessment
                CASE 
                    WHEN s.trend_correlation > 0.5 AND s.recent_7d_avg > 0.6 THEN 'STRONG_POSITIVE'
                    WHEN s.trend_correlation > 0.2 AND s.recent_7d_avg > 0.5 THEN 'POSITIVE'
                    WHEN s.trend_correlation > -0.2 AND s.recent_7d_avg > 0.4 THEN 'STABLE'
                    WHEN s.trend_correlation < -0.2 OR s.recent_7d_avg < 0.4 THEN 'NEGATIVE'
                    ELSE 'UNCLEAR'
                END as trend_assessment
            FROM trend_analysis t
            CROSS JOIN trend_summary s
            ORDER BY t.analysis_date DESC
            """
            
            trend_result = self.client.query(reward_trend_query).to_dataframe()
            
            if not trend_result.empty:
                latest = trend_result.iloc[0]
                print(f"📊 RL Reward Trend Analysis (Last 14 days):")
                print(f"  🎯 Overall Average Reward: {latest['overall_avg_reward']:.3f}")
                print(f"  📈 7-Day Moving Average: {latest['avg_7d_moving_avg']:.3f}")
                print(f"  📊 Recent 7-Day Average: {latest['recent_7d_avg']:.3f}")
                print(f"  📈 Improving Days: {latest['improving_days']}")
                print(f"  📉 Declining Days: {latest['declining_days']}")
                print(f"  ➡️ Stable Days: {latest['stable_days']}")
                print(f"  📊 Trend Correlation: {latest['trend_correlation']:.3f}")
                print(f"  🎯 Trend Assessment: {latest['trend_assessment']}")
                
                # Success criteria
                positive_trend = latest['trend_correlation'] > 0.2
                recent_positive = latest['recent_7d_avg'] > 0.5
                improving_majority = latest['improving_days'] > latest['declining_days']
                
                print(f"\n✅ RL REWARD SUCCESS CRITERIA:")
                print(f"  Positive Trend (correlation > 0.2): {'✅' if positive_trend else '❌'}")
                print(f"  Recent 7-Day Average > 0.5: {'✅' if recent_positive else '❌'}")
                print(f"  More Improving Days: {'✅' if improving_majority else '❌'}")
                
                success = positive_trend and recent_positive and improving_majority
                print(f"\n🏆 RL REWARD SIGNAL: {'✅ SUCCESS' if success else '❌ NEEDS IMPROVEMENT'}")
                return success
            else:
                print("❌ No RL reward data available")
                return False
                
        except Exception as e:
            print(f"❌ Error checking RL reward trend: {e}")
            return False
    
    def check_alert_entropy_decrease(self):
        """3. Check Alert Entropy: EI < 0.7 vs baseline > 0.9"""
        print("\n🧠 3. ALERT ENTROPY DECREASE ANALYSIS")
        print("=" * 50)
        
        if not self.client:
            print("❌ BigQuery client not available")
            return False
        
        try:
            # Get entropy analysis with baseline comparison
            entropy_query = f"""
            WITH daily_entropy AS (
                SELECT 
                    DATE(embedding_timestamp) as analysis_date,
                    COUNT(*) as total_alerts,
                    COUNT(DISTINCT ROUND(embedding_similarity, 2)) as unique_clusters,
                    -- Calculate Entropy Index: EI = 1 - (Unique Clusters / Total Alerts)
                    ROUND(1 - (COUNT(DISTINCT ROUND(embedding_similarity, 2)) / COUNT(*)), 4) as entropy_index,
                    AVG(embedding_similarity) as avg_similarity,
                    COUNT(CASE WHEN embedding_similarity > 0.8 THEN 1 END) as high_similarity_alerts
                FROM `{self.project_id}.soc_data.processed_alerts`
                WHERE embedding_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 14 DAY)
                GROUP BY DATE(embedding_timestamp)
                ORDER BY analysis_date ASC
            ),
            baseline_analysis AS (
                SELECT 
                    -- Baseline: First 7 days (older data)
                    AVG(CASE 
                        WHEN analysis_date < DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY) 
                        THEN entropy_index 
                    END) as baseline_entropy,
                    -- Current: Last 7 days (recent data)
                    AVG(CASE 
                        WHEN analysis_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY) 
                        THEN entropy_index 
                    END) as current_entropy,
                    -- Calculate improvement
                    AVG(CASE 
                        WHEN analysis_date < DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY) 
                        THEN entropy_index 
                    END) - AVG(CASE 
                        WHEN analysis_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY) 
                        THEN entropy_index 
                    END) as entropy_improvement
                FROM daily_entropy
            ),
            trend_analysis AS (
                SELECT 
                    analysis_date,
                    entropy_index,
                    total_alerts,
                    unique_clusters,
                    avg_similarity,
                    high_similarity_alerts,
                    -- Calculate 7-day moving average
                    AVG(entropy_index) OVER (
                        ORDER BY analysis_date 
                        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                    ) as moving_avg_7d,
                    -- Calculate daily change
                    LAG(entropy_index, 1) OVER (ORDER BY analysis_date) as prev_entropy,
                    entropy_index - LAG(entropy_index, 1) OVER (ORDER BY analysis_date) as daily_change
                FROM daily_entropy
            )
            SELECT 
                t.*,
                b.baseline_entropy,
                b.current_entropy,
                b.entropy_improvement,
                -- Success criteria
                CASE 
                    WHEN b.current_entropy < 0.7 AND b.baseline_entropy > 0.9 THEN 'EXCELLENT'
                    WHEN b.current_entropy < 0.7 AND b.baseline_entropy > 0.8 THEN 'GOOD'
                    WHEN b.current_entropy < 0.8 AND b.baseline_entropy > 0.7 THEN 'FAIR'
                    ELSE 'POOR'
                END as improvement_rating,
                -- Coherence level
                CASE 
                    WHEN t.entropy_index < 0.3 THEN 'Highly Coherent'
                    WHEN t.entropy_index < 0.5 THEN 'Moderately Coherent'
                    WHEN t.entropy_index < 0.7 THEN 'Somewhat Chaotic'
                    ELSE 'Highly Chaotic'
                END as coherence_level
            FROM trend_analysis t
            CROSS JOIN baseline_analysis b
            ORDER BY t.analysis_date DESC
            """
            
            entropy_result = self.client.query(entropy_query).to_dataframe()
            
            if not entropy_result.empty:
                latest = entropy_result.iloc[0]
                print(f"📊 Entropy Analysis (Last 14 days):")
                print(f"  🧠 Current Entropy (Last 7 days): {latest['current_entropy']:.3f}")
                print(f"  📈 Baseline Entropy (First 7 days): {latest['baseline_entropy']:.3f}")
                print(f"  📉 Entropy Improvement: {latest['entropy_improvement']:.3f}")
                print(f"  🎯 Improvement Rating: {latest['improvement_rating']}")
                print(f"  📊 Current Coherence: {latest['coherence_level']}")
                
                print(f"\n📈 Daily Entropy Trend:")
                for _, row in entropy_result.head(7).iterrows():
                    change_indicator = "📈" if row['daily_change'] > 0 else "📉" if row['daily_change'] < 0 else "➡️"
                    print(f"  {row['analysis_date']}: {row['entropy_index']:.3f} {change_indicator}")
                
                # Success criteria
                current_low = latest['current_entropy'] < 0.7
                baseline_high = latest['baseline_entropy'] > 0.9
                improvement_positive = latest['entropy_improvement'] > 0
                
                print(f"\n✅ ENTROPY SUCCESS CRITERIA:")
                print(f"  Current EI < 0.7: {'✅' if current_low else '❌'} ({latest['current_entropy']:.3f})")
                print(f"  Baseline EI > 0.9: {'✅' if baseline_high else '❌'} ({latest['baseline_entropy']:.3f})")
                print(f"  Positive Improvement: {'✅' if improvement_positive else '❌'} ({latest['entropy_improvement']:.3f})")
                
                success = current_low and baseline_high and improvement_positive
                print(f"\n🏆 ALERT ENTROPY: {'✅ SUCCESS' if success else '❌ NEEDS IMPROVEMENT'}")
                return success
            else:
                print("❌ No entropy data available")
                return False
                
        except Exception as e:
            print(f"❌ Error checking alert entropy: {e}")
            return False
    
    def generate_comprehensive_report(self):
        """Generate comprehensive Phase 1 critical indicators report"""
        print("🧠 AI-DRIVEN SOC PHASE 1 CRITICAL INDICATORS")
        print("=" * 60)
        print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Project: {self.project_id}")
        
        # Check all 3 critical indicators
        results = {
            "Embedding Pipeline": self.check_embedding_pipeline_health(),
            "RL Reward Trend": self.check_rl_reward_trend(),
            "Alert Entropy": self.check_alert_entropy_decrease()
        }
        
        # Overall assessment
        print(f"\n🏆 PHASE 1 CRITICAL INDICATORS SUMMARY")
        print("=" * 50)
        
        success_count = sum(results.values())
        total_indicators = len(results)
        
        for indicator, success in results.items():
            status = "✅ SUCCESS" if success else "❌ NEEDS WORK"
            print(f"  {indicator}: {status}")
        
        print(f"\n📊 OVERALL PHASE 1 STATUS: {success_count}/{total_indicators} indicators met")
        
        if success_count == total_indicators:
            print("🎉 PHASE 1 IS READY FOR PRODUCTION!")
            print("🚀 All critical indicators are successful")
        elif success_count >= 2:
            print("⚠️ PHASE 1 IS MOSTLY READY")
            print("🔧 Address the failed indicators before full deployment")
        else:
            print("❌ PHASE 1 NEEDS SIGNIFICANT WORK")
            print("🛠️ Focus on improving the critical indicators")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS")
        print("-" * 30)
        
        if not results["Embedding Pipeline"]:
            print("🔧 Fix embedding pipeline data quality issues")
        if not results["RL Reward Trend"]:
            print("🎯 Improve RL reward signal consistency")
        if not results["Alert Entropy"]:
            print("🧠 Optimize alert clustering to reduce entropy")
        
        if success_count == total_indicators:
            print("🚀 Proceed to Phase 2: Predictive Twin Fabric + Autonomous Bots")
        
        return results

def main():
    """Main function to run critical indicators analysis"""
    analyzer = Phase1CriticalIndicators()
    results = analyzer.generate_comprehensive_report()
    
    print(f"\n🎉 Analysis complete!")
    print(f"📊 Access KPI Dashboard: http://localhost:8528")
    print(f"🛡️ Access SOC Dashboard: http://localhost:8527")

if __name__ == "__main__":
    main()












