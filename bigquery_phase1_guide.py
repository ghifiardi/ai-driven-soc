#!/usr/bin/env python3
"""
BigQuery Phase 1 Success Measurement Guide
Real-time insights into AI-Driven SOC Performance
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

class Phase1SuccessGuide:
    """Guide for measuring Phase 1 success with BigQuery"""
    
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
    
    def check_phase1_infrastructure(self):
        """Check if Phase 1 infrastructure is properly set up"""
        print("🔍 Checking Phase 1 Infrastructure...")
        
        if not self.client:
            print("❌ BigQuery client not available")
            return False
        
        # Check if tables and views exist
        checks = {
            "processed_alerts_table": self._check_table_exists("soc_data.processed_alerts"),
            "embedding_analysis_view": self._check_view_exists("soc_data.embedding_analysis_view"),
            "entropy_index_view": self._check_view_exists("soc_data.entropy_index_view"),
            "triage_time_analysis": self._check_view_exists("soc_data.triage_time_analysis"),
            "kpi_summary": self._check_view_exists("soc_data.kpi_summary")
        }
        
        print("\n📊 Infrastructure Status:")
        for check, status in checks.items():
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {check}")
        
        return all(checks.values())
    
    def _check_table_exists(self, table_name):
        """Check if a table exists"""
        try:
            table_ref = self.client.get_table(table_name)
            return table_ref is not None
        except:
            return False
    
    def _check_view_exists(self, view_name):
        """Check if a view exists"""
        try:
            view_ref = self.client.get_table(view_name)
            return view_ref is not None
        except:
            return False
    
    def get_phase1_success_metrics(self, days: int = 7):
        """Get comprehensive Phase 1 success metrics"""
        print(f"\n📈 Phase 1 Success Metrics (Last {days} days)")
        print("=" * 50)
        
        if not self.client:
            print("❌ BigQuery client not available")
            return None
        
        try:
            # 1. Alert Quality Metrics
            print("\n🎯 1. ALERT QUALITY METRICS")
            print("-" * 30)
            quality_metrics = self._get_alert_quality_metrics(days)
            self._display_quality_metrics(quality_metrics)
            
            # 2. Entropy Index Analysis
            print("\n🧠 2. SOC COHERENCE (ENTROPY INDEX)")
            print("-" * 30)
            entropy_metrics = self._get_entropy_analysis(days)
            self._display_entropy_metrics(entropy_metrics)
            
            # 3. Triage Efficiency
            print("\n⏱️ 3. TRIAGE EFFICIENCY")
            print("-" * 30)
            triage_metrics = self._get_triage_efficiency(days)
            self._display_triage_metrics(triage_metrics)
            
            # 4. Overall Phase 1 Assessment
            print("\n🏆 4. PHASE 1 SUCCESS ASSESSMENT")
            print("-" * 30)
            self._assess_phase1_success(quality_metrics, entropy_metrics, triage_metrics)
            
            return {
                'quality': quality_metrics,
                'entropy': entropy_metrics,
                'triage': triage_metrics
            }
            
        except Exception as e:
            print(f"❌ Error getting Phase 1 metrics: {e}")
            return None
    
    def _get_alert_quality_metrics(self, days: int):
        """Get alert quality metrics"""
        query = f"""
        WITH recent_metrics AS (
            SELECT 
                analysis_date,
                total_alerts,
                redundancy_percentage,
                uniqueness_percentage,
                coherence_rating,
                entropy_index
            FROM `{self.project_id}.soc_data.kpi_summary`
            WHERE analysis_date >= DATE_SUB(CURRENT_DATE(), INTERVAL {days} DAY)
            ORDER BY analysis_date DESC
        ),
        baseline_metrics AS (
            SELECT 
                AVG(redundancy_percentage) as avg_redundancy,
                AVG(uniqueness_percentage) as avg_uniqueness,
                AVG(entropy_index) as avg_entropy,
                COUNT(*) as days_analyzed
            FROM recent_metrics
        )
        SELECT 
            r.*,
            b.avg_redundancy,
            b.avg_uniqueness,
            b.avg_entropy,
            b.days_analyzed
        FROM recent_metrics r
        CROSS JOIN baseline_metrics b
        ORDER BY r.analysis_date DESC
        """
        
        result = self.client.query(query).to_dataframe()
        return result if not result.empty else None
    
    def _get_entropy_analysis(self, days: int):
        """Get entropy index analysis"""
        query = f"""
        WITH daily_entropy AS (
            SELECT 
                analysis_date,
                entropy_index,
                total_alerts,
                unique_clusters,
                coherence_level,
                entropy_change
            FROM `{self.project_id}.soc_data.entropy_index_view`
            WHERE analysis_date >= DATE_SUB(CURRENT_DATE(), INTERVAL {days} DAY)
            ORDER BY analysis_date ASC
        ),
        trend_analysis AS (
            SELECT 
                AVG(entropy_index) as avg_entropy,
                MIN(entropy_index) as min_entropy,
                MAX(entropy_index) as max_entropy,
                STDDEV(entropy_index) as entropy_volatility,
                COUNT(CASE WHEN entropy_change < 0 THEN 1 END) as improving_days,
                COUNT(CASE WHEN entropy_change > 0 THEN 1 END) as worsening_days
            FROM daily_entropy
        )
        SELECT 
            d.*,
            t.avg_entropy,
            t.min_entropy,
            t.max_entropy,
            t.entropy_volatility,
            t.improving_days,
            t.worsening_days
        FROM daily_entropy d
        CROSS JOIN trend_analysis t
        ORDER BY d.analysis_date DESC
        """
        
        result = self.client.query(query).to_dataframe()
        return result if not result.empty else None
    
    def _get_triage_efficiency(self, days: int):
        """Get triage efficiency metrics"""
        query = f"""
        WITH daily_triage AS (
            SELECT 
                analysis_date,
                similarity_group,
                alert_count,
                avg_processing_time_seconds,
                fast_processing_ratio,
                slow_processing_ratio,
                estimated_time_savings
            FROM `{self.project_id}.soc_data.triage_time_analysis`
            WHERE analysis_date >= DATE_SUB(CURRENT_DATE(), INTERVAL {days} DAY)
        ),
        efficiency_summary AS (
            SELECT 
                similarity_group,
                AVG(avg_processing_time_seconds) as avg_processing_time,
                AVG(fast_processing_ratio) as avg_fast_ratio,
                AVG(slow_processing_ratio) as avg_slow_ratio,
                SUM(estimated_time_savings) as total_time_savings,
                SUM(alert_count) as total_alerts
            FROM daily_triage
            GROUP BY similarity_group
        ),
        overall_efficiency AS (
            SELECT 
                AVG(avg_processing_time_seconds) as overall_avg_time,
                AVG(fast_processing_ratio) as overall_fast_ratio,
                SUM(estimated_time_savings) as total_time_savings
            FROM daily_triage
        )
        SELECT 
            e.*,
            o.overall_avg_time,
            o.overall_fast_ratio,
            o.total_time_savings
        FROM efficiency_summary e
        CROSS JOIN overall_efficiency o
        ORDER BY e.avg_processing_time ASC
        """
        
        result = self.client.query(query).to_dataframe()
        return result if not result.empty else None
    
    def _display_quality_metrics(self, metrics):
        """Display alert quality metrics"""
        if metrics is None or metrics.empty:
            print("❌ No quality metrics data available")
            return
        
        latest = metrics.iloc[0]
        print(f"📊 Total Alerts: {latest['total_alerts']:,}")
        print(f"🎯 Redundancy: {latest['redundancy_percentage']:.1f}% (Target: <20%)")
        print(f"✨ Uniqueness: {latest['uniqueness_percentage']:.1f}%")
        print(f"🧠 Coherence: {latest['coherence_rating']}")
        print(f"📈 Entropy Index: {latest['entropy_index']:.3f}")
        
        # Success indicators
        redundancy_success = latest['redundancy_percentage'] < 20
        coherence_success = latest['coherence_rating'] in ['Excellent', 'Good']
        
        print(f"\n✅ Redundancy Target: {'ACHIEVED' if redundancy_success else 'NEEDS IMPROVEMENT'}")
        print(f"✅ Coherence Target: {'ACHIEVED' if coherence_success else 'NEEDS IMPROVEMENT'}")
    
    def _display_entropy_metrics(self, metrics):
        """Display entropy analysis"""
        if metrics is None or metrics.empty:
            print("❌ No entropy metrics data available")
            return
        
        latest = metrics.iloc[0]
        print(f"📊 Average Entropy: {latest['avg_entropy']:.3f}")
        print(f"📈 Min Entropy: {latest['min_entropy']:.3f}")
        print(f"📉 Max Entropy: {latest['max_entropy']:.3f}")
        print(f"📊 Volatility: {latest['entropy_volatility']:.3f}")
        print(f"📈 Improving Days: {latest['improving_days']}")
        print(f"📉 Worsening Days: {latest['worsening_days']}")
        
        # Trend analysis
        if latest['improving_days'] > latest['worsening_days']:
            print("✅ TREND: Improving (More organized)")
        elif latest['improving_days'] < latest['worsening_days']:
            print("⚠️ TREND: Worsening (More chaotic)")
        else:
            print("➡️ TREND: Stable")
    
    def _display_triage_metrics(self, metrics):
        """Display triage efficiency metrics"""
        if metrics is None or metrics.empty:
            print("❌ No triage metrics data available")
            return
        
        print("📊 Processing Time by Similarity Group:")
        for _, row in metrics.iterrows():
            group = row['similarity_group'].replace('_', ' ').title()
            time = row['avg_processing_time']
            fast_ratio = row['avg_fast_ratio']
            print(f"  {group}: {time:.1f}s (Fast: {fast_ratio:.1%})")
        
        overall = metrics.iloc[0]  # Overall metrics are in first row
        print(f"\n⏱️ Overall Average Time: {overall['overall_avg_time']:.1f}s")
        print(f"🚀 Overall Fast Ratio: {overall['overall_fast_ratio']:.1%}")
        print(f"💾 Total Time Saved: {overall['total_time_savings']:.1f}s")
    
    def _assess_phase1_success(self, quality, entropy, triage):
        """Assess overall Phase 1 success"""
        print("🎯 PHASE 1 SUCCESS CRITERIA:")
        print("-" * 30)
        
        success_criteria = {
            "Redundancy Reduction": False,
            "Coherence Improvement": False,
            "Triage Efficiency": False,
            "Overall Success": False
        }
        
        if quality is not None and not quality.empty:
            latest_quality = quality.iloc[0]
            success_criteria["Redundancy Reduction"] = latest_quality['redundancy_percentage'] < 20
            success_criteria["Coherence Improvement"] = latest_quality['coherence_rating'] in ['Excellent', 'Good']
        
        if triage is not None and not triage.empty:
            overall_triage = triage.iloc[0]
            success_criteria["Triage Efficiency"] = overall_triage['overall_fast_ratio'] > 0.6
        
        success_count = sum(success_criteria.values())
        success_criteria["Overall Success"] = success_count >= 2
        
        for criterion, achieved in success_criteria.items():
            status = "✅ ACHIEVED" if achieved else "❌ NEEDS WORK"
            print(f"  {criterion}: {status}")
        
        print(f"\n🏆 OVERALL PHASE 1 SUCCESS: {success_count}/3 criteria met")
        
        if success_criteria["Overall Success"]:
            print("🎉 PHASE 1 IS SUCCESSFUL! Ready for Phase 2.")
        else:
            print("⚠️ PHASE 1 NEEDS IMPROVEMENT. Focus on failed criteria.")
    
    def get_real_time_insights(self):
        """Get real-time insights for SOC operations"""
        print("\n🔍 REAL-TIME SOC INSIGHTS")
        print("=" * 50)
        
        if not self.client:
            print("❌ BigQuery client not available")
            return
        
        try:
            # Current day insights
            query = f"""
            WITH today_metrics AS (
                SELECT 
                    analysis_date,
                    total_alerts,
                    redundancy_percentage,
                    entropy_index,
                    coherence_rating,
                    processing_speed_rating
                FROM `{self.project_id}.soc_data.kpi_summary`
                WHERE analysis_date = CURRENT_DATE()
            ),
            yesterday_metrics AS (
                SELECT 
                    total_alerts as yesterday_alerts,
                    redundancy_percentage as yesterday_redundancy,
                    entropy_index as yesterday_entropy
                FROM `{self.project_id}.soc_data.kpi_summary`
                WHERE analysis_date = DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)
            ),
            hourly_trend AS (
                SELECT 
                    EXTRACT(HOUR FROM embedding_timestamp) as hour,
                    COUNT(*) as alerts_per_hour,
                    AVG(embedding_similarity) as avg_similarity
                FROM `{self.project_id}.soc_data.processed_alerts`
                WHERE DATE(embedding_timestamp) = CURRENT_DATE()
                GROUP BY hour
                ORDER BY hour
            )
            SELECT 
                t.*,
                y.yesterday_alerts,
                y.yesterday_redundancy,
                y.yesterday_entropy,
                (t.total_alerts - y.yesterday_alerts) as alert_change,
                (t.redundancy_percentage - y.yesterday_redundancy) as redundancy_change,
                (t.entropy_index - y.yesterday_entropy) as entropy_change
            FROM today_metrics t
            CROSS JOIN yesterday_metrics y
            """
            
            result = self.client.query(query).to_dataframe()
            
            if not result.empty:
                today = result.iloc[0]
                print(f"📅 Today's SOC Performance ({today['analysis_date']})")
                print(f"📊 Alerts: {today['total_alerts']:,} ({today['alert_change']:+.0f} vs yesterday)")
                print(f"🎯 Redundancy: {today['redundancy_percentage']:.1f}% ({today['redundancy_change']:+.1f}% vs yesterday)")
                print(f"🧠 Entropy: {today['entropy_index']:.3f} ({today['entropy_change']:+.3f} vs yesterday)")
                print(f"📈 Coherence: {today['coherence_rating']}")
                print(f"⏱️ Processing: {today['processing_speed_rating']}")
                
                # Alert trends
                if today['alert_change'] > 0:
                    print("⚠️ Alert volume increased - monitor for potential incidents")
                elif today['alert_change'] < -10:
                    print("✅ Alert volume decreased - system may be stabilizing")
                
                # Redundancy trends
                if today['redundancy_change'] > 5:
                    print("⚠️ Redundancy increased - check embedding clustering")
                elif today['redundancy_change'] < -2:
                    print("✅ Redundancy decreased - embedding clustering working")
                
                # Entropy trends
                if today['entropy_change'] > 0.1:
                    print("⚠️ SOC becoming more chaotic - review alert patterns")
                elif today['entropy_change'] < -0.05:
                    print("✅ SOC becoming more organized - good progress")
            else:
                print("❌ No data available for today")
                
        except Exception as e:
            print(f"❌ Error getting real-time insights: {e}")
    
    def generate_phase1_report(self, days: int = 7):
        """Generate comprehensive Phase 1 report"""
        print(f"\n📋 PHASE 1 IMPLEMENTATION REPORT")
        print("=" * 60)
        print(f"Analysis Period: Last {days} days")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check infrastructure
        if not self.check_phase1_infrastructure():
            print("\n❌ Phase 1 infrastructure not properly set up")
            return
        
        # Get metrics
        metrics = self.get_phase1_success_metrics(days)
        
        # Get real-time insights
        self.get_real_time_insights()
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS")
        print("-" * 30)
        if metrics and metrics['quality'] is not None:
            latest_quality = metrics['quality'].iloc[0]
            if latest_quality['redundancy_percentage'] >= 20:
                print("🎯 Focus on reducing redundant alerts through better clustering")
            if latest_quality['coherence_rating'] not in ['Excellent', 'Good']:
                print("🧠 Improve SOC coherence by analyzing alert patterns")
        
        if metrics and metrics['triage'] is not None:
            overall_triage = metrics['triage'].iloc[0]
            if overall_triage['overall_fast_ratio'] < 0.6:
                print("⏱️ Optimize triage processing for faster alert handling")
        
        print("📊 Continue monitoring KPIs daily for Phase 1 success")
        print("🚀 Prepare for Phase 2: Predictive Twin Fabric + Autonomous Bots")

def main():
    """Main function to run Phase 1 success measurement"""
    print("🧠 AI-Driven SOC Phase 1 Success Measurement")
    print("=" * 60)
    
    guide = Phase1SuccessGuide()
    
    # Generate comprehensive report
    guide.generate_phase1_report(days=7)
    
    print(f"\n🎉 Phase 1 measurement complete!")
    print(f"📊 Access KPI Dashboard: http://localhost:8528")
    print(f"🛡️ Access SOC Dashboard: http://localhost:8527")

if __name__ == "__main__":
    main()












