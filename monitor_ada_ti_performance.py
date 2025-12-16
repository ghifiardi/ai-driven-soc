#!/usr/bin/env python3
"""
Monitor ADA Performance Improvements with TI Integration
Tracks detection rates, false positive rates, and TI feature impact
"""

import json
import time
import asyncio
from datetime import datetime, timedelta
from google.cloud import bigquery
import pandas as pd
import numpy as np
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ADATIMonitor")

class ADATIPerformanceMonitor:
    def __init__(self, project_id: str = "chronicle-dev-2be9"):
        self.project_id = project_id
        self.bq_client = bigquery.Client(project=project_id)

    def get_recent_alerts_with_ti(self, hours: int = 24) -> pd.DataFrame:
        """Get recent processed alerts with TI information"""
        query = f"""
        SELECT
            alert_id,
            confidence_score,
            classification,
            is_anomaly,
            raw_alert
        FROM `chronicle-dev-2be9.soc_data.processed_alerts`
        WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {hours} HOUR)
        ORDER BY timestamp DESC
        LIMIT 1000
        """

        try:
            df = self.bq_client.query(query).to_dataframe()
            logger.info(f"Retrieved {len(df)} alerts from last {hours} hours")
            return df
        except Exception as e:
            logger.error(f"Error fetching alerts: {e}")
            return pd.DataFrame()

    def extract_ti_features_from_alert(self, alert_row) -> dict:
        """Extract TI features from a single alert row"""
        try:
            raw_alert = alert_row.get('raw_alert')
            if isinstance(raw_alert, str):
                raw_alert = json.loads(raw_alert)
            elif not isinstance(raw_alert, dict):
                return {}

            # Extract IPs
            src_ip = None
            dst_ip = None
            for key in ['src_ip', 'source_ip', 'dst_ip', 'dest_ip']:
                if key in raw_alert:
                    if 'src' in key:
                        src_ip = raw_alert[key]
                    elif 'dst' in key:
                        dst_ip = raw_alert[key]

            # Try to find TI in raw_alert
            ti_data = {}
            if isinstance(raw_alert, dict) and 'threat_intel' in raw_alert:
                ti_data = raw_alert['threat_intel'].get('abuseipdb', {})

            return {
                'alert_id': alert_row.get('alert_id'),
                'confidence': alert_row.get('confidence_score', 0),
                'classification': alert_row.get('classification'),
                'is_anomaly': alert_row.get('is_anomaly'),
                'src_ip': src_ip,
                'dst_ip': dst_ip,
                'src_ip_reputation': ti_data.get(src_ip, {}).get('abuseConfidenceScore', 0) if src_ip else 0,
                'dst_ip_reputation': ti_data.get(dst_ip, {}).get('abuseConfidenceScore', 0) if dst_ip else 0,
                'src_ip_reports': ti_data.get(src_ip, {}).get('totalReports', 0) if src_ip else 0,
                'dst_ip_reports': ti_data.get(dst_ip, {}).get('totalReports', 0) if dst_ip else 0
            }
        except Exception as e:
            logger.warning(f"Error extracting TI features: {e}")
            return {}

    def analyze_ti_impact(self, df: pd.DataFrame) -> dict:
        """Analyze the impact of TI features on detection performance"""

        if df.empty:
            return {"error": "No data available"}

        results = []

        for _, row in df.iterrows():
            features = self.extract_ti_features_from_alert(row)
            if features:
                results.append(features)

        if not results:
            return {"error": "No alerts with TI data found"}

        ti_df = pd.DataFrame(results)

        # Analyze detection performance by TI features
        analysis = {
            "total_alerts": len(ti_df),
            "anomalies_detected": len(ti_df[ti_df['is_anomaly'] == True]),
            "benign_classified": len(ti_df[ti_df['classification'] == 'benign']),
            "anomaly_classified": len(ti_df[ti_df['classification'] == 'anomaly']),
            "ti_coverage": {
                "src_ip_available": len(ti_df[ti_df['src_ip'].notna()]),
                "dst_ip_available": len(ti_df[ti_df['dst_ip'].notna()]),
                "ti_data_available": len(ti_df[(ti_df['src_ip_reputation'] > 0) | (ti_df['dst_ip_reputation'] > 0)])
            },
            "reputation_analysis": {
                "high_reputation_ips": len(ti_df[(ti_df['src_ip_reputation'] >= 70) | (ti_df['dst_ip_reputation'] >= 70)]),
                "avg_src_reputation": ti_df['src_ip_reputation'].mean(),
                "avg_dst_reputation": ti_df['dst_ip_reputation'].mean(),
                "max_reputation": max(ti_df['src_ip_reputation'].max(), ti_df['dst_ip_reputation'].max())
            },
            "correlation_analysis": {
                "confidence_vs_src_rep": ti_df['confidence'].corr(ti_df['src_ip_reputation']),
                "confidence_vs_dst_rep": ti_df['confidence'].corr(ti_df['dst_ip_reputation']),
                "anomaly_rate_by_high_rep": len(ti_df[
                    ((ti_df['src_ip_reputation'] >= 50) | (ti_df['dst_ip_reputation'] >= 50)) &
                    (ti_df['is_anomaly'] == True)
                ]) / len(ti_df[ti_df['src_ip_reputation'] >= 50]) if len(ti_df[ti_df['src_ip_reputation'] >= 50]) > 0 else 0
            }
        }

        return analysis

    def generate_performance_report(self, hours: int = 24) -> str:
        """Generate a comprehensive performance report"""
        df = self.get_recent_alerts_with_ti(hours)
        analysis = self.analyze_ti_impact(df)

        if "error" in analysis:
            return f"❌ Error: {analysis['error']}"

        report = f"""
# 📊 ADA TI Performance Report (Last {hours} hours)

## 📈 Overview
- **Total Alerts Processed**: {analysis['total_alerts']}
- **Anomalies Detected**: {analysis['anomalies_detected']} ({analysis['anomalies_detected']/max(analysis['total_alerts'],1)*100:.1f}%)
- **TI Coverage**: {analysis['ti_coverage']['ti_data_available']}/{analysis['total_alerts']} ({analysis['ti_coverage']['ti_data_available']/max(analysis['total_alerts'],1)*100:.1f}%)

## 🎯 Detection Performance
- **Anomaly Classification Rate**: {analysis['anomaly_classified']}/{analysis['total_alerts']} ({analysis['anomaly_classified']/max(analysis['total_alerts'],1)*100:.1f}%)
- **Benign Classification Rate**: {analysis['benign_classified']}/{analysis['total_alerts']} ({analysis['benign_classified']/max(analysis['total_alerts'],1)*100:.1f}%)

## 🔍 Threat Intelligence Impact
- **High Reputation IPs**: {analysis['reputation_analysis']['high_reputation_ips']} alerts with reputation ≥70
- **Average Source IP Reputation**: {analysis['reputation_analysis']['avg_src_reputation']:.2f}/100
- **Average Destination IP Reputation**: {analysis['reputation_analysis']['avg_dst_reputation']:.2f}/100
- **Highest Reputation Score**: {analysis['reputation_analysis']['max_reputation']:.0f}/100

## 📊 Correlation Analysis
- **Confidence vs Source Reputation**: {analysis['correlation_analysis']['confidence_vs_src_rep']:.3f}
- **Confidence vs Destination Reputation**: {analysis['correlation_analysis']['confidence_vs_dst_rep']:.3f}
- **Anomaly Rate for High Reputation IPs**: {analysis['correlation_analysis']['anomaly_rate_by_high_rep']:.1%} (vs overall {analysis['anomalies_detected']/max(analysis['total_alerts'],1)*100:.1f}%)

## 🎉 Key Insights
- **TI Integration Working**: {analysis['ti_coverage']['ti_data_available']} alerts have TI data
- **High Reputation Impact**: IPs with reputation ≥50 show {analysis['correlation_analysis']['anomaly_rate_by_high_rep']:.1%} anomaly rate vs {analysis['anomalies_detected']/max(analysis['total_alerts'],1)*100:.1f}% overall
- **Correlation Strength**: {abs(analysis['correlation_analysis']['confidence_vs_src_rep']):.2f} correlation between confidence and source IP reputation
"""
        return report

def main():
    """Main monitoring function"""
    print("🔍 Monitoring ADA TI Performance Improvements")
    print("=" * 60)

    monitor = ADATIPerformanceMonitor()

    # Generate reports for different time windows
    for hours in [1, 6, 24]:
        print(f"\n📊 {hours}-Hour Performance Report:")
        print("-" * 40)
        report = monitor.generate_performance_report(hours)
        print(report)

    print("\n" + "=" * 60)
    print("✅ Monitoring complete! Check the reports above for TI impact analysis.")

if __name__ == "__main__":
    main()
