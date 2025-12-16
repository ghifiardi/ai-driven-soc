#!/usr/bin/env python3
"""
Dashboard Improvement Monitor
============================

This script helps monitor the TAA dashboard for improvements from enhanced classification.
It tracks key metrics and provides alerts when improvements are detected.

Author: AI-Driven SOC Team
Date: 2025-09-18
"""

import requests
import time
import json
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import re

class DashboardMonitor:
    """Monitor TAA dashboard for improvements"""
    
    def __init__(self, dashboard_url="http://10.45.254.19:99"):
        self.dashboard_url = dashboard_url
        self.baseline_metrics = {
            'false_positive_rate': 66.7,  # Current baseline
            'false_alerts': 6777,
            'true_alerts': 3380,
            'total_alerts': 10157,
            'avg_confidence': 83.9,
            'avg_severity': 39.2,
            'low_confidence_alerts': 594,
            'low_severity_alerts': 7664,
            'system_status': 'Warning'
        }
        
        self.improvement_thresholds = {
            'false_positive_reduction': 5.0,  # 5% improvement
            'confidence_severity_improvement': 10.0,  # 10 point improvement
            'low_confidence_reduction': 50,  # 50 fewer alerts
            'low_severity_reduction': 100,  # 100 fewer alerts
        }
        
        print(f"🔍 Dashboard Monitor initialized")
        print(f"📊 Baseline metrics established:")
        print(f"   False Positive Rate: {self.baseline_metrics['false_positive_rate']}%")
        print(f"   System Status: {self.baseline_metrics['system_status']}")
        print(f"   Low Confidence Alerts: {self.baseline_metrics['low_confidence_alerts']}")
        print(f"   Low Severity Alerts: {self.baseline_metrics['low_severity_alerts']}")
    
    def extract_metrics_from_dashboard(self):
        """Extract metrics from the dashboard HTML"""
        try:
            response = requests.get(self.dashboard_url, timeout=10)
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            metrics = {}
            
            # Extract key metrics using text patterns
            text = soup.get_text()
            
            # Extract false positive rate
            false_positive_match = re.search(r'FALSE ALERTS.*?(\d+(?:\.\d+)?)%', text)
            if false_positive_match:
                metrics['false_positive_rate'] = float(false_positive_match.group(1))
            
            # Extract system status
            if 'WARNING' in text.upper():
                metrics['system_status'] = 'Warning'
            elif 'CRITICAL' in text.upper():
                metrics['system_status'] = 'Critical'
            elif 'OK' in text.upper() or 'GOOD' in text.upper():
                metrics['system_status'] = 'OK'
            else:
                metrics['system_status'] = 'Unknown'
            
            # Extract confidence and severity
            conf_match = re.search(r'Conf: ([\d.]+)%', text)
            if conf_match:
                metrics['avg_confidence'] = float(conf_match.group(1))
            
            sev_match = re.search(r'Sev: ([\d.]+)%', text)
            if sev_match:
                metrics['avg_severity'] = float(sev_match.group(1))
            
            # Extract alert counts
            false_alerts_match = re.search(r'FALSE ALERTS.*?(\d+) Alerts', text)
            if false_alerts_match:
                metrics['false_alerts'] = int(false_alerts_match.group(1))
            
            true_alerts_match = re.search(r'TRUE ALERTS.*?(\d+) Alerts', text)
            if true_alerts_match:
                metrics['true_alerts'] = int(true_alerts_match.group(1))
            
            total_alerts_match = re.search(r'TOTAL ALERTS.*?(\d+)', text)
            if total_alerts_match:
                metrics['total_alerts'] = int(total_alerts_match.group(1))
            
            # Extract low confidence/severity counts
            low_conf_match = re.search(r'LOW CONFIDENCE ALERTS.*?(\d+)', text)
            if low_conf_match:
                metrics['low_confidence_alerts'] = int(low_conf_match.group(1))
            
            low_sev_match = re.search(r'LOW SEVERITY ALERTS.*?(\d+)', text)
            if low_sev_match:
                metrics['low_severity_alerts'] = int(low_sev_match.group(1))
            
            return metrics
            
        except Exception as e:
            print(f"❌ Error extracting metrics: {e}")
            return None
    
    def analyze_improvements(self, current_metrics):
        """Analyze improvements compared to baseline"""
        improvements = []
        
        # Check false positive rate improvement
        if 'false_positive_rate' in current_metrics:
            improvement = self.baseline_metrics['false_positive_rate'] - current_metrics['false_positive_rate']
            if improvement >= self.improvement_thresholds['false_positive_reduction']:
                improvements.append({
                    'metric': 'False Positive Rate',
                    'improvement': f"{improvement:.1f}% reduction",
                    'baseline': f"{self.baseline_metrics['false_positive_rate']}%",
                    'current': f"{current_metrics['false_positive_rate']}%",
                    'status': '✅ IMPROVED'
                })
        
        # Check system status improvement
        if 'system_status' in current_metrics:
            if (self.baseline_metrics['system_status'] == 'Warning' and 
                current_metrics['system_status'] in ['OK', 'Good']):
                improvements.append({
                    'metric': 'System Status',
                    'improvement': 'Warning resolved',
                    'baseline': self.baseline_metrics['system_status'],
                    'current': current_metrics['system_status'],
                    'status': '🎉 MAJOR IMPROVEMENT'
                })
        
        # Check confidence/severity alignment
        if 'avg_confidence' in current_metrics and 'avg_severity' in current_metrics:
            conf_sev_diff = abs(current_metrics['avg_confidence'] - current_metrics['avg_severity'])
            baseline_diff = abs(self.baseline_metrics['avg_confidence'] - self.baseline_metrics['avg_severity'])
            alignment_improvement = baseline_diff - conf_sev_diff
            
            if alignment_improvement >= self.improvement_thresholds['confidence_severity_improvement']:
                improvements.append({
                    'metric': 'Confidence/Severity Alignment',
                    'improvement': f"{alignment_improvement:.1f} point improvement",
                    'baseline': f"Diff: {baseline_diff:.1f}",
                    'current': f"Diff: {conf_sev_diff:.1f}",
                    'status': '✅ IMPROVED'
                })
        
        # Check low confidence alerts reduction
        if 'low_confidence_alerts' in current_metrics:
            reduction = self.baseline_metrics['low_confidence_alerts'] - current_metrics['low_confidence_alerts']
            if reduction >= self.improvement_thresholds['low_confidence_reduction']:
                improvements.append({
                    'metric': 'Low Confidence Alerts',
                    'improvement': f"{reduction} fewer alerts",
                    'baseline': self.baseline_metrics['low_confidence_alerts'],
                    'current': current_metrics['low_confidence_alerts'],
                    'status': '✅ IMPROVED'
                })
        
        # Check low severity alerts reduction
        if 'low_severity_alerts' in current_metrics:
            reduction = self.baseline_metrics['low_severity_alerts'] - current_metrics['low_severity_alerts']
            if reduction >= self.improvement_thresholds['low_severity_reduction']:
                improvements.append({
                    'metric': 'Low Severity Alerts',
                    'improvement': f"{reduction} fewer alerts",
                    'baseline': self.baseline_metrics['low_severity_alerts'],
                    'current': current_metrics['low_severity_alerts'],
                    'status': '✅ IMPROVED'
                })
        
        return improvements
    
    def monitor_once(self):
        """Perform one monitoring check"""
        print(f"\n🔍 Monitoring check at {datetime.now().strftime('%H:%M:%S')}")
        
        current_metrics = self.extract_metrics_from_dashboard()
        if not current_metrics:
            print("❌ Could not extract metrics from dashboard")
            return False
        
        print("📊 Current Dashboard Metrics:")
        for metric, value in current_metrics.items():
            print(f"   {metric}: {value}")
        
        improvements = self.analyze_improvements(current_metrics)
        
        if improvements:
            print(f"\n🎉 IMPROVEMENTS DETECTED!")
            for improvement in improvements:
                print(f"   {improvement['status']} {improvement['metric']}")
                print(f"      Improvement: {improvement['improvement']}")
                print(f"      Baseline: {improvement['baseline']} → Current: {improvement['current']}")
            return True
        else:
            print("⏳ No significant improvements detected yet...")
            print("   Enhanced TAA is processing 25% of alerts - improvements should appear soon")
            return False
    
    def continuous_monitor(self, interval_minutes=5, max_checks=12):
        """Continuously monitor for improvements"""
        print(f"🚀 Starting continuous monitoring")
        print(f"📋 Check interval: {interval_minutes} minutes")
        print(f"📋 Maximum checks: {max_checks}")
        print(f"📋 Total monitoring time: {interval_minutes * max_checks} minutes")
        print(f"🔗 Dashboard URL: {self.dashboard_url}")
        
        improvements_detected = False
        check_count = 0
        
        try:
            while check_count < max_checks and not improvements_detected:
                improvements_detected = self.monitor_once()
                check_count += 1
                
                if not improvements_detected and check_count < max_checks:
                    print(f"⏳ Waiting {interval_minutes} minutes for next check...")
                    time.sleep(interval_minutes * 60)
            
            if improvements_detected:
                print(f"\n🎉 MONITORING COMPLETE - IMPROVEMENTS DETECTED!")
                print(f"📊 Enhanced TAA is successfully improving dashboard metrics")
                print(f"🎯 Consider increasing traffic to 50% or 100% for full benefits")
            else:
                print(f"\n⏳ MONITORING COMPLETE - No improvements detected yet")
                print(f"📊 Enhanced TAA may need more time to show improvements")
                print(f"💡 Consider monitoring for longer or checking enhanced TAA logs")
                
        except KeyboardInterrupt:
            print(f"\n🛑 Monitoring stopped by user")

def main():
    """Main monitoring function"""
    monitor = DashboardMonitor()
    
    print("\n" + "="*60)
    print("📊 TAA DASHBOARD IMPROVEMENT MONITOR")
    print("="*60)
    print("This monitor will check your dashboard for improvements")
    print("from the enhanced TAA system (currently processing 25% of alerts)")
    print("\nExpected improvements:")
    print("✅ Reduced false positive rate (currently 66.7%)")
    print("✅ Resolved confidence/severity mismatch warning")
    print("✅ Fewer low confidence alerts (currently 594)")
    print("✅ Fewer low severity alerts (currently 7,664)")
    print("✅ Better system status (currently Warning)")
    
    choice = input("\nChoose monitoring mode:\n1. Single check\n2. Continuous monitoring (5min intervals, 1 hour)\n3. Custom monitoring\nChoice (1-3): ")
    
    if choice == "1":
        monitor.monitor_once()
    elif choice == "2":
        monitor.continuous_monitor(interval_minutes=5, max_checks=12)
    elif choice == "3":
        interval = int(input("Check interval in minutes (default 5): ") or "5")
        max_checks = int(input("Maximum checks (default 12): ") or "12")
        monitor.continuous_monitor(interval_minutes=interval, max_checks=max_checks)
    else:
        print("Invalid choice, performing single check...")
        monitor.monitor_once()

if __name__ == "__main__":
    main()


