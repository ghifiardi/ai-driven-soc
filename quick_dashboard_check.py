#!/usr/bin/env python3
"""
Quick Dashboard Check
====================

Simple script to quickly check dashboard metrics for improvements.
"""

import requests
import re
from datetime import datetime

def check_dashboard():
    """Quick check of dashboard metrics"""
    try:
        print(f"🔍 Checking dashboard at {datetime.now().strftime('%H:%M:%S')}")
        
        response = requests.get("http://10.45.254.19:99", timeout=10)
        if response.status_code != 200:
            print(f"❌ Dashboard not accessible (Status: {response.status_code})")
            return
        
        text = response.text
        
        print("📊 Current Dashboard Metrics:")
        print("=" * 50)
        
        # Check for key metrics
        if "FALSE ALERTS" in text:
            print("✅ Dashboard is accessible and showing TAA metrics")
            
            # Extract key information
            if "WARNING" in text.upper():
                print("⚠️  System Status: WARNING (this should improve)")
            elif "CRITICAL" in text.upper():
                print("🚨 System Status: CRITICAL (this should improve)")
            else:
                print("✅ System Status: OK/Good (improvement detected!)")
            
            # Check for false positive indicators
            if "66.7%" in text:
                print("📊 False Positive Rate: Still 66.7% (baseline)")
            elif "%" in text:
                print("📊 False Positive Rate: Check dashboard for percentage")
            
            # Check for confidence/severity
            conf_match = re.search(r'Conf: ([\d.]+)%', text)
            sev_match = re.search(r'Sev: ([\d.]+)%', text)
            
            if conf_match and sev_match:
                conf = float(conf_match.group(1))
                sev = float(sev_match.group(1))
                diff = abs(conf - sev)
                print(f"📊 Confidence: {conf}% | Severity: {sev}% | Difference: {diff:.1f}")
                
                if diff < 30:
                    print("✅ Confidence/Severity: Well aligned (improvement!)")
                else:
                    print("⚠️  Confidence/Severity: Misaligned (should improve)")
            
            print("\n🎯 What to look for:")
            print("   • False positive rate should decrease from 66.7%")
            print("   • System status should change from WARNING")
            print("   • Confidence/Severity should become more aligned")
            print("   • Fewer low confidence/severity alerts")
            
        else:
            print("❌ Dashboard not showing TAA metrics")
            
    except Exception as e:
        print(f"❌ Error checking dashboard: {e}")

if __name__ == "__main__":
    check_dashboard()


