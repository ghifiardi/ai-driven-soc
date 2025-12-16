#!/usr/bin/env python3
"""
Test TAA Flash 2.5 with ACTUAL BigQuery Data
This will show you real analysis of your real SOC data
"""

import json
import time
from datetime import datetime
from google.cloud import bigquery
from enhanced_taa_flash25 import EnhancedTAA

def get_real_alerts_from_bigquery():
    """Fetch ACTUAL alerts from your BigQuery for testing"""
    
    print("📊 Fetching REAL alerts from your BigQuery...")
    
    try:
        client = bigquery.Client(project="chronicle-dev-2be9")
        
        # Query for REAL alerts from your actual data
        query = """
        SELECT 
            alarmId,
            events,
            createdTime,
            severity,
            state,
            ruleName,
            description
        FROM `chronicle-dev-2be9.soc_data.alerts`
        WHERE createdTime >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
        ORDER BY createdTime DESC
        LIMIT 3
        """
        
        query_job = client.query(query)
        results = query_job.result()
        
        alerts = []
        for row in results:
            alerts.append({
                "alarmId": row.alarmId,
                "events": row.events,
                "createdTime": str(row.createdTime),
                "severity": row.severity,
                "state": row.state,
                "ruleName": row.ruleName,
                "description": row.description
            })
        
        print(f"✅ Found {len(alerts)} REAL alerts from your BigQuery")
        return alerts
        
    except Exception as e:
        print(f"❌ Error fetching real alerts: {e}")
        return []

def test_with_real_bigquery_data():
    """Test TAA Flash 2.5 with your ACTUAL BigQuery data"""
    
    print("🧪 Testing TAA Flash 2.5 with YOUR REAL BigQuery Data")
    print("=" * 70)
    
    try:
        # Initialize TAA
        config = {
            "project_id": "chronicle-dev-2be9",
            "location": "us-central1"
        }
        taa = EnhancedTAA(config)
        print("✅ TAA Flash 2.5 initialized successfully")
        
        # Get REAL alerts from your BigQuery
        real_alerts = get_real_alerts_from_bigquery()
        
        if not real_alerts:
            print("⚠️  No real alerts found in your BigQuery")
            print("   This could mean:")
            print("   • No alerts in the last 7 days")
            print("   • Different table structure")
            print("   • Permission issues")
            return
        
        # Test each REAL alert
        for i, alert in enumerate(real_alerts, 1):
            print(f"\n🔍 REAL ALERT {i} from YOUR BigQuery:")
            print("=" * 50)
            
            # Show the REAL alert data
            print("📥 REAL ALERT DATA:")
            print(f"   Alarm ID: {alert['alarmId']}")
            print(f"   Created Time: {alert['createdTime']}")
            print(f"   Severity: {alert['severity']}")
            print(f"   State: {alert['state']}")
            print(f"   Rule Name: {alert['ruleName']}")
            print(f"   Description: {alert['description']}")
            
            # Parse events data
            try:
                events_data = json.loads(alert['events'])
                print(f"   Events Data: {json.dumps(events_data, indent=2)}")
            except:
                print(f"   Events (raw): {alert['events'][:200]}...")
            
            # Convert to TAA format
            taa_alert = {
                "alert_id": alert['alarmId'],
                "src_ip": "unknown",  # Extract from events if available
                "dst_ip": "unknown",
                "port": 0,
                "protocol": "unknown",
                "bytes_transferred": 0,
                "connection_count": 1,
                "timestamp": alert['createdTime'],
                "description": alert['description'],
                "raw_alert": alert  # Include full alert data
            }
            
            print(f"\n⚡ ANALYZING with Google Flash 2.5...")
            start_time = time.time()
            
            try:
                # This is the REAL analysis
                result = taa.analyze_alert(taa_alert)
                processing_time = (time.time() - start_time) * 1000
                
                print(f"\n📊 REAL ANALYSIS RESULTS:")
                print(f"   🤖 Model: {result.get('llm_model', 'Unknown')}")
                print(f"   ✅ True Positive: {result.get('is_true_positive', False)}")
                print(f"   🚨 Severity: {result.get('severity', 'Unknown').upper()}")
                print(f"   🎯 Confidence: {result.get('confidence', 0):.3f}")
                print(f"   ⚠️  Risk Score: {result.get('risk_score', 0):.3f}")
                print(f"   🔥 Urgency: {result.get('urgency', 'Unknown').upper()}")
                print(f"   🎯 Threat Category: {result.get('threat_category', 'Unknown').upper()}")
                print(f"   📡 Attack Vector: {result.get('attack_vector', 'Unknown').upper()}")
                print(f"   🕒 Processing Time: {processing_time:.1f}ms")
                
                print(f"\n🧠 REAL AI REASONING:")
                reasoning = result.get('reasoning', 'No reasoning provided')
                print(f"   {reasoning}")
                
                print(f"\n💡 REAL RECOMMENDED ACTIONS:")
                actions = result.get('recommended_actions', [])
                for action in actions:
                    print(f"   • {action}")
                
                print(f"\n✅ REAL ALERT {i} ANALYSIS COMPLETE")
                
            except Exception as e:
                print(f"\n❌ REAL ANALYSIS FAILED: {e}")
        
        # Show final metrics
        metrics = taa.get_performance_metrics()
        print(f"\n📊 REAL PERFORMANCE METRICS:")
        print(f"   Total Requests: {metrics['total_requests']}")
        print(f"   Average Response Time: {metrics['avg_response_time']:.2f}ms")
        print(f"   Success Rate: {metrics['success_rate']*100:.1f}%")
        
    except Exception as e:
        print(f"❌ Real data test failed: {e}")

def main():
    """Main function"""
    
    print("🚀 TAA Flash 2.5 with YOUR REAL BigQuery Data")
    print("=" * 70)
    print(f"Test Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    print("\n🔍 This test will:")
    print("   • Fetch REAL alerts from your BigQuery")
    print("   • Analyze them with Google Flash 2.5")
    print("   • Show you ACTUAL AI reasoning")
    print("   • Display REAL performance metrics")
    
    test_with_real_bigquery_data()
    
    print(f"\nTest Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

if __name__ == "__main__":
    main()
