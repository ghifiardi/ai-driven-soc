#!/usr/bin/env python3
"""
Test ADA Agent with Real Data - Fixed Version
Show how your ADA agents are working with real data
"""

import json
import time
from datetime import datetime
from google.cloud import bigquery
import requests

def check_ada_services():
    """Check status of running ADA services"""
    
    print("🔍 Checking ADA Services Status...")
    print("=" * 50)
    
    # Check production ADA service
    try:
        response = requests.get("http://localhost:8080/status", timeout=5)
        if response.status_code == 200:
            status = response.json()
            print("✅ Production ADA Service:")
            print(f"   Status: {status.get('status', 'Unknown')}")
            print(f"   Model Loaded: {status.get('model_loaded', False)}")
            print(f"   Predictions: {status.get('predictions_count', 0)}")
            print(f"   Accuracy: {status.get('accuracy', 0):.1%}")
        else:
            print("❌ Production ADA Service: Not responding")
    except Exception as e:
        print(f"❌ Production ADA Service: Error - {e}")
    
    print()

def get_processed_alerts_from_bigquery():
    """Fetch processed alerts from BigQuery to see what ADA has analyzed"""
    
    print("📊 Fetching Processed Alerts from BigQuery...")
    
    try:
        client = bigquery.Client(project="chronicle-dev-2be9")
        
        # Query for processed alerts (what ADA has analyzed)
        query = """
        SELECT 
            alert_id,
            confidence_score,
            classification,
            timestamp,
            is_anomaly,
            raw_alert
        FROM `chronicle-dev-2be9.soc_data.processed_alerts`
        WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
        ORDER BY timestamp DESC
        LIMIT 10
        """
        
        query_job = client.query(query)
        results = query_job.result()
        
        alerts = []
        for row in results:
            alerts.append({
                "alert_id": row.alert_id,
                "confidence_score": row.confidence_score,
                "classification": row.classification,
                "timestamp": str(row.timestamp),
                "is_anomaly": row.is_anomaly,
                "raw_alert": row.raw_alert
            })
        
        print(f"✅ Found {len(alerts)} processed alerts from ADA")
        return alerts
        
    except Exception as e:
        print(f"❌ Error fetching processed alerts: {e}")
        return []

def get_raw_alerts_from_bigquery():
    """Fetch raw alerts from BigQuery to see what ADA is processing"""
    
    print("📊 Fetching Raw Alerts from BigQuery...")
    
    try:
        client = bigquery.Client(project="chronicle-dev-2be9")
        
        # Query for raw alerts (what ADA processes)
        query = """
        SELECT 
            log_id,
            timestamp,
            source_ip,
            dest_ip,
            protocol,
            port,
            bytes_sent,
            bytes_received,
            duration,
            raw_log
        FROM `chronicle-dev-2be9.soc_data.raw_alerts`
        WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
        ORDER BY timestamp DESC
        LIMIT 5
        """
        
        query_job = client.query(query)
        results = query_job.result()
        
        alerts = []
        for row in results:
            alerts.append({
                "log_id": row.log_id,
                "timestamp": str(row.timestamp),
                "source_ip": row.source_ip,
                "dest_ip": row.dest_ip,
                "protocol": row.protocol,
                "port": row.port,
                "bytes_sent": row.bytes_sent,
                "bytes_received": row.bytes_received,
                "duration": row.duration,
                "raw_log": row.raw_log
            })
        
        print(f"✅ Found {len(alerts)} raw alerts for ADA processing")
        return alerts
        
    except Exception as e:
        print(f"❌ Error fetching raw alerts: {e}")
        return []

def test_ada_with_real_data():
    """Test ADA with real data from BigQuery"""
    
    print("🧪 Testing ADA with Real Data")
    print("=" * 60)
    
    # Check ADA services
    check_ada_services()
    
    # Get processed alerts (what ADA has already analyzed)
    processed_alerts = get_processed_alerts_from_bigquery()
    
    if processed_alerts:
        print(f"\n📊 ADA PROCESSED ALERTS ANALYSIS:")
        print("=" * 50)
        
        for i, alert in enumerate(processed_alerts, 1):
            print(f"\n🔍 Processed Alert {i}:")
            print(f"   Alert ID: {alert['alert_id']}")
            print(f"   Timestamp: {alert['timestamp']}")
            print(f"   Classification: {alert['classification']}")
            print(f"   Is Anomaly: {alert['is_anomaly']}")
            print(f"   Confidence: {alert['confidence_score']:.3f}")
            
            # Parse raw alert data safely
            try:
                raw_alert = alert['raw_alert']
                if isinstance(raw_alert, str):
                    raw_alert = json.loads(raw_alert)
                
                print(f"   Source IP: {raw_alert.get('src_ip', 'Unknown')}")
                print(f"   Dest IP: {raw_alert.get('dst_ip', 'Unknown')}")
                print(f"   Protocol: {raw_alert.get('protocol', 'Unknown')}")
                print(f"   Port: {raw_alert.get('port', 'Unknown')}")
            except Exception as e:
                print(f"   Raw Alert: {str(raw_alert)[:100]}...")
    
    # Get raw alerts (what ADA processes)
    raw_alerts = get_raw_alerts_from_bigquery()
    
    if raw_alerts:
        print(f"\n📥 RAW ALERTS FOR ADA PROCESSING:")
        print("=" * 50)
        
        for i, alert in enumerate(raw_alerts, 1):
            print(f"\n🔍 Raw Alert {i}:")
            print(f"   Log ID: {alert['log_id']}")
            print(f"   Timestamp: {alert['timestamp']}")
            print(f"   Source: {alert['source_ip']} → {alert['dest_ip']}")
            print(f"   Protocol: {alert['protocol']}:{alert['port']}")
            print(f"   Data: {alert['bytes_sent']} sent, {alert['bytes_received']} received")
            print(f"   Duration: {alert['duration']}s")
            
            # Parse raw log safely
            try:
                raw_log = alert['raw_log']
                if isinstance(raw_log, str):
                    raw_log = json.loads(raw_log)
                print(f"   Method: {raw_log.get('methodName', 'Unknown')}")
                print(f"   Service: {raw_log.get('serviceName', 'Unknown')}")
            except Exception as e:
                print(f"   Raw Log: {str(raw_log)[:100]}...")
    
    # Test ADA with a new alert
    print(f"\n🧪 TESTING ADA WITH NEW ALERT:")
    print("=" * 50)
    
    # Create a test alert
    test_alert = {
        "log_id": f"test-ada-{int(time.time())}",
        "timestamp": datetime.now().isoformat(),
        "source_ip": "192.168.1.100",
        "dest_ip": "10.0.0.1",
        "protocol": "SSH",
        "port": 22,
        "bytes_sent": 1000000,
        "bytes_received": 500000,
        "duration": 30.5,
        "raw_log": json.dumps({
            "methodName": "ssh_connection",
            "serviceName": "compute.googleapis.com"
        })
    }
    
    print(f"📥 Test Alert:")
    print(f"   Log ID: {test_alert['log_id']}")
    print(f"   Source: {test_alert['source_ip']} → {test_alert['dest_ip']}")
    print(f"   Protocol: {test_alert['protocol']}:{test_alert['port']}")
    print(f"   Data: {test_alert['bytes_sent']} sent, {test_alert['bytes_received']} received")
    
    # Try to send to ADA service
    try:
        print(f"\n⚡ Sending to ADA Service...")
        response = requests.post(
            "http://localhost:8080/classify",
            json=test_alert,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ ADA Analysis Complete:")
            print(f"   Classification: {result.get('classification', 'Unknown')}")
            print(f"   Confidence: {result.get('confidence', 0):.3f}")
            print(f"   Is Anomaly: {result.get('is_anomaly', False)}")
            print(f"   Processing Time: {result.get('processing_time_ms', 0):.1f}ms")
        else:
            print(f"❌ ADA Service Error: {response.status_code}")
            print(f"   Response: {response.text}")
    
    except Exception as e:
        print(f"❌ Error sending to ADA: {e}")

def main():
    """Main function"""
    
    print("🚀 ADA Agent Real Data Test")
    print("=" * 60)
    print(f"Test Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    print("\n🔍 This test will:")
    print("   • Check your running ADA services")
    print("   • Show processed alerts from BigQuery")
    print("   • Show raw alerts being processed")
    print("   • Test ADA with a new alert")
    
    test_ada_with_real_data()
    
    print(f"\nTest Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

if __name__ == "__main__":
    main()
