#!/usr/bin/env python3
"""
Populate test data for Enhanced TAA Dashboard
"""

import json
import time
from datetime import datetime
from google.cloud import bigquery, pubsub_v1
from enhanced_taa_flash25 import EnhancedTAA

def populate_bigquery_test_data():
    """Populate BigQuery with test data"""
    
    print("📊 Populating BigQuery with test data...")
    
    try:
        client = bigquery.Client(project="chronicle-dev-2be9")
        table_id = "chronicle-dev-2be9.soc_data.processed_alerts"
        
        # Test data
        test_data = [
            {
                "alert_id": "enhanced-test-001",
                "timestamp": datetime.now().isoformat(),
                "is_anomaly": True,
                "confidence_score": 0.85,
                "classification": "malware",
                "raw_alert": json.dumps({
                    "src_ip": "192.168.1.100",
                    "dst_ip": "10.0.0.1",
                    "protocol": "SSH",
                    "port": 22,
                    "bytes_transferred": 1000000
                })
            },
            {
                "alert_id": "enhanced-test-002", 
                "timestamp": datetime.now().isoformat(),
                "is_anomaly": False,
                "confidence_score": 0.25,
                "classification": "normal",
                "raw_alert": json.dumps({
                    "src_ip": "192.168.1.10",
                    "dst_ip": "192.168.1.1",
                    "protocol": "HTTP",
                    "port": 80,
                    "bytes_transferred": 1000
                })
            },
            {
                "alert_id": "enhanced-test-003",
                "timestamp": datetime.now().isoformat(),
                "is_anomaly": True,
                "confidence_score": 0.92,
                "classification": "network_intrusion",
                "raw_alert": json.dumps({
                    "src_ip": "10.0.0.50",
                    "dst_ip": "external-site.com",
                    "protocol": "HTTPS",
                    "port": 443,
                    "bytes_transferred": 500000
                })
            },
            {
                "alert_id": "enhanced-test-004",
                "timestamp": datetime.now().isoformat(),
                "is_anomaly": False,
                "confidence_score": 0.15,
                "classification": "normal",
                "raw_alert": json.dumps({
                    "src_ip": "192.168.1.20",
                    "dst_ip": "192.168.1.1",
                    "protocol": "DNS",
                    "port": 53,
                    "bytes_transferred": 512
                })
            },
            {
                "alert_id": "enhanced-test-005",
                "timestamp": datetime.now().isoformat(),
                "is_anomaly": True,
                "confidence_score": 0.78,
                "classification": "phishing",
                "raw_alert": json.dumps({
                    "src_ip": "192.168.1.200",
                    "dst_ip": "suspicious-site.com",
                    "protocol": "HTTPS",
                    "port": 443,
                    "bytes_transferred": 250000
                })
            }
        ]
        
        # Insert data
        errors = client.insert_rows_json(table_id, test_data)
        
        if errors:
            print(f"❌ Errors inserting data: {errors}")
            return False
        else:
            print(f"✅ Successfully inserted {len(test_data)} test records")
            return True
            
    except Exception as e:
        print(f"❌ Error populating BigQuery: {e}")
        return False

def test_taa_and_populate_metrics():
    """Test TAA and populate performance metrics"""
    
    print("⚡ Testing TAA and populating metrics...")
    
    try:
        # Initialize TAA
        config = {
            "project_id": "chronicle-dev-2be9",
            "location": "us-central1"
        }
        taa = EnhancedTAA(config)
        
        # Test alerts
        test_alerts = [
            {
                "alert_id": "metrics-test-001",
                "src_ip": "192.168.1.100",
                "dst_ip": "10.0.0.1",
                "port": 22,
                "protocol": "SSH",
                "bytes_transferred": 1000000,
                "connection_count": 150,
                "timestamp": datetime.now().isoformat()
            },
            {
                "alert_id": "metrics-test-002",
                "src_ip": "10.0.0.50",
                "dst_ip": "external-site.com",
                "port": 443,
                "protocol": "HTTPS",
                "bytes_transferred": 500000,
                "connection_count": 25,
                "timestamp": datetime.now().isoformat()
            },
            {
                "alert_id": "metrics-test-003",
                "src_ip": "192.168.1.10",
                "dst_ip": "192.168.1.1",
                "port": 80,
                "protocol": "HTTP",
                "bytes_transferred": 1000,
                "connection_count": 1,
                "timestamp": datetime.now().isoformat()
            }
        ]
        
        # Process alerts to populate metrics
        for alert in test_alerts:
            print(f"Processing alert: {alert['alert_id']}")
            result = taa.analyze_alert(alert)
            print(f"  Result: {result.get('severity')} severity, {result.get('confidence'):.2f} confidence")
            time.sleep(1)  # Small delay between requests
        
        # Get final metrics
        metrics = taa.get_performance_metrics()
        print(f"✅ TAA metrics populated:")
        print(f"  Total Requests: {metrics['total_requests']}")
        print(f"  Avg Response Time: {metrics['avg_response_time']:.2f}ms")
        print(f"  Success Rate: {metrics['success_rate']*100:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing TAA: {e}")
        return False

def publish_test_messages():
    """Publish test messages to Pub/Sub topics"""
    
    print("📡 Publishing test messages to Pub/Sub...")
    
    try:
        publisher = pubsub_v1.PublisherClient()
        project_id = "chronicle-dev-2be9"
        
        topics = [
            "containment-requests",
            "taa-feedback",
            "taa-reports"
        ]
        
        for topic_name in topics:
            topic_path = publisher.topic_path(project_id, topic_name)
            
            test_message = {
                "alert_id": f"pubsub-test-{topic_name}",
                "timestamp": datetime.now().isoformat(),
                "message_type": topic_name,
                "test_data": True
            }
            
            message_data = json.dumps(test_message).encode("utf-8")
            future = publisher.publish(topic_path, message_data)
            message_id = future.result()
            
            print(f"✅ Published to {topic_name}: {message_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error publishing to Pub/Sub: {e}")
        return False

def main():
    """Main function to populate all test data"""
    
    print("🧪 Populating Enhanced TAA Dashboard Test Data")
    print("=" * 50)
    
    # Step 1: Populate BigQuery
    bq_success = populate_bigquery_test_data()
    
    # Step 2: Test TAA and populate metrics
    taa_success = test_taa_and_populate_metrics()
    
    # Step 3: Publish test messages
    pubsub_success = publish_test_messages()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Population Summary")
    print("=" * 50)
    
    print(f"BigQuery Data: {'✅ SUCCESS' if bq_success else '❌ FAILED'}")
    print(f"TAA Metrics: {'✅ SUCCESS' if taa_success else '❌ FAILED'}")
    print(f"Pub/Sub Messages: {'✅ SUCCESS' if pubsub_success else '❌ FAILED'}")
    
    overall_success = bq_success and taa_success and pubsub_success
    
    if overall_success:
        print("\n🎉 All test data populated successfully!")
        print("\n📊 Dashboard should now show:")
        print("- Real BigQuery data with alerts")
        print("- TAA performance metrics")
        print("- Active Pub/Sub integration status")
        print("- Working charts and visualizations")
    else:
        print("\n⚠️  Some test data population failed. Check errors above.")

if __name__ == "__main__":
    main()
