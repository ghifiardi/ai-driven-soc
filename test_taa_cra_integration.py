#!/usr/bin/env python3
"""
Test script for TAA-CRA integration
"""

import json
import time
from google.cloud import pubsub_v1

def test_containment_request():
    """Test publishing a containment request to CRA"""
    
    # Initialize publisher
    publisher = pubsub_v1.PublisherClient()
    project_id = "chronicle-dev-2be9"
    topic_name = "containment-requests"
    topic_path = publisher.topic_path(project_id, topic_name)
    
    # Create test containment request
    containment_data = {
        "alert_id": "test-alert-001",
        "alert_data": {
            "src_ip": "192.168.1.100",
            "dst_ip": "10.0.0.1",
            "port": 22,
            "protocol": "SSH"
        },
        "llm_result": {
            "is_true_positive": True,
            "confidence": 0.95,
            "severity": "high",
            "reasoning": "Suspicious SSH activity detected"
        },
        "timestamp": "2025-09-25T02:47:00Z",
        "containment_request": {
            "action": "immediate_containment",
            "severity": "high",
            "confidence": 0.95,
            "reasoning": "High severity true positive detected: Suspicious SSH activity"
        }
    }
    
    # Publish the message
    message_data = json.dumps(containment_data).encode("utf-8")
    future = publisher.publish(topic_path, message_data)
    message_id = future.result()
    
    print(f"✅ Published containment request. Message ID: {message_id}")
    print(f"📋 Request details: {json.dumps(containment_data, indent=2)}")
    
    return message_id

if __name__ == "__main__":
    print("🧪 Testing TAA-CRA Integration...")
    test_containment_request()
    print("✅ Test completed. Check CRA service logs for processing.")
