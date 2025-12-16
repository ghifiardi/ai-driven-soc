#!/usr/bin/env python3
"""
Test script for TAA Flash 2.5 integration
"""

import json
import time
from enhanced_taa_flash25 import EnhancedTAA

def test_flash_25_integration():
    """Test Flash 2.5 integration"""
    
    print("🚀 Testing TAA Flash 2.5 Integration...")
    print("=" * 50)
    
    # Initialize TAA
    config = {
        "project_id": "chronicle-dev-2be9",
        "location": "us-central1"
    }
    
    try:
        taa = EnhancedTAA(config)
        print("✅ Enhanced TAA initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize TAA: {e}")
        return False
    
    # Test cases
    test_cases = [
        {
            "name": "High Severity SSH Attack",
            "alert": {
                "alert_id": "test-001",
                "src_ip": "192.168.1.100",
                "dst_ip": "10.0.0.1",
                "port": 22,
                "protocol": "SSH",
                "bytes_transferred": 1000000,
                "connection_count": 150,
                "timestamp": "2025-09-25T10:30:00Z"
            }
        },
        {
            "name": "Suspicious Web Traffic",
            "alert": {
                "alert_id": "test-002",
                "src_ip": "10.0.0.50",
                "dst_ip": "external-site.com",
                "port": 443,
                "protocol": "HTTPS",
                "bytes_transferred": 500000,
                "connection_count": 25,
                "timestamp": "2025-09-25T10:35:00Z"
            }
        },
        {
            "name": "Normal Network Activity",
            "alert": {
                "alert_id": "test-003",
                "src_ip": "192.168.1.10",
                "dst_ip": "192.168.1.1",
                "port": 80,
                "protocol": "HTTP",
                "bytes_transferred": 1000,
                "connection_count": 1,
                "timestamp": "2025-09-25T10:40:00Z"
            }
        }
    ]
    
    # Run tests
    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: {test_case['name']}")
        print("-" * 30)
        
        start_time = time.time()
        
        try:
            result = taa.analyze_alert(test_case['alert'])
            processing_time = (time.time() - start_time) * 1000
            
            print(f"✅ Analysis completed in {processing_time:.2f}ms")
            print(f"   Model: {result.get('llm_model', 'unknown')}")
            print(f"   Severity: {result.get('severity', 'unknown')}")
            print(f"   Confidence: {result.get('confidence', 0):.2f}")
            print(f"   True Positive: {result.get('is_true_positive', False)}")
            print(f"   Threat Category: {result.get('threat_category', 'unknown')}")
            print(f"   Reasoning: {result.get('reasoning', 'No reasoning provided')[:100]}...")
            
            results.append({
                "test": test_case['name'],
                "success": True,
                "processing_time": processing_time,
                "result": result
            })
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            results.append({
                "test": test_case['name'],
                "success": False,
                "error": str(e)
            })
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print("=" * 50)
    
    successful_tests = sum(1 for r in results if r['success'])
    total_tests = len(results)
    
    print(f"Total Tests: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {total_tests - successful_tests}")
    print(f"Success Rate: {successful_tests/total_tests*100:.1f}%")
    
    if successful_tests > 0:
        avg_time = sum(r['processing_time'] for r in results if r['success']) / successful_tests
        print(f"Average Processing Time: {avg_time:.2f}ms")
    
    # Performance metrics
    try:
        metrics = taa.get_performance_metrics()
        print(f"\n📈 Performance Metrics:")
        print(f"   Total Requests: {metrics['total_requests']}")
        print(f"   Average Response Time: {metrics['avg_response_time']:.2f}ms")
        print(f"   Success Rate: {metrics['success_rate']*100:.1f}%")
    except Exception as e:
        print(f"⚠️  Could not retrieve performance metrics: {e}")
    
    return successful_tests == total_tests

if __name__ == "__main__":
    success = test_flash_25_integration()
    if success:
        print("\n🎉 All tests passed! Flash 2.5 integration is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
