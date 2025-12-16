#!/usr/bin/env python3
"""
Test Enhanced ADA-TAA Integration with Flash 2.5
"""

import asyncio
import json
import time
from datetime import datetime
from enhanced_ada_taa_integration import EnhancedADATAAAgent, EnhancedADAConfig

def test_enhanced_integration():
    """Test the enhanced ADA-TAA integration"""
    
    print("🚀 Testing Enhanced ADA-TAA Integration with Flash 2.5...")
    print("=" * 60)
    
    try:
        # Initialize the enhanced agent
        config = EnhancedADAConfig()
        agent = EnhancedADATAAAgent(config)
        print("✅ Enhanced ADA-TAA Agent initialized successfully")
        
        # Test alert data
        test_alerts = [
            {
                "alarmId": "test-enhanced-001",
                "events": json.dumps({
                    "timestamp": datetime.now().isoformat(),
                    "source_ip": "192.168.1.100",
                    "dest_ip": "10.0.0.1",
                    "protocol": "SSH",
                    "port": 22,
                    "bytes_sent": 1000000,
                    "bytes_received": 500000,
                    "duration": 30.5
                })
            },
            {
                "alarmId": "test-enhanced-002", 
                "events": json.dumps({
                    "timestamp": datetime.now().isoformat(),
                    "source_ip": "10.0.0.50",
                    "dest_ip": "external-site.com",
                    "protocol": "HTTPS",
                    "port": 443,
                    "bytes_sent": 500000,
                    "bytes_received": 250000,
                    "duration": 15.2
                })
            },
            {
                "alarmId": "test-enhanced-003",
                "events": json.dumps({
                    "timestamp": datetime.now().isoformat(),
                    "source_ip": "192.168.1.10",
                    "dest_ip": "192.168.1.1",
                    "protocol": "HTTP",
                    "port": 80,
                    "bytes_sent": 1000,
                    "bytes_received": 2000,
                    "duration": 1.5
                })
            }
        ]
        
        # Test each alert
        for i, test_alert in enumerate(test_alerts, 1):
            print(f"\n🧪 Test {i}: Processing alert {test_alert['alarmId']}")
            print("-" * 40)
            
            start_time = time.time()
            
            try:
                # Process alert through enhanced workflow
                result = asyncio.run(agent.process_single_alert(test_alert))
                processing_time = (time.time() - start_time) * 1000
                
                print(f"✅ Processing completed in {processing_time:.2f}ms")
                
                # Display key results
                if result and not result.get("error_messages"):
                    alert_data = result.get("alert_data", {})
                    detection_result = result.get("detection_result", {})
                    taa_result = result.get("taa_analysis_result", {})
                    
                    print(f"   Alert ID: {alert_data.get('log_id', 'Unknown')}")
                    print(f"   Anomaly Detected: {detection_result.get('is_anomaly', False)}")
                    print(f"   Detection Confidence: {detection_result.get('confidence', 0):.2f}")
                    
                    if taa_result:
                        print(f"   TAA Model: {taa_result.get('llm_model', 'Unknown')}")
                        print(f"   TAA Severity: {taa_result.get('severity', 'Unknown')}")
                        print(f"   TAA Confidence: {taa_result.get('confidence', 0):.2f}")
                        print(f"   Threat Category: {taa_result.get('threat_category', 'Unknown')}")
                        print(f"   TAA Processing Time: {taa_result.get('processing_time_ms', 0):.2f}ms")
                        print(f"   TAA Decision: {result.get('taa_decision', 'Unknown')}")
                    
                    # Check for integration actions
                    if result.get('containment_message_id'):
                        print(f"   🔗 Containment Request Published: {result['containment_message_id']}")
                    if result.get('feedback_message_id'):
                        print(f"   🔗 Feedback Published: {result['feedback_message_id']}")
                    if result.get('manual_review_flagged'):
                        print(f"   🔗 Manual Review Flagged: Yes")
                    
                else:
                    error_messages = result.get('error_messages', ['Unknown error'])
                    print(f"❌ Processing failed: {error_messages}")
                
            except Exception as e:
                print(f"❌ Test {i} failed with exception: {e}")
        
        # Get performance metrics
        print("\n📊 Performance Metrics")
        print("-" * 30)
        
        taa_metrics = agent.get_performance_metrics()
        print(f"TAA Total Requests: {taa_metrics.get('total_requests', 0)}")
        print(f"TAA Avg Response Time: {taa_metrics.get('avg_response_time', 0):.2f}ms")
        print(f"TAA Success Rate: {taa_metrics.get('success_rate', 0)*100:.1f}%")
        
        print("\n🎉 Enhanced ADA-TAA Integration Test Completed!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

async def test_workflow_directly():
    """Test the workflow directly with sample data"""
    
    print("\n🔬 Testing Workflow Directly...")
    print("-" * 30)
    
    try:
        config = EnhancedADAConfig()
        agent = EnhancedADATAAAgent(config)
        
        # Create test alert data
        test_alert_data = {
            "log_id": "direct-test-001",
            "timestamp": datetime.now().isoformat(),
            "source_ip": "192.168.1.100",
            "dest_ip": "10.0.0.1",
            "protocol": "SSH",
            "port": 22,
            "bytes_sent": 1000000,
            "bytes_received": 500000,
            "duration": 30.5,
            "raw_log": {
                "methodName": "ssh_connection",
                "serviceName": "ssh_service"
            }
        }
        
        # Create initial state
        initial_state = {
            "alert_data": test_alert_data,
            "batch_id": test_alert_data['log_id'],
            "error_messages": [],
            "workflow_status": "started"
        }
        
        print("Processing through enhanced workflow...")
        final_state = await agent.workflow.ainvoke(initial_state)
        
        print("✅ Workflow completed successfully!")
        print(f"Final state keys: {list(final_state.keys())}")
        
        # Display results
        if final_state.get("detection_result"):
            detection = final_state["detection_result"]
            print(f"Detection: Anomaly={detection.get('is_anomaly')}, Confidence={detection.get('confidence'):.2f}")
        
        if final_state.get("taa_analysis_result"):
            taa = final_state["taa_analysis_result"]
            print(f"TAA Analysis: Severity={taa.get('severity')}, Confidence={taa.get('confidence'):.2f}")
            print(f"TAA Model: {taa.get('llm_model')}")
            print(f"TAA Processing Time: {taa.get('processing_time_ms'):.2f}ms")
        
        if final_state.get("taa_decision"):
            print(f"TAA Decision: {final_state['taa_decision']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Direct workflow test failed: {e}")
        return False

def main():
    """Main test function"""
    
    print("🧪 Enhanced ADA-TAA Integration Test Suite")
    print("=" * 50)
    
    # Test 1: Full integration test
    test1_success = test_enhanced_integration()
    
    # Test 2: Direct workflow test
    test2_success = asyncio.run(test_workflow_directly())
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary")
    print("=" * 50)
    
    print(f"Integration Test: {'✅ PASSED' if test1_success else '❌ FAILED'}")
    print(f"Workflow Test: {'✅ PASSED' if test2_success else '❌ FAILED'}")
    
    overall_success = test1_success and test2_success
    print(f"\nOverall Result: {'🎉 ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    
    if overall_success:
        print("\n🚀 Your Enhanced ADA-TAA with Flash 2.5 is ready for production!")
        print("\nNext Steps:")
        print("1. Deploy to production environment")
        print("2. Start the enhanced dashboard")
        print("3. Monitor performance and optimize")
    else:
        print("\n⚠️  Please check the errors above before deploying to production")

if __name__ == "__main__":
    main()
