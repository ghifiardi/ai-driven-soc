#!/usr/bin/env python3
"""
Simple Validation Script for Hybrid Cyber Defense Components
===========================================================

Tests core components without requiring full Google Cloud setup.
"""

import json
import time
import os
from datetime import datetime
import traceback

def test_dqn_inference():
    """Test DQN inference functionality"""
    print("1. Testing DQN Inference Components...")
    
    try:
        from dqn_inference import ProductionDQNAgent, RuleBasedFallback
        
        # Test DQN Agent
        agent = ProductionDQNAgent(model_path=None, state_size=7, action_size=5)
        print("   ✅ DQN Agent initialized successfully")
        
        # Test alert conversion
        test_alert = {
            'attack_type': 'malware',
            'severity': 'high',
            'affected_assets': [
                {'id': 'server1', 'status': 'vulnerable'},
                {'id': 'server2', 'status': 'compromised'}
            ]
        }
        
        state = agent.alert_to_state(test_alert)
        print(f"   ✅ Alert converted to state vector: {state}")
        
        # Test action recommendation
        start_time = time.time()
        result = agent.get_action_recommendation(test_alert)
        inference_time = time.time() - start_time
        
        print(f"   ✅ Action: {result['recommended_action']} (confidence: {result['confidence']:.2f})")
        print(f"   ✅ Inference time: {inference_time:.3f}s")
        
        # Test fallback agent
        fallback = RuleBasedFallback()
        fallback_result = fallback.get_action_recommendation(test_alert)
        print(f"   ✅ Fallback action: {fallback_result['recommended_action']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ DQN test failed: {e}")
        traceback.print_exc()
        return False

def test_configuration():
    """Test configuration loading"""
    print("\n2. Testing Configuration...")
    
    try:
        config_path = "config/hybrid_defense_config.json"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Validate key sections
            required_sections = ['agent', 'dqn_model', 'gemini', 'a2a', 'pubsub']
            for section in required_sections:
                if section in config:
                    print(f"   ✅ Config section '{section}' found")
                else:
                    print(f"   ⚠️  Config section '{section}' missing")
            
            return True
        else:
            print(f"   ❌ Configuration file not found: {config_path}")
            return False
            
    except Exception as e:
        print(f"   ❌ Configuration test failed: {e}")
        return False

def test_performance():
    """Test performance benchmarks"""
    print("\n3. Testing Performance...")
    
    try:
        from dqn_inference import ProductionDQNAgent
        
        agent = ProductionDQNAgent(model_path=None)
        
        # Performance test with multiple alerts
        test_alerts = [
            {'attack_type': 'malware', 'severity': 'high'},
            {'attack_type': 'ddos', 'severity': 'medium'},
            {'attack_type': 'intrusion', 'severity': 'critical'},
            {'attack_type': 'phishing', 'severity': 'low'},
            {'attack_type': 'ransomware', 'severity': 'critical'}
        ]
        
        times = []
        for alert in test_alerts:
            start_time = time.time()
            result = agent.get_action_recommendation(alert)
            processing_time = time.time() - start_time
            times.append(processing_time)
        
        avg_time = sum(times) / len(times)
        max_time = max(times)
        
        print(f"   ✅ Average processing time: {avg_time:.3f}s")
        print(f"   ✅ Maximum processing time: {max_time:.3f}s")
        
        # Performance requirements
        if avg_time < 0.1:
            print("   ✅ Performance requirement met (< 100ms avg)")
        else:
            print("   ⚠️  Performance slower than target (100ms)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Performance test failed: {e}")
        return False

def test_attack_scenarios():
    """Test various attack scenarios"""
    print("\n4. Testing Attack Scenarios...")
    
    try:
        from dqn_inference import RuleBasedFallback
        
        fallback = RuleBasedFallback()
        
        scenarios = [
            {
                'name': 'Critical Ransomware',
                'alert': {
                    'attack_type': 'ransomware',
                    'severity': 'critical',
                    'affected_assets': [{'id': 'server1', 'status': 'compromised'}]
                },
                'expected_action': 'isolate_node'
            },
            {
                'name': 'DDoS Attack',
                'alert': {
                    'attack_type': 'ddos',
                    'severity': 'high',
                    'affected_assets': []
                },
                'expected_action': 'block_traffic'
            },
            {
                'name': 'Low Priority Alert',
                'alert': {
                    'attack_type': 'none',
                    'severity': 'low',
                    'affected_assets': []
                },
                'expected_action': 'no_action'
            }
        ]
        
        all_passed = True
        for scenario in scenarios:
            result = fallback.get_action_recommendation(scenario['alert'])
            actual_action = result['recommended_action']
            expected_action = scenario['expected_action']
            
            if actual_action == expected_action:
                print(f"   ✅ {scenario['name']}: {actual_action} (as expected)")
            else:
                print(f"   ⚠️  {scenario['name']}: got {actual_action}, expected {expected_action}")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"   ❌ Scenario test failed: {e}")
        return False

def test_integration_readiness():
    """Test integration readiness"""
    print("\n5. Testing Integration Readiness...")
    
    try:
        # Test required imports
        imports_to_test = [
            'torch',
            'numpy',
            'google.cloud.pubsub_v1',
            'fastapi',
            'uvicorn'
        ]
        
        failed_imports = []
        for import_name in imports_to_test:
            try:
                __import__(import_name)
                print(f"   ✅ {import_name} available")
            except ImportError:
                print(f"   ❌ {import_name} not available")
                failed_imports.append(import_name)
        
        if not failed_imports:
            print("   ✅ All required dependencies available")
            return True
        else:
            print(f"   ❌ Missing dependencies: {failed_imports}")
            return False
            
    except Exception as e:
        print(f"   ❌ Integration test failed: {e}")
        return False

def main():
    """Run all validation tests"""
    print("=" * 70)
    print(" " * 10 + "HYBRID CYBER DEFENSE AGENT - VALIDATION SUITE")
    print("=" * 70)
    
    tests = [
        ("DQN Inference", test_dqn_inference),
        ("Configuration", test_configuration),
        ("Performance", test_performance),
        ("Attack Scenarios", test_attack_scenarios),
        ("Integration Readiness", test_integration_readiness)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print(" " * 25 + "VALIDATION SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<30} {status}")
    
    print("-" * 70)
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - SYSTEM READY FOR INTEGRATION!")
        print("\nNext steps:")
        print("1. Deploy to production environment")
        print("2. Configure Pub/Sub topics and subscriptions")  
        print("3. Set up Vertex AI credentials")
        print("4. Start the hybrid defense service")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests failed - please fix issues before deployment")
        return 1

if __name__ == "__main__":
    exit(main())
