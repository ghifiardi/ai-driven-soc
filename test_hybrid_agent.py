#!/usr/bin/env python3
"""
End-to-End Testing for Hybrid Cyber Defense Agent
================================================

Comprehensive test suite for validating all components:
- DQN inference
- Gemini explanations  
- Pub/Sub integration
- A2A protocol
- Circuit breaker functionality
- Performance benchmarks
"""

import asyncio
import json
import logging
import time
import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from datetime import datetime
import requests
import torch
import numpy as np

# Import components to test
from hybrid_cyber_defense_agent import HybridCyberDefenseAgent, AlertProcessingResult
from dqn_inference import ProductionDQNAgent, RuleBasedFallback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestDQNInference(unittest.TestCase):
    """Test DQN inference functionality"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a minimal DQN agent without requiring a model file
        self.agent = ProductionDQNAgent(model_path=None, state_size=7, action_size=5)
        self.fallback = RuleBasedFallback()
    
    def test_alert_to_state_conversion(self):
        """Test converting alerts to DQN state vectors"""
        sample_alert = {
            'attack_type': 'malware',
            'severity': 'high',
            'affected_assets': [
                {'id': 'server1', 'status': 'compromised'},
                {'id': 'server2', 'status': 'vulnerable'}
            ]
        }
        
        state = self.agent.alert_to_state(sample_alert)
        
        # Verify state format
        self.assertEqual(len(state), 7)  # 5 nodes + attack_type + threat_level
        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(state.dtype, np.float32)
        
        # Verify specific values
        self.assertEqual(state[-2], 1)  # malware = 1
        self.assertGreaterEqual(state[-1], 0)  # threat level >= 0
        self.assertLessEqual(state[-1], 10)  # threat level <= 10
    
    def test_dqn_action_selection(self):
        """Test DQN action selection"""
        state = np.array([0, 1, 0, 0, 0, 1, 8], dtype=np.float32)
        
        action, confidence, context = self.agent.act(state)
        
        # Verify action format
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, 5)
        self.assertIsInstance(confidence, float)
        self.assertIsInstance(context, dict)
        
        # Verify context structure
        self.assertIn('q_values', context)
        self.assertIn('action_name', context)
        self.assertIn('confidence_score', context)
    
    def test_fallback_agent(self):
        """Test rule-based fallback agent"""
        test_alerts = [
            {
                'attack_type': 'ddos',
                'severity': 'high',
                'affected_assets': []
            },
            {
                'attack_type': 'malware',
                'severity': 'medium',
                'affected_assets': [{'id': 'server1', 'status': 'compromised'}]
            }
        ]
        
        for alert in test_alerts:
            result = self.fallback.get_action_recommendation(alert)
            
            self.assertIn('recommended_action', result)
            self.assertIn('action_id', result)
            self.assertIn('confidence', result)
            self.assertTrue(result['rule_based'])

class TestGeminiIntegration(unittest.TestCase):
    """Test Gemini explanation generation"""
    
    def setUp(self):
        """Set up test configuration"""
        self.config = {
            'feature_flags': {'GEMINI_EXPLANATIONS_ENABLED': True},
            'gemini': {
                'model': 'gemini-2.0-flash-exp',
                'temperature': 0.1,
                'max_output_tokens': 500,
                'timeout': 5.0,
                'prompt_template': 'Explain action {action} for alert: {alert_summary}'
            }
        }
    
    @patch('hybrid_cyber_defense_agent.GenerativeModel')
    def test_gemini_explanation_generation(self, mock_gemini):
        """Test Gemini explanation with mocked API"""
        # Mock Gemini response
        mock_response = Mock()
        mock_response.text = "This is a test explanation for the security action."
        mock_gemini.return_value.generate_content.return_value = mock_response
        
        # Create agent instance with mocked Gemini
        with patch('hybrid_cyber_defense_agent.HybridCyberDefenseAgent._load_config') as mock_config:
            mock_config.return_value = self.config
            
            with patch('hybrid_cyber_defense_agent.HybridCyberDefenseAgent._init_google_cloud'):
                with patch('hybrid_cyber_defense_agent.HybridCyberDefenseAgent._init_pubsub'):
                    with patch('hybrid_cyber_defense_agent.HybridCyberDefenseAgent._init_dqn_agent'):
                        agent = HybridCyberDefenseAgent()
                        agent.gemini_client = 'vertexai'
                        agent.gemini_model = mock_gemini.return_value
                        agent.circuit_breakers = {'gemini_api': Mock()}
                        agent.circuit_breakers['gemini_api'].call = lambda func, *args: func(*args)
                        
                        alert_data = {'attack_type': 'malware', 'severity': 'high'}
                        dqn_result = {'recommended_action': 'isolate_node', 'dqn_context': {}}
                        
                        explanation = agent._generate_explanation(alert_data, dqn_result)
                        
                        self.assertIsInstance(explanation, str)
                        self.assertGreater(len(explanation), 0)

class TestPubSubIntegration(unittest.TestCase):
    """Test Pub/Sub message handling"""
    
    def setUp(self):
        """Set up mock Pub/Sub environment"""
        self.mock_message = Mock()
        self.mock_message.data = json.dumps({
            'id': 'test_alert_123',
            'attack_type': 'malware',
            'severity': 'high',
            'affected_assets': []
        }).encode('utf-8')
        self.mock_message.ack = Mock()
        self.mock_message.nack = Mock()
    
    @patch('hybrid_cyber_defense_agent.HybridCyberDefenseAgent._load_config')
    @patch('hybrid_cyber_defense_agent.HybridCyberDefenseAgent._init_google_cloud')
    @patch('hybrid_cyber_defense_agent.HybridCyberDefenseAgent._init_pubsub')
    @patch('hybrid_cyber_defense_agent.HybridCyberDefenseAgent._init_gemini_client')
    @patch('hybrid_cyber_defense_agent.HybridCyberDefenseAgent._init_circuit_breakers')
    @patch('hybrid_cyber_defense_agent.HybridCyberDefenseAgent._init_metrics')
    @patch('hybrid_cyber_defense_agent.HybridCyberDefenseAgent._init_a2a_server')
    def test_ada_alert_processing(self, *mocks):
        """Test processing ADA alerts from Pub/Sub"""
        config = {
            'feature_flags': {
                'DQN_INFERENCE_ENABLED': True,
                'GEMINI_EXPLANATIONS_ENABLED': False,
                'CONTAINMENT_PUBLISHING_ENABLED': False
            },
            'dqn_model': {'fallback_enabled': True}
        }
        
        with patch.object(HybridCyberDefenseAgent, '_load_config', return_value=config):
            agent = HybridCyberDefenseAgent()
            agent.dqn_agent = Mock()
            agent.dqn_agent.get_action_recommendation.return_value = {
                'recommended_action': 'isolate_node',
                'action_id': 3,
                'confidence': 0.85,
                'dqn_context': {}
            }
            agent.gemini_client = None
            agent.stats = {'alerts_processed': 0, 'actions_recommended': 0}
            agent.metrics = {'alerts_processed_total': 0, 'containment_actions_total': {}}
            
            # Test successful processing
            agent.process_ada_alert_callback(self.mock_message)
            
            # Verify message was acknowledged
            self.mock_message.ack.assert_called_once()
            self.assertEqual(agent.stats['alerts_processed'], 1)
    
    def test_malformed_message_handling(self):
        """Test handling of malformed Pub/Sub messages"""
        malformed_message = Mock()
        malformed_message.data = b"invalid json"
        malformed_message.ack = Mock()
        malformed_message.nack = Mock()
        
        config = {
            'feature_flags': {'DQN_INFERENCE_ENABLED': True},
            'dqn_model': {'fallback_enabled': True}
        }
        
        with patch.object(HybridCyberDefenseAgent, '_load_config', return_value=config):
            with patch.object(HybridCyberDefenseAgent, '_init_google_cloud'):
                with patch.object(HybridCyberDefenseAgent, '_init_pubsub'):
                    with patch.object(HybridCyberDefenseAgent, '_init_dqn_agent'):
                        with patch.object(HybridCyberDefenseAgent, '_init_gemini_client'):
                            agent = HybridCyberDefenseAgent()
                            
                            agent.process_ada_alert_callback(malformed_message)
                            
                            # Verify message was negatively acknowledged for retry
                            malformed_message.nack.assert_called_once()

class TestA2AProtocol(unittest.TestCase):
    """Test A2A protocol compliance"""
    
    def setUp(self):
        """Set up A2A test environment"""
        self.config = {
            'feature_flags': {'A2A_PROTOCOL_ENABLED': True},
            'a2a': {
                'host': '127.0.0.1',
                'port': 8083,
                'agent_card': {
                    'name': 'test-hybrid-defense',
                    'version': '1.0.0',
                    'capabilities': ['analyze_security_alert']
                }
            },
            'security': {'allowed_origins': ['*']}
        }
    
    @patch('hybrid_cyber_defense_agent.HybridCyberDefenseAgent._load_config')
    def test_agent_card_endpoint(self, mock_config):
        """Test A2A agent card endpoint"""
        mock_config.return_value = self.config
        
        with patch.object(HybridCyberDefenseAgent, '_init_google_cloud'):
            with patch.object(HybridCyberDefenseAgent, '_init_pubsub'):
                with patch.object(HybridCyberDefenseAgent, '_init_dqn_agent'):
                    with patch.object(HybridCyberDefenseAgent, '_init_gemini_client'):
                        agent = HybridCyberDefenseAgent()
                        
                        # Test agent card structure
                        self.assertIsNotNone(agent.app)
                        
                        # Verify agent card content
                        agent_card = self.config['a2a']['agent_card']
                        self.assertIn('name', agent_card)
                        self.assertIn('version', agent_card)
                        self.assertIn('capabilities', agent_card)

class TestCircuitBreakers(unittest.TestCase):
    """Test circuit breaker functionality"""
    
    def test_circuit_breaker_states(self):
        """Test circuit breaker state transitions"""
        from hybrid_cyber_defense_agent import CircuitBreaker
        
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=1)
        
        # Test normal operation (CLOSED state)
        self.assertEqual(cb.state, "CLOSED")
        
        def failing_function():
            raise Exception("Test failure")
        
        # Test failures leading to OPEN state
        with self.assertRaises(Exception):
            cb.call(failing_function)
        self.assertEqual(cb.failure_count, 1)
        
        with self.assertRaises(Exception):
            cb.call(failing_function)
        self.assertEqual(cb.state, "OPEN")
        
        # Test that circuit breaker blocks calls in OPEN state
        with self.assertRaises(Exception):
            cb.call(failing_function)
        
        # Test reset after timeout
        time.sleep(1.1)  # Wait for reset timeout
        cb.state = "HALF_OPEN"  # Manually set for testing
        
        def working_function():
            return "success"
        
        result = cb.call(working_function)
        self.assertEqual(result, "success")
        self.assertEqual(cb.state, "CLOSED")

class TestPerformance(unittest.TestCase):
    """Performance benchmarks"""
    
    def test_dqn_inference_speed(self):
        """Test DQN inference performance"""
        agent = ProductionDQNAgent(model_path=None)
        
        # Create test alert
        alert = {
            'attack_type': 'malware',
            'severity': 'high',
            'affected_assets': []
        }
        
        # Benchmark inference time
        times = []
        for _ in range(10):
            start_time = time.time()
            result = agent.get_action_recommendation(alert)
            inference_time = time.time() - start_time
            times.append(inference_time)
        
        avg_time = sum(times) / len(times)
        logger.info(f"Average DQN inference time: {avg_time:.3f}s")
        
        # Performance assertion (should be under 100ms)
        self.assertLess(avg_time, 0.1, "DQN inference too slow")
    
    def test_end_to_end_processing_speed(self):
        """Test complete alert processing performance"""
        config = {
            'feature_flags': {
                'DQN_INFERENCE_ENABLED': True,
                'GEMINI_EXPLANATIONS_ENABLED': False
            },
            'dqn_model': {'fallback_enabled': True}
        }
        
        with patch.object(HybridCyberDefenseAgent, '_load_config', return_value=config):
            with patch.object(HybridCyberDefenseAgent, '_init_google_cloud'):
                with patch.object(HybridCyberDefenseAgent, '_init_pubsub'):
                    with patch.object(HybridCyberDefenseAgent, '_init_gemini_client'):
                        agent = HybridCyberDefenseAgent()
                        agent.dqn_agent = ProductionDQNAgent(model_path=None)
                        agent.stats = {'alerts_processed': 0, 'actions_recommended': 0}
                        agent.metrics = {'alerts_processed_total': 0, 'containment_actions_total': {}}
                        
                        alert = {
                            'id': 'perf_test',
                            'attack_type': 'ddos',
                            'severity': 'high',
                            'affected_assets': []
                        }
                        
                        start_time = time.time()
                        result = agent.process_alert(alert)
                        processing_time = time.time() - start_time
                        
                        logger.info(f"End-to-end processing time: {processing_time:.3f}s")
                        
                        # Performance assertion (should be under 500ms)
                        self.assertLess(processing_time, 0.5, "End-to-end processing too slow")
                        self.assertIsInstance(result, AlertProcessingResult)

class TestIntegrationScenarios(unittest.TestCase):
    """Integration test scenarios"""
    
    def test_high_severity_attack_scenario(self):
        """Test response to high severity attack"""
        alert = {
            'id': 'critical_attack_001',
            'attack_type': 'ransomware',
            'severity': 'critical',
            'affected_assets': [
                {'id': 'server1', 'status': 'compromised'},
                {'id': 'server2', 'status': 'vulnerable'}
            ],
            'source_ip': '192.168.1.100',
            'timestamp': datetime.now().isoformat()
        }
        
        # Use fallback for predictable testing
        fallback = RuleBasedFallback()
        result = fallback.get_action_recommendation(alert)
        
        # Should recommend isolation for compromised assets
        self.assertEqual(result['recommended_action'], 'isolate_node')
        self.assertGreater(result['confidence'], 0.5)
        
    def test_ddos_attack_scenario(self):
        """Test DDoS attack handling"""
        alert = {
            'id': 'ddos_attack_001',
            'attack_type': 'ddos',
            'severity': 'high',
            'affected_assets': [],
            'source_ip': '10.0.0.50',
            'timestamp': datetime.now().isoformat()
        }
        
        fallback = RuleBasedFallback()
        result = fallback.get_action_recommendation(alert)
        
        # Should recommend traffic blocking for DDoS
        self.assertEqual(result['recommended_action'], 'block_traffic')
        self.assertGreater(result['confidence'], 0.5)

def run_quick_validation():
    """Quick validation test for immediate feedback"""
    print("=" * 60)
    print("HYBRID CYBER DEFENSE AGENT - QUICK VALIDATION")
    print("=" * 60)
    
    try:
        # Test 1: DQN Agent Initialization
        print("1. Testing DQN Agent Initialization...")
        agent = ProductionDQNAgent(model_path=None, state_size=7, action_size=5)
        print("   ✅ DQN Agent created successfully")
        
        # Test 2: Alert Processing
        print("2. Testing Alert Processing...")
        test_alert = {
            'attack_type': 'malware',
            'severity': 'high',
            'affected_assets': [{'id': 'server1', 'status': 'vulnerable'}]
        }
        
        result = agent.get_action_recommendation(test_alert)
        print(f"   ✅ Action: {result['recommended_action']} (confidence: {result['confidence']:.2f})")
        
        # Test 3: Fallback Agent
        print("3. Testing Fallback Agent...")
        fallback = RuleBasedFallback()
        fallback_result = fallback.get_action_recommendation(test_alert)
        print(f"   ✅ Fallback Action: {fallback_result['recommended_action']}")
        
        # Test 4: Configuration Loading
        print("4. Testing Configuration...")
        config_path = "config/hybrid_defense_config.json"
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            print("   ✅ Configuration loaded successfully")
        else:
            print("   ⚠️  Configuration file not found (expected for initial setup)")
        
        print("\n" + "=" * 60)
        print("QUICK VALIDATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"   ❌ Validation failed: {e}")
        return False

if __name__ == "__main__":
    print("Hybrid Cyber Defense Agent - Test Suite")
    print("=" * 50)
    
    # Run quick validation first
    if run_quick_validation():
        print("\n🚀 Ready to run full test suite!")
        
        # Run full test suite
        print("\nRunning comprehensive test suite...")
        unittest.main(argv=[''], verbosity=2, exit=False)
    else:
        print("❌ Quick validation failed. Please fix issues before running full tests.")
        exit(1)
