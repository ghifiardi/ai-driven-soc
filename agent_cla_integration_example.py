#!/usr/bin/env python3
"""
Agent-CLA Integration Example

This script demonstrates how ADA, TAA, and CRA agents can integrate with
the production CLA service for enhanced classification and learning.
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional

class ProductionCLAClient:
    """Client for interacting with Production CLA Service"""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.timeout = 10
    
    def classify_alert(self, alert_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Classify an alert using production CLA"""
        try:
            response = self.session.post(
                f"{self.base_url}/classify",
                json=alert_data,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Classification failed: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error classifying alert: {e}")
            return None
    
    def get_service_status(self) -> Optional[Dict[str, Any]]:
        """Get production CLA service status"""
        try:
            response = self.session.get(f"{self.base_url}/status", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return None
        except Exception as e:
            print(f"Error getting status: {e}")
            return None
    
    def trigger_retrain(self) -> bool:
        """Trigger model retraining"""
        try:
            response = self.session.post(f"{self.base_url}/retrain", timeout=30)
            return response.status_code == 200
        except Exception as e:
            print(f"Error triggering retrain: {e}")
            return False

class ADA_Agent:
    """Anomaly Detection Agent with CLA Integration"""
    
    def __init__(self, cla_client: ProductionCLAClient):
        self.cla_client = cla_client
        self.agent_name = "ADA"
    
    def process_alert(self, raw_alert: Dict[str, Any]) -> Dict[str, Any]:
        """Process alert with CLA integration"""
        
        # Extract features for CLA
        alert_features = {
            "confidence_score": raw_alert.get("confidence", 0.5),
            "severity": raw_alert.get("severity", "UNKNOWN"),
            "bytes_transferred": raw_alert.get("bytes_transferred", 0),
            "connection_count": raw_alert.get("connection_count", 0),
            "protocol": raw_alert.get("protocol", "UNKNOWN"),
            "timestamp": raw_alert.get("timestamp", datetime.now().isoformat()),
            "description": raw_alert.get("description", ""),
            "source": raw_alert.get("source", ""),
            "destination": raw_alert.get("destination", "")
        }
        
        # Get CLA classification
        cla_result = self.cla_client.classify_alert(alert_features)
        
        if cla_result and 'error' not in cla_result:
            # Enhanced classification using CLA
            classification = cla_result['classification']
            confidence = cla_result['confidence']
            model_version = cla_result['model_version']
            
            print(f"🎯 ADA + CLA Classification: {classification.upper()}")
            print(f"   Confidence: {confidence:.1%}")
            print(f"   Model Version: {model_version}")
            
            return {
                "agent": self.agent_name,
                "classification": classification,
                "confidence": confidence,
                "cla_model_version": model_version,
                "timestamp": datetime.now().isoformat(),
                "enhanced_by_cla": True
            }
        else:
            # Fallback to ADA-only classification
            print("⚠️  CLA unavailable, using ADA-only classification")
            return {
                "agent": self.agent_name,
                "classification": "unknown",
                "confidence": 0.5,
                "cla_model_version": None,
                "timestamp": datetime.now().isoformat(),
                "enhanced_by_cla": False
            }

class TAA_Agent:
    """Threat Analysis Agent with CLA Integration"""
    
    def __init__(self, cla_client: ProductionCLAClient):
        self.cla_client = cla_client
        self.agent_name = "TAA"
    
    def analyze_threat(self, threat_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze threat with CLA integration"""
        
        # Prepare data for CLA
        alert_features = {
            "confidence_score": threat_data.get("threat_level", 0.5),
            "severity": threat_data.get("severity", "MEDIUM"),
            "bytes_transferred": threat_data.get("data_volume", 0),
            "connection_count": threat_data.get("connection_attempts", 0),
            "protocol": threat_data.get("protocol", "UNKNOWN"),
            "timestamp": threat_data.get("timestamp", datetime.now().isoformat()),
            "description": threat_data.get("threat_description", ""),
            "source": threat_data.get("source_ip", ""),
            "destination": threat_data.get("target_ip", "")
        }
        
        # Get CLA analysis
        cla_result = self.cla_client.classify_alert(alert_features)
        
        if cla_result and 'error' not in cla_result:
            classification = cla_result['classification']
            confidence = cla_result['confidence']
            
            print(f"🎯 TAA + CLA Analysis: {classification.upper()}")
            print(f"   Threat Confidence: {confidence:.1%}")
            
            return {
                "agent": self.agent_name,
                "threat_assessment": classification,
                "confidence": confidence,
                "cla_enhanced": True,
                "timestamp": datetime.now().isoformat()
            }
        else:
            print("⚠️  CLA unavailable, using TAA-only analysis")
            return {
                "agent": self.agent_name,
                "threat_assessment": "unknown",
                "confidence": 0.5,
                "cla_enhanced": False,
                "timestamp": datetime.now().isoformat()
            }

class CRA_Agent:
    """Containment Response Agent with CLA Integration"""
    
    def __init__(self, cla_client: ProductionCLAClient):
        self.cla_client = cla_client
        self.agent_name = "CRA"
    
    def determine_response(self, incident_data: Dict[str, Any]) -> Dict[str, Any]:
        """Determine response with CLA integration"""
        
        # Prepare data for CLA
        alert_features = {
            "confidence_score": incident_data.get("incident_severity", 0.5),
            "severity": incident_data.get("severity", "HIGH"),
            "bytes_transferred": incident_data.get("data_impact", 0),
            "connection_count": incident_data.get("affected_systems", 0),
            "protocol": incident_data.get("attack_vector", "UNKNOWN"),
            "timestamp": incident_data.get("timestamp", datetime.now().isoformat()),
            "description": incident_data.get("incident_description", ""),
            "source": incident_data.get("attacker_ip", ""),
            "destination": incident_data.get("target_system", "")
        }
        
        # Get CLA recommendation
        cla_result = self.cla_client.classify_alert(alert_features)
        
        if cla_result and 'error' not in cla_result:
            classification = cla_result['classification']
            confidence = cla_result['confidence']
            
            # Map classification to response action
            if classification == "anomaly":
                response_action = "IMMEDIATE_CONTAINMENT"
                response_level = "HIGH"
            else:
                response_action = "MONITOR"
                response_level = "LOW"
            
            print(f"🎯 CRA + CLA Response: {response_action}")
            print(f"   Confidence: {confidence:.1%}")
            print(f"   Response Level: {response_level}")
            
            return {
                "agent": self.agent_name,
                "response_action": response_action,
                "response_level": response_level,
                "confidence": confidence,
                "cla_recommended": True,
                "timestamp": datetime.now().isoformat()
            }
        else:
            print("⚠️  CLA unavailable, using CRA-only response")
            return {
                "agent": self.agent_name,
                "response_action": "STANDARD_RESPONSE",
                "response_level": "MEDIUM",
                "confidence": 0.5,
                "cla_recommended": False,
                "timestamp": datetime.now().isoformat()
            }

def demonstrate_agent_integration():
    """Demonstrate agent integration with production CLA"""
    
    print("🚀 Starting Agent-CLA Integration Demonstration")
    print("=" * 60)
    
    # Initialize CLA client
    cla_client = ProductionCLAClient()
    
    # Check CLA status
    status = cla_client.get_service_status()
    if status:
        print(f"✅ Production CLA Status: {status.get('status', 'unknown')}")
        print(f"   Model Loaded: {status.get('model_loaded', False)}")
        print(f"   Model Version: {status.get('model_version', 'unknown')}")
        print(f"   Predictions Made: {status.get('predictions_count', 0)}")
    else:
        print("❌ Production CLA not available")
        return
    
    print("\n" + "=" * 60)
    
    # Initialize agents
    ada_agent = ADA_Agent(cla_client)
    taa_agent = TAA_Agent(cla_client)
    cra_agent = CRA_Agent(cla_client)
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "Suspicious Network Activity",
            "ada_data": {
                "confidence": 0.9,
                "severity": "HIGH",
                "bytes_transferred": 50000,
                "connection_count": 100,
                "protocol": "TCP",
                "description": "Large data transfer to external IP",
                "source": "192.168.1.100",
                "destination": "external-site.com"
            },
            "taa_data": {
                "threat_level": 0.8,
                "severity": "HIGH",
                "data_volume": 50000,
                "connection_attempts": 100,
                "protocol": "TCP",
                "threat_description": "Potential data exfiltration",
                "source_ip": "192.168.1.100",
                "target_ip": "external-site.com"
            },
            "cra_data": {
                "incident_severity": 0.9,
                "severity": "CRITICAL",
                "data_impact": 50000,
                "affected_systems": 5,
                "attack_vector": "TCP",
                "incident_description": "Data breach attempt detected",
                "attacker_ip": "192.168.1.100",
                "target_system": "external-site.com"
            }
        },
        {
            "name": "Normal Web Traffic",
            "ada_data": {
                "confidence": 0.3,
                "severity": "LOW",
                "bytes_transferred": 1000,
                "connection_count": 5,
                "protocol": "HTTP",
                "description": "Normal web browsing",
                "source": "192.168.1.50",
                "destination": "192.168.1.1"
            },
            "taa_data": {
                "threat_level": 0.2,
                "severity": "LOW",
                "data_volume": 1000,
                "connection_attempts": 5,
                "protocol": "HTTP",
                "threat_description": "Standard web traffic",
                "source_ip": "192.168.1.50",
                "target_ip": "192.168.1.1"
            },
            "cra_data": {
                "incident_severity": 0.1,
                "severity": "LOW",
                "data_impact": 1000,
                "affected_systems": 1,
                "attack_vector": "HTTP",
                "incident_description": "Routine web activity",
                "attacker_ip": "192.168.1.50",
                "target_system": "192.168.1.1"
            }
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n📋 Scenario {i}: {scenario['name']}")
        print("-" * 40)
        
        # ADA processing
        print("🔍 ADA (Anomaly Detection Agent):")
        ada_result = ada_agent.process_alert(scenario['ada_data'])
        
        time.sleep(1)  # Small delay between agents
        
        # TAA processing
        print("\n🎯 TAA (Threat Analysis Agent):")
        taa_result = taa_agent.analyze_threat(scenario['taa_data'])
        
        time.sleep(1)  # Small delay between agents
        
        # CRA processing
        print("\n🛡️  CRA (Containment Response Agent):")
        cra_result = cra_agent.determine_response(scenario['cra_data'])
        
        print(f"\n📊 Summary for {scenario['name']}:")
        print(f"   ADA: {ada_result['classification']} ({ada_result['confidence']:.1%})")
        print(f"   TAA: {taa_result['threat_assessment']} ({taa_result['confidence']:.1%})")
        print(f"   CRA: {cra_result['response_action']} ({cra_result['confidence']:.1%})")
        
        if i < len(test_scenarios):
            print("\n" + "=" * 60)
            time.sleep(2)
    
    print("\n🎉 Agent-CLA Integration Demonstration Complete!")
    print("\n📈 Benefits of Integration:")
    print("   ✅ Enhanced classification accuracy")
    print("   ✅ Real-time learning from all agents")
    print("   ✅ Consistent threat assessment")
    print("   ✅ Automated response recommendations")
    print("   ✅ Continuous model improvement")

if __name__ == "__main__":
    demonstrate_agent_integration()


