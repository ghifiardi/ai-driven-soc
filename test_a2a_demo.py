#!/usr/bin/env python3
"""
TAA A2A Demo Script
==================

Simple demonstration of TAA A2A (Agent-to-Agent) communication.
This script shows the core concepts without requiring full deployment.
"""

import json
import time
from datetime import datetime
from typing import Dict, Any

def simulate_agent_card() -> Dict[str, Any]:
    """Simulate TAA agent card for discovery"""
    return {
        "name": "taa-security-agent",
        "version": "1.0.0",
        "description": "Triage and Analysis Agent for security alerts with A2A capabilities",
        "agent_id": f"taa-agent-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "capabilities": [
            {
                "name": "analyze_security_alert",
                "description": "Analyze security alerts using LLM and threat intelligence",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "alert_data": {"type": "object"},
                        "enrichment_level": {"type": "string", "enum": ["basic", "full"]}
                    },
                    "required": ["alert_data"]
                }
            },
            {
                "name": "share_threat_intelligence",
                "description": "Share threat intelligence with other agents",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "indicators": {"type": "array"},
                        "confidence": {"type": "number"},
                        "source": {"type": "string"}
                    }
                }
            },
            {
                "name": "request_containment",
                "description": "Request containment actions from other agents",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "target_host": {"type": "string"},
                        "action_type": {"type": "string", "enum": ["isolate", "block", "monitor"]},
                        "priority": {"type": "string", "enum": ["low", "medium", "high", "critical"]}
                    },
                    "required": ["target_host", "action_type"]
                }
            },
            {
                "name": "collaborative_analysis",
                "description": "Collaborate with other agents for complex analysis",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "analysis_type": {"type": "string"},
                        "data": {"type": "object"},
                        "collaboration_mode": {"type": "string", "enum": ["parallel", "sequential", "hierarchical"]}
                    }
                }
            }
        ],
        "authentication": {
            "type": "oauth2",
            "scopes": ["https://www.googleapis.com/auth/cloud-platform"]
        },
        "endpoints": {
            "base_url": "https://taa-agent.agentspace.googleapis.com",
            "a2a_endpoint": "/a2a/tasks",
            "health_check": "/health",
            "agent_card": "/.well-known/agent.json"
        },
        "metadata": {
            "domain": "cybersecurity",
            "specialization": "threat_analysis",
            "supported_formats": ["json", "xml"],
            "response_time": "5s",
            "availability": "24/7"
        }
    }

def simulate_a2a_task(task_id: str, skill: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate A2A task execution"""
    print(f"🔍 Processing A2A task: {task_id}")
    print(f"   Skill: {skill}")
    print(f"   Parameters: {json.dumps(parameters, indent=2)}")
    
    # Simulate processing time
    time.sleep(1)
    
    # Generate response based on skill
    if skill == "analyze_security_alert":
        result = {
            "status": "success",
            "analysis": {
                "is_true_positive": True,
                "severity": "high",
                "confidence": 0.92,
                "attack_type": "malware",
                "indicators": ["suspicious_activity", "malicious_ip"],
                "recommended_actions": ["isolate_host", "block_ip"],
                "analysis_summary": "High-confidence malware detection",
                "timestamp": datetime.now().isoformat()
            },
            "agent_id": "taa-agent-demo"
        }
    elif skill == "share_threat_intelligence":
        indicators = parameters.get("indicators", [])
        result = {
            "status": "success",
            "shared_indicators": len(indicators),
            "message": "Threat intelligence shared successfully",
            "data": {
                "indicators": indicators,
                "confidence": parameters.get("confidence", 0.0),
                "source": parameters.get("source", "unknown"),
                "agent_id": "taa-agent-demo",
                "timestamp": datetime.now().isoformat()
            }
        }
    elif skill == "request_containment":
        result = {
            "status": "success",
            "containment_requested": True,
            "target_host": parameters.get("target_host"),
            "action_type": parameters.get("action_type"),
            "priority": parameters.get("priority", "medium"),
            "message": f"Containment request submitted for {parameters.get('target_host')}",
            "agent_id": "taa-agent-demo"
        }
    elif skill == "collaborative_analysis":
        result = {
            "status": "success",
            "collaboration_mode": parameters.get("collaboration_mode", "parallel"),
            "analysis_type": parameters.get("analysis_type"),
            "participating_agents": 3,  # Simulated
            "message": f"Collaborative analysis completed in {parameters.get('collaboration_mode', 'parallel')} mode",
            "results": {
                "primary_analysis": "completed",
                "secondary_analysis": "completed",
                "consensus": "high_confidence"
            },
            "agent_id": "taa-agent-demo"
        }
    else:
        result = {
            "status": "error",
            "message": f"Unknown skill: {skill}",
            "agent_id": "taa-agent-demo"
        }
    
    return {
        "task_id": task_id,
        "status": "completed",
        "result": result,
        "agent_id": "taa-agent-demo",
        "timestamp": datetime.now().isoformat()
    }

def demo_agent_discovery():
    """Demonstrate agent discovery"""
    print("🔍 === Agent Discovery Demo ===")
    
    # Simulate agent card
    agent_card = simulate_agent_card()
    
    print(f"📋 Agent Card:")
    print(f"   Name: {agent_card['name']}")
    print(f"   Version: {agent_card['version']}")
    print(f"   Description: {agent_card['description']}")
    print(f"   Agent ID: {agent_card['agent_id']}")
    print(f"   Capabilities: {len(agent_card['capabilities'])} skills")
    
    print("\n🛠️  Available Capabilities:")
    for cap in agent_card['capabilities']:
        print(f"   - {cap['name']}: {cap['description']}")
    
    print(f"\n🌐 Endpoints:")
    print(f"   Base URL: {agent_card['endpoints']['base_url']}")
    print(f"   A2A Tasks: {agent_card['endpoints']['a2a_endpoint']}")
    print(f"   Health Check: {agent_card['endpoints']['health_check']}")
    print(f"   Agent Card: {agent_card['endpoints']['agent_card']}")
    
    return agent_card

def demo_security_alert_analysis():
    """Demonstrate security alert analysis"""
    print("\n🚨 === Security Alert Analysis Demo ===")
    
    # Create test alert
    test_alert = {
        "alert_id": "ALERT-2025-001",
        "timestamp": datetime.now().isoformat(),
        "source": "firewall",
        "severity": "high",
        "log_data": {
            "event_type": "process_creation",
            "process_name": "powershell.exe",
            "command_line": "powershell.exe -EncodedCommand JABw...",
            "user": "admin\\user",
            "hostname": "WORKSTATION-001",
            "src_ip": "192.168.1.100",
            "dst_ip": "8.8.8.8"
        }
    }
    
    # Create A2A task
    task = {
        "task_id": f"analysis-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "skill": "analyze_security_alert",
        "parameters": {
            "alert_data": test_alert,
            "enrichment_level": "full"
        }
    }
    
    # Execute task
    response = simulate_a2a_task(task["task_id"], task["skill"], task["parameters"])
    
    print(f"📊 Analysis Results:")
    if response["result"]["status"] == "success":
        analysis = response["result"]["analysis"]
        print(f"   True Positive: {analysis['is_true_positive']}")
        print(f"   Severity: {analysis['severity']}")
        print(f"   Confidence: {analysis['confidence']}")
        print(f"   Attack Type: {analysis['attack_type']}")
        print(f"   Indicators: {', '.join(analysis['indicators'])}")
        print(f"   Recommended Actions: {', '.join(analysis['recommended_actions'])}")
        print(f"   Summary: {analysis['analysis_summary']}")
    
    return response

def demo_threat_intelligence_sharing():
    """Demonstrate threat intelligence sharing"""
    print("\n🔄 === Threat Intelligence Sharing Demo ===")
    
    # Create A2A task
    task = {
        "task_id": f"ti-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "skill": "share_threat_intelligence",
        "parameters": {
            "indicators": [
                "192.168.1.100",
                "malicious-domain.com",
                "abc123def456ghi789",
                "suspicious-process.exe"
            ],
            "confidence": 0.85,
            "source": "taa_agent_analysis"
        }
    }
    
    # Execute task
    response = simulate_a2a_task(task["task_id"], task["skill"], task["parameters"])
    
    print(f"📤 Threat Intelligence Shared:")
    if response["result"]["status"] == "success":
        ti_data = response["result"]["data"]
        print(f"   Indicators Shared: {response['result']['shared_indicators']}")
        print(f"   Confidence: {ti_data['confidence']}")
        print(f"   Source: {ti_data['source']}")
        print(f"   Message: {response['result']['message']}")
    
    return response

def demo_containment_request():
    """Demonstrate containment request"""
    print("\n🛡️ === Containment Request Demo ===")
    
    # Create A2A task
    task = {
        "task_id": f"containment-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "skill": "request_containment",
        "parameters": {
            "target_host": "192.168.1.100",
            "action_type": "isolate",
            "priority": "high"
        }
    }
    
    # Execute task
    response = simulate_a2a_task(task["task_id"], task["skill"], task["parameters"])
    
    print(f"🔒 Containment Request:")
    if response["result"]["status"] == "success":
        print(f"   Target Host: {response['result']['target_host']}")
        print(f"   Action Type: {response['result']['action_type']}")
        print(f"   Priority: {response['result']['priority']}")
        print(f"   Status: {response['result']['containment_requested']}")
        print(f"   Message: {response['result']['message']}")
    
    return response

def demo_collaborative_analysis():
    """Demonstrate collaborative analysis"""
    print("\n🤝 === Collaborative Analysis Demo ===")
    
    # Create A2A task
    task = {
        "task_id": f"collab-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "skill": "collaborative_analysis",
        "parameters": {
            "analysis_type": "malware_analysis",
            "data": {
                "file_hash": "abc123def456ghi789",
                "file_type": "executable",
                "source": "email_attachment",
                "file_size": "2.5MB"
            },
            "collaboration_mode": "parallel"
        }
    }
    
    # Execute task
    response = simulate_a2a_task(task["task_id"], task["skill"], task["parameters"])
    
    print(f"🔬 Collaborative Analysis Results:")
    if response["result"]["status"] == "success":
        print(f"   Analysis Type: {response['result']['analysis_type']}")
        print(f"   Collaboration Mode: {response['result']['collaboration_mode']}")
        print(f"   Participating Agents: {response['result']['participating_agents']}")
        print(f"   Message: {response['result']['message']}")
        print(f"   Results: {json.dumps(response['result']['results'], indent=4)}")
    
    return response

def demo_multi_agent_workflow():
    """Demonstrate multi-agent workflow"""
    print("\n🔄 === Multi-Agent Workflow Demo ===")
    
    print("1. 🔍 ADA Agent detects anomaly and sends alert")
    print("2. 📊 TAA Agent receives alert and analyzes it")
    print("3. 🔄 TAA Agent shares threat intelligence with other agents")
    print("4. 🛡️ TAA Agent requests containment from security agents")
    print("5. 🤝 TAA Agent collaborates with forensics agents")
    print("6. 📈 TAA Agent sends feedback to CLA agent")
    
    # Simulate the workflow
    workflow_steps = [
        ("ADA Alert Detection", "Anomaly detected in network traffic"),
        ("TAA Analysis", "High-confidence malware detection"),
        ("Threat Intel Sharing", "3 indicators shared with 5 agents"),
        ("Containment Request", "Host isolation requested"),
        ("Collaborative Analysis", "Multi-agent analysis completed"),
        ("Feedback Loop", "Results sent to CLA agent")
    ]
    
    for step, description in workflow_steps:
        print(f"   ✅ {step}: {description}")
        time.sleep(0.5)
    
    print("\n🎯 Workflow completed successfully!")

def main():
    """Main demonstration function"""
    print("🚀 TAA A2A (Agent-to-Agent) Communication Demo")
    print("=" * 50)
    
    try:
        # Demo 1: Agent Discovery
        agent_card = demo_agent_discovery()
        
        # Demo 2: Security Alert Analysis
        analysis_response = demo_security_alert_analysis()
        
        # Demo 3: Threat Intelligence Sharing
        ti_response = demo_threat_intelligence_sharing()
        
        # Demo 4: Containment Request
        containment_response = demo_containment_request()
        
        # Demo 5: Collaborative Analysis
        collab_response = demo_collaborative_analysis()
        
        # Demo 6: Multi-Agent Workflow
        demo_multi_agent_workflow()
        
        print("\n" + "=" * 50)
        print("🎉 A2A Communication Demo Completed Successfully!")
        print("\n📋 Summary:")
        print(f"   • Agent discovered with {len(agent_card['capabilities'])} capabilities")
        print(f"   • {analysis_response['result']['analysis']['attack_type']} attack analyzed")
        print(f"   • {ti_response['result']['shared_indicators']} threat indicators shared")
        print(f"   • {containment_response['result']['action_type']} containment requested")
        print(f"   • {collab_response['result']['participating_agents']} agents collaborated")
        
        print("\n🔗 Next Steps:")
        print("   1. Deploy the full A2A implementation: ./deploy_taa_a2a.sh")
        print("   2. Start the A2A server: python3 taa_a2a_server.py")
        print("   3. Run comprehensive tests: python3 taa_a2a_test_client.py")
        print("   4. Integrate with Google Agentspace")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    main() 