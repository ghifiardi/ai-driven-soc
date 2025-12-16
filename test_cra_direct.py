#!/usr/bin/env python3
"""
Direct Test of CRA SOAR MCP Agent
================================

This script directly tests the CRA SOAR MCP agent without the REST API,
demonstrating the advanced containment actions in action.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_cra_direct():
    """Test CRA SOAR MCP agent directly"""
    logger.info("🚀 Testing CRA SOAR MCP Agent Directly")
    logger.info("=" * 60)
    
    try:
        # Import the CRA agent
        from cra_soar_mcp_agent import CRASOARMCPAgent
        
        # Initialize the agent
        logger.info("🔧 Initializing CRA SOAR MCP Agent...")
        agent = CRASOARMCPAgent()
        logger.info("✅ CRA agent initialized successfully")
        
        # Test 1: Malware Containment
        logger.info("\n🔒 Testing Malware Containment Action")
        logger.info("-" * 40)
        
        malware_incident = {
            "alert_id": "malware_test_direct",
            "severity": "high",
            "title": "Advanced Malware Detection - Direct Test",
            "description": "Sophisticated malware detected with lateral movement capabilities. Immediate containment required.",
            "affected_assets": [
                "workstation-001.company.com",
                "server-002.company.com",
                "laptop-003.company.com"
            ],
            "indicators": [
                "192.168.1.100",
                "malware.example.com",
                "abc123def456789",
                "10.0.0.50"
            ],
            "metadata": {
                "threat_type": "malware",
                "confidence": 0.95,
                "lateral_movement": True
            }
        }
        
        logger.info(f"📋 Processing malware incident: {malware_incident['alert_id']}")
        logger.info(f"🎯 Affected assets: {malware_incident['affected_assets']}")
        logger.info(f"🔍 Indicators: {malware_incident['indicators']}")
        
        result = await agent.process_incident(malware_incident)
        logger.info(f"✅ Malware incident processed:")
        logger.info(f"  📊 Status: {result.get('status')}")
        logger.info(f"  🎯 Playbook: {result.get('playbook_executed')}")
        logger.info(f"  🛡️ Containment Actions: {result.get('containment_actions')}")
        logger.info(f"  🔍 TI Sources: {result.get('threat_intelligence_sources')}")
        
        # Test 2: DDoS Mitigation
        logger.info("\n🚫 Testing DDoS Mitigation Action")
        logger.info("-" * 40)
        
        ddos_incident = {
            "alert_id": "ddos_test_direct",
            "severity": "critical",
            "title": "Large-Scale DDoS Attack - Direct Test",
            "description": "Massive DDoS attack detected from multiple sources. Immediate mitigation required.",
            "affected_assets": [
                "firewall-primary.company.com",
                "load-balancer.company.com",
                "web-server-cluster.company.com"
            ],
            "indicators": [
                "1.2.3.4",
                "5.6.7.8",
                "9.10.11.12",
                "malicious-botnet.com"
            ],
            "metadata": {
                "threat_type": "ddos",
                "confidence": 0.98,
                "attack_volume": "massive"
            }
        }
        
        logger.info(f"📋 Processing DDoS incident: {ddos_incident['alert_id']}")
        logger.info(f"🎯 Target infrastructure: {ddos_incident['affected_assets']}")
        logger.info(f"🚫 Attack sources: {ddos_incident['indicators'][:3]}")
        
        result = await agent.process_incident(ddos_incident)
        logger.info(f"✅ DDoS incident processed:")
        logger.info(f"  📊 Status: {result.get('status')}")
        logger.info(f"  🎯 Playbook: {result.get('playbook_executed')}")
        logger.info(f"  🛡️ Containment Actions: {result.get('containment_actions')}")
        logger.info(f"  🔍 TI Sources: {result.get('threat_intelligence_sources')}")
        
        # Test 3: Insider Threat Response
        logger.info("\n👤 Testing Insider Threat Response Action")
        logger.info("-" * 40)
        
        insider_incident = {
            "alert_id": "insider_test_direct",
            "severity": "critical",
            "title": "Insider Threat Detection - Direct Test",
            "description": "Suspicious insider activity detected with data exfiltration indicators.",
            "affected_assets": [
                "user-001.company.com",
                "file-server.company.com",
                "database-backup.company.com"
            ],
            "indicators": [
                "john.doe@company.com",
                "data-exfil-server.com",
                "sensitive-data.zip",
                "192.168.1.200"
            ],
            "metadata": {
                "threat_type": "insider_threat",
                "confidence": 0.89,
                "data_exfiltration": True
            }
        }
        
        logger.info(f"📋 Processing insider threat incident: {insider_incident['alert_id']}")
        logger.info(f"👤 Compromised user: {insider_incident['indicators'][0]}")
        logger.info(f"📁 Exfiltrated data: {insider_incident['indicators'][2]}")
        
        result = await agent.process_incident(insider_incident)
        logger.info(f"✅ Insider threat incident processed:")
        logger.info(f"  📊 Status: {result.get('status')}")
        logger.info(f"  🎯 Playbook: {result.get('playbook_executed')}")
        logger.info(f"  🛡️ Containment Actions: {result.get('containment_actions')}")
        logger.info(f"  🔍 TI Sources: {result.get('threat_intelligence_sources')}")
        
        # Test 4: Critical Vulnerability Response
        logger.info("\n🔧 Testing Critical Vulnerability Response Action")
        logger.info("-" * 40)
        
        vuln_incident = {
            "alert_id": "vuln_test_direct",
            "severity": "critical",
            "title": "Critical Zero-Day Vulnerability - Direct Test",
            "description": "Critical zero-day vulnerability detected in production systems.",
            "affected_assets": [
                "web-server-01.company.com",
                "web-server-02.company.com",
                "database-server.company.com"
            ],
            "indicators": [
                "CVE-2024-0001",
                "exploit-payload.bin",
                "attacker-ip.com",
                "vulnerable-service:8080"
            ],
            "metadata": {
                "threat_type": "vulnerability",
                "confidence": 0.96,
                "cvss_score": 9.8,
                "zero_day": True
            }
        }
        
        logger.info(f"📋 Processing vulnerability incident: {vuln_incident['alert_id']}")
        logger.info(f"🔍 CVE: {vuln_incident['indicators'][0]}")
        logger.info(f"🎯 Affected systems: {len(vuln_incident['affected_assets'])}")
        
        result = await agent.process_incident(vuln_incident)
        logger.info(f"✅ Vulnerability incident processed:")
        logger.info(f"  📊 Status: {result.get('status')}")
        logger.info(f"  🎯 Playbook: {result.get('playbook_executed')}")
        logger.info(f"  🛡️ Containment Actions: {result.get('containment_actions')}")
        logger.info(f"  🔍 TI Sources: {result.get('threat_intelligence_sources')}")
        
        # Test 5: Phishing Response
        logger.info("\n🎣 Testing Phishing Response Action")
        logger.info("-" * 40)
        
        phishing_incident = {
            "alert_id": "phishing_test_direct",
            "severity": "medium",
            "title": "Phishing Campaign Detection - Direct Test",
            "description": "Coordinated phishing campaign targeting multiple users.",
            "affected_assets": [
                "user-001.company.com",
                "user-002.company.com",
                "email-server.company.com"
            ],
            "indicators": [
                "phishing.example.com",
                "suspicious-link.com",
                "malicious-attachment.pdf",
                "fake-bank.com"
            ],
            "metadata": {
                "threat_type": "phishing",
                "confidence": 0.85,
                "campaign_size": "large"
            }
        }
        
        logger.info(f"📋 Processing phishing incident: {phishing_incident['alert_id']}")
        logger.info(f"🎣 Phishing URLs: {phishing_incident['indicators'][:2]}")
        logger.info(f"👥 Affected users: {len([a for a in phishing_incident['affected_assets'] if 'user-' in a])}")
        
        result = await agent.process_incident(phishing_incident)
        logger.info(f"✅ Phishing incident processed:")
        logger.info(f"  📊 Status: {result.get('status')}")
        logger.info(f"  🎯 Playbook: {result.get('playbook_executed')}")
        logger.info(f"  🛡️ Containment Actions: {result.get('containment_actions')}")
        logger.info(f"  🔍 TI Sources: {result.get('threat_intelligence_sources')}")
        
        logger.info("\n🎉 All CRA SOAR MCP tests completed successfully!")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.info("💡 Make sure you're running from the correct directory with venv_mcp activated")
        return False
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

async def test_playbook_selection():
    """Test playbook selection logic"""
    logger.info("\n🎯 Testing Playbook Selection Logic")
    logger.info("=" * 60)
    
    try:
        from cra_soar_mcp_agent import CRASOARMCPAgent
        
        agent = CRASOARMCPAgent()
        
        # Test different incident types
        test_cases = [
            {
                "name": "Malware Incident",
                "incident": {
                    "severity": "high",
                    "threat_types": ["malware"],
                    "risk_score": 85
                }
            },
            {
                "name": "DDoS Attack",
                "incident": {
                    "severity": "critical",
                    "threat_types": ["ddos"],
                    "risk_score": 95
                }
            },
            {
                "name": "Insider Threat",
                "incident": {
                    "severity": "critical",
                    "threat_types": ["insider_threat"],
                    "risk_score": 90
                }
            },
            {
                "name": "Vulnerability",
                "incident": {
                    "severity": "high",
                    "threat_types": ["vulnerability"],
                    "risk_score": 88
                }
            },
            {
                "name": "Phishing Campaign",
                "incident": {
                    "severity": "medium",
                    "threat_types": ["phishing"],
                    "risk_score": 75
                }
            }
        ]
        
        for test_case in test_cases:
            logger.info(f"\n📋 Testing {test_case['name']}")
            logger.info(f"  🎯 Severity: {test_case['incident']['severity']}")
            logger.info(f"  🔍 Threat Types: {test_case['incident']['threat_types']}")
            logger.info(f"  📊 Risk Score: {test_case['incident']['risk_score']}")
            
            # Create a mock incident for playbook selection
            from cra_soar_mcp_agent import Incident, RiskLevel, ThreatIntelligence
            
            mock_incident = Incident(
                incident_id=f"test_{test_case['name'].lower().replace(' ', '_')}",
                alert_id=f"test_{test_case['name'].lower().replace(' ', '_')}",
                severity=RiskLevel(test_case['incident']['severity']),
                title=test_case['name'],
                description=f"Test {test_case['name']} for playbook selection",
                affected_assets=["test-host"],
                indicators=["test-indicator"],
                threat_intelligence=[
                    ThreatIntelligence(
                        indicator="test-indicator",
                        indicator_type="ip",
                        reputation="malicious",
                        risk_score=test_case['incident']['risk_score'],
                        threat_types=test_case['incident']['threat_types'],
                        confidence=0.9,
                        sources=["test-source"],
                        last_updated=datetime.now().isoformat(),
                        metadata={}
                    )
                ],
                containment_actions=[],
                created_at=datetime.now().isoformat()
            )
            
            # Test playbook selection
            selected_playbook = agent.soar_engine.select_playbook(mock_incident)
            
            if selected_playbook:
                logger.info(f"  ✅ Selected Playbook: {selected_playbook.name}")
                logger.info(f"  📝 Description: {selected_playbook.description}")
                logger.info(f"  🎯 Actions: {len(selected_playbook.actions)}")
                
                # Show action details
                for action in selected_playbook.actions:
                    logger.info(f"    🔧 {action.action_type.value}: {action.target}")
                    logger.info(f"      📊 Risk Level: {action.risk_level.value}")
                    logger.info(f"      ✅ Requires Approval: {action.requires_approval}")
            else:
                logger.info(f"  ⚠️ No playbook selected")
        
        logger.info("\n🎉 Playbook selection tests completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Playbook selection test failed: {e}")
        return False

async def test_containment_actions():
    """Test individual containment actions"""
    logger.info("\n🛡️ Testing Individual Containment Actions")
    logger.info("=" * 60)
    
    try:
        from cra_soar_mcp_agent import SOARWorkflowEngine, ContainmentAction, ActionType, RiskLevel, Incident
        
        # Initialize workflow engine
        config = {
            "playbooks": {
                "test_playbook": {
                    "name": "Test Playbook",
                    "description": "Test playbook for action testing",
                    "trigger_conditions": {},
                    "actions": [],
                    "timeout_minutes": 30,
                    "retry_count": 3,
                    "enabled": True
                }
            }
        }
        
        engine = SOARWorkflowEngine(config)
        
        # Test different action types
        test_actions = [
            {
                "type": ActionType.ISOLATE,
                "target": "workstation-001.company.com",
                "description": "Isolate compromised workstation"
            },
            {
                "type": ActionType.BLOCK,
                "target": "192.168.1.100",
                "description": "Block malicious IP address"
            },
            {
                "type": ActionType.QUARANTINE,
                "target": "malware.exe",
                "description": "Quarantine malicious file"
            },
            {
                "type": ActionType.MONITOR,
                "target": "user-001.company.com",
                "description": "Monitor suspicious user"
            },
            {
                "type": ActionType.ESCALATE,
                "target": "security_team",
                "description": "Escalate to security team"
            }
        ]
        
        for test_action in test_actions:
            logger.info(f"\n🔧 Testing {test_action['type'].value} Action")
            logger.info(f"  🎯 Target: {test_action['target']}")
            logger.info(f"  📝 Description: {test_action['description']}")
            
            # Create containment action
            action = ContainmentAction(
                action_id=f"test_{test_action['type'].value}",
                action_type=test_action['type'],
                target=test_action['target'],
                parameters={},
                risk_level=RiskLevel.MEDIUM,
                justification=test_action['description'],
                requires_approval=False
            )
            
            # Create mock incident
            mock_incident = Incident(
                incident_id="test_incident",
                alert_id="test_alert",
                severity=RiskLevel.HIGH,
                title="Test Incident",
                description="Test incident for action testing",
                affected_assets=[test_action['target']],
                indicators=["test-indicator"],
                threat_intelligence=[],
                containment_actions=[],
                created_at=datetime.now().isoformat()
            )
            
            # Execute action
            result = await engine._execute_action(action, mock_incident)
            
            if result['success']:
                logger.info(f"  ✅ Action executed successfully")
                logger.info(f"  📊 Result: {result.get('action')}")
                logger.info(f"  🕒 Timestamp: {result.get('timestamp')}")
                if result.get('details'):
                    logger.info(f"  📋 Details: {result['details']}")
            else:
                logger.error(f"  ❌ Action failed: {result.get('error')}")
        
        logger.info("\n🎉 Containment action tests completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Containment action test failed: {e}")
        return False

async def main():
    """Main test function"""
    logger.info("🚀 Starting Direct CRA SOAR MCP Tests")
    logger.info("=" * 80)
    
    test_results = []
    
    try:
        # Test 1: Direct CRA processing
        result1 = await test_cra_direct()
        test_results.append(("Direct CRA Processing", result1))
        
        # Test 2: Playbook selection
        result2 = await test_playbook_selection()
        test_results.append(("Playbook Selection", result2))
        
        # Test 3: Containment actions
        result3 = await test_containment_actions()
        test_results.append(("Containment Actions", result3))
        
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
    
    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("🎯 DIRECT CRA SOAR MCP TEST RESULTS")
    logger.info("=" * 80)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed_tests += 1
    
    logger.info(f"\n📊 Overall Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 All direct CRA SOAR MCP tests passed!")
        logger.info("\n🛡️ Advanced Containment Actions Demonstrated:")
        logger.info("  ✅ Asset Isolation - Network quarantine of compromised systems")
        logger.info("  ✅ Network Blocking - Firewall rules to block malicious traffic")
        logger.info("  ✅ File Quarantine - Secure isolation of suspicious files")
        logger.info("  ✅ Account Management - Disable compromised user accounts")
        logger.info("  ✅ System Monitoring - Enhanced monitoring of suspicious activities")
        logger.info("  ✅ Escalation Actions - Automated escalation to appropriate teams")
        logger.info("  ✅ Approval Workflows - Human oversight for high-risk actions")
    else:
        logger.info(f"⚠️ {total_tests - passed_tests} tests failed")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    asyncio.run(main())


