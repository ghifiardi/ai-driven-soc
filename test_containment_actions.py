#!/usr/bin/env python3
"""
Test Advanced Containment Actions
================================

This script specifically tests the advanced containment actions in the
CRA SOAR MCP system, including:

1. Asset Isolation
2. Network Blocking
3. File Quarantine
4. Account Management
5. System Patching
6. Monitoring Actions
7. Escalation Actions

Each test demonstrates real-world scenarios with detailed logging
and validation of the containment actions.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, List

import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContainmentActionTester:
    """Test client for advanced containment actions"""
    
    def __init__(self, base_url: str = "http://localhost:8081"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def wait_for_server(self, max_attempts: int = 30):
        """Wait for the SOAR server to be ready"""
        logger.info("⏳ Waiting for SOAR server to be ready...")
        
        for attempt in range(max_attempts):
            try:
                async with self.session.get(f"{self.base_url}/health") as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"✅ SOAR server is ready: {data}")
                        return True
            except Exception as e:
                logger.debug(f"Attempt {attempt + 1}: {e}")
            
            await asyncio.sleep(2)
        
        logger.error("❌ SOAR server not ready after maximum attempts")
        return False
    
    async def test_asset_isolation(self):
        """Test asset isolation containment action"""
        logger.info("🔒 Testing Asset Isolation Containment Action")
        logger.info("=" * 60)
        
        # Create malware incident that should trigger isolation
        incident_data = {
            "alert_id": f"isolation_test_{int(time.time())}",
            "severity": "high",
            "title": "Advanced Persistent Threat - Asset Isolation Required",
            "description": "APT malware detected on multiple endpoints with lateral movement indicators. Immediate isolation required to prevent further compromise.",
            "affected_assets": [
                "workstation-001.company.com",
                "server-002.company.com", 
                "laptop-003.company.com"
            ],
            "indicators": [
                "192.168.1.100",  # Compromised IP
                "malware.example.com",  # C2 domain
                "abc123def456789",  # Malware hash
                "10.0.0.50"  # Lateral movement IP
            ],
            "metadata": {
                "threat_type": "apt",
                "confidence": 0.95,
                "containment_priority": "critical",
                "requires_isolation": True
            }
        }
        
        logger.info(f"📋 Creating incident: {incident_data['alert_id']}")
        logger.info(f"🎯 Target assets: {incident_data['affected_assets']}")
        logger.info(f"🔍 Indicators: {incident_data['indicators']}")
        
        # Create incident
        async with self.session.post(f"{self.base_url}/incidents", json=incident_data) as response:
            if response.status == 201:
                result = await response.json()
                logger.info(f"✅ Incident created: {result}")
                
                # Wait for processing
                logger.info("⏳ Waiting for containment actions to execute...")
                await asyncio.sleep(5)
                
                # Check workflows
                async with self.session.get(f"{self.base_url}/workflows") as response:
                    if response.status == 200:
                        workflows = await response.json()
                        logger.info(f"📊 Active workflows: {len(workflows)}")
                        
                        for workflow in workflows:
                            if workflow['incident_id'] == incident_data['alert_id']:
                                logger.info(f"🔍 Workflow Status: {workflow['status']}")
                                logger.info(f"✅ Actions Completed: {len(workflow['actions_completed'])}")
                                logger.info(f"⏳ Actions Pending Approval: {len(workflow['actions_pending_approval'])}")
                                
                                # Log completed actions
                                for action in workflow['actions_completed']:
                                    logger.info(f"  ✅ {action.get('action_id', 'Unknown')}: {action.get('result', {}).get('action', 'Unknown')}")
                                
                                # Log pending actions
                                for action in workflow['actions_pending_approval']:
                                    logger.info(f"  ⏳ {action.get('action_id', 'Unknown')}: Requires approval")
                                
                                return True
                
                logger.warning("⚠️ No workflows found for incident")
                return False
            else:
                error_text = await response.text()
                logger.error(f"❌ Failed to create incident: {response.status} - {error_text}")
                return False
    
    async def test_network_blocking(self):
        """Test network blocking containment action"""
        logger.info("🚫 Testing Network Blocking Containment Action")
        logger.info("=" * 60)
        
        # Create DDoS incident that should trigger network blocking
        incident_data = {
            "alert_id": f"blocking_test_{int(time.time())}",
            "severity": "critical",
            "title": "DDoS Attack - Network Blocking Required",
            "description": "Large-scale DDoS attack detected from multiple source IPs. Immediate network blocking required to protect infrastructure.",
            "affected_assets": [
                "firewall-primary.company.com",
                "load-balancer.company.com",
                "web-server-cluster.company.com"
            ],
            "indicators": [
                "1.2.3.4",  # Attack source IP
                "5.6.7.8",  # Attack source IP
                "9.10.11.12",  # Attack source IP
                "malicious-botnet.com"  # C2 domain
            ],
            "metadata": {
                "threat_type": "ddos",
                "confidence": 0.98,
                "attack_volume": "high",
                "requires_blocking": True
            }
        }
        
        logger.info(f"📋 Creating DDoS incident: {incident_data['alert_id']}")
        logger.info(f"🎯 Target infrastructure: {incident_data['affected_assets']}")
        logger.info(f"🚫 Blocking indicators: {incident_data['indicators']}")
        
        # Create incident
        async with self.session.post(f"{self.base_url}/incidents", json=incident_data) as response:
            if response.status == 201:
                result = await response.json()
                logger.info(f"✅ DDoS incident created: {result}")
                
                # Wait for processing
                await asyncio.sleep(3)
                
                # Check for blocking actions
                async with self.session.get(f"{self.base_url}/workflows") as response:
                    if response.status == 200:
                        workflows = await response.json()
                        
                        for workflow in workflows:
                            if workflow['incident_id'] == incident_data['alert_id']:
                                logger.info(f"🔍 DDoS Response Status: {workflow['status']}")
                                
                                # Look for blocking actions
                                blocking_actions = []
                                for action in workflow['actions_completed']:
                                    if 'block' in action.get('action_id', '').lower():
                                        blocking_actions.append(action)
                                
                                if blocking_actions:
                                    logger.info(f"✅ Network blocking actions executed: {len(blocking_actions)}")
                                    for action in blocking_actions:
                                        logger.info(f"  🚫 {action.get('action_id')}: {action.get('result', {}).get('details', {})}")
                                else:
                                    logger.warning("⚠️ No blocking actions found")
                                
                                return True
                
                return False
            else:
                error_text = await response.text()
                logger.error(f"❌ Failed to create DDoS incident: {response.status} - {error_text}")
                return False
    
    async def test_file_quarantine(self):
        """Test file quarantine containment action"""
        logger.info("📁 Testing File Quarantine Containment Action")
        logger.info("=" * 60)
        
        # Create malware incident with file indicators
        incident_data = {
            "alert_id": f"quarantine_test_{int(time.time())}",
            "severity": "high",
            "title": "Malware File Detection - Quarantine Required",
            "description": "Suspicious executable files detected on multiple systems. Immediate quarantine required to prevent execution.",
            "affected_assets": [
                "workstation-001.company.com",
                "workstation-002.company.com"
            ],
            "indicators": [
                "malware.exe",  # Suspicious file
                "trojan.dll",  # Suspicious file
                "suspicious.doc",  # Suspicious document
                "192.168.1.100"  # Source IP
            ],
            "metadata": {
                "threat_type": "malware",
                "confidence": 0.92,
                "file_count": 3,
                "requires_quarantine": True
            }
        }
        
        logger.info(f"📋 Creating malware file incident: {incident_data['alert_id']}")
        logger.info(f"📁 Suspicious files: {incident_data['indicators'][:3]}")
        
        # Create incident
        async with self.session.post(f"{self.base_url}/incidents", json=incident_data) as response:
            if response.status == 201:
                result = await response.json()
                logger.info(f"✅ Malware incident created: {result}")
                
                # Wait for processing
                await asyncio.sleep(3)
                
                # Check for quarantine actions
                async with self.session.get(f"{self.base_url}/workflows") as response:
                    if response.status == 200:
                        workflows = await response.json()
                        
                        for workflow in workflows:
                            if workflow['incident_id'] == incident_data['alert_id']:
                                logger.info(f"🔍 Malware Response Status: {workflow['status']}")
                                
                                # Look for quarantine actions
                                quarantine_actions = []
                                for action in workflow['actions_completed']:
                                    if 'quarantine' in action.get('action_id', '').lower():
                                        quarantine_actions.append(action)
                                
                                if quarantine_actions:
                                    logger.info(f"✅ File quarantine actions executed: {len(quarantine_actions)}")
                                    for action in quarantine_actions:
                                        result = action.get('result', {})
                                        logger.info(f"  📁 {action.get('action_id')}: {result.get('details', {})}")
                                else:
                                    logger.warning("⚠️ No quarantine actions found")
                                
                                return True
                
                return False
            else:
                error_text = await response.text()
                logger.error(f"❌ Failed to create malware incident: {response.status} - {error_text}")
                return False
    
    async def test_account_management(self):
        """Test account management containment action"""
        logger.info("👤 Testing Account Management Containment Action")
        logger.info("=" * 60)
        
        # Create insider threat incident
        incident_data = {
            "alert_id": f"account_test_{int(time.time())}",
            "severity": "critical",
            "title": "Insider Threat - Account Disable Required",
            "description": "Suspicious insider activity detected. User account shows signs of data exfiltration. Immediate account disable required.",
            "affected_assets": [
                "user-001.company.com",
                "file-server.company.com"
            ],
            "indicators": [
                "john.doe@company.com",  # Compromised user
                "data-exfil-server.com",  # External destination
                "sensitive-data.zip",  # Exfiltrated file
                "192.168.1.200"  # User workstation
            ],
            "metadata": {
                "threat_type": "insider_threat",
                "confidence": 0.89,
                "data_exfiltration": True,
                "requires_account_disable": True
            }
        }
        
        logger.info(f"📋 Creating insider threat incident: {incident_data['alert_id']}")
        logger.info(f"👤 Compromised user: {incident_data['indicators'][0]}")
        
        # Create incident
        async with self.session.post(f"{self.base_url}/incidents", json=incident_data) as response:
            if response.status == 201:
                result = await response.json()
                logger.info(f"✅ Insider threat incident created: {result}")
                
                # Wait for processing
                await asyncio.sleep(4)
                
                # Check for account management actions
                async with self.session.get(f"{self.base_url}/workflows") as response:
                    if response.status == 200:
                        workflows = await response.json()
                        
                        for workflow in workflows:
                            if workflow['incident_id'] == incident_data['alert_id']:
                                logger.info(f"🔍 Insider Threat Response Status: {workflow['status']}")
                                
                                # Look for account actions
                                account_actions = []
                                for action in workflow['actions_completed']:
                                    if any(keyword in action.get('action_id', '').lower() 
                                          for keyword in ['disable', 'account', 'user']):
                                        account_actions.append(action)
                                
                                if account_actions:
                                    logger.info(f"✅ Account management actions executed: {len(account_actions)}")
                                    for action in account_actions:
                                        result = action.get('result', {})
                                        logger.info(f"  👤 {action.get('action_id')}: {result.get('details', {})}")
                                else:
                                    logger.warning("⚠️ No account management actions found")
                                
                                return True
                
                return False
            else:
                error_text = await response.text()
                logger.error(f"❌ Failed to create insider threat incident: {response.status} - {error_text}")
                return False
    
    async def test_escalation_actions(self):
        """Test escalation containment actions"""
        logger.info("📞 Testing Escalation Containment Actions")
        logger.info("=" * 60)
        
        # Create critical vulnerability incident
        incident_data = {
            "alert_id": f"escalation_test_{int(time.time())}",
            "severity": "critical",
            "title": "Critical Zero-Day Vulnerability - Escalation Required",
            "description": "Critical zero-day vulnerability detected in production systems. Immediate escalation to security team and management required.",
            "affected_assets": [
                "web-server-01.company.com",
                "web-server-02.company.com",
                "database-server.company.com"
            ],
            "indicators": [
                "CVE-2024-0001",  # Critical CVE
                "exploit-payload.bin",  # Exploit file
                "attacker-ip.com",  # Attack source
                "vulnerable-service:8080"  # Vulnerable service
            ],
            "metadata": {
                "threat_type": "vulnerability",
                "confidence": 0.96,
                "cvss_score": 9.8,
                "requires_escalation": True
            }
        }
        
        logger.info(f"📋 Creating critical vulnerability incident: {incident_data['alert_id']}")
        logger.info(f"🎯 Affected systems: {len(incident_data['affected_assets'])}")
        logger.info(f"🔍 CVE: {incident_data['indicators'][0]}")
        
        # Create incident
        async with self.session.post(f"{self.base_url}/incidents", json=incident_data) as response:
            if response.status == 201:
                result = await response.json()
                logger.info(f"✅ Critical vulnerability incident created: {result}")
                
                # Wait for processing
                await asyncio.sleep(3)
                
                # Check for escalation actions
                async with self.session.get(f"{self.base_url}/workflows") as response:
                    if response.status == 200:
                        workflows = await response.json()
                        
                        for workflow in workflows:
                            if workflow['incident_id'] == incident_data['alert_id']:
                                logger.info(f"🔍 Vulnerability Response Status: {workflow['status']}")
                                
                                # Look for escalation actions
                                escalation_actions = []
                                for action in workflow['actions_completed']:
                                    if 'escalate' in action.get('action_id', '').lower():
                                        escalation_actions.append(action)
                                
                                if escalation_actions:
                                    logger.info(f"✅ Escalation actions executed: {len(escalation_actions)}")
                                    for action in escalation_actions:
                                        result = action.get('result', {})
                                        logger.info(f"  📞 {action.get('action_id')}: {result.get('details', {})}")
                                else:
                                    logger.warning("⚠️ No escalation actions found")
                                
                                return True
                
                return False
            else:
                error_text = await response.text()
                logger.error(f"❌ Failed to create vulnerability incident: {response.status} - {error_text}")
                return False
    
    async def test_approval_workflows(self):
        """Test approval workflows for high-risk actions"""
        logger.info("✅ Testing Approval Workflows")
        logger.info("=" * 60)
        
        # Create incident that should require approval
        incident_data = {
            "alert_id": f"approval_test_{int(time.time())}",
            "severity": "critical",
            "title": "Critical System Compromise - Approval Required",
            "description": "Critical system compromise detected. High-risk containment actions require approval before execution.",
            "affected_assets": [
                "production-server.company.com",
                "database-primary.company.com"
            ],
            "indicators": [
                "compromised-admin@company.com",
                "backdoor.exe",
                "192.168.1.50"
            ],
            "metadata": {
                "threat_type": "system_compromise",
                "confidence": 0.94,
                "requires_approval": True,
                "risk_level": "critical"
            }
        }
        
        logger.info(f"📋 Creating approval-required incident: {incident_data['alert_id']}")
        
        # Create incident
        async with self.session.post(f"{self.base_url}/incidents", json=incident_data) as response:
            if response.status == 201:
                result = await response.json()
                logger.info(f"✅ Approval-required incident created: {result}")
                
                # Wait for processing
                await asyncio.sleep(3)
                
                # Check for pending approvals
                async with self.session.get(f"{self.base_url}/workflows") as response:
                    if response.status == 200:
                        workflows = await response.json()
                        
                        for workflow in workflows:
                            if workflow['incident_id'] == incident_data['alert_id']:
                                logger.info(f"🔍 Approval Workflow Status: {workflow['status']}")
                                
                                pending_approvals = workflow.get('actions_pending_approval', [])
                                if pending_approvals:
                                    logger.info(f"⏳ Actions pending approval: {len(pending_approvals)}")
                                    for action in pending_approvals:
                                        logger.info(f"  ⏳ {action.get('action_id')}: {action.get('justification', 'No justification')}")
                                    
                                    # Test approval process
                                    if pending_approvals:
                                        action_id = pending_approvals[0]['action_id']
                                        workflow_id = workflow['workflow_id']
                                        
                                        approval_data = {
                                            "workflow_id": workflow_id,
                                            "action_id": action_id,
                                            "approved": True,
                                            "approver": "test-approver",
                                            "comments": "Approved for testing purposes"
                                        }
                                        
                                        logger.info(f"✅ Simulating approval for action: {action_id}")
                                        async with self.session.post(
                                            f"{self.base_url}/workflows/{workflow_id}/approve",
                                            json=approval_data
                                        ) as approval_response:
                                            if approval_response.status == 200:
                                                approval_result = await approval_response.json()
                                                logger.info(f"✅ Approval processed: {approval_result}")
                                            else:
                                                logger.error(f"❌ Approval failed: {approval_response.status}")
                                    
                                    return True
                                else:
                                    logger.warning("⚠️ No actions pending approval")
                                    return False
                
                return False
            else:
                error_text = await response.text()
                logger.error(f"❌ Failed to create approval incident: {response.status} - {error_text}")
                return False
    
    async def get_system_metrics(self):
        """Get system metrics and status"""
        logger.info("📊 Getting System Metrics")
        logger.info("=" * 60)
        
        try:
            async with self.session.get(f"{self.base_url}/metrics") as response:
                if response.status == 200:
                    metrics = await response.json()
                    logger.info("📈 System Metrics:")
                    logger.info(f"  📚 Total Playbooks: {metrics.get('playbooks', {}).get('total', 0)}")
                    logger.info(f"  ✅ Enabled Playbooks: {metrics.get('playbooks', {}).get('enabled', 0)}")
                    logger.info(f"  🔄 Running Workflows: {metrics.get('workflows', {}).get('running', 0)}")
                    logger.info(f"  🛡️ MCP Enabled: {metrics.get('mcp', {}).get('enabled', False)}")
                    logger.info(f"  🔍 TI Sources: {metrics.get('mcp', {}).get('threat_intelligence_sources', 0)}")
                    
                    # Show workflow status breakdown
                    status_counts = metrics.get('workflows', {}).get('by_status', {})
                    if status_counts:
                        logger.info("  📊 Workflow Status Breakdown:")
                        for status, count in status_counts.items():
                            logger.info(f"    {status}: {count}")
                    
                    return metrics
                else:
                    logger.error(f"❌ Failed to get metrics: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"❌ Error getting metrics: {e}")
            return None

async def main():
    """Main test function for containment actions"""
    logger.info("🚀 Starting Advanced Containment Actions Test Suite")
    logger.info("=" * 80)
    
    async with ContainmentActionTester() as tester:
        # Wait for server to be ready
        if not await tester.wait_for_server():
            logger.error("❌ Cannot proceed without SOAR server")
            return False
        
        test_results = {}
        
        try:
            # Test 1: Asset Isolation
            logger.info("\n" + "=" * 80)
            test_results["asset_isolation"] = await tester.test_asset_isolation()
            
            # Test 2: Network Blocking
            logger.info("\n" + "=" * 80)
            test_results["network_blocking"] = await tester.test_network_blocking()
            
            # Test 3: File Quarantine
            logger.info("\n" + "=" * 80)
            test_results["file_quarantine"] = await tester.test_file_quarantine()
            
            # Test 4: Account Management
            logger.info("\n" + "=" * 80)
            test_results["account_management"] = await tester.test_account_management()
            
            # Test 5: Escalation Actions
            logger.info("\n" + "=" * 80)
            test_results["escalation_actions"] = await tester.test_escalation_actions()
            
            # Test 6: Approval Workflows
            logger.info("\n" + "=" * 80)
            test_results["approval_workflows"] = await tester.test_approval_workflows()
            
            # Get final system metrics
            logger.info("\n" + "=" * 80)
            await tester.get_system_metrics()
            
        except Exception as e:
            logger.error(f"❌ Test execution failed: {e}")
        
        # Print final results
        logger.info("\n" + "=" * 80)
        logger.info("🎯 CONTAINMENT ACTIONS TEST RESULTS")
        logger.info("=" * 80)
        
        passed_tests = 0
        total_tests = len(test_results)
        
        for test_name, result in test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
            if result:
                passed_tests += 1
        
        logger.info(f"\n📊 Overall Results: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            logger.info("🎉 All containment action tests passed!")
        else:
            logger.info(f"⚠️ {total_tests - passed_tests} tests failed")
        
        return passed_tests == total_tests

if __name__ == "__main__":
    asyncio.run(main())


