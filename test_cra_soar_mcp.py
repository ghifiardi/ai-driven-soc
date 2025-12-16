#!/usr/bin/env python3
"""
Test script for CRA SOAR MCP Integration
========================================

This script demonstrates the enhanced CRA with SOAR and MCP integration
capabilities including threat intelligence enrichment and automated response.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any

import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CRASOARMCPTester:
    """Test client for CRA SOAR MCP integration"""
    
    def __init__(self, base_url: str = "http://localhost:8081"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_health_check(self):
        """Test health check endpoint"""
        logger.info("Testing health check...")
        
        async with self.session.get(f"{self.base_url}/health") as response:
            if response.status == 200:
                data = await response.json()
                logger.info(f"Health check passed: {data}")
                return True
            else:
                logger.error(f"Health check failed: {response.status}")
                return False
    
    async def test_list_playbooks(self):
        """Test listing playbooks"""
        logger.info("Testing playbook listing...")
        
        async with self.session.get(f"{self.base_url}/playbooks") as response:
            if response.status == 200:
                playbooks = await response.json()
                logger.info(f"Found {len(playbooks)} playbooks:")
                for playbook in playbooks:
                    logger.info(f"  - {playbook['name']} ({playbook['playbook_id']})")
                return playbooks
            else:
                logger.error(f"Failed to list playbooks: {response.status}")
                return []
    
    async def test_create_incident(self, incident_data: Dict[str, Any]):
        """Test incident creation"""
        logger.info(f"Testing incident creation: {incident_data['alert_id']}")
        
        async with self.session.post(f"{self.base_url}/incidents", json=incident_data) as response:
            if response.status == 201:
                result = await response.json()
                logger.info(f"Incident created: {result}")
                return result
            else:
                error_text = await response.text()
                logger.error(f"Failed to create incident: {response.status} - {error_text}")
                return None
    
    async def test_execute_playbook(self, playbook_id: str, incident_id: str):
        """Test direct playbook execution"""
        logger.info(f"Testing playbook execution: {playbook_id}")
        
        execution_data = {
            "incident_id": incident_id,
            "playbook_id": playbook_id,
            "parameters": {}
        }
        
        async with self.session.post(
            f"{self.base_url}/playbooks/{playbook_id}/execute",
            json=execution_data
        ) as response:
            if response.status == 200:
                result = await response.json()
                logger.info(f"Playbook execution started: {result}")
                return result
            else:
                error_text = await response.text()
                logger.error(f"Failed to execute playbook: {response.status} - {error_text}")
                return None
    
    async def test_workflow_status(self, workflow_id: str):
        """Test workflow status retrieval"""
        logger.info(f"Testing workflow status: {workflow_id}")
        
        async with self.session.get(f"{self.base_url}/workflows/{workflow_id}") as response:
            if response.status == 200:
                result = await response.json()
                logger.info(f"Workflow status: {result}")
                return result
            else:
                error_text = await response.text()
                logger.error(f"Failed to get workflow status: {response.status} - {error_text}")
                return None
    
    async def test_list_workflows(self):
        """Test listing all workflows"""
        logger.info("Testing workflow listing...")
        
        async with self.session.get(f"{self.base_url}/workflows") as response:
            if response.status == 200:
                workflows = await response.json()
                logger.info(f"Found {len(workflows)} workflows:")
                for workflow in workflows:
                    logger.info(f"  - {workflow['workflow_id']}: {workflow['status']}")
                return workflows
            else:
                logger.error(f"Failed to list workflows: {response.status}")
                return []
    
    async def test_metrics(self):
        """Test metrics endpoint"""
        logger.info("Testing metrics...")
        
        async with self.session.get(f"{self.base_url}/metrics") as response:
            if response.status == 200:
                metrics = await response.json()
                logger.info(f"System metrics: {json.dumps(metrics, indent=2)}")
                return metrics
            else:
                logger.error(f"Failed to get metrics: {response.status}")
                return None

async def test_malware_incident():
    """Test malware incident scenario"""
    logger.info("=== Testing Malware Incident Scenario ===")
    
    async with CRASOARMCPTester() as tester:
        # Test health check
        if not await tester.test_health_check():
            return False
        
        # List available playbooks
        playbooks = await tester.test_list_playbooks()
        if not playbooks:
            return False
        
        # Create malware incident
        malware_incident = {
            "alert_id": "malware_alert_001",
            "severity": "high",
            "title": "Malware Detection on Multiple Hosts",
            "description": "Suspicious executable detected on multiple endpoints with network connections to known malicious IPs",
            "affected_assets": ["host-001", "host-002", "host-003"],
            "indicators": [
                "192.168.1.100",
                "malware.example.com",
                "abc123def456789",
                "10.0.0.50"
            ],
            "metadata": {
                "detection_engine": "ADA",
                "confidence": 0.95,
                "threat_type": "malware"
            }
        }
        
        # Create incident
        incident_result = await tester.test_create_incident(malware_incident)
        if not incident_result:
            return False
        
        # Wait a moment for processing
        await asyncio.sleep(2)
        
        # Check workflows
        workflows = await tester.test_list_workflows()
        
        # Get metrics
        metrics = await tester.test_metrics()
        
        return True

async def test_phishing_incident():
    """Test phishing incident scenario"""
    logger.info("=== Testing Phishing Incident Scenario ===")
    
    async with CRASOARMCPTester() as tester:
        # Create phishing incident
        phishing_incident = {
            "alert_id": "phishing_alert_001",
            "severity": "medium",
            "title": "Phishing Email Campaign Detected",
            "description": "Multiple users received suspicious emails with malicious links",
            "affected_assets": ["user-001", "user-002"],
            "indicators": [
                "phishing.example.com",
                "suspicious.link.com",
                "malicious-attachment.pdf"
            ],
            "metadata": {
                "detection_engine": "Email_Security",
                "confidence": 0.85,
                "threat_type": "phishing"
            }
        }
        
        # Create incident
        incident_result = await tester.test_create_incident(phishing_incident)
        if not incident_result:
            return False
        
        # Wait for processing
        await asyncio.sleep(2)
        
        # Check workflows
        workflows = await tester.test_list_workflows()
        
        return True

async def test_direct_playbook_execution():
    """Test direct playbook execution"""
    logger.info("=== Testing Direct Playbook Execution ===")
    
    async with CRASOARMCPTester() as tester:
        # List playbooks
        playbooks = await tester.test_list_playbooks()
        if not playbooks:
            return False
        
        # Find a suitable playbook
        malware_playbook = next(
            (p for p in playbooks if "malware" in p["name"].lower()),
            playbooks[0] if playbooks else None
        )
        
        if not malware_playbook:
            logger.error("No suitable playbook found")
            return False
        
        # Execute playbook directly
        execution_result = await tester.test_execute_playbook(
            malware_playbook["playbook_id"],
            "direct_execution_001"
        )
        
        if execution_result:
            # Wait for execution
            await asyncio.sleep(3)
            
            # Check workflow status
            workflows = await tester.test_list_workflows()
            
            return True
        
        return False

async def test_threat_intelligence_integration():
    """Test threat intelligence integration"""
    logger.info("=== Testing Threat Intelligence Integration ===")
    
    # This would test the MCP threat intelligence integration
    # by creating incidents with various indicators and checking
    # if they get enriched with threat intelligence
    
    async with CRASOARMCPTester() as tester:
        # Create incident with various indicators
        ti_incident = {
            "alert_id": "ti_test_001",
            "severity": "high",
            "title": "Threat Intelligence Test",
            "description": "Testing MCP threat intelligence integration",
            "affected_assets": ["test-host"],
            "indicators": [
                "8.8.8.8",  # Google DNS (should be clean)
                "1.1.1.1",  # Cloudflare DNS (should be clean)
                "malware.example.com",  # Should be flagged
                "abc123def456"  # Test hash
            ],
            "metadata": {
                "test_type": "threat_intelligence",
                "expected_sources": ["threatfox", "malwarebazaar"]
            }
        }
        
        # Create incident
        incident_result = await tester.test_create_incident(ti_incident)
        
        if incident_result:
            # Wait for processing and TI enrichment
            await asyncio.sleep(5)
            
            # Check if threat intelligence was enriched
            # This would require checking the incident in Firestore
            # or through an API endpoint
            
            logger.info("Threat intelligence integration test completed")
            return True
        
        return False

async def main():
    """Main test function"""
    logger.info("Starting CRA SOAR MCP Integration Tests")
    
    test_results = {}
    
    try:
        # Test malware incident scenario
        test_results["malware_incident"] = await test_malware_incident()
        
        # Test phishing incident scenario
        test_results["phishing_incident"] = await test_phishing_incident()
        
        # Test direct playbook execution
        test_results["direct_playbook"] = await test_direct_playbook_execution()
        
        # Test threat intelligence integration
        test_results["threat_intelligence"] = await test_threat_intelligence_integration()
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
    
    # Print test results
    logger.info("=== Test Results ===")
    for test_name, result in test_results.items():
        status = "PASSED" if result else "FAILED"
        logger.info(f"{test_name}: {status}")
    
    # Overall result
    all_passed = all(test_results.values())
    logger.info(f"Overall Result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    asyncio.run(main())


