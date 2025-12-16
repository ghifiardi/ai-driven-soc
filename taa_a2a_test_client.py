#!/usr/bin/env python3
"""
TAA A2A Test Client
==================

Test client to demonstrate A2A communication with the TAA agent.
This shows how other agents can discover and communicate with the TAA agent.
"""

import json
import logging
import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class A2ATask:
    """A2A task structure"""
    task_id: str
    skill: str
    parameters: Dict[str, Any]
    callback_url: Optional[str] = None
    priority: str = "normal"
    timestamp: str = None

class TAAA2ATestClient:
    """Test client for TAA A2A communication"""
    
    def __init__(self, taa_agent_url: str = "http://localhost:8080"):
        self.taa_agent_url = taa_agent_url
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def discover_agent(self) -> Dict[str, Any]:
        """Discover TAA agent capabilities"""
        try:
            async with self.session.get(f"{self.taa_agent_url}/.well-known/agent.json") as response:
                if response.status == 200:
                    agent_card = await response.json()
                    logger.info(f"Discovered TAA agent: {agent_card['name']} v{agent_card['version']}")
                    return agent_card
                else:
                    raise Exception(f"Failed to discover agent: {response.status}")
        except Exception as e:
            logger.error(f"Error discovering agent: {e}")
            raise
    
    async def get_health(self) -> Dict[str, Any]:
        """Check TAA agent health"""
        try:
            async with self.session.get(f"{self.taa_agent_url}/health") as response:
                if response.status == 200:
                    health = await response.json()
                    logger.info(f"TAA agent health: {health['status']}")
                    return health
                else:
                    raise Exception(f"Health check failed: {response.status}")
        except Exception as e:
            logger.error(f"Error checking health: {e}")
            raise
    
    async def get_capabilities(self) -> Dict[str, Any]:
        """Get TAA agent capabilities"""
        try:
            async with self.session.get(f"{self.taa_agent_url}/a2a/capabilities") as response:
                if response.status == 200:
                    capabilities = await response.json()
                    logger.info(f"TAA agent capabilities: {len(capabilities['capabilities'])} skills")
                    return capabilities
                else:
                    raise Exception(f"Failed to get capabilities: {response.status}")
        except Exception as e:
            logger.error(f"Error getting capabilities: {e}")
            raise
    
    async def send_task(self, task: A2ATask) -> Dict[str, Any]:
        """Send A2A task to TAA agent"""
        try:
            headers = {"Content-Type": "application/json"}
            payload = asdict(task)
            
            async with self.session.post(
                f"{self.taa_agent_url}/a2a/tasks",
                json=payload,
                headers=headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Task completed: {task.task_id} - {result['status']}")
                    return result
                else:
                    error_text = await response.text()
                    raise Exception(f"Task failed: {response.status} - {error_text}")
        except Exception as e:
            logger.error(f"Error sending task: {e}")
            raise
    
    async def test_alert_analysis(self) -> Dict[str, Any]:
        """Test security alert analysis capability"""
        try:
            # Create test alert data
            test_alert = {
                "alert_id": "TEST-ALERT-001",
                "timestamp": datetime.now().isoformat(),
                "source": "test_client",
                "severity": "medium",
                "log_data": {
                    "event_type": "process_creation",
                    "process_name": "powershell.exe",
                    "command_line": "powershell.exe -EncodedCommand test",
                    "user": "test\\user",
                    "hostname": "TEST-HOST-001",
                    "src_ip": "192.168.1.100",
                    "dst_ip": "8.8.8.8"
                }
            }
            
            task = A2ATask(
                task_id=f"test-analysis-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                skill="analyze_security_alert",
                parameters={
                    "alert_data": test_alert,
                    "enrichment_level": "full"
                },
                timestamp=datetime.now().isoformat()
            )
            
            logger.info("Testing security alert analysis...")
            return await self.send_task(task)
            
        except Exception as e:
            logger.error(f"Error testing alert analysis: {e}")
            raise
    
    async def test_threat_intelligence_sharing(self) -> Dict[str, Any]:
        """Test threat intelligence sharing capability"""
        try:
            task = A2ATask(
                task_id=f"test-ti-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                skill="share_threat_intelligence",
                parameters={
                    "indicators": ["malicious_ip_1", "suspicious_domain_1", "malware_hash_1"],
                    "confidence": 0.85,
                    "source": "test_client"
                },
                timestamp=datetime.now().isoformat()
            )
            
            logger.info("Testing threat intelligence sharing...")
            return await self.send_task(task)
            
        except Exception as e:
            logger.error(f"Error testing threat intelligence sharing: {e}")
            raise
    
    async def test_containment_request(self) -> Dict[str, Any]:
        """Test containment request capability"""
        try:
            task = A2ATask(
                task_id=f"test-containment-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                skill="request_containment",
                parameters={
                    "target_host": "192.168.1.100",
                    "action_type": "isolate",
                    "priority": "high"
                },
                timestamp=datetime.now().isoformat()
            )
            
            logger.info("Testing containment request...")
            return await self.send_task(task)
            
        except Exception as e:
            logger.error(f"Error testing containment request: {e}")
            raise
    
    async def test_collaborative_analysis(self) -> Dict[str, Any]:
        """Test collaborative analysis capability"""
        try:
            task = A2ATask(
                task_id=f"test-collab-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                skill="collaborative_analysis",
                parameters={
                    "analysis_type": "malware_analysis",
                    "data": {
                        "file_hash": "test_hash_123",
                        "file_type": "executable",
                        "source": "email_attachment"
                    },
                    "collaboration_mode": "parallel"
                },
                timestamp=datetime.now().isoformat()
            )
            
            logger.info("Testing collaborative analysis...")
            return await self.send_task(task)
            
        except Exception as e:
            logger.error(f"Error testing collaborative analysis: {e}")
            raise
    
    async def run_comprehensive_test(self):
        """Run comprehensive A2A communication test"""
        try:
            logger.info("=== TAA A2A Communication Test ===")
            
            # Step 1: Discover agent
            logger.info("\n1. Discovering TAA agent...")
            agent_card = await self.discover_agent()
            print(f"   Agent: {agent_card['name']} v{agent_card['version']}")
            print(f"   Description: {agent_card['description']}")
            print(f"   Capabilities: {len(agent_card['capabilities'])} skills")
            
            # Step 2: Check health
            logger.info("\n2. Checking agent health...")
            health = await self.get_health()
            print(f"   Status: {health['status']}")
            print(f"   Version: {health['version']}")
            
            # Step 3: Get capabilities
            logger.info("\n3. Getting agent capabilities...")
            capabilities = await self.get_capabilities()
            for cap in capabilities['capabilities']:
                print(f"   - {cap['name']}: {cap['description']}")
            
            # Step 4: Test security alert analysis
            logger.info("\n4. Testing security alert analysis...")
            analysis_result = await self.test_alert_analysis()
            print(f"   Status: {analysis_result['status']}")
            if analysis_result['status'] == 'completed':
                analysis = analysis_result['result']['analysis']
                print(f"   True Positive: {analysis['is_true_positive']}")
                print(f"   Severity: {analysis['severity']}")
                print(f"   Confidence: {analysis['confidence']}")
            
            # Step 5: Test threat intelligence sharing
            logger.info("\n5. Testing threat intelligence sharing...")
            ti_result = await self.test_threat_intelligence_sharing()
            print(f"   Status: {ti_result['status']}")
            if ti_result['status'] == 'completed':
                ti_data = ti_result['result']
                print(f"   Shared Indicators: {ti_data['shared_indicators']}")
                print(f"   Message: {ti_data['message']}")
            
            # Step 6: Test containment request
            logger.info("\n6. Testing containment request...")
            containment_result = await self.test_containment_request()
            print(f"   Status: {containment_result['status']}")
            if containment_result['status'] == 'completed':
                cont_data = containment_result['result']
                print(f"   Target Host: {cont_data['target_host']}")
                print(f"   Action Type: {cont_data['action_type']}")
                print(f"   Priority: {cont_data['priority']}")
            
            # Step 7: Test collaborative analysis
            logger.info("\n7. Testing collaborative analysis...")
            collab_result = await self.test_collaborative_analysis()
            print(f"   Status: {collab_result['status']}")
            if collab_result['status'] == 'completed':
                collab_data = collab_result['result']
                print(f"   Collaboration Mode: {collab_data['collaboration_mode']}")
                print(f"   Participating Agents: {collab_data['participating_agents']}")
                print(f"   Message: {collab_data['message']}")
            
            logger.info("\n=== A2A Communication Test Completed Successfully ===")
            
        except Exception as e:
            logger.error(f"Comprehensive test failed: {e}")
            raise

async def main():
    """Main function to run the A2A test client"""
    try:
        # Create test client
        async with TAAA2ATestClient() as client:
            # Run comprehensive test
            await client.run_comprehensive_test()
            
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    # Run the test client
    asyncio.run(main()) 