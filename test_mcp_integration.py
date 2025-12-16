#!/usr/bin/env python3
"""
Test Client for TAA A2A MCP Integration
=======================================

This test client demonstrates the MCP integration capabilities with Google Security Operations.
It tests various MCP endpoints and validates the integration functionality.
"""

import asyncio
import json
import logging
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MCPIntegrationTester:
    """Test client for MCP integration"""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_health_check(self) -> bool:
        """Test server health check"""
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Health check passed: {result}")
                    return True
                else:
                    logger.error(f"Health check failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return False
    
    async def test_agent_card(self) -> bool:
        """Test agent card endpoint"""
        try:
            async with self.session.get(f"{self.base_url}/.well-known/agent.json") as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Agent card retrieved: {result['name']} v{result['version']}")
                    logger.info(f"MCP integration enabled: {result.get('mcp_integration', {}).get('enabled', False)}")
                    return True
                else:
                    logger.error(f"Agent card failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Agent card error: {e}")
            return False
    
    async def test_chronicle_query(self) -> bool:
        """Test Chronicle query via MCP"""
        try:
            query_request = {
                "query_text": "metadata.event_type='NETWORK_CONNECTION' AND network.destination.ip_address='8.8.8.8'",
                "time_range": {
                    "start_time": (datetime.now() - timedelta(hours=24)).isoformat(),
                    "end_time": datetime.now().isoformat()
                },
                "max_results": 100
            }
            
            async with self.session.post(
                f"{self.base_url}/mcp/chronicle/query",
                json=query_request
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Chronicle query successful: {result['status']}")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Chronicle query failed: {response.status} - {error_text}")
                    return False
        except Exception as e:
            logger.error(f"Chronicle query error: {e}")
            return False
    
    async def test_threat_intelligence_query(self) -> bool:
        """Test threat intelligence query via MCP"""
        try:
            ti_request = {
                "indicators": ["8.8.8.8", "malware.example.com", "abc123def456"],
                "intelligence_type": "ioc"
            }
            
            async with self.session.post(
                f"{self.base_url}/mcp/threat-intelligence/query",
                json=ti_request
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Threat intelligence query successful: {result['status']}")
                    logger.info(f"Queried {result['indicators_queried']} indicators")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Threat intelligence query failed: {response.status} - {error_text}")
                    return False
        except Exception as e:
            logger.error(f"Threat intelligence query error: {e}")
            return False
    
    async def test_investigation(self) -> bool:
        """Test incident investigation via MCP"""
        try:
            investigation_request = {
                "incident_id": "test-incident-123",
                "investigation_scope": "deep"
            }
            
            async with self.session.post(
                f"{self.base_url}/mcp/investigate",
                json=investigation_request
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Investigation successful: {result['status']}")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Investigation failed: {response.status} - {error_text}")
                    return False
        except Exception as e:
            logger.error(f"Investigation error: {e}")
            return False
    
    async def test_enhanced_alert_analysis(self) -> bool:
        """Test enhanced alert analysis with MCP"""
        try:
            alert_data = {
                "alert_id": "test-alert-456",
                "log_data": {
                    "src_ip": "192.168.1.100",
                    "dst_ip": "8.8.8.8",
                    "domain": "suspicious.example.com",
                    "event_type": "network_connection",
                    "timestamp": datetime.now().isoformat()
                },
                "enrichment_level": "chronicle"
            }
            
            async with self.session.post(
                f"{self.base_url}/analyze/enhanced",
                json=alert_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Enhanced analysis successful: {result['status']}")
                    logger.info(f"Enrichment level: {result['enrichment_level']}")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Enhanced analysis failed: {response.status} - {error_text}")
                    return False
        except Exception as e:
            logger.error(f"Enhanced analysis error: {e}")
            return False
    
    async def test_a2a_task(self) -> bool:
        """Test A2A task processing"""
        try:
            a2a_task = {
                "task_id": "test-task-789",
                "skill": "analyze_security_alert",
                "parameters": {
                    "alert_data": {
                        "alert_id": "test-alert-789",
                        "log_data": {
                            "src_ip": "10.0.0.1",
                            "dst_ip": "1.1.1.1",
                            "event_type": "file_download"
                        }
                    },
                    "enrichment_level": "chronicle"
                },
                "priority": "high"
            }
            
            async with self.session.post(
                f"{self.base_url}/a2a/tasks",
                json=a2a_task
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"A2A task successful: {result['status']}")
                    logger.info(f"Agent ID: {result['agent_id']}")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"A2A task failed: {response.status} - {error_text}")
                    return False
        except Exception as e:
            logger.error(f"A2A task error: {e}")
            return False
    
    async def test_mcp_servers_status(self) -> bool:
        """Test MCP servers status"""
        try:
            async with self.session.get(f"{self.base_url}/mcp/servers") as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"MCP servers status retrieved")
                    for server_name, server_info in result.get('mcp_servers', {}).items():
                        logger.info(f"  {server_name}: {server_info['status']}")
                    return True
                else:
                    logger.error(f"MCP servers status failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"MCP servers status error: {e}")
            return False
    
    async def test_share_threat_intelligence(self) -> bool:
        """Test threat intelligence sharing"""
        try:
            ti_data = {
                "indicators": ["malicious.ip.example", "suspicious-domain.com"],
                "confidence": 0.95,
                "source": "test_agent",
                "share_with_chronicle": True
            }
            
            async with self.session.post(
                f"{self.base_url}/share/intelligence",
                json=ti_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Threat intelligence sharing successful: {result['status']}")
                    logger.info(f"Shared {result['indicators_shared']} indicators")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Threat intelligence sharing failed: {response.status} - {error_text}")
                    return False
        except Exception as e:
            logger.error(f"Threat intelligence sharing error: {e}")
            return False
    
    async def run_all_tests(self) -> Dict[str, bool]:
        """Run all integration tests"""
        logger.info("Starting MCP Integration Tests")
        logger.info("=" * 50)
        
        tests = {
            "health_check": self.test_health_check,
            "agent_card": self.test_agent_card,
            "chronicle_query": self.test_chronicle_query,
            "threat_intelligence_query": self.test_threat_intelligence_query,
            "investigation": self.test_investigation,
            "enhanced_alert_analysis": self.test_enhanced_alert_analysis,
            "a2a_task": self.test_a2a_task,
            "mcp_servers_status": self.test_mcp_servers_status,
            "share_threat_intelligence": self.test_share_threat_intelligence
        }
        
        results = {}
        for test_name, test_func in tests.items():
            logger.info(f"Running test: {test_name}")
            try:
                result = await test_func()
                results[test_name] = result
                status = "PASS" if result else "FAIL"
                logger.info(f"Test {test_name}: {status}")
            except Exception as e:
                logger.error(f"Test {test_name} failed with exception: {e}")
                results[test_name] = False
            
            logger.info("-" * 30)
        
        # Summary
        passed = sum(results.values())
        total = len(results)
        logger.info("=" * 50)
        logger.info(f"Test Summary: {passed}/{total} tests passed")
        logger.info("=" * 50)
        
        return results

async def main():
    """Main test function"""
    base_url = "http://localhost:8080"
    
    async with MCPIntegrationTester(base_url) as tester:
        results = await tester.run_all_tests()
        
        # Exit with error code if any tests failed
        if not all(results.values()):
            logger.error("Some tests failed. Check the logs above for details.")
            exit(1)
        else:
            logger.info("All tests passed successfully!")

if __name__ == "__main__":
    asyncio.run(main())


