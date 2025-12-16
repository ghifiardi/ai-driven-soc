#!/usr/bin/env python3
"""
Test Script for MSSP Platform Integration
=========================================

Verifies the functionality of the MSSP Platform Server including:
1. Tenant Registration
2. Event Ingestion
3. MCP Tool Usage
"""

import asyncio
import logging
import urllib.request
import urllib.error
import json
import time
from multiprocessing import Process
import uvicorn
from mssp_platform_server import MSSPPlatformServer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8081"

def run_server():
    """Run the server in a separate process."""
    server = MSSPPlatformServer()
    server.run(port=8081)

def wait_for_server():
    """Wait for the server to be ready."""
    for _ in range(10):
        try:
            with urllib.request.urlopen(BASE_URL) as response:
                if response.status == 200:
                    return True
        except urllib.error.URLError:
            pass
        time.sleep(1)
    return False

def make_request(method, endpoint, data=None):
    url = f"{BASE_URL}{endpoint}"
    headers = {'Content-Type': 'application/json'}
    
    if data:
        json_data = json.dumps(data).encode('utf-8')
        req = urllib.request.Request(url, data=json_data, headers=headers, method=method)
    else:
        req = urllib.request.Request(url, headers=headers, method=method)
        
    try:
        with urllib.request.urlopen(req) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())
    except Exception as e:
        return 500, str(e)

def test_tenant_registration():
    logger.info("Testing Tenant Registration...")
    payload = {
        "tenant_id": "test_mssp_tenant_001",
        "display_name": "Test MSSP Client",
        "region": "us-central1",
        "service_level": "starter"
    }
    
    # Register
    status, response = make_request("POST", "/api/v1/tenants", payload)
    
    if status == 200:
        logger.info("✅ Tenant registered successfully")
    elif status == 409:
        logger.info("⚠️ Tenant already exists (expected if re-running)")
    else:
        logger.error(f"❌ Registration failed: {response}")
        return False

    # Verify listing
    status, response = make_request("GET", "/api/v1/tenants")
    tenants = response.get("tenants", [])
    found = any(t["tenant_id"] == "test_mssp_tenant_001" for t in tenants)
    
    if found:
        logger.info("✅ Tenant found in list")
        return True
    else:
        logger.error("❌ Tenant not found in list")
        return False

def test_event_ingestion():
    logger.info("Testing Event Ingestion...")
    payload = {
        "tenant_id": "test_mssp_tenant_001",
        "events": [
            {"id": 1, "type": "login_success", "user": "alice"},
            {"id": 2, "type": "file_access", "file": "secret.txt"}
        ]
    }
    
    status, response = make_request("POST", "/api/v1/events", payload)
    if status == 200:
        logger.info("✅ Events ingested successfully")
        return True
    else:
        logger.error(f"❌ Ingestion failed: {response}")
        return False

def test_mcp_tool():
    logger.info("Testing MCP Tool (get_tenant_status)...")
    payload = {"tenant_id": "test_mssp_tenant_001"}
    
    status, response = make_request("POST", "/mcp/tools/get_tenant_status", payload)
    if status == 200:
        if response.get("status") == "active":
            logger.info("✅ MCP tool returned active status")
            return True
        else:
            logger.error(f"❌ MCP tool returned unexpected data: {response}")
            return False
    else:
        logger.error(f"❌ MCP tool call failed: {response}")
        return False

if __name__ == "__main__":
    # Start server in background
    server_process = Process(target=run_server)
    server_process.start()
    
    try:
        if not wait_for_server():
            logger.error("Server failed to start")
            exit(1)
            
        logger.info("Server is up. Starting tests...")
        
        success = True
        success &= test_tenant_registration()
        success &= test_event_ingestion()
        success &= test_mcp_tool()
        
        if success:
            logger.info("🎉 ALL TESTS PASSED")
        else:
            logger.error("💥 SOME TESTS FAILED")
            exit(1)
            
    finally:
        server_process.terminate()
        server_process.join()
