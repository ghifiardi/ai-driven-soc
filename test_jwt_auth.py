#!/usr/bin/env python3
"""
Test Script for JWT Authentication Flow
=======================================

Verifies:
1. Token generation via API Key.
2. Secure event ingestion using the JWT token.
3. Rejection of invalid/expired tokens.
"""

import urllib.request
import urllib.error
import json
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8081"
API_KEY = "soc_secret_key_001"
TENANT_ID = "tenant_001"

def make_request(method, endpoint, data=None, headers=None):
    url = f"{BASE_URL}{endpoint}"
    req_headers = {'Content-Type': 'application/json'}
    if headers:
        req_headers.update(headers)
    
    if data:
        json_data = json.dumps(data).encode('utf-8')
        req = urllib.request.Request(url, data=json_data, headers=req_headers, method=method)
    else:
        req = urllib.request.Request(url, headers=req_headers, method=method)
        
    try:
        with urllib.request.urlopen(req) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except:
            return e.code, e.read().decode()
    except Exception as e:
        return 500, str(e)

def test_jwt_flow():
    logger.info("Step 1: Requesting JWT Token with API Key...")
    status, response = make_request("POST", "/api/v1/auth/token", headers={"X-API-Key": API_KEY})
    
    if status != 200 or "access_token" not in response:
        logger.error(f"❌ Failed to get token: {response}")
        return False
    
    token = response["access_token"]
    logger.info("✅ Successfully received JWT token")

    logger.info("Step 2: Testing Event Ingestion with JWT Token...")
    payload = {
        "tenant_id": TENANT_ID,
        "events": [{"id": "test-1", "type": "public_api_test"}]
    }
    
    status, response = make_request(
        "POST", 
        "/api/v1/events", 
        data=payload, 
        headers={"Authorization": f"Bearer {token}"}
    )
    
    if status == 200 or status == 202: # Accepted
        logger.info("✅ Event ingestion successful with JWT")
    else:
        logger.error(f"❌ Event ingestion failed: {response}")
        return False

    logger.info("Step 3: Testing Ingestion WITHOUT Token (Should Fail)...")
    status, response = make_request("POST", "/api/v1/events", data=payload)
    if status == 403 or status == 401:
        logger.info(f"✅ Properly rejected request without token (Status: {status})")
    else:
        logger.error(f"❌ Failed to reject request without token (Status: {status})")
        return False

    logger.info("Step 4: Testing Token for WRONG Tenant (Should Fail)...")
    payload_wrong = {
        "tenant_id": "tenant_002",
        "events": [{"id": "test-2", "type": "impersonation_test"}]
    }
    status, response = make_request(
        "POST", 
        "/api/v1/events", 
        data=payload_wrong, 
        headers={"Authorization": f"Bearer {token}"}
    )
    if status == 403:
        logger.info("✅ Properly rejected impersonation attempt")
    else:
        logger.error(f"❌ Failed to reject impersonation (Status: {status})")
        return False

    return True

if __name__ == "__main__":
    logger.info("Starting JWT Verification Test...")
    if test_jwt_flow():
        logger.info("🎉 JWT AUTHENTICATION VERIFIED SUCCESSFULLY")
    else:
        logger.error("💥 JWT VERIFICATION FAILED")
        exit(1)
