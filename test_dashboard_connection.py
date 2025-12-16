#!/usr/bin/env python3
"""
Test Dashboard-Agent Connection
==============================

Quick test script to verify dashboard connectivity with the hybrid agent.
"""

import requests
import json
import time

def test_connection():
    """Test the connection between dashboard and agent"""
    print("🔍 Testing Dashboard-Agent Connection...")
    print("=" * 50)
    
    # Test agent health
    try:
        response = requests.get("http://localhost:8083/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Agent Health: {data['status']}")
            print(f"🆔 Agent ID: {data['agent_id']}")
            print(f"📊 Components: {data['components']}")
        else:
            print(f"❌ Agent Health Error: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Agent Health Failed: {e}")
        return False
    
    # Test agent status
    try:
        response = requests.get("http://localhost:8083/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"\n📈 Agent Status:")
            print(f"   • Alerts Processed: {data['stats']['alerts_processed']}")
            print(f"   • Actions Recommended: {data['stats']['actions_recommended']}")
            print(f"   • Explanations Generated: {data['stats']['explanations_generated']}")
            print(f"   • Errors: {data['stats']['errors']}")
        else:
            print(f"❌ Agent Status Error: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Agent Status Failed: {e}")
        return False
    
    # Test dashboard health
    try:
        response = requests.get("http://localhost:8529/_stcore/health", timeout=5)
        if response.status_code == 200:
            print(f"\n✅ Dashboard Health: {response.text}")
        else:
            print(f"❌ Dashboard Health Error: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Dashboard Health Failed: {e}")
        return False
    
    # Test dashboard accessibility
    try:
        response = requests.get("http://localhost:8529", timeout=5)
        if response.status_code == 200:
            print(f"✅ Dashboard Accessible: {len(response.text)} bytes")
            if "Hybrid Cyber Defense" in response.text:
                print("✅ Dashboard Content: Hybrid Defense detected")
            else:
                print("⚠️ Dashboard Content: Hybrid Defense not found")
        else:
            print(f"❌ Dashboard Access Error: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Dashboard Access Failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED!")
    print("\n📊 Dashboard URLs:")
    print("   • Dashboard: http://localhost:8529")
    print("   • Agent Health: http://localhost:8083/health")
    print("   • Agent Status: http://localhost:8083/status")
    
    return True

if __name__ == "__main__":
    success = test_connection()
    if success:
        print("\n✅ Dashboard and Agent are working correctly!")
        print("🛡️ You can now access the Hybrid Cyber Defense Dashboard!")
    else:
        print("\n❌ Connection test failed. Please check the services.")
        exit(1)


















