#!/usr/bin/env python3
"""
SOC Admin: New Customer Onboarding
==================================

Use this script to:
1. Register a new customer (Tenant ID) in the system.
2. Generate their secure API Key.
3. Automatically provision their GCP infrastructure.
"""

import requests
import subprocess
import sys
import json

SERVER_URL = "http://localhost:8081"

def onboard_customer(tenant_id, display_name):
    print(f"🚀 Onboarding New Customer: {display_name} ({tenant_id})")
    
    # 1. Register in the Platform Server
    print("Step 1: Registering tenant in the platform server...")
    payload = {
        "tenant_id": tenant_id,
        "display_name": display_name,
        "service_level": "professional"
    }
    
    try:
        response = requests.post(f"{SERVER_URL}/api/v1/tenants", json=payload)
        response.raise_for_status()
        result = response.json()
        api_key = result["api_key"]
        print(f"✅ Tenant registered successfully.")
        print(f"🔑 GENERATED API KEY: {api_key}")
    except Exception as e:
        print(f"❌ Failed to register tenant: {e}")
        return

    # 2. Provision Infrastructure
    print("\nStep 2: Provisioning GCP Infrastructure (BigQuery/PubSub)...")
    try:
        # Run the existing infra script for this specific tenant
        subprocess.run([
            "python3", "rebuild_soc_infra.py", 
            "--tenant_id", tenant_id
        ], check=True)
        print(f"✅ Infrastructure provisioned for {tenant_id}.")
    except Exception as e:
        print(f"❌ Failed to provision infrastructure: {e}")
        print("💡 Note: Ensure your local server is running and you have gcloud permissions.")

    print("\n--- ONBOARDING COMPLETE ---")
    print(f"Give these details to your customer:")
    print(f"  Tenant ID: {tenant_id}")
    print(f"  API Key:   {api_key}")
    print("----------------------------")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 onboard_customer.py [TENANT_ID] [DISPLAY_NAME]")
        print("Example: python3 onboard_customer.py customer_mega_corp 'Mega Corp Inc'")
        sys.exit(1)
        
    onboard_customer(sys.argv[1], sys.argv[2])
