import requests
import time
import subprocess
import os

def test_platform_gatra_integration():
    print("🚀 Starting MSSP Platform Integration Test...")
    
    # Start server in background
    server_process = subprocess.Popen(
        ["./venv/bin/python3.14", "mssp_platform_server.py"],
        env={**os.environ, "MULTITENANT_CONFIG_PATH": "config/gatra_multitenant_config.json"}
    )
    time.sleep(3) # Wait for startup
    
    try:
        # 1. Get Token (using a known API Key from the config)
        api_key = "soc_secret_key_001" 
        
        print("Auth: Exchanging API Key for JWT...")
        resp = requests.post("http://localhost:8081/api/v1/auth/token", headers={"X-API-Key": api_key})
        if resp.status_code != 200:
            print(f"❌ Auth failed: {resp.text}")
            return
            
        token = resp.json()["access_token"]
        print("✅ Auth Success.")

        # 2. Ingest Events
        print("Ingestion: Sending 5 events to GATRA...")
        payload = {
            "tenant_id": "tenant_001",
            "events": [
                {"id": "evt_1", "duration": 10, "bytes_sent": 5000, "bytes_received": 10000, "port": 443},
                {"id": "evt_2", "duration": 5, "bytes_sent": 100, "bytes_received": 200, "port": 80},
                {"id": "evt_3", "duration": 1, "bytes_sent": 1000000, "bytes_received": 0, "port": 22}, # Outlier?
                {"id": "evt_4", "duration": 30, "bytes_sent": 10, "bytes_received": 10, "port": 53},
                {"id": "evt_5", "duration": 20, "bytes_sent": 200, "bytes_received": 200, "port": 443}
            ]
        }
        
        headers = {"Authorization": f"Bearer {token}"}
        resp = requests.post("http://localhost:8081/api/v1/events", json=payload, headers=headers)
        
        if resp.status_code == 202 or resp.status_code == 200:
            print(f"✅ Ingestion Accepted: {resp.json()}")
        else:
            print(f"❌ Ingestion Failed: {resp.text}")

        # Wait a bit for background processing logs to appear
        time.sleep(2)
        print("\nIntegration test complete. Check server logs for GATRA analysis details.")

    finally:
        server_process.terminate()
        print("🛑 Server stopped.")

if __name__ == "__main__":
    test_platform_gatra_integration()
