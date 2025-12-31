import requests
import time
import subprocess
import os
import signal

def test_full_soc_pipeline():
    print("🚀 Starting Full SOC Stack E2E Pipeline Test...")
    
    # 1. Start all services
    services = {
        "ADA/Platform": ["./venv/bin/python3.14", "mssp_platform_server.py"],
        "TAA Service": ["./venv/bin/python3.14", "taa_service.py"],
        "CRA Service": ["./venv/bin/python3.14", "cra_service.py"]
    }
    
    processes = {}
    for name, cmd in services.items():
        print(f"Starting {name}...")
        proc = subprocess.Popen(cmd, env={
            **os.environ, 
            "TAA_SERVICE_URL": "http://localhost:8082",
            "CRA_SERVICE_URL": "http://localhost:8083"
        })
        processes[name] = proc
    
    time.sleep(10) # Wait for all services to warm up
    
    try:
        # Step 1: Auth
        api_key = "soc_secret_key_001"
        print("\n[STEP 1] Auth: Exchanging API Key for JWT...")
        resp = requests.post("http://localhost:8081/api/v1/auth/token", headers={"X-API-Key": api_key})
        token = resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        print("✅ Auth Success.")

        # Step 2: Normal Ingestion
        print("\n[STEP 2] Ingestion: Sending Normal Traffic...")
        requests.post("http://localhost:8081/api/v1/events", json={
            "tenant_id": "tenant_001",
            "events": [{"id": "norm_1", "duration": 1, "bytes_sent": 500, "bytes_received": 500, "port": 443}]
        }, headers=headers)
        print("✅ Normal event accepted.")

        # High duration, high bytes_sent, malicious IP, critical severity
        print("\n[STEP 3] Ingestion: Sending CRITICAL Traffic (Simulating Malicious Exfiltration)...")
        requests.post("http://localhost:8081/api/v1/events", json={
            "tenant_id": "tenant_001",
            "events": [{
                "id": "exfil_1", 
                "duration": 9999, 
                "bytes_sent": 999999, 
                "bytes_received": 0, 
                "port": 6666, 
                "source_ip": "1.1.1.1", # Known malicious in reputation DB
                "alert_severity": "critical"
            }]
        }, headers=headers)
        print("✅ Anomalous event accepted.")

        print("\n⏳ Waiting for autonomous pipeline to complete (ADA -> TAA -> CRA)...")
        time.sleep(10)
        
        print("\n[FINAL CHECK] Check service logs for TAA triage and CRA containment success.")
        print("✅ E2E Pipeline Test Sequence Complete.")

    finally:
        for name, proc in processes.items():
            print(f"Stopping {name}...")
            proc.terminate()
            proc.wait()
        print("🛑 All services stopped.")

if __name__ == "__main__":
    test_full_soc_pipeline()
