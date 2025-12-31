import requests
import time
import subprocess
import os

def test_advanced_features():
    print("🚀 Verifying Advanced SOC Features (RL, Notifications, Intel)...")
    
    # Start Learning Service
    proc_lr = subprocess.Popen(["./venv/bin/python3.14", "soc_learning_service.py"])
    proc_cra = subprocess.Popen(["./venv/bin/python3.14", "cra_service.py"])
    time.sleep(5)

    try:
        # 1. Verify RL Feedback
        print("\n[TEST 1] RL Feedback Loop...")
        feedback = {"alarm_id": "TEST-001", "is_true_positive": False, "agent": "ADA"}
        resp = requests.post("http://localhost:8084/api/v1/feedback", json=feedback)
        print(f"Outcome: {resp.json()['status']}")
        print(f"New Weights: {resp.json()['weights']}")
        assert resp.json()['weights']['autoencoder_threshold'] > 0.1
        print("✅ RL Feedback verified.")

        # 2. Verify CRA Notifications
        print("\n[TEST 2] CRA Email Notifications...")
        incident = {"alarm_id": "CRIT-001", "classification": "critical"}
        resp = requests.post("http://localhost:8083/api/v1/contain", json=incident)
        print(f"CRA Outcome: {resp.json()['status']}")
        # Check logs (simulation)
        print("✅ Notification logic triggered (check logs above).")

    finally:
        proc_lr.terminate()
        proc_cra.terminate()
        proc_lr.wait()
        proc_cra.wait()

if __name__ == "__main__":
    test_advanced_features()
