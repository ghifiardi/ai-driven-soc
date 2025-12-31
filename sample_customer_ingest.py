import requests
import json
import time

# --- CUSTOMER CONFIGURATION ---
API_KEY = "YOUR_API_KEY"
TENANT_ID = "YOUR_TENANT_ID"
BASE_URL = "https://api.your-soc.com/api/v1"

def get_token():
    """Exchanges API Key for a JWT Access Token."""
    print("🔑 Authenticating with SOC...")
    response = requests.post(
        f"{BASE_URL}/auth/token",
        headers={"X-API-Key": API_KEY}
    )
    response.raise_for_status()
    return response.json()["access_token"]

def send_events(token, events):
    """Sends a batch of events to the SOC."""
    print(f"🚀 Sending {len(events)} events...")
    response = requests.post(
        f"{BASE_URL}/events",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        },
        json={
            "tenant_id": TENANT_ID,
            "events": events
        }
    )
    response.raise_for_status()
    print("✅ Ingestion Successful.")

if __name__ == "__main__":
    try:
        # 1. Authenticate
        jwt_token = get_token()

        # 2. Sample Data
        my_events = [
            {
                "event_id": f"evt-{int(time.time())}",
                "timestamp": "2024-05-20T10:00:00Z",
                "type": "firewall_alert",
                "source_ip": "1.2.3.4",
                "details": {"action": "blocked", "reason": "unauthorized_access"}
            }
        ]

        # 3. Send
        send_events(jwt_token, my_events)

    except Exception as e:
        print(f"❌ Error: {e}")
