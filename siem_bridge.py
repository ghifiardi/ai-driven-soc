import json
import logging
from datetime import datetime
from typing import Dict, Any, List
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GATRA-SIEM-Bridge")

class SIEMBridge:
    def __init__(self, gatra_api_url: str, api_key: str):
        self.api_url = gatra_api_url
        self.api_key = api_key
        self.token = None

    def _get_token(self):
        """Exchange API Key for JWT."""
        try:
            response = requests.post(
                f"{self.api_url}/api/v1/auth/token",
                headers={"X-API-Key": self.api_key}
            )
            response.raise_for_status()
            self.token = response.json()["access_token"]
            logger.info("Successfully refreshed GATRA token.")
        except Exception as e:
            logger.error(f"Failed to get token: {e}")

    def map_generic_log_to_gatra(self, log: Dict[str, Any]) -> Dict[str, Any]:
        """
        Maps a generic SIEM log (Splunk/Elastic) to GATRA 10D vector schema.
        This is a template and should be customized for specific SIEM exports.
        """
        return {
            "id": log.get("event_id") or log.get("_id"),
            "timestamp": log.get("@timestamp") or log.get("time") or datetime.now().isoformat(),
            "source_ip": log.get("src_ip") or log.get("source.ip"),
            "dest_ip": log.get("dest_ip") or log.get("destination.ip"),
            "port": int(log.get("dest_port") or log.get("destination.port") or 0),
            "protocol": log.get("proto") or log.get("network.transport") or "tcp",
            "bytes_sent": int(log.get("bytes_out") or log.get("source.bytes") or 0),
            "bytes_received": int(log.get("bytes_in") or log.get("destination.bytes") or 0),
            "duration": float(log.get("duration") or 0)
        }

    def push_to_gatra(self, tenant_id: str, raw_logs: List[Dict[str, Any]]):
        """Send mapped logs to the GATRA API."""
        if not self.token:
            self._get_token()

        mapped_events = [self.map_generic_log_to_gatra(log) for log in raw_logs]
        
        payload = {
            "tenant_id": tenant_id,
            "events": mapped_events
        }

        try:
            headers = {"Authorization": f"Bearer {self.token}"}
            response = requests.post(
                f"{self.api_url}/api/v1/events",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 401:
                self._get_token() # Token might be expired
                headers["Authorization"] = f"Bearer {self.token}"
                response = requests.post(f"{self.api_url}/api/v1/events", json=payload, headers=headers)
            
            response.raise_for_status()
            logger.info(f"Successfully pushed {len(mapped_events)} events to GATRA.")
        except Exception as e:
            logger.error(f"Error pushing to GATRA API: {e}")

if __name__ == "__main__":
    # Example usage for a Splunk-style log
    bridge = SIEMBridge(gatra_api_url="http://localhost:8081", api_key="your-api-key-here")
    
    sample_logs = [
        {
            "event_id": "splunk_123",
            "@timestamp": "2025-12-21T16:00:00Z",
            "src_ip": "10.0.0.5",
            "dest_ip": "8.8.8.8",
            "dest_port": 443,
            "bytes_out": 5000,
            "bytes_in": 12000,
            "duration": 1.5
        }
    ]
    
    # bridge.push_to_gatra(tenant_id="tenant-alpha", raw_logs=sample_logs)
