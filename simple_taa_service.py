#!/usr/bin/env python3
"""
Simple TAA Service for Dashboard Status
"""
import time
import json
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleTAAService:
    def __init__(self):
        self.status = "active"
        self.processed_count = 0
        self.start_time = datetime.now()
        
    def process_alert(self, alert_data):
        """Simulate alert processing"""
        self.processed_count += 1
        logger.info(f"Processing alert #{self.processed_count}: {alert_data.get('id', 'unknown')}")
        
        # Simulate processing time
        time.sleep(0.1)
        
        return {
            "status": "processed",
            "confidence": 0.85,
            "severity": "medium",
            "processed_at": datetime.now().isoformat()
        }
    
    def get_status(self):
        """Get service status"""
        uptime = datetime.now() - self.start_time
        return {
            "status": self.status,
            "uptime_seconds": uptime.total_seconds(),
            "processed_count": self.processed_count,
            "last_update": datetime.now().isoformat()
        }

def main():
    """Main service loop"""
    logger.info("Starting Simple TAA Service...")
    
    service = SimpleTAAService()
    
    # Simulate processing alerts
    test_alerts = [
        {"id": "ALERT-001", "type": "suspicious_login"},
        {"id": "ALERT-002", "type": "data_exfiltration"},
        {"id": "ALERT-003", "type": "malware_detection"}
    ]
    
    try:
        while True:
            # Process a test alert every 30 seconds
            for alert in test_alerts:
                result = service.process_alert(alert)
                logger.info(f"Alert {alert['id']} processed: {result['status']}")
                time.sleep(30)
                
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
    except Exception as e:
        logger.error(f"Service error: {e}")

if __name__ == "__main__":
    main()


