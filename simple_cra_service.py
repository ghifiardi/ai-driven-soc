#!/usr/bin/env python3
"""
Simple CRA Service for Dashboard Status
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

class SimpleCRAService:
    def __init__(self):
        self.status = "active"
        self.contained_count = 0
        self.start_time = datetime.now()
        
    def contain_threat(self, threat_data):
        """Simulate threat containment"""
        self.contained_count += 1
        logger.info(f"Containing threat #{self.contained_count}: {threat_data.get('id', 'unknown')}")
        
        # Simulate containment time
        time.sleep(0.2)
        
        return {
            "status": "contained",
            "action": "isolated",
            "contained_at": datetime.now().isoformat()
        }
    
    def get_status(self):
        """Get service status"""
        uptime = datetime.now() - self.start_time
        return {
            "status": self.status,
            "uptime_seconds": uptime.total_seconds(),
            "contained_count": self.contained_count,
            "last_update": datetime.now().isoformat()
        }

def main():
    """Main service loop"""
    logger.info("Starting Simple CRA Service...")
    
    service = SimpleCRAService()
    
    # Simulate containing threats
    test_threats = [
        {"id": "THREAT-001", "type": "malware"},
        {"id": "THREAT-002", "type": "intrusion"},
        {"id": "THREAT-003", "type": "data_breach"}
    ]
    
    try:
        while True:
            # Process a test threat every 45 seconds
            for threat in test_threats:
                result = service.contain_threat(threat)
                logger.info(f"Threat {threat['id']} contained: {result['status']}")
                time.sleep(45)
                
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
    except Exception as e:
        logger.error(f"Service error: {e}")

if __name__ == "__main__":
    main()


