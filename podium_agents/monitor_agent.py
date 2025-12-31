import random
import logging
import sys
import os

# Add root directory to path to allow importing GATRA
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from advanced_anomaly_detection import GATRAAnomalyDetectionSystem, SeverityLevel
import numpy as np

logger = logging.getLogger(__name__)

class MonitorAgent:
    def __init__(self):
        self.engine = GATRAAnomalyDetectionSystem()
        self.metrics_to_watch = [
            "message_volume",
            "response_time",
            "sentiment_score",
            "customer_retention_rate",
            "payment_dispute_rate"
        ]
        self.anomaly_thresholds = {
            "critical": 0.8,  # GATRA scores are 0.0 - 1.0
            "high": 0.6,
            "medium": 0.4
        }
    
    def run_monitoring_cycle(self):
        """Siklus pemantauan utama"""
        alerts = []
        for metric in self.metrics_to_watch:
            current_value = self.get_current_metric(metric)
            
            # Construct a "virtual" alert for the engine to analyze
            # We map metric values to "alert_severity" or other features the engine understands
            alert_data = self._create_virtual_alert(metric, current_value)
            
            # Use GATRA for calculations
            gatra_result = self.calculate_anomaly_score(alert_data)
            
            if gatra_result.score > self.anomaly_thresholds["medium"]:
                alert = self.create_alert(metric, gatra_result, current_value)
                self.escalate_to_investigator(alert)
                alerts.append(alert)
        return alerts

    def get_current_metric(self, metric_name):
        """Mock Sigma API call - returns random realistic values"""
        return random.uniform(10, 1000)
    
    def _create_virtual_alert(self, metric, value):
        """Maps a metric update to an alert structure for the engine"""
        # Determine severity based on simple thresholds for demonstration
        severity = "info"
        if metric == "payment_dispute_rate" and value > 50:
            severity = "critical"
        elif value > 800:
            severity = "high"
            
        return {
            "alarm_id": f"mon_{random.randint(1000,9999)}",
            "alert_severity": severity,
            "attack_category": "Normal", # Default
            "confidence": 0.9,
            "source_ip": "192.168.1.10", # Internal
            "metric_name": metric,
            "metric_value": value,
            "timestamp": "now"
        }

    def calculate_anomaly_score(self, alert_data):
        """Menghitung skor anomali menggunakan GATRA engine"""
        # Prepare feature vector for GATRA
        features = np.zeros(10)
        features[0] = alert_data.get('metric_value', 0)
        # Mock other features
        alert_data['features'] = features
        return self.engine.process_telemetry(alert_data)

    def create_alert(self, metric, gatra_result, value):
        return {
            "metric": metric,
            "anomaly_score": gatra_result.score,
            "classification": gatra_result.severity.name,
            "reasoning": gatra_result.reasoning,
            "current_value": value,
            "timestamp": "now"
        }

    def escalate_to_investigator(self, alert):
        logger.info(f"Escalating alert for {alert['metric']} (Score: {alert['anomaly_score']:.2f}) - {alert['classification']}")
    
    def continuous_monitoring(self):
        """Used by Coordinator"""
        alerts = self.run_monitoring_cycle()
        return {"critical": [a for a in alerts if a['anomaly_score'] > self.anomaly_thresholds["high"]]}
    
    def get_current_state(self):
        return {"status": "monitoring (local engine)", "active_alerts": 0}
