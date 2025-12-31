from .monitor_agent import MonitorAgent
from .investigator_agent import InvestigatorAgent
from .forecaster_agent import ForecasterAgent
import logging

logger = logging.getLogger(__name__)

class CoordinatorAgent:
    def __init__(self):
        self.monitor = MonitorAgent()
        self.investigator = InvestigatorAgent()
        self.forecaster = ForecasterAgent()
        self.workflow_state = {}
        
    def execute_daily_workflow(self):
        """Workflow operasional harian"""
        # Langkah 1: Pemantauan berkelanjutan
        anomalies = self.monitor.continuous_monitoring()
        
        # Langkah 2: Investigasi anomali kritis
        for anomaly in anomalies.get("critical", []):
            investigation = self.investigator.investigate_anomaly(anomaly)
            
            if investigation.get("requires_immediate_action"):
                self.trigger_automated_response(investigation)
                self.notify_operations_team(investigation)
        
        # Langkah 3: Update forecasting harian
        if self.is_forecast_update_time():
            latest_data = self.fetch_latest_cleaned_data()
            forecasts = self.forecaster.generate_forecasts(latest_data)
            self.update_sigma_dashboard(forecasts)
        
        # Langkah 4: Generate laporan performa
        performance_report = self.generate_performance_report()
        self.distribute_report(performance_report)
    
    def trigger_automated_response(self, investigation):
        """Memicu respons otomatis berdasarkan investigasi"""
        response_actions = {
            "high_fraud_risk": self.block_suspicious_transactions,
            "operational_bottleneck": self.reallocate_resources,
            "sentiment_crisis": self.trigger_customer_retention_protocol
        }
        
        for risk_type, action in response_actions.items():
            if risk_type in investigation.get("identified_risks", []):
                action(investigation)

    # --- MOCKED METHODS ---
    
    def is_forecast_update_time(self):
        return True # Always run for demo

    def fetch_latest_cleaned_data(self):
        return [1, 2, 3, 4, 5]

    def update_sigma_dashboard(self, forecasts):
        logger.info("Updated Sigma Dashboard with new forecasts")

    def generate_performance_report(self):
        return {"uptime": 99.9, "alerts_handled": 10}

    def distribute_report(self, report):
        logger.info(f"Distributed report: {report}")

    def block_suspicious_transactions(self, investigation):
        logger.info("Blocking suspicious transaction...")

    def reallocate_resources(self, investigation):
        logger.info("Reallocating resources...")

    def trigger_customer_retention_protocol(self, investigation):
        logger.info("Triggering customer retention protocol...")
        
    def notify_operations_team(self, investigation):
        logger.info("Notifying operations team...")
