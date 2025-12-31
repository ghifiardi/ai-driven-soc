import logging
import sys
import os
import json

# Add root directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from podium_agents.monitor_agent import MonitorAgent

logging.basicConfig(level=logging.INFO)

def test_monitor_agent():
    print("Initializing MonitorAgent with GATRA...")
    agent = MonitorAgent()
    
    print("Running monitoring cycle...")
    alerts = agent.run_monitoring_cycle()
    
    print(f"Generated {len(alerts)} alerts.")
    for alert in alerts:
        print(f"Alert: {alert['metric']} - Score: {alert['anomaly_score']:.2f} - Severity: {alert['classification']}")
        # print(f"Reasoning: {alert['reasoning']}")

if __name__ == "__main__":
    test_monitor_agent()
