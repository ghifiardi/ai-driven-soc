#!/usr/bin/env python3
"""
Fix script for the CLA deployment on the VM.
This script:
1. Creates the config directory and config file
2. Patches the continuous-learning-agent.py to remove Firestore references
3. Updates the systemd service file
"""

import os
import re
import json
import subprocess
import sys

# Configuration
PROJECT_ID = "chronicle-dev-2be9"
APP_DIR = "/home/raditio.ghifiardigmail.com/ai-driven-soc"
CLA_FILE = os.path.join(APP_DIR, "continuous-learning-agent.py")
CONFIG_DIR = os.path.join(APP_DIR, "config")
CONFIG_FILE = os.path.join(CONFIG_DIR, "cla_config.json")
SERVICE_FILE = "/etc/systemd/system/cla.service"

# Config content
config = {
    "project_id": PROJECT_ID,
    "bigquery_dataset": "soc_data",
    "bigquery_alerts_table": "alerts",
    "bigquery_feedback_table": "feedback",
    "bigquery_metrics_table": "metrics", 
    "bigquery_models_table": "models",
    "bigquery_evaluations_table": "evaluations",
    "bigquery_incidents_table": "incidents",
    "bigquery_patterns_table": "patterns",
    "bigquery_improvements_table": "improvements",
    "bigquery_agent_state_table": "agent_state",
    "feedback_topic": "soc-feedback",
    "feedback_subscription": "cla-feedback-sub",
    "metrics_topic": "soc-metrics",
    "metrics_subscription": "cla-metrics-sub",
    "langgraph_feedback_topic": "langgraph-feedback",
    "feedback_threshold": 10,
    "retraining_interval_days": 7,
    "model_improvement_threshold": 0.05,
    "metrics_collection_days": 30,
    "pattern_analysis_days": 30
}

# Service file content
service_content = """[Unit]
Description=Continuous Learning Agent for AI-Driven SOC
After=network.target

[Service]
Type=simple
User=raditio.ghifiardigmail.com
Group=raditio.ghifiardigmail.com
WorkingDirectory=/home/raditio.ghifiardigmail.com/ai-driven-soc
ExecStart=/home/raditio.ghifiardigmail.com/ai-driven-soc/venv/bin/python /home/raditio.ghifiardigmail.com/ai-driven-soc/continuous-learning-agent.py --config /home/raditio.ghifiardigmail.com/ai-driven-soc/config/cla_config.json
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""

def create_config_file():
    """Create the config directory and file"""
    print(f"Creating config directory: {CONFIG_DIR}")
    os.makedirs(CONFIG_DIR, exist_ok=True)
    
    print(f"Writing config file: {CONFIG_FILE}")
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

def patch_cla_code():
    """Patch the CLA file to fix Firestore references"""
    print(f"Creating backup of CLA file")
    backup_file = f"{CLA_FILE}.bak"
    if not os.path.exists(backup_file):
        os.system(f"cp {CLA_FILE} {backup_file}")
    
    print(f"Patching CLA file to remove Firestore references")
    with open(CLA_FILE, 'r') as file:
        content = file.read()
    
    # Replace the process_feedback method to use BigQuery instead of Firestore
    process_feedback_pattern = r'def process_feedback\(self, feedback_data\):.*?(?=\n    def|$)'
    process_feedback_replacement = '''def process_feedback(self, feedback_data):
        """Process analyst feedback on an alert.
        
        Args:
            feedback_data (dict): Feedback data containing alert_id, is_true_positive, 
                                 analyst_comments, and confidence.
        
        Returns:
            bool: True if feedback was processed successfully.
        """
        try:
            alert_id = feedback_data.get("alert_id")
            is_true_positive = feedback_data.get("is_true_positive", False)
            analyst_comments = feedback_data.get("analyst_comments", "")
            confidence = feedback_data.get("confidence", 1.0)
            
            # Log the received feedback
            logger.info(f"Processing feedback for alert {alert_id}: TP={is_true_positive}, confidence={confidence}")
            
            # Store feedback in BigQuery
            feedback_doc = {
                "alert_id": alert_id,
                "is_true_positive": is_true_positive,
                "analyst_comments": analyst_comments,
                "confidence": confidence,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "processed": False
            }
            
            # Insert into BigQuery using the BigQueryClient
            table_id = f"{self.project_id}.{self.bigquery_dataset}.{self.bigquery_feedback_table}"
            self.bq_client.insert_rows_json(table_id, [feedback_doc])
            
            # If it's a true positive, provide positive feedback to LangGraph ADA
            if is_true_positive:
                self.provide_positive_feedback(feedback_data)
            
            # Update alert status in BigQuery
            alert_update = {
                "feedback_received": True,
                "is_true_positive": is_true_positive,
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            
            # Update alert in BigQuery
            alerts_table_id = f"{self.project_id}.{self.bigquery_dataset}.{self.bigquery_alerts_table}"
            query = f"""
            UPDATE `{alerts_table_id}`
            SET feedback_received = TRUE,
                is_true_positive = {str(is_true_positive).upper()},
                last_updated = CURRENT_TIMESTAMP()
            WHERE alert_id = '{alert_id}'
            """
            
            query_job = self.bq_client.query(query)
            query_job.result()
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing feedback: {e}")
            return False'''
    
    content = re.sub(process_feedback_pattern, process_feedback_replacement, content, flags=re.DOTALL)
    
    # Update main function to properly handle the config path
    main_pattern = r'if __name__ == "__main__":(.*?)(?=\n# End of file|\Z)'
    main_replacement = '''if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Continuous Learning Agent')
    parser.add_argument('--config', type=str, help='Path to config file')
    args = parser.parse_args()
    
    config_path = args.config if args.config else "config/cla_config.json"
    
    # Check if config exists
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found at {config_path}, using environment variables or defaults")
    
    # Initialize and start the agent
    agent = ContinuousLearningAgent(config_path=config_path)
    
    # For testing
    test_feedback = {
        "alert_id": "ADA-DEMO-001", 
        "is_true_positive": True, 
        "analyst_comments": "This is a valid detection", 
        "confidence": 0.95
    }
    
    # Start the feedback and metrics listeners
    agent.start_listeners()
    
    # Keep the agent running
    try:
        while True:
            agent.run_scheduled_tasks()
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Shutting down Continuous Learning Agent")'''
    
    content = re.sub(main_pattern, main_replacement, content, flags=re.DOTALL)
    
    # Make sure timezone is imported
    if 'from datetime import timezone' not in content:
        content = content.replace('from datetime import datetime', 'from datetime import datetime, timezone')
    
    # Write the patched content back to the file
    with open(CLA_FILE, 'w') as file:
        file.write(content)
    
    print("Successfully patched CLA code")

def update_service_file():
    """Update the systemd service file"""
    print(f"Writing service file to {SERVICE_FILE}")
    
    # Need to use sudo to write to /etc/systemd
    with open('/tmp/cla.service', 'w') as f:
        f.write(service_content)
    
    os.system(f"sudo mv /tmp/cla.service {SERVICE_FILE}")
    print("Service file updated")

def restart_service():
    """Reload and restart the service"""
    print("Reloading systemd daemon")
    os.system("sudo systemctl daemon-reload")
    
    print("Restarting CLA service")
    os.system("sudo systemctl restart cla.service")
    
    print("Service restarted. To check status:")
    print("sudo systemctl status cla.service")
    print("To view logs:")
    print("sudo journalctl -u cla.service -f")

def main():
    """Main function to run all fixes"""
    print("Starting CLA fixes...")
    
    # First stop the service
    print("Stopping CLA service")
    os.system("sudo systemctl stop cla.service")
    
    # Create config file
    create_config_file()
    
    # Patch the CLA code
    patch_cla_code()
    
    # Update service file
    update_service_file()
    
    # Restart service
    restart_service()
    
    print("\nAll fixes applied. Please monitor the service logs for any remaining issues.")

if __name__ == "__main__":
    main()
