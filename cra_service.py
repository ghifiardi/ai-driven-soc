import logging
import os
import json
import smtplib
from email.mime.text import MIMEText
from fastapi import FastAPI, HTTPException, Body
from typing import Dict, Any, List
from containment_response_agent import ContainmentResponseAgent
from bigquery_client import BigQueryClient
from datetime import datetime

# Set BigQuery Credentials for Baseline Integration
SA_PATH = "Service Account BigQuery/sa-gatra-bigquery.json"
if os.path.exists(SA_PATH):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SA_PATH

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CRA-Service")

# Use a mockable version of CRA for affordable/local deployment
class LiteCRA(ContainmentResponseAgent):
    """A version of CRA that doesn't require GCP/Firestore for local work."""
    def __init__(self, config_path=None):
        self.config = {
            "project_id": "soc-lite",
            "approval_required": {"high": True, "medium": False, "low": False},
            "playbooks": {
                "default": [{"action": "mock_block", "params": {}, "risk_level": "medium"}]
            }
        }
        self.actions_log = []
        logger.info("Lite CRA initialized (Mock mode)")

    def _execute_playbook(self, playbook_tuple, incident, incident_ref=None):
        name, actions = playbook_tuple
        logger.info(f"Executing playbook: {name}")
        results = []
        for action in actions:
            logger.info(f"Action: {action['action']} executed for {incident.get('alarm_id')}")
            results.append({"action": action['action'], "status": "completed"})
        return {"containment_status": "complete", "actions_executed": results}

# Initialize Lite CRA
cra_agent = LiteCRA()

# Initialize BigQuery Persistence
try:
    bq_client = BigQueryClient(
        project_id="chronicle-dev-2be9",
        dataset_id="gatra_database",
        table_id="taa_enhanced_results"
    )
    logger.info("CRA BigQuery Persistence active")
except Exception as e:
    logger.error(f"Failed to initialize CRA BQ Persistence: {e}")
    bq_client = None

def send_notification(subject: str, message: str):
    """SMTP Notification Handler."""
    recipient = os.getenv("SOC_NOTIFICATION_EMAIL", "analyst@example.com")
    logger.info(f"--- EMAIL NOTIFICATION ---")
    logger.info(f"To: {recipient}")
    logger.info(f"Subject: {subject}")
    logger.info(f"Body: {message}")
    logger.info(f"--------------------------")
    
    # Real SMTP Logic (Placeholder)
    # try:
    #     msg = MIMEText(message)
    #     msg['Subject'] = subject
    #     msg['From'] = "soc-noreply@gatra.ai"
    #     msg['To'] = recipient
    #     with smtplib.SMTP('localhost') as s:
    #         s.send_message(msg)
    # except Exception as e:
    #     logger.warning(f"Failed to send real email: {e}")

app = FastAPI(title="GATRA SOC - Containment & Response Service (CRA)")

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/api/v1/contain")
async def contain_incident(incident: Dict[str, Any] = Body(...)):
    """Execute containment actions based on a triaged threat."""
    try:
        logger.info(f"Received incident for containment: {incident.get('alarm_id')}")
        
        # Select playbook (mock logic)
        playbook = ("default", [{"action": "isolate_host", "risk_level": "medium"}])
        
        # Execute
        result = cra_agent._execute_playbook(playbook, incident)
        
        if result.get("containment_status") == "complete":
            send_notification(
                f"INCIDENT CONTAINED: {incident.get('alarm_id')}",
                f"Incident {incident.get('alarm_id')} has been autonomously contained.\n"
                f"Classification: {incident.get('classification')}\n"
                f"Actions Taken: {result.get('actions_executed')}"
            )
        
        # Persist Result to BigQuery (Baseline for Dashboard)
        if bq_client:
            try:
                bq_client.insert_rows_json([{
                    "alarm_id": incident.get("alarm_id"),
                    "cra_action_type": result.get("actions_executed", [{}])[0].get("action", "unknown"),
                    "cra_success": result.get("containment_status") == "complete",
                    "cra_reasoning": result.get("reasoning", "Autonomous containment executed"),
                    "event_time": datetime.now().isoformat()
                }])
            except Exception as e:
                logger.error(f"CRA BQ Persistence error: {e}")
        
        return {
            "status": "active",
            "incident_id": incident.get("alarm_id"),
            "actions": result.get("actions_executed"),
            "containment_status": result.get("containment_status")
        }
    except Exception as e:
        logger.error(f"Containment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8083)