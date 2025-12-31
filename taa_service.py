import logging
import time
import os
from typing import Dict, Any, List

from fastapi import FastAPI, Body, HTTPException
from bigquery_client import BigQueryClient
from enhanced_taa_agent import EnhancedTAAgent

# Set BigQuery Credentials for Baseline Integration
SA_PATH = "Service Account BigQuery/sa-gatra-bigquery.json"
if os.path.exists(SA_PATH):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SA_PATH

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TAA-Service")

app = FastAPI(title="GATRA SOC - Triage & Analysis Service (TAA)")
agent = EnhancedTAAgent()

# Initialize BigQuery Persistence
try:
    bq_client = BigQueryClient(
        project_id="chronicle-dev-2be9",
        dataset_id="gatra_database",
        table_id="taa_state"
    )
    logger.info("TAA BigQuery Persistence active")
except Exception as e:
    logger.error(f"Failed to initialize TAA BQ Persistence: {e}")
    bq_client = None

# Dynamic Intelligence Cache
INTELLIGENCE_CACHE = {
    "malicious_ips": set(["1.1.1.1", "8.8.8.8"]), # Initial seeds
    "last_update": 0
}

async def update_threat_intel():
    """Simulate polling a public threat intel feed."""
    now = time.time()
    if now - INTELLIGENCE_CACHE["last_update"] > 300: # Every 5 mins
        logger.info("Updating dynamic threat intelligence feeds...")
        # In a real app: resp = await requests.get("https://intel-feed/api/v1/bad-ips")
        INTELLIGENCE_CACHE["malicious_ips"].add("192.168.100.200") 
        INTELLIGENCE_CACHE["last_update"] = now

@app.on_event("startup")
async def startup_event():
    await update_threat_intel()

@app.get("/health")
async def health():
    return {"status": "healthy", "stats": agent.get_statistics()}

@app.post("/api/v1/triage")
async def triage_alert(alert_data: Dict[str, Any] = Body(...)):
    """Analyze an alert and produce a classification."""
    try:
        logger.info(f"Received alert for triage: {alert_data.get('alarm_id')}")
        
        # Dynamic Intel Check
        src_ip = alert_data.get("source_ip")
        if src_ip in INTELLIGENCE_CACHE["malicious_ips"]:
            logger.warning(f"THREAT INTEL MATCH: Source IP {src_ip} found in malicious cache!")
            alert_data["alert_severity"] = "critical" # Force escalation
        
        result = await agent.analyze_alert(alert_data)

        # Convert result to dict for serialization
        return {
            "alarm_id": result.alarm_id,
            "classification": result.classification,
            "confidence": result.confidence,
            "threat_score": result.threat_score,
            "is_anomaly": result.is_anomaly,
            "processing_time": result.processing_time
        }
    except Exception as e:
        logger.error(f"Triage error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/triage/batch")
async def triage_batch(alerts: List[Dict[str, Any]] = Body(...)):
    """Analyze a batch of alerts."""
    try:
        results = await agent.analyze_batch(alerts)
        return [ {
            "alarm_id": r.alarm_id,
            "classification": r.classification,
            "confidence": r.confidence,
            "threat_score": r.threat_score,
            "is_anomaly": r.is_anomaly
        } for r in results]
    except Exception as e:
        logger.error(f"Batch triage error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8082)