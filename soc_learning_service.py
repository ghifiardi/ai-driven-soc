import logging
import json
from fastapi import FastAPI, Body, HTTPException
from typing import Dict, Any
from bigquery_client import BigQueryClient
import os

# Set BigQuery Credentials for Baseline Integration
SA_PATH = "Service Account BigQuery/sa-gatra-bigquery.json"
if os.path.exists(SA_PATH):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SA_PATH

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Learning-Service")

app = FastAPI(title="GATRA SOC - Self-Learning Service (RL)")

# Initialize BigQuery Persistence
try:
    bq_client = BigQueryClient(
        project_id="chronicle-dev-2be9",
        dataset_id="soc_data",
        table_id="rl_feedback_metrics"
    )
    logger.info("Learning BigQuery Persistence active")
except Exception as e:
    logger.error(f"Failed to initialize Learning BQ Persistence: {e}")
    bq_client = None

class SOCRLModel:
    """A simple Reinforcement Learning model (Q-Learning style) for tuning SOC thresholds."""
    def __init__(self):
        # Initial 'weights' for detectors
        self.weights = {
            "autoencoder_threshold": 0.1, # ADA
            "graph_anomaly_threshold": 2.0, # ADA
            "threat_score_offset": 0.0 # TAA
        }
        self.learning_rate = 0.05

    def apply_feedback(self, feedback: Dict[str, Any]):
        """
        Adjust weights based on analyst feedback.
        feedback = {'alarm_id': '...', 'is_true_positive': True/False, 'agent': 'ADA'/'TAA'}
        """
        is_tp = feedback.get('is_true_positive', True)
        agent = feedback.get('agent', 'ADA')
        
        # Reward Signal: +1 for True Positive, -1 for False Positive
        reward = 1.0 if is_tp else -1.0
        
        logger.info(f"Applying feedback loop for {agent}. Reward: {reward}")
        
        if not is_tp:
            # If False Positive, increase thresholds to be more selective (less sensitive)
            if agent == 'ADA':
                self.weights['autoencoder_threshold'] += self.learning_rate
                self.weights['graph_anomaly_threshold'] += (self.learning_rate * 5)
            elif agent == 'TAA':
                self.weights['threat_score_offset'] += self.learning_rate
        else:
            # If True Positive, slightly decrease thresholds (be more sensitive)
            if agent == 'ADA':
                self.weights['autoencoder_threshold'] = max(0.01, self.weights['autoencoder_threshold'] - (self.learning_rate * 0.1))
            elif agent == 'TAA':
                self.weights['threat_score_offset'] -= (self.learning_rate * 0.1)

        logger.info(f"New state after learning: {self.weights}")

model = SOCRLModel()

@app.get("/health")
async def health():
    return {"status": "active", "current_weights": model.weights}

@app.post("/api/v1/feedback")
async def submit_feedback(feedback: Dict[str, Any] = Body(...)):
    """API for analysts or CRA outcomes to provide feedback to the RL model."""
    try:
        model.apply_feedback(feedback)
        
        # Persist to BigQuery
        if bq_client:
            try:
                bq_client.insert_rows_json([{
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "classification": feedback.get("agent", "unknown"),
                    "avg_reward_score": 1.0 if feedback.get("is_true_positive") else -1.0,
                    "avg_similarity": model.weights.get("autoencoder_threshold", 0.0), # Drift proxy
                    "total_alerts": 1,
                    "high_reward_alerts": 1 if feedback.get("is_true_positive") else 0
                }])
            except Exception as e:
                logger.error(f"Learning BQ Persistence error: {e}")
                
        return {"status": "learned", "weights": model.weights}
    except Exception as e:
        logger.error(f"Learning error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8084)
