#!/usr/bin/env python3
"""
TAA LangGraph Microservice
==========================
This service listens to 'ada-alerts' from the ADA agent and executes 
a semantic triage workflow using LangGraph and LLM reasoning.
"""

import os
import json
import logging
import time
from datetime import datetime
from google.cloud import pubsub_v1
from taa_langgraph_agent import build_taa_workflow, TAAState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("taa_langgraph_service.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TAA-LangGraph-Service")

class TAALangGraphService:
    def __init__(self, project_id: str, subscription_id: str):
        self.project_id = project_id
        self.subscription_id = subscription_id
        self.subscriber = pubsub_v1.SubscriberClient()
        self.subscription_path = self.subscriber.subscription_path(project_id, subscription_id)
        
        # Build and compile the LangGraph workflow
        logger.info("Building TAA LangGraph workflow...")
        self.workflow = build_taa_workflow().compile()
        logger.info("TAA LangGraph workflow compiled successfully.")

    def _callback(self, message):
        """Callback for processing received alerts from ADA"""
        try:
            logger.info(f"Received message from ADA: {message.message_id}")
            data = json.loads(message.data.decode("utf-8"))
            
            # Prepare initial state for the TAA LangGraph
            initial_state = TAAState(
                alert_id=data.get("alarm_id"),
                alert_data=data,
                timestamp=datetime.now().isoformat()
            )
            
            logger.info(f"Starting LangGraph triage for Alert ID: {data.get('alarm_id')}")
            
            # Execute the workflow
            final_state = self.workflow.invoke(initial_state)
            
            # Log results
            result = final_state.get('llm_result', {})
            logger.info(f"Triage Complete: Alert={data.get('alarm_id')}, "
                        f"TP={result.get('is_true_positive')}, "
                        f"Severity={result.get('severity')}")
            
            # Acknowledge the message
            message.ack()
            
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            # Nack to allow retry
            message.nack()

    def start(self):
        """Start listening for alerts"""
        logger.info(f"Listening for ADA alerts on {self.subscription_path}...")
        streaming_pull_future = self.subscriber.subscribe(
            self.subscription_path, 
            callback=self._callback
        )
        
        try:
            # Keep the main thread alive
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            streaming_pull_future.cancel()
            logger.info("TAA Service stopped by user.")
        except Exception as e:
            streaming_pull_future.cancel()
            logger.error(f"TAA Service crashed: {e}")

if __name__ == "__main__":
    PROJECT_ID = os.getenv("GCP_PROJECT_ID", "chronicle-dev-2be9")
    SUBSCRIPTION_ID = os.getenv("TAA_SUBSCRIPTION_ID", "ada-alerts-sub")
    
    service = TAALangGraphService(PROJECT_ID, SUBSCRIPTION_ID)
    service.start()
