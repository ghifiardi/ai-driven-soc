"""
LangGraph-Enhanced Anomaly Detection Agent (ADA) for AI-driven SOC

This enhanced version integrates LangGraph for sophisticated workflow orchestration,
providing better state management, conditional routing, and multi-agent coordination.

Core Enhancements:
- Graph-based workflow orchestration
- Advanced state management with persistent context
- Conditional routing based on detection confidence
- Parallel processing for multiple data sources
- Enhanced coordination with other SOC agents
- Retry mechanisms and fault tolerance
- Real-time feedback integration
"""

import os
import json
import logging
import asyncio
import time
from datetime import datetime
import sys
from typing import Dict, List, Any, Optional
# Literal was added in Python 3.8, use typing_extensions for older versions
if sys.version_info >= (3, 8):
    from typing import Literal, TypedDict
else:
    from typing_extensions import Literal, TypedDict
from typing_extensions import NotRequired
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np

# LangGraph imports
from langgraph.graph import StateGraph, END

# Google Cloud imports
from google.cloud import storage, pubsub_v1, aiplatform, firestore

# ML and data processing
from sklearn.ensemble import IsolationForest
import pickle
import joblib
import redis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ada_workflow.log", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LangGraphADA")

# State definitions for the ADA workflow
class AlertData(TypedDict):
    log_id: str
    timestamp: str
    source_ip: Optional[str]
    dest_ip: Optional[str]
    protocol: Optional[str]
    port: Optional[int]
    bytes_sent: Optional[int]
    bytes_received: Optional[int]
    duration: Optional[float]
    raw_log: Dict[str, Any]

class DetectionResult(TypedDict):
    anomaly_score: float
    confidence: float
    is_anomaly: bool
    model_version: str
    detection_timestamp: str
    feature_importance: Dict[str, float]

class EnrichmentData(TypedDict):
    geo_location: Optional[Dict[str, Any]]
    threat_intelligence: Optional[Dict[str, Any]]
    historical_context: Optional[Dict[str, Any]]
    asset_context: Optional[Dict[str, Any]]

class ADAState(TypedDict):
    alert_data: AlertData
    batch_id: str
    detection_result: Optional[DetectionResult]
    geo_location: NotRequired[Dict[str, Any]]
    threat_intelligence: NotRequired[Dict[str, Any]]
    historical_context: NotRequired[Dict[str, Any]]
    asset_context: NotRequired[Dict[str, Any]]
    enrichment_data: Optional[EnrichmentData]
    workflow_status: Literal["started", "processing", "enriching", "validating", "completed", "failed"]
    error_messages: List[str]
    final_alert: Optional[Dict[str, Any]]
    should_publish: bool

@dataclass
class ADAConfig:
    project_id: str = "chronicle-dev-2be9"
    location: str = "us-central1"
    redis_url: str = ""
    model_bucket: str = "security-models-chronicle-dev-2be9" # Use project-specific bucket
    model_path: str = "anomaly-detection/isolation_forest.pkl"
    supervised_model_path: str = "supervised_model_v1.joblib" # Path to the new supervised model
    confidence_threshold: float = 0.8
    alert_topic: str = "security-alerts"
    features: List[str] = None
    local_mode: bool = False

    def __post_init__(self):
        if self.features is None:
            self.features = [
                "duration", "bytes_sent", "bytes_received", "port",
                "protocol_encoded", "hour_of_day", "day_of_week"
            ]

class LangGraphAnomalyDetectionAgent:
    def __init__(self, config: ADAConfig):
        self.config = config
        self.storage_client = storage.Client(project=config.project_id)
        self.publisher = pubsub_v1.PublisherClient()
        self.db = firestore.Client(project=config.project_id) if not config.local_mode else None
        aiplatform.init(project=config.project_id, location=config.location)
        self.model = self._load_model(self.config.model_path)
        self.supervised_model = self._load_model(self.config.supervised_model_path, model_type='joblib')
        self.workflow = self._build_workflow()
        logger.info("LangGraph ADA initialized successfully")

    async def process_alert(self, alert_data: dict) -> dict:
        """
        Processes a single alert through the LangGraph workflow.
        This is the primary entry point for external clients.
        """
        logger.info(f"Processing alert: {alert_data.get('log_id', 'N/A')}")
        try:
            # The input to the graph is a dictionary with a 'messages' key and direct alert_data
            graph_input = {"messages": [("user", alert_data)], "alert_data": alert_data}
            # Create a unique configuration for each alert to enable parallel processing
            config = {"configurable": {"thread_id": alert_data.get('log_id')}}
            final_state = await self.workflow.ainvoke(graph_input, config=config)
            
            # Check if the final state has the expected structure
            if isinstance(final_state, dict) and "final_alert" in final_state:
                return {"success": True, "result": final_state}
            else:
                logger.error(f"Graph execution for alert {alert_data.get('log_id')} returned an invalid final state: {final_state}")
                return {"error": "Invalid final state from graph"}
        except Exception as e:
            logger.exception(f"An exception occurred while processing alert {alert_data.get('log_id')}: {e}")
            return {"error": str(e)}

    def _load_model(self, model_path, model_type='pickle'):
        """Loads a model from GCS, with appropriate fallbacks."""
        try:
            bucket = self.storage_client.bucket(self.config.model_bucket)
            blob = bucket.blob(model_path)
            
            logger.info(f"Attempting to load {model_type} model from gs://{self.config.model_bucket}/{model_path}")
            model_data = blob.download_as_bytes()

            if model_type == 'pickle':
                model = pickle.loads(model_data)
            elif model_type == 'joblib':
                from io import BytesIO
                model = joblib.load(BytesIO(model_data))
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            logger.info(f"Successfully loaded model from GCS.")
            return model

        except Exception as e:
            logger.warning(f"Could not load model '{model_path}' from GCS: {e}")
            if model_type == 'pickle':
                logger.info("Creating new Isolation Forest model as a fallback.")
                return IsolationForest(contamination='auto', random_state=42)
            else:
                logger.warning("Supervised model could not be loaded. Validation will fall back to default behavior.")
                return None

    def _generate_synthetic_data(self):
        np.random.seed(42)
        return np.random.rand(1000, len(self.config.features))

    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(ADAState)
        workflow.add_node("preprocess_data", self.preprocess_data_node)
        workflow.add_node("extract_features", self.extract_features_node)
        workflow.add_node("detect_anomalies", self.detect_anomalies_node)
        workflow.add_node("geo_enrichment", self.geo_enrichment_node)
        workflow.add_node("threat_intel_enrichment", self.threat_intel_enrichment_node)
        workflow.add_node("historical_enrichment", self.historical_enrichment_node)
        workflow.add_node("asset_enrichment", self.asset_enrichment_node)
        workflow.add_node("aggregate_enrichment", self.aggregate_enrichment_node)
        workflow.add_node("validate_detection", self.validate_detection_node)
        workflow.add_node("generate_alert", self.generate_alert_node)
        workflow.add_node("publish_alert", self.publish_alert_node)
        workflow.add_node("handle_error", self.handle_error_node)

        workflow.set_entry_point("preprocess_data")
        workflow.add_edge("preprocess_data", "extract_features")
        workflow.add_edge("extract_features", "detect_anomalies")
        workflow.add_conditional_edges("detect_anomalies", self.route_after_detection)
        
        for node in ["geo_enrichment", "threat_intel_enrichment", "historical_enrichment", "asset_enrichment"]:
            workflow.add_edge(node, "aggregate_enrichment")

        workflow.add_edge("aggregate_enrichment", "validate_detection")
        workflow.add_conditional_edges("validate_detection", self.route_after_validation, {"generate_alert": "generate_alert", "discard": END, "error": "handle_error"})
        workflow.add_edge("generate_alert", "publish_alert")
        workflow.add_edge("publish_alert", END)
        workflow.add_edge("handle_error", END)
        return workflow.compile()

    def preprocess_data_node(self, state: ADAState) -> Dict[str, Any]:
        logger.info("Preprocessing data")
        try:
            # Alert data should already be in the state from process_alert
            if "alert_data" not in state:
                # As a fallback, try to extract from messages
                messages = state.get("messages", [])
                for role, content in reversed(messages):
                    if role == "user":
                        state["alert_data"] = content
                        return state
                
                # If we get here, we didn't find a user message
                logger.error("No user message found in state")
                state["error_message"] = "No user message found in state"
            
            return state
        except Exception as e:
            logger.error(f"Error extracting alert data: {e}")
            state["error_message"] = f"Error extracting alert data: {e}"
            return state

    def extract_features_node(self, state: ADAState) -> Dict[str, Any]:
        logger.info("Extracting features")
        alert_data = state["alert_data"]
        
        # Handle potentially missing timestamp
        timestamp_str = alert_data.get("timestamp")
        timestamp = pd.to_datetime(timestamp_str) if timestamp_str else pd.Timestamp.now()
        
        # Safely convert values to float/int with proper None handling
        def safe_float(value, default=0.0):
            if value is None:
                return default
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        
        def safe_int(value, default=0):
            if value is None:
                return default
            try:
                return int(value)
            except (ValueError, TypeError):
                return default
        
        features = {
            "duration": safe_float(alert_data.get("duration")),
            "bytes_sent": safe_float(alert_data.get("bytes_sent")),
            "bytes_received": safe_float(alert_data.get("bytes_received")),
            "port": safe_int(alert_data.get("port")),
            "protocol_encoded": {"tcp": 1, "udp": 2, "icmp": 3}.get(
                str(alert_data.get("protocol", "")).lower(), 0
            ),
            "hour_of_day": timestamp.hour,
            "day_of_week": timestamp.dayofweek,
        }
        alert_data["extracted_features"] = features
        return {"alert_data": alert_data}

    def detect_anomalies_node(self, state: ADAState) -> Dict[str, Any]:
        logger.info("Detecting anomalies")
        features = state["alert_data"]["extracted_features"]
        feature_vector = np.array([features.get(f, 0.0) for f in self.config.features]).reshape(1, -1)
        score = self.model.decision_function(feature_vector)[0]
        normalized_score = max(0, min(1, (0.5 - score) * 2))
        is_anomaly = normalized_score > self.config.confidence_threshold
        result = DetectionResult(anomaly_score=normalized_score, confidence=abs(normalized_score - 0.5) * 2, is_anomaly=is_anomaly, model_version="1.0", detection_timestamp=datetime.now().isoformat(), feature_importance={})
        return {"detection_result": result}

    def route_after_detection(self, state: ADAState) -> str:
        if state["detection_result"]["is_anomaly"]:
            return ["geo_enrichment", "threat_intel_enrichment", "historical_enrichment", "asset_enrichment"]
        return "validate_detection"

    def geo_enrichment_node(self, state: ADAState) -> Dict[str, Any]:
        return {"geo_location": {"country": "USA"}}

    def threat_intel_enrichment_node(self, state: ADAState) -> Dict[str, Any]:
        return {"threat_intelligence": {"status": "suspicious"}}

    def historical_enrichment_node(self, state: ADAState) -> Dict[str, Any]:
        return {"historical_context": {"past_events": 5}}

    def asset_enrichment_node(self, state: ADAState) -> Dict[str, Any]:
        return {"asset_context": {"owner": "test-owner"}}

    def aggregate_enrichment_node(self, state: ADAState) -> Dict[str, Any]:
        logger.info("Aggregating enrichment results")
        enrichment_data = EnrichmentData(
            geo_location=state.get("geo_location"),
            threat_intelligence=state.get("threat_intelligence"),
            historical_context=state.get("historical_context"),
            asset_context=state.get("asset_context"),
        )
        logger.info(f"Aggregated data: {json.dumps(enrichment_data, indent=2)}")
        return {"enrichment_data": enrichment_data}

    def validate_detection_node(self, state: ADAState) -> Dict[str, Any]:
        logger.info("--- Stage 2 Validation: Supervised Model Classification ---")
        detection_result = state.get("detection_result")

        if not detection_result or not detection_result.get("is_anomaly"):
            logger.info("Initial scan did not flag as anomaly. Skipping supervised validation.")
            return {"should_publish": False}

        if not self.supervised_model:
            logger.warning("Supervised model not loaded. Falling back to simple threshold check.")
            should_publish = detection_result["confidence"] > self.config.confidence_threshold
            return {"detection_result": detection_result, "should_publish": should_publish}

        logger.info("Initial anomaly detected. Preparing features for supervised model...")

        # Prepare the feature vector for the supervised model
        # This must match the features used during retraining
        try:
            raw_alert = state['alert_data']['raw_log']
            features = {
                'raw_alert_data_anomaly_score': detection_result.get('anomaly_score', 0.0),
                'raw_alert_data_method_name': raw_alert.get('methodName'),
                'raw_alert_data_service_name': raw_alert.get('serviceName'),
            }
            feature_df = pd.DataFrame([features])
            logger.info(f"Feature DataFrame for prediction:\n{feature_df.to_string()}")

            # Predict using the supervised model pipeline
            prediction = self.supervised_model.predict(feature_df)
            final_classification = prediction[0]
            logger.info(f"Supervised model prediction: '{final_classification}'")

            # Update the detection result with the new, more accurate classification
            detection_result['supervised_classification'] = final_classification

            # The final decision is now based on the supervised model's output
            should_publish = final_classification == 'true_positive'

        except Exception as e:
            logger.error(f"Error during supervised model prediction: {e}", exc_info=True)
            # Fallback to default behavior in case of error
            should_publish = False

        logger.info(f"Final validation decision based on supervised model: should_publish={should_publish}")
        return {"detection_result": detection_result, "should_publish": should_publish}

    def route_after_validation(self, state: ADAState) -> str:
        return "generate_alert" if state.get("should_publish") else "discard"

    def generate_alert_node(self, state: ADAState) -> Dict[str, Any]:
        final_alert = {
            "alert_id": f"ada-{state['alert_data']['log_id']}",
            "details": state["alert_data"],
            "detection": state["detection_result"],
            "enrichment": state.get("enrichment_data"),
        }
        return {"final_alert": final_alert}

    def publish_alert_node(self, state: ADAState) -> Dict[str, Any]:
        logger.info(f"Publishing alert: {json.dumps(state['final_alert'], indent=2)}")
        return {}

    def handle_error_node(self, state: ADAState) -> Dict[str, Any]:
        logger.error(f"Workflow failed. Errors: {state.get('error_messages')}")
        return {}

    async def process_single_alert(self, alert_data: AlertData) -> Dict[str, Any]:
        initial_state = ADAState(alert_data=alert_data, batch_id=alert_data['log_id'], error_messages=[])
        return await self.workflow.ainvoke(initial_state)

async def main():
    config = ADAConfig(local_mode=False)
    agent = LangGraphAnomalyDetectionAgent(config)
    sample_alert = {
        "log_id": "log-12345",
        "timestamp": datetime.now().isoformat(),
        "source_ip": "192.168.1.100",
        "dest_ip": "10.0.0.5",
        "protocol": "tcp",
        "port": 443,
        "bytes_sent": 1500,
        "bytes_received": 8000,
        "duration": 60.5,
        "raw_log": {"message": "Suspicious connection"}
    }
    result = await agent.process_single_alert(sample_alert)
    logger.info(f"Final state: {result.get('workflow_status')}")

if __name__ == "__main__":
    if "PROJECT_ID" not in os.environ:
        os.environ["PROJECT_ID"] = "chronicle-dev-2be9"
    asyncio.run(main())
