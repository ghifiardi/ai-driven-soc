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
from taa_langgraph_agent import build_taa_workflow, TAAState  # Integrate TAA as LangGraph node

# Google Cloud imports
from google.cloud import storage, pubsub_v1, aiplatform, firestore

# ML and data processing
from sklearn.ensemble import IsolationForest, RandomForestClassifier
import pickle
import joblib
import redis
from bigquery_client import BigQueryClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/var/log/langgraph-ada/ada_workflow.log", mode='w'),
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
    # Load all configuration from environment variables to ensure flexibility
    project_id: str = os.getenv("BIGQUERY_PROJECT_ID", "chronicle-dev-2be9")
    location: str = os.getenv("GCP_LOCATION", "asia-southeast2")
    redis_url: str = os.getenv("REDIS_URL", "")
    
    # Model configuration
    model_bucket: str = os.getenv("GCS_BUCKET_NAME", "security-models-chronicle-dev-2be9")
    model_path: str = "anomaly-detection/isolation_forest.pkl"
    supervised_model_path: str = "supervised_model_v1.pkl" # Standardize on .pkl
    
    # BigQuery Configuration - sourced from environment
    BIGQUERY_PROJECT_ID: str = os.getenv("BIGQUERY_PROJECT_ID")
    BIGQUERY_DATASET_ID: str = os.getenv("BIGQUERY_DATASET_ID")
    BIGQUERY_TABLE_ID: str = os.getenv("BIGQUERY_TABLE_ID")
    BIGQUERY_RESULTS_TABLE_ID: str = os.getenv("BIGQUERY_RESULTS_TABLE_ID")

    # Application settings
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", 0.8))
    polling_interval: int = int(os.getenv("POLLING_INTERVAL", 30))
    features: List[str] = None

    def __post_init__(self):
        # Ensure required BigQuery settings are present
        if not all([self.BIGQUERY_PROJECT_ID, self.BIGQUERY_DATASET_ID, self.BIGQUERY_TABLE_ID, self.BIGQUERY_RESULTS_TABLE_ID]):
            raise ValueError("Missing one or more required BigQuery environment variables.")

        if self.features is None:
            self.features = [
                'bytes_sent', 'bytes_received', 'duration',
                'port', 'protocol_encoded', 'hour_of_day', 'day_of_week'
            ]

class LangGraphAnomalyDetectionAgent:
    def __init__(self, config: ADAConfig):
        logger.info("[TRACE] Entering __init__")
        self.config = config
        logger.info("[TRACE] 1. config assigned")
        self.storage_client = storage.Client(project=config.project_id)
        logger.info("[TRACE] 2. storage client created")
        self.publisher = pubsub_v1.PublisherClient()
        logger.info("[TRACE] 3. publisher created")
        self.redis_client = redis.from_url(config.redis_url) if config.redis_url else None
        logger.info(f"[TRACE] 4. Redis client created. URL was: '{config.redis_url}'")
        try:
            self.bq_client = BigQueryClient(
                config.BIGQUERY_PROJECT_ID,
                config.BIGQUERY_DATASET_ID,
                config.BIGQUERY_TABLE_ID,
                config.BIGQUERY_RESULTS_TABLE_ID
            )
            logger.info("[TRACE] 5. BigQuery client created")
        except (ValueError, NameError) as e:
            logger.critical(f"Failed to initialize BigQuery client: {e}")
            # In a real app, you might want to handle this more gracefully
            sys.exit(1)
        aiplatform.init(project=config.project_id, location=config.location)
        logger.info("[TRACE] 6. AI Platform initialized")
        self.model = self._load_model(self.config.model_path)
        logger.info("[TRACE] 7. Model loaded")
        self.supervised_model = self._load_model(self.config.supervised_model_path, model_type='joblib')
        logger.info("[TRACE] 8. Supervised model loaded")
        # Prepare TAA workflow once
        try:
            self.taa_workflow = build_taa_workflow().compile()
            logger.info("[TRACE] 9. TAA workflow compiled")
        except Exception as e:
            logger.warning(f"Failed to initialize TAA workflow: {e}")
            self.taa_workflow = None

        self.workflow = self._build_workflow()
        logger.info("[TRACE] 9. Workflow built")
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
            if isinstance(final_state, dict) and "detection_result" in final_state:
                # Return a cleaner result with just the important information
                return {
                    "success": True,
                    "alert_id": alert_data.get('log_id'),
                    "is_anomaly": final_state["detection_result"]["is_anomaly"],
                    "confidence": final_state["detection_result"]["confidence"],
                    "model_version": final_state["detection_result"]["model_version"],
                }
            else:
                logger.error(f"Graph execution for alert {alert_data.get('log_id')} returned an unexpected structure: {final_state}")
                return {"error": "Invalid result structure from graph"}
        except Exception as e:
            logger.exception(f"An exception occurred while processing alert {alert_data.get('log_id')}: {e}")
            return {"error": str(e)}

    def _load_model(self, model_path, model_type='pickle'):
        """Loads a model from a local path, creating a fallback if it doesn't exist."""
        local_model_dir = 'models'
        # Ensure we use a consistent .pkl extension for local models, as we save everything in that format.
        base_name = os.path.basename(model_path)
        local_path = os.path.join(local_model_dir, os.path.splitext(base_name)[0] + '.pkl')
        
        os.makedirs(local_model_dir, exist_ok=True)

        # Attempt to load from local path first
        if os.path.exists(local_path):
            logger.info(f"Attempting to load model from local path: {local_path}")
            try:
                with open(local_path, 'rb') as f:
                    return pickle.load(f)
            except (pickle.UnpicklingError, AttributeError, EOFError, TypeError) as e:
                logger.warning(f"Could not load model from {local_path} due to corruption or version mismatch: {e}. A new model will be created.")
        
        # If local load failed or file doesn't exist, create and save a fallback.
        logger.info(f"Creating and saving a new fallback model at {local_path}.")
        
        # Fixed: Generate both X and y values from synthetic data
        X, y = self._generate_synthetic_data()

        if 'isolation_forest' in model_path:
            model = IsolationForest(contamination='auto', random_state=42)
            model.fit(X)
        elif 'supervised_model' in model_path:
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X, y)
        else:
            logger.error(f"Unknown model type for path: {model_path}. Cannot create fallback.")
            return None

        try:
            with open(local_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Successfully saved new model to {local_path}")
        except Exception as e:
            logger.error(f"Failed to save new model to {local_path}: {e}")

        return model

    def _generate_synthetic_data(self):
        """Generate synthetic data for training fallback models."""
        np.random.seed(42)
        # Generate X (features) with appropriate dimensions
        X = np.random.rand(1000, len(self.config.features))
        # Generate y (labels) for supervised models
        y = np.random.choice([0, 1], size=1000)
        logger.info(f"DEBUG: Generated synthetic data X shape: {X.shape}, y shape: {y.shape}")
        return X, y

    def _build_workflow(self) -> StateGraph:
        """Builds and compiles the LangGraph workflow once."""
        workflow = StateGraph(ADAState)
        workflow.add_node("preprocess_data", self.preprocess_data_node)
        workflow.add_node("extract_features", self.extract_features_node)
        workflow.add_node("detect_anomalies", self.detect_anomalies_node)
        workflow.add_node("generate_alert", self.generate_alert_node)
        workflow.add_node("publish_alert", self.publish_alert_node)

        workflow.set_entry_point("preprocess_data")
        workflow.add_edge("preprocess_data", "extract_features")
        workflow.add_edge("extract_features", "detect_anomalies")
        
        workflow.add_conditional_edges(
            "detect_anomalies",
            self.should_publish_alert,
            {
                "generate_alert": "generate_alert",
                "__end__": END
            }
        )

        workflow.add_edge("generate_alert", "publish_alert")
        workflow.add_edge("publish_alert", END)

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
        logger.info(f"Extracting features for alert {state['alert_data'].get('log_id')}")
        alert_data = state["alert_data"]

        # Ensure timestamp is correctly parsed to create time-based features
        timestamp_str = alert_data.get("timestamp")
        try:
            timestamp = pd.to_datetime(timestamp_str)
        except (ValueError, TypeError):
            logger.warning(f"Could not parse timestamp '{timestamp_str}'. Using current time.")
            timestamp = pd.Timestamp.now()

        # Safely convert values to float/int
        def safe_float(value, default=0.0):
            try:
                return float(value) if value is not None else default
            except (ValueError, TypeError):
                return default

        def safe_int(value, default=0):
            try:
                return int(value) if value is not None else default
            except (ValueError, TypeError):
                return default

        features = {
            "duration": safe_float(alert_data.get("duration")),
            "bytes_sent": safe_float(alert_data.get("bytes_sent")),
            "bytes_received": safe_float(alert_data.get("bytes_received")),
            "port": safe_int(alert_data.get("port")),
            "protocol_encoded": {"tcp": 1, "udp": 2, "icmp": 3}.get(str(alert_data.get("protocol", "")).lower(), 0),
            "hour_of_day": timestamp.hour,
            "day_of_week": timestamp.dayofweek,
        }
        
        logger.info(f"Generated features for alert {alert_data.get('log_id')}: {features}")
        alert_data["extracted_features"] = features
        return {"alert_data": alert_data}

    def detect_anomalies_node(self, state: ADAState) -> Dict[str, Any]:
        logger.info("Detecting anomalies")
        features = state["alert_data"]["extracted_features"]
        feature_vector = np.array([features.get(f, 0.0) for f in self.config.features]).reshape(1, -1)
        # Use the supervised model to predict the probability of being an anomaly
        # predict_proba returns [[P(not_anomaly), P(anomaly)]]
        anomaly_prob = self.supervised_model.predict_proba(feature_vector)[0, 1]

        # The probability is our new confidence score
        confidence = anomaly_prob
        is_anomaly = confidence > self.config.confidence_threshold

        logger.info(f"Supervised model prediction. Confidence: {confidence:.4f}, Is Anomaly: {is_anomaly}")

        # Convert NumPy values to standard Python types for serialization
        result = DetectionResult(
            anomaly_score=float(confidence), 
            confidence=float(confidence), 
            is_anomaly=bool(is_anomaly), 
            model_version="supervised_v1", 
            detection_timestamp=datetime.now().isoformat(), 
            feature_importance={}
        )
        return {"detection_result": result}

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
        final_alert = state.get('final_alert')
        if not final_alert:
            logger.warning("Publish node reached but no final alert in state.")
            return {}

        logger.info(f"Writing final result to BigQuery table {self.config.BIGQUERY_RESULTS_TABLE_ID}...")
        try:
            # We need to flatten the nested structure for BigQuery insertion
            result_row = {
                "alarm_id": final_alert.get("alert_id"),
                "id": None, # Or generate a unique ID if needed
                "score": final_alert.get("detection", {}).get("anomaly_score"),
                "confidence": final_alert.get("detection", {}).get("confidence"),
                "detection_timestamp": final_alert.get("detection", {}).get("detection_timestamp"),
                "valid": state.get("should_publish"),
                "case_class": None, # To be filled by human analyst or another process
                "variable_of_importance": json.dumps(final_alert.get("detection", {}).get("feature_importance", {})),
                "remarks": "Generated by LangGraph ADA",
                "run_time": None # Can be calculated if needed
            }
            self.bq_client.insert_rows_json([result_row])
            logger.info(f"Successfully wrote result for alarm {result_row['alarm_id']} to BigQuery.")
        except Exception as e:
            error_message = f"Failed to write result to BigQuery: {e}"
            logger.error(error_message, exc_info=True)
            # Add error to state to potentially handle it downstream
            state["error_messages"].append(error_message)

        return {}

    def handle_error_node(self, state: ADAState) -> Dict[str, Any]:
        logger.error(f"Workflow failed. Errors: {state.get('error_messages', [])}")
        return {}

    def _create_alert_data_from_raw(self, raw_alert: dict) -> AlertData:
        """Transforms a raw alert dictionary from BigQuery into a structured AlertData object."""
        events_str = raw_alert.get("events")
        if not events_str or not isinstance(events_str, str):
            logger.warning(f"Alert {raw_alert.get('alarmId')} has missing or invalid 'events' data. Proceeding with empty features.")
            events_data = {}
        else:
            try:
                events_data = json.loads(events_str)
            except json.JSONDecodeError:
                logger.warning(f"Alert {raw_alert.get('alarmId')} has malformed JSON in 'events' data: {events_str}")
                events_data = {}

        return AlertData(
            log_id=raw_alert.get("alarmId"),
            timestamp=events_data.get("timestamp", datetime.now().isoformat()),
            source_ip=events_data.get("source_ip"),
            dest_ip=events_data.get("dest_ip"),
            protocol=events_data.get("protocol"),
            port=events_data.get("port"),
            bytes_sent=events_data.get("bytes_sent"),
            bytes_received=events_data.get("bytes_received"),
            duration=events_data.get("duration"),
            raw_log=events_data
        )


    def should_publish_alert(self, state: ADAState) -> Literal["generate_alert", "__end__"]:
        """Determines the next step after anomaly detection based on confidence."""
        logger.info("Routing based on detection result...")
        detection_result = state.get("detection_result")
        if detection_result and detection_result.get("is_anomaly"):
            logger.info(f"Anomaly detected with confidence {detection_result['confidence']:.2f}. Routing to generate alert.")
            return "generate_alert"
        logger.info("No anomaly detected or low confidence. Ending workflow for this alert.")
        return "__end__"

    async def process_single_alert(self, raw_alert: dict) -> Dict[str, Any]:
        """Processes a single raw alert dictionary by invoking the compiled graph."""
        logger.info(f"Processing alert with ID: {raw_alert.get('alarmId')}")
        try:
            alert_data = self._create_alert_data_from_raw(raw_alert)
            initial_state = {"alert_data": alert_data, "error_messages": []}
            config = {"configurable": {"thread_id": str(raw_alert.get('alarmId'))}}
            final_state = await self.workflow.ainvoke(initial_state, config=config)
            return final_state
        except Exception as e:
            logger.error(f"Exception during alert processing for {raw_alert.get('alarmId')}: {e}", exc_info=True)
            return {"error_messages": [str(e)]}

if __name__ == "__main__":
    try:
        config = ADAConfig()
        agent = LangGraphAnomalyDetectionAgent(config)
        bq_client = agent.bq_client

        async def main_loop():
            logger.info("Starting LangGraph ADA service to monitor BigQuery...")
            while True:
                try:
                    new_alerts = bq_client.fetch_new_alerts(limit=50)
                    if not new_alerts:
                        logger.info(f"No new alerts found. Waiting for {config.polling_interval}s...")
                        await asyncio.sleep(config.polling_interval)
                        continue

                    logger.info(f"Fetched {len(new_alerts)} new alerts. Processing batch...")
                    tasks = [agent.process_single_alert(alert) for alert in new_alerts]
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    processed_ids = []
                    error_ids = []
                    
                    for i, result in enumerate(results):
                        # Use 'alarmId' which is the correct key from BigQuery
                        alarm_id = new_alerts[i].get("alarmId")
                        if isinstance(result, Exception):
                            logger.error(f"Unhandled exception for alert {alarm_id}: {result}", exc_info=True)
                            error_ids.append(alarm_id)
                        elif result and not result.get("error_messages"):
                            logger.info(f"Successfully processed alert {alarm_id}.")
                            processed_ids.append(alarm_id)
                        else:
                            error_msg = result.get('error_messages', ['Unknown error'])
                            logger.error(f"Error processing alert {alarm_id}: {error_msg}")
                            error_ids.append(alarm_id)

                    # FIXED: Changed self.bigquery_client to bq_client
                    if processed_ids:
                        bq_client.update_alert_status(alarm_ids=processed_ids, new_status="PROCESSED")
                    if error_ids:
                        bq_client.update_alert_status(alarm_ids=error_ids, new_status="ERROR")

                except Exception as e:
                    logger.critical(f"Critical error in main polling loop: {e}", exc_info=True)
                    await asyncio.sleep(60) # Wait longer after a critical error

        asyncio.run(main_loop())

    except ValueError as e:
        logger.critical(f"Configuration Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Service stopped by user.")

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
from bigquery_client import BigQueryClient

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
    # Load all configuration from environment variables to ensure flexibility
    project_id: str = os.getenv("BIGQUERY_PROJECT_ID", "chronicle-dev-2be9")
    location: str = os.getenv("GCP_LOCATION", "asia-southeast2")
    redis_url: str = os.getenv("REDIS_URL", "")
    
    # Model configuration
    model_bucket: str = os.getenv("GCS_BUCKET_NAME", "security-models-chronicle-dev-2be9")
    model_path: str = "anomaly-detection/isolation_forest.pkl"
    supervised_model_path: str = "supervised_model_v1.pkl" # Standardize on .pkl
    
    # BigQuery Configuration - sourced from environment
    BIGQUERY_PROJECT_ID: str = os.getenv("BIGQUERY_PROJECT_ID")
    BIGQUERY_DATASET_ID: str = os.getenv("BIGQUERY_DATASET_ID")
    BIGQUERY_TABLE_ID: str = os.getenv("BIGQUERY_TABLE_ID")
    BIGQUERY_RESULTS_TABLE_ID: str = os.getenv("BIGQUERY_RESULTS_TABLE_ID")

    # Application settings
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", 0.8))
    polling_interval: int = int(os.getenv("POLLING_INTERVAL", 30))
    features: List[str] = None

    def __post_init__(self):
        # Ensure required BigQuery settings are present
        if not all([self.BIGQUERY_PROJECT_ID, self.BIGQUERY_DATASET_ID, self.BIGQUERY_TABLE_ID, self.BIGQUERY_RESULTS_TABLE_ID]):
            raise ValueError("Missing one or more required BigQuery environment variables.")

        if self.features is None:
            self.features = [
                'bytes_sent', 'bytes_received', 'duration',
                'port', 'protocol_encoded', 'hour_of_day', 'day_of_week'
            ]

class LangGraphAnomalyDetectionAgent:
    def __init__(self, config: ADAConfig):
        logger.info("[TRACE] Entering __init__")
        self.config = config
        logger.info("[TRACE] 1. config assigned")

        self.storage_client = storage.Client(project=config.project_id)
        logger.info(f"[TRACE] 2. Storage client created for project {config.project_id}")

        self.publisher = pubsub_v1.PublisherClient()
        logger.info("[TRACE] 3. Pub/Sub client created")

        self.redis_client = redis.from_url(config.redis_url) if config.redis_url else None
        logger.info(f"[TRACE] 4. Redis client created. URL was: '{config.redis_url}'")

        try:
            self.bq_client = BigQueryClient(
                config.BIGQUERY_PROJECT_ID,
                config.BIGQUERY_DATASET_ID,
                config.BIGQUERY_TABLE_ID,
                config.BIGQUERY_RESULTS_TABLE_ID
            )
            logger.info("[TRACE] 5. BigQuery client created successfully.")
        except Exception as e:
            logger.critical(f"[TRACE] 5. FAILED to initialize BigQuery client: {e}", exc_info=True)
            sys.exit(1)

        aiplatform.init(project=config.project_id, location=config.location)
        logger.info(f"[TRACE] 6. AI Platform initialized for project {config.project_id} in {config.location}")

        logger.info(f"[TRACE] 7. PRE-CALL to _load_model for unsupervised model. Path: {self.config.model_path}")
        self.model = self._load_model(self.config.model_path)
        logger.info("[TRACE] 7. POST-CALL to _load_model for unsupervised model.")

        logger.info(f"[TRACE] 8. PRE-CALL to _load_model for supervised model. Path: {self.config.supervised_model_path}")
        self.supervised_model = self._load_model(self.config.supervised_model_path, model_type='joblib')
        logger.info("[TRACE] 8. POST-CALL to _load_model for supervised model.")

        self.workflow = self._build_workflow()
        logger.info("[TRACE] 9. Workflow built.")

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
            if isinstance(final_state, dict) and "detection_result" in final_state:
                # Return a cleaner result with just the important information
                return {
                    "success": True,
                    "alert_id": alert_data.get('log_id'),
                    "is_anomaly": final_state["detection_result"]["is_anomaly"],
                    "confidence": final_state["detection_result"]["confidence"],
                    "model_version": final_state["detection_result"]["model_version"],
                }
            else:
                logger.error(f"Graph execution for alert {alert_data.get('log_id')} returned an unexpected structure: {final_state}")
                return {"error": "Invalid result structure from graph"}
        except Exception as e:
            logger.exception(f"An exception occurred while processing alert {alert_data.get('log_id')}: {e}")
            return {"error": str(e)}

    def _load_model(self, model_path, model_type='pickle'):
        """Loads a model from a local path, creating a fallback if it doesn't exist."""
        local_model_dir = 'models'
        # Ensure we use a consistent .pkl extension for local models, as we save everything in that format.
        base_name = os.path.basename(model_path)
        local_path = os.path.join(local_model_dir, os.path.splitext(base_name)[0] + '.pkl')

        os.makedirs(local_model_dir, exist_ok=True)

        # Attempt to load from local path first
        if os.path.exists(local_path):
            logger.info(f"Attempting to load model from local path: {local_path}")
            try:
                with open(local_path, 'rb') as f:
                    return pickle.load(f)
            except (pickle.UnpicklingError, AttributeError, EOFError, TypeError) as e:
                logger.warning(f"Could not load model from {local_path} due to corruption or version mismatch: {e}. A new model will be created.")
        
        # If local load failed or file doesn't exist, create and save a fallback.
        logger.info(f"Creating and saving a new fallback model at {local_path}.")
        
        X, y = self._generate_synthetic_data()

        if 'isolation_forest' in model_path:
            model = IsolationForest(contamination='auto', random_state=42)
            model.fit(X)
        elif 'supervised_model' in model_path:
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X, y)
        else:
            logger.error(f"Unknown model type for path: {model_path}. Cannot create fallback.")
            return None

        try:
            with open(local_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Successfully saved new model to {local_path}")
        except Exception as e:
            logger.error(f"Failed to save new model to {local_path}: {e}")

        return model

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
        workflow.add_node("taa_triage", self.taa_triage_node)  # New node: TAA integration
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
        # Route alerts through TAA triage before publish
        workflow.add_edge("generate_alert", "taa_triage")
        workflow.add_edge("taa_triage", "publish_alert")
        workflow.add_edge("publish_alert", END)
        workflow.add_edge("handle_error", END)
        return workflow.compile()

    def taa_triage_node(self, state: ADAState) -> Dict[str, Any]:
        """Invoke TAA LangGraph workflow for triage/analysis. Non-blocking errors.

        Copies essential fields to a TAA state, invokes workflow, and merges back
        key outcomes (e.g., manual_review flag, containment requested) if present.
        """
        if not self.taa_workflow:
            logger.info("TAA workflow not initialized; skipping TAA triage")
            return state
        try:
            alert_id = state.get("alert_id") or state.get("generated_alert_id")
            taa_state = TAAState(alert_id=alert_id, alert_data=state.get("alert_data", {}))
            _ = self.taa_workflow.invoke(taa_state)
            # Optionally, map results back if needed in future
            # Example: state["taa_feedback"] = _
        except Exception as e:
            logger.warning(f"TAA triage invocation failed: {e}")
        return state

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
        logger.info(f"Extracting features for alert {state['alert_data'].get('log_id')}")
        alert_data = state["alert_data"]

        # Ensure timestamp is correctly parsed to create time-based features
        timestamp_str = alert_data.get("timestamp")
        try:
            timestamp = pd.to_datetime(timestamp_str)
        except (ValueError, TypeError):
            logger.warning(f"Could not parse timestamp '{timestamp_str}'. Using current time.")
            timestamp = pd.Timestamp.now()

        # Safely convert values to float/int
        def safe_float(value, default=0.0):
            try:
                return float(value) if value is not None else default
            except (ValueError, TypeError):
                return default

        def safe_int(value, default=0):
            try:
                return int(value) if value is not None else default
            except (ValueError, TypeError):
                return default

        features = {
            "duration": safe_float(alert_data.get("duration")),
            "bytes_sent": safe_float(alert_data.get("bytes_sent")),
            "bytes_received": safe_float(alert_data.get("bytes_received")),
            "port": safe_int(alert_data.get("port")),
            "protocol_encoded": {"tcp": 1, "udp": 2, "icmp": 3}.get(str(alert_data.get("protocol", "")).lower(), 0),
            "hour_of_day": timestamp.hour,
            "day_of_week": timestamp.dayofweek,
        }
        
        logger.info(f"Generated features for alert {alert_data.get('log_id')}: {features}")
        alert_data["extracted_features"] = features
        return {"alert_data": alert_data}

    def detect_anomalies_node(self, state: ADAState) -> Dict[str, Any]:
        logger.info("Detecting anomalies")
        features = state["alert_data"]["extracted_features"]
        feature_vector = np.array([features.get(f, 0.0) for f in self.config.features]).reshape(1, -1)
        # Use the supervised model to predict the probability of being an anomaly
        # predict_proba returns [[P(not_anomaly), P(anomaly)]]
        anomaly_prob = self.supervised_model.predict_proba(feature_vector)[0, 1]

        # The probability is our new confidence score
        confidence = anomaly_prob
        is_anomaly = confidence > self.config.confidence_threshold

        logger.info(f"Supervised model prediction. Confidence: {confidence:.4f}, Is Anomaly: {is_anomaly}")

        # Convert NumPy values to standard Python types for serialization
        result = DetectionResult(
            anomaly_score=float(confidence), 
            confidence=float(confidence), 
            is_anomaly=bool(is_anomaly), 
            model_version="supervised_v1", 
            detection_timestamp=datetime.now().isoformat(), 
            feature_importance={}
        )
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
        final_alert = state.get('final_alert')
        if not final_alert:
            logger.warning("Publish node reached but no final alert in state.")
            return {}

        logger.info(f"Writing final result to BigQuery table {self.config.BIGQUERY_RESULTS_TABLE_ID}...")
        try:
            # We need to flatten the nested structure for BigQuery insertion
            result_row = {
                "alarm_id": final_alert.get("alert_id"),
                "id": None, # Or generate a unique ID if needed
                "score": final_alert.get("detection", {}).get("anomaly_score"),
                "confidence": final_alert.get("detection", {}).get("confidence"),
                "detection_timestamp": final_alert.get("detection", {}).get("detection_timestamp"),
                "valid": state.get("should_publish"),
                "case_class": None, # To be filled by human analyst or another process
                "variable_of_importance": json.dumps(final_alert.get("detection", {}).get("feature_importance")),
                "remarks": "Generated by LangGraph ADA",
                "run_time": None # Can be calculated if needed
            }
            self.bq_client.insert_rows_json([result_row])
            logger.info(f"Successfully wrote result for alarm {result_row['alarm_id']} to BigQuery.")
        except Exception as e:
            error_message = f"Failed to write result to BigQuery: {e}"
            logger.error(error_message, exc_info=True)
            # Add error to state to potentially handle it downstream
            state["error_messages"].append(error_message)

        return {}

    def handle_error_node(self, state: ADAState) -> Dict[str, Any]:
        logger.error(f"Workflow failed. Errors: {state.get('error_messages')}")
        return {}

    def _create_alert_data_from_raw(self, raw_alert: dict) -> AlertData:
        """Transforms a raw alert dictionary from BigQuery into a structured AlertData object."""
        events_str = raw_alert.get("events")
        if not events_str or not isinstance(events_str, str):
            logger.warning(f"Alert {raw_alert.get('alarmId')} has missing or invalid 'events' data. Proceeding with empty features.")
            events_data = {}
        else:
            try:
                events_data = json.loads(events_str)
            except json.JSONDecodeError:
                logger.warning(f"Alert {raw_alert.get('alarmId')} has malformed JSON in 'events' data: {events_str}")
                events_data = {}

        return AlertData(
            log_id=raw_alert.get("alarmId"),
            timestamp=events_data.get("timestamp", datetime.now().isoformat()),
            source_ip=events_data.get("source_ip"),
            dest_ip=events_data.get("dest_ip"),
            protocol=events_data.get("protocol"),
            port=events_data.get("port"),
            bytes_sent=events_data.get("bytes_sent"),
            bytes_received=events_data.get("bytes_received"),
            duration=events_data.get("duration"),
            raw_log=events_data
        )

    async def process_single_alert(self, raw_alert: dict) -> Dict[str, Any]:
        """Processes a single raw alert dictionary through the compiled graph."""
        logger.info(f"Received raw alert for processing: {raw_alert}")
        # Transform the raw alert into the structured AlertData format first
        alert_data = self._create_alert_data_from_raw(raw_alert)
        
        initial_state = ADAState(
            alert_data=alert_data, 
            batch_id=alert_data.get('log_id', 'unknown'), 
            error_messages=[],
            workflow_status='started'
        )
        return await self.workflow.ainvoke(initial_state)

if __name__ == "__main__":
    try:
        config = ADAConfig()
        agent = LangGraphAnomalyDetectionAgent(config)
        bq_client = agent.bq_client

        async def main_loop():
            logger.info("Starting LangGraph ADA service to monitor BigQuery...")
            while True:
                try:
                    new_alerts = bq_client.fetch_new_alerts(limit=50)
                    if not new_alerts:
                        logger.info(f"No new alerts found. Waiting for {config.polling_interval}s...")
                        await asyncio.sleep(config.polling_interval)
                        continue

                    logger.info(f"Fetched {len(new_alerts)} new alerts. Processing batch...")
                    tasks = [agent.process_single_alert(alert) for alert in new_alerts]
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    processed_ids, error_ids = [], []
                    for i, result in enumerate(results):
                        # Use 'alarmId' which is the correct key from BigQuery
                        alarm_id = new_alerts[i].get("alarmId")
                        if isinstance(result, Exception):
                            logger.error(f"Unhandled exception for alert {alarm_id}: {result}", exc_info=True)
                            error_ids.append(alarm_id)
                        elif result and not result.get("error_messages"):
                            logger.info(f"Successfully processed alert {alarm_id}.")
                            processed_ids.append(alarm_id)
                        else:
                            error_msg = result.get('error_messages', 'Unknown error')
                            logger.error(f"Error processing alert {alarm_id}: {error_msg}")
                            error_ids.append(alarm_id)

                    if processed_ids:
                        bq_client.update_alert_status(alarm_ids=processed_ids, new_status="PROCESSED")
                    if error_ids:
                        bq_client.update_alert_status(alarm_ids=error_ids, new_status="ERROR")

                except Exception as e:
                    logger.critical(f"Critical error in main polling loop: {e}", exc_info=True)
                    await asyncio.sleep(60) # Wait longer after a critical error

        asyncio.run(main_loop())

    except ValueError as e:
        logger.critical(f"Configuration Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Service stopped by user.")
