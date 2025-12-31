"""
LangGraph-Enhanced Anomaly Detection Agent (ADA) for AI-driven SOC
Integrated with GATRA Anomaly Detection System.
"""

import os
import json
import logging
import asyncio
import time
from datetime import datetime
import sys
from typing import Dict, List, Any, Optional, Union
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

# GATRA imports
from advanced_anomaly_detection import GATRAAnomalyDetectionSystem, AnomalyResult, SeverityLevel

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
    extracted_features: NotRequired[Dict[str, Any]]

class DetectionResult(TypedDict):
    anomaly_score: float
    confidence: float
    is_anomaly: bool
    model_version: str
    detection_timestamp: str
    feature_importance: Dict[str, float]
    reasoning: Optional[List[str]]
    severity: Optional[str]

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
    project_id: str = os.getenv("BIGQUERY_PROJECT_ID", "chronicle-dev-2be9")
    location: str = os.getenv("GCP_LOCATION", "asia-southeast2")
    redis_url: str = os.getenv("REDIS_URL", "")
    
    # Model configuration
    model_path: str = "anomaly-detection/isolation_forest.pkl"
    supervised_model_path: str = "supervised_model_v1.pkl"
    
    # BigQuery Configuration
    BIGQUERY_PROJECT_ID: str = os.getenv("BIGQUERY_PROJECT_ID")
    BIGQUERY_DATASET_ID: str = os.getenv("BIGQUERY_DATASET_ID")
    BIGQUERY_TABLE_ID: str = os.getenv("BIGQUERY_TABLE_ID")
    BIGQUERY_RESULTS_TABLE_ID: str = os.getenv("BIGQUERY_RESULTS_TABLE_ID")

    # Application settings
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", 0.8))
    polling_interval: int = int(os.getenv("POLLING_INTERVAL", 30))
    features: List[str] = None

    def __post_init__(self):
        if self.features is None:
            self.features = [
                'bytes_sent', 'bytes_received', 'duration',
                'port', 'protocol_encoded', 'hour_of_day', 'day_of_week'
            ]

class LangGraphAnomalyDetectionAgent:
    def __init__(self, config: ADAConfig):
        self.config = config
        self.storage_client = storage.Client(project=config.project_id) if config.project_id else None
        self.publisher = pubsub_v1.PublisherClient()
        self.redis_client = redis.from_url(config.redis_url) if config.redis_url else None
        try:
            self.bq_client = BigQueryClient(
                config.BIGQUERY_PROJECT_ID,
                config.BIGQUERY_DATASET_ID,
                config.BIGQUERY_TABLE_ID,
                config.BIGQUERY_RESULTS_TABLE_ID
            )
        except Exception as e:
            logger.warning(f"Failed to initialize BigQuery client: {e}")
            self.bq_client = None

        aiplatform.init(project=config.project_id, location=config.location)
        
        # Initialize GATRA instead of legacy models
        self.gatra = GATRAAnomalyDetectionSystem()
        # Fallback for supervised stage if needed
        self.supervised_model = self._load_model(self.config.supervised_model_path, model_type='joblib')

        self.workflow = self._build_workflow()
        logger.info("LangGraph ADA with GATRA initialized successfully")

    def _load_model(self, model_path, model_type='pickle'):
        local_path = os.path.join('models', os.path.basename(model_path))
        if os.path.exists(local_path):
            try:
                with open(local_path, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                pass
        return None

    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(ADAState)
        workflow.add_node("preprocess_data", self.preprocess_data_node)
        workflow.add_node("extract_features", self.extract_features_node)
        workflow.add_node("detect_anomalies_gatra", self.detect_anomalies_gatra_node)
        workflow.add_node("geo_enrichment", self.geo_enrichment_node)
        workflow.add_node("threat_intel_enrichment", self.threat_intel_enrichment_node)
        workflow.add_node("aggregate_enrichment", self.aggregate_enrichment_node)
        workflow.add_node("validate_detection", self.validate_detection_node)
        workflow.add_node("generate_alert", self.generate_alert_node)
        workflow.add_node("publish_alert", self.publish_alert_node)

        workflow.set_entry_point("preprocess_data")
        workflow.add_edge("preprocess_data", "extract_features")
        workflow.add_edge("extract_features", "detect_anomalies_gatra")
        workflow.add_conditional_edges("detect_anomalies_gatra", self.route_after_detection)
        
        workflow.add_edge("geo_enrichment", "aggregate_enrichment")
        workflow.add_edge("threat_intel_enrichment", "aggregate_enrichment")

        workflow.add_edge("aggregate_enrichment", "validate_detection")
        workflow.add_conditional_edges("validate_detection", self.route_after_validation, {"generate_alert": "generate_alert", "discard": END})
        workflow.add_edge("generate_alert", "publish_alert")
        workflow.add_edge("publish_alert", END)
        return workflow.compile()

    def preprocess_data_node(self, state: ADAState) -> Dict[str, Any]:
        logger.info("Preprocessing data")
        return state

    def extract_features_node(self, state: ADAState) -> Dict[str, Any]:
        alert_data = state["alert_data"]
        timestamp = pd.to_datetime(alert_data.get("timestamp", datetime.now()))
        
        features = {
            "duration": float(alert_data.get("duration") or 0),
            "bytes_sent": float(alert_data.get("bytes_sent") or 0),
            "bytes_received": float(alert_data.get("bytes_received") or 0),
            "port": int(alert_data.get("port") or 0),
            "protocol_encoded": {"tcp": 1, "udp": 2, "icmp": 3}.get(str(alert_data.get("protocol") or "").lower(), 0),
            "hour_of_day": timestamp.hour,
            "day_of_week": timestamp.dayofweek,
        }
        # Add padding to reach GATRA expected dim (10) if needed
        feature_vals = [features.get(f, 0.0) for f in self.config.features]
        while len(feature_vals) < 10:
            feature_vals.append(0.0)
            
        alert_data["extracted_features"] = features
        alert_data["gatra_vector"] = np.array(feature_vals)
        return {"alert_data": alert_data}

    def detect_anomalies_gatra_node(self, state: ADAState) -> Dict[str, Any]:
        logger.info("GATRA Anomaly Detection")
        alert_data = state["alert_data"]
        
        telemetry = {
            'source_ip': alert_data.get('source_ip'),
            'dest_ip': alert_data.get('dest_ip'),
            'features': alert_data.get('gatra_vector'),
            'event_type': 'network_log',
            'timestamp': alert_data.get('timestamp')
        }
        
        gatra_result = self.gatra.process_telemetry(telemetry)
        
        result = DetectionResult(
            anomaly_score=gatra_result.score,
            confidence=gatra_result.score,
            is_anomaly=gatra_result.is_anomaly,
            model_version="GATRA_Ensemble_v1",
            detection_timestamp=gatra_result.timestamp,
            feature_importance={},
            reasoning=gatra_result.reasoning,
            severity=gatra_result.severity.name
        )
        logger.info(f"GATRA result: {gatra_result.is_anomaly} (Score: {gatra_result.score:.4f})")
        return {"detection_result": result}

    def route_after_detection(self, state: ADAState) -> List[str]:
        if state["detection_result"]["is_anomaly"]:
            return ["geo_enrichment", "threat_intel_enrichment"]
        return ["geo_enrichment"] # Still enrich even if not anomaly for context

    def geo_enrichment_node(self, state: ADAState) -> Dict[str, Any]:
        return {"geo_location": {"country": "Unknown"}}

    def threat_intel_enrichment_node(self, state: ADAState) -> Dict[str, Any]:
        return {"threat_intelligence": {"status": "checked"}}

    def aggregate_enrichment_node(self, state: ADAState) -> Dict[str, Any]:
        enrichment_data = EnrichmentData(
            geo_location=state.get("geo_location"),
            threat_intelligence=state.get("threat_intelligence"),
            historical_context=None,
            asset_context=None,
        )
        return {"enrichment_data": enrichment_data}

    def validate_detection_node(self, state: ADAState) -> Dict[str, Any]:
        detection_result = state.get("detection_result")
        # GATRA handles validation via its ensemble, so we just check the flag here
        should_publish = detection_result.get("is_anomaly", False)
        return {"should_publish": should_publish}

    def route_after_validation(self, state: ADAState) -> str:
        return "generate_alert" if state.get("should_publish") else "discard"

    def generate_alert_node(self, state: ADAState) -> Dict[str, Any]:
        final_alert = {
            "alert_id": f"gatra-{state['alert_data']['log_id']}",
            "details": state["alert_data"],
            "detection": state["detection_result"],
            "enrichment": state.get("enrichment_data"),
        }
        return {"final_alert": final_alert}

    def publish_alert_node(self, state: ADAState) -> Dict[str, Any]:
        final_alert = state.get('final_alert')
        if not final_alert:
            logger.warning("No final alert found to publish")
            return {}

        logger.info(f"Publishing alert: {final_alert['alert_id']} - Severity: {final_alert['detection'].get('severity')}")
        
        try:
            # Topic name for TAA consumption
            topic_path = self.publisher.topic_path(self.config.project_id, "ada-alerts")
            
            # Prepare message for TAA
            message_data = {
                "alarm_id": final_alert['alert_id'],
                "timestamp": datetime.now().isoformat(),
                "confidence": float(final_alert['detection'].get('confidence', 0.0)),
                "is_anomaly": True,
                "alert_severity": final_alert['detection'].get('severity', 'LOW'),
                "source_ip": final_alert['details'].get('source_ip', 'Unknown'),
                "destination_ip": final_alert['details'].get('dest_ip', 'Unknown'),
                "protocol": final_alert['details'].get('protocol', 'Unknown'),
                "port": final_alert['details'].get('port'),
                "detection_reasons": final_alert['detection'].get('reasoning', []),
                "raw_log": final_alert['details'].get('raw_log', {}),
                "source": "langgraph_ada"
            }
            
            message_bytes = json.dumps(message_data).encode("utf-8")
            future = self.publisher.publish(topic_path, message_bytes)
            message_id = future.result()
            
            logger.info(f"Successfully published alert {final_alert['alert_id']} to {topic_path}. Message ID: {message_id}")
            return {"workflow_status": "completed"}
            
        except Exception as e:
            logger.error(f"Failed to publish alert to Pub/Sub: {e}")
            return {"workflow_status": "failed", "error_messages": [str(e)]}

    def _create_alert_data_from_raw(self, raw_alert: dict) -> AlertData:
        events_str = raw_alert.get("events", "{}")
        events_data = json.loads(events_str) if isinstance(events_str, str) else events_str
        return AlertData(
            log_id=str(raw_alert.get("alarmId", datetime.now().timestamp())),
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
        alert_data = self._create_alert_data_from_raw(raw_alert)
        initial_state = {"alert_data": alert_data, "error_messages": [], "workflow_status": "started"}
        return await self.workflow.ainvoke(initial_state)

if __name__ == "__main__":
    # Quick test
    async def test():
        config = ADAConfig(BIGQUERY_PROJECT_ID="test", BIGQUERY_DATASET_ID="test", BIGQUERY_TABLE_ID="test", BIGQUERY_RESULTS_TABLE_ID="test")
        agent = LangGraphAnomalyDetectionAgent(config)
        test_alert = {"alarmId": "test-123", "events": json.dumps({"source_ip": "1.2.3.4", "dest_ip": "8.8.8.8", "protocol": "tcp", "bytes_sent": 5000})}
        result = await agent.process_single_alert(test_alert)
        print(f"Final State: {result['detection_result']}")

    asyncio.run(test())
