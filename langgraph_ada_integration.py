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
print("DEBUG: PROJECT_ID is", os.environ.get("PROJECT_ID"))
import json
import logging
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional, Literal, TypedDict, Annotated
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
# from langgraph.checkpoint.postgres import PostgresSaver
# from langgraph.checkpoint.redis import RedisSaver
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# Google Cloud imports
from google.cloud import storage, pubsub_v1, aiplatform, firestore
from google.cloud.sql.connector import Connector
import sqlalchemy

# ML and data processing
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
import pickle
import redis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LangGraphADA")

# State definitions for the ADA workflow
class AlertData(TypedDict):
    """Structure for raw alert data"""
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
    """Structure for detection results"""
    anomaly_score: float
    confidence: float
    is_anomaly: bool
    model_version: str
    detection_timestamp: str
    feature_importance: Dict[str, float]

class EnrichmentData(TypedDict):
    """Structure for enriched context data"""
    geo_location: Optional[Dict[str, Any]]
    threat_intelligence: Optional[Dict[str, Any]]
    historical_context: Optional[Dict[str, Any]]
    asset_context: Optional[Dict[str, Any]]

class ADAState(TypedDict):
    """Main state for the ADA workflow"""
    # Input data
    alert_data: AlertData
    batch_id: str
    processing_mode: Literal["realtime", "batch", "training"]
    
    # Detection results
    detection_result: Optional[DetectionResult]
    enrichment_data: Optional[EnrichmentData]
    
    # Workflow control
    workflow_status: Literal["started", "processing", "enriching", "validating", "completed", "failed"]
    retry_count: int
    error_messages: List[str]
    
    # Output
    final_alert: Optional[Dict[str, Any]]
    should_publish: bool
    confidence_threshold: float
    
    # Coordination with other agents
    taa_feedback: Optional[Dict[str, Any]]
    model_updates: List[Dict[str, Any]]
    
    # Performance metrics
    processing_time: Optional[float]
    model_performance: Optional[Dict[str, Any]]

@dataclass
class ADAConfig:
    """Configuration for the ADA system"""
    project_id: str = "chronicle-dev-2be9"
    location: str = "us-central1"
    
    # Database connections
    postgres_connection_string: str = ""
    redis_url: str = ""
    
    # Model configuration
    model_bucket: str = "security-models"
    model_path: str = "anomaly-detection/isolation_forest.pkl"
    confidence_threshold: float = 0.8
    batch_size: int = 100
    
    # Pub/Sub topics
    alert_topic: str = "security-alerts"
    feedback_topic: str = "analysis-feedback"
    model_update_topic: str = "ada-model-updates"
    
    # Performance settings
    max_retries: int = 3
    processing_timeout: int = 300
    parallel_enrichment: bool = True
    
    # Feature extraction
    features: List[str] = None
    
    # Local mode disables cloud calls and fits the model on synthetic data
    local_mode: bool = False
    
    def __post_init__(self):
        if self.features is None:
            self.features = [
                "duration", "bytes_sent", "bytes_received", "port", 
                "protocol_encoded", "time_of_day", "day_of_week"
            ]

class LangGraphAnomalyDetectionAgent:
    """Enhanced Anomaly Detection Agent with LangGraph orchestration"""
    
    def __init__(self, config: ADAConfig):
        self.config = config
        
        # Initialize Google Cloud clients
        self.storage_client = storage.Client(project=config.project_id)
        self.publisher = pubsub_v1.PublisherClient()
        if not config.local_mode:
            self.db = firestore.Client(project=config.project_id)
        
        # Initialize Vertex AI
        aiplatform.init(project=config.project_id, location=config.location)
        
        # Load ML model
        self.model = self._load_model()
        
        # Initialize Redis client for checkpointing or caching (only if URL is set)
        if config.redis_url and config.redis_url.strip():
            self.redis_client = redis.from_url(config.redis_url)
        else:
            self.redis_client = None
        
        # Create checkpointer for state persistence
        self.checkpointer = self._create_checkpointer()
        
        # Build the LangGraph workflow
        self.workflow = self._build_workflow()
        
        logger.info("LangGraph ADA initialized successfully")
    
    def _create_checkpointer(self):
        """Checkpointing disabled for local testing"""
        return None
    
    def _load_model(self):
        """Load the anomaly detection model"""
        try:
            bucket = self.storage_client.bucket(self.config.model_bucket)
            blob = bucket.blob(self.config.model_path)
            
            if blob.exists():
                model_data = blob.download_as_bytes()
                model = pickle.loads(model_data)
                logger.info("Model loaded from GCS")
                if not hasattr(model, "estimators_"):
                    try:
                        synthetic_data = self._generate_synthetic_data()
                        model.fit(synthetic_data)
                        logger.info("Loaded IsolationForest model was not fitted; fitted on synthetic data.")
                    except Exception as e:
                        logger.warning(f"Could not fit loaded IsolationForest model: {e}")
                return model
        except Exception as e:
            logger.warning(f"Could not load model from GCS: {e}")
        
        # Create new model if none exists
        logger.info("Creating new Isolation Forest model")
        model = IsolationForest(
            contamination=0.01,
            random_state=42,
            n_estimators=100
        )
        
        try:
            # Fit model on synthetic data in local mode
            synthetic_data = self._generate_synthetic_data()
            model.fit(synthetic_data)
            logger.info("IsolationForest model fitted on synthetic data.")
        except Exception as e:
            logger.warning(f"Could not fit IsolationForest model: {e}")
        return model

    
    def _generate_synthetic_data(self):
        """Generate synthetic data for local mode testing"""
        np.random.seed(42)
        data = np.random.rand(1000, len(self.config.features))
        return data
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for anomaly detection"""
        
        # Create the workflow graph
        workflow = StateGraph(ADAState)
        
        # Add Scale AI preprocessing node before other processing
        workflow.add_node("scale_ai_preprocessing", self.scale_ai_preprocessing_node)
        workflow.add_node("preprocess_data", self.preprocess_data_node)
        workflow.add_node("extract_features", self.extract_features_node)
        workflow.add_node("detect_anomalies", self.detect_anomalies_node)
        workflow.add_node("parallel_enrichment", self.parallel_enrichment_node)
        workflow.add_node("validate_detection", self.validate_detection_node)
        workflow.add_node("generate_alert", self.generate_alert_node)
        workflow.add_node("publish_alert", self.publish_alert_node)
        workflow.add_node("handle_error", self.handle_error_node)
        workflow.add_node("update_metrics", self.update_metrics_node)
        
        # Add enrichment sub-nodes
        workflow.add_node("geo_enrichment", self.geo_enrichment_node)
        workflow.add_node("threat_intel_enrichment", self.threat_intel_enrichment_node)
        workflow.add_node("historical_enrichment", self.historical_enrichment_node)
        workflow.add_node("asset_enrichment", self.asset_enrichment_node)
        
        # Define the workflow edges
        workflow.add_edge(START, "scale_ai_preprocessing")
        workflow.add_edge("scale_ai_preprocessing", "preprocess_data")
        workflow.add_edge("preprocess_data", "extract_features")
        workflow.add_edge("extract_features", "detect_anomalies")
        
        # Conditional routing after detection
        workflow.add_conditional_edges(
            "detect_anomalies",
            self.route_after_detection,
            {
                "enrich": "parallel_enrichment",
                "validate": "validate_detection",
                "error": "handle_error"
            }
        )
        
        workflow.add_edge("parallel_enrichment", "validate_detection")
        
        # Conditional routing after validation
        workflow.add_conditional_edges(
            "validate_detection",
            self.route_after_validation,
            {
                "generate_alert": "generate_alert",
                "discard": "update_metrics",
                "retry": "detect_anomalies",
                "error": "handle_error"
            }
        )
        
        workflow.add_edge("generate_alert", "publish_alert")
        workflow.add_edge("publish_alert", "update_metrics")
        workflow.add_edge("handle_error", "update_metrics")
        workflow.add_edge("update_metrics", END)
        
        # Compile the workflow
        compiled_workflow = workflow.compile(
            checkpointer=self.checkpointer,
            interrupt_before=["publish_alert"],  # Allow human intervention before publishing
        )
        
        return compiled_workflow
    
    def scale_ai_preprocessing_node(self, state: ADAState) -> dict:
        """
        Scale AI Preprocessing & Ingestion Phase:
        - Deduplicate logs
        - Redact/standardize sensitive fields
        - Label logs (Normal, Suspicious, Attack, Kill Chain stage)
        """
        import hashlib
        import re
        logger.info(f"[Scale AI] Preprocessing for batch: {state.get('batch_id', 'unknown')}")
        alert_data = state.get("alert_data")
        if not alert_data:
            return {"workflow_status": "failed", "error_messages": ["No alert_data found"]}

        # 1. Deduplication (by log_id or hash of raw_log)
        seen_hashes = set()
        logs = alert_data if isinstance(alert_data, list) else [alert_data]
        deduped_logs = []
        for log in logs:
            raw = json.dumps(log.get("raw_log", log), sort_keys=True)
            h = hashlib.md5(raw.encode()).hexdigest()
            if h not in seen_hashes:
                seen_hashes.add(h)
                deduped_logs.append(log)
        logger.info(f"[Scale AI] Deduplicated logs: {len(deduped_logs)}")

        # 2. Redact/standardize sensitive fields
        def redact_username(name):
            return "user" if name else "user"
        for log in deduped_logs:
            # Keep source_ip and dest_ip for detection and investigation
            if "username" in log:
                log["username"] = redact_username(log["username"])
            if "timestamp" in log:
                try:
                    log["timestamp"] = pd.to_datetime(log["timestamp"]).isoformat()
                except Exception:
                    log["timestamp"] = datetime.now().isoformat()

        # 3. Labeling & semantic enrichment
        for log in deduped_logs:
            # Simple heuristics for demonstration
            if "malware" in str(log.get("raw_log", "")).lower() or "attack" in str(log.get("raw_log", "")).lower():
                log["label"] = "Confirmed Attack"
                log["kill_chain_stage"] = "Action on Objectives"
            elif "scan" in str(log.get("raw_log", "")).lower() or "suspicious" in str(log.get("raw_log", "")).lower():
                log["label"] = "Suspicious Pattern"
                log["kill_chain_stage"] = "Reconnaissance"
            else:
                log["label"] = "Normal Traffic"
                log["kill_chain_stage"] = "None"

        # If batch, keep as list; if single, return single dict
        processed = deduped_logs if isinstance(alert_data, list) else deduped_logs[0]
        return {
            "alert_data": processed,
            "workflow_status": "processing",
            "processing_time": 0.0
        }

    
    # Node implementations
    def preprocess_data_node(self, state: ADAState) -> Dict[str, Any]:
        """Preprocess and validate input data"""
        logger.info(f"Preprocessing data for batch: {state.get('batch_id', 'unknown')}")
        
        start_time = time.time()
        
        try:
            alert_data = state["alert_data"]
            
            # Validate required fields
            if not alert_data.get("log_id"):
                raise ValueError("Missing log_id in alert data")
            
            # Normalize timestamp format
            if alert_data.get("timestamp"):
                try:
                    # Convert to standard format if needed
                    timestamp = pd.to_datetime(alert_data["timestamp"])
                    alert_data["timestamp"] = timestamp.isoformat()
                except Exception as e:
                    logger.warning(f"Could not parse timestamp: {e}")
                    alert_data["timestamp"] = datetime.now().isoformat()
            
            # Set default values for missing fields
            defaults = {
                "source_ip": "unknown",
                "dest_ip": "unknown",
                "protocol": "unknown",
                "port": 0,
                "bytes_sent": 0,
                "bytes_received": 0,
                "duration": 0.0
            }
            
            for key, default_value in defaults.items():
                if alert_data.get(key) is None:
                    alert_data[key] = default_value
            
            processing_time = time.time() - start_time
            
            return {
                "alert_data": alert_data,
                "workflow_status": "processing",
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            return {
                "workflow_status": "failed",
                "error_messages": state.get("error_messages", []) + [str(e)]
            }
    
    def extract_features_node(self, state: ADAState) -> Dict[str, Any]:
        """Extract features for ML model"""
        logger.info("Extracting features for anomaly detection")
        
        try:
            alert_data = state["alert_data"]
            
            # Extract timestamp features
            timestamp = pd.to_datetime(alert_data["timestamp"])
            time_features = {
                "hour_of_day": timestamp.hour,
                "day_of_week": timestamp.dayofweek,
                "is_weekend": timestamp.dayofweek >= 5
            }
            
            # Extract network features
            network_features = {
                "bytes_sent": float(alert_data.get("bytes_sent", 0)),
                "bytes_received": float(alert_data.get("bytes_received", 0)),
                "duration": float(alert_data.get("duration", 0)),
                "port": int(alert_data.get("port", 0))
            }
            
            # Calculate derived features
            total_bytes = network_features["bytes_sent"] + network_features["bytes_received"]
            network_features["total_bytes"] = total_bytes
            network_features["bytes_ratio"] = (
                network_features["bytes_sent"] / max(network_features["bytes_received"], 1)
            )
            
            # Encode categorical features
            protocol_encoding = {
                "tcp": 1, "udp": 2, "icmp": 3, "unknown": 0
            }
            categorical_features = {
                "protocol_encoded": protocol_encoding.get(
                    alert_data.get("protocol", "unknown").lower(), 0
                )
            }
            
            # Combine all features
            extracted_features = {
                **time_features,
                **network_features,
                **categorical_features
            }
            
            # Update alert data with extracted features
            updated_alert_data = state["alert_data"].copy()
            updated_alert_data["extracted_features"] = extracted_features
            
            return {
                "alert_data": updated_alert_data,
                "workflow_status": "processing"
            }
            
        except Exception as e:
            logger.error(f"Error in feature extraction: {e}")
            return {
                "workflow_status": "failed",
                "error_messages": state.get("error_messages", []) + [str(e)]
            }
    
    def detect_anomalies_node(self, state: ADAState) -> Dict[str, Any]:
        """Perform anomaly detection using ML model"""
        logger.info("Performing anomaly detection")
        
        try:
            alert_data = state["alert_data"]
            extracted_features = alert_data.get("extracted_features", {})
            
            # Prepare feature vector
            feature_vector = []
            feature_names = []
            
            for feature_name in self.config.features:
                if feature_name in extracted_features:
                    feature_vector.append(extracted_features[feature_name])
                    feature_names.append(feature_name)
                else:
                    # Use default value for missing features
                    feature_vector.append(0.0)
                    feature_names.append(feature_name)
            
            # Reshape for single prediction
            feature_array = np.array(feature_vector).reshape(1, -1)
            
            # Get anomaly score
            anomaly_scores = self.model.decision_function(feature_array)
            raw_score = anomaly_scores[0]
            
            # Convert to 0-1 range (higher = more anomalous)
            normalized_score = max(0, min(1, (0.5 - raw_score) * 2))
            
            # Determine if anomalous
            is_anomaly = normalized_score > self.config.confidence_threshold
            
            # Calculate confidence based on distance from threshold
            confidence = abs(normalized_score - self.config.confidence_threshold) + 0.5
            confidence = min(1.0, confidence)
            
            # Create detection result
            detection_result = DetectionResult(
                anomaly_score=float(normalized_score),
                confidence=float(confidence),
                is_anomaly=bool(is_anomaly),
                model_version="1.0",  # Would be dynamic in production
                detection_timestamp=datetime.now().isoformat(),
                feature_importance={
                    name: abs(float(value)) for name, value in zip(feature_names, feature_vector)
                }
            )
            
            return {
                "detection_result": detection_result,
                "workflow_status": "processing"
            }
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return {
                "workflow_status": "failed",
                "error_messages": state.get("error_messages", []) + [str(e)]
            }
    
    def parallel_enrichment_node(self, state: ADAState) -> Dict[str, Any]:
        """Coordinate parallel enrichment processes"""
        logger.info("Starting parallel enrichment")
        
        alert_data = state["alert_data"]
        
        # Create enrichment tasks
        enrichment_tasks = []
        
        if alert_data.get("source_ip") and alert_data["source_ip"] != "unknown":
            enrichment_tasks.append("geo_enrichment")
            enrichment_tasks.append("threat_intel_enrichment")
        
        if alert_data.get("log_id"):
            enrichment_tasks.append("historical_enrichment")
        
        if alert_data.get("dest_ip") and alert_data["dest_ip"] != "unknown":
            enrichment_tasks.append("asset_enrichment")
        
        return enrichment_tasks
    
    def geo_enrichment_node(self, state: ADAState) -> Dict[str, Any]:
        """Enrich with geographical information"""
        logger.info("Performing geo enrichment")
        try:
            alert_data = state.get("alert_data")
            if alert_data is None or not isinstance(alert_data, dict):
                logger.error(f"Geo enrichment error: alert_data is None or not a dict. Value: {alert_data}")
                return {"workflow_status": "failed", "error_messages": ["Invalid alert_data for geo enrichment"]}
            source_ip = alert_data.get("source_ip")
            geo_data = {
                "country": "Unknown",
                "city": "Unknown",
                "latitude": 0.0,
                "longitude": 0.0,
                "is_tor_exit": False,
                "is_known_vpn": False
            }
            if source_ip and isinstance(source_ip, str) and source_ip.startswith("10."):
                geo_data.update({
                    "country": "Internal",
                    "city": "Corporate Network",
                    "is_internal": True
                })
            enrichment_data = state.get("enrichment_data", {})
            if enrichment_data is None or not isinstance(enrichment_data, dict):
                logger.error(f"Geo enrichment: enrichment_data is None or not a dict. Value: {enrichment_data}")
                enrichment_data = {}
            enrichment_data["geo_location"] = geo_data
            return {"enrichment_data": enrichment_data}
        except Exception as e:
            logger.error(f"Error in geo enrichment: {e}")
            logger.debug(f"State at error: alert_data={state.get('alert_data')}, enrichment_data={state.get('enrichment_data')}")
            return {"workflow_status": "failed", "error_messages": [f"Geo enrichment exception: {type(e).__name__}: {e}"]}
    
    def threat_intel_enrichment_node(self, state: ADAState) -> Dict[str, Any]:
        """Enrich with threat intelligence"""
        logger.info("Performing threat intelligence enrichment")
        try:
            alert_data = state.get("alert_data")
            if alert_data is None or not isinstance(alert_data, dict):
                logger.error(f"Threat intel enrichment error: alert_data is None or not a dict. Value: {alert_data}")
                return {"workflow_status": "failed", "error_messages": ["Invalid alert_data for threat intel enrichment"]}
            source_ip = alert_data.get("source_ip")
            threat_data = {
                "reputation_score": 50,  # 0-100, lower is worse
                "threat_categories": [],
                "last_seen_malicious": None,
                "associated_campaigns": [],
                "confidence": 0.5
            }
            if source_ip and isinstance(source_ip, str) and (source_ip.startswith("192.168.1.666") or source_ip.startswith("evil")):
                threat_data.update({
                    "reputation_score": 10,
                    "threat_categories": ["malware", "botnet"],
                    "last_seen_malicious": datetime.now().isoformat(),
                    "confidence": 0.9
                })
            enrichment_data = state.get("enrichment_data", {})
            if enrichment_data is None or not isinstance(enrichment_data, dict):
                logger.error(f"Threat intel enrichment: enrichment_data is None or not a dict. Value: {enrichment_data}")
                enrichment_data = {}
            enrichment_data["threat_intelligence"] = threat_data
            return {"enrichment_data": enrichment_data}
        except Exception as e:
            logger.error(f"Error in threat intel enrichment: {e}")
            logger.debug(f"State at error: alert_data={state.get('alert_data')}, enrichment_data={state.get('enrichment_data')}")
            return {"workflow_status": "failed", "error_messages": [f"Threat intel enrichment exception: {type(e).__name__}: {e}"]}
    
    def historical_enrichment_node(self, state: ADAState) -> Dict[str, Any]:
        """Enrich with historical context"""
        logger.info("Performing historical enrichment")
        try:
            alert_data = state.get("alert_data")
            if alert_data is None or not isinstance(alert_data, dict):
                logger.error(f"Historical enrichment error: alert_data is None or not a dict. Value: {alert_data}")
                return {"workflow_status": "failed", "error_messages": ["Invalid alert_data for historical enrichment"]}
            source_ip = alert_data.get("source_ip")
            historical_data = {
                "previous_alerts": 0,
                "first_seen": None,
                "last_seen": None,
                "alert_frequency": 0.0,
                "typical_behavior": {},
                "anomaly_history": []
            }
            if source_ip:
                # Simulate some previous activity
                historical_data.update({
                    "previous_alerts": 3,
                    "first_seen": (datetime.now() - timedelta(days=30)).isoformat(),
                    "last_seen": (datetime.now() - timedelta(hours=6)).isoformat(),
                    "alert_frequency": 0.1  # alerts per day
                })
            enrichment_data = state.get("enrichment_data", {})
            if enrichment_data is None or not isinstance(enrichment_data, dict):
                logger.error(f"Historical enrichment: enrichment_data is None or not a dict. Value: {enrichment_data}")
                enrichment_data = {}
            enrichment_data["historical_context"] = historical_data
            return {"enrichment_data": enrichment_data}
        except Exception as e:
            logger.error(f"Error in historical enrichment: {e}")
            logger.debug(f"State at error: alert_data={state.get('alert_data')}, enrichment_data={state.get('enrichment_data')}")
            return {"workflow_status": "failed", "error_messages": [f"Historical enrichment exception: {type(e).__name__}: {e}"]}
    
    def asset_enrichment_node(self, state: ADAState) -> Dict[str, Any]:
        """Enrich with asset context"""
        logger.info("Performing asset enrichment")
        try:
            alert_data = state.get("alert_data")
            if alert_data is None or not isinstance(alert_data, dict):
                logger.error(f"Asset enrichment error: alert_data is None or not a dict. Value: {alert_data}")
                return {"workflow_status": "failed", "error_messages": ["Invalid alert_data for asset enrichment"]}
            dest_ip = alert_data.get("dest_ip")
            asset_data = {
                "asset_type": "unknown",
                "criticality": "low",
                "owner": "unknown",
                "services": [],
                "security_group": "default",
                "compliance_requirements": []
            }
            if dest_ip and isinstance(dest_ip, str):
                if dest_ip.startswith("10.1."):
                    asset_data.update({
                        "asset_type": "database_server",
                        "criticality": "high",
                        "owner": "database_team",
                        "services": ["mysql", "postgresql"],
                        "compliance_requirements": ["pci-dss", "sox"]
                    })
                elif dest_ip.startswith("10.2."):
                    asset_data.update({
                        "asset_type": "web_server",
                        "criticality": "medium",
                        "owner": "web_team",
                        "services": ["http", "https"]
                    })
            enrichment_data = state.get("enrichment_data", {})
            if enrichment_data is None or not isinstance(enrichment_data, dict):
                logger.error(f"Asset enrichment: enrichment_data is None or not a dict. Value: {enrichment_data}")
                enrichment_data = {}
            enrichment_data["asset_context"] = asset_data
            return {"enrichment_data": enrichment_data}
        except Exception as e:
            logger.error(f"Error in asset enrichment: {e}")
            logger.debug(f"State at error: alert_data={state.get('alert_data')}, enrichment_data={state.get('enrichment_data')}")
            return {"workflow_status": "failed", "error_messages": [f"Asset enrichment exception: {type(e).__name__}: {e}"]}
    
    def validate_detection_node(self, state: ADAState) -> Dict[str, Any]:
        """Validate detection results with enriched context"""
        logger.info("Validating detection with enriched context")
        try:
            detection_result = state.get("detection_result")
            enrichment_data = state.get("enrichment_data", {})
            if detection_result is None or not isinstance(detection_result, dict):
                logger.error(f"Validation error: detection_result is None or not a dict. Value: {detection_result}")
                return {
                    "workflow_status": "failed",
                    "error_messages": state.get("error_messages", []) + [
                        f"No valid detection result available (got {type(detection_result).__name__})"
                    ]
                }
            if enrichment_data is None or not isinstance(enrichment_data, dict):
                logger.error(f"Validation error: enrichment_data is None or not a dict. Value: {enrichment_data}")
                enrichment_data = {}
            # Apply enrichment-based adjustments
            adjusted_confidence = detection_result.get("confidence", 0.0)
            adjusted_score = detection_result.get("anomaly_score", 0.0)
            # Adjust based on threat intelligence
            threat_intel = enrichment_data.get("threat_intelligence", {})
            if isinstance(threat_intel, dict) and threat_intel:
                reputation_score = threat_intel.get("reputation_score", 50)
                if reputation_score < 20:  # Known bad IP
                    adjusted_confidence = min(1.0, adjusted_confidence * 1.5)
                    adjusted_score = min(1.0, adjusted_score * 1.3)
                elif reputation_score > 80:  # Known good IP
                    adjusted_confidence = max(0.1, adjusted_confidence * 0.8)
                    adjusted_score = max(0.0, adjusted_score * 0.7)
            # Adjust based on asset criticality
            asset_context = enrichment_data.get("asset_context", {})
            if isinstance(asset_context, dict) and asset_context:
                criticality = asset_context.get("criticality", "low")
                if criticality == "high":
                    adjusted_confidence = min(1.0, adjusted_confidence * 1.2)
                elif criticality == "low":
                    adjusted_confidence = max(0.1, adjusted_confidence * 0.9)
            # Adjust based on historical context
            historical_context = enrichment_data.get("historical_context", {})
            if isinstance(historical_context, dict) and historical_context:
                previous_alerts = historical_context.get("previous_alerts", 0)
                if previous_alerts > 5:  # Frequent offender
                    adjusted_confidence = min(1.0, adjusted_confidence * 1.1)
            # Update detection result
            updated_detection = detection_result.copy()
            updated_detection["confidence"] = float(adjusted_confidence)
            updated_detection["anomaly_score"] = float(adjusted_score)
            updated_detection["is_anomaly"] = adjusted_score > self.config.confidence_threshold
            return {
                "detection_result": updated_detection,
                "workflow_status": "processing"
            }
        except Exception as e:
            logger.error(f"Error in validation: {e}")
            logger.debug(f"State at error: detection_result={state.get('detection_result')}, enrichment_data={state.get('enrichment_data')}")
            return {
                "workflow_status": "failed",
                "error_messages": state.get("error_messages", []) + [
                    f"Validation exception: {type(e).__name__}: {e}"
                ]
            }
    
    def generate_alert_node(self, state: ADAState) -> Dict[str, Any]:
        """Generate the final alert structure"""
        logger.info("Generating final alert")
        
        try:
            alert_data = state["alert_data"]
            detection_result = state["detection_result"]
            enrichment_data = state.get("enrichment_data", {})
            
            # Create comprehensive alert
            final_alert = {
                "alert_id": f"ADA-{datetime.now().strftime('%Y%m%d%H%M%S')}-{hash(str(alert_data)) % 10000}",
                "timestamp": datetime.now().isoformat(),
                "source": "anomaly_detection_agent",
                "version": "2.0",  # LangGraph version
                
                # Detection information
                "anomaly_score": detection_result["anomaly_score"],
                "confidence": detection_result["confidence"],
                "severity": self._calculate_severity(detection_result, enrichment_data),
                "alert_type": "ANOMALY_DETECTION",
                
                # Original data
                "log_data": alert_data,
                
                # Enrichment context
                "enrichment": enrichment_data,
                
                # Model information
                "model_info": {
                    "model_version": detection_result["model_version"],
                    "detection_timestamp": detection_result["detection_timestamp"],
                    "feature_importance": detection_result["feature_importance"]
                },
                
                # Workflow metadata
                "workflow_metadata": {
                    "batch_id": state.get("batch_id"),
                    "processing_mode": state.get("processing_mode"),
                    "processing_time": state.get("processing_time"),
                    "retry_count": state.get("retry_count", 0)
                }
            }
            
            return {
                "final_alert": final_alert,
                "should_publish": True,
                "workflow_status": "processing"
            }
            
        except Exception as e:
            logger.error(f"Error in alert generation: {e}")
            return {
                "workflow_status": "failed",
                "error_messages": state.get("error_messages", []) + [str(e)]
            }
    
    def publish_alert_node(self, state: ADAState) -> Dict[str, Any]:
        """Publish the alert to Pub/Sub"""
        logger.info("Publishing alert")
        
        try:
            if not state.get("should_publish"):
                return {"workflow_status": "completed"}
            
            final_alert = state["final_alert"]
            
            # Publish to Pub/Sub
            topic_path = self.publisher.topic_path(
                self.config.project_id, 
                self.config.alert_topic
            )
            
            message_data = json.dumps(final_alert).encode("utf-8")
            future = self.publisher.publish(topic_path, message_data)
            message_id = future.result()
            
            logger.info(f"Alert published with message ID: {message_id}")
            
            # Save feedback to Firestore for traceability (skip if local_mode)
            if not self.config.local_mode:
                self._save_feedback_to_firestore(state, final_alert)
            
            return {
                "workflow_status": "completed",
                "message_id": message_id
            }
            
        except Exception as e:
            logger.error(f"Error in publishing: {e}")
            return {
                "workflow_status": "failed",
                "error_messages": state.get("error_messages", []) + [str(e)]
            }
    
    def handle_error_node(self, state: ADAState) -> Dict[str, Any]:
        """Handle errors and determine retry strategy"""
        logger.info("Handling workflow errors")
        
        error_messages = state.get("error_messages", [])
        retry_count = state.get("retry_count", 0)
        
        # Log errors
        for error in error_messages:
            logger.error(f"Workflow error: {error}")
        
        # Determine if retry is appropriate
        if retry_count < self.config.max_retries:
            logger.info(f"Retrying workflow (attempt {retry_count + 1})")
            return {
                "retry_count": retry_count + 1,
                "workflow_status": "processing",
                "error_messages": []  # Clear errors for retry
            }
        else:
            logger.error(f"Max retries exceeded, marking as failed")
            
            # Save failed alert to Firestore for manual review (skip if local_mode)
            if not self.config.local_mode:
                self._save_failed_alert_to_firestore(state)
            
            return {"workflow_status": "failed"}
    
    def update_metrics_node(self, state: ADAState) -> Dict[str, Any]:
        """Update performance metrics"""
        logger.info("Updating performance metrics")
        
        try:
            # Calculate total processing time
            total_processing_time = time.time() - state.get("start_time", time.time())
            
            # Create metrics
            metrics = {
                "batch_id": state.get("batch_id"),
                "processing_time": total_processing_time,
                "workflow_status": state.get("workflow_status"),
                "retry_count": state.get("retry_count", 0),
                "alert_generated": state.get("should_publish", False),
                "timestamp": datetime.now().isoformat()
            }
            
            if state.get("detection_result"):
                metrics.update({
                    "anomaly_score": state["detection_result"]["anomaly_score"],
                    "confidence": state["detection_result"]["confidence"],
                    "is_anomaly": state["detection_result"]["is_anomaly"]
                })
            
            # Save metrics to Firestore (skip if local_mode)
            if not self.config.local_mode:
                self._update_performance_metrics(state)
            
            return {"model_performance": metrics}
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
            return {}
    
    def _update_performance_metrics(self, state: ADAState):
        """Update model performance metrics in Firestore (skipped in local_mode)"""
        if self.config.local_mode:
            logger.info("Local mode: skipping Firestore metrics update.")
            return
        try:
            # Connect to Firestore
            db = firestore.Client(project=self.config.project_id)
            metrics_ref = db.collection("model_performance").document("isolation_forest")
            metrics_ref.set(state.get("model_performance", {}), merge=True)
            logger.info("Model performance metrics updated in Firestore")
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
            return {}
    
    def _save_feedback_to_firestore(self, state: ADAState, feedback: Dict[str, Any]):
        """Save feedback to Firestore for traceability (skipped in local_mode)"""
        if self.config.local_mode:
            logger.info("Local mode: skipping Firestore feedback save.")
            return
        try:
            # Connect to Firestore
            db = firestore.Client(project=self.config.project_id)
            feedback_ref = db.collection("feedback").document(feedback["alert_id"])
            feedback_ref.set(feedback, merge=True)
            logger.info("Feedback saved to Firestore")
        except Exception as e:
            logger.error(f"Error saving feedback: {e}")
            return {}
    
    def _save_failed_alert_to_firestore(self, state: ADAState):
        """Save failed alert to Firestore for manual review (skipped in local_mode)"""
        if self.config.local_mode:
            logger.info("Local mode: skipping Firestore failed alert save.")
            return
        try:
            # Connect to Firestore
            db = firestore.Client(project=self.config.project_id)
            failed_alert = {
                "alert_id": f"ADA-FAILED-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "timestamp": datetime.now().isoformat(),
                "original_data": state.get("alert_data"),
                "error_messages": state.get("error_messages", []),
                "retry_count": state.get("retry_count", 0),
                "batch_id": state.get("batch_id")
            }
            db.collection("ada_failed_alerts").add(failed_alert)
            logger.info("Failed alert saved to Firestore")
        except Exception as e:
            logger.error(f"Error saving failed alert: {e}")
            return {}
    
    # Conditional routing functions
    def route_after_detection(self, state: ADAState) -> Literal["enrich", "validate", "error"]:
        """Route after anomaly detection based on results"""
        
        if state.get("workflow_status") == "failed":
            return "error"
        
        detection_result = state.get("detection_result")
        if not detection_result:
            return "error"
        
        # If high confidence, proceed to enrichment
        if detection_result["confidence"] > 0.7:
            return "enrich"
        
        # Otherwise, validate directly
        return "validate"
    
    def route_after_validation(self, state: ADAState) -> Literal["generate_alert", "discard", "retry", "error"]:
        """Route after validation based on final assessment"""
        
        if state.get("workflow_status") == "failed":
            error_count = len(state.get("error_messages", []))
            retry_count = state.get("retry_count", 0)
            
            if retry_count < self.config.max_retries and error_count < 3:
                return "retry"
            else:
                return "error"
        
        detection_result = state.get("detection_result")
        if not detection_result:
            return "error"
        
        # Generate alert if anomaly detected with sufficient confidence
        if detection_result["is_anomaly"] and detection_result["confidence"] > self.config.confidence_threshold:
            return "generate_alert"
        
        return "discard"
    
    def _calculate_severity(self, detection_result: DetectionResult, enrichment_data: EnrichmentData) -> str:
        """Calculate alert severity based on detection and enrichment"""
        
        score = detection_result["anomaly_score"]
        confidence = detection_result["confidence"]
        
        # Base severity from score and confidence
        composite_score = (score * 0.7) + (confidence * 0.3)
        
        # Adjust based on enrichment data
        threat_intel = enrichment_data.get("threat_intelligence", {})
        if threat_intel:
            reputation = threat_intel.get("reputation_score", 50)
            if reputation < 20:
                composite_score = min(1.0, composite_score * 1.3)
        
        asset_context = enrichment_data.get("asset_context", {})
        if asset_context:
            criticality = asset_context.get("criticality", "low")
            if criticality == "high":
                composite_score = min(1.0, composite_score * 1.2)
        
        # Determine severity level
        if composite_score > 0.8:
            return "critical"
        elif composite_score > 0.6:
            return "high"
        elif composite_score > 0.4:
            return "medium"
        else:
            return "low"
    
    # Public interface methods
    async def process_alert(self, alert_data: AlertData, batch_id: str = None) -> Dict[str, Any]:
        """Process a single alert through the LangGraph workflow"""
        
        if batch_id is None:
            batch_id = f"batch_{int(time.time())}_{hash(str(alert_data)) % 1000}"
        
        # Create initial state
        initial_state = ADAState(
            alert_data=alert_data,
            batch_id=batch_id,
            processing_mode="realtime",
            workflow_status="started",
            retry_count=0,
            error_messages=[],
            final_alert=None,
            should_publish=False,
            confidence_threshold=self.config.confidence_threshold,
            taa_feedback=None,
            model_updates=[],
            processing_time=None,
            model_performance=None,
            detection_result=None,
            enrichment_data=None
        )
        
        # Add start time for metrics
        initial_state["start_time"] = time.time()
        
        # Execute the workflow
        config = {
            "configurable": {
                "thread_id": batch_id,
                "checkpoint_ns": "ada_workflow"
            }
        }
        
        try:
            result = await self.workflow.ainvoke(initial_state, config)
            logger.info(f"Workflow completed for batch {batch_id}")
            return result
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            raise
    
    async def process_batch(self, alerts: List[AlertData]) -> List[Dict[str, Any]]:
        """Process multiple alerts in parallel"""
        
        batch_id = f"multi_batch_{int(time.time())}"
        logger.info(f"Processing batch of {len(alerts)} alerts: {batch_id}")
        
        # Create tasks for parallel processing
        tasks = []
        for i, alert_data in enumerate(alerts):
            alert_batch_id = f"{batch_id}_{i}"
            task = self.process_alert(alert_data, alert_batch_id)
            tasks.append(task)
        
        # Execute all tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Alert {i} failed: {result}")
                processed_results.append({
                    "status": "failed",
                    "error": str(result),
                    "alert_index": i
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_workflow_state(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Get the current state of a workflow thread"""
        
        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "ada_workflow"
            }
        }
        
        try:
            state = self.workflow.get_state(config)
            return state.values if state else None
        except Exception as e:
            logger.error(f"Error getting workflow state: {e}")
            return None
    
    def update_model(self, model_data: bytes, model_metadata: Dict[str, Any]) -> bool:
        """Update the ML model used for anomaly detection"""
        
        try:
            # Load new model
            new_model = pickle.loads(model_data)
            
            # Validate model (basic check)
            if not hasattr(new_model, 'decision_function'):
                raise ValueError("Invalid model: missing decision_function method")
            
            # Update the model
            old_model = self.model
            self.model = new_model
            
            # Save to GCS
            bucket = self.storage_client.bucket(self.config.model_bucket)
            blob = bucket.blob(self.config.model_path)
            blob.upload_from_string(model_data)
            
            # Save metadata
            metadata_blob = bucket.blob(f"{self.config.model_path}.metadata.json")
            metadata_blob.upload_from_string(json.dumps(model_metadata, indent=2))
            
            logger.info("Model updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error updating model: {e}")
            # Restore old model on failure
            if 'old_model' in locals():
                self.model = old_model
            return False

# Helper functions for deployment
async def create_ada_config_from_env() -> ADAConfig:
    """Create ADA configuration from environment variables"""
    
    return ADAConfig(
        project_id=os.environ.get("PROJECT_ID", "ai-driven-soc"),
        location=os.environ.get("VERTEX_AI_LOCATION", "us-central1"),
        postgres_connection_string=os.environ.get("POSTGRES_CONNECTION_STRING", ""),
        redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379"),
        model_bucket=os.environ.get("MODEL_BUCKET", "security-models"),
        confidence_threshold=float(os.environ.get("CONFIDENCE_THRESHOLD", "0.8")),
        batch_size=int(os.environ.get("BATCH_SIZE", "100")),
        max_retries=int(os.environ.get("MAX_RETRIES", "3"))
    )

# Cloud Function entry points
async def process_alert_cloud_function(request):
    """Cloud Function entry point for processing single alerts"""
    
    try:
        request_json = request.get_json()
        
        if not request_json or "alert_data" not in request_json:
            return {"error": "Missing alert_data in request"}, 400
        
        # Create ADA instance
        config = await create_ada_config_from_env()
        ada = LangGraphAnomalyDetectionAgent(config)
        
        # Process the alert
        result = await ada.process_alert(
            alert_data=request_json["alert_data"],
            batch_id=request_json.get("batch_id")
        )
        
        return {"status": "success", "result": result}
        
    except Exception as e:
        logger.error(f"Cloud Function error: {e}")
        return {"error": str(e)}, 500

def process_pubsub_alert_cloud_function(event, context):
    """Cloud Function entry point for Pub/Sub triggered processing"""
    
    import base64
    
    try:
        # Decode Pub/Sub message
        pubsub_message = base64.b64decode(event['data']).decode('utf-8')
        alert_data = json.loads(pubsub_message)
        
        # Process asynchronously
        asyncio.run(process_single_pubsub_alert(alert_data))
        
        return "Alert processed successfully"
        
    except Exception as e:
        logger.error(f"Pub/Sub Cloud Function error: {e}")
        raise

async def process_single_pubsub_alert(alert_data: Dict[str, Any]):
    """Process a single alert from Pub/Sub"""
    
    config = await create_ada_config_from_env()
    ada = LangGraphAnomalyDetectionAgent(config)
    
    result = await ada.process_alert(alert_data)
    logger.info(f"Processed Pub/Sub alert: {result.get('workflow_status')}")

if __name__ == "__main__":
    import random
    import time
    import argparse

    def generate_random_alert(idx):
        return {
            "log_id": f"test-log-{idx}",
            "timestamp": "2023-05-03T12:34:56Z",
            "source_ip": f"192.168.1.{random.randint(1,254)}",
            "dest_ip": f"10.1.1.{random.randint(1,254)}",
            "protocol": random.choice(["tcp", "udp"]),
            "port": random.choice([80, 443, 22, 8080]),
            "bytes_sent": random.randint(100, 10000),
            "bytes_received": random.randint(100, 10000),
            "duration": random.uniform(0.1, 10.0),
            "raw_log": {"test": "data"}
        }

    async def test_ada():
        config = ADAConfig()
        ada = LangGraphAnomalyDetectionAgent(config)
        # Test alert
        test_alert = {
            "log_id": "test-log-123",
            "timestamp": "2023-05-03T12:34:56Z",
            "source_ip": "192.168.1.100",
            "dest_ip": "10.1.1.50",
            "protocol": "tcp",
            "port": 443,
            "bytes_sent": 1500,
            "bytes_received": 8500,
            "duration": 2.5,
            "raw_log": {"test": "data"}
        }
        result = await ada.process_alert(test_alert)
        print(json.dumps(result, indent=2, default=str))

    async def test_large_batch(batch_size=1000, chunk_size=500):
        # Enable local mode for synthetic batch tests
        config = ADAConfig(batch_size=chunk_size, local_mode=True)
        ada = LangGraphAnomalyDetectionAgent(config)
        alerts = [generate_random_alert(i) for i in range(batch_size)]

        # Fit IsolationForest on real data if in local mode
        if config.local_mode:
            import pandas as pd
            from sklearn.ensemble import IsolationForest
            # Load real training data
            training_path = "../training_data/augmented_with_cicids.csv"
            df = pd.read_csv(training_path)
            # Use all columns except 'is_anomaly' as features
            feature_cols = [
                "Flow Duration",
                "Total Fwd Packets",
                "Total Backward Packets",
                "Total Length of Fwd Packets",
                "Total Length of Bwd Packets",
                "Flow Bytes/s",
                "Flow Packets/s"
            ]
            X = df[feature_cols].values
            ada.model = IsolationForest()
            ada.model.fit(X)

        start = time.time()
        results = []
        status_counts = {}
        total_chunks = (batch_size + chunk_size - 1) // chunk_size
        for chunk_idx in range(total_chunks):
            chunk_alerts = alerts[chunk_idx*chunk_size : (chunk_idx+1)*chunk_size]
            chunk_results = []
            for alert in chunk_alerts:
                alert_with_mode = dict(alert)
                alert_with_mode["processing_mode"] = "batch"
                result = await ada.process_alert(alert_with_mode)
                chunk_results.append(result)
                results.append(result)
            # Intermediate stats for this chunk
            chunk_status_counts = {}
            for r in chunk_results:
                status = r.get("workflow_status", "unknown")
                chunk_status_counts[status] = chunk_status_counts.get(status, 0) + 1
                status_counts[status] = status_counts.get(status, 0) + 1
            print(f"Chunk {chunk_idx+1}/{total_chunks}: Processed {len(chunk_alerts)} alerts. Status breakdown: {chunk_status_counts}")
        end = time.time()
        print(f"\nProcessed {batch_size} alerts in {end - start:.2f} seconds.")
        print(f"Final status breakdown: {status_counts}")

    parser = argparse.ArgumentParser(description="ADA Test Harness")
    parser.add_argument('--large', action='store_true', help='Run large-scale synthetic batch test')
    parser.add_argument('--batch_size', type=int, default=1000, help='Number of alerts for large-scale test')
    parser.add_argument('--chunk_size', type=int, default=500, help='Number of alerts to process per chunk')
    args = parser.parse_args()

    if args.large:
        asyncio.run(test_large_batch(batch_size=args.batch_size, chunk_size=args.chunk_size))
    else:
        asyncio.run(test_ada())

