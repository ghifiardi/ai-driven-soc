#!/usr/bin/env python3
"""
Enhanced ADA-TAA Integration with Google Flash 2.5
Integrates the enhanced TAA with Flash 2.5 into the existing LangGraph ADA workflow
"""

import os
import json
import logging
import asyncio
import time
from datetime import datetime
import sys
from typing import Dict, List, Any, Optional
from typing_extensions import Literal, TypedDict, NotRequired
from dataclasses import dataclass
import pandas as pd
import numpy as np

# LangGraph imports
from langgraph.graph import StateGraph, END

# Google Cloud imports
from google.cloud import storage, pubsub_v1, aiplatform, firestore, bigquery

# ML and data processing
from sklearn.ensemble import IsolationForest, RandomForestClassifier
import pickle
import redis
from bigquery_client import BigQueryClient

# Enhanced TAA imports
from enhanced_taa_flash25 import EnhancedTAA

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/var/log/enhanced-ada-taa/ada_taa_workflow.log", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EnhancedADATAA")

# State definitions for the enhanced ADA workflow
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

class TAAAnalysisResult(TypedDict):
    llm_model: str
    is_true_positive: bool
    severity: str
    confidence: float
    threat_indicators: List[str]
    attack_patterns: List[str]
    recommended_actions: List[str]
    reasoning: str
    risk_score: float
    urgency: str
    threat_category: str
    attack_vector: str
    processing_time_ms: float
    tokens_used: int

class EnhancedADAState(TypedDict):
    alert_data: AlertData
    batch_id: str
    detection_result: Optional[DetectionResult]
    taa_analysis_result: Optional[TAAAnalysisResult]
    workflow_status: Literal["started", "processing", "enriching", "validating", "taa_analysis", "completed", "failed"]
    error_messages: List[str]
    final_alert: Optional[Dict[str, Any]]
    should_publish: bool
    taa_decision: Optional[str]  # 'containment', 'manual_review', 'feedback'

@dataclass
class EnhancedADAConfig:
    # Load configuration from environment variables
    project_id: str = os.getenv("BIGQUERY_PROJECT_ID", "chronicle-dev-2be9")
    location: str = os.getenv("GCP_LOCATION", "us-central1")
    redis_url: str = os.getenv("REDIS_URL", "")
    
    # Model configuration
    model_bucket: str = os.getenv("GCS_BUCKET_NAME", "security-models-chronicle-dev-2be9")
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
        if not all([self.BIGQUERY_PROJECT_ID, self.BIGQUERY_DATASET_ID, self.BIGQUERY_TABLE_ID, self.BIGQUERY_RESULTS_TABLE_ID]):
            raise ValueError("Missing one or more required BigQuery environment variables.")

        if self.features is None:
            self.features = [
                'bytes_sent', 'bytes_received', 'duration',
                'port', 'protocol_encoded', 'hour_of_day', 'day_of_week'
            ]

class EnhancedADATAAAgent:
    def __init__(self, config: EnhancedADAConfig):
        logger.info("Initializing Enhanced ADA-TAA Agent with Flash 2.5...")
        self.config = config
        
        # Initialize Google Cloud clients
        self.storage_client = storage.Client(project=config.project_id)
        self.publisher = pubsub_v1.PublisherClient()
        self.redis_client = redis.from_url(config.redis_url) if config.redis_url else None
        self.bq_client = bigquery.Client(project=config.project_id)
        
        # Initialize BigQuery client
        try:
            self.bq_client_custom = BigQueryClient(
                config.BIGQUERY_PROJECT_ID,
                config.BIGQUERY_DATASET_ID,
                config.BIGQUERY_TABLE_ID,
                config.BIGQUERY_RESULTS_TABLE_ID
            )
            logger.info("BigQuery client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize BigQuery client: {e}")
            raise
        
        # Initialize AI Platform
        aiplatform.init(project=config.project_id, location=config.location)
        
        # Load ML models
        self.model = self._load_model(self.config.model_path)
        self.supervised_model = self._load_model(self.config.supervised_model_path, model_type='joblib')
        
        # Initialize Enhanced TAA with Flash 2.5
        taa_config = {
            "project_id": config.project_id,
            "location": config.location
        }
        self.enhanced_taa = EnhancedTAA(taa_config)
        logger.info("Enhanced TAA with Flash 2.5 initialized successfully")
        
        # Build workflow
        self.workflow = self._build_workflow()
        logger.info("Enhanced ADA-TAA workflow built successfully")

    def _load_model(self, model_path, model_type='pickle'):
        """Load ML model with fallback creation"""
        local_model_dir = 'models'
        base_name = os.path.basename(model_path)
        local_path = os.path.join(local_model_dir, os.path.splitext(base_name)[0] + '.pkl')
        
        os.makedirs(local_model_dir, exist_ok=True)

        if os.path.exists(local_path):
            logger.info(f"Loading model from: {local_path}")
            try:
                with open(local_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Could not load model from {local_path}: {e}")
        
        # Create fallback model
        logger.info(f"Creating fallback model at: {local_path}")
        X, y = self._generate_synthetic_data()
        
        if 'isolation_forest' in model_path:
            model = IsolationForest(contamination='auto', random_state=42)
            model.fit(X)
        elif 'supervised_model' in model_path:
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X, y)
        else:
            logger.error(f"Unknown model type for path: {model_path}")
            return None

        try:
            with open(local_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Successfully saved model to {local_path}")
        except Exception as e:
            logger.error(f"Failed to save model to {local_path}: {e}")

        return model

    def _generate_synthetic_data(self):
        """Generate synthetic data for training fallback models"""
        np.random.seed(42)
        X = np.random.rand(1000, len(self.config.features))
        y = np.random.choice([0, 1], size=1000)
        return X, y

    def _build_workflow(self) -> StateGraph:
        """Build enhanced ADA-TAA workflow"""
        workflow = StateGraph(EnhancedADAState)
        
        # Add nodes
        workflow.add_node("preprocess_data", self.preprocess_data_node)
        workflow.add_node("extract_features", self.extract_features_node)
        workflow.add_node("detect_anomalies", self.detect_anomalies_node)
        workflow.add_node("taa_analysis", self.taa_analysis_node)  # New Flash 2.5 analysis
        workflow.add_node("taa_decision", self.taa_decision_node)
        workflow.add_node("containment_action", self.containment_action_node)
        workflow.add_node("manual_review", self.manual_review_node)
        workflow.add_node("feedback_processing", self.feedback_processing_node)
        workflow.add_node("generate_alert", self.generate_alert_node)
        workflow.add_node("publish_alert", self.publish_alert_node)
        workflow.add_node("store_results", self.store_results_node)

        # Set entry point
        workflow.set_entry_point("preprocess_data")

        # Define workflow edges
        workflow.add_edge("preprocess_data", "extract_features")
        workflow.add_edge("extract_features", "detect_anomalies")
        
        # Route based on anomaly detection
        workflow.add_conditional_edges(
            "detect_anomalies",
            self.route_after_detection,
            {
                "taa_analysis": "taa_analysis",
                "store_results": "store_results"
            }
        )
        
        # TAA analysis flow
        workflow.add_edge("taa_analysis", "taa_decision")
        workflow.add_conditional_edges(
            "taa_decision",
            self.route_after_taa_decision,
            {
                "containment_action": "containment_action",
                "manual_review": "manual_review",
                "feedback_processing": "feedback_processing",
                "generate_alert": "generate_alert"
            }
        )
        
        # Final processing
        workflow.add_edge("containment_action", "generate_alert")
        workflow.add_edge("manual_review", "generate_alert")
        workflow.add_edge("feedback_processing", "generate_alert")
        workflow.add_edge("generate_alert", "publish_alert")
        workflow.add_edge("publish_alert", "store_results")
        workflow.add_edge("store_results", END)

        return workflow.compile()

    def preprocess_data_node(self, state: EnhancedADAState) -> Dict[str, Any]:
        """Preprocess incoming alert data"""
        logger.info("Preprocessing alert data")
        try:
            if "alert_data" not in state:
                messages = state.get("messages", [])
                for role, content in reversed(messages):
                    if role == "user":
                        state["alert_data"] = content
                        break
                
                if "alert_data" not in state:
                    logger.error("No alert data found in state")
                    state["error_messages"] = state.get("error_messages", []) + ["No alert data found"]
            
            return state
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            state["error_messages"] = state.get("error_messages", []) + [str(e)]
            return state

    def extract_features_node(self, state: EnhancedADAState) -> Dict[str, Any]:
        """Extract features from alert data with TI enrichment"""
        logger.info(f"Extracting features for alert {state['alert_data'].get('log_id')}")
        alert_data = state["alert_data"]

        # Parse timestamp
        timestamp_str = alert_data.get("timestamp")
        try:
            timestamp = pd.to_datetime(timestamp_str)
        except (ValueError, TypeError):
            logger.warning(f"Could not parse timestamp '{timestamp_str}'. Using current time.")
            timestamp = pd.Timestamp.now()

        # Safe conversion functions
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

        # Extract basic features
        features = {
            "duration": safe_float(alert_data.get("duration")),
            "bytes_sent": safe_float(alert_data.get("bytes_sent")),
            "bytes_received": safe_float(alert_data.get("bytes_received")),
            "port": safe_int(alert_data.get("port")),
            "protocol_encoded": {"tcp": 1, "udp": 2, "icmp": 3}.get(str(alert_data.get("protocol", "")).lower(), 0),
            "hour_of_day": timestamp.hour,
            "day_of_week": timestamp.dayofweek,
        }

        # Add Threat Intelligence features (optional)
        ti_features = self._extract_ti_features(alert_data)
        features.update(ti_features)
        alert_data["ti_features"] = ti_features

        logger.info(f"Extracted features: {features}")
        alert_data["extracted_features"] = features
        return {"alert_data": alert_data}

    def _extract_ti_features(self, alert_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract TI-based features from IPs"""
        ti_features = {
            "src_ip_reputation": 0.0,
            "dst_ip_reputation": 0.0,
            "src_ip_reports": 0.0,
            "dst_ip_reports": 0.0,
            "src_ip_whitelisted": 0.0,
            "dst_ip_whitelisted": 0.0,
        }

        try:
            # Import TI client (optional)
            try:
                import ti_abuseipdb
            except ImportError:
                logger.debug("TI client not available, skipping TI features")
                return ti_features

            # Extract IPs
            src_ip = alert_data.get("source_ip") or alert_data.get("src_ip")
            dst_ip = alert_data.get("dest_ip") or alert_data.get("dst_ip")

            # Lookup source IP
            if src_ip and isinstance(src_ip, str):
                try:
                    ti_data = ti_abuseipdb.query_abuseipdb(src_ip)
                    if isinstance(ti_data, dict):
                        ti_features["src_ip_reputation"] = ti_data.get("abuseConfidenceScore", 0) / 100.0
                        ti_features["src_ip_reports"] = min(ti_data.get("totalReports", 0) / 1000.0, 1.0)  # Normalize
                        ti_features["src_ip_whitelisted"] = 1.0 if ti_data.get("isWhitelisted", False) else 0.0
                except Exception as e:
                    logger.debug(f"TI lookup failed for {src_ip}: {e}")

            # Lookup destination IP
            if dst_ip and isinstance(dst_ip, str):
                try:
                    ti_data = ti_abuseipdb.query_abuseipdb(dst_ip)
                    if isinstance(ti_data, dict):
                        ti_features["dst_ip_reputation"] = ti_data.get("abuseConfidenceScore", 0) / 100.0
                        ti_features["dst_ip_reports"] = min(ti_data.get("totalReports", 0) / 1000.0, 1.0)
                        ti_features["dst_ip_whitelisted"] = 1.0 if ti_data.get("isWhitelisted", False) else 0.0
                except Exception as e:
                    logger.debug(f"TI lookup failed for {dst_ip}: {e}")

        except Exception as e:
            logger.warning(f"Error extracting TI features: {e}")

        logger.debug(f"TI features: {ti_features}")
        return ti_features

    def detect_anomalies_node(self, state: EnhancedADAState) -> Dict[str, Any]:
        """Detect anomalies using ML models"""
        logger.info("Detecting anomalies")
        features = state["alert_data"]["extracted_features"]
        feature_vector = np.array([features.get(f, 0.0) for f in self.config.features]).reshape(1, -1)
        
        # Use supervised model for prediction
        anomaly_prob = self.supervised_model.predict_proba(feature_vector)[0, 1]
        confidence = anomaly_prob
        is_anomaly = confidence > self.config.confidence_threshold

        logger.info(f"Anomaly detection: Confidence={confidence:.4f}, Is_Anomaly={is_anomaly}")

        result = DetectionResult(
            anomaly_score=float(confidence),
            confidence=float(confidence),
            is_anomaly=bool(is_anomaly),
            model_version="enhanced_supervised_v1",
            detection_timestamp=datetime.now().isoformat(),
            feature_importance={}
        )
        return {"detection_result": result}

    def taa_analysis_node(self, state: EnhancedADAState) -> Dict[str, Any]:
        """Enhanced TAA analysis using Google Flash 2.5"""
        logger.info("Performing TAA analysis with Flash 2.5")
        
        try:
            # Prepare alert data for TAA analysis
            alert_data = state["alert_data"]
            detection_result = state["detection_result"]
            
            # Create comprehensive alert data for TAA
            taa_alert_data = {
                "alert_id": alert_data.get("log_id"),
                "src_ip": alert_data.get("source_ip"),
                "dst_ip": alert_data.get("dest_ip"),
                "port": alert_data.get("port"),
                "protocol": alert_data.get("protocol"),
                "bytes_transferred": alert_data.get("bytes_sent", 0) + alert_data.get("bytes_received", 0),
                "connection_count": 1,  # Single connection for now
                "timestamp": alert_data.get("timestamp"),
                "anomaly_score": detection_result.get("anomaly_score", 0.0),
                "detection_confidence": detection_result.get("confidence", 0.0),
                "raw_log": alert_data.get("raw_log", {})
            }
            
            # Perform Flash 2.5 analysis
            analysis_result = self.enhanced_taa.analyze_alert(taa_alert_data)
            
            # Store TAA analysis result
            taa_result = TAAAnalysisResult(
                llm_model=analysis_result.get("llm_model", "unknown"),
                is_true_positive=analysis_result.get("is_true_positive", False),
                severity=analysis_result.get("severity", "low"),
                confidence=analysis_result.get("confidence", 0.0),
                threat_indicators=analysis_result.get("threat_indicators", []),
                attack_patterns=analysis_result.get("attack_patterns", []),
                recommended_actions=analysis_result.get("recommended_actions", []),
                reasoning=analysis_result.get("reasoning", ""),
                risk_score=analysis_result.get("risk_score", 0.0),
                urgency=analysis_result.get("urgency", "low"),
                threat_category=analysis_result.get("threat_category", "other"),
                attack_vector=analysis_result.get("attack_vector", "other"),
                processing_time_ms=analysis_result.get("processing_time_ms", 0.0),
                tokens_used=analysis_result.get("tokens_used", 0)
            )
            
            logger.info(f"TAA analysis completed: {taa_result['severity']} severity, {taa_result['confidence']:.2f} confidence")
            
            return {"taa_analysis_result": taa_result}
            
        except Exception as e:
            logger.error(f"Error in TAA analysis: {e}")
            state["error_messages"] = state.get("error_messages", []) + [f"TAA analysis error: {str(e)}"]
            return state

    def taa_decision_node(self, state: EnhancedADAState) -> Dict[str, Any]:
        """Make decision based on TAA analysis results"""
        logger.info("Making TAA decision")
        
        taa_result = state.get("taa_analysis_result", {})
        
        # Decision logic based on TAA analysis
        if taa_result.get("is_true_positive") and taa_result.get("severity") == "high":
            decision = "containment_action"
            logger.info("Routing to containment action")
        elif taa_result.get("confidence", 1.0) < 0.7:
            decision = "manual_review"
            logger.info("Routing to manual review")
        else:
            decision = "feedback_processing"
            logger.info("Routing to feedback processing")
        
        state["taa_decision"] = decision
        return state

    def containment_action_node(self, state: EnhancedADAState) -> Dict[str, Any]:
        """Handle containment actions"""
        logger.info("Processing containment action")
        
        try:
            # Publish containment request to CRA
            pub = self.publisher
            project_id = self.config.project_id
            topic_name = "containment-requests"
            topic_path = pub.topic_path(project_id, topic_name)
            
            alert_data = state["alert_data"]
            taa_result = state["taa_analysis_result"]
            
            containment_data = {
                "alert_id": alert_data.get("log_id"),
                "alert_data": alert_data,
                "taa_analysis": taa_result,
                "timestamp": datetime.now().isoformat(),
                "containment_request": {
                    "action": "immediate_containment",
                    "severity": taa_result.get("severity", "high"),
                    "confidence": taa_result.get("confidence", 0.95),
                    "reasoning": f"High severity true positive: {taa_result.get('reasoning', 'Automated analysis')}"
                }
            }
            
            message_data = json.dumps(containment_data).encode("utf-8")
            future = pub.publish(topic_path, message_data)
            message_id = future.result()
            
            logger.info(f"Published containment request. Message ID: {message_id}")
            state["containment_message_id"] = message_id
            
        except Exception as e:
            logger.error(f"Failed to publish containment request: {e}")
            state["error_messages"] = state.get("error_messages", []) + [f"Containment error: {str(e)}"]
        
        return state

    def manual_review_node(self, state: EnhancedADAState) -> Dict[str, Any]:
        """Flag for manual review"""
        logger.info("Flagging for manual review")
        state["manual_review_flagged"] = True
        return state

    def feedback_processing_node(self, state: EnhancedADAState) -> Dict[str, Any]:
        """Process feedback for CLA learning"""
        logger.info("Processing feedback for CLA")
        
        try:
            # Publish feedback to CLA
            pub = self.publisher
            project_id = self.config.project_id
            topic_name = "taa-feedback"
            topic_path = pub.topic_path(project_id, topic_name)
            
            alert_data = state["alert_data"]
            taa_result = state["taa_analysis_result"]
            
            feedback_data = {
                "alert_id": alert_data.get("log_id"),
                "is_true_positive": taa_result.get("is_true_positive", True),
                "confidence": taa_result.get("confidence", 0.95),
                "severity": taa_result.get("severity", "medium"),
                "timestamp": datetime.now().isoformat(),
                "reasoning": taa_result.get("reasoning", "Automated analysis"),
                "source": "enhanced_ada_taa_flash25"
            }
            
            message_data = json.dumps(feedback_data).encode("utf-8")
            future = pub.publish(topic_path, message_data)
            message_id = future.result()
            
            logger.info(f"Published feedback. Message ID: {message_id}")
            state["feedback_message_id"] = message_id
            
        except Exception as e:
            logger.error(f"Failed to publish feedback: {e}")
            state["error_messages"] = state.get("error_messages", []) + [f"Feedback error: {str(e)}"]
        
        return state

    def generate_alert_node(self, state: EnhancedADAState) -> Dict[str, Any]:
        """Generate final alert"""
        logger.info("Generating final alert")
        
        alert_data = state["alert_data"]
        detection_result = state["detection_result"]
        taa_result = state.get("taa_analysis_result")
        
        final_alert = {
            "alert_id": f"enhanced-ada-{alert_data.get('log_id')}",
            "timestamp": datetime.now().isoformat(),
            "alert_data": alert_data,
            "detection_result": detection_result,
            "taa_analysis": taa_result,
            "taa_decision": state.get("taa_decision"),
            "containment_requested": state.get("containment_message_id") is not None,
            "manual_review_flagged": state.get("manual_review_flagged", False),
            "feedback_published": state.get("feedback_message_id") is not None,
            "processing_metadata": {
                "workflow_version": "enhanced_ada_taa_v1",
                "flash25_model": taa_result.get("llm_model") if taa_result else None,
                "processing_time_ms": taa_result.get("processing_time_ms") if taa_result else None
            }
        }
        
        return {"final_alert": final_alert}

    def publish_alert_node(self, state: EnhancedADAState) -> Dict[str, Any]:
        """Publish alert to BigQuery"""
        logger.info("Publishing alert to BigQuery")
        
        final_alert = state.get("final_alert")
        if not final_alert:
            logger.warning("No final alert to publish")
            return {}
        
        try:
            # Prepare BigQuery row
            result_row = {
                "alarm_id": final_alert.get("alert_id"),
                "id": None,
                "score": final_alert.get("detection_result", {}).get("anomaly_score"),
                "confidence": final_alert.get("detection_result", {}).get("confidence"),
                "detection_timestamp": final_alert.get("detection_result", {}).get("detection_timestamp"),
                "valid": True,  # Enhanced ADA-TAA always publishes valid alerts
                "case_class": final_alert.get("taa_analysis", {}).get("threat_category"),
                "variable_of_importance": json.dumps(final_alert.get("taa_analysis", {}).get("threat_indicators", [])),
                "remarks": f"Enhanced ADA-TAA Flash 2.5: {final_alert.get('taa_decision', 'unknown')}",
                "run_time": final_alert.get("processing_metadata", {}).get("processing_time_ms")
            }
            
            self.bq_client_custom.insert_rows_json([result_row])
            logger.info(f"Successfully published alert {result_row['alarm_id']} to BigQuery")
            
        except Exception as e:
            logger.error(f"Failed to publish alert to BigQuery: {e}")
            state["error_messages"] = state.get("error_messages", []) + [f"Publish error: {str(e)}"]
        
        return {}

    def store_results_node(self, state: EnhancedADAState) -> Dict[str, Any]:
        """Store processing results and metrics"""
        logger.info("Storing processing results")
        
        try:
            # Store metrics for monitoring
            metrics_data = {
                "timestamp": datetime.now().isoformat(),
                "alert_id": state["alert_data"].get("log_id"),
                "processing_time_ms": state.get("taa_analysis_result", {}).get("processing_time_ms", 0),
                "taa_model": state.get("taa_analysis_result", {}).get("llm_model", "unknown"),
                "severity": state.get("taa_analysis_result", {}).get("severity", "unknown"),
                "confidence": state.get("taa_analysis_result", {}).get("confidence", 0),
                "decision": state.get("taa_decision", "unknown"),
                "success": len(state.get("error_messages", [])) == 0
            }
            
            # Store in Redis if available
            if self.redis_client:
                self.redis_client.hset(
                    "enhanced_ada_taa_metrics",
                    state["alert_data"].get("log_id"),
                    json.dumps(metrics_data)
                )
            
            logger.info(f"Stored metrics for alert {state['alert_data'].get('log_id')}")
            
        except Exception as e:
            logger.error(f"Failed to store results: {e}")
        
        return {}

    def route_after_detection(self, state: EnhancedADAState) -> str:
        """Route after anomaly detection"""
        detection_result = state.get("detection_result")
        if detection_result and detection_result.get("is_anomaly"):
            return "taa_analysis"
        else:
            return "store_results"

    def route_after_taa_decision(self, state: EnhancedADAState) -> str:
        """Route after TAA decision"""
        return state.get("taa_decision", "generate_alert")

    async def process_single_alert(self, raw_alert: dict) -> Dict[str, Any]:
        """Process a single alert through the enhanced workflow"""
        logger.info(f"Processing alert: {raw_alert.get('alarmId')}")
        
        try:
            # Create alert data
            alert_data = self._create_alert_data_from_raw(raw_alert)
            
            # Initialize state
            initial_state = EnhancedADAState(
                alert_data=alert_data,
                batch_id=alert_data.get('log_id', 'unknown'),
                error_messages=[],
                workflow_status='started'
            )
            
            # Process through workflow
            final_state = await self.workflow.ainvoke(initial_state)
            
            return final_state
            
        except Exception as e:
            logger.error(f"Error processing alert {raw_alert.get('alarmId')}: {e}")
            return {"error_messages": [str(e)]}

    def _create_alert_data_from_raw(self, raw_alert: dict) -> AlertData:
        """Transform raw alert to structured format and normalize IP keys.
        Ensures `src_ip` and `dst_ip` are present in raw_log when possible
        so TI lookups work consistently in the dashboard.
        """
        events_str = raw_alert.get("events")
        if not events_str or not isinstance(events_str, str):
            logger.warning(f"Alert {raw_alert.get('alarmId')} has missing events data")
            events_data = {}
        else:
            try:
                events_data = json.loads(events_str)
            except json.JSONDecodeError:
                logger.warning(f"Alert {raw_alert.get('alarmId')} has malformed JSON")
                events_data = {}

        # Normalize IP field names
        def first_nonempty(d: dict, keys):
            for k in keys:
                v = d.get(k)
                if isinstance(v, str) and v:
                    return v
            return None

        src_ip = first_nonempty(events_data, [
            "src_ip", "source_ip", "client_ip", "ip_src", "source", "src"
        ])
        dst_ip = first_nonempty(events_data, [
            "dst_ip", "dest_ip", "destination_ip", "server_ip", "ip_dst", "destination", "dst"
        ])

        # Write back normalized keys to raw payload for downstream consumers
        if src_ip and events_data.get("src_ip") != src_ip:
            events_data["src_ip"] = src_ip
        if dst_ip and events_data.get("dst_ip") != dst_ip:
            events_data["dst_ip"] = dst_ip

        return AlertData(
            log_id=raw_alert.get("alarmId"),
            timestamp=events_data.get("timestamp", datetime.now().isoformat()),
            source_ip=src_ip,
            dest_ip=dst_ip,
            protocol=events_data.get("protocol"),
            port=events_data.get("port"),
            bytes_sent=events_data.get("bytes_sent"),
            bytes_received=events_data.get("bytes_received"),
            duration=events_data.get("duration"),
            raw_log=events_data
        )

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from Enhanced TAA"""
        return self.enhanced_taa.get_performance_metrics()

if __name__ == "__main__":
    try:
        config = EnhancedADAConfig()
        agent = EnhancedADATAAAgent(config)
        bq_client = agent.bq_client_custom

        async def main_loop():
            logger.info("Starting Enhanced ADA-TAA service with Flash 2.5...")
            while True:
                try:
                    new_alerts = bq_client.fetch_new_alerts(limit=50)
                    if not new_alerts:
                        logger.info(f"No new alerts found. Waiting for {config.polling_interval}s...")
                        await asyncio.sleep(config.polling_interval)
                        continue

                    logger.info(f"Fetched {len(new_alerts)} new alerts. Processing with Enhanced ADA-TAA...")
                    tasks = [agent.process_single_alert(alert) for alert in new_alerts]
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    processed_ids = []
                    error_ids = []
                    
                    for i, result in enumerate(results):
                        alarm_id = new_alerts[i].get("alarmId")
                        if isinstance(result, Exception):
                            logger.error(f"Exception for alert {alarm_id}: {result}")
                            error_ids.append(alarm_id)
                        elif result and not result.get("error_messages"):
                            logger.info(f"Successfully processed alert {alarm_id}")
                            processed_ids.append(alarm_id)
                        else:
                            error_msg = result.get('error_messages', ['Unknown error'])
                            logger.error(f"Error processing alert {alarm_id}: {error_msg}")
                            error_ids.append(alarm_id)

                    # Update alert status
                    if processed_ids:
                        bq_client.update_alert_status(alarm_ids=processed_ids, new_status="PROCESSED")
                    if error_ids:
                        bq_client.update_alert_status(alarm_ids=error_ids, new_status="ERROR")

                    # Log performance metrics
                    metrics = agent.get_performance_metrics()
                    logger.info(f"Performance metrics: {metrics}")

                except Exception as e:
                    logger.critical(f"Critical error in main loop: {e}")
                    await asyncio.sleep(60)

        asyncio.run(main_loop())

    except ValueError as e:
        logger.critical(f"Configuration Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Enhanced ADA-TAA service stopped by user.")
