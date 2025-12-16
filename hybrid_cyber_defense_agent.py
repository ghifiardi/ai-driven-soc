#!/usr/bin/env python3
"""
Hybrid Cyber Defense Agent
==========================

Integrates DQN (Deep Q-Network) for fast decision making with Gemini explanations
for explainable AI in cybersecurity defense. Follows A2A protocol for multi-agent
communication within the ADA/TAA/CRA/CLA framework.

Features:
- Real-time Pub/Sub integration with ADA alerts
- DQN-based action recommendations (100ms response)
- Gemini-generated explanations for decisions
- A2A protocol compliance for agent communication
- Circuit breaker patterns for resilience
- Comprehensive monitoring and metrics
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import traceback

# Google Cloud imports
from google.cloud import pubsub_v1
import vertexai
from vertexai.generative_models import GenerativeModel

# FastAPI for A2A protocol
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Local imports
from dqn_inference import ProductionDQNAgent, RuleBasedFallback
from gemini_rest_integration import GeminiRESTClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class AlertProcessingResult:
    """Result of processing a security alert"""
    alert_id: str
    recommended_action: str
    action_id: int
    confidence: float
    explanation: str
    processing_time: float
    dqn_context: dict
    gemini_used: bool
    timestamp: str

@dataclass
class A2ATask:
    """A2A task structure"""
    task_id: str
    skill: str
    parameters: Dict[str, Any]
    callback_url: Optional[str] = None
    priority: str = "normal"
    timestamp: str = None

@dataclass
class A2AResponse:
    """A2A response structure"""
    task_id: str
    status: str
    result: Dict[str, Any]
    agent_id: str
    timestamp: str
    metadata: Dict[str, Any] = None

class CircuitBreaker:
    """Circuit breaker implementation for resilience"""
    
    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            raise e

class HybridCyberDefenseAgent:
    """Main hybrid agent class integrating DQN + Gemini"""
    
    def __init__(self, config_path: str = "config/hybrid_defense_config.json"):
        """Initialize the hybrid cyber defense agent"""
        self.config = self._load_config(config_path)
        self.agent_id = f"hybrid-defense-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Initialize components
        self._init_logging()
        self._init_google_cloud()
        self._init_dqn_agent()
        self._init_gemini_client()
        self._init_pubsub()
        self._init_circuit_breakers()
        self._init_metrics()
        self._init_a2a_server()
        
        # State tracking
        self.is_running = False
        self.stats = {
            'alerts_processed': 0,
            'actions_recommended': 0,
            'explanations_generated': 0,
            'errors': 0,
            'start_time': datetime.now()
        }
        
        # Thread pool for concurrent processing
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        logger.info(f"Hybrid Cyber Defense Agent initialized: {self.agent_id}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load agent configuration"""
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}")
            raise
    
    def _init_logging(self):
        """Initialize structured logging"""
        log_level = self.config.get('monitoring', {}).get('log_level', 'INFO')
        logger.setLevel(getattr(logging, log_level))
    
    def _init_google_cloud(self):
        """Initialize Google Cloud services"""
        try:
            project_id = self.config['google_cloud']['project_id']
            location = self.config['google_cloud']['location']
            
            # Initialize Vertex AI
            vertexai.init(project=project_id, location=location)
            
            # Set credentials if provided
            creds_path = self.config['google_cloud'].get('credentials_path')
            if creds_path and os.path.exists(creds_path):
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = creds_path
            
            logger.info(f"Google Cloud initialized for project: {project_id}")
        except Exception as e:
            logger.error(f"Failed to initialize Google Cloud: {e}")
            raise
    
    def _init_dqn_agent(self):
        """Initialize DQN inference agent"""
        try:
            if not self.config['feature_flags']['DQN_INFERENCE_ENABLED']:
                logger.info("DQN inference disabled by feature flag")
                self.dqn_agent = None
                return
            
            dqn_config = self.config['dqn_model']
            model_path = dqn_config.get('model_path')
            
            # Create DQN agent with fallback
            self.dqn_agent = ProductionDQNAgent(
                model_path=model_path,
                state_size=dqn_config['state_size'],
                action_size=dqn_config['action_size']
            )
            
            # Initialize fallback if enabled
            if dqn_config['fallback_enabled']:
                self.fallback_agent = RuleBasedFallback()
            else:
                self.fallback_agent = None
            
            logger.info("DQN agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize DQN agent: {e}")
            if self.config['dqn_model']['fallback_enabled']:
                self.dqn_agent = None
                self.fallback_agent = RuleBasedFallback()
                logger.info("Using rule-based fallback only")
            else:
                raise
    
    def _init_gemini_client(self):
        """Initialize Gemini AI client"""
        try:
            if not self.config['feature_flags']['GEMINI_EXPLANATIONS_ENABLED']:
                logger.info("Gemini explanations disabled by feature flag")
                self.gemini_client = None
                return
            
            gemini_config = self.config['gemini']
            
            # Try Vertex AI first, fallback to REST API
            try:
                self.gemini_model = GenerativeModel(gemini_config['model'])
                self.gemini_client = 'vertexai'
                logger.info("Gemini initialized via Vertex AI")
            except Exception as e:
                logger.warning(f"Vertex AI Gemini failed, trying REST API: {e}")
                # Fallback to REST API if available
                api_key = os.getenv('GEMINI_API_KEY')
                if api_key:
                    self.gemini_model = GeminiRESTClient(api_key)
                    self.gemini_client = 'rest'
                    logger.info("Gemini initialized via REST API")
                else:
                    logger.error("No Gemini API key found, explanations disabled")
                    self.gemini_client = None
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            self.gemini_client = None
    
    def _init_pubsub(self):
        """Initialize Pub/Sub publishers and subscribers"""
        try:
            project_id = self.config['google_cloud']['project_id']
            
            # Initialize publisher
            self.publisher = pubsub_v1.PublisherClient()
            
            # Initialize subscriber
            self.subscriber = pubsub_v1.SubscriberClient()
            
            # Set up topic paths
            self.containment_topic_path = self.publisher.topic_path(
                project_id, 
                self.config['pubsub']['publishers']['containment_requests']['topic_name']
            )
            
            self.feedback_topic_path = self.publisher.topic_path(
                project_id,
                self.config['pubsub']['publishers']['taa_feedback']['topic_name']
            )
            
            # Set up subscription path
            self.alerts_subscription_path = self.subscriber.subscription_path(
                project_id,
                self.config['pubsub']['subscriptions']['ada_alerts']['subscription_name']
            )
            
            logger.info("Pub/Sub clients initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Pub/Sub: {e}")
            raise
    
    def _init_circuit_breakers(self):
        """Initialize circuit breakers for resilience"""
        self.circuit_breakers = {}
        
        for name, config in self.config['circuit_breakers'].items():
            if config['enabled']:
                self.circuit_breakers[name] = CircuitBreaker(
                    failure_threshold=config['failure_threshold'],
                    reset_timeout=config['reset_timeout']
                )
        
        logger.info(f"Initialized {len(self.circuit_breakers)} circuit breakers")
    
    def _init_metrics(self):
        """Initialize metrics collection"""
        self.metrics = {
            'alerts_processed_total': 0,
            'dqn_inference_duration': [],
            'gemini_api_calls_total': 0,
            'containment_actions_total': {},
            'errors_total': {}
        }
    
    def _init_a2a_server(self):
        """Initialize A2A FastAPI server"""
        if not self.config['feature_flags']['A2A_PROTOCOL_ENABLED']:
            self.app = None
            return
        
        self.app = FastAPI(
            title="Hybrid Cyber Defense Agent",
            description="DQN + Gemini hybrid agent for cybersecurity",
            version=self.config['agent']['version']
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config['security']['allowed_origins'],
            allow_methods=["*"],
            allow_headers=["*"]
        )
        
        # Add routes
        self._setup_a2a_routes()
    
    def _setup_a2a_routes(self):
        """Setup A2A protocol routes"""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy" if self.is_running else "stopped",
                "agent_id": self.agent_id,
                "timestamp": datetime.now().isoformat(),
                "components": {
                    "dqn_agent": self.dqn_agent is not None,
                    "gemini_client": self.gemini_client is not None,
                    "pubsub": True
                }
            }
        
        @self.app.get("/status")
        async def get_status():
            """Get agent status and metrics"""
            uptime = datetime.now() - self.stats['start_time']
            return {
                "agent_id": self.agent_id,
                "uptime_seconds": uptime.total_seconds(),
                "stats": self.stats,
                "metrics": self.metrics,
                "circuit_breakers": {
                    name: cb.state for name, cb in self.circuit_breakers.items()
                }
            }
        
        @self.app.get("/agent-card")
        async def get_agent_card():
            """Return A2A agent card"""
            return self.config['a2a']['agent_card']
        
        @self.app.post("/a2a/process_alert")
        async def process_alert_a2a(task: dict):
            """A2A endpoint for processing alerts"""
            try:
                result = await self._process_alert_async(task['parameters']['alert'])
                return {
                    "task_id": task.get('task_id'),
                    "status": "completed",
                    "result": asdict(result),
                    "agent_id": self.agent_id,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"A2A alert processing failed: {e}")
                return {
                    "task_id": task.get('task_id'),
                    "status": "failed",
                    "error": str(e),
                    "agent_id": self.agent_id,
                    "timestamp": datetime.now().isoformat()
                }
    
    async def _process_alert_async(self, alert_data: dict) -> AlertProcessingResult:
        """Async wrapper for alert processing"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.process_alert, alert_data)
    
    def process_alert(self, alert_data: dict) -> AlertProcessingResult:
        """
        Core alert processing pipeline
        
        Steps:
        1. DQN inference for action recommendation
        2. Gemini explanation generation
        3. Result packaging
        """
        start_time = time.time()
        alert_id = alert_data.get('id', f"alert_{int(time.time())}")
        
        try:
            logger.info(f"Processing alert {alert_id}")
            
            # Step 1: Get DQN recommendation
            dqn_start = time.time()
            if self.dqn_agent and self.config['feature_flags']['DQN_INFERENCE_ENABLED']:
                try:
                    dqn_result = self.circuit_breakers['dqn_model'].call(
                        self.dqn_agent.get_action_recommendation,
                        alert_data
                    )
                    dqn_used = True
                except Exception as e:
                    logger.warning(f"DQN inference failed, using fallback: {e}")
                    if self.fallback_agent:
                        dqn_result = self.fallback_agent.get_action_recommendation(alert_data)
                        dqn_used = False
                    else:
                        raise
            else:
                if self.fallback_agent:
                    dqn_result = self.fallback_agent.get_action_recommendation(alert_data)
                    dqn_used = False
                else:
                    raise Exception("No decision engine available")
            
            dqn_duration = time.time() - dqn_start
            self.metrics['dqn_inference_duration'].append(dqn_duration)
            
            # Step 2: Generate Gemini explanation
            explanation = self._generate_explanation(alert_data, dqn_result)
            gemini_used = self.gemini_client is not None
            
            # Step 3: Create result
            processing_time = time.time() - start_time
            
            result = AlertProcessingResult(
                alert_id=alert_id,
                recommended_action=dqn_result['recommended_action'],
                action_id=dqn_result['action_id'],
                confidence=dqn_result['confidence'],
                explanation=explanation,
                processing_time=processing_time,
                dqn_context=dqn_result.get('dqn_context', {}),
                gemini_used=gemini_used,
                timestamp=datetime.now().isoformat()
            )
            
            # Update metrics
            self.stats['alerts_processed'] += 1
            self.stats['actions_recommended'] += 1
            self.metrics['alerts_processed_total'] += 1
            
            action_name = result.recommended_action
            self.metrics['containment_actions_total'][action_name] = \
                self.metrics['containment_actions_total'].get(action_name, 0) + 1
            
            logger.info(f"Alert {alert_id} processed: {action_name} (confidence: {result.confidence:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process alert {alert_id}: {e}")
            self.stats['errors'] += 1
            error_type = type(e).__name__
            self.metrics['errors_total'][error_type] = \
                self.metrics['errors_total'].get(error_type, 0) + 1
            
            # Return safe fallback result
            return AlertProcessingResult(
                alert_id=alert_id,
                recommended_action="no_action",
                action_id=4,
                confidence=0.1,
                explanation=f"Error during processing: {str(e)}",
                processing_time=time.time() - start_time,
                dqn_context={},
                gemini_used=False,
                timestamp=datetime.now().isoformat()
            )
    
    def _generate_explanation(self, alert_data: dict, dqn_result: dict) -> str:
        """Generate human-readable explanation using Gemini"""
        if not self.gemini_client or not self.config['feature_flags']['GEMINI_EXPLANATIONS_ENABLED']:
            return self._generate_simple_explanation(dqn_result)
        
        try:
            # Prepare context for Gemini
            alert_summary = self._summarize_alert(alert_data)
            action = dqn_result['recommended_action']
            context = dqn_result.get('dqn_context', {})
            
            prompt = self.config['gemini']['prompt_template'].format(
                alert_summary=alert_summary,
                action=action,
                context=json.dumps(context, indent=2)
            )
            
            # Generate explanation
            explanation = self.circuit_breakers['gemini_api'].call(
                self._call_gemini_api,
                prompt
            )
            
            self.stats['explanations_generated'] += 1
            self.metrics['gemini_api_calls_total'] += 1
            
            return explanation
            
        except Exception as e:
            logger.warning(f"Gemini explanation failed, using fallback: {e}")
            return self._generate_simple_explanation(dqn_result)
    
    def _call_gemini_api(self, prompt: str) -> str:
        """Call Gemini API with timeout"""
        timeout = self.config['gemini']['timeout']
        
        if self.gemini_client == 'vertexai':
            response = self.gemini_model.generate_content(
                prompt,
                generation_config={
                    'temperature': self.config['gemini']['temperature'],
                    'max_output_tokens': self.config['gemini']['max_output_tokens']
                }
            )
            return response.text
        elif self.gemini_client == 'rest':
            result = self.gemini_model.analyze_alert(prompt, "security")
            return result.get('response', 'No explanation available')
        else:
            raise Exception("No Gemini client available")
    
    def _generate_simple_explanation(self, dqn_result: dict) -> str:
        """Generate simple explanation without Gemini"""
        action = dqn_result['recommended_action']
        confidence = dqn_result['confidence']
        
        explanations = {
            'monitor': f"Recommended enhanced monitoring (confidence: {confidence:.2f}). The system detected suspicious activity that requires closer observation.",
            'block_traffic': f"Recommended blocking traffic (confidence: {confidence:.2f}). Network-based attack detected, traffic filtering will help contain the threat.",
            'patch_system': f"Recommended system patching (confidence: {confidence:.2f}). Vulnerability-based attack detected, patching will close security gaps.",
            'isolate_node': f"Recommended node isolation (confidence: {confidence:.2f}). Compromised system detected, isolation prevents lateral movement.",
            'no_action': f"No immediate action recommended (confidence: {confidence:.2f}). Alert appears to be low risk or false positive."
        }
        
        return explanations.get(action, f"Action: {action} (confidence: {confidence:.2f})")
    
    def _summarize_alert(self, alert_data: dict) -> str:
        """Create summary of alert for Gemini context"""
        summary_parts = []
        
        if 'attack_type' in alert_data:
            summary_parts.append(f"Attack Type: {alert_data['attack_type']}")
        
        if 'severity' in alert_data:
            summary_parts.append(f"Severity: {alert_data['severity']}")
        
        if 'affected_assets' in alert_data:
            asset_count = len(alert_data['affected_assets'])
            summary_parts.append(f"Affected Assets: {asset_count}")
        
        if 'source_ip' in alert_data:
            summary_parts.append(f"Source: {alert_data['source_ip']}")
        
        return " | ".join(summary_parts) if summary_parts else "Security alert detected"
    
    def publish_containment_request(self, result: AlertProcessingResult):
        """Publish containment request to CRA"""
        if not self.config['feature_flags']['CONTAINMENT_PUBLISHING_ENABLED']:
            logger.debug("Containment publishing disabled")
            return
        
        try:
            message_data = {
                'alert_id': result.alert_id,
                'action': result.recommended_action,
                'action_id': result.action_id,
                'confidence': result.confidence,
                'explanation': result.explanation,
                'agent_id': self.agent_id,
                'timestamp': result.timestamp,
                'processing_time': result.processing_time
            }
            
            message_json = json.dumps(message_data)
            message_bytes = message_json.encode('utf-8')
            
            # Publish message
            future = self.publisher.publish(self.containment_topic_path, message_bytes)
            future.result()  # Wait for publish to complete
            
            logger.info(f"Containment request published for alert {result.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to publish containment request: {e}")
            raise
    
    def publish_feedback(self, result: AlertProcessingResult, feedback: dict):
        """Publish feedback to CLA for learning"""
        if not self.config['feature_flags']['CLA_FEEDBACK_ENABLED']:
            logger.debug("CLA feedback disabled")
            return
        
        try:
            feedback_data = {
                'alert_id': result.alert_id,
                'action_taken': result.recommended_action,
                'confidence': result.confidence,
                'outcome': feedback.get('outcome', 'unknown'),
                'effectiveness': feedback.get('effectiveness', 0.5),
                'agent_id': self.agent_id,
                'timestamp': datetime.now().isoformat(),
                'metadata': {
                    'processing_time': result.processing_time,
                    'dqn_context': result.dqn_context
                }
            }
            
            message_json = json.dumps(feedback_data)
            message_bytes = message_json.encode('utf-8')
            
            # Publish feedback
            future = self.publisher.publish(self.feedback_topic_path, message_bytes)
            future.result()
            
            logger.info(f"Feedback published for alert {result.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to publish feedback: {e}")
    
    def process_ada_alert_callback(self, message):
        """Callback for processing ADA alerts from Pub/Sub"""
        try:
            # Decode message
            alert_data = json.loads(message.data.decode('utf-8'))
            
            # Process alert
            result = self.process_alert(alert_data)
            
            # Publish containment request if action is needed
            if result.action_id != 4:  # Not "no_action"
                self.publish_containment_request(result)
            
            # Acknowledge message
            message.ack()
            
            logger.info(f"ADA alert processed and acknowledged: {result.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to process ADA alert: {e}")
            message.nack()  # Negative acknowledgment for retry
    
    def start_pubsub_listener(self):
        """Start listening for ADA alerts"""
        try:
            flow_control = pubsub_v1.types.FlowControl(
                max_messages=self.config['pubsub']['subscriptions']['ada_alerts']['flow_control']['max_messages'],
                max_bytes=self.config['pubsub']['subscriptions']['ada_alerts']['flow_control']['max_bytes']
            )
            
            # Start subscriber
            streaming_pull_future = self.subscriber.subscribe(
                self.alerts_subscription_path,
                callback=self.process_ada_alert_callback,
                flow_control=flow_control
            )
            
            logger.info(f"Listening for ADA alerts on {self.alerts_subscription_path}")
            
            # Keep the main thread running
            try:
                streaming_pull_future.result()
            except KeyboardInterrupt:
                streaming_pull_future.cancel()
                logger.info("Pub/Sub listener stopped by user")
            
        except Exception as e:
            logger.error(f"Pub/Sub listener failed: {e}")
            raise
    
    def start_a2a_server(self):
        """Start A2A protocol server"""
        if not self.app or not self.config['feature_flags']['A2A_PROTOCOL_ENABLED']:
            logger.info("A2A server disabled")
            return
        
        try:
            host = self.config['a2a']['host']
            port = self.config['a2a']['port']
            
            uvicorn.run(
                self.app,
                host=host,
                port=port,
                log_level="info"
            )
        except Exception as e:
            logger.error(f"A2A server failed: {e}")
            raise
    
    def run(self):
        """Start the hybrid agent"""
        logger.info("Starting Hybrid Cyber Defense Agent...")
        self.is_running = True
        
        try:
            # Start A2A server in background thread
            if self.config['feature_flags']['A2A_PROTOCOL_ENABLED']:
                a2a_thread = threading.Thread(target=self.start_a2a_server, daemon=True)
                a2a_thread.start()
                logger.info("A2A server started in background")
            
            # Start Pub/Sub listener (blocking)
            self.start_pubsub_listener()
            
        except KeyboardInterrupt:
            logger.info("Shutting down agent...")
        except Exception as e:
            logger.error(f"Agent failed: {e}")
            raise
        finally:
            self.stop()
    
    def stop(self):
        """Stop the hybrid agent"""
        logger.info("Stopping Hybrid Cyber Defense Agent...")
        self.is_running = False
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Close Pub/Sub clients
        if hasattr(self, 'subscriber'):
            self.subscriber.close()
        if hasattr(self, 'publisher'):
            self.publisher.close()
        
        logger.info("Agent stopped successfully")


def main():
    """Main entry point"""
    try:
        # Create and run agent
        agent = HybridCyberDefenseAgent()
        agent.run()
    except Exception as e:
        logger.error(f"Failed to start agent: {e}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())


















