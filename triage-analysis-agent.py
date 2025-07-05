"""
Triage and Analysis Agent (TAA) for AI-driven SOC

Core Responsibilities:
- Analysis of security alerts to determine validity
- Enrichment of alerts with additional context and threat intelligence
- Classification of alerts as true or false positives
- Prioritization of confirmed incidents based on severity
- Detailed analysis of attack patterns and techniques
"""

import os
import json
import logging
import base64
import requests
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from google.cloud import pubsub_v1, storage
import vertexai
from vertexai.language_models import TextGenerationModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TriageAnalysisAgent")

class TriageAnalysisAgent:
    """Triage and Analysis Agent for security alerts"""
    
    def __init__(self, config_path: str = "config/taa_config.json"):
        """Initialize the Triage and Analysis Agent
        
        Args:
            config_path: Path to the agent configuration file
        """
        self.config = self._load_config(config_path)
        
        # Initialize publishers and subscribers
        self.publisher = pubsub_v1.PublisherClient()
        self.subscriber = pubsub_v1.SubscriberClient()
        
        # Topic paths
        self.incident_topic_path = self.publisher.topic_path(
            self.config["project_id"], 
            self.config["incident_topic"]
        )
        self.feedback_topic_path = self.publisher.topic_path(
            self.config["project_id"], 
            self.config["feedback_topic"]
        )
        
        # Initialize Vertex AI for LLM analysis
        vertexai.init(project=self.config["project_id"], location=self.config["location"])
        self.llm = TextGenerationModel.from_pretrained(self.config["llm_model"])
        
        # Threat intelligence APIs cache
        self.ti_cache = {}
        
        logger.info("Triage and Analysis Agent initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load the agent configuration from a JSON file
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Dict containing the configuration
        """
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            # Use default configuration if file not found
            return {
                "project_id": "ai-driven-soc",
                "location": "us-central1",
                "alert_subscription": "security-alerts-sub",
                "incident_topic": "confirmed-incidents",
                "feedback_topic": "analysis-feedback",
                "llm_model": "gemini-1.0-pro",
                "ti_apis": {
                    "virus_total": {
                        "url": "https://www.virustotal.com/api/v3/",
                        "api_key": os.environ.get("VT_API_KEY", "")
                    },
                    "abuse_ipdb": {
                        "url": "https://api.abuseipdb.com/api/v2/",
                        "api_key": os.environ.get("AIPDB_API_KEY", "")
                    }
                },
                "severity_thresholds": {
                    "high": 0.8,
                    "medium": 0.6,
                    "low": 0.4
                },
                "cache_ttl": 3600  # Cache TI data for 1 hour
            }
    
    def process_alert(self, alert: Dict) -> Dict:
        """Process a security alert
        
        Args:
            alert: Alert data from the Anomaly Detection Agent
            
        Returns:
            Dict containing analysis results with a decision (true/false positive)
        """
        logger.info(f"Processing alert: {alert.get('alert_id', 'unknown')}")
        
        # Step 1: Enrich the alert with context
        enriched_alert = self._enrich_alert(alert)
        
        # Step 2: Perform analysis using LLM
        analysis_result = self._analyze_with_llm(enriched_alert)
        
        # Step 3: Determine if it's a true or false positive
        decision = self._make_decision(enriched_alert, analysis_result)
        
        # Step 4: Prepare the result
        result = {
            "alert_id": alert.get("alert_id"),
            "timestamp": datetime.now().isoformat(),
            "original_alert": alert,
            "enriched_data": enriched_alert.get("enriched_data", {}),
            "analysis": analysis_result,
            "decision": decision,
            "is_true_positive": decision["is_true_positive"],
            "severity": decision["severity"] if decision["is_true_positive"] else "none",
            "confidence": decision["confidence"],
            "recommended_actions": decision.get("recommended_actions", [])
        }
        
        # Send feedback to the ADA 
        self._send_feedback(result)
        
        return result
    
    def _enrich_alert(self, alert: Dict) -> Dict:
        """Enrich the alert with additional context and threat intelligence
        
        Args:
            alert: Original alert data
            
        Returns:
            Enriched alert with additional context
        """
        enriched_alert = alert.copy()
        enriched_data = {}
        
        # Extract relevant fields from the alert
        log_data = alert.get("log_data", {})
        src_ip = log_data.get("src_ip")
        dst_ip = log_data.get("dst_ip")
        domain = log_data.get("domain")
        
        # Enrich with threat intelligence if available
        if src_ip:
            enriched_data["src_ip_intel"] = self._get_ip_intelligence(src_ip)
        
        if dst_ip:
            enriched_data["dst_ip_intel"] = self._get_ip_intelligence(dst_ip)
        
        if domain:
            enriched_data["domain_intel"] = self._get_domain_intelligence(domain)
        
        # Add historical context
        enriched_data["historical_context"] = self._get_historical_context(alert)
        
        # Add the enriched data to the alert
        enriched_alert["enriched_data"] = enriched_data
        
        return enriched_alert
    
    def _get_ip_intelligence(self, ip: str) -> Dict:
        """Get threat intelligence for an IP address
        
        Args:
            ip: IP address to check
            
        Returns:
            Dict containing threat intelligence for the IP
        """
        # Check cache first
        cache_key = f"ip_{ip}"
        if cache_key in self.ti_cache:
            cache_entry = self.ti_cache[cache_key]
            if cache_entry["timestamp"] > datetime.now() - timedelta(seconds=self.config["cache_ttl"]):
                return cache_entry["data"]
        
        # Initialize result
        result = {
            "reputation_score": None,
            "is_malicious": False,
            "categories": [],
            "recent_reports": 0,
            "country": None,
            "asn": None,
            "last_reported": None
        }
        
        # Try AbuseIPDB
        try:
            abuse_api = self.config["ti_apis"]["abuse_ipdb"]
            headers = {
                "Key": abuse_api["api_key"],
                "Accept": "application/json"
            }
            params = {
                "ipAddress": ip,
                "maxAgeInDays": 90
            }
            response = requests.get(
                f"{abuse_api['url']}check", 
                headers=headers, 
                params=params
            )
            
            if response.status_code == 200:
                data = response.json().get("data", {})
                result["reputation_score"] = 100 - data.get("abuseConfidenceScore", 0)
                result["is_malicious"] = data.get("abuseConfidenceScore", 0) > 50
                result["recent_reports"] = data.get("totalReports", 0)
                result["country"] = data.get("countryCode")
                result["last_reported"] = data.get("lastReportedAt")
                
                # Categories derived from reports
                if "reports" in data and len(data["reports"]) > 0:
                    categories = set()
                    for report in data["reports"]:
                        for cat in report.get("categories", []):
                            categories.add(cat)
                    result["categories"] = list(categories)
        except Exception as e:
            logger.warning(f"Error getting AbuseIPDB intelligence for {ip}: {e}")
        
        # Cache the result
        self.ti_cache[cache_key] = {
            "timestamp": datetime.now(),
            "data": result
        }
        
        return result
    
    def _get_domain_intelligence(self, domain: str) -> Dict:
        """Get threat intelligence for a domain
        
        Args:
            domain: Domain to check
            
        Returns:
            Dict containing threat intelligence for the domain
        """
        # Check cache first
        cache_key = f"domain_{domain}"
        if cache_key in self.ti_cache:
            cache_entry = self.ti_cache[cache_key]
            if cache_entry["timestamp"] > datetime.now() - timedelta(seconds=self.config["cache_ttl"]):
                return cache_entry["data"]
        
        # Initialize result
        result = {
            "reputation_score": None,
            "is_malicious": False,
            "categories": [],
            "last_analysis_date": None,
            "creation_date": None,
            "registrar": None
        }
        
        # Try VirusTotal
        try:
            vt_api = self.config["ti_apis"]["virus_total"]
            headers = {
                "x-apikey": vt_api["api_key"]
            }
            response = requests.get(
                f"{vt_api['url']}domains/{domain}", 
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json().get("data", {}).get("attributes", {})
                last_analysis = data.get("last_analysis_stats", {})
                total_engines = sum(last_analysis.values())
                malicious_engines = last_analysis.get("malicious", 0)
                
                # Calculate reputation score (0-100)
                if total_engines > 0:
                    result["reputation_score"] = 100 - (malicious_engines / total_engines * 100)
                    result["is_malicious"] = malicious_engines > 2  # More than 2 engines report malicious
                
                result["last_analysis_date"] = data.get("last_analysis_date")
                result["creation_date"] = data.get("creation_date")
                result["registrar"] = data.get("registrar")
                
                # Categories from analysis
                categories = set()
                for engine_result in data.get("last_analysis_results", {}).values():
                    if engine_result.get("category") and engine_result.get("category") != "undetected":
                        categories.add(engine_result.get("category"))
                result["categories"] = list(categories)
        except Exception as e:
            logger.warning(f"Error getting VirusTotal intelligence for {domain}: {e}")
        
        # Cache the result
        self.ti_cache[cache_key] = {
            "timestamp": datetime.now(),
            "data": result
        }
        
        return result
    
    def _get_historical_context(self, alert: Dict) -> Dict:
        """Get historical context for the alert
        
        Args:
            alert: Alert data
            
        Returns:
            Dict containing historical context
        """
        # This would typically query a database or data warehouse
        # Simplified implementation for now
        log_data = alert.get("log_data", {})
        src_ip = log_data.get("src_ip")
        
        result = {
            "previous_alerts": 0,
            "previous_incidents": 0,
            "first_seen": None,
            "last_seen": None,
            "recent_activity": []
        }
        
        # Simulated historical data lookup
        if src_ip:
            # This would be replaced with actual queries to BigQuery or other data stores
            # Simulated data for demonstration
            result["previous_alerts"] = 5 if src_ip.startswith("10.") else 0
            result["previous_incidents"] = 2 if src_ip.startswith("10.") else 0
            result["first_seen"] = (datetime.now() - timedelta(days=30)).isoformat() if result["previous_alerts"] > 0 else None
            result["last_seen"] = (datetime.now() - timedelta(days=1)).isoformat() if result["previous_alerts"] > 0 else None
        
        return result
    
    def _analyze_with_llm(self, enriched_alert: Dict) -> Dict:
        """Analyze the enriched alert using LLM
        
        Args:
            enriched_alert: Alert data enriched with context
            
        Returns:
            Analysis results from the LLM
        """
        # Prepare the prompt for the LLM
        alert_json = json.dumps(enriched_alert, indent=2)
        
        prompt = f"""You are an expert security analyst. Analyze the following security alert and determine if it is a true or false positive.
        
Alert details:
{alert_json}

Please analyze this alert and provide:
1. Is this a true positive or false positive?
2. What is the severity level (high, medium, low)?
3. What is the confidence in your determination (0.0-1.0)?
4. What is the likely attack type or category?
5. What are the indicators of compromise?
6. What recommended actions should be taken?
7. Provide a brief summary of your analysis.

Format your response as JSON with the following structure:
{{
  "is_true_positive": true/false,
  "severity": "high/medium/low",
  "confidence": 0.XX,
  "attack_type": "...",
  "indicators": ["..."],
  "recommended_actions": ["..."],
  "analysis_summary": "..."
}}
"""
        
        try:
            # Call the LLM for analysis
            response = self.llm.predict(
                prompt,
                temperature=0.2,
                max_output_tokens=1024,
                top_k=40,
                top_p=0.8
            )
            
            # Parse the response
            text_response = response.text
            # Extract JSON - the model might include markdown formatting
            json_start = text_response.find("{")
            json_end = text_response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_text = text_response[json_start:json_end]
                analysis = json.loads(json_text)
                return analysis
            else:
                logger.error("Could not extract JSON from LLM response")
                # Return a default analysis with low confidence
                return {
                    "is_true_positive": False,
                    "severity": "low",
                    "confidence": 0.5,
                    "attack_type": "unknown",
                    "indicators": [],
                    "recommended_actions": [],
                    "analysis_summary": "Error parsing LLM analysis"
                }
        except Exception as e:
            logger.error(f"Error in LLM analysis: {e}")
            # Return a default analysis with low confidence
            return {
                "is_true_positive": False,
                "severity": "low",
                "confidence": 0.3,
                "attack_type": "unknown",
                "indicators": [],
                "recommended_actions": [],
                "analysis_summary": f"Error in analysis: {str(e)}"
            }
    
    def _make_decision(self, enriched_alert: Dict, analysis_result: Dict) -> Dict:
        """Make a final decision based on the analysis
        
        Args:
            enriched_alert: Enriched alert data
            analysis_result: Results from LLM analysis
            
        Returns:
            Decision dict with final determination
        """
        # Start with the LLM analysis
        decision = {
            "is_true_positive": analysis_result.get("is_true_positive", False),
            "severity": analysis_result.get("severity", "low"),
            "confidence": analysis_result.get("confidence", 0.5),
            "attack_type": analysis_result.get("attack_type", "unknown"),
            "indicators": analysis_result.get("indicators", []),
            "recommended_actions": analysis_result.get("recommended_actions", []),
            "analysis_summary": analysis_result.get("analysis_summary", "")
        }
        
        # Apply additional heuristic rules
        
        # Check threat intelligence scores for IPs and domains
        enriched_data = enriched_alert.get("enriched_data", {})
        
        # Check if any IP is confirmed malicious by threat intelligence
        ip_is_malicious = False
        ip_malicious_confidence = 0.0
        
        if "src_ip_intel" in enriched_data and enriched_data["src_ip_intel"].get("is_malicious", False):
            ip_is_malicious = True
            ip_malicious_confidence = max(ip_malicious_confidence, 
                                         (100 - enriched_data["src_ip_intel"].get("reputation_score", 0)) / 100)
        
        if "dst_ip_intel" in enriched_data and enriched_data["dst_ip_intel"].get("is_malicious", False):
            ip_is_malicious = True
            ip_malicious_confidence = max(ip_malicious_confidence, 
                                         (100 - enriched_data["dst_ip_intel"].get("reputation_score", 0)) / 100)
        
        # Check if domain is confirmed malicious
        domain_is_malicious = False
        domain_malicious_confidence = 0.0
        
        if "domain_intel" in enriched_data and enriched_data["domain_intel"].get("is_malicious", False):
            domain_is_malicious = True
            domain_malicious_confidence = max(domain_malicious_confidence, 
                                            (100 - enriched_data["domain_intel"].get("reputation_score", 0)) / 100)
        
        # If threat intelligence confirms malicious activity, increase confidence
        if ip_is_malicious or domain_is_malicious:
            max_ti_confidence = max(ip_malicious_confidence, domain_malicious_confidence)
            
            # If LLM already said it's a true positive, increase confidence
            if decision["is_true_positive"]:
                decision["confidence"] = min(1.0, decision["confidence"] + (max_ti_confidence * 0.3))
            else:
                # If TI is very confident but LLM said false positive, reconsider
                if max_ti_confidence > 0.8:
                    decision["is_true_positive"] = True
                    decision["confidence"] = max_ti_confidence
        
        # Check historical context
        historical = enriched_data.get("historical_context", {})
        if historical.get("previous_incidents", 0) > 0:
            # If there have been previous incidents, increase confidence
            if decision["is_true_positive"]:
                decision["confidence"] = min(1.0, decision["confidence"] + 0.1)
        
        # Apply severity thresholds
        thresholds = self.config["severity_thresholds"]
        if decision["is_true_positive"]:
            if decision["confidence"] >= thresholds["high"]:
                decision["severity"] = "high"
            elif decision["confidence"] >= thresholds["medium"]:
                decision["severity"] = "medium"
            elif decision["confidence"] >= thresholds["low"]:
                decision["severity"] = "low"
            else:
                decision["severity"] = "informational"
        
        # Add timestamp for the decision
        decision["timestamp"] = datetime.now().isoformat()
        
        return decision
    
    def _send_feedback(self, result: Dict) -> None:
        """Send feedback to the Anomaly Detection Agent
        
        Args:
            result: Analysis result
        """
        try:
            # Prepare feedback message
            feedback = {
                "alert_id": result.get("alert_id"),
                "timestamp": datetime.now().isoformat(),
                "is_true_positive": result.get("is_true_positive", False),
                "confidence": result.get("confidence", 0.0),
                "feedback_type": "triage_analysis"
            }
            
            # Publish to feedback topic
            message_data = json.dumps(feedback).encode("utf-8")
            future = self.publisher.publish(self.feedback_topic_path, message_data)
            message_id = future.result()
            logger.info(f"Feedback published with message ID: {message_id}")
        except Exception as e:
            logger.error(f"Error sending feedback: {e}")
    
    def publish_incident(self, incident: Dict) -> None:
        """Publish a confirmed incident to the incident topic
        
        Args:
            incident: Confirmed incident data
        """
        try:
            message_data = json.dumps(incident).encode("utf-8")
            future = self.publisher.publish(self.incident_topic_path, message_data)
            message_id = future.result()
            logger.info(f"Incident published with message ID: {message_id}")
        except Exception as e:
            logger.error(f"Error publishing incident: {e}")
    
    def process_message(self, message) -> None:
        """Process a message from the alert subscription
        
        Args:
            message: Pub/Sub message
        """
        try:
            # Decode the message data
            alert_data = json.loads(message.data.decode("utf-8"))
            
            # Process the alert
            result = self.process_alert(alert_data)
            
            # If it's a true positive, publish as an incident
            if result.get("is_true_positive", False):
                self.publish_incident(result)
                logger.info(f"Incident confirmed and published: {result.get('alert_id')}")
            else:
                logger.info(f"Alert classified as false positive: {result.get('alert_id')}")
            
            # Acknowledge the message
            message.ack()
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            # Don't acknowledge to allow retry
    
    def listen_for_alerts(self):
        """Listen for alerts from the subscription"""
        subscription_path = self.subscriber.subscription_path(
            self.config["project_id"],
            self.config["alert_subscription"]
        )
        
        def callback(message):
            self.process_message(message)
        
        streaming_pull_future = self.subscriber.subscribe(
            subscription_path, callback=callback
        )
        logger.info(f"Listening for messages on {subscription_path}")
        
        try:
            # Keep the main thread alive
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            streaming_pull_future.cancel()
            logger.info("Listener stopped")

# Cloud Function entry point
def process_alert_function(event, context):
    """Cloud Function entry point for processing alerts
    
    Args:
        event: Pub/Sub event
        context: Event context
    """
    import base64
    
    agent = TriageAnalysisAgent()
    
    # Decode the alert data
    alert_data = json.loads(base64.b64decode(event["data"]).decode("utf-8"))
    
    # Process the alert
    result = agent.process_alert(alert_data)
    
    # If it's a true positive, publish as an incident
    if result.get("is_true_positive", False):
        agent.publish_incident(result)
    
    return "Alert processed successfully"

if __name__ == "__main__":
    # For testing
    agent = TriageAnalysisAgent()
    
    # Test alert
    test_alert = {
        "alert_id": "ADA-20230503123456-7890",
        "timestamp": "2023-05-03T12:34:56Z",
        "anomaly_score": 0.85,
        "confidence": 0.85,
        "alert_type": "ANOMALY_DETECTION",
        "description": "Anomalous network activity detected with confidence 0.85",
        "raw_data_reference": "log-123456",
        "log_data": {
            "log_id": "log-123456",
            "timestamp": "2023-05-03T12:34:56Z",
            "duration": 120,
            "protocol_type": "tcp",
            "service": "http",
            "flag": "SF",
            "src_bytes": 1200,
            "dst_bytes": 15000,
            "src_ip": "10.0.0.1",
            "dst_ip": "203.0.113.1",
            "domain": "example.com"
        }
    }
    
    # Process the test alert
    result = agent.process_alert(test_alert)
    
    # Print the result
    print(json.dumps(result, indent=2))
