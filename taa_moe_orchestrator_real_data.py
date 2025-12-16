#!/usr/bin/env python3
"""
TAA MoE Orchestrator with Real BigQuery and Pub/Sub Integration
Connects to actual security data instead of mock responses
"""

import json
import logging
import asyncio
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import os
import sys

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from google.cloud import bigquery
    from google.cloud import pubsub_v1
    from google.oauth2 import service_account
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Install with: pip install google-cloud-bigquery google-cloud-pubsub")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaaMoeOrchestrator:
    """TAA MoE Orchestrator with real data integration"""
    
    def __init__(self):
        self.project_id = "chronicle-dev-2be9"
        self.bigquery_client = None
        self.pubsub_publisher = None
        self.pubsub_subscriber = None
        
        # Initialize BigQuery and Pub/Sub clients
        self._initialize_clients()
        
        # Expert agents configuration
        self.expert_agents = {
            "malware_detection": {
                "name": "Malware Detection Expert",
                "specialization": ["malware", "virus", "trojan", "ransomware"],
                "confidence_threshold": 0.7
            },
            "network_intrusion": {
                "name": "Network Intrusion Expert", 
                "specialization": ["intrusion", "breach", "unauthorized_access", "lateral_movement"],
                "confidence_threshold": 0.8
            },
            "phishing_detection": {
                "name": "Phishing Detection Expert",
                "specialization": ["phishing", "social_engineering", "email_spoofing"],
                "confidence_threshold": 0.6
            },
            "insider_threat": {
                "name": "Insider Threat Expert",
                "specialization": ["insider", "privilege_abuse", "data_exfiltration"],
                "confidence_threshold": 0.75
            },
            "data_breach": {
                "name": "Data Breach Expert",
                "specialization": ["data_breach", "exfiltration", "unauthorized_access"],
                "confidence_threshold": 0.85
            }
        }
    
    def _initialize_clients(self):
        """Initialize BigQuery and Pub/Sub clients"""
        try:
            # Initialize BigQuery client
            self.bigquery_client = bigquery.Client(project=self.project_id)
            logger.info("BigQuery client initialized successfully")
            
            # Initialize Pub/Sub clients
            self.pubsub_publisher = pubsub_v1.PublisherClient()
            self.pubsub_subscriber = pubsub_v1.SubscriberClient()
            logger.info("Pub/Sub clients initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize clients: {e}")
            self.bigquery_client = None
            self.pubsub_publisher = None
            self.pubsub_subscriber = None
    
    def get_real_time_alerts(self, limit=10):
        """Get real-time alerts from BigQuery dashboard_alerts table"""
        try:
            if not self.bigquery_client:
                return {"error": "BigQuery client not initialized"}
            
            query = f"""
            SELECT 
                alarm_id,
                ada_case_class,
                ada_confidence,
                taa_confidence,
                ada_score,
                taa_severity,
                ada_reasoning,
                taa_reasoning,
                taa_created
            FROM `{self.project_id}.gatra_database.dashboard_alerts`
            WHERE taa_created >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
            ORDER BY taa_created DESC
            LIMIT {limit}
            """
            
            query_job = self.bigquery_client.query(query)
            results = query_job.result()
            
            alerts = []
            for row in results:
                alerts.append({
                    "alarm_id": row.alarm_id,
                    "case_class": row.ada_case_class,
                    "ada_confidence": float(row.ada_confidence) if row.ada_confidence else 0.0,
                    "taa_confidence": float(row.taa_confidence) if row.taa_confidence else 0.0,
                    "ada_score": float(row.ada_score) if row.ada_score else 0.0,
                    "taa_severity": float(row.taa_severity) if row.taa_severity else 0.0,
                    "ada_reasoning": row.ada_reasoning,
                    "taa_reasoning": row.taa_reasoning,
                    "timestamp": row.taa_created.isoformat() if row.taa_created else datetime.now().isoformat()
                })
            
            return {"alerts": alerts, "count": len(alerts)}
            
        except Exception as e:
            logger.error(f"Error getting real-time alerts: {e}")
            return {"error": str(e)}
    
    def get_threat_statistics(self):
        """Get threat statistics from real BigQuery data"""
        try:
            if not self.bigquery_client:
                return {"error": "BigQuery client not initialized"}
            
            query = f"""
            SELECT 
                ada_case_class,
                COUNT(*) as count,
                AVG(ada_confidence) as avg_confidence,
                AVG(taa_severity) as avg_severity
            FROM `{self.project_id}.gatra_database.dashboard_alerts`
            WHERE taa_created >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
            GROUP BY ada_case_class
            ORDER BY count DESC
            """
            
            query_job = self.bigquery_client.query(query)
            results = query_job.result()
            
            stats = []
            for row in results:
                stats.append({
                    "threat_type": row.ada_case_class or "unknown",
                    "count": row.count,
                    "avg_confidence": float(row.avg_confidence) if row.avg_confidence else 0.0,
                    "avg_severity": float(row.avg_severity) if row.avg_severity else 0.0
                })
            
            return {"statistics": stats, "total_alerts": sum(s["count"] for s in stats)}
            
        except Exception as e:
            logger.error(f"Error getting threat statistics: {e}")
            return {"error": str(e)}
    
    def route_to_expert_agent(self, alert_data):
        """Route alert to appropriate expert agent based on content analysis"""
        try:
            # Analyze alert content to determine best expert
            alert_text = f"{alert_data.get('description', '')} {alert_data.get('alert_type', '')}".lower()
            
            best_expert = None
            best_score = 0
            
            for agent_id, agent_config in self.expert_agents.items():
                score = 0
                for keyword in agent_config["specialization"]:
                    if keyword in alert_text:
                        score += 1
                
                # Normalize score
                score = score / len(agent_config["specialization"])
                
                if score > best_score:
                    best_score = score
                    best_expert = agent_id
            
            # If no expert matches well, default to network_intrusion
            if not best_expert or best_score < 0.3:
                best_expert = "network_intrusion"
            
            expert_config = self.expert_agents[best_expert]
            
            # Simulate expert analysis (in real implementation, this would call actual AI models)
            analysis_result = {
                "expert_agent": expert_config["name"],
                "agent_id": best_expert,
                "confidence": min(0.95, best_score + 0.5),  # Boost confidence based on match
                "threat_level": self._calculate_threat_level(alert_data, best_score),
                "recommendations": self._generate_recommendations(best_expert, alert_data),
                "analysis_timestamp": datetime.now().isoformat(),
                "routing_score": best_score
            }
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error routing to expert agent: {e}")
            return {"error": str(e)}
    
    def _calculate_threat_level(self, alert_data, routing_score):
        """Calculate threat level based on alert data and routing score"""
        severity_map = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        base_severity = severity_map.get(alert_data.get("severity", "medium").lower(), 2)
        
        # Adjust based on routing confidence
        adjusted_level = base_severity * (0.5 + routing_score)
        
        if adjusted_level >= 3.5:
            return "critical"
        elif adjusted_level >= 2.5:
            return "high"
        elif adjusted_level >= 1.5:
            return "medium"
        else:
            return "low"
    
    def _generate_recommendations(self, expert_id, alert_data):
        """Generate recommendations based on expert type and alert data"""
        recommendations = {
            "malware_detection": [
                "Immediately isolate affected systems",
                "Run full antivirus scan",
                "Check for lateral movement indicators",
                "Review recent file downloads and executions"
            ],
            "network_intrusion": [
                "Block suspicious IP addresses",
                "Review network logs for lateral movement",
                "Check for unauthorized access attempts",
                "Implement network segmentation"
            ],
            "phishing_detection": [
                "Quarantine suspicious emails",
                "Educate users about phishing indicators",
                "Check for credential compromise",
                "Review email security policies"
            ],
            "insider_threat": [
                "Review user access patterns",
                "Check for unusual data access",
                "Monitor privileged account usage",
                "Implement additional access controls"
            ],
            "data_breach": [
                "Assess data exposure scope",
                "Notify relevant stakeholders",
                "Implement containment measures",
                "Prepare incident response procedures"
            ]
        }
        
        return recommendations.get(expert_id, [
            "Investigate the incident thoroughly",
            "Document all findings",
            "Implement appropriate containment measures",
            "Review security controls"
        ])
    
    def publish_analysis_result(self, analysis_result):
        """Publish analysis result to Pub/Sub topic"""
        try:
            if not self.pubsub_publisher:
                logger.warning("Pub/Sub publisher not initialized")
                return False
            
            topic_path = self.pubsub_publisher.topic_path(
                self.project_id, "taa-moe-analysis-results"
            )
            
            message_data = json.dumps(analysis_result).encode('utf-8')
            future = self.pubsub_publisher.publish(topic_path, message_data)
            
            logger.info(f"Published analysis result: {future.result()}")
            return True
            
        except Exception as e:
            logger.error(f"Error publishing to Pub/Sub: {e}")
            return False

class RealDataOrchestratorHandler(BaseHTTPRequestHandler):
    """HTTP handler with real data integration"""
    
    def __init__(self, *args, **kwargs):
        self.orchestrator = TaaMoeOrchestrator()
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests"""
        try:
            parsed_path = urlparse(self.path)
            
            if parsed_path.path == '/':
                self._send_response({
                    'message': 'TAA-MoE Orchestrator API Server with Real Data',
                    'status': 'running',
                    'features': [
                        'BigQuery integration',
                        'Pub/Sub integration', 
                        'Expert agent routing',
                        'Real-time threat analysis'
                    ]
                })
                
            elif parsed_path.path == '/health':
                health_status = {
                    'status': 'healthy',
                    'timestamp': datetime.now().isoformat(),
                    'bigquery_connected': self.orchestrator.bigquery_client is not None,
                    'pubsub_connected': self.orchestrator.pubsub_publisher is not None,
                    'expert_agents': len(self.orchestrator.expert_agents)
                }
                self._send_response(health_status)
                
            elif parsed_path.path == '/alerts':
                # Get real-time alerts from BigQuery
                limit = int(parse_qs(parsed_path.query).get('limit', ['10'])[0])
                alerts_data = self.orchestrator.get_real_time_alerts(limit)
                self._send_response(alerts_data)
                
            elif parsed_path.path == '/statistics':
                # Get threat statistics from BigQuery
                stats_data = self.orchestrator.get_threat_statistics()
                self._send_response(stats_data)
                
            elif parsed_path.path == '/experts':
                # Get expert agent information
                experts_info = {
                    'expert_agents': self.orchestrator.expert_agents,
                    'total_experts': len(self.orchestrator.expert_agents)
                }
                self._send_response(experts_info)
                
            else:
                self._send_error_response(404, "Endpoint not found")
                
        except Exception as e:
            logger.error(f"Error in GET request: {e}")
            self._send_error_response(500, str(e))
    
    def do_POST(self):
        """Handle POST requests"""
        try:
            parsed_path = urlparse(self.path)
            
            if parsed_path.path.startswith('/orchestrator/run'):
                # Process alert with real data
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                alert_data = json.loads(post_data.decode('utf-8'))
                
                # Route to expert agent
                analysis_result = self.orchestrator.route_to_expert_agent(alert_data)
                
                # Add original alert data
                analysis_result['original_alert'] = alert_data
                analysis_result['processing_timestamp'] = datetime.now().isoformat()
                
                # Publish to Pub/Sub
                self.orchestrator.publish_analysis_result(analysis_result)
                
                self._send_response(analysis_result)
                
            else:
                self._send_error_response(404, "Endpoint not found")
                
        except Exception as e:
            logger.error(f"Error in POST request: {e}")
            self._send_error_response(500, str(e))
    
    def _send_response(self, data):
        """Send JSON response"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())
    
    def _send_error_response(self, status_code, message):
        """Send error response"""
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        error_response = {
            'error': message,
            'status_code': status_code,
            'timestamp': datetime.now().isoformat()
        }
        self.wfile.write(json.dumps(error_response).encode())

def run_server(port=8000):
    """Run the orchestrator server"""
    server_address = ('', port)
    httpd = HTTPServer(server_address, RealDataOrchestratorHandler)
    print(f'Starting TAA-MoE Orchestrator with Real Data on port {port}')
    print('Features:')
    print('- BigQuery integration for real alert data')
    print('- Pub/Sub integration for real-time processing')
    print('- Expert agent routing based on threat analysis')
    print('- Real-time threat statistics and analytics')
    httpd.serve_forever()

if __name__ == '__main__':
    run_server()


