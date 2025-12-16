#!/usr/bin/env python3
"""
AI-Driven SOC Operations Platform with GLM-4.6 Integration
Custom application for SOC operations using GLM-4.6 API
"""

import os
import json
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SOCAlert:
    """SOC Alert data structure"""
    id: str
    timestamp: datetime
    severity: str
    source: str
    description: str
    status: str
    confidence: float
    ai_reasoning: str
    recommended_action: str

@dataclass
class ThreatIntelligence:
    """Threat Intelligence data structure"""
    ioc: str
    ioc_type: str
    threat_level: str
    source: str
    first_seen: datetime
    last_seen: datetime
    related_attacks: List[str]
    ai_analysis: str

class GLM4Integration:
    """GLM-4.6 API Integration for SOC Operations"""
    
    def __init__(self, api_key: str, base_url: str = "https://open.bigmodel.cn/api/paas/v4/chat/completions"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    async def analyze_threat(self, alert_data: Dict) -> Dict:
        """Analyze threat using GLM-4.6"""
        prompt = f"""
        As an AI SOC analyst, analyze this security alert:
        
        Alert ID: {alert_data.get('id', 'Unknown')}
        Timestamp: {alert_data.get('timestamp', 'Unknown')}
        Source: {alert_data.get('source', 'Unknown')}
        Description: {alert_data.get('description', 'Unknown')}
        Severity: {alert_data.get('severity', 'Unknown')}
        
        Provide analysis in JSON format:
        {{
            "threat_level": "Low/Medium/High/Critical",
            "attack_type": "Specific attack type if identified",
            "confidence": 0.0-1.0,
            "reasoning": "Detailed analysis reasoning",
            "recommended_actions": ["Action 1", "Action 2"],
            "iocs": ["Indicator 1", "Indicator 2"],
            "mitigation_steps": ["Step 1", "Step 2"]
        }}
        """
        
        payload = {
            "model": "glm-4",
            "messages": [
                {"role": "system", "content": "You are an expert SOC analyst with 20+ years experience in cybersecurity."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 2000
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, headers=self.headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return json.loads(result['choices'][0]['message']['content'])
                    else:
                        logger.error(f"GLM-4.6 API error: {response.status}")
                        return self._fallback_analysis(alert_data)
        except Exception as e:
            logger.error(f"Error calling GLM-4.6: {e}")
            return self._fallback_analysis(alert_data)
    
    async def generate_incident_response(self, incident_data: Dict) -> Dict:
        """Generate incident response plan using GLM-4.6"""
        prompt = f"""
        As an AI incident response coordinator, create a response plan for this incident:
        
        Incident ID: {incident_data.get('id', 'Unknown')}
        Severity: {incident_data.get('severity', 'Unknown')}
        Attack Type: {incident_data.get('attack_type', 'Unknown')}
        Affected Systems: {incident_data.get('affected_systems', 'Unknown')}
        Current Status: {incident_data.get('status', 'Unknown')}
        
        Provide response plan in JSON format:
        {{
            "immediate_actions": ["Action 1", "Action 2"],
            "containment_steps": ["Step 1", "Step 2"],
            "eradication_plan": ["Plan 1", "Plan 2"],
            "recovery_steps": ["Step 1", "Step 2"],
            "lessons_learned": ["Lesson 1", "Lesson 2"],
            "timeline": "Estimated completion time",
            "resources_needed": ["Resource 1", "Resource 2"]
        }}
        """
        
        payload = {
            "model": "glm-4",
            "messages": [
                {"role": "system", "content": "You are an expert incident response coordinator with extensive experience in cybersecurity incident management."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 2500
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, headers=self.headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return json.loads(result['choices'][0]['message']['content'])
                    else:
                        return self._fallback_response_plan(incident_data)
        except Exception as e:
            logger.error(f"Error generating incident response: {e}")
            return self._fallback_response_plan(incident_data)
    
    async def compliance_check(self, compliance_data: Dict) -> Dict:
        """Perform compliance check using GLM-4.6"""
        prompt = f"""
        As an AI compliance officer, assess this security event for compliance:
        
        Event: {compliance_data.get('event', 'Unknown')}
        Data Type: {compliance_data.get('data_type', 'Unknown')}
        User: {compliance_data.get('user', 'Unknown')}
        System: {compliance_data.get('system', 'Unknown')}
        
        Check compliance against:
        - GDPR (General Data Protection Regulation)
        - PDP (Personal Data Protection)
        - SOX (Sarbanes-Oxley)
        - HIPAA (if applicable)
        - PCI-DSS (if applicable)
        
        Provide assessment in JSON format:
        {{
            "compliance_status": "Compliant/Non-Compliant/Needs Review",
            "violations": ["Violation 1", "Violation 2"],
            "recommendations": ["Recommendation 1", "Recommendation 2"],
            "risk_level": "Low/Medium/High/Critical",
            "required_actions": ["Action 1", "Action 2"],
            "audit_trail": "Required audit trail entries"
        }}
        """
        
        payload = {
            "model": "glm-4",
            "messages": [
                {"role": "system", "content": "You are an expert compliance officer specializing in data protection regulations and cybersecurity compliance."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 2000
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, headers=self.headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return json.loads(result['choices'][0]['message']['content'])
                    else:
                        return self._fallback_compliance_check(compliance_data)
        except Exception as e:
            logger.error(f"Error performing compliance check: {e}")
            return self._fallback_compliance_check(compliance_data)
    
    def _fallback_analysis(self, alert_data: Dict) -> Dict:
        """Fallback analysis when GLM-4.6 is unavailable"""
        return {
            "threat_level": "Medium",
            "attack_type": "Unknown",
            "confidence": 0.5,
            "reasoning": "Fallback analysis - GLM-4.6 unavailable",
            "recommended_actions": ["Manual review required"],
            "iocs": [],
            "mitigation_steps": ["Isolate affected systems"]
        }
    
    def _fallback_response_plan(self, incident_data: Dict) -> Dict:
        """Fallback response plan when GLM-4.6 is unavailable"""
        return {
            "immediate_actions": ["Isolate affected systems", "Preserve evidence"],
            "containment_steps": ["Block malicious IPs", "Disable compromised accounts"],
            "eradication_plan": ["Remove malware", "Patch vulnerabilities"],
            "recovery_steps": ["Restore from backups", "Verify system integrity"],
            "lessons_learned": ["Review security controls", "Update procedures"],
            "timeline": "24-48 hours",
            "resources_needed": ["Security team", "IT support"]
        }
    
    def _fallback_compliance_check(self, compliance_data: Dict) -> Dict:
        """Fallback compliance check when GLM-4.6 is unavailable"""
        return {
            "compliance_status": "Needs Review",
            "violations": ["Manual review required"],
            "recommendations": ["Consult compliance team"],
            "risk_level": "Medium",
            "required_actions": ["Manual assessment"],
            "audit_trail": "Log event for manual review"
        }

class SOCOperationsPlatform:
    """Main SOC Operations Platform"""
    
    def __init__(self, glm_api_key: str):
        self.glm = GLM4Integration(glm_api_key)
        self.alerts = []
        self.incidents = []
        self.threat_intel = []
        self.metrics = {
            "total_alerts": 0,
            "resolved_alerts": 0,
            "active_incidents": 0,
            "mttr": 0,
            "false_positive_rate": 0,
            "compliance_score": 0
        }
    
    async def process_alert(self, alert_data: Dict) -> SOCAlert:
        """Process new alert with GLM-4.6 analysis"""
        # Analyze with GLM-4.6
        analysis = await self.glm.analyze_threat(alert_data)
        
        # Create SOC alert
        alert = SOCAlert(
            id=alert_data.get('id', f"ALERT_{len(self.alerts)+1}"),
            timestamp=datetime.now(),
            severity=analysis.get('threat_level', 'Medium'),
            source=alert_data.get('source', 'Unknown'),
            description=alert_data.get('description', 'No description'),
            status='New',
            confidence=analysis.get('confidence', 0.5),
            ai_reasoning=analysis.get('reasoning', 'No reasoning available'),
            recommended_action=analysis.get('recommended_actions', ['Manual review'])[0]
        )
        
        self.alerts.append(alert)
        self.metrics["total_alerts"] += 1
        
        return alert
    
    async def create_incident(self, incident_data: Dict) -> Dict:
        """Create incident with GLM-4.6 response plan"""
        # Generate response plan
        response_plan = await self.glm.generate_incident_response(incident_data)
        
        incident = {
            "id": incident_data.get('id', f"INC_{len(self.incidents)+1}"),
            "timestamp": datetime.now(),
            "severity": incident_data.get('severity', 'Medium'),
            "status": 'Active',
            "response_plan": response_plan,
            "assigned_team": "SOC Team",
            "progress": 0
        }
        
        self.incidents.append(incident)
        self.metrics["active_incidents"] += 1
        
        return incident
    
    async def check_compliance(self, event_data: Dict) -> Dict:
        """Check compliance using GLM-4.6"""
        compliance_result = await self.glm.compliance_check(event_data)
        
        # Update compliance score
        if compliance_result.get('compliance_status') == 'Compliant':
            self.metrics["compliance_score"] = min(100, self.metrics["compliance_score"] + 1)
        elif compliance_result.get('compliance_status') == 'Non-Compliant':
            self.metrics["compliance_score"] = max(0, self.metrics["compliance_score"] - 5)
        
        return compliance_result
    
    def get_dashboard_data(self) -> Dict:
        """Get dashboard data for real-time display"""
        return {
            "alerts": [
                {
                    "id": alert.id,
                    "timestamp": alert.timestamp.isoformat(),
                    "severity": alert.severity,
                    "source": alert.source,
                    "description": alert.description,
                    "status": alert.status,
                    "confidence": alert.confidence,
                    "ai_reasoning": alert.ai_reasoning,
                    "recommended_action": alert.recommended_action
                }
                for alert in self.alerts[-10:]  # Last 10 alerts
            ],
            "incidents": [
                {
                    "id": incident["id"],
                    "timestamp": incident["timestamp"].isoformat(),
                    "severity": incident["severity"],
                    "status": incident["status"],
                    "progress": incident["progress"]
                }
                for incident in self.incidents[-5:]  # Last 5 incidents
            ],
            "metrics": self.metrics,
            "threat_intel": [
                {
                    "ioc": ti.ioc,
                    "ioc_type": ti.ioc_type,
                    "threat_level": ti.threat_level,
                    "ai_analysis": ti.ai_analysis
                }
                for ti in self.threat_intel[-10:]  # Last 10 IOCs
            ]
        }

# Flask Application
app = Flask(__name__)
app.config['SECRET_KEY'] = 'soc_glm_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize SOC Platform
soc_platform = None

@app.route('/')
def index():
    """Main dashboard"""
    return render_template('soc_dashboard.html')

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Get all alerts"""
    if soc_platform:
        return jsonify(soc_platform.get_dashboard_data()['alerts'])
    return jsonify([])

@app.route('/api/alerts', methods=['POST'])
async def create_alert():
    """Create new alert"""
    alert_data = request.json
    if soc_platform:
        alert = await soc_platform.process_alert(alert_data)
        socketio.emit('new_alert', {
            'id': alert.id,
            'severity': alert.severity,
            'source': alert.source,
            'ai_reasoning': alert.ai_reasoning
        })
        return jsonify({'status': 'success', 'alert_id': alert.id})
    return jsonify({'status': 'error'})

@app.route('/api/incidents', methods=['POST'])
async def create_incident():
    """Create new incident"""
    incident_data = request.json
    if soc_platform:
        incident = await soc_platform.create_incident(incident_data)
        socketio.emit('new_incident', {
            'id': incident['id'],
            'severity': incident['severity'],
            'status': incident['status']
        })
        return jsonify({'status': 'success', 'incident_id': incident['id']})
    return jsonify({'status': 'error'})

@app.route('/api/compliance', methods=['POST'])
async def check_compliance():
    """Check compliance"""
    event_data = request.json
    if soc_platform:
        result = await soc_platform.check_compliance(event_data)
        return jsonify(result)
    return jsonify({'status': 'error'})

@app.route('/api/dashboard')
def get_dashboard():
    """Get dashboard data"""
    if soc_platform:
        return jsonify(soc_platform.get_dashboard_data())
    return jsonify({})

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('status', {'message': 'Connected to SOC Operations Platform'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

if __name__ == '__main__':
    # Initialize SOC Platform with GLM-4.6 API key
    glm_api_key = os.getenv('GLM_API_KEY', 'your_glm_api_key_here')
    soc_platform = SOCOperationsPlatform(glm_api_key)
    
    print("🚀 Starting AI-Driven SOC Operations Platform with GLM-4.6")
    print("📊 Dashboard: http://localhost:5000")
    print("🔧 API Endpoints: /api/alerts, /api/incidents, /api/compliance")
    print("⚡ Real-time updates via WebSocket")
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)