"""
Reporting and Visualization Agent (RVA) for AI-driven SOC

Core Responsibilities:
- Generation of comprehensive incident reports
- Creation and updating of security dashboards
- Documentation of incident timelines and response actions
- Preparation of executive summaries and technical details
- Archiving of incident data for compliance and reference
"""

import os
import io
import json
import base64
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from google.cloud import pubsub_v1, storage, bigquery
from google.cloud import firestore
import jinja2
from matplotlib.dates import DateFormatter, WeekdayLocator
import matplotlib.ticker as ticker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ReportingVisualizationAgent")

class ReportingVisualizationAgent:
    """Reporting and Visualization Agent for security incidents"""
    
    def __init__(self, config_path: str = "config/rva_config.json"):
        """Initialize the Reporting and Visualization Agent
        
        Args:
            config_path: Path to the agent configuration file
        """
        self.config = self._load_config(config_path)
        
        # Initialize clients
        self.storage_client = storage.Client(project=self.config["project_id"])
        self.bigquery_client = bigquery.Client(project=self.config["project_id"])
        self.db = firestore.Client(project=self.config["project_id"])
        
        # Initialize Pub/Sub
        self.publisher = pubsub_v1.PublisherClient()
        self.subscriber = pubsub_v1.SubscriberClient()
        
        # Metrics topic for continuous learning
        self.metrics_topic_path = self.publisher.topic_path(
            self.config["project_id"], 
            self.config["metrics_topic"]
        )
        
        # Initialize Jinja2 environment for templates
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.config.get("template_dir", "templates")),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
        logger.info("Reporting and Visualization Agent initialized successfully")
    
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
                "incident_subscription": "incident-reports-sub",
                "metrics_topic": "performance-metrics",
                "incident_bucket": "security-incidents",
                "dashboard_bucket": "security-dashboards",
                "bigquery_dataset": "security_analytics",
                "report_templates": {
                    "incident": "incident_report.html.j2",
                    "executive": "executive_summary.html.j2",
                    "technical": "technical_report.html.j2",
                    "daily": "daily_summary.html.j2"
                },
                "dashboard_update_interval": 300,  # 5 minutes
                "template_dir": "templates",
                "retention_policy": {
                    "incident_reports": 365,  # days
                    "dashboards": 90,         # days
                    "raw_data": 180           # days
                }
            }
    
    def process_incident_report(self, incident_data: Dict) -> Dict:
        """Process an incident report and generate artifacts
        
        Args:
            incident_data: Incident data from the Containment and Response Agent
            
        Returns:
            Dict containing processing results
        """
        incident_id = incident_data.get("incident_id", "unknown")
        logger.info(f"Processing incident report: {incident_id}")
        
        # Store the raw incident data
        self._store_incident_data(incident_id, incident_data)
        
        # Generate reports
        reports = self._generate_incident_reports(incident_id, incident_data)
        
        # Store reports
        report_urls = self._store_reports(incident_id, reports)
        
        # Update dashboards
        self._update_dashboards(incident_data)
        
        # Send performance metrics to Continuous Learning Agent
        metrics = self._extract_performance_metrics(incident_data)
        self._send_metrics(metrics)
        
        # Return processing results
        results = {
            "incident_id": incident_id,
            "timestamp": datetime.now().isoformat(),
            "reports": report_urls,
            "metrics": metrics
        }
        
        return results
    
    def _store_incident_data(self, incident_id: str, incident_data: Dict) -> None:
        """Store the raw incident data in Cloud Storage and BigQuery
        
        Args:
            incident_id: Incident ID
            incident_data: Incident data
        """
        # Store in Cloud Storage
        bucket = self.storage_client.bucket(self.config["incident_bucket"])
        blob = bucket.blob(f"raw/{incident_id}.json")
        blob.upload_from_string(json.dumps(incident_data, indent=2), content_type="application/json")
        
        # Store in BigQuery
        table_id = f"{self.config['project_id']}.{self.config['bigquery_dataset']}.incidents"
        
        # Flatten the nested structure for BigQuery
        flattened_data = self._flatten_incident_data(incident_data)
        
        # Load into BigQuery
        errors = self.bigquery_client.insert_rows_json(table_id, [flattened_data])
        if errors:
            logger.error(f"Error inserting data into BigQuery: {errors}")
        
        # Update Firestore document
        self.db.collection("incident_reports").document(incident_id).set({
            "incident_id": incident_id,
            "created_at": datetime.now(),
            "data": incident_data,
            "reports": {},
            "archived": False
        })
    
    def _flatten_incident_data(self, incident_data: Dict) -> Dict:
        """Flatten nested incident data for BigQuery
        
        Args:
            incident_data: Nested incident data
            
        Returns:
            Flattened data suitable for BigQuery
        """
        flat_data = {
            "incident_id": incident_data.get("incident_id"),
            "timestamp": incident_data.get("timestamp"),
            "severity": incident_data.get("incident", {}).get("severity"),
            "confidence": incident_data.get("incident", {}).get("confidence"),
            "attack_type": incident_data.get("incident", {}).get("analysis", {}).get("attack_type"),
            "playbook_name": incident_data.get("playbook_name"),
            "containment_status": incident_data.get("containment_status"),
            "ticket_id": incident_data.get("ticket_id"),
            "actions_executed_count": len(incident_data.get("actions_executed", [])),
            "actions_pending_count": len(incident_data.get("actions_pending_approval", [])),
            "processed_at": datetime.now().isoformat()
        }
        
        return flat_data
    
    def _generate_incident_reports(self, incident_id: str, incident_data: Dict) -> Dict[str, str]:
        """Generate different reports for the incident
        
        Args:
            incident_id: Incident ID
            incident_data: Incident data
            
        Returns:
            Dict mapping report types to report content
        """
        reports = {}
        
        # Load templates
        templates = self.config["report_templates"]
        
        # Prepare context for templates
        context = {
            "incident": incident_data,
            "incident_id": incident_id,
            "generated_at": datetime.now().isoformat(),
            "report_id": f"RPT-{incident_id}-{int(time.time())}",
            "agent_version": "1.0.0"
        }
        
        # Generate each report type
        for report_type, template_name in templates.items():
            try:
                # Try to load the template
                template = self._load_template(template_name)
                if template:
                    # Render the template
                    reports[report_type] = template.render(**context)
                    logger.info(f"Generated {report_type} report for incident {incident_id}")
            except Exception as e:
                logger.error(f"Error generating {report_type} report: {e}")
        
        return reports
    
    def _load_template(self, template_name: str) -> Optional[jinja2.Template]:
        """Load a Jinja2 template
        
        Args:
            template_name: Name of the template
            
        Returns:
            Jinja2 Template or None if not found
        """
        try:
            return self.jinja_env.get_template(template_name)
        except jinja2.exceptions.TemplateNotFound:
            # Fall back to default templates embedded in the code
            default_templates = {
                "incident_report.html.j2": """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Incident Report: {{ incident_id }}</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 20px; }
                        h1 { color: #003366; }
                        .section { margin: 20px 0; }
                        .severity-high { color: #cc0000; }
                        .severity-medium { color: #ff9900; }
                        .severity-low { color: #009900; }
                        table { border-collapse: collapse; width: 100%; }
                        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                        th { background-color: #f2f2f2; }
                    </style>
                </head>
                <body>
                    <h1>Security Incident Report</h1>
                    <div class="section">
                        <h2>Incident Summary</h2>
                        <p><strong>Incident ID:</strong> {{ incident_id }}</p>
                        <p><strong>Timestamp:</strong> {{ incident.timestamp }}</p>
                        <p><strong>Severity:</strong> <span class="severity-{{ incident.incident.severity }}">{{ incident.incident.severity|upper }}</span></p>
                        <p><strong>Attack Type:</strong> {{ incident.incident.analysis.attack_type }}</p>
                        <p><strong>Confidence:</strong> {{ "%.2f"|format(incident.incident.confidence * 100) }}%</p>
                        <p><strong>Containment Status:</strong> {{ incident.containment_status|upper }}</p>
                        {% if incident.ticket_id %}
                        <p><strong>Ticket ID:</strong> <a href="{{ incident.ticket_url }}">{{ incident.ticket_id }}</a></p>
                        {% endif %}
                    </div>
                    
                    <div class="section">
                        <h2>Analysis</h2>
                        <p>{{ incident.incident.analysis.analysis_summary }}</p>
                        
                        <h3>Indicators of Compromise</h3>
                        <ul>
                        {% for indicator in incident.incident.analysis.indicators %}
                            <li>{{ indicator }}</li>
                        {% endfor %}
                        </ul>
                    </div>
                    
                    <div class="section">
                        <h2>Actions Taken</h2>
                        <table>
                            <tr>
                                <th>Action</th>
                                <th>Time</th>
                                <th>Status</th>
                                <th>Details</th>
                            </tr>
                            {% for action in incident.actions_executed %}
                            <tr>
                                <td>{{ action.action }}</td>
                                <td>{{ action.timestamp }}</td>
                                <td>{{ action.status }}</td>
                                <td>{{ action.result|tojson }}</td>
                            </tr>
                            {% endfor %}
                        </table>
                    </div>
                    
                    {% if incident.actions_pending_approval %}
                    <div class="section">
                        <h2>Pending Actions</h2>
                        <table>
                            <tr>
                                <th>Action</th>
                                <th>Risk Level</th>
                                <th>Parameters</th>
                            </tr>
                            {% for action in incident.actions_pending_approval %}
                            <tr>
                                <td>{{ action.action }}</td>
                                <td>{{ action.risk_level }}</td>
                                <td>{{ action.parameters|tojson }}</td>
                            </tr>
                            {% endfor %}
                        </table>
                    </div>
                    {% endif %}
                    
                    <div class="section">
                        <h2>Technical Details</h2>
                        <pre>{{ incident.incident.log_data|tojson(indent=2) }}</pre>
                    </div>
                    
                    <div class="footer">
                        <p>Generated at: {{ generated_at }} | Report ID: {{ report_id }}</p>
                    </div>
                </body>
                </html>
                """,
                
                "executive_summary.html.j2": """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Executive Summary: {{ incident_id }}</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 20px; }
                        h1 { color: #003366; }
                        .section { margin: 20px 0; }
                        .severity-high { color: #cc0000; }
                        .severity-medium { color: #ff9900; }
                        .severity-low { color: #009900; }
                    </style>
                </head>
                <body>
                    <h1>Security Incident: Executive Summary</h1>
                    <div class="section">
                        <h2>Incident Overview</h2>
                        <p><strong>Incident ID:</strong> {{ incident_id }}</p>
                        <p><strong>Date/Time:</strong> {{ incident.timestamp }}</p>
                        <p><strong>Severity:</strong> <span class="severity-{{ incident.incident.severity }}">{{ incident.incident.severity|upper }}</span></p>
                        <p><strong>Type:</strong> {{ incident.incident.analysis.attack_type }}</p>
                        <p><strong>Status:</strong> {{ incident.containment_status|upper }}</p>
                    </div>
                    
                    <div class="section">
                        <h2>Summary</h2>
                        <p>{{ incident.incident.analysis.analysis_summary }}</p>
                    </div>
                    
                    <div class="section">
                        <h2>Business Impact</h2>
                        <p>Based on the analysis, this incident had the following business impact:</p>
                        <ul>
                            {% if incident.incident.severity == "high" %}
                            <li>Potential data breach risk</li>
                            <li>Possible regulatory implications</li>
                            <li>Service disruption for affected systems</li>
                            {% elif incident.incident.severity == "medium" %}
                            <li>Limited access to sensitive data</li>
                            <li>Minor service degradation</li>
                            {% else %}
                            <li>Minimal business impact</li>
                            <li>No service disruptions</li>
                            {% endif %}
                        </ul>
                    </div>
                    
                    <div class="section">
                        <h2>Response Summary</h2>
                        <p><strong>Actions Taken:</strong> {{ incident.actions_executed|length }}</p>
                        <p><strong>Pending Actions:</strong> {{ incident.actions_pending_approval|length }}</p>
                        {% if incident.ticket_id %}
                        <p><strong>Ticket:</strong> <a href="{{ incident.ticket_url }}">{{ incident.ticket_id }}</a></p>
                        {% endif %}
                    </div>
                    
                    <div class="footer">
                        <p>Generated at: {{ generated_at }} | Report ID: {{ report_id }}</p>
                    </div>
                </body>
                </html>
                """,
                
                "technical_report.html.j2": """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Technical Report: {{ incident_id }}</title>
                    <style>
                        body { font-family: monospace; margin: 20px; }
                        h1 { color: #003366; }
                        .section { margin: 20px 0; }
                        pre { background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow: auto; }
                        table { border-collapse: collapse; width: 100%; }
                        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                        th { background-color: #f2f2f2; }
                    </style>
                </head>
                <body>
                    <h1>Security Incident: Technical Report</h1>
                    <div class="section">
                        <h2>Incident Details</h2>
                        <pre>
Incident ID: {{ incident_id }}
Timestamp: {{ incident.timestamp }}
Severity: {{ incident.incident.severity }}
Confidence: {{ "%.2f"|format(incident.incident.confidence * 100) }}%
Attack Type: {{ incident.incident.analysis.attack_type }}
Containment: {{ incident.containment_status }}
                        </pre>
                    </div>
                    
                    <div class="section">
                        <h2>Technical Analysis</h2>
                        <pre>{{ incident.incident.analysis.analysis_summary }}</pre>
                        
                        <h3>Indicators of Compromise</h3>
                        <pre>{% for indicator in incident.incident.analysis.indicators %}
- {{ indicator }}{% endfor %}
                        </pre>
                    </div>
                    
                    <div class="section">
                        <h2>Log Data</h2>
                        <pre>{{ incident.incident.log_data|tojson(indent=2) }}</pre>
                    </div>
                    
                    <div class="section">
                        <h2>Response Actions</h2>
                        <table>
                            <tr>
                                <th>Action</th>
                                <th>Time</th>
                                <th>Status</th>
                                <th>Result</th>
                            </tr>
                            {% for action in incident.actions_executed %}
                            <tr>
                                <td>{{ action.action }}</td>
                                <td>{{ action.timestamp }}</td>
                                <td>{{ action.status }}</td>
                                <td><pre>{{ action.result|tojson(indent=2) }}</pre></td>
                            </tr>
                            {% endfor %}
                        </table>
                    </div>
                    
                    <div class="section">
                        <h2>Raw Incident Data</h2>
                        <pre>{{ incident|tojson(indent=2) }}</pre>
                    </div>
                    
                    <div class="footer">
                        <p>Generated at: {{ generated_at }} | Report ID: {{ report_id }}</p>
                    </div>
                </body>
                </html>
                """,
                
                "daily_summary.html.j2": """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Daily Security Summary</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 20px; }
                        h1 { color: #003366; }
                        .section { margin: 20px 0; }
                        .chart { width: 100%; height: 300px; margin: 20px 0; }
                        table { border-collapse: collapse; width: 100%; }
                        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                        th { background-color: #f2f2f2; }
                    </style>
                </head>
                <body>
                    <h1>Daily Security Summary</h1>
                    <div class="section">
                        <h2>Summary for {{ day_date }}</h2>
                        <p><strong>Total Incidents:</strong> {{ total_incidents }}</p>
                        <p><strong>High Severity:</strong> {{ high_severity_incidents }}</p>
                        <p><strong>Medium Severity:</strong> {{ medium_severity_incidents }}</p>
                        <p><strong>Low Severity:</strong> {{ low_severity_incidents }}</p>
                        <p><strong>Containment Rate:</strong> {{ containment_rate }}%</p>
                    </div>
                    
                    <div class="section">
                        <h2>Incident Distribution</h2>
                        <div class="chart">
                            <!-- Chart would be embedded here -->
                            <img src="data:image/png;base64,{{ incident_distribution_chart }}" alt="Incident Distribution" />
                        </div>
                    </div>
                    
                    <div class="section">
                        <h2>Attack Types</h2>
                        <div class="chart">
                            <!-- Chart would be embedded here -->
                            <img src="data:image/png;base64,{{ attack_types_chart }}" alt="Attack Types" />
                        </div>
                    </div>
                    
                    <div class="section">
                        <h2>Recent Incidents</h2>
                        <table>
                            <tr>
                                <th>ID</th>
                                <th>Time</th>
                                <th>Severity</th>
                                <th>Attack Type</th>
                                <th>Status</th>
                            </tr>
                            {% for incident in recent_incidents %}
                            <tr>
                                <td>{{ incident.id }}</td>
                                <td>{{ incident.time }}</td>
                                <td>{{ incident.severity }}</td>
                                <td>{{ incident.attack_type }}</td>
                                <td>{{ incident.status }}</td>
                            </tr>
                            {% endfor %}
                        </table>
                    </div>
                    
                    <div class="footer">
                        <p>Generated at: {{ generated_at }} | Report ID: {{ report_id }}</p>
                    </div>
                </body>
                </html>
                """
            }
            
            if template_name in default_templates:
                return jinja2.Template(default_templates[template_name])
            else:
                logger.error(f"Template not found: {template_name}")
                return None
    
    def _store_reports(self, incident_id: str, reports: Dict[str, str]) -> Dict[str, str]:
        """Store the generated reports in Cloud Storage
        
        Args:
            incident_id: Incident ID
            reports: Dict mapping report types to report content
            
        Returns:
            Dict mapping report types to report URLs
        """
        report_urls = {}
        bucket = self.storage_client.bucket(self.config["incident_bucket"])
        
        # Store each report
        for report_type, content in reports.items():
            timestamp = int(time.time())
            blob_path = f"reports/{incident_id}/{report_type}_{timestamp}.html"
            blob = bucket.blob(blob_path)
            blob.upload_from_string(content, content_type="text/html")
            
            # Make the blob publicly readable if configured
            if self.config.get("make_reports_public", False):
                blob.make_public()
            
            # Get the URL
            report_urls[report_type] = blob.public_url if self.config.get("make_reports_public", False) else blob.self_link
            
            logger.info(f"Stored {report_type} report for incident {incident_id}")
        
        # Update Firestore document
        self.db.collection("incident_reports").document(incident_id).update({
            "reports": report_urls,
            "updated_at": datetime.now()
        })
        
        return report_urls
    
    def _update_dashboards(self, incident_data: Dict) -> None:
        """Update security dashboards with the latest incident data
        
        Args:
            incident_data: Incident data
        """
        # This would be implemented with a more sophisticated dashboard system
        # For now, this is a simplified implementation
        
        # Get recent incidents for dashboards
        query = self.db.collection("incident_reports").order_by(
            "created_at", direction=firestore.Query.DESCENDING
        ).limit(100)
        
        docs = query.stream()
        incidents = [doc.to_dict() for doc in docs]
        
        # Generate dashboards only if enough time has passed since the last update
        last_update_key = "dashboard_last_update"
        last_update = self.db.collection("agent_state").document("rva").get()
        
        current_time = datetime.now()
        should_update = True
        
        if last_update.exists:
            last_update_time = last_update.get(last_update_key)
            if last_update_time:
                elapsed = current_time - last_update_time.timestamp()
                if elapsed.total_seconds() < self.config["dashboard_update_interval"]:
                    should_update = False
        
        if should_update:
            # Update daily dashboard
            self._generate_daily_dashboard(incidents)
            
            # Update weekly dashboard
            self._generate_weekly_dashboard(incidents)
            
            # Update trend dashboard
            self._generate_trend_dashboard(incidents)
            
            # Record update time
            self.db.collection("agent_state").document("rva").set({
                last_update_key: current_time
            }, merge=True)
    
    def _generate_daily_dashboard(self, incidents: List[Dict]) -> None:
        """Generate daily dashboard
        
        Args:
            incidents: List of recent incidents
        """
        # Calculate today's date
        today = datetime.now().date()
        today_str = today.strftime("%Y-%m-%d")
        
        # Filter incidents for today
        today_incidents = []
        for incident in incidents:
            created_at = incident.get("created_at")
            if created_at and created_at.date() == today:
                today_incidents.append(incident)
        
        # Generate charts and tables
        charts = self._generate_dashboard_charts(today_incidents, "daily")
        
        # Calculate metrics
        total_incidents = len(today_incidents)
        high_severity = sum(1 for inc in today_incidents if inc.get("data", {}).get("incident", {}).get("severity") == "high")
        medium_severity = sum(1 for inc in today_incidents if inc.get("data", {}).get("incident", {}).get("severity") == "medium")
        low_severity = sum(1 for inc in today_incidents if inc.get("data", {}).get("incident", {}).get("severity") == "low")
        
        contained = sum(1 for inc in today_incidents if inc.get("data", {}).get("containment_status") in ["complete", "partial"])
        containment_rate = (contained / total_incidents * 100) if total_incidents > 0 else 0
        
        # Prepare dashboard data
        dashboard_data = {
            "day_date": today_str,
            "total_incidents": total_incidents,
            "high_severity_incidents": high_severity,
            "medium_severity_incidents": medium_severity,
            "low_severity_incidents": low_severity,
            "containment_rate": round(containment_rate, 1),
            "generated_at": datetime.now().isoformat(),
            "report_id": f"DASH-DAILY-{int(time.time())}",
            "incident_distribution_chart": charts.get("severity_distribution"),
            "attack_types_chart": charts.get("attack_types"),
            "recent_incidents": self._format_recent_incidents(today_incidents)
        }
        
        # Generate the dashboard using the daily summary template
        template = self._load_template("daily_summary.html.j2")
        if template:
            dashboard_html = template.render(**dashboard_data)
            
            # Store the dashboard
            bucket = self.storage_client.bucket(self.config["dashboard_bucket"])
            blob = bucket.blob(f"daily/{today_str}.html")
            blob.upload_from_string(dashboard_html, content_type="text/html")
            
            # Also store as latest
            latest_blob = bucket.blob("daily/latest.html")
            latest_blob.upload_from_string(dashboard_html, content_type="text/html")
            
            logger.info("Daily dashboard updated")
    
    def _generate_weekly_dashboard(self, incidents: List[Dict]) -> None:
        """Generate weekly dashboard
        
        Args:
            incidents: List of recent incidents
        """
        # Calculate current week
        today = datetime.now().date()
        # Find the Monday of the current week
        monday = today - timedelta(days=today.weekday())
        week_start = monday
        week_end = monday + timedelta(days=6)  # Sunday
        
        week_str = f"{week_start.strftime('%Y-%m-%d')} to {week_end.strftime('%Y-%m-%d')}"
        
        # Filter incidents for this week
        week_incidents = []
        for incident in incidents:
            created_at = incident.get("created_at")
            if created_at and week_start <= created_at.date() <= week_end:
                week_incidents.append(incident)
        
        # Generate charts
        charts = self._generate_dashboard_charts(week_incidents, "weekly")
        
        # Calculate metrics
        total_incidents = len(week_incidents)
        high_severity = sum(1 for inc in week_incidents if inc.get("data", {}).get("incident", {}).get("severity") == "high")
        medium_severity = sum(1 for inc in week_incidents if inc.get("data", {}).get("incident", {}).get("severity") == "medium")
        low_severity = sum(1 for inc in week_incidents if inc.get("data", {}).get("incident", {}).get("severity") == "low")
        
        contained = sum(1 for inc in week_incidents if inc.get("data", {}).get("containment_status") in ["complete", "partial"])
        containment_rate = (contained / total_incidents * 100) if total_incidents > 0 else 0
        
        # Group by day
        daily_counts = {}
        for incident in week_incidents:
            created_at = incident.get("created_at")
            if created_at:
                day_str = created_at.date().strftime("%Y-%m-%d")
                if day_str not in daily_counts:
                    daily_counts[day_str] = 0
                daily_counts[day_str] += 1
        
        # Prepare dashboard data
        dashboard_data = {
            "week_date": week_str,
            "total_incidents": total_incidents,
            "high_severity_incidents": high_severity,
            "medium_severity_incidents": medium_severity,
            "low_severity_incidents": low_severity,
            "containment_rate": round(containment_rate, 1),
            "generated_at": datetime.now().isoformat(),
            "report_id": f"DASH-WEEKLY-{int(time.time())}",
            "incident_distribution_chart": charts.get("severity_distribution"),
            "attack_types_chart": charts.get("attack_types"),
            "daily_trend_chart": charts.get("daily_trend"),
            "recent_incidents": self._format_recent_incidents(week_incidents)
        }
        
        # Generate weekly dashboard HTML
        # This would typically use a weekly template
        # For simplicity, we'll adapt the daily template
        template = self._load_template("daily_summary.html.j2")
        if template:
            dashboard_html = template.render(**dashboard_data)
            
            # Store the dashboard
            bucket = self.storage_client.bucket(self.config["dashboard_bucket"])
            blob = bucket.blob(f"weekly/{week_start.strftime('%Y-%m-%d')}.html")
            blob.upload_from_string(dashboard_html, content_type="text/html")
            
            # Also store as latest
            latest_blob = bucket.blob("weekly/latest.html")
            latest_blob.upload_from_string(dashboard_html, content_type="text/html")
            
            logger.info("Weekly dashboard updated")
    
    def _generate_trend_dashboard(self, incidents: List[Dict]) -> None:
        """Generate trend dashboard showing longer-term trends
        
        Args:
            incidents: List of recent incidents
        """
        # This would be implemented with a more sophisticated dashboard
        # For now, we'll create a simple trend dashboard
        today = datetime.now().date()
        
        # Look back 30 days
        start_date = today - timedelta(days=30)
        
        # Filter incidents for the last 30 days
        period_incidents = []
        for incident in incidents:
            created_at = incident.get("created_at")
            if created_at and start_date <= created_at.date() <= today:
                period_incidents.append(incident)
        
        # Generate trend charts
        charts = self._generate_dashboard_charts(period_incidents, "trend")
        
        # Calculate metrics for the period
        total_incidents = len(period_incidents)
        
        # Calculate daily averages
        days_in_period = (today - start_date).days + 1
        avg_daily_incidents = total_incidents / days_in_period if days_in_period > 0 else 0
        
        # Prepare dashboard data
        dashboard_data = {
            "period_start": start_date.strftime("%Y-%m-%d"),
            "period_end": today.strftime("%Y-%m-%d"),
            "total_incidents": total_incidents,
            "avg_daily_incidents": round(avg_daily_incidents, 1),
            "generated_at": datetime.now().isoformat(),
            "report_id": f"DASH-TREND-{int(time.time())}",
            "trend_chart": charts.get("trend"),
            "severity_trend_chart": charts.get("severity_trend"),
            "attack_type_trend_chart": charts.get("attack_type_trend")
        }
        
        # For now, we'll just log this
        logger.info(f"Trend dashboard would be generated with {total_incidents} incidents")
        
        # In a real implementation, we would render a template and store the dashboard
    
    def _generate_dashboard_charts(self, incidents: List[Dict], dashboard_type: str) -> Dict[str, str]:
        """Generate charts for dashboards
        
        Args:
            incidents: List of incidents
            dashboard_type: Type of dashboard (daily, weekly, trend)
            
        Returns:
            Dict mapping chart names to base64-encoded PNG images
        """
        charts = {}
        
        # Set Seaborn style
        sns.set_style("whitegrid")
        
        if not incidents:
            # If no incidents, return empty charts
            return charts
        
        # Prepare data for charts
        df = self._prepare_incident_dataframe(incidents)
        
        # ---------------------------------------------------------------
        # Severity distribution chart
        fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
        
        # Count by severity
        severity_counts = df['severity'].value_counts()
        
        # Map colors
        colors = {'high': '#cc0000', 'medium': '#ff9900', 'low': '#009900'}
        
        # Generate bar chart
        severity_bars = ax.bar(severity_counts.index, severity_counts.values, 
                              color=[colors.get(s, '#666666') for s in severity_counts.index])
        
        # Add labels
        for bar in severity_bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.0f}', ha='center', va='bottom')
        
        ax.set_title('Incident Severity Distribution')
        ax.set_xlabel('Severity')
        ax.set_ylabel('Number of Incidents')
        
        # Save chart
        charts['severity_distribution'] = self._fig_to_base64(fig)
        plt.close(fig)
        
        # ---------------------------------------------------------------
        # Attack types chart
        fig, ax = plt.figure(figsize=(12, 6)), plt.gca()
        
        # Count by attack type
        attack_counts = df['attack_type'].value_counts().head(10)  # Top 10
        
        # Generate horizontal bar chart for better label readability
        attack_bars = ax.barh(attack_counts.index, attack_counts.values, color='#3366cc')
        
        # Add labels
        for bar in attack_bars:
            width = bar.get_width()
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2.,
                    f'{width:.0f}', ha='left', va='center')
        
        ax.set_title('Top Attack Types')
        ax.set_xlabel('Number of Incidents')
        plt.tight_layout()
        
        # Save chart
        charts['attack_types'] = self._fig_to_base64(fig)
        plt.close(fig)
        
        # ---------------------------------------------------------------
        # Additional charts based on dashboard type
        if dashboard_type in ['weekly', 'trend']:
            # Daily trend chart
            fig, ax = plt.figure(figsize=(12, 6)), plt.gca()
            
            # Group by date
            df['date'] = df['timestamp'].dt.date
            daily_counts = df.groupby('date').size()
            
            # Create line chart
            ax.plot(daily_counts.index, daily_counts.values, marker='o', linestyle='-', color='#3366cc')
            
            # Format x-axis
            ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            
            ax.set_title('Incident Trend')
            ax.set_xlabel('Date')
            ax.set_ylabel('Number of Incidents')
            plt.tight_layout()
            
            # Save chart
            charts['daily_trend'] = self._fig_to_base64(fig)
            plt.close(fig)
        
        if dashboard_type == 'trend':
            # Severity trend over time
            fig, ax = plt.figure(figsize=(12, 6)), plt.gca()
            
            # Group by date and severity
            severity_trend = df.groupby(['date', 'severity']).size().unstack().fillna(0)
            
            # Create stacked area chart
            severity_trend.plot(kind='area', stacked=True, ax=ax, 
                               color=[colors.get(s, '#666666') for s in severity_trend.columns])
            
            # Format x-axis
            ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            
            ax.set_title('Severity Trend')
            ax.set_xlabel('Date')
            ax.set_ylabel('Number of Incidents')
            plt.tight_layout()
            
            # Save chart
            charts['severity_trend'] = self._fig_to_base64(fig)
            plt.close(fig)
            
            # Attack type trend
            # Only include top 5 attack types for readability
            top_types = df['attack_type'].value_counts().head(5).index
            df_top_types = df[df['attack_type'].isin(top_types)]
            
            if not df_top_types.empty:
                fig, ax = plt.figure(figsize=(12, 6)), plt.gca()
                
                # Group by date and attack type
                type_trend = df_top_types.groupby(['date', 'attack_type']).size().unstack().fillna(0)
                
                # Create line chart
                type_trend.plot(kind='line', marker='o', ax=ax)
                
                # Format x-axis
                ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
                plt.xticks(rotation=45)
                
                ax.set_title('Attack Type Trend (Top 5)')
                ax.set_xlabel('Date')
                ax.set_ylabel('Number of Incidents')
                plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
                plt.tight_layout()
                
                # Save chart
                charts['attack_type_trend'] = self._fig_to_base64(fig)
                plt.close(fig)
        
        return charts
    
    def _prepare_incident_dataframe(self, incidents: List[Dict]) -> pd.DataFrame:
        """Prepare a pandas DataFrame from incident data
        
        Args:
            incidents: List of incidents
            
        Returns:
            DataFrame with incident data
        """
        # Extract key fields from incidents
        data = []
        for incident in incidents:
            incident_data = incident.get("data", {})
            
            # Extract timestamp
            timestamp = None
            if "timestamp" in incident_data:
                try:
                    timestamp = datetime.fromisoformat(incident_data["timestamp"].replace('Z', '+00:00'))
                except:
                    # Fall back to created_at from the document
                    timestamp = incident.get("created_at")
            else:
                timestamp = incident.get("created_at")
            
            # Skip if no timestamp
            if not timestamp:
                continue
            
            # Extract other fields
            incident_obj = incident_data.get("incident", {})
            analysis = incident_obj.get("analysis", {})
            
            data.append({
                "incident_id": incident_data.get("incident_id"),
                "timestamp": timestamp,
                "severity": incident_obj.get("severity", "unknown"),
                "attack_type": analysis.get("attack_type", "unknown"),
                "confidence": incident_obj.get("confidence", 0.0),
                "containment_status": incident_data.get("containment_status", "unknown")
            })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Fill missing values
        df["severity"] = df["severity"].fillna("unknown")
        df["attack_type"] = df["attack_type"].fillna("unknown")
        
        return df
    
    def _fig_to_base64(self, fig: plt.Figure) -> str:
        """Convert a matplotlib figure to base64-encoded PNG
        
        Args:
            fig: Matplotlib figure
            
        Returns:
            Base64-encoded PNG
        """
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        return img_str
    
    def _format_recent_incidents(self, incidents: List[Dict]) -> List[Dict]:
        """Format recent incidents for display in dashboards
        
        Args:
            incidents: List of incidents
            
        Returns:
            List of formatted incidents
        """
        formatted = []
        for incident in incidents:
            incident_data = incident.get("data", {})
            incident_obj = incident_data.get("incident", {})
            
            formatted.append({
                "id": incident_data.get("incident_id", "unknown"),
                "time": incident_data.get("timestamp", ""),
                "severity": incident_obj.get("severity", "unknown"),
                "attack_type": incident_obj.get("analysis", {}).get("attack_type", "unknown"),
                "status": incident_data.get("containment_status", "unknown")
            })
        
        # Sort by time (descending)
        formatted.sort(key=lambda x: x.get("time", ""), reverse=True)
        
        # Return at most 10 incidents
        return formatted[:10]
    
    def _extract_performance_metrics(self, incident_data: Dict) -> Dict:
        """Extract performance metrics from incident data for Continuous Learning
        
        Args:
            incident_data: Incident data
            
        Returns:
            Dict containing performance metrics
        """
        # Extract metrics relevant for the Continuous Learning Agent
        metrics = {
            "incident_id": incident_data.get("incident_id"),
            "timestamp": datetime.now().isoformat(),
            "detection": {
                "original_alert_id": incident_data.get("incident", {}).get("alert_id"),
                "confidence": incident_data.get("incident", {}).get("confidence", 0.0),
                "severity": incident_data.get("incident", {}).get("severity")
            },
            "analysis": {
                "attack_type": incident_data.get("incident", {}).get("analysis", {}).get("attack_type"),
                "is_true_positive": True  # If it reached CRA, it's a true positive
            },
            "response": {
                "containment_status": incident_data.get("containment_status"),
                "actions_executed": len(incident_data.get("actions_executed", [])),
                "actions_pending": len(incident_data.get("actions_pending_approval", [])),
                "response_time": None  # Would calculate time from detection to containment
            }
        }
        
        # Calculate response time if possible
        original_timestamp = incident_data.get("incident", {}).get("timestamp")
        if original_timestamp:
            try:
                detection_time = datetime.fromisoformat(original_timestamp.replace('Z', '+00:00'))
                action_times = []
                
                for action in incident_data.get("actions_executed", []):
                    if "timestamp" in action:
                        try:
                            action_time = datetime.fromisoformat(action["timestamp"].replace('Z', '+00:00'))
                            action_times.append(action_time)
                        except:
                            pass
                
                if action_times:
                    latest_action = max(action_times)
                    response_time = (latest_action - detection_time).total_seconds()
                    metrics["response"]["response_time"] = response_time
            except:
                logger.warning(f"Could not calculate response time for incident {incident_data.get('incident_id')}")
        
        return metrics
    
    def _send_metrics(self, metrics: Dict) -> None:
        """Send performance metrics to the Continuous Learning Agent
        
        Args:
            metrics: Performance metrics
        """
        try:
            message_data = json.dumps(metrics).encode("utf-8")
            future = self.publisher.publish(self.metrics_topic_path, message_data)
            message_id = future.result()
            logger.info(f"Metrics published with message ID: {message_id}")
        except Exception as e:
            logger.error(f"Error sending metrics: {e}")
    
    def process_message(self, message) -> None:
        """Process a message from the incident report subscription
        
        Args:
            message: Pub/Sub message
        """
        try:
            # Decode the message data
            incident_data = json.loads(message.data.decode("utf-8"))
            
            # Process the incident report
            result = self.process_incident_report(incident_data)
            
            # Acknowledge the message
            message.ack()
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            # Don't acknowledge to allow retry
    
    def listen_for_reports(self):
        """Listen for incident reports from the subscription"""
        subscription_path = self.subscriber.subscription_path(
            self.config["project_id"],
            self.config["incident_subscription"]
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
    
    def generate_daily_summary(self, date_str: Optional[str] = None) -> str:
        """Generate a daily summary report
        
        Args:
            date_str: Date string in YYYY-MM-DD format, or None for today
            
        Returns:
            URL of the generated report
        """
        # Get the date
        if date_str:
            try:
                report_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            except ValueError:
                raise ValueError(f"Invalid date format: {date_str}. Use YYYY-MM-DD.")
        else:
            report_date = datetime.now().date()
        
        # Query incidents for the date
        start_time = datetime.combine(report_date, datetime.min.time())
        end_time = datetime.combine(report_date, datetime.max.time())
        
        query = self.db.collection("incident_reports").where(
            "created_at", ">=", start_time
        ).where(
            "created_at", "<=", end_time
        )
        
        docs = query.stream()
        incidents = [doc.to_dict() for doc in docs]
        
        if not incidents:
            logger.info(f"No incidents found for date: {report_date}")
            return None
        
        # Generate charts
        charts = self._generate_dashboard_charts(incidents, "daily")
        
        # Calculate metrics
        total_incidents = len(incidents)
        high_severity = sum(1 for inc in incidents if inc.get("data", {}).get("incident", {}).get("severity") == "high")
        medium_severity = sum(1 for inc in incidents if inc.get("data", {}).get("incident", {}).get("severity") == "medium")
        low_severity = sum(1 for inc in incidents if inc.get("data", {}).get("incident", {}).get("severity") == "low")
        
        contained = sum(1 for inc in incidents if inc.get("data", {}).get("containment_status") in ["complete", "partial"])
        containment_rate = (contained / total_incidents * 100) if total_incidents > 0 else 0
        
        # Prepare report data
        report_data = {
            "day_date": report_date.strftime("%Y-%m-%d"),
            "total_incidents": total_incidents,
            "high_severity_incidents": high_severity,
            "medium_severity_incidents": medium_severity,
            "low_severity_incidents": low_severity,
            "containment_rate": round(containment_rate, 1),
            "generated_at": datetime.now().isoformat(),
            "report_id": f"SUMMARY-DAILY-{report_date.strftime('%Y%m%d')}-{int(time.time())}",
            "incident_distribution_chart": charts.get("severity_distribution"),
            "attack_types_chart": charts.get("attack_types"),
            "recent_incidents": self._format_recent_incidents(incidents)
        }
        
        # Generate the report
        template = self._load_template("daily_summary.html.j2")
        if template:
            report_html = template.render(**report_data)
            
            # Store the report
            bucket = self.storage_client.bucket(self.config["incident_bucket"])
            blob_path = f"summary/daily/{report_date.strftime('%Y-%m-%d')}.html"
            blob = bucket.blob(blob_path)
            blob.upload_from_string(report_html, content_type="text/html")
            
            # Make the blob publicly readable if configured
            if self.config.get("make_reports_public", False):
                blob.make_public()
            
            # Get the URL
            report_url = blob.public_url if self.config.get("make_reports_public", False) else blob.self_link
            
            logger.info(f"Generated daily summary for {report_date}")
            return report_url
        
        return None

# Cloud Function entry point for processing incident reports
def process_report_function(event, context):
    """Cloud Function entry point for processing incident reports
    
    Args:
        event: Pub/Sub event
        context: Event context
    """
    agent = ReportingVisualizationAgent()
    
    # Decode the incident data
    incident_data = json.loads(base64.b64decode(event["data"]).decode("utf-8"))
    
    # Process the incident report
    result = agent.process_incident_report(incident_data)
    
    return "Incident report processed successfully"

# Cloud Function entry point for generating daily summary
def generate_daily_summary_function(request):
    """Cloud Function entry point for generating daily summary
    
    Args:
        request: HTTP request
    """
    request_json = request.get_json(silent=True)
    
    # Get date from request or use today
    date_str = None
    if request_json and "date" in request_json:
        date_str = request_json["date"]
    
    agent = ReportingVisualizationAgent()
    report_url = agent.generate_daily_summary(date_str)
    
    if report_url:
        return {"status": "success", "report_url": report_url}
    else:
        return {"status": "error", "message": "Could not generate report"}

if __name__ == "__main__":
    # For testing
    agent = ReportingVisualizationAgent()
    
    # Test incident data
    test_incident = {
        "incident_id": "INC-20230503123456",
        "timestamp": "2023-05-03T12:34:56Z",
        "incident": {
            "alert_id": "ADA-20230503123456-7890",
            "timestamp": "2023-05-03T12:34:56Z",
            "severity": "high",
            "confidence": 0.85,
            "analysis": {
                "is_true_positive": True,
                "severity": "high",
                "confidence": 0.85,
                "attack_type": "brute_force",
                "indicators": ["Repeated failed login attempts", "Unusual source IP"],
                "recommended_actions": ["Block source IP", "Reset affected user credentials"],
                "analysis_summary": "Multiple failed login attempts from unusual IP address"
            },
            "log_data": {
                "log_id": "log-123456",
                "timestamp": "2023-05-03T12:34:56Z",
                "src_ip": "10.0.0.1",
                "dst_ip": "203.0.113.1",
                "target_user": "admin",
                "src_hostname": "workstation-123"
            }
        },
        "playbook_name": "brute_force",
        "actions_executed": [
            {
                "action": "block_ip",
                "parameters": {"ip_field": "src_ip"},
                "risk_level": "medium",
                "timestamp": "2023-05-03T12:35:10Z",
                "status": "completed",
                "result": {"ip_blocked": "10.0.0.1", "firewall_type": "gcp", "rule_id": "block-ip-10-0-0-1-1683115510"}
            },
            {
                "action": "create_ticket",
                "parameters": {},
                "risk_level": "low",
                "timestamp": "2023-05-03T12:35:15Z",
                "status": "completed",
                "result": {"ticket_id": "SEC-123", "ticket_url": "https://example.atlassian.net/browse/SEC-123"}
            }
        ],
        "actions_pending_approval": [
            {
                "action": "reset_credentials",
                "parameters": {"user_field": "target_user"},
                "risk_level": "high"
            }
        ],
        "ticket_id": "SEC-123",
        "ticket_url": "https://example.atlassian.net/browse/SEC-123",
        "containment_status": "partial"
    }
    
    # Process the test incident
    result = agent.process_incident_report(test_incident)
    
    # Print the result
    print(json.dumps(result, indent=2))
