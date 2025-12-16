#!/usr/bin/env python3
"""
ADA Monitoring Dashboard with BigQuery Integration
Displays real-time metrics from processed_alerts table with proper filtering
"""

from flask import Flask, render_template, jsonify
from flask_cors import CORS
from google.cloud import bigquery
import os
from datetime import datetime, timedelta
import json

app = Flask(__name__)
CORS(app)

# BigQuery configuration
PROJECT_ID = os.getenv("BIGQUERY_PROJECT_ID", "chronicle-dev-2be9")
DATASET_ID = os.getenv("BIGQUERY_DATASET_ID", "soc_data")
PROCESSED_ALERTS_TABLE = os.getenv("BIGQUERY_PROCESSED_ALERTS_TABLE_ID", "processed_alerts")

def get_bigquery_client():
    """Initialize BigQuery client"""
    return bigquery.Client(project=PROJECT_ID)

def get_processed_alerts_metrics():
    """
    Query BigQuery for processed alerts metrics with proper filtering
    to exclude invalid/NULL records
    """
    client = get_bigquery_client()
    
    # Query to get metrics with filtering for valid records only
    query = f"""
    SELECT 
        COUNT(*) as total_processed,
        COUNT(CASE WHEN classification = 'anomaly' THEN 1 END) as anomalies_detected,
        COUNT(CASE WHEN classification = 'benign' THEN 1 END) as benign_alerts,
        AVG(confidence_score) as avg_confidence,
        MAX(confidence_score) as max_confidence,
        MIN(confidence_score) as min_confidence,
        COUNT(CASE WHEN confidence_score > 0.5 THEN 1 END) as high_confidence_alerts
    FROM `{PROJECT_ID}.{DATASET_ID}.{PROCESSED_ALERTS_TABLE}`
    WHERE 
        alert_id IS NOT NULL 
        AND alert_id != 'NULL' 
        AND alert_id != ''
        AND confidence_score IS NOT NULL
        AND confidence_score > 0
        AND classification IS NOT NULL
        AND classification != ''
        AND timestamp IS NOT NULL
        AND timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
    """
    
    try:
        query_job = client.query(query)
        results = query_job.result()
        
        for row in results:
            return {
                'total_processed': row.total_processed or 0,
                'anomalies_detected': row.anomalies_detected or 0,
                'benign_alerts': row.benign_alerts or 0,
                'avg_confidence': round(float(row.avg_confidence or 0), 3),
                'max_confidence': round(float(row.max_confidence or 0), 3),
                'min_confidence': round(float(row.min_confidence or 0), 3),
                'high_confidence_alerts': row.high_confidence_alerts or 0
            }
    except Exception as e:
        print(f"Error querying BigQuery: {e}")
        return {
            'total_processed': 0,
            'anomalies_detected': 0,
            'benign_alerts': 0,
            'avg_confidence': 0,
            'max_confidence': 0,
            'min_confidence': 0,
            'high_confidence_alerts': 0,
            'error': str(e)
        }

def get_recent_alerts():
    """Get recent processed alerts for display"""
    client = get_bigquery_client()
    
    query = f"""
    SELECT 
        alert_id,
        timestamp,
        confidence_score,
        classification,
        SUBSTR(raw_alert, 1, 200) as alert_preview
    FROM `{PROJECT_ID}.{DATASET_ID}.{PROCESSED_ALERTS_TABLE}`
    WHERE 
        alert_id IS NOT NULL 
        AND alert_id != 'NULL' 
        AND alert_id != ''
        AND confidence_score IS NOT NULL
        AND confidence_score > 0
        AND classification IS NOT NULL
        AND classification != ''
        AND timestamp IS NOT NULL
    ORDER BY timestamp DESC
    LIMIT 20
    """
    
    try:
        query_job = client.query(query)
        results = query_job.result()
        
        alerts = []
        for row in results:
            alerts.append({
                'alert_id': row.alert_id,
                'timestamp': row.timestamp.isoformat() if row.timestamp else None,
                'confidence_score': round(float(row.confidence_score), 3),
                'classification': row.classification,
                'alert_preview': row.alert_preview
            })
        
        return alerts
    except Exception as e:
        print(f"Error querying recent alerts: {e}")
        return []

def get_service_status():
    """Check if ADA service is running by looking for recent data"""
    client = get_bigquery_client()
    
    query = f"""
    SELECT 
        COUNT(*) as recent_count,
        MAX(timestamp) as last_processed
    FROM `{PROJECT_ID}.{DATASET_ID}.{PROCESSED_ALERTS_TABLE}`
    WHERE 
        timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)
        AND alert_id IS NOT NULL 
        AND alert_id != 'NULL' 
        AND alert_id != ''
    """
    
    try:
        query_job = client.query(query)
        results = query_job.result()
        
        for row in results:
            return {
                'status': 'active' if (row.recent_count or 0) > 0 else 'inactive',
                'recent_count': row.recent_count or 0,
                'last_processed': row.last_processed.isoformat() if row.last_processed else None
            }
    except Exception as e:
        print(f"Error checking service status: {e}")
        return {
            'status': 'error',
            'recent_count': 0,
            'last_processed': None,
            'error': str(e)
        }

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('ada_dashboard.html')

@app.route('/api/metrics')
def api_metrics():
    """API endpoint for metrics data"""
    metrics = get_processed_alerts_metrics()
    return jsonify(metrics)

@app.route('/api/recent-alerts')
def api_recent_alerts():
    """API endpoint for recent alerts"""
    alerts = get_recent_alerts()
    return jsonify(alerts)

@app.route('/api/service-status')
def api_service_status():
    """API endpoint for service status"""
    status = get_service_status()
    return jsonify(status)

@app.route('/api/dashboard-data')
def api_dashboard_data():
    """Combined API endpoint for all dashboard data"""
    return jsonify({
        'metrics': get_processed_alerts_metrics(),
        'recent_alerts': get_recent_alerts(),
        'service_status': get_service_status(),
        'last_updated': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("Starting ADA BigQuery Dashboard...")
    print(f"Project: {PROJECT_ID}")
    print(f"Dataset: {DATASET_ID}")
    print(f"Table: {PROCESSED_ALERTS_TABLE}")
    
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    app.run(host='0.0.0.0', port=3002, debug=True)
