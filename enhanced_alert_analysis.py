#!/usr/bin/env python3
"""
Enhanced Alert Analysis Functions with Real ML/LLM Integration
This module provides real ML model inference and LLM enrichment for the Alert Review tab
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from google.cloud import bigquery
import google.generativeai as genai

# Initialize Gemini
def initialize_gemini():
    """Initialize Google Gemini with multiple authentication methods"""
    try:
        # Method 1: Try API Key first (most reliable for Gemini API)
        api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        if api_key:
            try:
                genai.configure(api_key=api_key)
                # Test the configuration by trying to load a model
                test_model = genai.GenerativeModel('gemini-2.0-flash-exp')
                print("✅ Gemini initialized successfully with API key")
                return True
            except Exception as e:
                print(f"⚠️ API key configuration failed: {e}")

        # Method 2: Try Application Default Credentials (Service Account)
        service_account_paths = [
            'Service Account BigQuery/gatra-user-gemini.json',  # Gemini-specific service account (primary)
            'Service Account BigQuery/gatra-user-bigquery.json',  # BigQuery service account (fallback)
            'Service Account BigQuery/sa-gatra-bigquery.json',  # Original service account (fallback)
            '/home/app/ai-driven-soc/Service Account BigQuery/gatra-user-gemini.json',  # VM path
            os.path.expanduser('~/Service Account BigQuery/gatra-user-gemini.json'),  # Home directory
        ]

        for sa_path in service_account_paths:
            if os.path.exists(sa_path):
                try:
                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = sa_path
                    genai.configure()  # Configure with ADC
                    # Test the configuration
                    test_model = genai.GenerativeModel('gemini-2.0-flash-exp')
                    print(f"✅ Gemini initialized successfully with service account: {sa_path}")
                    return True
                except Exception as e:
                    print(f"⚠️ Service account {sa_path} failed: {e}")
                    continue

        # Method 3: Try to use already configured ADC
        try:
            genai.configure()  # This will use existing GOOGLE_APPLICATION_CREDENTIALS
            test_model = genai.GenerativeModel('gemini-2.0-flash-exp')
            print("✅ Gemini initialized with existing ADC")
            return True
        except:
            pass

        print("❌ Gemini initialization failed: No valid authentication method found")
        print("💡 SOLUTION: Set GEMINI_API_KEY environment variable (RECOMMENDED)")
        print("   Get your API key from: https://aistudio.google.com/app/apikey")
        print("   Then run: export GEMINI_API_KEY='your-api-key'")
        print()
        print("   Alternative: Ensure service account has these OAuth scopes:")
        print("   - https://www.googleapis.com/auth/generative-language")
        print("   - https://www.googleapis.com/auth/generative-language.retriever")
        return False

    except Exception as e:
        print(f"❌ Gemini initialization failed: {e}")
        return False

# Initialize Gemini
GEMINI_AVAILABLE = initialize_gemini()

def load_latest_cla_model():
    """Load the latest trained CLA model"""
    try:
        # Try to load enhanced model
        if os.path.exists('enhanced_cla_model.pkl'):
            model_data = joblib.load('enhanced_cla_model.pkl')
            return model_data
        # Fallback to supervised model
        elif os.path.exists('supervised_model_v1.joblib'):
            model = joblib.load('supervised_model_v1.joblib')
            return {'models': {'main': model}, 'scaler': None}
        else:
            return None
    except Exception as e:
        print(f"Error loading CLA model: {e}")
        return None

def get_real_ml_threat_score(alert_data, extracted_params):
    """
    Real ML model inference for threat scoring
    Uses trained CLA ensemble models
    """
    try:
        # Load model
        model_data = load_latest_cla_model()
        
        if not model_data:
            return {
                'threat_score': alert_data.get('confidence_score', 0.5),
                'classification': alert_data.get('classification', 'unknown'),
                'confidence': 'medium',
                'model_used': 'fallback',
                'models_count': 1
            }
        
        # Extract features for ML model
        features = []
        
        # Feature 1: Confidence score
        features.append(alert_data.get('confidence_score', 0.5))
        
        # Feature 2: Severity (encoded)
        severity_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
        features.append(severity_map.get(alert_data.get('severity', 'Medium'), 2))
        
        # Feature 3: Is anomaly
        features.append(1 if alert_data.get('is_anomaly') else 0)
        
        # Feature 4: Classification type
        classification = str(alert_data.get('classification', '')).lower()
        features.append(1 if 'threat' in classification or 'malware' in classification else 0)
        
        # Feature 5: IP count
        features.append(len(extracted_params.get('ip_addresses', [])))
        
        # Feature 6: Has network flow
        features.append(1 if extracted_params.get('network_flow') else 0)
        
        # Feature 7: Bytes transferred (normalized)
        bytes_val = extracted_params.get('bytes_transferred', 0)
        features.append(min(bytes_val / 1000000000, 1.0))  # Normalize to 0-1
        
        # Pad features to match expected model input
        while len(features) < 10:
            features.append(0)
        
        features_array = np.array(features[:10]).reshape(1, -1)
        
        # Get prediction from model
        models = model_data.get('models', {})
        scaler = model_data.get('scaler')
        
        if scaler:
            features_array = scaler.transform(features_array)
        
        # Ensemble prediction
        predictions = []
        for name, model in models.items():
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(features_array)[0][1]
            elif hasattr(model, 'decision_function'):
                score = model.decision_function(features_array)[0]
                pred = 1 / (1 + np.exp(-score))  # Sigmoid
            else:
                pred = 0.5
            predictions.append(pred)
        
        # Average ensemble prediction
        threat_score = np.mean(predictions) if predictions else 0.5
        
        # Classify confidence level
        if threat_score > 0.8:
            confidence = 'high'
            classification_label = 'Confirmed Threat'
        elif threat_score > 0.6:
            confidence = 'medium'
            classification_label = 'Potential Threat'
        else:
            confidence = 'low'
            classification_label = 'Likely Benign'
        
        return {
            'threat_score': float(threat_score),
            'classification': classification_label,
            'confidence': confidence,
            'model_used': 'CLA Ensemble',
            'models_count': len(models)
        }
        
    except Exception as e:
        print(f"ML inference error: {e}")
        return {
            'threat_score': alert_data.get('confidence_score', 0.5),
            'classification': alert_data.get('classification', 'unknown'),
            'confidence': 'medium',
            'model_used': 'fallback',
            'models_count': 1
        }

def get_gemini_alert_enrichment(alert_data, extracted_params, ml_score):
    """
    Real Google Gemini Flash 2.5 integration for alert enrichment
    Provides LLM-powered context and analysis
    """
    if not GEMINI_AVAILABLE:
        return {
            'summary': 'LLM enrichment unavailable - Gemini authentication failed. Please check GOOGLE_API_KEY, GEMINI_API_KEY, or service account configuration.',
            'context': [],
            'recommendations': []
        }

    try:
        # Try multiple Gemini model names (prioritize working models)
        model_names = ['gemini-2.0-flash-exp', 'gemini-1.5-flash', 'gemini-1.5-flash-latest', 'gemini-flash-2.5']

        model = None
        for model_name in model_names:
            try:
                model = genai.GenerativeModel(model_name)
                break
            except Exception as e:
                print(f"Failed to load model {model_name}: {e}")
                continue

        if not model:
            return {
                'summary': 'Gemini models not available. Please check your Google Cloud project and API access.',
                'context': [],
                'recommendations': []
            }
        
        # Create comprehensive prompt
        prompt = f"""
You are a cybersecurity analyst assistant. Analyze this security alert and provide:

ALERT DETAILS:
- Alert ID: {alert_data.get('alert_id', 'Unknown')}
- Classification: {alert_data.get('classification', 'Unknown')}
- Severity: {alert_data.get('severity', 'Unknown')}
- Confidence Score: {alert_data.get('confidence_score', 0)}
- ML Threat Score: {ml_score['threat_score']:.2f}
- IP Addresses: {', '.join(extracted_params.get('ip_addresses', ['None']))}
- Bytes Transferred: {extracted_params.get('bytes_transferred', 'Unknown')}

Provide:
1. SUMMARY: One-sentence assessment of this alert
2. CONTEXT: 2-3 key contextual insights
3. RECOMMENDATIONS: 3 specific actions the analyst should take

Be concise and actionable. Format as JSON with keys: summary, context (array), recommendations (array)
"""
        
        # Generate response
        response = model.generate_content(prompt)
        response_text = response.text
        
        # Try to parse JSON response
        try:
            # Clean markdown code blocks if present
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0]
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0]
            
            result = json.loads(response_text.strip())
            return result
        except:
            # If JSON parsing fails, return as-is
            return {
                'summary': response_text[:200],
                'context': [response_text],
                'recommendations': ['Review alert details', 'Investigate IP addresses', 'Check for similar patterns']
            }
            
    except Exception as e:
        print(f"Gemini enrichment error: {e}")
        error_msg = str(e)

        # Provide specific guidance based on error type
        if "API_KEY" in error_msg or "ADC" in error_msg or "credentials" in error_msg.lower():
            guidance = " Please check your GOOGLE_API_KEY, GEMINI_API_KEY environment variables or service account file path."
        elif "permission" in error_msg.lower() or "403" in error_msg:
            guidance = " Please check that your service account or API key has Gemini API access permissions."
        elif "quota" in error_msg.lower() or "429" in error_msg:
            guidance = " Please check your Gemini API quota limits."
        else:
            guidance = " Please check your Google Cloud project configuration and internet connection."

        return {
            'summary': f'Alert enrichment failed: {error_msg}{guidance}',
            'context': [],
            'recommendations': []
        }

def get_social_media_alerts(client, limit: int = 50):
    """
    Get social media alerts from BigQuery for SOC analysis
    """
    try:
        query = f"""
        SELECT
            tweet_id,
            text,
            created_at,
            user_screen_name,
            user_followers,
            retweet_count,
            favorite_count,
            threat_score,
            severity,
            risk_factors,
            keywords_found,
            analysis_timestamp,
            language
        FROM `chronicle-dev-2be9.soc_data.social_media_alerts`
        ORDER BY analysis_timestamp DESC
        LIMIT {limit}
        """

        result = client.query(query).result()
        rows = list(result)

        alerts = []
        for row in rows:
            alerts.append({
                'alert_id': f"social_{row.tweet_id}",
                'timestamp': row.created_at,
                'classification': f"Social Media - {row.severity}",
                'severity': row.severity,
                'confidence_score': row.threat_score,
                'is_anomaly': row.threat_score > 0.5,
                'text': row.text,
                'user_screen_name': row.user_screen_name,
                'user_followers': row.user_followers,
                'retweet_count': row.retweet_count,
                'favorite_count': row.favorite_count,
                'risk_factors': row.risk_factors.split(',') if row.risk_factors else [],
                'keywords_found': row.keywords_found.split(',') if row.keywords_found else [],
                'source': 'twitter',
                'analysis_timestamp': row.analysis_timestamp,
                'language': row.language
            })

        return alerts

    except Exception as e:
        print(f"Error getting social media alerts: {e}")
        return []

def get_real_historical_correlation(alert_data, client):
    """
    Real BigQuery historical incident correlation
    Searches for similar past incidents
    """
    try:
        # Search for similar alerts by classification and severity
        query = f"""
        SELECT 
            alert_id,
            timestamp,
            classification,
            severity,
            is_anomaly
        FROM `chronicle-dev-2be9.soc_data.alerts`
        WHERE classification = '{alert_data.get('classification', 'unknown')}'
        AND severity = '{alert_data.get('severity', 'Medium')}'
        AND alert_id != '{alert_data.get('alert_id', '')}'
        ORDER BY timestamp DESC
        LIMIT 5
        """
        
        result = client.query(query).to_dataframe()
        
        if not result.empty:
            correlations = []
            for _, row in result.iterrows():
                correlations.append({
                    'alert_id': row['alert_id'],
                    'timestamp': str(row['timestamp'])[:19],
                    'classification': row['classification'],
                    'severity': row['severity']
                })
            return correlations
        else:
            return []
            
    except Exception as e:
        print(f"Historical correlation error: {e}")
        return []

def get_dynamic_mitre_attack_mapping(alert_data, extracted_params, gemini_analysis):
    """
    Dynamic MITRE ATT&CK mapping based on alert behavior
    """
    ttps = []
    
    try:
        classification = str(alert_data.get('classification', '')).lower()
        severity = alert_data.get('severity', 'Medium')
        
        # Data exfiltration indicators
        if extracted_params.get('bytes_transferred', 0) > 100000000:  # >100MB
            ttps.append({
                'technique': 'T1041',
                'name': 'Exfiltration Over C2 Channel',
                'tactic': 'Exfiltration',
                'confidence': 'high' if severity == 'High' else 'medium'
            })
            ttps.append({
                'technique': 'T1048',
                'name': 'Exfiltration Over Alternative Protocol',
                'tactic': 'Exfiltration',
                'confidence': 'medium'
            })
        
        # Anomaly behavior
        if 'anomaly' in classification:
            ttps.append({
                'technique': 'T1071',
                'name': 'Application Layer Protocol',
                'tactic': 'Command and Control',
                'confidence': 'medium'
            })
        
        # Network scanning
        if extracted_params.get('has_port_scan'):
            ttps.append({
                'technique': 'T1046',
                'name': 'Network Service Scanning',
                'tactic': 'Discovery',
                'confidence': 'high'
            })
        
        # Malware indicators
        if 'malware' in classification or 'threat' in classification:
            ttps.append({
                'technique': 'T1059',
                'name': 'Command and Scripting Interpreter',
                'tactic': 'Execution',
                'confidence': 'high'
            })
            ttps.append({
                'technique': 'T1566',
                'name': 'Phishing',
                'tactic': 'Initial Access',
                'confidence': 'medium'
            })
        
        return ttps if ttps else [{
            'technique': 'T1071',
            'name': 'Application Layer Protocol',
            'tactic': 'Command and Control',
            'confidence': 'low'
        }]
        
    except Exception as e:
        print(f"MITRE mapping error: {e}")
        return []

# Export functions for use in dashboard
__all__ = [
    'get_real_ml_threat_score',
    'get_gemini_alert_enrichment',
    'get_real_historical_correlation',
    'get_dynamic_mitre_attack_mapping',
    'get_social_media_alerts'
]

