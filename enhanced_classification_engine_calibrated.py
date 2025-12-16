#!/usr/bin/env python3
"""
Enhanced Classification Engine with Confidence Calibration
- Fixes confidence miscalibration (83.9% confidence vs 33.3% precision)
- Improves severity scoring (average 0.392 -> proper distribution)
- Implements suppression policies for noise reduction
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ThreatLevel(Enum):
    BENIGN = "benign"
    LOW_RISK = "low_risk"
    SUSPICIOUS = "suspicious"
    MALICIOUS = "malicious"
    CRITICAL = "critical"

@dataclass
class CalibratedThreatScore:
    classification: ThreatLevel
    calibrated_confidence: float
    raw_confidence: float
    threat_score: float
    severity_score: float
    is_anomaly: bool
    reasoning: List[str]
    suppression_recommended: bool

class CalibratedEnhancedClassificationEngine:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # CALIBRATED CONFIDENCE MAPPING (based on observed precision: 33.3%)
        self.confidence_calibration = {
            'critical': {'raw': 0.95, 'calibrated': 0.75},
            'malicious': {'raw': 0.90, 'calibrated': 0.65},
            'suspicious': {'raw': 0.75, 'calibrated': 0.45},
            'low_risk': {'raw': 0.60, 'calibrated': 0.30},
            'benign': {'raw': 0.40, 'calibrated': 0.15}
        }
        
        # SEVERITY SCORING (0-1 scale, properly distributed)
        self.severity_weights = {
            'critical': 1.0,
            'malicious': 0.8,
            'suspicious': 0.6,
            'low_risk': 0.3,
            'benign': 0.1
        }
        
        # SUPPRESSION THRESHOLDS (to reduce noise - adjusted for better balance)
        self.suppression_thresholds = {
            'confidence_threshold': 0.2,  # Suppress if confidence < 20%
            'severity_threshold': 0.1,    # Suppress if severity < 10%
            'combined_threshold': 0.15    # Suppress if combined score < 15%
        }
        
        # THREAT INDICATORS (enhanced)
        self.threat_indicators = {
            'attack_categories': {
                'exploit': 8.0,
                'malware': 9.0,
                'phishing': 7.0,
                'brute_force': 6.0,
                'data_exfiltration': 9.5,
                'privilege_escalation': 8.5,
                'lateral_movement': 8.0,
                'persistence': 7.5,
                'command_control': 9.0,
                'defense_evasion': 8.5
            },
            'severity_levels': {
                'critical': 10.0,
                'high': 8.0,
                'medium': 5.0,
                'low': 2.0,
                'info': 0.5
            },
            'keywords': {
                'exploit': 3.0,
                'malware': 4.0,
                'backdoor': 5.0,
                'trojan': 4.5,
                'virus': 4.0,
                'rootkit': 5.5,
                'keylogger': 4.5,
                'ransomware': 6.0,
                'botnet': 5.0,
                'c2': 6.5,
                'exfiltration': 7.0,
                'privilege': 6.0,
                'escalation': 6.5,
                'lateral': 5.5,
                'persistence': 5.0,
                'evasion': 6.0
            },
            'ip_reputation': {
                'malicious': 8.0,
                'suspicious': 5.0,
                'unknown': 2.0,
                'clean': 0.0
            }
        }
        
        self.logger.info("🎯 Calibrated Enhanced Classification Engine initialized")
        self.logger.info(f"📊 Confidence calibration: {self.confidence_calibration}")
        self.logger.info(f"📊 Suppression thresholds: {self.suppression_thresholds}")

    def calculate_threat_score(self, alert_data: Dict[str, Any]) -> float:
        """Calculate comprehensive threat score"""
        score = 0.0
        reasoning = []
        
        # 1. Attack Category Score
        attack_category = str(alert_data.get('attack_category', '')).lower()
        if attack_category in self.threat_indicators['attack_categories']:
            category_score = self.threat_indicators['attack_categories'][attack_category]
            score += category_score
            reasoning.append(f"Attack category '{attack_category}': {category_score}")
        
        # 2. Severity Score
        alert_severity = str(alert_data.get('alert_severity', '')).lower()
        if alert_severity in self.threat_indicators['severity_levels']:
            severity_score = self.threat_indicators['severity_levels'][alert_severity]
            score += severity_score
            reasoning.append(f"Severity '{alert_severity}': {severity_score}")
        
        # 3. Keyword Analysis
        raw_text = str(alert_data.get('alert_data', '')).lower()
        for keyword, kw_score in self.threat_indicators['keywords'].items():
            if keyword in raw_text:
                score += kw_score
                reasoning.append(f"Threat keyword '{keyword}': {kw_score}")
        
        # 4. IP Reputation
        source_ip = str(alert_data.get('source_ip', '')).strip()
        if source_ip and not self._is_private_ip(source_ip):
            # Assume unknown public IPs have some risk
            score += 2.0
            reasoning.append(f"Public IP {source_ip}: 2.0")
        
        # 5. Confidence Inversion (low confidence = higher threat potential)
        confidence = float(alert_data.get('confidence', 0.5))
        if confidence < 0.5:
            confidence_penalty = (0.5 - confidence) * 5.0
            score += confidence_penalty
            reasoning.append(f"Low confidence penalty: {confidence_penalty}")
        
        return score

    def _is_private_ip(self, ip: str) -> bool:
        """Check if IP is private"""
        try:
            parts = ip.split('.')
            if len(parts) != 4:
                return False
            
            first = int(parts[0])
            second = int(parts[1])
            
            return (first == 10 or 
                   (first == 172 and 16 <= second <= 31) or
                   (first == 192 and second == 168))
        except:
            return False

    def classify_alert_calibrated(self, alert_data: Dict[str, Any]) -> CalibratedThreatScore:
        """Classify alert with calibrated confidence and proper severity"""
        
        # Calculate threat score
        threat_score = self.calculate_threat_score(alert_data)
        
        # Determine classification based on threat score
        if threat_score >= 15.0:
            classification = ThreatLevel.CRITICAL
            raw_confidence = 0.95
        elif threat_score >= 10.0:
            classification = ThreatLevel.MALICIOUS
            raw_confidence = 0.85
        elif threat_score >= 6.0:
            classification = ThreatLevel.SUSPICIOUS
            raw_confidence = 0.70
        elif threat_score >= 3.0:
            classification = ThreatLevel.LOW_RISK
            raw_confidence = 0.55
        else:
            classification = ThreatLevel.BENIGN
            raw_confidence = 0.30
        
        # Apply confidence calibration
        calibrated_confidence = self.confidence_calibration[classification.value]['calibrated']
        
        # Calculate proper severity score (0-1)
        severity_score = self.severity_weights[classification.value]
        
        # Determine if anomaly
        is_anomaly = classification != ThreatLevel.BENIGN
        
        # Check suppression recommendations
        suppression_recommended = (
            calibrated_confidence < self.suppression_thresholds['confidence_threshold'] or
            severity_score < self.suppression_thresholds['severity_threshold'] or
            (calibrated_confidence * severity_score) < self.suppression_thresholds['combined_threshold']
        )
        
        # Generate reasoning
        reasoning = [
            f"Threat score: {threat_score:.1f}",
            f"Classification: {classification.value}",
            f"Raw confidence: {raw_confidence:.2f}",
            f"Calibrated confidence: {calibrated_confidence:.2f}",
            f"Severity: {severity_score:.2f}",
            f"Suppression recommended: {suppression_recommended}"
        ]
        
        return CalibratedThreatScore(
            classification=classification,
            calibrated_confidence=calibrated_confidence,
            raw_confidence=raw_confidence,
            threat_score=threat_score,
            severity_score=severity_score,
            is_anomaly=is_anomaly,
            reasoning=reasoning,
            suppression_recommended=suppression_recommended
        )

    def classify_alert(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main classification method with calibrated outputs"""
        try:
            result = self.classify_alert_calibrated(alert_data)
            
            return {
                'enhanced_classification': result.classification.value,
                'enhanced_confidence': result.calibrated_confidence,
                'raw_confidence': result.raw_confidence,
                'threat_score': result.threat_score,
                'enhanced_is_anomaly': result.is_anomaly,
                'severity_score': result.severity_score,
                'classification_reasoning': '; '.join(result.reasoning),
                'suppression_recommended': result.suppression_recommended,
                'enhanced_classification': True,
                'calibrated': True
            }
        except Exception as e:
            self.logger.error(f"Error in calibrated classification: {e}")
            return {
                'enhanced_classification': 'benign',
                'enhanced_confidence': 0.15,
                'raw_confidence': 0.30,
                'threat_score': 0.0,
                'enhanced_is_anomaly': False,
                'severity_score': 0.1,
                'classification_reasoning': f"Error: {str(e)}",
                'suppression_recommended': True,
                'enhanced_classification': False,
                'calibrated': True
            }

# Example usage and testing
if __name__ == "__main__":
    # Test the calibrated engine
    engine = CalibratedEnhancedClassificationEngine()
    
    # Test alerts
    test_alerts = [
        {
            'alarm_id': 'test-1',
            'attack_category': 'malware',
            'alert_severity': 'high',
            'source_ip': '192.168.1.100',
            'alert_data': 'Malware detected: trojan horse',
            'confidence': 0.3
        },
        {
            'alarm_id': 'test-2',
            'attack_category': 'benign',
            'alert_severity': 'low',
            'source_ip': '10.0.0.1',
            'alert_data': 'Normal network activity',
            'confidence': 0.8
        }
    ]
    
    print("🧪 Testing Calibrated Enhanced Classification Engine")
    print("=" * 60)
    
    for alert in test_alerts:
        result = engine.classify_alert(alert)
        print(f"\n📨 Alert: {alert['alarm_id']}")
        print(f"   Classification: {result['enhanced_classification']}")
        print(f"   Calibrated Confidence: {result['enhanced_confidence']:.2f}")
        print(f"   Raw Confidence: {result['raw_confidence']:.2f}")
        print(f"   Severity Score: {result['severity_score']:.2f}")
        print(f"   Threat Score: {result['threat_score']:.1f}")
        print(f"   Suppression Recommended: {result['suppression_recommended']}")
        print(f"   Reasoning: {result['classification_reasoning']}")
