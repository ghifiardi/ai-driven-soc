#!/usr/bin/env python3
"""
Enhanced Classification Engine
=============================

Fixes the broken classification logic by properly utilizing all available
threat intelligence data for accurate threat detection and classification.
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ThreatLevel(Enum):
    """Threat classification levels"""
    BENIGN = "benign"
    LOW_RISK = "low_risk"
    SUSPICIOUS = "suspicious"
    MALICIOUS = "malicious"
    CRITICAL = "critical"

class AttackCategory(Enum):
    """Attack category classifications"""
    NORMAL = "Normal"
    EXPLOITS = "Exploits"
    MALWARE = "Malware"
    INTRUSION = "Intrusion"
    DATA_EXFILTRATION = "Data_Exfiltration"
    LATERAL_MOVEMENT = "Lateral_Movement"
    PRIVILEGE_ESCALATION = "Privilege_Escalation"
    PERSISTENCE = "Persistence"
    DEFENSE_EVASION = "Defense_Evasion"
    DISCOVERY = "Discovery"
    COLLECTION = "Collection"
    COMMAND_AND_CONTROL = "Command_And_Control"
    EXECUTION = "Execution"
    IMPACT = "Impact"

@dataclass
class ThreatScore:
    """Comprehensive threat score breakdown"""
    total_score: float
    attack_category_score: float
    severity_score: float
    confidence_score: float
    ip_reputation_score: float
    behavioral_score: float
    temporal_score: float
    contextual_score: float
    classification: ThreatLevel
    confidence: float
    reasoning: List[str]

class EnhancedClassificationEngine:
    """Enhanced classification engine that properly utilizes threat intelligence"""
    
    def __init__(self, config_file: str = None):
        """Initialize the enhanced classification engine"""
        self.config = self._load_config(config_file)
        self.threat_indicators = self._initialize_threat_indicators()
        self.ip_reputation_db = self._initialize_ip_reputation()
        self.attack_patterns = self._initialize_attack_patterns()
        
        logger.info("🚀 Enhanced Classification Engine initialized")
        logger.info(f"📊 Loaded {len(self.threat_indicators)} threat indicators")
        logger.info(f"🌐 Loaded {len(self.ip_reputation_db)} IP reputation entries")
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration for classification thresholds"""
        default_config = {
            "classification_thresholds": {
                "critical": 8.0,
                "malicious": 6.0,
                "suspicious": 4.0,
                "low_risk": 2.0,
                "benign": 0.0
            },
            "attack_category_weights": {
                "Exploits": 5.0,
                "Malware": 4.5,
                "Intrusion": 4.0,
                "Data_Exfiltration": 4.5,
                "Lateral_Movement": 3.5,
                "Privilege_Escalation": 4.0,
                "Persistence": 3.0,
                "Defense_Evasion": 3.5,
                "Discovery": 2.0,
                "Collection": 2.5,
                "Command_And_Control": 3.5,
                "Execution": 3.0,
                "Impact": 4.0,
                "Normal": 0.0
            },
            "severity_weights": {
                "critical": 4.0,
                "high": 3.0,
                "medium": 2.0,
                "low": 1.0,
                "info": 0.5
            },
            "confidence_thresholds": {
                "high_confidence": 0.8,
                "medium_confidence": 0.5,
                "low_confidence": 0.3
            },
            "ip_reputation_weights": {
                "known_malicious": 4.0,
                "suspicious": 2.0,
                "unknown": 1.0,
                "trusted": -1.0
            }
        }
        
        if config_file:
            try:
                with open(config_file, 'r') as f:
                    custom_config = json.load(f)
                    default_config.update(custom_config)
            except Exception as e:
                logger.warning(f"Could not load config file {config_file}: {e}")
        
        return default_config
    
    def _initialize_threat_indicators(self) -> Dict[str, float]:
        """Initialize threat indicator patterns and weights"""
        return {
            # Network-based indicators
            "port_scanning": 3.0,
            "brute_force": 4.0,
            "ddos_attack": 4.5,
            "data_exfiltration": 5.0,
            "command_control": 4.0,
            "lateral_movement": 3.5,
            
            # Application-based indicators
            "sql_injection": 4.5,
            "xss_attack": 3.0,
            "csrf_attack": 3.5,
            "file_upload": 2.5,
            "privilege_escalation": 4.0,
            
            # System-based indicators
            "malware_signature": 5.0,
            "rootkit_detection": 5.0,
            "backdoor_activity": 5.0,
            "persistence_mechanism": 3.5,
            "defense_evasion": 3.0,
            
            # Behavioral indicators
            "unusual_access_pattern": 2.5,
            "off_hours_activity": 2.0,
            "geographic_anomaly": 2.0,
            "volume_anomaly": 3.0,
            "frequency_anomaly": 2.5
        }
    
    def _initialize_ip_reputation(self) -> Dict[str, str]:
        """Initialize IP reputation database"""
        # In a real implementation, this would be loaded from threat intelligence feeds
        return {
            # Known malicious IPs (example)
            "1.1.1.1": "known_malicious",
            "2.2.2.2": "suspicious",
            "10.0.0.0/8": "trusted",
            "192.168.0.0/16": "trusted",
            "172.16.0.0/12": "trusted"
        }
    
    def _initialize_attack_patterns(self) -> Dict[str, List[str]]:
        """Initialize attack pattern recognition"""
        return {
            "exploit_patterns": [
                "exploit", "buffer_overflow", "injection", "vulnerability",
                "cve-", "payload", "shellcode", "rop_chain"
            ],
            "malware_patterns": [
                "malware", "virus", "trojan", "backdoor", "rootkit",
                "ransomware", "botnet", "keylogger", "spyware"
            ],
            "intrusion_patterns": [
                "intrusion", "breach", "unauthorized", "privilege",
                "escalation", "persistence", "lateral", "movement"
            ],
            "data_exfiltration_patterns": [
                "exfiltration", "data_theft", "sensitive", "confidential",
                "large_upload", "encrypted_traffic", "steganography"
            ]
        }
    
    def calculate_threat_score(self, alert_data: Dict[str, Any]) -> ThreatScore:
        """Calculate comprehensive threat score using all available intelligence"""
        
        # Initialize score components
        scores = {
            'attack_category_score': 0.0,
            'severity_score': 0.0,
            'confidence_score': 0.0,
            'ip_reputation_score': 0.0,
            'behavioral_score': 0.0,
            'temporal_score': 0.0,
            'contextual_score': 0.0
        }
        
        reasoning = []
        
        # 1. Attack Category Analysis
        attack_category = alert_data.get('attack_category', 'Normal')
        if attack_category in self.config['attack_category_weights']:
            scores['attack_category_score'] = self.config['attack_category_weights'][attack_category]
            if scores['attack_category_score'] > 0:
                reasoning.append(f"Attack category '{attack_category}' (score: {scores['attack_category_score']})")
        
        # 2. Severity Analysis
        severity = alert_data.get('alert_severity', 'info')
        if severity in self.config['severity_weights']:
            scores['severity_score'] = self.config['severity_weights'][severity]
            if scores['severity_score'] > 0:
                reasoning.append(f"Severity level '{severity}' (score: {scores['severity_score']})")
        
        # 3. Confidence Analysis (inverted - lower confidence = higher threat)
        confidence = alert_data.get('confidence', 0.5)
        if confidence < self.config['confidence_thresholds']['low_confidence']:
            scores['confidence_score'] = 3.0  # Low confidence indicates uncertainty/evasion
            reasoning.append(f"Low confidence score {confidence} indicates potential evasion")
        elif confidence < self.config['confidence_thresholds']['medium_confidence']:
            scores['confidence_score'] = 1.5
            reasoning.append(f"Medium confidence score {confidence}")
        else:
            scores['confidence_score'] = 0.0
        
        # 4. IP Reputation Analysis
        source_ip = alert_data.get('source_ip', '')
        dest_ip = alert_data.get('destination_ip', '')
        
        if source_ip:
            ip_reputation = self._get_ip_reputation(source_ip)
            if ip_reputation in self.config['ip_reputation_weights']:
                scores['ip_reputation_score'] = self.config['ip_reputation_weights'][ip_reputation]
                if scores['ip_reputation_score'] > 0:
                    reasoning.append(f"Source IP {source_ip} reputation: {ip_reputation}")
        
        # 5. Behavioral Analysis
        behavioral_score = self._analyze_behavioral_patterns(alert_data)
        scores['behavioral_score'] = behavioral_score
        if behavioral_score > 0:
            reasoning.append(f"Behavioral anomalies detected (score: {behavioral_score})")
        
        # 6. Temporal Analysis
        temporal_score = self._analyze_temporal_patterns(alert_data)
        scores['temporal_score'] = temporal_score
        if temporal_score > 0:
            reasoning.append(f"Temporal anomalies detected (score: {temporal_score})")
        
        # 7. Contextual Analysis
        contextual_score = self._analyze_contextual_patterns(alert_data)
        scores['contextual_score'] = contextual_score
        if contextual_score > 0:
            reasoning.append(f"Contextual threats detected (score: {contextual_score})")
        
        # Calculate total score
        total_score = sum(scores.values())
        
        # Determine classification
        classification = self._determine_classification(total_score)
        
        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(scores, reasoning)
        
        return ThreatScore(
            total_score=total_score,
            attack_category_score=scores['attack_category_score'],
            severity_score=scores['severity_score'],
            confidence_score=scores['confidence_score'],
            ip_reputation_score=scores['ip_reputation_score'],
            behavioral_score=scores['behavioral_score'],
            temporal_score=scores['temporal_score'],
            contextual_score=scores['contextual_score'],
            classification=classification,
            confidence=overall_confidence,
            reasoning=reasoning
        )
    
    def _get_ip_reputation(self, ip: str) -> str:
        """Get IP reputation from database"""
        # Check exact match
        if ip in self.ip_reputation_db:
            return self.ip_reputation_db[ip]
        
        # Check CIDR ranges
        for cidr, reputation in self.ip_reputation_db.items():
            if '/' in cidr and self._ip_in_cidr(ip, cidr):
                return reputation
        
        # Default to unknown
        return "unknown"
    
    def _ip_in_cidr(self, ip: str, cidr: str) -> bool:
        """Check if IP is in CIDR range"""
        try:
            import ipaddress
            return ipaddress.ip_address(ip) in ipaddress.ip_network(cidr)
        except:
            return False
    
    def _analyze_behavioral_patterns(self, alert_data: Dict[str, Any]) -> float:
        """Analyze behavioral patterns for anomalies"""
        score = 0.0
        
        # Check for unusual processing time patterns
        processing_time = alert_data.get('processing_time', 0.0)
        if processing_time > 0.01:  # Unusually slow processing
            score += 1.0
        
        # Check for repeated alarm IDs (potential replay attacks)
        # This would require historical data in a real implementation
        
        return score
    
    def _analyze_temporal_patterns(self, alert_data: Dict[str, Any]) -> float:
        """Analyze temporal patterns for anomalies"""
        score = 0.0
        
        # Check timestamp for off-hours activity
        timestamp_str = alert_data.get('timestamp', '')
        if timestamp_str:
            try:
                timestamp = pd.to_datetime(timestamp_str)
                hour = timestamp.hour
                # Off-hours activity (11 PM - 6 AM)
                if hour >= 23 or hour <= 6:
                    score += 1.0
            except:
                pass
        
        return score
    
    def _analyze_contextual_patterns(self, alert_data: Dict[str, Any]) -> float:
        """Analyze contextual patterns for threats"""
        score = 0.0
        
        # Check protocol anomalies
        protocol = (alert_data.get('protocol') or '').lower()
        if protocol in ['udp', 'icmp'] and alert_data.get('attack_category') == 'Exploits':
            score += 1.0  # Suspicious protocol for exploits
        
        # Check for external IP connections with high severity
        source_ip = alert_data.get('source_ip', '')
        if source_ip and not self._is_private_ip(source_ip):
            if alert_data.get('alert_severity') == 'high':
                score += 1.5  # External high-severity connection
        
        return score
    
    def _is_private_ip(self, ip: str) -> bool:
        """Check if IP is private"""
        try:
            parts = ip.split('.')
            if len(parts) != 4:
                return False
            
            first = int(parts[0])
            second = int(parts[1])
            
            # Private IP ranges
            if first == 10:
                return True
            elif first == 172 and 16 <= second <= 31:
                return True
            elif first == 192 and second == 168:
                return True
            
            return False
        except:
            return False
    
    def _determine_classification(self, total_score: float) -> ThreatLevel:
        """Determine threat classification based on total score"""
        thresholds = self.config['classification_thresholds']
        
        if total_score >= thresholds['critical']:
            return ThreatLevel.CRITICAL
        elif total_score >= thresholds['malicious']:
            return ThreatLevel.MALICIOUS
        elif total_score >= thresholds['suspicious']:
            return ThreatLevel.SUSPICIOUS
        elif total_score >= thresholds['low_risk']:
            return ThreatLevel.LOW_RISK
        else:
            return ThreatLevel.BENIGN
    
    def _calculate_overall_confidence(self, scores: Dict[str, float], reasoning: List[str]) -> float:
        """Calculate overall confidence in the classification"""
        # Base confidence on number of indicators and score consistency
        num_indicators = len([s for s in scores.values() if s > 0])
        
        if num_indicators == 0:
            return 0.1  # Very low confidence if no indicators
        
        # Higher confidence with more indicators and higher scores
        confidence = min(0.95, 0.3 + (num_indicators * 0.1) + (sum(scores.values()) * 0.05))
        
        return confidence
    
    def classify_alert(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """Classify a single alert using enhanced logic"""
        try:
            threat_score = self.calculate_threat_score(alert_data)
            
            return {
                'classification': threat_score.classification.value,
                'confidence': threat_score.confidence,
                'threat_score': threat_score.total_score,
                'is_anomaly': threat_score.classification != ThreatLevel.BENIGN,
                'threat_breakdown': {
                    'attack_category_score': threat_score.attack_category_score,
                    'severity_score': threat_score.severity_score,
                    'confidence_score': threat_score.confidence_score,
                    'ip_reputation_score': threat_score.ip_reputation_score,
                    'behavioral_score': threat_score.behavioral_score,
                    'temporal_score': threat_score.temporal_score,
                    'contextual_score': threat_score.contextual_score
                },
                'reasoning': threat_score.reasoning,
                'enhanced_classification': True
            }
        except Exception as e:
            logger.error(f"Error classifying alert: {e}")
            return {
                'classification': 'benign',
                'confidence': 0.1,
                'threat_score': 0.0,
                'is_anomaly': False,
                'error': str(e),
                'reasoning': [f"Classification error: {str(e)}"],
                'threat_breakdown': {},
                'enhanced_classification': False
            }
    
    def process_batch(self, alerts_df: pd.DataFrame) -> pd.DataFrame:
        """Process a batch of alerts with enhanced classification"""
        logger.info(f"🔄 Processing {len(alerts_df)} alerts with enhanced classification...")
        
        enhanced_results = []
        
        for idx, row in alerts_df.iterrows():
            alert_data = row.to_dict()
            classification_result = self.classify_alert(alert_data)
            
            # Update the row with enhanced classification
            enhanced_row = row.copy()
            enhanced_row['enhanced_classification'] = classification_result['classification']
            enhanced_row['enhanced_confidence'] = classification_result['confidence']
            enhanced_row['threat_score'] = classification_result['threat_score']
            enhanced_row['enhanced_is_anomaly'] = classification_result['is_anomaly']
            enhanced_row['classification_reasoning'] = '; '.join(classification_result['reasoning'])
            
            enhanced_results.append(enhanced_row)
        
        result_df = pd.DataFrame(enhanced_results)
        
        # Generate summary statistics
        self._generate_classification_summary(result_df)
        
        return result_df
    
    def _generate_classification_summary(self, df: pd.DataFrame):
        """Generate summary of classification results"""
        if 'enhanced_classification' not in df.columns:
            return
        
        classification_counts = df['enhanced_classification'].value_counts()
        anomaly_count = df['enhanced_is_anomaly'].sum()
        
        logger.info("📊 Enhanced Classification Results:")
        for classification, count in classification_counts.items():
            percentage = (count / len(df)) * 100
            logger.info(f"   {classification}: {count:,} alerts ({percentage:.1f}%)")
        
        logger.info(f"🚨 Total Anomalies Detected: {anomaly_count:,} ({anomaly_count/len(df)*100:.1f}%)")
        
        # Show threat score distribution
        if 'threat_score' in df.columns:
            high_threat = df[df['threat_score'] >= 6.0]
            medium_threat = df[(df['threat_score'] >= 4.0) & (df['threat_score'] < 6.0)]
            low_threat = df[(df['threat_score'] >= 2.0) & (df['threat_score'] < 4.0)]
            
            logger.info(f"🎯 Threat Score Distribution:")
            logger.info(f"   High Threat (≥6.0): {len(high_threat):,} alerts")
            logger.info(f"   Medium Threat (4.0-5.9): {len(medium_threat):,} alerts")
            logger.info(f"   Low Threat (2.0-3.9): {len(low_threat):,} alerts")

def main():
    """Main function to demonstrate enhanced classification"""
    print("🚀 Enhanced Classification Engine Demo")
    print("=" * 60)
    
    # Initialize the engine
    engine = EnhancedClassificationEngine()
    
    # Load the TAA data
    try:
        df = pd.read_csv("taa_extracted_data_20250918_115033.csv")
        print(f"📊 Loaded {len(df):,} TAA alerts")
        
        # Process a sample of alerts
        sample_size = min(1000, len(df))
        sample_df = df.head(sample_size)
        
        print(f"🔄 Processing {sample_size:,} alerts with enhanced classification...")
        
        # Apply enhanced classification
        enhanced_df = engine.process_batch(sample_df)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"enhanced_classification_results_{timestamp}.csv"
        enhanced_df.to_csv(output_file, index=False)
        
        print(f"💾 Enhanced classification results saved to: {output_file}")
        
        # Show comparison
        print("\n📊 Classification Comparison:")
        print("Original (Broken) Classification:")
        original_counts = sample_df['classification'].value_counts()
        for classification, count in original_counts.items():
            print(f"   {classification}: {count:,} alerts")
        
        print("\nEnhanced (Fixed) Classification:")
        enhanced_counts = enhanced_df['enhanced_classification'].value_counts()
        for classification, count in enhanced_counts.items():
            print(f"   {classification}: {count:,} alerts")
        
        print("\n✅ Enhanced Classification Engine Demo Complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Demo failed: {e}")

if __name__ == "__main__":
    main()


