#!/usr/bin/env python3
"""
Real-World Case Study: TAA Agent LLM Analysis - FALSE POSITIVE
==============================================================

This demonstrates how the TAA agent correctly identifies a false positive
using LLM analysis in a production SOC environment.

Case: Legitimate PowerShell Script Execution for System Administration
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FalsePositiveTAACase:
    """False positive case study for TAA agent LLM analysis"""
    
    def __init__(self):
        self.case_id = "CASE-2024-002"
        self.alert_data = self._create_false_positive_alert()
        
    def _create_false_positive_alert(self) -> Dict[str, Any]:
        """Create a realistic false positive alert"""
        
        return {
            "alert_id": "ALERT-2024-07-13-002",
            "timestamp": "2024-07-13T15:45:30Z",
            "source": "EDR_Sensor_002",
            "severity": "medium",
            "category": "suspicious_activity",
            "log_data": {
                "event_type": "process_creation",
                "process_name": "powershell.exe",
                "process_id": 23456,
                "parent_process": "svchost.exe",
                "command_line": "powershell.exe -ExecutionPolicy Bypass -File C:\\Scripts\\deploy_updates.ps1 -ServerList servers.txt -UpdateType security",
                "user": "CORP\\admin.smith",
                "hostname": "ADMIN-WORKSTATION-01",
                "src_ip": "192.168.1.50",
                "dst_ip": "192.168.1.100",
                "dst_port": 5985,
                "domain": "internal.corp.com",
                "file_hash": "f1e2d3c4b5a6789012345678901234567890efgh",
                "file_path": "C:\\Scripts\\deploy_updates.ps1"
            },
            "indicators": [
                "powershell_execution",
                "bypass_execution_policy",
                "network_connection",
                "script_execution"
            ],
            "metadata": {
                "detection_rule": "PowerShell Execution with Network Activity",
                "rule_id": "PS-002",
                "confidence_score": 0.65
            }
        }
    
    def _enrich_alert_with_threat_intelligence(self, alert: Dict) -> Dict:
        """Simulate threat intelligence enrichment for false positive"""
        
        enriched_data = {
            "src_ip_intel": {
                "reputation_score": 95,
                "is_malicious": False,
                "categories": ["corporate", "admin"],
                "country": "US",
                "asn": "AS12345",
                "last_reported": None
            },
            "dst_ip_intel": {
                "reputation_score": 90,
                "is_malicious": False,
                "categories": ["corporate"],
                "country": "US",
                "asn": "AS12345",
                "last_reported": None
            },
            "domain_intel": {
                "reputation_score": 95,
                "is_malicious": False,
                "categories": ["corporate"],
                "last_analysis_date": "2024-07-10T12:00:00Z",
                "creation_date": "2020-03-15T10:00:00Z",
                "registrar": "Corporate IT Department"
            },
            "file_intel": {
                "reputation_score": 95,
                "is_malicious": False,
                "categories": ["legitimate_script"],
                "detection_ratio": "0/70",
                "last_analysis_date": "2024-07-13T08:00:00Z"
            },
            "historical_context": {
                "previous_alerts": 0,
                "previous_incidents": 0,
                "first_seen": "2024-01-15T09:00:00Z",
                "last_seen": "2024-07-13T15:45:30Z",
                "recent_activity": [
                    "Regular scheduled script execution",
                    "Normal administrative activity"
                ]
            },
            "user_context": {
                "role": "System Administrator",
                "department": "IT Operations",
                "privilege_level": "admin",
                "usual_activity": "System maintenance and updates",
                "last_login": "2024-07-13T08:30:00Z"
            }
        }
        
        return enriched_data
    
    def _create_llm_prompt(self, enriched_alert: Dict) -> str:
        """Create a detailed LLM prompt for false positive analysis"""
        
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
1. The PowerShell command is executing a legitimate script (deploy_updates.ps1) for system administration
2. The user (admin.smith) is a system administrator with legitimate privileges
3. The destination IP (192.168.1.100) is an internal corporate server
4. The domain "internal.corp.com" is a legitimate internal corporate domain
5. The file hash has a 0/70 detection ratio (no antivirus flags)
6. There's no historical suspicious activity from this user or host
7. The command line shows legitimate administrative parameters (-ServerList, -UpdateType)
8. The execution policy bypass is common for legitimate administrative scripts

RESPOND ONLY with a valid JSON object. Do not include any text, explanation, or markdown formatting before or after the JSON. The output must be a single valid JSON object matching this structure:

{{
  "is_true_positive": true/false,
  "severity": "high/medium/low",
  "confidence": 0.XX,
  "attack_type": "malware/phishing/brute_force/credential_theft/privilege_escalation/data_exfiltration/legitimate_admin/other",
  "indicators": ["indicator1", "indicator2", "indicator3"],
  "recommended_actions": ["action1", "action2", "action3"],
  "analysis_summary": "Detailed explanation of why this is a true/false positive and what type of activity this represents"
}}
"""
        return prompt
    
    def _simulate_llm_analysis(self, prompt: str) -> Dict:
        """Simulate LLM analysis response for false positive"""
        
        # This simulates what a real LLM would return for this false positive case
        llm_response = {
            "is_true_positive": False,
            "severity": "low",
            "confidence": 0.88,
            "attack_type": "legitimate_admin",
            "indicators": [
                "legitimate_powershell_script",
                "internal_network_communication",
                "admin_user_execution",
                "scheduled_maintenance_activity",
                "no_malicious_indicators"
            ],
            "recommended_actions": [
                "No immediate action required",
                "Consider whitelisting this script for future executions",
                "Monitor for any unusual variations in script execution",
                "Verify script integrity periodically",
                "Document this as legitimate administrative activity"
            ],
            "analysis_summary": "This is a false positive representing legitimate system administration activity. The PowerShell command is executing a legitimate deployment script (deploy_updates.ps1) with proper administrative parameters. The user (admin.smith) is a system administrator with appropriate privileges, and the destination is an internal corporate server. The script hash has no antivirus detections, and there's no historical suspicious activity. The execution policy bypass is common and necessary for legitimate administrative scripts. This appears to be routine system maintenance and update deployment activity."
        }
        
        return llm_response
    
    def _make_final_decision(self, llm_analysis: Dict, enriched_data: Dict) -> Dict:
        """Combine LLM analysis with threat intelligence for final decision"""
        
        decision = llm_analysis.copy()
        
        # Enhance confidence based on threat intelligence
        ti_confidence_boost = 0.0
        
        # Check source IP intelligence (admin workstation)
        src_ip_intel = enriched_data["src_ip_intel"]
        if not src_ip_intel["is_malicious"] and "admin" in src_ip_intel["categories"]:
            ti_confidence_boost += 0.05
        
        # Check destination IP intelligence (internal server)
        dst_ip_intel = enriched_data["dst_ip_intel"]
        if not dst_ip_intel["is_malicious"]:
            ti_confidence_boost += 0.05
        
        # Check domain intelligence (internal corporate domain)
        domain_intel = enriched_data["domain_intel"]
        if not domain_intel["is_malicious"]:
            ti_confidence_boost += 0.05
        
        # Check file intelligence (legitimate script)
        file_intel = enriched_data["file_intel"]
        if not file_intel["is_malicious"]:
            ti_confidence_boost += 0.08
        
        # Check historical context (no previous incidents)
        historical = enriched_data["historical_context"]
        if historical["previous_incidents"] == 0:
            ti_confidence_boost += 0.05
        
        # Check user context (legitimate admin)
        user_context = enriched_data["user_context"]
        if user_context["role"] == "System Administrator":
            ti_confidence_boost += 0.05
        
        # Apply confidence boost
        decision["confidence"] = min(1.0, decision["confidence"] + ti_confidence_boost)
        
        # Adjust severity based on final confidence
        if decision["confidence"] >= 0.9:
            decision["severity"] = "low"
        elif decision["confidence"] >= 0.7:
            decision["severity"] = "low"
        else:
            decision["severity"] = "medium"
        
        decision["final_confidence"] = decision["confidence"]
        decision["ti_confidence_boost"] = ti_confidence_boost
        decision["timestamp"] = datetime.now().isoformat()
        
        return decision
    
    def run_case_study(self):
        """Run the complete false positive case study"""
        
        logger.info(f"=== TAA Agent False Positive Case Study: {self.case_id} ===")
        logger.info(f"Alert ID: {self.alert_data['alert_id']}")
        logger.info(f"Event: {self.alert_data['log_data']['event_type']}")
        logger.info(f"Process: {self.alert_data['log_data']['process_name']}")
        logger.info(f"User: {self.alert_data['log_data']['user']}")
        logger.info(f"Host: {self.alert_data['log_data']['hostname']}")
        
        print("\n" + "="*80)
        print("STEP 1: ORIGINAL ALERT DATA (FALSE POSITIVE)")
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
        print("FALSE POSITIVE CASE STUDY SUMMARY")
        print("="*80)
        print(f"Alert ID: {self.alert_data['alert_id']}")
        print(f"Decision: {'TRUE POSITIVE' if final_decision['is_true_positive'] else 'FALSE POSITIVE'}")
        print(f"Severity: {final_decision['severity'].upper()}")
        print(f"Confidence: {final_decision['final_confidence']:.2%}")
        print(f"Activity Type: {final_decision['attack_type']}")
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
    """Run the false positive case study"""
    case = FalsePositiveTAACase()
    results = case.run_case_study()
    
    # Save results to file
    with open("taa_false_positive_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nFalse positive results saved to: taa_false_positive_results.json")

if __name__ == "__main__":
    main() 