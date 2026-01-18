"""
GATRA Super Agent - Master Orchestrator Implementation
Version: 2.0
Author: Ghifi - PT Numedia Atrya Dinamika
Date: January 10, 2026
"""

import asyncio
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from abc import ABC, abstractmethod
import os
# Optional runtime dependency: allow importing classifier without aiohttp installed
try:
    from aiohttp import web  # type: ignore
except ImportError:  # pragma: no cover
    web = None  # Only required when running the API server
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('GATRASuperAgent')


class AgentDomain(Enum):
    """Specialized agent domains"""
    THREAT_DETECTION = "threat_detection"
    RESPONSE_OPS = "response_ops"
    FORENSICS = "forensics"
    COMPLIANCE = "compliance"
    OFFSEC = "offensive_security"


class RiskLevel(Enum):
    """Risk severity levels"""
    CRITICAL = 90
    HIGH = 70
    MEDIUM = 50
    LOW = 30
    INFO = 10


class ResponseMode(Enum):
    """Response execution modes"""
    AUTONOMOUS = "autonomous"
    ASSISTED = "assisted"
    MANUAL = "manual"


@dataclass
class SecurityEvent:
    """Security event data structure"""
    event_id: str
    event_type: str
    timestamp: datetime
    source: str
    severity: int
    data: Dict[str, Any]
    affected_assets: List[str]
    raw_log: str
    
    def to_dict(self) -> Dict:
        return {
            'event_id': self.event_id,
            'event_type': self.event_type,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'severity': self.severity,
            'data': self.data,
            'affected_assets': self.affected_assets,
            'raw_log': self.raw_log
        }


@dataclass
class OrchestrationIntent:
    """Parsed intent from security event"""
    event_type: str
    required_domains: List[AgentDomain]
    priority: int
    context: Dict[str, Any]
    expected_outcome: str


@dataclass
class AgentTask:
    """Task assigned to specialized agent"""
    task_id: str
    domain: AgentDomain
    action: str
    parameters: Dict[str, Any]
    priority: int
    timeout_seconds: int = 300


@dataclass
class AgentResult:
    """Result from specialized agent execution"""
    task_id: str
    agent_id: str
    domain: AgentDomain
    success: bool
    findings: Dict[str, Any]
    confidence: float
    execution_time: float
    recommendations: List[str]


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    return " ".join(s.lower().split())


def extract_iocs(event_data: Dict[str, Any], iocs: Optional[List[str]] = None) -> List[str]:
    collected: List[str] = []
    seen = set()

    def _add(value: Any) -> None:
        if not value:
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                _add(item)
            return
        text = str(value).strip()
        if not text or text in seen:
            return
        seen.add(text)
        collected.append(text)

    if iocs:
        _add(iocs)

    if isinstance(event_data, dict):
        for key in ("iocs", "indicators", "indicator", "ioc", "ioc_list", "threat_indicators"):
            if key in event_data:
                _add(event_data.get(key))

    return collected


def _require_aiohttp() -> None:
    if web is None:
        raise RuntimeError(
            "aiohttp is not installed. Install it to run the API server: pip install aiohttp"
        )


def classify_threat(
    event_type: str,
    event_data: Dict[str, Any],
    iocs: Optional[List[str]] = None,
    raw_log: str = ""
) -> Dict[str, Any]:
    """
    Pure, deterministic threat classifier.
    Returns threat_type, severity, mitre_tactics, matched_signatures, and classification_reason.
    """
    if not isinstance(event_data, dict):
        event_data = {"raw_event_data": event_data}

    event_type_norm = normalize_text(event_type)
    raw_log_norm = normalize_text(raw_log)

    try:
        payload_text = json.dumps(event_data, sort_keys=True, default=str)
    except TypeError:
        payload_text = str(event_data)

    ioc_list = extract_iocs(event_data, iocs)
    combined_text = " ".join(
        [
            event_type_norm,
            raw_log_norm,
            normalize_text(payload_text),
            normalize_text(" ".join(ioc_list)),
        ]
    ).strip()

    def _contains_any(text: str, keywords: List[str]) -> bool:
        return any(keyword in text for keyword in keywords)

    def _to_number(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(str(value).strip())
        except (TypeError, ValueError):
            return None

    login_keywords = [
        "login",
        "logon",
        "4625",
        "mfa",
        "impossible travel",
        "impossible_travel",
        "new device",
        "new_device",
        "failed password",
        "failed_password",
    ]
    c2_keywords = [
        "beacon",
        "c2",
        "command_and_control",
        "command and control",
        "dns tunneling",
        "dns_tunneling",
        "callback",
    ]
    exfil_keywords = [
        "exfil",
        "bucket",
        "s3",
        "cloudtrail",
        "large upload",
        "data transfer",
    ]

    login_signals: List[str] = []
    c2_signals: List[str] = []
    exfil_signals: List[str] = []

    if _contains_any(combined_text, login_keywords):
        login_signals.append("login_keywords")
    mfa_failures = _to_number(event_data.get("mfa_failures"))
    if mfa_failures is not None and mfa_failures >= 1:
        login_signals.append("mfa_failures")
    geo_from = normalize_text(event_data.get("geo_from"))
    geo_to = normalize_text(event_data.get("geo_to"))
    if geo_from and geo_to and geo_from != geo_to:
        login_signals.append("geo_jump")
    logon_type = _to_number(event_data.get("logon_type"))
    if logon_type in (3, 10):
        login_signals.append("logon_type")
    action = normalize_text(event_data.get("action"))
    if action == "failure":
        login_signals.append("action_failure")

    if _contains_any(combined_text, c2_keywords):
        c2_signals.append("c2_keywords")
    if ioc_list:
        c2_signals.append("ioc_presence")
    network_keys = (
        "dest_domain",
        "destination_domain",
        "dest_ip",
        "destination_ip",
        "ja3",
        "uri",
        "uri_path",
        "url",
        "domain",
        "hostname",
        "host",
    )
    if any(event_data.get(key) for key in network_keys):
        c2_signals.append("network_indicators")

    if _contains_any(combined_text, exfil_keywords):
        exfil_signals.append("exfil_keywords")
    storage_keys = ("bucket", "object", "service_account", "service_account_name", "serviceaccount")
    if any(event_data.get(key) for key in storage_keys):
        exfil_signals.append("storage_fields")
    volume_bytes_keys = (
        "bytes",
        "bytes_uploaded",
        "bytes_sent",
        "bytes_transferred",
        "bytes_out",
        "bytes_in",
    )
    volume_records_keys = ("records", "record_count", "rows", "row_count", "object_count")
    bytes_threshold = 10_000_000
    records_threshold = 10_000
    max_bytes = max(
        (_to_number(event_data.get(key)) or 0 for key in volume_bytes_keys),
        default=0,
    )
    max_records = max(
        (_to_number(event_data.get(key)) or 0 for key in volume_records_keys),
        default=0,
    )
    if max_bytes >= bytes_threshold or max_records >= records_threshold:
        exfil_signals.append("volume_indicators")

    login_score = len(login_signals)
    c2_score = len(c2_signals)
    exfil_score = len(exfil_signals)

    classification = "unknown_security_event"
    tie_breaker_applied = False
    tie_breaker_rule = "highest_score"
    if max(login_score, c2_score, exfil_score) == 0:
        if event_type_norm == "unauthorized_access":
            classification = "login_anomaly"
            tie_breaker_applied = True
            tie_breaker_rule = "fallback_event_type_unauthorized_access"
        else:
            classification = "unknown_security_event"
            tie_breaker_rule = "no_signals_default_unknown"
    else:
        scores = {
            "data_exfiltration": exfil_score,
            "command_and_control": c2_score,
            "login_anomaly": login_score,
        }
        best_score = max(scores.values())
        candidates = [name for name, score in scores.items() if score == best_score]
        for name in ("data_exfiltration", "command_and_control", "login_anomaly"):
            if name in candidates:
                classification = name
                break
        tie_breaker_applied = len(candidates) > 1
        if tie_breaker_applied:
            tie_breaker_rule = "priority_exfil_c2_login"

    if classification == "unknown_security_event":
        reason_parts = ["No strong signals detected"]
        classification_summary = "No strong signals detected."
    elif classification == "login_anomaly":
        reason_map = {
            "login_keywords": "Login keywords",
            "mfa_failures": "MFA failures",
            "geo_jump": "Geo jump",
            "logon_type": "Logon type",
            "action_failure": "Failure action",
        }
        reason_parts = [reason_map[signal] for signal in login_signals]
        if not reason_parts:
            reason_parts = ["Authentication anomalies"]
        classification_summary = f"Login anomaly signals: {', '.join(reason_parts)}."
    elif classification == "command_and_control":
        reason_map = {
            "c2_keywords": "C2 keywords",
            "ioc_presence": "IOC list",
            "network_indicators": "Network indicators",
        }
        reason_parts = [reason_map[signal] for signal in c2_signals]
        if not reason_parts:
            reason_parts = ["C2 indicators"]
        classification_summary = f"C2 indicators: {', '.join(reason_parts)}."
    else:
        reason_map = {
            "exfil_keywords": "Exfil keywords",
            "storage_fields": "Storage fields",
            "volume_indicators": "Large transfer",
        }
        reason_parts = [reason_map[signal] for signal in exfil_signals]
        if not reason_parts:
            reason_parts = ["Exfiltration indicators"]
        classification_summary = f"Exfiltration signals: {', '.join(reason_parts)}."

    reason_line = " + ".join(reason_parts)
    classification_reason = {
        "login_score": login_score,
        "c2_score": c2_score,
        "exfil_score": exfil_score,
        "login_signals": login_signals,
        "c2_signals": c2_signals,
        "exfil_signals": exfil_signals,
        "tie_breaker_applied": tie_breaker_applied,
        "tie_breaker_rule": tie_breaker_rule,
        "summary": classification_summary,
    }

    if classification == "command_and_control":
        mitre_tactics = ["TA0011"]
        severity = 85
        matched_signatures = ["ET-MALWARE-2024-001"]
    elif classification == "data_exfiltration":
        mitre_tactics = ["TA0010"]
        severity = 80
        matched_signatures = ["CLOUD-EXFIL-S3-ANOMALY-001"]
    elif classification == "unknown_security_event":
        mitre_tactics = []
        severity = 30
        matched_signatures = ["NO-SIGNALS-000"]
    else:
        mitre_tactics = ["TA0006"]
        severity = 60
        matched_signatures = ["AUTH-ANOMALY-2024-LOGIN-001"]

    return {
        "threat_type": classification,
        "severity": severity,
        "mitre_tactics": mitre_tactics,
        "matched_signatures": matched_signatures,
        "classification_reason": classification_reason,
        "reason": reason_line,
        "classification_summary": classification_summary,
    }


class SpecializedAgent(ABC):
    """Abstract base class for specialized agents"""
    
    def __init__(self, agent_id: str, domain: AgentDomain):
        self.agent_id = agent_id
        self.domain = domain
        self.logger = logging.getLogger(f'Agent.{domain.value}.{agent_id}')
    
    @abstractmethod
    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute assigned task"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check agent health status"""
        pass


class ThreatDetectionAgent(SpecializedAgent):
    """Network traffic and threat analysis agent"""
    
    async def execute(self, task: AgentTask) -> AgentResult:
        start_time = asyncio.get_event_loop().time()
        params = task.parameters or {}
        event_data = params.get("event_data") or params or {}
        iocs = params.get("iocs") or params.get("indicators") or []
        event_type = params.get("event_type") or params.get("intent_event_type") or task.action or ""
        raw_log = params.get("raw_log") or params.get("raw_text") or params.get("input") or ""

        classification = classify_threat(
            event_type=event_type,
            event_data=event_data,
            iocs=iocs,
            raw_log=raw_log,
        )

        findings = {
            "event_type": event_type,
            "iocs": iocs,
            **classification,
        }
        
        recommendations = [
            'Isolate affected endpoint',
            'Block IOCs at network perimeter',
            'Investigate lateral movement'
        ]
        
        execution_time = asyncio.get_event_loop().time() - start_time
        
        return AgentResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            domain=self.domain,
            success=True,
            findings=findings,
            confidence=0.87,
            execution_time=execution_time,
            recommendations=recommendations
        )
    
    async def health_check(self) -> bool:
        return True


class ResponseOpsAgent(SpecializedAgent):
    """Incident response and containment agent"""
    
    async def execute(self, task: AgentTask) -> AgentResult:
        start_time = asyncio.get_event_loop().time()
        
        self.logger.info(f"Executing response action: {task.action}")
        await asyncio.sleep(0.3)

        threat_type = str(task.parameters.get('threat_type', '')).lower()
        if threat_type in ['command_and_control', 'c2', 'command-and-control', 'malware', 'malware_detection', 'ransomware', 'trojan']:
            recommendations = [
                'Isolate affected endpoints from the network',
                'Block C2 domains/IPs at perimeter and DNS',
                'Collect beaconing telemetry for scope analysis',
                'Reset credentials for impacted accounts'
            ]
        elif threat_type in ['login_anomaly', 'unauthorized_access', 'failed_login', 'brute_force'] or 'login' in threat_type:
            recommendations = [
                'Force password reset and enforce MFA',
                'Lock or step-up authenticate impacted accounts',
                'Block suspicious source IPs and geos',
                'Review authentication logs for related attempts'
            ]
        elif 'exfiltration' in threat_type or 'data_leak' in threat_type:
            recommendations = [
                'Block outbound data transfers to suspicious destinations',
                'Preserve network flow logs and proxy records',
                'Initiate DLP review on impacted data stores',
                'Engage legal/compliance for potential disclosure'
            ]
        elif 'vulnerability' in threat_type or 'scan' in threat_type:
            recommendations = [
                'Block scanning sources and rate-limit probes',
                'Validate exposed services and patch high-risk CVEs',
                'Run targeted vulnerability validation on affected assets'
            ]
        else:
            recommendations = [
                'Monitor for additional malicious activity',
                'Conduct post-incident review'
            ]
        
        findings = {
            'action_taken': task.action,
            'affected_systems': task.parameters.get('targets', []),
            'containment_status': 'successful',
            'rollback_available': True,
            'threat_type': threat_type
        }
        
        execution_time = asyncio.get_event_loop().time() - start_time
        
        return AgentResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            domain=self.domain,
            success=True,
            findings=findings,
            confidence=0.95,
            execution_time=execution_time,
            recommendations=recommendations
        )
    
    async def health_check(self) -> bool:
        return True


class ForensicsAgent(SpecializedAgent):
    """Digital forensics and investigation agent"""
    
    async def execute(self, task: AgentTask) -> AgentResult:
        start_time = asyncio.get_event_loop().time()
        
        self.logger.info(f"Conducting forensics: {task.action}")
        await asyncio.sleep(1.0)  # Forensics takes longer
        
        findings = {
            'timeline': [
                {'timestamp': '2026-01-10T05:30:00Z', 'event': 'Initial compromise'},
                {'timestamp': '2026-01-10T05:45:00Z', 'event': 'Privilege escalation'},
                {'timestamp': '2026-01-10T06:00:00Z', 'event': 'Data exfiltration attempt'}
            ],
            'attack_path': 'web_vuln -> shell -> privesc -> exfil',
            'evidence_collected': ['memory_dump.bin', 'network_capture.pcap'],
            'attribution_indicators': ['APT-X toolkit signatures']
        }
        
        recommendations = [
            'Preserve evidence for legal proceedings',
            'Patch vulnerable web application',
            'Review privileged account access'
        ]
        
        execution_time = asyncio.get_event_loop().time() - start_time
        
        return AgentResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            domain=self.domain,
            success=True,
            findings=findings,
            confidence=0.82,
            execution_time=execution_time,
            recommendations=recommendations
        )
    
    async def health_check(self) -> bool:
        return True


class ComplianceAgent(SpecializedAgent):
    """Compliance monitoring and reporting agent"""
    
    async def execute(self, task: AgentTask) -> AgentResult:
        start_time = asyncio.get_event_loop().time()
        
        self.logger.info(f"Checking compliance: {task.action}")
        await asyncio.sleep(0.4)
        
        findings = {
            'frameworks_affected': ['ISO27001', 'PCI-DSS'],
            'violations_detected': [
                {'control': 'A.12.6.1', 'description': 'Unpatched vulnerability'},
                {'control': '6.5.10', 'description': 'Insufficient logging'}
            ],
            'breach_notification_required': True,
            'notification_deadline': '72 hours'
        }
        
        recommendations = [
            'Initiate breach notification procedure',
            'Document incident for audit trail',
            'Update risk register'
        ]
        
        execution_time = asyncio.get_event_loop().time() - start_time
        
        return AgentResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            domain=self.domain,
            success=True,
            findings=findings,
            confidence=0.92,
            execution_time=execution_time,
            recommendations=recommendations
        )
    
    async def health_check(self) -> bool:
        return True


class OffensiveSecurityAgent(SpecializedAgent):
    """Automated penetration testing agent"""
    
    async def execute(self, task: AgentTask) -> AgentResult:
        start_time = asyncio.get_event_loop().time()
        
        self.logger.info(f"Running offensive security test: {task.action}")
        await asyncio.sleep(0.8)
        
        findings = {
            'vulnerabilities_found': [
                {'cve': 'CVE-2024-1234', 'severity': 'high', 'exploitable': True},
                {'cve': 'CVE-2024-5678', 'severity': 'medium', 'exploitable': False}
            ],
            'attack_vectors_tested': ['web', 'network', 'cloud'],
            'successful_exploits': 1,
            'detection_rate': '50%'  # Only 50% of exploits were detected
        }
        
        recommendations = [
            'Patch CVE-2024-1234 immediately',
            'Improve detection rules for web attacks',
            'Conduct security awareness training'
        ]
        
        execution_time = asyncio.get_event_loop().time() - start_time
        
        return AgentResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            domain=self.domain,
            success=True,
            findings=findings,
            confidence=0.88,
            execution_time=execution_time,
            recommendations=recommendations
        )
    
    async def health_check(self) -> bool:
        return True


class RiskBasedDecisionEngine:
    """Risk assessment and autonomous decision-making engine"""
    
    def __init__(self, autonomy_level: int = 75):
        self.autonomy_level = autonomy_level  # 0-100
        self.threat_thresholds = {
            RiskLevel.CRITICAL: 90,
            RiskLevel.HIGH: 70,
            RiskLevel.MEDIUM: 50,
            RiskLevel.LOW: 30,
            RiskLevel.INFO: 10
        }
    
    def assess_risk(self, event: SecurityEvent, agent_results: List[AgentResult]) -> int:
        """Calculate composite risk score"""
        
        # Base score from event severity
        base_score = event.severity
        
        # Adjust based on agent findings
        threat_multiplier = 1.0
        for result in agent_results:
            if result.domain == AgentDomain.THREAT_DETECTION:
                threat_severity = result.findings.get('severity', 50)
                threat_multiplier = max(threat_multiplier, threat_severity / 100)
            
            if result.domain == AgentDomain.FORENSICS:
                if 'APT' in str(result.findings):
                    threat_multiplier *= 1.2
        
        # Asset criticality
        critical_assets = ['production_db', 'payment_gateway', 'customer_data']
        if any(asset in event.affected_assets for asset in critical_assets):
            threat_multiplier *= 1.3
        
        final_score = min(100, int(base_score * threat_multiplier))
        
        return final_score
    
    def decide_response_mode(self, risk_score: int) -> ResponseMode:
        """Determine if response should be autonomous, assisted, or manual"""
        
        if risk_score >= self.threat_thresholds[RiskLevel.CRITICAL]:
            return ResponseMode.AUTONOMOUS if self.autonomy_level >= 80 else ResponseMode.ASSISTED
        
        elif risk_score >= self.threat_thresholds[RiskLevel.HIGH]:
            return ResponseMode.AUTONOMOUS if self.autonomy_level >= 60 else ResponseMode.ASSISTED
        
        elif risk_score >= self.threat_thresholds[RiskLevel.MEDIUM]:
            return ResponseMode.ASSISTED
        
        else:
            return ResponseMode.MANUAL
    
    def generate_recommendations(self, risk_score: int, agent_results: List[AgentResult]) -> List[str]:
        """Synthesize recommendations from all agents"""
        
        all_recommendations = []
        for result in agent_results:
            all_recommendations.extend(result.recommendations)
        
        # Deduplicate and prioritize
        unique_recommendations = list(set(all_recommendations))
        
        # Add risk-based recommendations
        if risk_score >= self.threat_thresholds[RiskLevel.CRITICAL]:
            unique_recommendations.insert(0, 'CRITICAL: Immediate executive escalation required')
        
        return unique_recommendations[:10]  # Top 10 recommendations


class KnowledgeSynthesizer:
    """Synthesizes insights from multiple agent domains"""
    
    def synthesize(self, agent_results: List[AgentResult]) -> Dict[str, Any]:
        """Create unified intelligence from all agent findings"""
        
        synthesis = {
            'summary': '',
            'key_findings': [],
            'cross_domain_correlations': [],
            'unified_recommendations': [],
            'confidence_score': 0.0
        }
        
        # Extract key findings by domain
        findings_by_domain = {}
        for result in agent_results:
            findings_by_domain[result.domain.value] = result.findings
        
        # Cross-domain correlation: Threat detection + Compliance
        if (AgentDomain.THREAT_DETECTION.value in findings_by_domain and 
            AgentDomain.COMPLIANCE.value in findings_by_domain):
            
            threat_data = findings_by_domain[AgentDomain.THREAT_DETECTION.value]
            compliance_data = findings_by_domain[AgentDomain.COMPLIANCE.value]
            
            if compliance_data.get('breach_notification_required'):
                synthesis['cross_domain_correlations'].append({
                    'type': 'threat_compliance',
                    'insight': 'Data breach incident requires regulatory notification',
                    'deadline': compliance_data.get('notification_deadline'),
                    'frameworks': compliance_data.get('frameworks_affected')
                })
        
        # Cross-domain correlation: OffSec + Threat Detection
        if (AgentDomain.OFFSEC.value in findings_by_domain and 
            AgentDomain.THREAT_DETECTION.value in findings_by_domain):
            
            offsec_data = findings_by_domain[AgentDomain.OFFSEC.value]
            detection_rate = offsec_data.get('detection_rate', '0%')
            
            if int(detection_rate.rstrip('%')) < 75:
                synthesis['cross_domain_correlations'].append({
                    'type': 'detection_gap',
                    'insight': 'Low detection rate for penetration test exploits',
                    'detection_rate': detection_rate,
                    'action': 'Tune detection rules based on pentest findings'
                })
        
        # Calculate overall confidence
        confidence_scores = [r.confidence for r in agent_results]
        synthesis['confidence_score'] = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        # Generate summary
        synthesis['summary'] = self._generate_summary(findings_by_domain, synthesis['cross_domain_correlations'])
        
        return synthesis
    
    def _generate_summary(self, findings: Dict, correlations: List[Dict]) -> str:
        """Generate human-readable summary"""
        
        summary_parts = []
        
        if AgentDomain.THREAT_DETECTION.value in findings:
            threat = findings[AgentDomain.THREAT_DETECTION.value]
            summary_parts.append(f"Detected {threat.get('threat_type', 'security incident')}")
        
        if AgentDomain.FORENSICS.value in findings:
            forensics = findings[AgentDomain.FORENSICS.value]
            summary_parts.append(f"Attack path: {forensics.get('attack_path', 'unknown')}")
        
        if correlations:
            summary_parts.append(f"Found {len(correlations)} cross-domain correlations")
        
        return '. '.join(summary_parts) + '.'


class GATRASuperAgent:
    """
    Master orchestrator for GATRA cybersecurity platform
    Coordinates specialized agents across threat detection, response, forensics, compliance, and offensive security
    """
    
    def __init__(self, autonomy_level: int = 75):
        self.autonomy_level = autonomy_level
        self.logger = logging.getLogger('GATRASuperAgent')
        self.decision_engine = RiskBasedDecisionEngine(autonomy_level)
        self.knowledge_synthesizer = KnowledgeSynthesizer()
        self.specialized_agents = self._initialize_agents()
    
    def _initialize_agents(self) -> Dict[AgentDomain, List[SpecializedAgent]]:
        """Initialize all specialized agent pools"""
        
        agents = {
            AgentDomain.THREAT_DETECTION: [
                ThreatDetectionAgent('td-001', AgentDomain.THREAT_DETECTION),
                ThreatDetectionAgent('td-002', AgentDomain.THREAT_DETECTION),
            ],
            AgentDomain.RESPONSE_OPS: [
                ResponseOpsAgent('ro-001', AgentDomain.RESPONSE_OPS),
            ],
            AgentDomain.FORENSICS: [
                ForensicsAgent('fr-001', AgentDomain.FORENSICS),
            ],
            AgentDomain.COMPLIANCE: [
                ComplianceAgent('cp-001', AgentDomain.COMPLIANCE),
            ],
            AgentDomain.OFFSEC: [
                OffensiveSecurityAgent('os-001', AgentDomain.OFFSEC),
            ]
        }
        
        self.logger.info(f"Initialized {sum(len(pool) for pool in agents.values())} specialized agents")
        return agents
    
    async def orchestrate(self, event: SecurityEvent) -> Dict[str, Any]:
        """
        Main orchestration workflow
        
        1. Intent Recognition
        2. Risk Assessment
        3. Task Decomposition
        4. Agent Selection & Delegation
        5. Parallel Execution
        6. Knowledge Synthesis
        7. Autonomous Decision
        8. Execute & Monitor
        """
        
        orchestration_id = f"orch-{event.event_id}"
        self.logger.info(f"Starting orchestration {orchestration_id} for event {event.event_id}")
        
        # Step 1: Intent Recognition
        intent = self._recognize_intent(event)
        self.logger.info(f"Recognized intent: {intent.event_type}, domains: {[d.value for d in intent.required_domains]}")
        
        # Step 2: Initial Risk Assessment
        initial_risk = event.severity
        
        # Step 3: Task Decomposition
        tasks = self._decompose_tasks(intent, event)
        self.logger.info(f"Decomposed into {len(tasks)} tasks")
        
        # Step 4: Agent Selection & Delegation
        agent_assignments = self._assign_agents(tasks)
        
        # Step 5: Parallel Execution
        self.logger.info("Executing agents in parallel...")
        agent_results = await self._execute_parallel(agent_assignments)
        self.logger.info(f"Completed {len(agent_results)} agent executions")
        
        # Step 6: Knowledge Synthesis
        unified_insight = self.knowledge_synthesizer.synthesize(agent_results)

        td_result = next(
            (result for result in agent_results if result.domain == AgentDomain.THREAT_DETECTION),
            None
        )
        threat_classification = None
        if td_result:
            td_findings = td_result.findings or {}
            threat_classification = {
                'threat_type': td_findings.get('threat_type'),
                'classification_reason': td_findings.get('classification_reason'),
                'scores': td_findings.get('scores'),
                'signals': td_findings.get('signals'),
                'mitre_tactics': td_findings.get('mitre_tactics'),
                'matched_signatures': td_findings.get('matched_signatures'),
                'severity': td_findings.get('severity')
            }

        # Step 7: Final Risk Assessment
        final_risk_score = self.decision_engine.assess_risk(event, agent_results)
        if threat_classification and threat_classification.get('severity') is not None:
            try:
                classification_severity = int(threat_classification.get('severity'))
                final_risk_score = max(final_risk_score, classification_severity)
            except (TypeError, ValueError):
                pass
        response_mode = self.decision_engine.decide_response_mode(final_risk_score)
        
        # Step 8: Generate Recommendations
        recommendations = self.decision_engine.generate_recommendations(final_risk_score, agent_results)
        
        # Orchestration result
        result = {
            'orchestration_id': orchestration_id,
            'event_id': event.event_id,
            'timestamp': datetime.utcnow().isoformat(),
            'intent': {
                'event_type': intent.event_type,
                'domains_engaged': [d.value for d in intent.required_domains]
            },
            'risk_assessment': {
                'initial_score': initial_risk,
                'final_score': final_risk_score,
                'level': self._get_risk_level(final_risk_score).name
            },
            'agent_results': [
                {
                    'agent_id': r.agent_id,
                    'domain': r.domain.value,
                    'success': r.success,
                    'confidence': r.confidence,
                    'execution_time': r.execution_time
                }
                for r in agent_results
            ],
            'unified_insight': unified_insight,
            'response_mode': response_mode.value,
            'recommendations': recommendations,
            'autonomy_level': self.autonomy_level
        }

        if threat_classification:
            result['threat_classification'] = threat_classification
        
        self.logger.info(f"Orchestration {orchestration_id} complete: Risk={final_risk_score}, Mode={response_mode.value}")
        
        return result
    
    def _recognize_intent(self, event: SecurityEvent) -> OrchestrationIntent:
        """Classify event and determine required agent domains"""
        
        # Pattern matching for event types
        intent_patterns = {
            'malware_detection': {
                'keywords': ['malware', 'virus', 'trojan', 'ransomware'],
                'domains': [AgentDomain.THREAT_DETECTION, AgentDomain.RESPONSE_OPS, AgentDomain.FORENSICS]
            },
            'data_exfiltration': {
                'keywords': ['exfiltration', 'data_leak', 'unauthorized_transfer'],
                'domains': [AgentDomain.THREAT_DETECTION, AgentDomain.RESPONSE_OPS, AgentDomain.FORENSICS, AgentDomain.COMPLIANCE]
            },
            'unauthorized_access': {
                'keywords': ['brute_force', 'unauthorized', 'access_denied', 'failed_login'],
                'domains': [AgentDomain.THREAT_DETECTION, AgentDomain.RESPONSE_OPS]
            },
            'compliance_violation': {
                'keywords': ['policy_violation', 'compliance', 'audit'],
                'domains': [AgentDomain.COMPLIANCE, AgentDomain.RESPONSE_OPS]
            },
            'vulnerability_scan': {
                'keywords': ['vulnerability', 'cve', 'patch'],
                'domains': [AgentDomain.OFFSEC, AgentDomain.RESPONSE_OPS]
            }
        }
        
        # Determine event type
        event_type = event.event_type.lower()
        detected_pattern = None
        required_domains = []
        
        for pattern_name, pattern_config in intent_patterns.items():
            if any(keyword in event_type for keyword in pattern_config['keywords']):
                detected_pattern = pattern_name
                required_domains = pattern_config['domains']
                break
        
        # Default to threat detection if no pattern matched
        if not detected_pattern:
            detected_pattern = 'unknown_security_event'
            required_domains = [AgentDomain.THREAT_DETECTION]
        
        return OrchestrationIntent(
            event_type=detected_pattern,
            required_domains=required_domains,
            priority=self._calculate_priority(event),
            context={'original_event_type': event.event_type},
            expected_outcome='incident_resolution'
        )
    
    def _calculate_priority(self, event: SecurityEvent) -> int:
        """Calculate task priority (1-10, 10 = highest)"""
        
        if event.severity >= 90:
            return 10
        elif event.severity >= 70:
            return 8
        elif event.severity >= 50:
            return 5
        else:
            return 3
    
    def _decompose_tasks(self, intent: OrchestrationIntent, event: SecurityEvent) -> List[AgentTask]:
        """Break down intent into specific agent tasks"""
        
        tasks = []
        task_counter = 0
        
        for domain in intent.required_domains:
            task_counter += 1
            
            # Domain-specific task generation
            if domain == AgentDomain.THREAT_DETECTION:
                event_iocs = []
                for key in ['indicators', 'iocs', 'ioc_list']:
                    values = event.data.get(key, [])
                    if isinstance(values, list):
                        event_iocs.extend(values)
                task = AgentTask(
                    task_id=f"task-{event.event_id}-{task_counter}",
                    domain=domain,
                    action='analyze_threat',
                    parameters={
                        'event_data': event.data,
                        'iocs': event_iocs,
                        'intent_event_type': intent.event_type
                    },
                    priority=intent.priority,
                    timeout_seconds=300
                )
                tasks.append(task)
            
            elif domain == AgentDomain.RESPONSE_OPS:
                task = AgentTask(
                    task_id=f"task-{event.event_id}-{task_counter}",
                    domain=domain,
                    action='prepare_containment',
                    parameters={
                        'targets': event.affected_assets,
                        'threat_type': event.data.get('threat_type') or intent.event_type or event.event_type
                    },
                    priority=intent.priority,
                    timeout_seconds=180
                )
                tasks.append(task)
            
            elif domain == AgentDomain.FORENSICS:
                task = AgentTask(
                    task_id=f"task-{event.event_id}-{task_counter}",
                    domain=domain,
                    action='investigate_timeline',
                    parameters={
                        'affected_assets': event.affected_assets,
                        'time_window': '24h'
                    },
                    priority=intent.priority,
                    timeout_seconds=600
                )
                tasks.append(task)
            
            elif domain == AgentDomain.COMPLIANCE:
                task = AgentTask(
                    task_id=f"task-{event.event_id}-{task_counter}",
                    domain=domain,
                    action='check_compliance_impact',
                    parameters={
                        'frameworks': ['ISO27001', 'PCI-DSS', 'GDPR']
                    },
                    priority=intent.priority,
                    timeout_seconds=200
                )
                tasks.append(task)
            
            elif domain == AgentDomain.OFFSEC:
                task = AgentTask(
                    task_id=f"task-{event.event_id}-{task_counter}",
                    domain=domain,
                    action='validate_security_controls',
                    parameters={
                        'targets': event.affected_assets
                    },
                    priority=intent.priority,
                    timeout_seconds=400
                )
                tasks.append(task)
        
        return tasks
    
    def _assign_agents(self, tasks: List[AgentTask]) -> List[tuple[SpecializedAgent, AgentTask]]:
        """Assign tasks to available specialized agents"""
        
        assignments = []
        
        for task in tasks:
            # Get agent pool for this domain
            agent_pool = self.specialized_agents.get(task.domain, [])
            
            if not agent_pool:
                self.logger.warning(f"No agents available for domain {task.domain.value}")
                continue
            
            # Simple round-robin selection (in production, use load-based selection)
            selected_agent = agent_pool[0]
            assignments.append((selected_agent, task))
        
        return assignments
    
    async def _execute_parallel(self, assignments: List[tuple[SpecializedAgent, AgentTask]]) -> List[AgentResult]:
        """Execute all agent tasks in parallel"""
        
        # Create tasks for async execution
        async_tasks = [
            agent.execute(task)
            for agent, task in assignments
        ]
        
        # Execute all in parallel
        results = await asyncio.gather(*async_tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = [r for r in results if isinstance(r, AgentResult)]
        
        return valid_results
    
    def _get_risk_level(self, risk_score: int) -> RiskLevel:
        """Convert risk score to risk level"""
        
        if risk_score >= RiskLevel.CRITICAL.value:
            return RiskLevel.CRITICAL
        elif risk_score >= RiskLevel.HIGH.value:
            return RiskLevel.HIGH
        elif risk_score >= RiskLevel.MEDIUM.value:
            return RiskLevel.MEDIUM
        elif risk_score >= RiskLevel.LOW.value:
            return RiskLevel.LOW
        else:
            return RiskLevel.INFO


# Example usage
async def demo_orchestration():
    """Demonstrate GATRA Super Agent orchestration"""
    
    # Initialize Super Agent with 75% autonomy
    super_agent = GATRASuperAgent(autonomy_level=75)
    
    # Simulate a security event
    event = SecurityEvent(
        event_id='evt-20260110-001',
        event_type='malware_detection',
        timestamp=datetime.utcnow(),
        source='EDR_CrowdStrike',
        severity=85,
        data={
            'indicators': ['192.168.1.100', 'malicious.example.com'],
            'process': 'suspicious.exe',
            'user': 'compromised_user'
        },
        affected_assets=['workstation-42', 'file_server-01'],
        raw_log='[EDR] Malware detected on workstation-42...'
    )
    
    print("=" * 80)
    print("GATRA SUPER AGENT DEMONSTRATION")
    print("=" * 80)
    print(f"\nIncoming Security Event: {event.event_id}")
    print(f"Type: {event.event_type}")
    print(f"Severity: {event.severity}")
    print(f"Affected Assets: {', '.join(event.affected_assets)}")
    print("\nInitiating orchestration...\n")
    
    # Orchestrate response
    result = await super_agent.orchestrate(event)
    
    # Display results
    print("=" * 80)
    print("ORCHESTRATION RESULTS")
    print("=" * 80)
    print(f"\nOrchestration ID: {result['orchestration_id']}")
    print(f"Event Type Detected: {result['intent']['event_type']}")
    print(f"Domains Engaged: {', '.join(result['intent']['domains_engaged'])}")
    
    print(f"\nRisk Assessment:")
    print(f"  Initial Score: {result['risk_assessment']['initial_score']}")
    print(f"  Final Score: {result['risk_assessment']['final_score']}")
    print(f"  Risk Level: {result['risk_assessment']['level']}")
    
    print(f"\nAgent Execution Summary:")
    for agent in result['agent_results']:
        print(f"  [{agent['domain']}] {agent['agent_id']}: {'✓' if agent['success'] else '✗'} "
              f"(confidence: {agent['confidence']:.2f}, time: {agent['execution_time']:.3f}s)")
    
    print(f"\nUnified Insight:")
    print(f"  Summary: {result['unified_insight']['summary']}")
    print(f"  Overall Confidence: {result['unified_insight']['confidence_score']:.2f}")
    print(f"  Cross-Domain Correlations: {len(result['unified_insight']['cross_domain_correlations'])}")
    
    for correlation in result['unified_insight']['cross_domain_correlations']:
        print(f"    - {correlation['type']}: {correlation['insight']}")
    
    print(f"\nResponse Mode: {result['response_mode'].upper()}")
    print(f"Autonomy Level: {result['autonomy_level']}%")
    
    print(f"\nTop Recommendations:")
    for i, rec in enumerate(result['recommendations'][:5], 1):
        print(f"  {i}. {rec}")
    
    print("\n" + "=" * 80)
    
    return result


async def start_orchestrator_server():
    """Start the Orchestrator API server"""
    _require_aiohttp()
    super_agent = GATRASuperAgent(autonomy_level=int(os.getenv('AUTONOMY_LEVEL', 75)))
    port_env = os.getenv("ORCH_PORT") or os.getenv("PORT") or "8080"
    host = os.getenv("ORCH_HOST", "0.0.0.0")
    try:
        port = int(port_env)
    except (TypeError, ValueError):
        port = 8080
    
    routes = web.RouteTableDef()
    
    @routes.post('/api/v2/orchestrate')
    async def handle_orchestrate(request):
        try:
            data = await request.json()
            # Convert JSON to SecurityEvent
            event = SecurityEvent(
                event_id=data.get('event_id', f"evt-{int(asyncio.get_event_loop().time())}"),
                event_type=data.get('event_type', 'unknown'),
                timestamp=datetime.utcnow(),
                source=data.get('source', 'api'),
                severity=int(data.get('severity', 50)),
                data=data.get('data', {}),
                affected_assets=data.get('affected_assets', []),
                raw_log=data.get('raw_log', '')
            )
            
            result = await super_agent.orchestrate(event)
            return web.json_response(result)
        except Exception as e:
            logger.error(f"Orchestration error: {e}")
            return web.json_response({'error': str(e)}, status=500)

    @routes.get('/health')
    async def health_check(request):
        return web.json_response({'status': 'ok'})

    app = web.Application()
    app.add_routes(routes)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    
    logger.info("Starting Orchestrator API on %s:%s", host, port)
    await site.start()
    
    # Keep alive
    while True:
        await asyncio.sleep(3600)

async def start_agent_worker():
    """Start specialized agent worker loop"""
    domain = os.getenv('AGENT_DOMAIN', 'generic')
    logger.info(f"Starting Agent Worker for domain: {domain}")
    
    # In a real implementation, this would consume from Kafka
    # For now, we simulate being 'ready' and kept alive
    while True:
        logger.info(f"Agent {domain} heartbeat...")
        await asyncio.sleep(60)

if __name__ == "__main__":
    if os.getenv('AGENT_DOMAIN'):
        # This is an agent container
        asyncio.run(start_agent_worker())
    elif os.getenv('NEO4J_URI'): 
        # This is likely the orchestrator (has DB config)
        asyncio.run(start_orchestrator_server())
    else:
        # Local run / Demo
        asyncio.run(demo_orchestration())
