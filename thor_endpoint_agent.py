#!/usr/bin/env python3
"""
THOR Endpoint Scanning Agent
=============================
Mimics Nextron Systems THOR scanner capabilities:
- YARA rule scanning (file system, memory, processes)
- IOC matching (IPs, domains, file hashes, registry keys)
- Sigma rule evaluation for log events
- File integrity monitoring
- Process inspection and behavioral analysis
- Network connection analysis

Integration with AI-Driven SOC:
- Publishes findings to TAA for enrichment
- Triggers CRA playbooks for automated response
- Feeds detection data to CLA for model training
"""

import asyncio
import hashlib
import json
import logging
import os
import psutil
import re
import yara
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict

# Google Cloud imports
from google.cloud import pubsub_v1, firestore, bigquery

# LangGraph for workflow orchestration
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScanType(Enum):
    """Types of scans THOR can perform"""
    FILESYSTEM = "filesystem"
    MEMORY = "memory"
    PROCESS = "process"
    NETWORK = "network"
    REGISTRY = "registry"  # For Windows
    LOG_EVENTS = "log_events"
    FULL = "full"


class ThreatSeverity(Enum):
    """Threat severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class MatchType(Enum):
    """Type of threat match"""
    YARA = "yara"
    IOC_IP = "ioc_ip"
    IOC_DOMAIN = "ioc_domain"
    IOC_HASH = "ioc_hash"
    IOC_FILENAME = "ioc_filename"
    IOC_REGISTRY = "ioc_registry"
    SIGMA = "sigma"
    BEHAVIORAL = "behavioral"


@dataclass
class IOCSet:
    """Collection of Indicators of Compromise"""
    ips: Set[str]
    domains: Set[str]
    file_hashes: Dict[str, str]  # hash -> malware_name
    filenames: Set[str]
    registry_keys: Set[str]  # Windows registry paths
    mutex_names: Set[str]
    user_agents: Set[str]

    def to_dict(self) -> Dict:
        return {
            "ips": list(self.ips),
            "domains": list(self.domains),
            "file_hashes": self.file_hashes,
            "filenames": list(self.filenames),
            "registry_keys": list(self.registry_keys),
            "mutex_names": list(self.mutex_names),
            "user_agents": list(self.user_agents)
        }


@dataclass
class ThreatMatch:
    """Individual threat detection"""
    match_id: str
    timestamp: str
    match_type: MatchType
    severity: ThreatSeverity
    rule_name: str
    confidence: float
    target: str  # File path, IP, process name, etc.
    context: Dict[str, Any]
    remediation: List[str]
    false_positive_likelihood: float

    def to_dict(self) -> Dict:
        data = asdict(self)
        data['match_type'] = self.match_type.value
        data['severity'] = self.severity.value
        return data


@dataclass
class ScanResult:
    """Results from a THOR scan"""
    scan_id: str
    hostname: str
    start_time: str
    end_time: str
    scan_types: List[ScanType]
    total_files_scanned: int
    total_processes_scanned: int
    total_network_connections: int
    matches: List[ThreatMatch]
    iocs_detected: List[str]
    yara_rules_matched: List[str]
    sigma_rules_matched: List[str]
    scan_statistics: Dict[str, Any]

    def to_dict(self) -> Dict:
        data = asdict(self)
        data['scan_types'] = [st.value for st in self.scan_types]
        data['matches'] = [m.to_dict() for m in self.matches]
        return data


class THORState(TypedDict):
    """LangGraph state for THOR scanning workflow"""
    scan_request: Dict[str, Any]
    scan_config: Dict[str, Any]
    yara_rules: Optional[Any]
    ioc_set: Optional[IOCSet]
    scan_results: Optional[ScanResult]
    enrichment_data: Optional[Dict[str, Any]]
    workflow_status: str
    errors: List[str]


class THOREndpointAgent:
    """
    THOR-like endpoint scanning agent

    Capabilities:
    - Multi-format YARA scanning (compiled rules, plaintext)
    - IOC matching across filesystem, memory, network
    - Sigma rule evaluation for Windows/Linux/macOS events
    - Behavioral analysis (process trees, network anomalies)
    - Integration with VALHALLA feed manager for rule updates
    """

    def __init__(self, config_path: str = "config/thor_config.json"):
        self.config = self._load_config(config_path)
        self.project_id = self.config.get("gcp_project_id", os.getenv("GCP_PROJECT_ID"))
        self.tenant_id = self.config.get("tenant_id", "default")

        # GCP clients
        self.publisher = pubsub_v1.PublisherClient()
        self.firestore_client = firestore.Client(project=self.project_id)
        self.bq_client = bigquery.Client(project=self.project_id)

        # Pub/Sub topics
        self.scan_requests_topic = self.config["pubsub_topics"]["scan_requests"]
        self.findings_topic = self.config["pubsub_topics"]["findings"]

        # YARA rules
        self.yara_rules_path = Path(self.config["yara"]["rules_path"])
        self.compiled_rules: Optional[yara.Rules] = None

        # IOC database
        self.ioc_set: Optional[IOCSet] = None

        # Sigma rules (stored as dicts for now)
        self.sigma_rules: List[Dict] = []

        # Scan exclusions
        self.excluded_paths = set(self.config.get("scan_exclusions", {}).get("paths", []))
        self.excluded_extensions = set(self.config.get("scan_exclusions", {}).get("extensions", []))

        # Statistics
        self.scan_stats = defaultdict(int)

        logger.info(f"THOR Agent initialized for tenant {self.tenant_id}")

    def _load_config(self, config_path: str) -> Dict:
        """Load THOR configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self._default_config()

    def _default_config(self) -> Dict:
        """Default configuration if file doesn't exist"""
        return {
            "tenant_id": "default",
            "yara": {
                "rules_path": "/opt/thor/yara_rules",
                "enable_memory_scan": True,
                "enable_process_scan": True
            },
            "ioc": {
                "feeds": ["valhalla", "threatfox", "malwarebazaar"],
                "update_interval_hours": 6
            },
            "sigma": {
                "rules_path": "/opt/thor/sigma_rules",
                "backends": ["splunk", "elk", "windows_event_log"]
            },
            "scan_exclusions": {
                "paths": ["/proc", "/sys", "/dev"],
                "extensions": [".tmp", ".log"]
            },
            "pubsub_topics": {
                "scan_requests": "thor-scan-requests",
                "findings": "thor-findings"
            },
            "behavioral_analysis": {
                "enable_process_tree": True,
                "enable_network_baseline": True,
                "suspicious_process_patterns": [
                    "powershell.*-encodedcommand",
                    "cmd.exe.*whoami",
                    ".*mimikatz.*"
                ]
            }
        }

    def load_yara_rules(self) -> None:
        """Load and compile YARA rules from directory, skipping invalid rules"""
        try:
            if not self.yara_rules_path.exists():
                logger.warning(f"YARA rules path {self.yara_rules_path} does not exist")
                return

            # Collect all .yar and .yara files
            rule_files = {}
            for ext in ['.yar', '.yara']:
                for rule_file in self.yara_rules_path.rglob(f"*{ext}"):
                    namespace = rule_file.stem
                    # Handle duplicate names by adding parent directory
                    if namespace in rule_files:
                        namespace = f"{rule_file.parent.name}_{namespace}"
                    rule_files[namespace] = str(rule_file)

            if not rule_files:
                logger.warning("No YARA rules found")
                return

            # Try to compile all rules at once first (faster)
            try:
                self.compiled_rules = yara.compile(filepaths=rule_files)
                logger.info(f"Loaded {len(rule_files)} YARA rule files")
            except yara.SyntaxError as e:
                # If compilation fails, try loading rules individually
                logger.warning(f"Bulk compilation failed: {e}. Loading rules individually...")
                valid_rules = {}
                errors = 0
                
                for namespace, filepath in rule_files.items():
                    try:
                        # Test compile individual rule
                        test_rule = yara.compile(filepath=filepath)
                        valid_rules[namespace] = filepath
                    except yara.SyntaxError as rule_error:
                        errors += 1
                        logger.debug(f"Skipping invalid rule {filepath}: {rule_error}")
                    except Exception as rule_error:
                        errors += 1
                        logger.debug(f"Skipping rule {filepath}: {rule_error}")
                
                if valid_rules:
                    try:
                        self.compiled_rules = yara.compile(filepaths=valid_rules)
                        logger.info(f"Loaded {len(valid_rules)}/{len(rule_files)} YARA rule files "
                                  f"({errors} skipped due to errors)")
                    except Exception as compile_error:
                        logger.error(f"Failed to compile valid rules: {compile_error}")
                        self.compiled_rules = None
                else:
                    logger.warning(f"No valid YARA rules found (all {len(rule_files)} had errors)")
                    self.compiled_rules = None

        except Exception as e:
            logger.error(f"Failed to load YARA rules: {e}")
            self.compiled_rules = None

    def load_iocs(self, ioc_data: Optional[Dict] = None) -> None:
        """
        Load IOCs from VALHALLA feed or provided data

        Args:
            ioc_data: Optional pre-loaded IOC set, otherwise fetch from feeds
        """
        if ioc_data:
            self.ioc_set = IOCSet(**ioc_data)
        else:
            # In production, this would fetch from VALHALLA agent
            # For now, load from local cache or Firestore
            try:
                doc_ref = self.firestore_client.collection('threat_intel').document('iocs')
                doc = doc_ref.get()

                if doc.exists:
                    data = doc.to_dict()
                    self.ioc_set = IOCSet(
                        ips=set(data.get('ips', [])),
                        domains=set(data.get('domains', [])),
                        file_hashes=data.get('file_hashes', {}),
                        filenames=set(data.get('filenames', [])),
                        registry_keys=set(data.get('registry_keys', [])),
                        mutex_names=set(data.get('mutex_names', [])),
                        user_agents=set(data.get('user_agents', []))
                    )
                    logger.info(f"Loaded {len(self.ioc_set.ips)} IP IOCs, "
                              f"{len(self.ioc_set.file_hashes)} hash IOCs")
                else:
                    logger.warning("No IOCs found in Firestore, initializing empty set")
                    self.ioc_set = IOCSet(set(), set(), {}, set(), set(), set(), set())

            except Exception as e:
                logger.error(f"Failed to load IOCs: {e}")
                self.ioc_set = IOCSet(set(), set(), {}, set(), set(), set(), set())

    def scan_file_with_yara(self, file_path: str) -> List[ThreatMatch]:
        """Scan a single file with YARA rules"""
        matches = []

        if not self.compiled_rules:
            return matches

        try:
            yara_matches = self.compiled_rules.match(file_path)

            for match in yara_matches:
                # Determine severity from rule metadata or name
                severity = self._determine_severity(match)

                threat_match = ThreatMatch(
                    match_id=f"yara_{match.rule}_{hashlib.md5(file_path.encode()).hexdigest()[:8]}",
                    timestamp=datetime.utcnow().isoformat(),
                    match_type=MatchType.YARA,
                    severity=severity,
                    rule_name=match.rule,
                    confidence=0.85,  # YARA matches are generally high confidence
                    target=file_path,
                    context={
                        "namespace": match.namespace,
                        "tags": match.tags,
                        "meta": match.meta,
                        "strings": [(s[0], s[1], s[2].decode('utf-8', errors='ignore'))
                                   for s in match.strings[:5]]  # Limit string output
                    },
                    remediation=[
                        "Isolate the affected system",
                        f"Quarantine file: {file_path}",
                        "Perform memory dump for forensic analysis",
                        "Check for lateral movement indicators"
                    ],
                    false_positive_likelihood=0.1
                )
                matches.append(threat_match)
                self.scan_stats['yara_matches'] += 1

        except yara.Error as e:
            logger.debug(f"YARA scan error for {file_path}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error scanning {file_path}: {e}")

        return matches

    def scan_file_for_iocs(self, file_path: str) -> List[ThreatMatch]:
        """Scan file for IOC matches (filename, hash)"""
        matches = []

        if not self.ioc_set:
            return matches

        try:
            # Check filename IOCs
            filename = os.path.basename(file_path)
            if filename in self.ioc_set.filenames:
                matches.append(ThreatMatch(
                    match_id=f"ioc_filename_{hashlib.md5(file_path.encode()).hexdigest()[:8]}",
                    timestamp=datetime.utcnow().isoformat(),
                    match_type=MatchType.IOC_FILENAME,
                    severity=ThreatSeverity.HIGH,
                    rule_name=f"IOC_Filename_{filename}",
                    confidence=0.75,
                    target=file_path,
                    context={"filename": filename, "ioc_type": "filename"},
                    remediation=["Quarantine file", "Investigate file origin"],
                    false_positive_likelihood=0.2
                ))
                self.scan_stats['ioc_filename_matches'] += 1

            # Check file hash IOCs
            if os.path.getsize(file_path) < 100 * 1024 * 1024:  # Only hash files < 100MB
                file_hash = self._calculate_file_hash(file_path)
                if file_hash in self.ioc_set.file_hashes:
                    malware_name = self.ioc_set.file_hashes[file_hash]
                    matches.append(ThreatMatch(
                        match_id=f"ioc_hash_{file_hash[:16]}",
                        timestamp=datetime.utcnow().isoformat(),
                        match_type=MatchType.IOC_HASH,
                        severity=ThreatSeverity.CRITICAL,
                        rule_name=f"IOC_Hash_{malware_name}",
                        confidence=0.95,
                        target=file_path,
                        context={
                            "sha256": file_hash,
                            "malware_family": malware_name,
                            "ioc_type": "file_hash"
                        },
                        remediation=[
                            "IMMEDIATE: Isolate system from network",
                            f"Quarantine known malware: {malware_name}",
                            "Initiate incident response procedure",
                            "Perform full system scan"
                        ],
                        false_positive_likelihood=0.01
                    ))
                    self.scan_stats['ioc_hash_matches'] += 1

        except Exception as e:
            logger.error(f"IOC scan error for {file_path}: {e}")

        return matches

    def scan_filesystem(self, target_paths: List[str], max_depth: int = 10) -> List[ThreatMatch]:
        """
        Scan filesystem for threats

        Args:
            target_paths: List of directories or files to scan
            max_depth: Maximum recursion depth

        Returns:
            List of threat matches
        """
        all_matches = []

        for target in target_paths:
            target_path = Path(target)

            if not target_path.exists():
                logger.warning(f"Target path does not exist: {target}")
                continue

            if target_path.is_file():
                # Scan single file
                matches = self._scan_single_file(str(target_path))
                all_matches.extend(matches)
                self.scan_stats['files_scanned'] += 1

            elif target_path.is_dir():
                # Scan directory recursively
                for root, dirs, files in os.walk(target_path):
                    # Check depth
                    depth = str(root).count(os.sep) - str(target_path).count(os.sep)
                    if depth > max_depth:
                        continue

                    # Filter excluded directories
                    dirs[:] = [d for d in dirs if not self._is_excluded_path(os.path.join(root, d))]

                    for file in files:
                        file_path = os.path.join(root, file)

                        if self._is_excluded_path(file_path):
                            continue

                        matches = self._scan_single_file(file_path)
                        all_matches.extend(matches)
                        self.scan_stats['files_scanned'] += 1

                        # Log progress every 1000 files
                        if self.scan_stats['files_scanned'] % 1000 == 0:
                            logger.info(f"Scanned {self.scan_stats['files_scanned']} files, "
                                      f"found {len(all_matches)} threats")

        return all_matches

    def scan_processes(self) -> List[ThreatMatch]:
        """Scan running processes for threats"""
        matches = []

        for proc in psutil.process_iter(['pid', 'name', 'exe', 'cmdline', 'connections']):
            try:
                proc_info = proc.info

                # Behavioral analysis - suspicious command lines
                cmdline = ' '.join(proc_info.get('cmdline', []))
                for pattern in self.config["behavioral_analysis"]["suspicious_process_patterns"]:
                    if re.search(pattern, cmdline, re.IGNORECASE):
                        matches.append(ThreatMatch(
                            match_id=f"behavioral_proc_{proc_info['pid']}",
                            timestamp=datetime.utcnow().isoformat(),
                            match_type=MatchType.BEHAVIORAL,
                            severity=ThreatSeverity.HIGH,
                            rule_name=f"Suspicious_Process_{pattern[:30]}",
                            confidence=0.70,
                            target=proc_info.get('exe', proc_info['name']),
                            context={
                                "pid": proc_info['pid'],
                                "name": proc_info['name'],
                                "cmdline": cmdline,
                                "pattern": pattern
                            },
                            remediation=[
                                f"Investigate process PID {proc_info['pid']}",
                                "Collect process memory dump",
                                "Check parent process and process tree"
                            ],
                            false_positive_likelihood=0.3
                        ))
                        self.scan_stats['behavioral_matches'] += 1

                # YARA scan process memory (if enabled and rules loaded)
                if self.config["yara"].get("enable_process_scan") and self.compiled_rules:
                    try:
                        # Note: Requires elevated privileges
                        yara_matches = self.compiled_rules.match(pid=proc_info['pid'])
                        for match in yara_matches:
                            matches.append(ThreatMatch(
                                match_id=f"yara_proc_{match.rule}_{proc_info['pid']}",
                                timestamp=datetime.utcnow().isoformat(),
                                match_type=MatchType.YARA,
                                severity=self._determine_severity(match),
                                rule_name=match.rule,
                                confidence=0.90,
                                target=f"Process: {proc_info['name']} (PID {proc_info['pid']})",
                                context={
                                    "pid": proc_info['pid'],
                                    "process_name": proc_info['name'],
                                    "yara_rule": match.rule,
                                    "tags": match.tags
                                },
                                remediation=[
                                    f"URGENT: Suspend process PID {proc_info['pid']}",
                                    "Capture memory forensics",
                                    "Analyze process injection/hollowing"
                                ],
                                false_positive_likelihood=0.05
                            ))
                            self.scan_stats['yara_memory_matches'] += 1
                    except:
                        pass  # Process might have terminated or access denied

                self.scan_stats['processes_scanned'] += 1

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return matches

    def scan_network_connections(self) -> List[ThreatMatch]:
        """Scan active network connections for IOC matches"""
        matches = []

        if not self.ioc_set:
            return matches

        for conn in psutil.net_connections(kind='inet'):
            try:
                if conn.raddr:
                    remote_ip = conn.raddr.ip

                    # Check against IP IOCs
                    if remote_ip in self.ioc_set.ips:
                        # Try to get process info
                        proc_name = "unknown"
                        if conn.pid:
                            try:
                                proc = psutil.Process(conn.pid)
                                proc_name = proc.name()
                            except:
                                pass

                        matches.append(ThreatMatch(
                            match_id=f"ioc_ip_{remote_ip}_{conn.laddr.port}",
                            timestamp=datetime.utcnow().isoformat(),
                            match_type=MatchType.IOC_IP,
                            severity=ThreatSeverity.CRITICAL,
                            rule_name=f"IOC_MaliciousIP_{remote_ip}",
                            confidence=0.85,
                            target=f"{remote_ip}:{conn.raddr.port}",
                            context={
                                "remote_ip": remote_ip,
                                "remote_port": conn.raddr.port,
                                "local_port": conn.laddr.port,
                                "status": conn.status,
                                "process": proc_name,
                                "pid": conn.pid
                            },
                            remediation=[
                                f"IMMEDIATE: Block IP {remote_ip} at firewall",
                                f"Terminate process {proc_name} (PID {conn.pid})",
                                "Investigate data exfiltration",
                                "Check for C2 communication patterns"
                            ],
                            false_positive_likelihood=0.1
                        ))
                        self.scan_stats['ioc_ip_matches'] += 1

                self.scan_stats['network_connections_scanned'] += 1

            except Exception as e:
                logger.debug(f"Error scanning connection: {e}")

        return matches

    def _scan_single_file(self, file_path: str) -> List[ThreatMatch]:
        """Scan a single file with all available methods"""
        matches = []

        # YARA scan
        if self.compiled_rules:
            matches.extend(self.scan_file_with_yara(file_path))

        # IOC scan
        if self.ioc_set:
            matches.extend(self.scan_file_for_iocs(file_path))

        return matches

    def _is_excluded_path(self, path: str) -> bool:
        """Check if path should be excluded from scanning"""
        # Check excluded directories
        for excluded in self.excluded_paths:
            if path.startswith(excluded):
                return True

        # Check excluded extensions
        _, ext = os.path.splitext(path)
        if ext in self.excluded_extensions:
            return True

        return False

    def _calculate_file_hash(self, file_path: str, algorithm: str = "sha256") -> str:
        """Calculate file hash"""
        hash_obj = hashlib.new(algorithm)

        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)

        return hash_obj.hexdigest()

    def _determine_severity(self, yara_match) -> ThreatSeverity:
        """Determine threat severity from YARA match metadata"""
        # Check rule metadata for severity
        if hasattr(yara_match, 'meta'):
            severity_meta = yara_match.meta.get('severity', '').lower()
            if 'critical' in severity_meta:
                return ThreatSeverity.CRITICAL
            elif 'high' in severity_meta:
                return ThreatSeverity.HIGH
            elif 'medium' in severity_meta:
                return ThreatSeverity.MEDIUM
            elif 'low' in severity_meta:
                return ThreatSeverity.LOW

        # Check rule tags
        tags = [tag.lower() for tag in yara_match.tags]
        if any(t in tags for t in ['apt', 'ransomware', 'trojan', 'rootkit']):
            return ThreatSeverity.CRITICAL
        elif any(t in tags for t in ['exploit', 'backdoor', 'c2']):
            return ThreatSeverity.HIGH
        elif any(t in tags for t in ['suspicious', 'pua']):
            return ThreatSeverity.MEDIUM

        # Default
        return ThreatSeverity.MEDIUM

    def perform_scan(self, scan_types: List[ScanType], target_paths: Optional[List[str]] = None) -> ScanResult:
        """
        Perform comprehensive endpoint scan

        Args:
            scan_types: Types of scans to perform
            target_paths: Filesystem paths to scan (for FILESYSTEM scan type)

        Returns:
            ScanResult with all findings
        """
        import socket

        scan_id = f"thor_scan_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        hostname = socket.gethostname()
        start_time = datetime.utcnow().isoformat()

        # Reset statistics
        self.scan_stats = defaultdict(int)

        all_matches = []

        logger.info(f"Starting THOR scan {scan_id} on {hostname}")

        # Load rules and IOCs if not already loaded
        if not self.compiled_rules:
            self.load_yara_rules()
        if not self.ioc_set:
            self.load_iocs()

        # Perform requested scans
        if ScanType.FULL in scan_types:
            scan_types = [ScanType.FILESYSTEM, ScanType.PROCESS, ScanType.NETWORK]

        if ScanType.FILESYSTEM in scan_types:
            logger.info("Performing filesystem scan...")
            if not target_paths:
                # Default to common malware locations
                target_paths = ["/tmp", "/var/tmp", os.path.expanduser("~")]
            all_matches.extend(self.scan_filesystem(target_paths))

        if ScanType.PROCESS in scan_types:
            logger.info("Performing process scan...")
            all_matches.extend(self.scan_processes())

        if ScanType.NETWORK in scan_types:
            logger.info("Performing network connection scan...")
            all_matches.extend(self.scan_network_connections())

        end_time = datetime.utcnow().isoformat()

        # Compile results
        scan_result = ScanResult(
            scan_id=scan_id,
            hostname=hostname,
            start_time=start_time,
            end_time=end_time,
            scan_types=scan_types,
            total_files_scanned=self.scan_stats['files_scanned'],
            total_processes_scanned=self.scan_stats['processes_scanned'],
            total_network_connections=self.scan_stats['network_connections_scanned'],
            matches=all_matches,
            iocs_detected=[m.rule_name for m in all_matches if m.match_type in [MatchType.IOC_IP, MatchType.IOC_HASH, MatchType.IOC_FILENAME]],
            yara_rules_matched=[m.rule_name for m in all_matches if m.match_type == MatchType.YARA],
            sigma_rules_matched=[],  # Not yet implemented
            scan_statistics=dict(self.scan_stats)
        )

        logger.info(f"THOR scan {scan_id} complete: {len(all_matches)} threats detected")

        # Publish results to Pub/Sub for TAA enrichment
        self._publish_findings(scan_result)

        # Store in BigQuery
        self._store_results_bigquery(scan_result)

        return scan_result

    def _publish_findings(self, scan_result: ScanResult) -> None:
        """Publish scan findings to Pub/Sub for downstream processing"""
        try:
            topic_path = self.publisher.topic_path(self.project_id, self.findings_topic)

            message_data = {
                "scan_id": scan_result.scan_id,
                "hostname": scan_result.hostname,
                "timestamp": scan_result.end_time,
                "total_threats": len(scan_result.matches),
                "critical_threats": len([m for m in scan_result.matches if m.severity == ThreatSeverity.CRITICAL]),
                "high_threats": len([m for m in scan_result.matches if m.severity == ThreatSeverity.HIGH]),
                "matches": [m.to_dict() for m in scan_result.matches],
                "source": "thor_endpoint_agent",
                "tenant_id": self.tenant_id
            }

            future = self.publisher.publish(
                topic_path,
                json.dumps(message_data).encode('utf-8'),
                source="thor_agent",
                scan_id=scan_result.scan_id
            )

            logger.info(f"Published findings to {self.findings_topic}: {future.result()}")

        except Exception as e:
            logger.error(f"Failed to publish findings: {e}")

    def _store_results_bigquery(self, scan_result: ScanResult) -> None:
        """Store scan results in BigQuery for historical analysis"""
        try:
            table_id = f"{self.project_id}.soc_data.thor_scan_results"

            rows_to_insert = [{
                "scan_id": scan_result.scan_id,
                "hostname": scan_result.hostname,
                "start_time": scan_result.start_time,
                "end_time": scan_result.end_time,
                "scan_types": [st.value for st in scan_result.scan_types],
                "total_files_scanned": scan_result.total_files_scanned,
                "total_processes_scanned": scan_result.total_processes_scanned,
                "total_network_connections": scan_result.total_network_connections,
                "total_threats": len(scan_result.matches),
                "critical_count": len([m for m in scan_result.matches if m.severity == ThreatSeverity.CRITICAL]),
                "high_count": len([m for m in scan_result.matches if m.severity == ThreatSeverity.HIGH]),
                "medium_count": len([m for m in scan_result.matches if m.severity == ThreatSeverity.MEDIUM]),
                "matches": json.dumps([m.to_dict() for m in scan_result.matches]),
                "statistics": json.dumps(scan_result.scan_statistics),
                "tenant_id": self.tenant_id
            }]

            errors = self.bq_client.insert_rows_json(table_id, rows_to_insert)

            if errors:
                logger.error(f"BigQuery insert errors: {errors}")
            else:
                logger.info(f"Stored scan results in BigQuery: {table_id}")

        except Exception as e:
            logger.error(f"Failed to store results in BigQuery: {e}")


def main():
    """Main entry point for THOR agent"""
    import argparse

    parser = argparse.ArgumentParser(description="THOR Endpoint Scanning Agent")
    parser.add_argument("--config", default="config/thor_config.json", help="Config file path")
    parser.add_argument("--scan-type", choices=["filesystem", "process", "network", "full"],
                       default="full", help="Type of scan to perform")
    parser.add_argument("--target", nargs="+", help="Target paths for filesystem scan")
    parser.add_argument("--load-yara", action="store_true", help="Preload YARA rules")
    parser.add_argument("--load-iocs", action="store_true", help="Preload IOCs")

    args = parser.parse_args()

    # Initialize agent
    agent = THOREndpointAgent(config_path=args.config)

    # Preload rules if requested
    if args.load_yara:
        agent.load_yara_rules()
    if args.load_iocs:
        agent.load_iocs()

    # Perform scan
    scan_type_map = {
        "filesystem": ScanType.FILESYSTEM,
        "process": ScanType.PROCESS,
        "network": ScanType.NETWORK,
        "full": ScanType.FULL
    }

    result = agent.perform_scan(
        scan_types=[scan_type_map[args.scan_type]],
        target_paths=args.target
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"THOR Scan Complete: {result.scan_id}")
    print(f"{'='*60}")
    print(f"Hostname: {result.hostname}")
    print(f"Duration: {result.start_time} - {result.end_time}")
    print(f"Files scanned: {result.total_files_scanned}")
    print(f"Processes scanned: {result.total_processes_scanned}")
    print(f"Network connections: {result.total_network_connections}")
    print(f"\nThreats detected: {len(result.matches)}")

    if result.matches:
        severity_counts = {}
        for match in result.matches:
            severity_counts[match.severity.value] = severity_counts.get(match.severity.value, 0) + 1

        print("\nBy severity:")
        for severity, count in sorted(severity_counts.items()):
            print(f"  {severity.upper()}: {count}")

        print(f"\nTop threats:")
        for match in sorted(result.matches, key=lambda m: m.confidence, reverse=True)[:5]:
            print(f"  [{match.severity.value.upper()}] {match.rule_name}")
            print(f"    Target: {match.target}")
            print(f"    Confidence: {match.confidence:.2%}")


if __name__ == "__main__":
    main()
