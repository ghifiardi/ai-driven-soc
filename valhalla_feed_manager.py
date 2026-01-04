#!/usr/bin/env python3
"""
VALHALLA Threat Intelligence Feed Manager
==========================================
Mimics Nextron Systems VALHALLA threat intelligence platform capabilities:
- YARA rule repository and distribution
- IOC feed aggregation from multiple sources
- Sigma rule management
- Automated rule updates and version control
- Rule quality scoring and validation
- Custom rule creation and testing
- Integration with THOR and ASGARD for rule deployment

Integration with AI-Driven SOC:
- Provides curated threat intelligence to THOR agents
- Integrates with existing threat intel MCP server
- Feeds detection patterns to ADA for ML training
- Supplies context to TAA for alert enrichment
"""

import asyncio
import hashlib
import json
import logging
import requests
import yara
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict

# Google Cloud imports
from google.cloud import firestore, storage, pubsub_v1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RuleType(Enum):
    """Types of detection rules"""
    YARA = "yara"
    SIGMA = "sigma"
    SNORT = "snort"
    SURICATA = "suricata"
    CUSTOM = "custom"


class RuleQuality(Enum):
    """Rule quality/confidence levels"""
    VERIFIED = "verified"  # Tested and validated
    HIGH = "high"  # High confidence, well-tested
    MEDIUM = "medium"  # Medium confidence
    EXPERIMENTAL = "experimental"  # New or untested
    DEPRECATED = "deprecated"  # Old, should be replaced


class ThreatCategory(Enum):
    """Threat categories for rules"""
    APT = "apt"
    RANSOMWARE = "ransomware"
    TROJAN = "trojan"
    MALWARE = "malware"
    EXPLOIT = "exploit"
    WEBSHELL = "webshell"
    BACKDOOR = "backdoor"
    ROOTKIT = "rootkit"
    CRYPTOMINER = "cryptominer"
    PUA = "pua"  # Potentially Unwanted Application
    HACKTOOL = "hacktool"
    GENERIC = "generic"


class FeedSource(Enum):
    """External threat intelligence sources"""
    VALHALLA_PUBLIC = "valhalla_public"
    THREATFOX = "threatfox"
    MALWAREBAZAAR = "malwarebazaar"
    ABUSE_CH = "abuse_ch"
    EMERGINGTHREATS = "emergingthreats"
    ALIENVAULT_OTX = "alienvault_otx"
    MISP = "misp"
    CUSTOM = "custom"


@dataclass
class YARARule:
    """YARA rule metadata and content"""
    rule_id: str
    rule_name: str
    rule_content: str
    description: str
    author: str
    reference: str
    created_at: str
    updated_at: str
    version: str
    quality: RuleQuality
    categories: List[ThreatCategory]
    tags: List[str]
    severity: str  # critical, high, medium, low
    false_positive_rate: float  # 0.0 - 1.0
    detection_rate: float  # 0.0 - 1.0
    source: FeedSource
    is_compiled: bool
    compiled_path: Optional[str]
    sha256_hash: str

    def to_dict(self) -> Dict:
        data = asdict(self)
        data['quality'] = self.quality.value
        data['categories'] = [c.value for c in self.categories]
        data['source'] = self.source.value
        return data


@dataclass
class SigmaRule:
    """Sigma rule metadata and content"""
    rule_id: str
    title: str
    rule_content: str  # YAML content
    description: str
    author: str
    created_at: str
    updated_at: str
    logsource: Dict[str, str]
    detection: Dict[str, Any]
    level: str  # critical, high, medium, low, informational
    tags: List[str]
    false_positives: List[str]
    references: List[str]
    sha256_hash: str

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class IOCFeed:
    """IOC feed metadata"""
    feed_id: str
    feed_name: str
    source: FeedSource
    feed_type: str  # ip, domain, hash, url, etc.
    last_updated: str
    next_update: str
    update_interval_hours: int
    total_iocs: int
    active_iocs: int
    expired_iocs: int
    quality_score: float  # 0.0 - 1.0

    def to_dict(self) -> Dict:
        data = asdict(self)
        data['source'] = self.source.value
        return data


@dataclass
class IOCEntry:
    """Individual IOC entry"""
    ioc_id: str
    ioc_value: str
    ioc_type: str  # ip, domain, hash, url, filename, etc.
    threat_type: str
    malware_family: Optional[str]
    first_seen: str
    last_seen: str
    expires_at: Optional[str]
    confidence: float  # 0.0 - 1.0
    tags: List[str]
    source: FeedSource
    reference_url: Optional[str]
    context: Dict[str, Any]

    def to_dict(self) -> Dict:
        data = asdict(self)
        data['source'] = self.source.value
        return data


class VALHALLAFeedManager:
    """
    VALHALLA-like threat intelligence feed manager

    Capabilities:
    - Aggregate and normalize IOCs from multiple sources
    - Manage YARA rule repository with versioning
    - Distribute Sigma rules for log analysis
    - Automated feed updates and synchronization
    - Rule quality scoring and validation
    - Custom rule creation and testing
    - GCP Cloud Storage for rule distribution
    - Pub/Sub notifications for rule updates
    """

    def __init__(self, config_path: str = "config/valhalla_config.json"):
        self.config = self._load_config(config_path)
        self.project_id = self.config.get("gcp_project_id", os.getenv("GCP_PROJECT_ID"))
        self.tenant_id = self.config.get("tenant_id", "default")

        # GCP clients
        self.firestore_client = firestore.Client(project=self.project_id)
        self.storage_client = storage.Client(project=self.project_id)
        self.publisher = pubsub_v1.PublisherClient()

        # GCS bucket for rule storage
        self.rules_bucket_name = self.config["gcs"]["rules_bucket"]
        try:
            self.rules_bucket = self.storage_client.bucket(self.rules_bucket_name)
        except Exception as e:
            logger.warning(f"Rules bucket not accessible: {e}")
            self.rules_bucket = None

        # Pub/Sub topics
        self.rule_updates_topic = self.config["pubsub_topics"]["rule_updates"]
        self.ioc_updates_topic = self.config["pubsub_topics"]["ioc_updates"]

        # In-memory caches
        self.yara_rules: Dict[str, YARARule] = {}
        self.sigma_rules: Dict[str, SigmaRule] = {}
        self.ioc_feeds: Dict[str, IOCFeed] = {}
        self.iocs: Dict[str, List[IOCEntry]] = defaultdict(list)  # type -> [IOCs]

        # API keys for external sources
        self.api_keys = self.config.get("api_keys", {})

        # Load cached rules and IOCs
        self._load_cached_data()

        logger.info(f"VALHALLA Feed Manager initialized for tenant {self.tenant_id}")
        logger.info(f"YARA rules: {len(self.yara_rules)}, Sigma rules: {len(self.sigma_rules)}, "
                   f"IOC feeds: {len(self.ioc_feeds)}")

    def _load_config(self, config_path: str) -> Dict:
        """Load VALHALLA configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self._default_config()

    def _default_config(self) -> Dict:
        """Default configuration"""
        import os
        return {
            "tenant_id": "default",
            "gcp_project_id": os.getenv("GCP_PROJECT_ID", "my-soc-project"),
            "gcs": {
                "rules_bucket": "valhalla-threat-intel"
            },
            "pubsub_topics": {
                "rule_updates": "valhalla-rule-updates",
                "ioc_updates": "valhalla-ioc-updates"
            },
            "feeds": {
                "threatfox": {
                    "enabled": True,
                    "url": "https://threatfox-api.abuse.ch/api/v1/",
                    "update_interval_hours": 1,
                    "types": ["ip", "domain", "hash", "url"]
                },
                "malwarebazaar": {
                    "enabled": True,
                    "url": "https://mb-api.abuse.ch/api/v1/",
                    "update_interval_hours": 6,
                    "types": ["hash"]
                },
                "emergingthreats": {
                    "enabled": True,
                    "url": "https://rules.emergingthreats.net/",
                    "update_interval_hours": 24,
                    "types": ["yara", "snort"]
                }
            },
            "yara": {
                "auto_compile": True,
                "validation": True,
                "min_quality": "medium"
            },
            "sigma": {
                "validation": True,
                "backends": ["splunk", "elk", "qradar"]
            },
            "api_keys": {
                "virustotal": os.getenv("VIRUSTOTAL_API_KEY", ""),
                "alienvault": os.getenv("ALIENVAULT_API_KEY", "")
            }
        }

    def _load_cached_data(self) -> None:
        """Load cached rules and IOCs from Firestore"""
        try:
            # Load YARA rules
            yara_docs = self.firestore_client.collection('yara_rules').where(
                'tenant_id', '==', self.tenant_id
            ).stream()

            for doc in yara_docs:
                data = doc.to_dict()
                rule = YARARule(
                    rule_id=data['rule_id'],
                    rule_name=data['rule_name'],
                    rule_content=data['rule_content'],
                    description=data['description'],
                    author=data['author'],
                    reference=data['reference'],
                    created_at=data['created_at'],
                    updated_at=data['updated_at'],
                    version=data['version'],
                    quality=RuleQuality(data['quality']),
                    categories=[ThreatCategory(c) for c in data['categories']],
                    tags=data['tags'],
                    severity=data['severity'],
                    false_positive_rate=data['false_positive_rate'],
                    detection_rate=data['detection_rate'],
                    source=FeedSource(data['source']),
                    is_compiled=data['is_compiled'],
                    compiled_path=data.get('compiled_path'),
                    sha256_hash=data['sha256_hash']
                )
                self.yara_rules[rule.rule_id] = rule

            # Load IOC feeds metadata
            feed_docs = self.firestore_client.collection('ioc_feeds').where(
                'tenant_id', '==', self.tenant_id
            ).stream()

            for doc in feed_docs:
                data = doc.to_dict()
                feed = IOCFeed(**data)
                feed.source = FeedSource(data['source'])
                self.ioc_feeds[feed.feed_id] = feed

            logger.info(f"Loaded {len(self.yara_rules)} YARA rules and {len(self.ioc_feeds)} IOC feeds from cache")

        except Exception as e:
            logger.error(f"Failed to load cached data: {e}")

    # ===== YARA Rule Management =====

    def fetch_yara_rules_emergingthreats(self) -> List[YARARule]:
        """Fetch YARA rules from Emerging Threats"""
        rules = []

        try:
            # Example: Fetch from public ET YARA repo (GitHub)
            url = "https://raw.githubusercontent.com/emergingthreats/threatview/master/yara/ransomware.yar"

            response = requests.get(url, timeout=30)
            response.raise_for_status()

            rule_content = response.text

            # Parse YARA rules (simplified - in production use proper parser)
            rule_id = f"et_yara_{hashlib.md5(rule_content.encode()).hexdigest()[:16]}"

            rule = YARARule(
                rule_id=rule_id,
                rule_name="EmergingThreats_Ransomware",
                rule_content=rule_content,
                description="Ransomware detection rules from Emerging Threats",
                author="Emerging Threats",
                reference="https://github.com/emergingthreats/threatview",
                created_at=datetime.utcnow().isoformat(),
                updated_at=datetime.utcnow().isoformat(),
                version="1.0",
                quality=RuleQuality.HIGH,
                categories=[ThreatCategory.RANSOMWARE],
                tags=["ransomware", "emergingthreats"],
                severity="high",
                false_positive_rate=0.05,
                detection_rate=0.85,
                source=FeedSource.EMERGINGTHREATS,
                is_compiled=False,
                compiled_path=None,
                sha256_hash=hashlib.sha256(rule_content.encode()).hexdigest()
            )

            rules.append(rule)
            logger.info(f"Fetched YARA rules from Emerging Threats")

        except Exception as e:
            logger.error(f"Failed to fetch Emerging Threats YARA rules: {e}")

        return rules

    def add_yara_rule(self, rule: YARARule, validate: bool = True) -> bool:
        """
        Add a YARA rule to the repository

        Args:
            rule: YARA rule to add
            validate: Whether to validate rule syntax

        Returns:
            True if successful
        """
        try:
            # Validate YARA rule if requested
            if validate:
                if not self._validate_yara_rule(rule.rule_content):
                    logger.error(f"YARA rule validation failed: {rule.rule_name}")
                    return False

            # Compile rule if configured
            if self.config["yara"]["auto_compile"]:
                compiled_path = self._compile_yara_rule(rule)
                rule.is_compiled = True
                rule.compiled_path = compiled_path

            # Store in Firestore
            doc_ref = self.firestore_client.collection('yara_rules').document(rule.rule_id)
            data = rule.to_dict()
            data['tenant_id'] = self.tenant_id
            doc_ref.set(data)

            # Store rule content in GCS
            if self.rules_bucket:
                blob = self.rules_bucket.blob(f"yara/{rule.rule_id}.yar")
                blob.upload_from_string(rule.rule_content)

            # Update in-memory cache
            self.yara_rules[rule.rule_id] = rule

            # Publish update notification
            self._publish_rule_update("yara", rule.rule_id, "added")

            logger.info(f"Added YARA rule: {rule.rule_name} ({rule.rule_id})")
            return True

        except Exception as e:
            logger.error(f"Failed to add YARA rule: {e}")
            return False

    def _validate_yara_rule(self, rule_content: str) -> bool:
        """Validate YARA rule syntax"""
        try:
            # Try to compile the rule
            yara.compile(source=rule_content)
            return True
        except yara.SyntaxError as e:
            logger.error(f"YARA syntax error: {e}")
            return False
        except Exception as e:
            logger.error(f"YARA validation error: {e}")
            return False

    def _compile_yara_rule(self, rule: YARARule) -> str:
        """Compile YARA rule and store in GCS"""
        try:
            import tempfile

            compiled_rules = yara.compile(source=rule.rule_content)

            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.yarc') as tmp:
                compiled_rules.save(tmp.name)

                # Upload to GCS
                if self.rules_bucket:
                    blob = self.rules_bucket.blob(f"yara/compiled/{rule.rule_id}.yarc")
                    blob.upload_from_filename(tmp.name)

                    compiled_path = f"gs://{self.rules_bucket_name}/yara/compiled/{rule.rule_id}.yarc"
                    logger.info(f"Compiled YARA rule: {compiled_path}")
                    return compiled_path

        except Exception as e:
            logger.error(f"Failed to compile YARA rule: {e}")

        return None

    def get_yara_rules(
        self,
        categories: Optional[List[ThreatCategory]] = None,
        min_quality: Optional[RuleQuality] = None,
        tags: Optional[List[str]] = None
    ) -> List[YARARule]:
        """
        Get YARA rules matching criteria

        Args:
            categories: Filter by threat categories
            min_quality: Minimum rule quality
            tags: Filter by tags

        Returns:
            List of matching rules
        """
        rules = list(self.yara_rules.values())

        # Filter by categories
        if categories:
            rules = [r for r in rules if any(c in r.categories for c in categories)]

        # Filter by quality
        if min_quality:
            quality_order = [RuleQuality.DEPRECATED, RuleQuality.EXPERIMENTAL,
                           RuleQuality.MEDIUM, RuleQuality.HIGH, RuleQuality.VERIFIED]
            min_index = quality_order.index(min_quality)
            rules = [r for r in rules if quality_order.index(r.quality) >= min_index]

        # Filter by tags
        if tags:
            rules = [r for r in rules if any(t in r.tags for t in tags)]

        return rules

    def export_yara_rules_bundle(self, output_path: str, rule_ids: Optional[List[str]] = None) -> bool:
        """
        Export YARA rules as a bundle file

        Args:
            output_path: Output file path
            rule_ids: Specific rule IDs to export (None = all)

        Returns:
            True if successful
        """
        try:
            if rule_ids:
                rules = [self.yara_rules[rid] for rid in rule_ids if rid in self.yara_rules]
            else:
                rules = list(self.yara_rules.values())

            # Combine all rule contents
            bundle_content = "\n\n".join([
                f"// Rule: {rule.rule_name}\n"
                f"// Source: {rule.source.value}\n"
                f"// Quality: {rule.quality.value}\n"
                f"// Severity: {rule.severity}\n"
                f"{rule.rule_content}"
                for rule in rules
            ])

            # Write to file
            with open(output_path, 'w') as f:
                f.write(bundle_content)

            logger.info(f"Exported {len(rules)} YARA rules to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export YARA rules: {e}")
            return False

    # ===== IOC Management =====

    def fetch_iocs_threatfox(self, days: int = 7) -> List[IOCEntry]:
        """Fetch recent IOCs from ThreatFox (abuse.ch)"""
        iocs = []

        try:
            url = "https://threatfox-api.abuse.ch/api/v1/"

            # Query recent IOCs
            payload = {
                "query": "get_iocs",
                "days": days
            }

            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()

            data = response.json()

            if data.get("query_status") == "ok":
                for item in data.get("data", []):
                    ioc = IOCEntry(
                        ioc_id=f"threatfox_{item['id']}",
                        ioc_value=item.get("ioc"),
                        ioc_type=item.get("ioc_type", "unknown"),
                        threat_type=item.get("threat_type", "unknown"),
                        malware_family=item.get("malware"),
                        first_seen=item.get("first_seen", datetime.utcnow().isoformat()),
                        last_seen=item.get("last_seen", datetime.utcnow().isoformat()),
                        expires_at=None,
                        confidence=item.get("confidence_level", 50) / 100.0,
                        tags=item.get("tags", []),
                        source=FeedSource.THREATFOX,
                        reference_url=item.get("reference"),
                        context={
                            "reporter": item.get("reporter"),
                            "malware_printable": item.get("malware_printable")
                        }
                    )
                    iocs.append(ioc)

                logger.info(f"Fetched {len(iocs)} IOCs from ThreatFox")

        except Exception as e:
            logger.error(f"Failed to fetch ThreatFox IOCs: {e}")

        return iocs

    def fetch_iocs_malwarebazaar(self, limit: int = 100) -> List[IOCEntry]:
        """Fetch recent malware hashes from MalwareBazaar"""
        iocs = []

        try:
            url = "https://mb-api.abuse.ch/api/v1/"

            payload = {
                "query": "get_recent",
                "selector": limit
            }

            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()

            data = response.json()

            if data.get("query_status") == "ok":
                for item in data.get("data", []):
                    ioc = IOCEntry(
                        ioc_id=f"malwarebazaar_{item.get('sha256_hash', '')[:16]}",
                        ioc_value=item.get("sha256_hash"),
                        ioc_type="hash",
                        threat_type="malware",
                        malware_family=item.get("signature"),
                        first_seen=item.get("first_seen", datetime.utcnow().isoformat()),
                        last_seen=item.get("last_seen", datetime.utcnow().isoformat()),
                        expires_at=None,
                        confidence=0.90,  # MalwareBazaar is high quality
                        tags=item.get("tags", []),
                        source=FeedSource.MALWAREBAZAAR,
                        reference_url=f"https://bazaar.abuse.ch/sample/{item.get('sha256_hash')}/",
                        context={
                            "file_type": item.get("file_type"),
                            "file_name": item.get("file_name"),
                            "file_size": item.get("file_size")
                        }
                    )
                    iocs.append(ioc)

                logger.info(f"Fetched {len(iocs)} hash IOCs from MalwareBazaar")

        except Exception as e:
            logger.error(f"Failed to fetch MalwareBazaar IOCs: {e}")

        return iocs

    def add_ioc(self, ioc: IOCEntry) -> bool:
        """Add IOC to repository"""
        try:
            # Store in Firestore
            doc_ref = self.firestore_client.collection('iocs').document(ioc.ioc_id)
            data = ioc.to_dict()
            data['tenant_id'] = self.tenant_id
            doc_ref.set(data)

            # Update in-memory cache
            self.iocs[ioc.ioc_type].append(ioc)

            logger.debug(f"Added IOC: {ioc.ioc_value} ({ioc.ioc_type})")
            return True

        except Exception as e:
            logger.error(f"Failed to add IOC: {e}")
            return False

    def update_all_feeds(self) -> Dict[str, int]:
        """
        Update all enabled threat intelligence feeds

        Returns:
            Dictionary of feed_name -> count of new items
        """
        stats = {}

        # Update ThreatFox IOCs
        if self.config["feeds"]["threatfox"]["enabled"]:
            iocs = self.fetch_iocs_threatfox(days=1)
            for ioc in iocs:
                self.add_ioc(ioc)
            stats["threatfox"] = len(iocs)

        # Update MalwareBazaar IOCs
        if self.config["feeds"]["malwarebazaar"]["enabled"]:
            iocs = self.fetch_iocs_malwarebazaar(limit=100)
            for ioc in iocs:
                self.add_ioc(ioc)
            stats["malwarebazaar"] = len(iocs)

        # Update Emerging Threats YARA rules
        if self.config["feeds"]["emergingthreats"]["enabled"]:
            rules = self.fetch_yara_rules_emergingthreats()
            for rule in rules:
                self.add_yara_rule(rule)
            stats["emergingthreats_yara"] = len(rules)

        # Publish update notification
        self._publish_ioc_update(stats)

        logger.info(f"Feed update complete: {stats}")
        return stats

    def get_iocs_by_type(self, ioc_type: str, limit: Optional[int] = None) -> List[IOCEntry]:
        """Get IOCs by type"""
        # Query Firestore for fresh data
        try:
            query = self.firestore_client.collection('iocs').where('ioc_type', '==', ioc_type)
            query = query.where('tenant_id', '==', self.tenant_id)

            if limit:
                query = query.limit(limit)

            docs = query.stream()

            iocs = []
            for doc in docs:
                data = doc.to_dict()
                ioc = IOCEntry(**data)
                ioc.source = FeedSource(data['source'])
                iocs.append(ioc)

            return iocs

        except Exception as e:
            logger.error(f"Failed to query IOCs: {e}")
            return []

    def export_iocs_for_thor(self) -> Dict[str, Any]:
        """
        Export IOCs in THOR-compatible format

        Returns:
            Dictionary with IOC sets by type
        """
        try:
            export_data = {
                "ips": [],
                "domains": [],
                "file_hashes": {},
                "filenames": [],
                "registry_keys": [],
                "mutex_names": [],
                "user_agents": [],
                "metadata": {
                    "generated_at": datetime.utcnow().isoformat(),
                    "tenant_id": self.tenant_id,
                    "total_iocs": 0
                }
            }

            # Fetch IOCs by type
            for ioc_type in ["ip", "domain", "hash", "filename"]:
                iocs = self.get_iocs_by_type(ioc_type)

                if ioc_type == "ip":
                    export_data["ips"] = [ioc.ioc_value for ioc in iocs]
                elif ioc_type == "domain":
                    export_data["domains"] = [ioc.ioc_value for ioc in iocs]
                elif ioc_type == "hash":
                    export_data["file_hashes"] = {
                        ioc.ioc_value: ioc.malware_family or "unknown"
                        for ioc in iocs
                    }
                elif ioc_type == "filename":
                    export_data["filenames"] = [ioc.ioc_value for ioc in iocs]

            export_data["metadata"]["total_iocs"] = (
                len(export_data["ips"]) +
                len(export_data["domains"]) +
                len(export_data["file_hashes"]) +
                len(export_data["filenames"])
            )

            # Store in Firestore for THOR agents to retrieve
            doc_ref = self.firestore_client.collection('threat_intel').document('iocs')
            doc_ref.set(export_data)

            logger.info(f"Exported {export_data['metadata']['total_iocs']} IOCs for THOR")
            return export_data

        except Exception as e:
            logger.error(f"Failed to export IOCs: {e}")
            return {}

    # ===== Pub/Sub Notifications =====

    def _publish_rule_update(self, rule_type: str, rule_id: str, action: str) -> None:
        """Publish rule update notification"""
        try:
            topic_path = self.publisher.topic_path(self.project_id, self.rule_updates_topic)

            message = {
                "rule_type": rule_type,
                "rule_id": rule_id,
                "action": action,
                "timestamp": datetime.utcnow().isoformat(),
                "tenant_id": self.tenant_id
            }

            future = self.publisher.publish(
                topic_path,
                json.dumps(message).encode('utf-8'),
                rule_type=rule_type,
                action=action
            )

            logger.debug(f"Published rule update notification: {rule_id}")

        except Exception as e:
            logger.error(f"Failed to publish rule update: {e}")

    def _publish_ioc_update(self, stats: Dict[str, int]) -> None:
        """Publish IOC update notification"""
        try:
            topic_path = self.publisher.topic_path(self.project_id, self.ioc_updates_topic)

            message = {
                "stats": stats,
                "timestamp": datetime.utcnow().isoformat(),
                "tenant_id": self.tenant_id
            }

            future = self.publisher.publish(
                topic_path,
                json.dumps(message).encode('utf-8')
            )

            logger.debug(f"Published IOC update notification")

        except Exception as e:
            logger.error(f"Failed to publish IOC update: {e}")

    # ===== Scheduled Updates =====

    async def run_scheduled_updates(self) -> None:
        """Run scheduled feed updates"""
        logger.info("Starting scheduled feed updates")

        while True:
            try:
                # Update all feeds
                stats = self.update_all_feeds()

                # Export IOCs for THOR
                self.export_iocs_for_thor()

                # Sleep until next update (default: 1 hour)
                await asyncio.sleep(3600)

            except Exception as e:
                logger.error(f"Scheduled update error: {e}")
                await asyncio.sleep(300)  # Retry in 5 minutes


async def main():
    """Main entry point"""
    import os
    manager = VALHALLAFeedManager()

    # Perform initial feed update
    print("Updating threat intelligence feeds...")
    stats = manager.update_all_feeds()
    print(f"Feed update stats: {json.dumps(stats, indent=2)}")

    # Export IOCs for THOR agents
    print("\nExporting IOCs for THOR agents...")
    ioc_export = manager.export_iocs_for_thor()
    print(f"Exported {ioc_export['metadata']['total_iocs']} IOCs")

    # Get YARA rules by category
    print("\nFetching ransomware YARA rules...")
    rules = manager.get_yara_rules(categories=[ThreatCategory.RANSOMWARE])
    print(f"Found {len(rules)} ransomware YARA rules")

    # Run scheduled updates (uncomment for daemon mode)
    # await manager.run_scheduled_updates()


if __name__ == "__main__":
    asyncio.run(main())
