"""
GATRA A2UI Component Definitions

Type-safe component classes for building A2UI schemas.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Literal
from enum import Enum


class Severity(str, Enum):
    """Severity levels for threats and alerts"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComponentStatus(str, Enum):
    """Status indicators"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class MetricCard:
    """Metric display card component"""
    title: str
    value: str
    severity: Optional[Severity] = None
    trend: Optional[Literal["up", "down", "stable"]] = None
    icon: Optional[str] = None

    def to_a2ui(self, component_id: str) -> Dict[str, Any]:
        return {
            "id": component_id,
            "component": {
                "MetricCard": {
                    "title": {"literalString": self.title},
                    "value": {"literalString": str(self.value)},
                    **({"severity": self.severity.value} if self.severity else {}),
                    **({"trend": self.trend} if self.trend else {}),
                    **({"icon": self.icon} if self.icon else {}),
                }
            }
        }


@dataclass
class MITRETechnique:
    """MITRE ATT&CK technique reference"""
    technique_id: str  # e.g., "T1566.001"
    technique_name: str  # e.g., "Spearphishing Attachment"
    tactic: Optional[str] = None  # e.g., "Initial Access"


@dataclass
class TimelineStep:
    """Attack timeline step component"""
    step_number: int
    title: str
    description: str
    technique: Optional[MITRETechnique] = None
    confidence: Optional[float] = None
    evidence_ids: List[str] = field(default_factory=list)
    uncertainty: Optional[str] = None

    def to_a2ui(self, component_id: str) -> Dict[str, Any]:
        component_data = {
            "step_number": self.step_number,
            "title": {"literalString": self.title},
            "description": {"literalString": self.description},
        }

        if self.technique:
            component_data["technique"] = {
                "mitre_id": self.technique.technique_id,
                "name": self.technique.technique_name,
            }
            if self.technique.tactic:
                component_data["technique"]["tactic"] = self.technique.tactic

        if self.confidence is not None:
            component_data["confidence"] = self.confidence

        if self.evidence_ids:
            component_data["evidence_ids"] = self.evidence_ids

        if self.uncertainty:
            component_data["uncertainty"] = {"literalString": self.uncertainty}

        return {
            "id": component_id,
            "component": {
                "TimelineStep": component_data
            }
        }


@dataclass
class RiskIndicator:
    """Risk level indicator component"""
    risk_level: Severity
    confidence: float
    label: str
    description: Optional[str] = None

    def to_a2ui(self, component_id: str) -> Dict[str, Any]:
        return {
            "id": component_id,
            "component": {
                "RiskIndicator": {
                    "risk_level": self.risk_level.value,
                    "confidence": self.confidence,
                    "label": {"literalString": self.label},
                    **({"description": {"literalString": self.description}} if self.description else {}),
                }
            }
        }


@dataclass
class MITREBadge:
    """MITRE ATT&CK technique badge"""
    technique_id: str
    technique_name: str
    tactic: str
    confidence: Optional[float] = None

    def to_a2ui(self, component_id: str) -> Dict[str, Any]:
        return {
            "id": component_id,
            "component": {
                "MITREBadge": {
                    "technique_id": self.technique_id,
                    "technique_name": self.technique_name,
                    "tactic": self.tactic,
                    **({"confidence": self.confidence} if self.confidence else {}),
                }
            }
        }


@dataclass
class EvidenceCard:
    """Evidence display card"""
    evidence_id: str
    evidence_type: str  # e.g., "log", "file", "network"
    title: str
    description: str
    timestamp: Optional[str] = None
    source: Optional[str] = None
    severity: Optional[Severity] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_a2ui(self, component_id: str) -> Dict[str, Any]:
        return {
            "id": component_id,
            "component": {
                "EvidenceCard": {
                    "evidence_id": self.evidence_id,
                    "evidence_type": self.evidence_type,
                    "title": {"literalString": self.title},
                    "description": {"literalString": self.description},
                    **({"timestamp": self.timestamp} if self.timestamp else {}),
                    **({"source": self.source} if self.source else {}),
                    **({"severity": self.severity.value} if self.severity else {}),
                    **({"metadata": self.metadata} if self.metadata else {}),
                }
            }
        }


@dataclass
class ActionItem:
    """Containment/response action item"""
    action_id: str
    title: str
    description: str
    risk: Severity
    impact: str
    recommended: bool = False
    estimated_time: Optional[str] = None

    def to_a2ui(self, component_id: str) -> Dict[str, Any]:
        return {
            "id": component_id,
            "component": {
                "ActionItem": {
                    "action_id": self.action_id,
                    "title": {"literalString": self.title},
                    "description": {"literalString": self.description},
                    "risk": self.risk.value,
                    "impact": {"literalString": self.impact},
                    "recommended": self.recommended,
                    **({"estimated_time": self.estimated_time} if self.estimated_time else {}),
                }
            }
        }


@dataclass
class ConfidenceGauge:
    """Agent confidence gauge component"""
    agent_name: str
    overall_confidence: float
    data_coverage: float
    model_drift: float
    status: ComponentStatus
    last_training: Optional[str] = None
    last_action: Optional[str] = None
    alerts: List[str] = field(default_factory=list)

    def to_a2ui(self, component_id: str) -> Dict[str, Any]:
        return {
            "id": component_id,
            "component": {
                "ConfidenceGauge": {
                    "agent_name": self.agent_name,
                    "overall_confidence": self.overall_confidence,
                    "data_coverage": self.data_coverage,
                    "model_drift": self.model_drift,
                    "status": self.status.value,
                    **({"last_training": self.last_training} if self.last_training else {}),
                    **({"last_action": self.last_action} if self.last_action else {}),
                    **({"alerts": self.alerts} if self.alerts else {}),
                }
            }
        }


@dataclass
class TextComponent:
    """Basic text component"""
    text: str
    style: Optional[Literal["heading1", "heading2", "heading3", "body", "caption"]] = "body"

    def to_a2ui(self, component_id: str) -> Dict[str, Any]:
        return {
            "id": component_id,
            "component": {
                "Text": {
                    "text": {"literalString": self.text},
                    "style": self.style,
                }
            }
        }


@dataclass
class ButtonComponent:
    """Button component with action"""
    label: str
    action_name: str
    action_params: Dict[str, Any] = field(default_factory=dict)
    style: Optional[Literal["primary", "secondary", "danger"]] = "primary"

    def to_a2ui(self, component_id: str) -> Dict[str, Any]:
        return {
            "id": component_id,
            "component": {
                "Button": {
                    "label": {"literalString": self.label},
                    "action": {
                        "name": self.action_name,
                        **({"params": self.action_params} if self.action_params else {}),
                    },
                    "style": self.style,
                }
            }
        }


@dataclass
class CardComponent:
    """Card container component"""
    children: List[str] = field(default_factory=list)
    title: Optional[str] = None

    def to_a2ui(self, component_id: str) -> Dict[str, Any]:
        return {
            "id": component_id,
            "component": {
                "Card": {
                    "children": self.children,
                    **({"title": {"literalString": self.title}} if self.title else {}),
                }
            }
        }
