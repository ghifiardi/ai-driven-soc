"""
GATRA A2UI - Agent-to-User Interface for AI-Driven SOC

This package provides tools for GATRA agents (ADA, TAA, CRA, CLA) to generate
declarative user interfaces using Google's A2UI protocol.

Key Features:
- Security-first: No code execution, only declarative JSON
- Type-safe builders for common SOC screens
- Audit trail integration
- Multi-agent composition support

Example:
    from gatra_a2ui import SituationBriefBuilder

    builder = SituationBriefBuilder()
    builder.add_metric_card("Active Incidents", value=7, severity="high")
    schema = builder.build()
"""

from .builders import (
    SituationBriefBuilder,
    AttackNarrativeBuilder,
    DecisionPanelBuilder,
    ConfidenceMonitorBuilder,
    InvestigationWorkspaceBuilder,
    ExecutiveViewBuilder,
)
from .components import (
    MetricCard,
    TimelineStep,
    RiskIndicator,
    MITREBadge,
    EvidenceCard,
    ActionItem,
    ConfidenceGauge,
)
from .publisher import A2UIPublisher
from .catalog import GATRA_COMPONENT_CATALOG

__version__ = "1.0.0"
__all__ = [
    "SituationBriefBuilder",
    "AttackNarrativeBuilder",
    "DecisionPanelBuilder",
    "ConfidenceMonitorBuilder",
    "InvestigationWorkspaceBuilder",
    "ExecutiveViewBuilder",
    "MetricCard",
    "TimelineStep",
    "RiskIndicator",
    "MITREBadge",
    "EvidenceCard",
    "ActionItem",
    "ConfidenceGauge",
    "A2UIPublisher",
    "GATRA_COMPONENT_CATALOG",
]
