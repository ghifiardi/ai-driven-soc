"""
GATRA A2UI Screen Builders

High-level builders for constructing common SOC screens using A2UI.
"""

from typing import List, Dict, Any, Optional
from .components import (
    MetricCard,
    TimelineStep,
    RiskIndicator,
    MITREBadge,
    EvidenceCard,
    ActionItem,
    ConfidenceGauge,
    TextComponent,
    ButtonComponent,
    CardComponent,
    Severity,
    ComponentStatus,
    MITRETechnique,
)


class A2UIBuilder:
    """Base builder class for A2UI schemas"""

    def __init__(self):
        self.components: List[Dict[str, Any]] = []
        self._component_counter = 0

    def _generate_id(self, prefix: str = "component") -> str:
        """Generate unique component ID"""
        self._component_counter += 1
        return f"{prefix}_{self._component_counter}"

    def _add_component(self, component: Any, component_id: Optional[str] = None) -> str:
        """Add a component to the schema"""
        if component_id is None:
            component_id = self._generate_id()

        a2ui_component = component.to_a2ui(component_id)
        self.components.append(a2ui_component)
        return component_id

    def build(self) -> Dict[str, Any]:
        """Build final A2UI schema"""
        return {
            "surfaceUpdate": {
                "components": self.components
            }
        }


class SituationBriefBuilder(A2UIBuilder):
    """Builder for Situation Brief screen (landing view)"""

    def __init__(self):
        super().__init__()
        self.add_header("GATRA SOC - Situation Brief")

    def add_header(self, title: str) -> str:
        """Add screen header"""
        return self._add_component(
            TextComponent(text=title, style="heading1"),
            component_id="header"
        )

    def add_metric_card(
        self,
        title: str,
        value: str,
        severity: Optional[Severity] = None,
        trend: Optional[str] = None,
        icon: Optional[str] = None
    ) -> str:
        """Add a metric display card"""
        return self._add_component(
            MetricCard(
                title=title,
                value=value,
                severity=severity,
                trend=trend,
                icon=icon
            )
        )

    def add_threat_actor_card(self, threat_actor: str, severity: Severity = Severity.CRITICAL) -> str:
        """Add threat actor identification card"""
        return self.add_metric_card(
            title="Likely Threat Actor",
            value=threat_actor,
            severity=severity,
            icon="threat_actor"
        )

    def add_incident_list(self, incidents: List[str]) -> str:
        """Add list of active incidents"""
        card_id = self._generate_id("incident_list_card")
        list_id = self._generate_id("incident_list")

        # Add individual incident items
        for incident in incidents:
            self._add_component(
                TextComponent(text=f"• {incident}", style="body")
            )

        return card_id

    def add_primary_action(
        self,
        label: str,
        action_name: str,
        action_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add primary action button"""
        return self._add_component(
            ButtonComponent(
                label=label,
                action_name=action_name,
                action_params=action_params or {},
                style="primary"
            ),
            component_id="primary_action"
        )


class AttackNarrativeBuilder(A2UIBuilder):
    """Builder for Attack Narrative screen (explainability view)"""

    def __init__(self, incident_id: str):
        super().__init__()
        self.incident_id = incident_id
        self.add_header(f"Attack Narrative: {incident_id}")

    def add_header(self, title: str) -> str:
        """Add screen header"""
        return self._add_component(
            TextComponent(text=title, style="heading2"),
            component_id="narrative_header"
        )

    def add_step(
        self,
        step: int,
        title: str,
        description: str,
        mitre_technique: Optional[str] = None,
        mitre_name: Optional[str] = None,
        mitre_tactic: Optional[str] = None,
        confidence: Optional[float] = None,
        evidence: Optional[List[str]] = None,
        uncertainty: Optional[str] = None
    ) -> str:
        """Add timeline step to narrative"""
        technique = None
        if mitre_technique and mitre_name:
            technique = MITRETechnique(
                technique_id=mitre_technique,
                technique_name=mitre_name,
                tactic=mitre_tactic
            )

        return self._add_component(
            TimelineStep(
                step_number=step,
                title=title,
                description=description,
                technique=technique,
                confidence=confidence,
                evidence_ids=evidence or [],
                uncertainty=uncertainty
            )
        )

    def add_expand_action(self, step_number: Optional[int] = None) -> str:
        """Add button to request more evidence"""
        return self._add_component(
            ButtonComponent(
                label="Request More Evidence",
                action_name="expand_analysis",
                action_params={"step": step_number} if step_number else {},
                style="secondary"
            ),
            component_id="expand_action"
        )

    def add_challenge_button(self, step_number: int) -> str:
        """Add button to challenge an assumption"""
        return self._add_component(
            ButtonComponent(
                label="Challenge This Assumption",
                action_name="challenge_assumption",
                action_params={"step": step_number},
                style="secondary"
            )
        )


class DecisionPanelBuilder(A2UIBuilder):
    """Builder for Decision Panel screen (human-in-the-loop)"""

    def __init__(self, incident_id: str):
        super().__init__()
        self.incident_id = incident_id
        self.action_items: List[str] = []
        self.add_header("Containment Recommendation")

    def add_header(self, title: str) -> str:
        """Add screen header"""
        return self._add_component(
            TextComponent(text=title, style="heading2"),
            component_id="decision_header"
        )

    def add_action(
        self,
        action_id: str,
        title: str,
        description: str,
        risk: Severity,
        impact: str,
        recommended: bool = False,
        estimated_time: Optional[str] = None
    ) -> str:
        """Add containment action option"""
        component_id = self._add_component(
            ActionItem(
                action_id=action_id,
                title=title,
                description=description,
                risk=risk,
                impact=impact,
                recommended=recommended,
                estimated_time=estimated_time
            )
        )
        self.action_items.append(action_id)
        return component_id

    def add_approval_workflow(
        self,
        approve_label: str = "Execute Containment",
        reject_label: str = "Manual Review First",
        note_required: bool = True
    ) -> str:
        """Add approval workflow buttons"""
        return self._add_component(
            {
                "to_a2ui": lambda cid: {
                    "id": cid,
                    "component": {
                        "ApprovalWorkflow": {
                            "actions": self.action_items,
                            "approve_label": {"literalString": approve_label},
                            "reject_label": {"literalString": reject_label},
                            "note_required": note_required,
                            "note_placeholder": {"literalString": "Document decision rationale for audit..."}
                        }
                    }
                }
            },
            component_id="approval_workflow"
        )

    def add_impact_simulation(
        self,
        affected_users: int,
        affected_systems: int,
        estimated_downtime: str,
        business_impact: str
    ) -> str:
        """Add impact simulation display"""
        return self._add_component(
            {
                "to_a2ui": lambda cid: {
                    "id": cid,
                    "component": {
                        "ImpactSimulation": {
                            "affected_users": affected_users,
                            "affected_systems": affected_systems,
                            "estimated_downtime": estimated_downtime,
                            "business_impact": business_impact
                        }
                    }
                }
            },
            component_id="impact_simulation"
        )


class InvestigationWorkspaceBuilder(A2UIBuilder):
    """Builder for Investigation Workspace screen"""

    def __init__(self, incident_id: str):
        super().__init__()
        self.incident_id = incident_id
        self.add_header(f"Investigation: {incident_id}")

    def add_header(self, title: str) -> str:
        """Add screen header"""
        return self._add_component(
            TextComponent(text=title, style="heading2"),
            component_id="investigation_header"
        )

    def add_timeline(self, events: List[Dict[str, Any]]) -> str:
        """Add event timeline"""
        # Implementation would create timeline visualization
        pass

    def add_evidence_card(
        self,
        evidence_id: str,
        evidence_type: str,
        title: str,
        description: str,
        timestamp: Optional[str] = None,
        source: Optional[str] = None,
        severity: Optional[Severity] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add evidence card"""
        return self._add_component(
            EvidenceCard(
                evidence_id=evidence_id,
                evidence_type=evidence_type,
                title=title,
                description=description,
                timestamp=timestamp,
                source=source,
                severity=severity,
                metadata=metadata or {}
            )
        )

    def add_mitre_badge(
        self,
        technique_id: str,
        technique_name: str,
        tactic: str,
        confidence: Optional[float] = None
    ) -> str:
        """Add MITRE ATT&CK technique badge"""
        return self._add_component(
            MITREBadge(
                technique_id=technique_id,
                technique_name=technique_name,
                tactic=tactic,
                confidence=confidence
            )
        )


class ConfidenceMonitorBuilder(A2UIBuilder):
    """Builder for Agent Confidence Monitor screen"""

    def __init__(self):
        super().__init__()
        self.add_header("Agent Confidence & Health")

    def add_header(self, title: str) -> str:
        """Add screen header"""
        return self._add_component(
            TextComponent(text=title, style="heading2"),
            component_id="confidence_header"
        )

    def add_agent_gauge(
        self,
        agent_name: str,
        overall_confidence: float,
        data_coverage: float,
        model_drift: float,
        status: ComponentStatus,
        last_training: Optional[str] = None,
        last_action: Optional[str] = None,
        alerts: Optional[List[str]] = None
    ) -> str:
        """Add confidence gauge for an agent"""
        return self._add_component(
            ConfidenceGauge(
                agent_name=agent_name,
                overall_confidence=overall_confidence,
                data_coverage=data_coverage,
                model_drift=model_drift,
                status=status,
                last_training=last_training,
                last_action=last_action,
                alerts=alerts or []
            )
        )

    def add_alert(self, severity: Severity, message: str, action_label: str, action_name: str) -> str:
        """Add system alert"""
        return self._add_component(
            {
                "to_a2ui": lambda cid: {
                    "id": cid,
                    "component": {
                        "Alert": {
                            "severity": severity.value,
                            "message": {"literalString": message},
                            "action": {
                                "label": {"literalString": action_label},
                                "action": {"name": action_name}
                            }
                        }
                    }
                }
            }
        )


class ExecutiveViewBuilder(A2UIBuilder):
    """Builder for Executive & Regulator View screen"""

    def __init__(self):
        super().__init__()
        self.add_header("Executive Security Posture")

    def add_header(self, title: str) -> str:
        """Add screen header"""
        return self._add_component(
            TextComponent(text=title, style="heading1"),
            component_id="executive_header"
        )

    def add_business_impact_summary(
        self,
        current_risk_level: Severity,
        active_threats: int,
        incidents_today: int,
        mttr: str
    ) -> str:
        """Add business impact summary"""
        return self._add_component(
            {
                "to_a2ui": lambda cid: {
                    "id": cid,
                    "component": {
                        "BusinessImpactSummary": {
                            "current_risk_level": current_risk_level.value,
                            "active_threats": active_threats,
                            "incidents_today": incidents_today,
                            "mttr": mttr
                        }
                    }
                }
            },
            component_id="business_impact"
        )

    def add_compliance_indicator(
        self,
        framework: str,
        status: str,
        score: float,
        last_audit: str
    ) -> str:
        """Add compliance framework indicator"""
        return self._add_component(
            {
                "to_a2ui": lambda cid: {
                    "id": cid,
                    "component": {
                        "ComplianceIndicator": {
                            "framework": framework,
                            "status": status,
                            "score": score,
                            "last_audit": last_audit
                        }
                    }
                }
            }
        )

    def add_export_button(self, format_type: str, label: str) -> str:
        """Add export button"""
        return self._add_component(
            ButtonComponent(
                label=label,
                action_name="export_report",
                action_params={"format": format_type},
                style="secondary"
            )
        )
