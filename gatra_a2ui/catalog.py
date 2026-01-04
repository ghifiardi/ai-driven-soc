"""
GATRA A2UI Component Catalog

Defines the trusted component catalog for the GATRA SOC frontend.
"""

GATRA_COMPONENT_CATALOG = {
    "catalog_version": "1.0.0",
    "namespace": "gatra.soc",
    "description": "Component catalog for GATRA AI-Driven SOC A2UI interfaces",
    "components": {
        # Basic Components
        "Text": {
            "description": "Text display component",
            "props": {
                "text": {"type": "object", "required": True},
                "style": {
                    "type": "enum",
                    "values": ["heading1", "heading2", "heading3", "body", "caption"],
                    "default": "body"
                }
            },
            "security": {
                "allows_user_content": False,
                "allows_html": False,
                "allows_scripts": False
            }
        },
        "Button": {
            "description": "Interactive button component",
            "props": {
                "label": {"type": "object", "required": True},
                "action": {"type": "object", "required": True},
                "style": {
                    "type": "enum",
                    "values": ["primary", "secondary", "danger"],
                    "default": "primary"
                }
            },
            "security": {
                "allows_user_content": False,
                "requires_authentication": True,
                "logs_all_actions": True
            }
        },
        "Card": {
            "description": "Card container component",
            "props": {
                "children": {"type": "array", "items": "string"},
                "title": {"type": "object", "required": False}
            },
            "security": {
                "allows_user_content": False
            }
        },

        # SOC-Specific Components
        "MetricCard": {
            "description": "Metric display card for KPIs and counts",
            "props": {
                "title": {"type": "object", "required": True},
                "value": {"type": "object", "required": True},
                "severity": {
                    "type": "enum",
                    "values": ["low", "medium", "high", "critical"],
                    "required": False
                },
                "trend": {
                    "type": "enum",
                    "values": ["up", "down", "stable"],
                    "required": False
                },
                "icon": {"type": "string", "required": False}
            },
            "security": {
                "allows_user_content": False
            }
        },
        "TimelineStep": {
            "description": "Attack timeline step with MITRE mapping",
            "props": {
                "step_number": {"type": "integer", "required": True},
                "title": {"type": "object", "required": True},
                "description": {"type": "object", "required": True},
                "technique": {
                    "type": "object",
                    "required": False,
                    "fields": {
                        "mitre_id": "string",
                        "name": "string",
                        "tactic": "string"
                    }
                },
                "confidence": {"type": "float", "min": 0.0, "max": 1.0, "required": False},
                "evidence_ids": {"type": "array", "items": "string", "required": False},
                "uncertainty": {"type": "object", "required": False}
            },
            "security": {
                "allows_user_content": False
            }
        },
        "RiskIndicator": {
            "description": "Risk level indicator with confidence",
            "props": {
                "risk_level": {
                    "type": "enum",
                    "values": ["low", "medium", "high", "critical"],
                    "required": True
                },
                "confidence": {"type": "float", "min": 0.0, "max": 1.0, "required": True},
                "label": {"type": "object", "required": True},
                "description": {"type": "object", "required": False}
            },
            "security": {
                "allows_user_content": False
            }
        },
        "MITREBadge": {
            "description": "MITRE ATT&CK technique badge",
            "props": {
                "technique_id": {"type": "string", "required": True, "pattern": "T\\d{4}(\\.\\d{3})?"},
                "technique_name": {"type": "string", "required": True},
                "tactic": {"type": "string", "required": True},
                "confidence": {"type": "float", "min": 0.0, "max": 1.0, "required": False}
            },
            "security": {
                "allows_user_content": False
            }
        },
        "EvidenceCard": {
            "description": "Evidence display card for logs, files, network data",
            "props": {
                "evidence_id": {"type": "string", "required": True},
                "evidence_type": {
                    "type": "enum",
                    "values": ["log", "file", "network", "process", "registry", "other"],
                    "required": True
                },
                "title": {"type": "object", "required": True},
                "description": {"type": "object", "required": True},
                "timestamp": {"type": "string", "format": "ISO8601", "required": False},
                "source": {"type": "string", "required": False},
                "severity": {
                    "type": "enum",
                    "values": ["low", "medium", "high", "critical"],
                    "required": False
                },
                "metadata": {"type": "object", "required": False}
            },
            "security": {
                "allows_user_content": False,
                "sanitize_metadata": True
            }
        },
        "ActionItem": {
            "description": "Containment/response action recommendation",
            "props": {
                "action_id": {"type": "string", "required": True},
                "title": {"type": "object", "required": True},
                "description": {"type": "object", "required": True},
                "risk": {
                    "type": "enum",
                    "values": ["low", "medium", "high", "critical"],
                    "required": True
                },
                "impact": {"type": "object", "required": True},
                "recommended": {"type": "boolean", "default": False},
                "estimated_time": {"type": "string", "required": False}
            },
            "security": {
                "allows_user_content": False,
                "requires_approval": True
            }
        },
        "ConfidenceGauge": {
            "description": "Agent confidence and health indicator",
            "props": {
                "agent_name": {"type": "string", "required": True},
                "overall_confidence": {"type": "float", "min": 0.0, "max": 1.0, "required": True},
                "data_coverage": {"type": "float", "min": 0.0, "max": 1.0, "required": True},
                "model_drift": {"type": "float", "min": 0.0, "required": True},
                "status": {
                    "type": "enum",
                    "values": ["healthy", "warning", "critical", "unknown"],
                    "required": True
                },
                "last_training": {"type": "string", "format": "ISO8601", "required": False},
                "last_action": {"type": "string", "format": "ISO8601", "required": False},
                "alerts": {"type": "array", "items": "string", "required": False}
            },
            "security": {
                "allows_user_content": False
            }
        },
        "ApprovalWorkflow": {
            "description": "Human-in-the-loop approval workflow",
            "props": {
                "actions": {"type": "array", "items": "string", "required": True},
                "approve_label": {"type": "object", "required": True},
                "reject_label": {"type": "object", "required": True},
                "note_required": {"type": "boolean", "default": True},
                "note_placeholder": {"type": "object", "required": False}
            },
            "security": {
                "requires_authentication": True,
                "requires_audit_trail": True,
                "logs_all_actions": True,
                "two_factor_auth": False  # Can be enabled for high-risk actions
            }
        },
        "ImpactSimulation": {
            "description": "Containment action impact simulation",
            "props": {
                "affected_users": {"type": "integer", "required": True},
                "affected_systems": {"type": "integer", "required": True},
                "estimated_downtime": {"type": "string", "required": True},
                "business_impact": {"type": "string", "required": True}
            },
            "security": {
                "allows_user_content": False
            }
        },
        "Alert": {
            "description": "System alert/notification component",
            "props": {
                "severity": {
                    "type": "enum",
                    "values": ["info", "warning", "error", "critical"],
                    "required": True
                },
                "message": {"type": "object", "required": True},
                "action": {"type": "object", "required": False}
            },
            "security": {
                "allows_user_content": False
            }
        },
        "BusinessImpactSummary": {
            "description": "Executive-level business impact summary",
            "props": {
                "current_risk_level": {
                    "type": "enum",
                    "values": ["low", "medium", "high", "critical"],
                    "required": True
                },
                "active_threats": {"type": "integer", "required": True},
                "incidents_today": {"type": "integer", "required": True},
                "mttr": {"type": "string", "required": True}
            },
            "security": {
                "allows_user_content": False
            }
        },
        "ComplianceIndicator": {
            "description": "Compliance framework status indicator",
            "props": {
                "framework": {
                    "type": "enum",
                    "values": ["GDPR", "NIS2", "PCI-DSS", "SOC2", "ISO27001", "HIPAA"],
                    "required": True
                },
                "status": {
                    "type": "enum",
                    "values": ["compliant", "non_compliant", "in_progress", "not_applicable"],
                    "required": True
                },
                "score": {"type": "float", "min": 0.0, "max": 1.0, "required": False},
                "last_audit": {"type": "string", "format": "ISO8601", "required": False}
            },
            "security": {
                "allows_user_content": False
            }
        }
    },
    "security_policies": {
        "default_policy": {
            "allows_user_content": False,
            "allows_html": False,
            "allows_scripts": False,
            "requires_authentication": True,
            "requires_audit_trail": True
        },
        "component_validation": {
            "enabled": True,
            "strict_mode": True,
            "reject_unknown_components": True
        },
        "content_sanitization": {
            "enabled": True,
            "strip_html_tags": True,
            "escape_special_chars": True
        }
    }
}
