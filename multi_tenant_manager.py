"""
Multi-tenant configuration manager for the AI-driven SOC stack.

This module centralises tenant metadata (BigQuery datasets, Pub/Sub topics,
rate limits, etc.) so that ADA/TAA components can align with the GATRA
multi-tenant blueprint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, Optional


class TenantConfigError(ValueError):
    """Raised when the tenant configuration is invalid."""


class TenantNotFoundError(KeyError):
    """Raised when a tenant_id is missing from the configuration."""


@dataclass(frozen=True)
class TenantTables:
    events: str
    alerts: str
    results: str


@dataclass(frozen=True)
class TenantPubSubTopics:
    ingest: str
    alerts: str
    priority: str


@dataclass(frozen=True)
class TenantRateLimits:
    ingest_eps: int
    alerts_per_min: int


@dataclass(frozen=True)
class FirewallConfig:
    type: str
    mgmt_ip: str
    api_key: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    device_group: Optional[str] = None
    domain: Optional[str] = None


@dataclass(frozen=True)
class TenantConfig:
    tenant_id: str
    display_name: str
    region: str
    dataset: str
    results_dataset: str
    tables: TenantTables
    pubsub_topics: TenantPubSubTopics
    rate_limits: TenantRateLimits
    service_level: str
    api_key: Optional[str] = None
    firewall_config: Optional[FirewallConfig] = None


@dataclass(frozen=True)
class DefaultsConfig:
    project_id: str
    location: str
    dataset_template: str
    results_dataset_template: str
    metrics_namespace: str


@dataclass
class MultiTenantConfig:
    defaults: DefaultsConfig
    tenants: Dict[str, TenantConfig] = field(default_factory=dict)
    default_tenant_id: str = ""

    def get_tenant(self, tenant_id: Optional[str] = None) -> TenantConfig:
        tenant_key = tenant_id or self.default_tenant_id
        try:
            return self.tenants[tenant_key]
        except KeyError as exc:
            raise TenantNotFoundError(f"Unknown tenant_id '{tenant_key}'") from exc

    def list_tenants(self) -> list[str]:
        return sorted(self.tenants.keys())

    def bigquery_fqn(self, tenant_id: str, table_key: str) -> str:
        tenant = self.get_tenant(tenant_id)
        table_name = getattr(tenant.tables, table_key, None)
        if not table_name:
            raise TenantConfigError(f"Tenant '{tenant_id}' missing table mapping for '{table_key}'")
        dataset = tenant.dataset if table_key != "results" else tenant.results_dataset
        return f"{self.defaults.project_id}.{dataset}.{table_name}"

    def results_dataset(self, tenant_id: str) -> str:
        return self.get_tenant(tenant_id).results_dataset

    def pubsub_topic(self, tenant_id: str, topic_key: str) -> str:
        tenant = self.get_tenant(tenant_id)
        topic = getattr(tenant.pubsub_topics, topic_key, None)
        if not topic:
            raise TenantConfigError(f"Tenant '{tenant_id}' missing pubsub topic for '{topic_key}'")
        return topic

    def build_bigquery_args(self, tenant_id: str) -> Dict[str, str]:
        tenant = self.get_tenant(tenant_id)
        return {
            "project_id": self.defaults.project_id,
            "dataset_id": tenant.dataset,
            "table_id": tenant.tables.events,
            "location": tenant.region or self.defaults.location,
            "results_dataset_id": tenant.results_dataset,
            "results_table_id": tenant.tables.results,
        }


class MultiTenantManager:
    """Factory for loading and validating multi-tenant configuration."""

    def __init__(self, config: MultiTenantConfig):
        self._config = config

    @property
    def project_id(self) -> str:
        return self._config.defaults.project_id

    @property
    def metrics_namespace(self) -> str:
        return self._config.defaults.metrics_namespace

    @classmethod
    def from_file(cls, path: str | Path) -> "MultiTenantManager":
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Multi-tenant config file not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as handle:
            raw_config = json.load(handle)

        config = cls._parse_config(raw_config)
        cls._validate_config(config)
        return cls(config)

    def get_tenant(self, tenant_id: Optional[str] = None) -> TenantConfig:
        return self._config.get_tenant(tenant_id)

    def list_tenants(self) -> list[str]:
        return self._config.list_tenants()

    def bigquery_fqn(self, tenant_id: str, table_key: str) -> str:
        return self._config.bigquery_fqn(tenant_id, table_key)

    def pubsub_topic(self, tenant_id: str, topic_key: str) -> str:
        return self._config.pubsub_topic(tenant_id, topic_key)

    def build_bigquery_args(self, tenant_id: str) -> Dict[str, str]:
        return self._config.build_bigquery_args(tenant_id)

    def bigquery_location(self, tenant_id: Optional[str] = None) -> str:
        if tenant_id:
            tenant = self._config.get_tenant(tenant_id)
            if tenant.region:
                return tenant.region
        return self._config.defaults.location

    @property
    def default_tenant_id(self) -> str:
        return self._config.default_tenant_id

    def get_default_tenant_id(self) -> str:
        return self.default_tenant_id

    def tenants_count(self) -> int:
        return len(self._config.tenants)

    @staticmethod
    def _parse_config(raw: Dict[str, object]) -> MultiTenantConfig:
        try:
            defaults_section = raw["defaults"]
            tenants_section = raw["tenants"]
            default_tenant_id = raw["default_tenant_id"]
        except KeyError as exc:
            raise TenantConfigError(f"Missing required section in configuration: {exc}") from exc

        defaults = DefaultsConfig(
            project_id=defaults_section["project_id"],
            location=defaults_section["location"],
            dataset_template=defaults_section["dataset_template"],
            results_dataset_template=defaults_section["results_dataset_template"],
            metrics_namespace=defaults_section["metrics_namespace"],
        )

        tenants: Dict[str, TenantConfig] = {}
        for tenant_payload in tenants_section:
            tenant = TenantConfig(
                tenant_id=tenant_payload["tenant_id"],
                display_name=tenant_payload["display_name"],
                region=tenant_payload.get("region", defaults.location),
                dataset=tenant_payload.get(
                    "dataset",
                    defaults.dataset_template.format(tenant_id=tenant_payload["tenant_id"]),
                ),
                results_dataset=tenant_payload.get(
                    "results_dataset",
                    defaults.results_dataset_template.format(tenant_id=tenant_payload["tenant_id"]),
                ),
                tables=TenantTables(
                    events=tenant_payload["tables"]["events"],
                    alerts=tenant_payload["tables"]["alerts"],
                    results=tenant_payload["tables"]["results"],
                ),
                pubsub_topics=TenantPubSubTopics(
                    ingest=tenant_payload["pubsub_topics"]["ingest"],
                    alerts=tenant_payload["pubsub_topics"]["alerts"],
                    priority=tenant_payload["pubsub_topics"]["priority"],
                ),
                rate_limits=TenantRateLimits(
                    ingest_eps=tenant_payload["rate_limits"]["ingest_eps"],
                    alerts_per_min=tenant_payload["rate_limits"]["alerts_per_min"],
                ),
                service_level=tenant_payload.get("service_level", "starter"),
                api_key=tenant_payload.get("api_key"),
                firewall_config=FirewallConfig(**tenant_payload["firewall_config"]) if tenant_payload.get("firewall_config") else None,
            )
            tenants[tenant.tenant_id] = tenant

        return MultiTenantConfig(
            defaults=defaults,
            tenants=tenants,
            default_tenant_id=default_tenant_id,
        )

    @staticmethod
    def _validate_config(config: MultiTenantConfig) -> None:
        if not config.tenants:
            raise TenantConfigError("No tenants defined in configuration.")

        if config.default_tenant_id not in config.tenants:
            raise TenantConfigError(
                f"default_tenant_id '{config.default_tenant_id}' is not present in tenants list."
            )

        seen_datasets: Dict[str, str] = {}
        for tenant_id in config.list_tenants():
            tenant = config.get_tenant(tenant_id)
            if not tenant.dataset or not tenant.results_dataset:
                raise TenantConfigError(f"Tenant '{tenant.tenant_id}' missing dataset definitions.")

            if tenant.dataset in seen_datasets and seen_datasets[tenant.dataset] != tenant.tenant_id:
                raise TenantConfigError(
                    f"Dataset '{tenant.dataset}' shared by multiple tenants: "
                    f"{seen_datasets[tenant.dataset]} and {tenant.tenant_id}"
                )
            seen_datasets[tenant.dataset] = tenant.tenant_id

            for key in ("events", "alerts", "results"):
                table_value = getattr(tenant.tables, key, None)
                if not table_value:
                    raise TenantConfigError(f"Tenant '{tenant.tenant_id}' missing table for '{key}'")

            for topic_key in ("ingest", "alerts", "priority"):
                topic_value = getattr(tenant.pubsub_topics, topic_key, None)
                if not topic_value:
                    raise TenantConfigError(
                        f"Tenant '{tenant.tenant_id}' missing pubsub topic for '{topic_key}'"
                    )

            if tenant.rate_limits.ingest_eps <= 0 or tenant.rate_limits.alerts_per_min <= 0:
                raise TenantConfigError(
                    f"Tenant '{tenant.tenant_id}' must have positive rate limit values."
                )

    def add_tenant(self, tenant_config: TenantConfig) -> None:
        """Add or update a tenant configuration dynamically."""
        self._config.tenants[tenant_config.tenant_id] = tenant_config
        # Re-validate to ensure integrity
        self._validate_config(self._config)

    def save_config(self, path: str | Path) -> None:
        """Save the current configuration back to the JSON file."""
        config_path = Path(path)
        
        # Convert config back to dictionary format
        defaults = self._config.defaults
        tenants_list = []
        
        for tenant in self._config.tenants.values():
            tenant_dict = {
                "tenant_id": tenant.tenant_id,
                "display_name": tenant.display_name,
                "region": tenant.region,
                "dataset": tenant.dataset,
                "results_dataset": tenant.results_dataset,
                "tables": {
                    "events": tenant.tables.events,
                    "alerts": tenant.tables.alerts,
                    "results": tenant.tables.results
                },
                "pubsub_topics": {
                    "ingest": tenant.pubsub_topics.ingest,
                    "alerts": tenant.pubsub_topics.alerts,
                    "priority": tenant.pubsub_topics.priority
                },
                "rate_limits": {
                    "ingest_eps": tenant.rate_limits.ingest_eps,
                    "alerts_per_min": tenant.rate_limits.alerts_per_min
                },
                "service_level": tenant.service_level,
                "api_key": tenant.api_key,
                "firewall_config": asdict(tenant.firewall_config) if tenant.firewall_config else None
            }
            tenants_list.append(tenant_dict)
            
        config_dict = {
            "defaults": {
                "project_id": defaults.project_id,
                "location": defaults.location,
                "dataset_template": defaults.dataset_template,
                "results_dataset_template": defaults.results_dataset_template,
                "metrics_namespace": defaults.metrics_namespace
            },
            "tenants": tenants_list,
            "default_tenant_id": self._config.default_tenant_id
        }
        
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(config_dict, handle, indent=2)
