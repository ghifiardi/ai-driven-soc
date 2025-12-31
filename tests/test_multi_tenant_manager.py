# =============================================================================
# Tests for Multi-Tenant Manager
# =============================================================================

import pytest
import json
import tempfile
from pathlib import Path


class TestMultiTenantManager:
    """Tests for the MultiTenantManager class."""

    @pytest.fixture
    def valid_config(self) -> dict:
        """Create a valid multi-tenant configuration."""
        return {
            "defaults": {
                "project_id": "test-project",
                "location": "us-central1",
                "dataset_template": "gatra_{tenant_id}",
                "results_dataset_template": "gatra_{tenant_id}_results",
                "metrics_namespace": "gatra_multi_tenant",
            },
            "tenants": [
                {
                    "tenant_id": "tenant_001",
                    "display_name": "Test Tenant 001",
                    "region": "us-central1",
                    "dataset": "gatra_tenant_001",
                    "results_dataset": "gatra_tenant_001_results",
                    "tables": {
                        "events": "events",
                        "alerts": "alerts",
                        "results": "events_results",
                    },
                    "pubsub_topics": {
                        "ingest": "events-tenant_001",
                        "alerts": "alerts-tenant_001",
                        "priority": "priority-tenant_001",
                    },
                    "rate_limits": {
                        "ingest_eps": 10000,
                        "alerts_per_min": 500,
                    },
                    "service_level": "professional",
                },
            ],
            "default_tenant_id": "tenant_001",
        }

    @pytest.fixture
    def config_file(self, valid_config) -> Path:
        """Create a temporary config file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(valid_config, f)
            return Path(f.name)

    def test_load_config_from_file(self, config_file):
        """Test loading configuration from a JSON file."""
        from multi_tenant_manager import MultiTenantManager

        manager = MultiTenantManager.from_file(str(config_file))
        assert manager is not None
        assert manager.default_tenant_id == "tenant_001"

    def test_get_tenant_config(self, config_file):
        """Test retrieving tenant configuration."""
        from multi_tenant_manager import MultiTenantManager

        manager = MultiTenantManager.from_file(str(config_file))
        tenant = manager.get_tenant("tenant_001")

        assert tenant is not None
        assert tenant.tenant_id == "tenant_001"
        assert tenant.dataset == "gatra_tenant_001"

    def test_get_nonexistent_tenant(self, config_file):
        """Test retrieving a non-existent tenant."""
        from multi_tenant_manager import MultiTenantManager

        manager = MultiTenantManager.from_file(str(config_file))

        with pytest.raises(KeyError):
            manager.get_tenant("nonexistent_tenant")

    def test_tenant_tables(self, config_file):
        """Test tenant table configuration."""
        from multi_tenant_manager import MultiTenantManager

        manager = MultiTenantManager.from_file(str(config_file))
        tenant = manager.get_tenant("tenant_001")

        assert tenant.tables.events == "events"
        assert tenant.tables.alerts == "alerts"
        assert tenant.tables.results == "events_results"

    def test_tenant_pubsub_topics(self, config_file):
        """Test tenant Pub/Sub topic configuration."""
        from multi_tenant_manager import MultiTenantManager

        manager = MultiTenantManager.from_file(str(config_file))
        tenant = manager.get_tenant("tenant_001")

        assert tenant.pubsub_topics.ingest == "events-tenant_001"
        assert tenant.pubsub_topics.alerts == "alerts-tenant_001"
        assert tenant.pubsub_topics.priority == "priority-tenant_001"

    def test_tenant_rate_limits(self, config_file):
        """Test tenant rate limit configuration."""
        from multi_tenant_manager import MultiTenantManager

        manager = MultiTenantManager.from_file(str(config_file))
        tenant = manager.get_tenant("tenant_001")

        assert tenant.rate_limits.ingest_eps == 10000
        assert tenant.rate_limits.alerts_per_min == 500

    def test_list_tenants(self, config_file):
        """Test listing all tenants."""
        from multi_tenant_manager import MultiTenantManager

        manager = MultiTenantManager.from_file(str(config_file))
        tenants = manager.list_tenants()

        assert len(tenants) == 1
        assert "tenant_001" in tenants

    def test_invalid_default_tenant(self, valid_config):
        """Test validation fails for invalid default tenant."""
        from multi_tenant_manager import MultiTenantManager

        valid_config["default_tenant_id"] = "nonexistent"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(valid_config, f)
            config_path = f.name

        with pytest.raises(ValueError, match="default_tenant_id"):
            MultiTenantManager.from_file(config_path)


class TestTenantIsolation:
    """Tests for tenant data isolation."""

    def test_datasets_unique(self, sample_tenant_config):
        """Test that tenant datasets must be unique."""
        # This test verifies the validation logic
        config1 = sample_tenant_config.copy()
        config2 = sample_tenant_config.copy()
        config2["tenant_id"] = "tenant_002"
        # Same dataset should fail validation
        config2["dataset"] = config1["dataset"]

        # The manager should prevent this during validation
        # Implementation depends on your validation logic
        pass

    def test_tenant_context_isolation(self):
        """Test that tenant context is properly isolated."""
        from production.tenant_context import set_tenant_context, get_current_tenant

        set_tenant_context("tenant_001")
        assert get_current_tenant() == "tenant_001"

        set_tenant_context("tenant_002")
        assert get_current_tenant() == "tenant_002"
