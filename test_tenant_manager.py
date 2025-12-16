#!/usr/bin/env python3
"""
Test Script for MultiTenantManager
==================================

Verifies the dynamic tenant management logic in MultiTenantManager.
"""

import unittest
import json
import tempfile
import os
from pathlib import Path
from multi_tenant_manager import MultiTenantManager, TenantConfig, TenantTables, TenantPubSubTopics, TenantRateLimits

class TestMultiTenantManager(unittest.TestCase):
    def setUp(self):
        # Create a temporary config file
        self.test_dir = tempfile.TemporaryDirectory()
        self.config_path = Path(self.test_dir.name) / "test_config.json"
        
        self.initial_config = {
            "defaults": {
                "project_id": "test-project",
                "location": "us-central1",
                "dataset_template": "gatra_{tenant_id}",
                "results_dataset_template": "gatra_{tenant_id}_results",
                "metrics_namespace": "gatra_test"
            },
            "tenants": [
                {
                    "tenant_id": "default_tenant",
                    "display_name": "Default Tenant",
                    "region": "us-central1",
                    "dataset": "gatra_default_tenant",
                    "results_dataset": "gatra_default_tenant_results",
                    "tables": {
                        "events": "events",
                        "alerts": "alerts",
                        "results": "results"
                    },
                    "pubsub_topics": {
                        "ingest": "ingest-topic",
                        "alerts": "alerts-topic",
                        "priority": "priority-topic"
                    },
                    "rate_limits": {
                        "ingest_eps": 100,
                        "alerts_per_min": 10
                    },
                    "service_level": "starter"
                }
            ],
            "default_tenant_id": "default_tenant"
        }
        
        with open(self.config_path, "w") as f:
            json.dump(self.initial_config, f)
            
        self.manager = MultiTenantManager.from_file(self.config_path)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_add_tenant(self):
        new_tenant = TenantConfig(
            tenant_id="new_tenant",
            display_name="New Tenant",
            region="us-west1",
            dataset="gatra_new_tenant",
            results_dataset="gatra_new_tenant_results",
            tables=TenantTables("events", "alerts", "results"),
            pubsub_topics=TenantPubSubTopics("ingest-new", "alerts-new", "priority-new"),
            rate_limits=TenantRateLimits(200, 20),
            service_level="pro"
        )
        
        self.manager.add_tenant(new_tenant)
        
        # Verify in memory
        self.assertEqual(self.manager.tenants_count(), 2)
        fetched_tenant = self.manager.get_tenant("new_tenant")
        self.assertEqual(fetched_tenant.display_name, "New Tenant")

    def test_save_config(self):
        new_tenant = TenantConfig(
            tenant_id="saved_tenant",
            display_name="Saved Tenant",
            region="us-east1",
            dataset="gatra_saved",
            results_dataset="gatra_saved_results",
            tables=TenantTables("events", "alerts", "results"),
            pubsub_topics=TenantPubSubTopics("ingest-saved", "alerts-saved", "priority-saved"),
            rate_limits=TenantRateLimits(300, 30),
            service_level="enterprise"
        )
        
        self.manager.add_tenant(new_tenant)
        self.manager.save_config(self.config_path)
        
        # Reload from file
        new_manager = MultiTenantManager.from_file(self.config_path)
        self.assertEqual(new_manager.tenants_count(), 2)
        saved_tenant = new_manager.get_tenant("saved_tenant")
        self.assertEqual(saved_tenant.region, "us-east1")

if __name__ == "__main__":
    unittest.main()
