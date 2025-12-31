# =============================================================================
# Pytest Configuration and Fixtures
# =============================================================================

import os
import pytest
import asyncio
from typing import Generator, AsyncGenerator
from unittest.mock import MagicMock, patch


# =============================================================================
# Environment Setup
# =============================================================================

# Set test environment before importing application modules
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("JWT_SECRET", "test-secret-key-for-testing-only-32chars")
os.environ.setdefault("MULTITENANT_CONFIG_PATH", "config/gatra_multitenant_config.json")


# =============================================================================
# Async Event Loop
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_bigquery_client():
    """Mock BigQuery client for tests."""
    with patch("google.cloud.bigquery.Client") as mock:
        client = MagicMock()
        mock.return_value = client
        yield client


@pytest.fixture
def mock_redis():
    """Mock Redis client for tests."""
    with patch("redis.asyncio.from_url") as mock:
        redis_client = MagicMock()
        mock.return_value = redis_client
        yield redis_client


@pytest.fixture
def mock_pubsub():
    """Mock Pub/Sub client for tests."""
    with patch("google.cloud.pubsub_v1.PublisherClient") as pub_mock:
        with patch("google.cloud.pubsub_v1.SubscriberClient") as sub_mock:
            yield {
                "publisher": pub_mock.return_value,
                "subscriber": sub_mock.return_value,
            }


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def sample_tenant_config() -> dict:
    """Sample tenant configuration for tests."""
    return {
        "tenant_id": "test_tenant",
        "display_name": "Test Tenant",
        "region": "us-central1",
        "dataset": "gatra_test_tenant",
        "results_dataset": "gatra_test_tenant_results",
        "tables": {
            "events": "events",
            "alerts": "alerts",
            "results": "events_results",
        },
        "pubsub_topics": {
            "ingest": "events-test_tenant",
            "alerts": "alerts-test_tenant",
            "priority": "priority-test_tenant",
        },
        "rate_limits": {
            "ingest_eps": 1000,
            "alerts_per_min": 100,
        },
        "service_level": "professional",
    }


@pytest.fixture
def sample_security_event() -> dict:
    """Sample security event for tests."""
    return {
        "event_id": "evt_12345",
        "timestamp": "2024-01-15T10:30:00Z",
        "source": "firewall",
        "event_type": "intrusion_attempt",
        "severity": "high",
        "source_ip": "192.168.1.100",
        "destination_ip": "10.0.0.1",
        "destination_port": 443,
        "protocol": "TCP",
        "action": "blocked",
        "raw_log": "Sample raw log data",
    }


@pytest.fixture
def sample_alert() -> dict:
    """Sample alert for tests."""
    return {
        "alert_id": "alert_67890",
        "timestamp": "2024-01-15T10:35:00Z",
        "severity": "critical",
        "title": "Suspicious Network Activity Detected",
        "description": "Multiple failed login attempts from external IP",
        "source": "ADA",
        "confidence": 0.95,
        "related_events": ["evt_12345"],
        "recommended_action": "investigate",
        "tenant_id": "test_tenant",
    }


# =============================================================================
# FastAPI Test Client
# =============================================================================

@pytest.fixture
def test_client():
    """Create a test client for FastAPI application."""
    # Import here to avoid circular imports
    from fastapi.testclient import TestClient

    # This would import your actual FastAPI app
    # from mssp_platform_server import app
    # return TestClient(app)

    # For now, return a mock
    return MagicMock()


# =============================================================================
# Cleanup
# =============================================================================

@pytest.fixture(autouse=True)
def cleanup_env():
    """Clean up environment after each test."""
    yield
    # Any cleanup code here
