"""
Production Security Module
==========================

Provides secure secret management, service-to-service authentication,
and input validation for production deployment.
"""

import os
import hmac
import hashlib
import time
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from functools import wraps
import secrets

logger = logging.getLogger(__name__)


# =============================================================================
# Secret Management
# =============================================================================

class SecretManager:
    """
    Production-ready secret management.

    Supports:
    - Google Cloud Secret Manager (production)
    - Environment variables (development fallback)
    - Local file-based secrets (testing only)
    """

    _instance = None
    _secrets_cache: Dict[str, str] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._backend = os.getenv("SECRET_BACKEND", "env")  # env, gcp, file
        self._gcp_project = os.getenv("GOOGLE_CLOUD_PROJECT", "")
        self._secrets_cache = {}
        self._initialized = True

        if self._backend == "gcp":
            try:
                from google.cloud import secretmanager
                self._sm_client = secretmanager.SecretManagerServiceClient()
                logger.info("SecretManager: Using Google Cloud Secret Manager")
            except ImportError:
                logger.warning("SecretManager: GCP SDK not available, falling back to env")
                self._backend = "env"
        else:
            logger.info(f"SecretManager: Using backend '{self._backend}'")

    def get_secret(self, secret_name: str, version: str = "latest") -> Optional[str]:
        """Retrieve a secret value."""
        cache_key = f"{secret_name}:{version}"

        if cache_key in self._secrets_cache:
            return self._secrets_cache[cache_key]

        value = None

        if self._backend == "gcp":
            value = self._get_gcp_secret(secret_name, version)
        elif self._backend == "file":
            value = self._get_file_secret(secret_name)
        else:
            value = os.getenv(secret_name)

        if value:
            self._secrets_cache[cache_key] = value

        return value

    def _get_gcp_secret(self, secret_name: str, version: str) -> Optional[str]:
        """Fetch secret from Google Cloud Secret Manager."""
        try:
            name = f"projects/{self._gcp_project}/secrets/{secret_name}/versions/{version}"
            response = self._sm_client.access_secret_version(request={"name": name})
            return response.payload.data.decode("UTF-8")
        except Exception as e:
            logger.error(f"Failed to fetch secret '{secret_name}' from GCP: {e}")
            return None

    def _get_file_secret(self, secret_name: str) -> Optional[str]:
        """Fetch secret from local file (for testing only)."""
        secrets_dir = os.getenv("SECRETS_DIR", "/run/secrets")
        secret_path = os.path.join(secrets_dir, secret_name)
        try:
            with open(secret_path, "r") as f:
                return f.read().strip()
        except FileNotFoundError:
            return os.getenv(secret_name)

    def clear_cache(self):
        """Clear the secrets cache (useful for rotation)."""
        self._secrets_cache.clear()


# Singleton instance
secret_manager = SecretManager()


def get_jwt_secret() -> str:
    """Get JWT secret with secure fallback."""
    secret = secret_manager.get_secret("JWT_SECRET")
    if not secret:
        # In production, fail if no secret is configured
        if os.getenv("ENVIRONMENT", "development") == "production":
            raise RuntimeError("JWT_SECRET must be configured in production")
        # Development fallback - generate a random secret
        logger.warning("Using auto-generated JWT secret - NOT FOR PRODUCTION")
        secret = secrets.token_urlsafe(32)
    return secret


def get_service_api_key(service_name: str) -> str:
    """Get API key for service-to-service authentication."""
    key = secret_manager.get_secret(f"{service_name.upper()}_API_KEY")
    if not key:
        if os.getenv("ENVIRONMENT", "development") == "production":
            raise RuntimeError(f"API key for {service_name} must be configured")
        key = secrets.token_urlsafe(32)
        logger.warning(f"Using auto-generated API key for {service_name}")
    return key


# =============================================================================
# Service-to-Service Authentication
# =============================================================================

@dataclass
class ServiceCredentials:
    """Credentials for inter-service communication."""
    service_name: str
    api_key: str
    timestamp: float
    signature: str


class ServiceAuthenticator:
    """
    HMAC-based service-to-service authentication.

    Each request includes:
    - Service name
    - Timestamp (for replay protection)
    - HMAC signature
    """

    SIGNATURE_VALIDITY_SECONDS = 300  # 5 minutes

    def __init__(self, service_name: str):
        self.service_name = service_name
        self._signing_key = get_service_api_key(service_name)

    def create_auth_headers(self, body: bytes = b"") -> Dict[str, str]:
        """Create authentication headers for outgoing request."""
        timestamp = str(int(time.time()))

        # Create signature: HMAC(service_name + timestamp + body_hash)
        body_hash = hashlib.sha256(body).hexdigest()
        message = f"{self.service_name}:{timestamp}:{body_hash}"
        signature = hmac.new(
            self._signing_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()

        return {
            "X-Service-Name": self.service_name,
            "X-Service-Timestamp": timestamp,
            "X-Service-Signature": signature,
        }

    @classmethod
    def verify_request(
        cls,
        service_name: str,
        timestamp: str,
        signature: str,
        body: bytes = b""
    ) -> bool:
        """Verify an incoming service request."""
        try:
            # Check timestamp freshness
            request_time = int(timestamp)
            current_time = int(time.time())
            if abs(current_time - request_time) > cls.SIGNATURE_VALIDITY_SECONDS:
                logger.warning(f"Request from {service_name} has expired timestamp")
                return False

            # Verify signature
            expected_key = get_service_api_key(service_name)
            body_hash = hashlib.sha256(body).hexdigest()
            message = f"{service_name}:{timestamp}:{body_hash}"
            expected_signature = hmac.new(
                expected_key.encode(),
                message.encode(),
                hashlib.sha256
            ).hexdigest()

            if not hmac.compare_digest(signature, expected_signature):
                logger.warning(f"Invalid signature from {service_name}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error verifying service request: {e}")
            return False


# =============================================================================
# Input Validation
# =============================================================================

class InputValidator:
    """Production-grade input validation."""

    # Maximum sizes
    MAX_EVENT_SIZE_BYTES = 1024 * 1024  # 1MB per event
    MAX_BATCH_SIZE = 1000  # Events per batch
    MAX_STRING_LENGTH = 10000

    # Allowed characters for IDs
    ID_PATTERN = r'^[a-zA-Z0-9_-]+$'

    @classmethod
    def validate_tenant_id(cls, tenant_id: str) -> bool:
        """Validate tenant ID format."""
        import re
        if not tenant_id or len(tenant_id) > 64:
            return False
        return bool(re.match(cls.ID_PATTERN, tenant_id))

    @classmethod
    def validate_event_batch(cls, events: list, tenant_id: str) -> tuple[bool, str]:
        """
        Validate an event batch.

        Returns:
            (is_valid, error_message)
        """
        if not isinstance(events, list):
            return False, "Events must be a list"

        if len(events) > cls.MAX_BATCH_SIZE:
            return False, f"Batch size {len(events)} exceeds maximum {cls.MAX_BATCH_SIZE}"

        if len(events) == 0:
            return False, "Event batch cannot be empty"

        import json
        total_size = 0

        for i, event in enumerate(events):
            if not isinstance(event, dict):
                return False, f"Event {i} must be a dictionary"

            # Check individual event size
            try:
                event_json = json.dumps(event)
                event_size = len(event_json.encode('utf-8'))
                if event_size > cls.MAX_EVENT_SIZE_BYTES:
                    return False, f"Event {i} size {event_size} exceeds maximum {cls.MAX_EVENT_SIZE_BYTES}"
                total_size += event_size
            except (TypeError, ValueError) as e:
                return False, f"Event {i} is not JSON serializable: {e}"

        # Check total batch size
        max_batch_bytes = cls.MAX_BATCH_SIZE * cls.MAX_EVENT_SIZE_BYTES
        if total_size > max_batch_bytes:
            return False, f"Total batch size {total_size} exceeds maximum"

        return True, ""

    @classmethod
    def sanitize_log_message(cls, message: str) -> str:
        """Remove potentially sensitive data from log messages."""
        import re

        # Patterns to redact
        patterns = [
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_REDACTED]'),
            (r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '[IP_REDACTED]'),
            (r'api[_-]?key["\']?\s*[:=]\s*["\']?[\w-]+', 'api_key=[REDACTED]'),
            (r'password["\']?\s*[:=]\s*["\']?[\w-]+', 'password=[REDACTED]'),
            (r'token["\']?\s*[:=]\s*["\']?[\w.-]+', 'token=[REDACTED]'),
        ]

        sanitized = message
        for pattern, replacement in patterns:
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)

        return sanitized


# =============================================================================
# API Key Hashing (for storage)
# =============================================================================

def hash_api_key(api_key: str) -> str:
    """Hash an API key for secure storage."""
    return hashlib.sha256(api_key.encode()).hexdigest()


def verify_api_key(api_key: str, stored_hash: str) -> bool:
    """Verify an API key against its stored hash."""
    return hmac.compare_digest(hash_api_key(api_key), stored_hash)
