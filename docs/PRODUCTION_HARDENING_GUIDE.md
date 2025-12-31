# GATRA SOC Platform - Production Hardening Guide

## Executive Summary

This document describes the production hardening implementation for the GATRA (Generalized Anomaly Detection & Triage Response Architecture) SOC Platform. The hardening effort transforms the development-ready codebase into a production-grade, multi-tenant security operations platform suitable for enterprise deployment.

**Version:** 1.0
**Last Updated:** December 2024
**Classification:** Internal Technical Documentation

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Security Hardening](#3-security-hardening)
4. [Tenant Isolation](#4-tenant-isolation)
5. [Observability](#5-observability)
6. [Resilience Patterns](#6-resilience-patterns)
7. [Deployment Guide](#7-deployment-guide)
8. [Configuration Reference](#8-configuration-reference)
9. [Operations Runbook](#9-operations-runbook)
10. [Security Checklist](#10-security-checklist)

---

## 1. Overview

### 1.1 Purpose

The production hardening initiative addresses critical gaps identified in the initial production readiness assessment, which scored the system at 33/100. After hardening, the system achieves enterprise-grade standards for:

- **Security**: Proper secret management, service authentication, and input validation
- **Multi-tenancy**: Complete tenant isolation with context propagation
- **Observability**: Structured logging, metrics, and comprehensive health checks
- **Resilience**: Circuit breakers, retry logic, and graceful shutdown

### 1.2 Scope

The following components were hardened:

| Component | File | Description |
|-----------|------|-------------|
| MSSP Platform | `mssp_platform_server_production.py` | Main orchestration service |
| TAA Service | `taa_service_production.py` | Triage & Analysis Agent |
| CRA Service | `cra_service_production.py` | Containment & Response Agent |
| Production Module | `production/` | Shared hardening utilities |

### 1.3 Key Improvements

| Area | Before | After |
|------|--------|-------|
| Secret Management | Hardcoded JWT secrets | GCP Secret Manager + secure env vars |
| Service Auth | None | HMAC-based service-to-service authentication |
| Rate Limiting | Configured but not enforced | Token bucket with Redis backend |
| Logging | Basic text logs | Structured JSON with correlation IDs |
| Health Checks | Simple `/health` endpoint | Liveness + Readiness with dependency checks |
| Resilience | None | Circuit breakers, retry, graceful shutdown |
| Tenant Isolation | Partial | Full context propagation across services |

---

## 2. Architecture

### 2.1 Production Architecture Diagram

```
                                    ┌─────────────────────────────────────┐
                                    │         Load Balancer               │
                                    │    (TLS Termination, WAF)           │
                                    └──────────────┬──────────────────────┘
                                                   │
                    ┌──────────────────────────────┼──────────────────────────────┐
                    │                              │                              │
                    ▼                              ▼                              ▼
        ┌───────────────────┐      ┌───────────────────┐      ┌───────────────────┐
        │   MSSP Platform   │      │   MSSP Platform   │      │   MSSP Platform   │
        │   (Instance 1)    │      │   (Instance 2)    │      │   (Instance N)    │
        └─────────┬─────────┘      └─────────┬─────────┘      └─────────┬─────────┘
                  │                          │                          │
                  └──────────────────────────┼──────────────────────────┘
                                             │
                    ┌────────────────────────┼────────────────────────┐
                    │                        │                        │
                    ▼                        ▼                        ▼
        ┌───────────────────┐    ┌───────────────────┐    ┌───────────────────┐
        │    TAA Service    │    │    CRA Service    │    │ Learning Service  │
        │  (Triage Agent)   │    │ (Response Agent)  │    │   (Feedback)      │
        └─────────┬─────────┘    └─────────┬─────────┘    └─────────┬─────────┘
                  │                        │                        │
                  └────────────────────────┼────────────────────────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    │                      │                      │
                    ▼                      ▼                      ▼
        ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐
        │      Redis        │  │     BigQuery      │  │   GCP Secret      │
        │  (Rate Limiting)  │  │   (Data Store)    │  │     Manager       │
        └───────────────────┘  └───────────────────┘  └───────────────────┘
```

### 2.2 Production Module Structure

```
production/
├── __init__.py           # Module exports
├── security.py           # Secret management, service auth, validation
├── rate_limiter.py       # Token bucket rate limiting
├── observability.py      # Logging, metrics, health checks
├── resilience.py         # Circuit breakers, retry, shutdown
└── tenant_context.py     # Tenant context propagation
```

---

## 3. Security Hardening

### 3.1 Secret Management

The `SecretManager` class provides a secure, hierarchical approach to secret retrieval:

```python
from production import SecretManager, secret_manager

# Automatic priority: GCP Secret Manager > Environment Variables > File
jwt_secret = secret_manager.get_secret("JWT_SECRET")
```

**Priority Order:**
1. **GCP Secret Manager** (if `GCP_SECRET_MANAGER_PROJECT` is set)
2. **Environment Variables** (for container deployments)
3. **File-based secrets** (for development)

**Configuration:**
```bash
# For GCP Secret Manager
export GCP_SECRET_MANAGER_PROJECT=your-project-id

# For environment-based secrets
export JWT_SECRET=$(openssl rand -hex 32)
export SERVICE_API_KEY=$(openssl rand -hex 32)
```

### 3.2 Service-to-Service Authentication

All internal service calls are authenticated using HMAC-SHA256 signatures:

```python
from production import ServiceAuthenticator

# Create authenticator for a service
auth = ServiceAuthenticator(service_name="taa-service")

# Generate auth headers for outgoing request
headers = auth.create_auth_headers(request_body)

# Verify incoming request
is_valid = ServiceAuthenticator.verify_request(
    service_name="mssp-platform",
    timestamp=request.headers["X-Service-Timestamp"],
    signature=request.headers["X-Service-Signature"],
    body=request.body
)
```

**Security Features:**
- HMAC-SHA256 signatures prevent tampering
- Timestamp validation prevents replay attacks (5-minute window)
- Service identification for audit trails

### 3.3 Input Validation

The `InputValidator` class provides comprehensive input sanitization:

```python
from production import InputValidator

validator = InputValidator()

# Validate tenant ID (alphanumeric + hyphens, 3-64 chars)
tenant_id = validator.validate_tenant_id(user_input)

# Validate alert ID (UUID format)
alert_id = validator.validate_alert_id(user_input)

# Validate IP address
ip = validator.validate_ip_address(user_input)

# Sanitize free-text input
safe_text = validator.sanitize_string(user_input, max_length=1000)
```

### 3.4 API Key Hashing

API keys are stored as salted SHA-256 hashes:

```python
from production import hash_api_key, verify_api_key

# Hash an API key for storage
hashed = hash_api_key(plain_api_key)

# Verify a provided key against stored hash
is_valid = verify_api_key(provided_key, stored_hash)
```

---

## 4. Tenant Isolation

### 4.1 Tenant Context Propagation

Every request carries tenant context through the entire processing pipeline:

```python
from production import TenantContext, set_current_tenant, get_current_tenant_id

# Set tenant context at request entry
context = TenantContext(
    tenant_id="acme-corp",
    tenant_config=config,
    correlation_id="req-abc123"
)
set_current_tenant(context)

# Access anywhere in the call stack
tenant_id = get_current_tenant_id()  # Returns "acme-corp"
```

### 4.2 Context Middleware

FastAPI middleware automatically extracts and propagates tenant context:

```python
from production import TenantContextMiddleware

app = FastAPI()
app.add_middleware(TenantContextMiddleware)

# Tenant context is now available in all routes
@app.get("/alerts")
@require_tenant_context
async def get_alerts():
    tenant_id = get_current_tenant_id()
    # Process request for specific tenant
```

### 4.3 Tenant-Scoped Clients

All external service calls automatically include tenant context:

```python
from production import TenantAwareClient, TenantScopedBigQueryClient

# HTTP client with automatic tenant header injection
async with TenantAwareClient() as client:
    response = await client.post(taa_url, json=alert_data)
    # X-Tenant-ID header automatically added

# BigQuery client scoped to tenant dataset
bq_client = TenantScopedBigQueryClient()
table_ref = bq_client.get_scoped_table("alerts")
# Returns: project.tenant_dataset.alerts
```

### 4.4 Tenant Configuration

Each tenant has isolated configuration:

```json
{
  "tenant_id": "acme-corp",
  "tenant_name": "Acme Corporation",
  "bigquery_dataset": "gatra_acme_corp",
  "pubsub_topic": "gatra-alerts-acme-corp",
  "rate_limits": {
    "ingest_eps": 500,
    "alerts_per_min": 100
  },
  "features": {
    "advanced_ml": true,
    "auto_containment": false
  }
}
```

---

## 5. Observability

### 5.1 Structured Logging

All logs are emitted as JSON for easy parsing by log aggregators:

```python
from production import configure_structured_logging, StructuredLogger

# Configure at startup
configure_structured_logging(service_name="mssp-platform", log_level="INFO")

# Use structured logger
logger = StructuredLogger("mssp-platform")
logger.info("Processing alert", alert_id=alert_id, severity="high")
```

**Log Output:**
```json
{
  "timestamp": "2024-12-27T10:30:45.123Z",
  "level": "INFO",
  "service": "mssp-platform",
  "correlation_id": "req-abc123",
  "tenant_id": "acme-corp",
  "message": "Processing alert",
  "alert_id": "alert-xyz",
  "severity": "high"
}
```

### 5.2 Correlation IDs

Every request is assigned a unique correlation ID that flows through all services:

```python
from production import generate_correlation_id, set_correlation_id, get_correlation_id

# Generate at request entry
correlation_id = generate_correlation_id()
set_correlation_id(correlation_id)

# Retrieve anywhere in the call stack
current_id = get_correlation_id()

# Automatically included in:
# - All log messages
# - Outgoing HTTP headers (X-Correlation-ID)
# - Error responses
```

### 5.3 Metrics Collection

Prometheus-compatible metrics for monitoring:

```python
from production import metrics, timed_operation

# Record custom metrics
metrics.increment("alerts_processed", tenant_id="acme-corp")
metrics.gauge("active_connections", 42)
metrics.histogram("response_time", 0.125)

# Time operations automatically
@timed_operation("process_alert")
async def process_alert(alert):
    # Execution time automatically recorded
    pass
```

**Available Metrics:**
| Metric | Type | Description |
|--------|------|-------------|
| `requests_total` | Counter | Total requests by endpoint and status |
| `request_duration_seconds` | Histogram | Request latency distribution |
| `rate_limit_exceeded` | Counter | Rate limit violations by tenant |
| `circuit_breaker_state` | Gauge | Circuit breaker status (0=closed, 1=open) |
| `active_requests` | Gauge | Current in-flight requests |

### 5.4 Health Checks

Kubernetes-compatible liveness and readiness probes:

```python
from production import HealthChecker

health = HealthChecker(service_name="mssp-platform")

# Register dependency checks
health.register_check("redis", check_redis_connection)
health.register_check("bigquery", check_bigquery_connection)
health.register_check("taa_service", check_taa_health)

# Liveness: Is the process alive?
@app.get("/health/live")
async def liveness():
    result = await health.check_liveness()
    return {"status": result.status}

# Readiness: Can we serve traffic?
@app.get("/health/ready")
async def readiness():
    is_ready, checks = await health.check_readiness()
    return {
        "ready": is_ready,
        "checks": [{"name": c.name, "status": c.status} for c in checks]
    }
```

**Health Check Response:**
```json
{
  "ready": true,
  "checks": [
    {"name": "redis", "status": "healthy", "latency_ms": 2},
    {"name": "bigquery", "status": "healthy", "latency_ms": 45},
    {"name": "taa_service", "status": "healthy", "latency_ms": 12}
  ]
}
```

---

## 6. Resilience Patterns

### 6.1 Circuit Breaker

Prevents cascade failures when downstream services are unhealthy:

```python
from production import CircuitBreaker, CircuitBreakerConfig, get_circuit_breaker

# Configure circuit breaker
config = CircuitBreakerConfig(
    failure_threshold=5,      # Open after 5 failures
    recovery_timeout=30.0,    # Try again after 30 seconds
    half_open_requests=3      # Allow 3 test requests in half-open state
)

breaker = get_circuit_breaker("taa-service", config)

# Use in requests
if await breaker.can_proceed():
    try:
        response = await call_taa_service()
        await breaker.record_success()
    except Exception:
        await breaker.record_failure()
        raise
else:
    raise CircuitBreakerError("TAA service circuit is open")
```

**Circuit States:**
| State | Description | Behavior |
|-------|-------------|----------|
| CLOSED | Normal operation | All requests allowed |
| OPEN | Failures exceeded threshold | All requests rejected |
| HALF_OPEN | Testing recovery | Limited requests allowed |

### 6.2 Retry Logic

Automatic retry with exponential backoff:

```python
from production import retry_async, RetryConfig

config = RetryConfig(
    max_retries=3,
    base_delay=1.0,
    max_delay=30.0,
    exponential_base=2,
    retryable_exceptions=(ConnectionError, TimeoutError)
)

@retry_async(config)
async def call_external_service():
    # Will retry up to 3 times with delays: 1s, 2s, 4s
    return await http_client.get(url)
```

### 6.3 Graceful Shutdown

Ensures in-flight requests complete before shutdown:

```python
from production import GracefulShutdown, setup_signal_handlers

shutdown = GracefulShutdown(grace_period=30.0)

# Register cleanup tasks
shutdown.register_cleanup(close_database_connections)
shutdown.register_cleanup(flush_metrics)

# Setup signal handlers (SIGTERM, SIGINT)
setup_signal_handlers(shutdown)

# Track active requests
async def handle_request():
    async with shutdown.track_request():
        # Request processing
        pass

# On shutdown signal:
# 1. Stop accepting new requests
# 2. Wait for in-flight requests (up to 30s)
# 3. Run cleanup tasks
# 4. Exit
```

### 6.4 Bulkhead Pattern

Isolates resources to prevent one component from exhausting shared resources:

```python
from production import Bulkhead, get_bulkhead

# Create bulkhead with max 10 concurrent operations
bulkhead = get_bulkhead("bigquery-queries", max_concurrent=10)

async def query_bigquery():
    async with bulkhead.acquire():
        # Only 10 concurrent queries allowed
        return await bq_client.query(sql)
```

### 6.5 Timeout Handling

Configurable timeouts for all operations:

```python
from production import timeout, TimeoutError

@timeout(seconds=30)
async def process_with_timeout():
    # Raises TimeoutError if exceeds 30 seconds
    return await long_running_operation()
```

---

## 7. Deployment Guide

### 7.1 Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- GCP Service Account with:
  - BigQuery Data Editor
  - Secret Manager Secret Accessor (optional)
- Redis 7.0+ (included in docker-compose)

### 7.2 Quick Start

```bash
# 1. Clone and navigate to project
cd ai-driven-soc.backup

# 2. Create production environment file
cp .env.production.template .env.production

# 3. Generate secure secrets
export JWT_SECRET=$(openssl rand -hex 32)
export SERVICE_API_KEY=$(openssl rand -hex 32)

# 4. Edit .env.production with your values
nano .env.production

# 5. Deploy with Docker Compose
docker-compose -f docker-compose.production.yml \
  --env-file .env.production up -d

# 6. Verify deployment
curl http://localhost:8080/health/ready
```

### 7.3 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `JWT_SECRET` | Yes | JWT signing secret (min 32 chars) |
| `SERVICE_API_KEY` | Yes | Service-to-service auth key |
| `GCP_PROJECT_ID` | No | GCP project for Secret Manager |
| `REDIS_URL` | Yes | Redis connection URL |
| `LOG_LEVEL` | No | Logging level (default: INFO) |
| `ENVIRONMENT` | No | Environment name (default: production) |

### 7.4 Kubernetes Deployment

For Kubernetes deployments, create secrets and ConfigMaps:

```yaml
# secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: gatra-secrets
type: Opaque
stringData:
  jwt-secret: "${JWT_SECRET}"
  service-api-key: "${SERVICE_API_KEY}"

---
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: gatra-config
data:
  LOG_LEVEL: "INFO"
  ENVIRONMENT: "production"
  REDIS_URL: "redis://redis-service:6379"
```

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mssp-platform
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: mssp-platform
        image: gatra-mssp:latest
        ports:
        - containerPort: 8080
        env:
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: gatra-secrets
              key: jwt-secret
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 15
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
```

---

## 8. Configuration Reference

### 8.1 Rate Limiting Configuration

Per-tenant rate limits in `gatra_multitenant_config.json`:

```json
{
  "tenants": [
    {
      "tenant_id": "acme-corp",
      "rate_limits": {
        "ingest_eps": 500,
        "alerts_per_min": 100
      }
    }
  ]
}
```

### 8.2 Circuit Breaker Configuration

Environment variables:

```bash
CIRCUIT_BREAKER_FAILURE_THRESHOLD=5    # Failures before opening
CIRCUIT_BREAKER_RECOVERY_TIMEOUT=30    # Seconds before half-open
```

### 8.3 Logging Configuration

```bash
LOG_LEVEL=INFO           # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT=json          # json or text
```

---

## 9. Operations Runbook

### 9.1 Monitoring Alerts

| Alert | Threshold | Action |
|-------|-----------|--------|
| High Error Rate | >5% 5xx responses | Check service logs, circuit breaker state |
| Rate Limit Exceeded | >10 per minute | Review tenant limits, check for abuse |
| Circuit Breaker Open | Any | Check downstream service health |
| High Latency | p99 > 5s | Scale services, optimize queries |

### 9.2 Common Issues

**Issue: Rate limit errors for valid traffic**
```bash
# Check current rate limit status
curl http://localhost:8080/api/v1/rate-limit/status?tenant_id=acme-corp

# Increase tenant limits in config
# Edit config/gatra_multitenant_config.json
```

**Issue: Circuit breaker stuck open**
```bash
# Check circuit breaker state
curl http://localhost:8080/health/ready

# Force reset (if safe)
curl -X POST http://localhost:8080/admin/circuit-breaker/reset?service=taa
```

**Issue: Tenant context not propagating**
```bash
# Check logs for missing tenant context
grep "tenant_id" /app/logs/mssp-platform.log | head -20

# Ensure X-Tenant-ID header is being sent
curl -H "X-Tenant-ID: acme-corp" http://localhost:8080/api/v1/alerts
```

### 9.3 Scaling Guidelines

| Service | CPU Threshold | Memory Threshold | Recommended Action |
|---------|---------------|------------------|-------------------|
| MSSP Platform | 70% | 80% | Add replicas (horizontal) |
| TAA Service | 80% | 70% | Add replicas + increase CPU |
| CRA Service | 60% | 60% | Add replicas |
| Redis | 70% | 80% | Upgrade instance size |

---

## 10. Security Checklist

### Pre-Deployment

- [ ] Generate unique JWT_SECRET (min 32 chars)
- [ ] Generate unique SERVICE_API_KEY
- [ ] Configure GCP Secret Manager (recommended)
- [ ] Review tenant rate limits
- [ ] Enable TLS at load balancer
- [ ] Configure WAF rules
- [ ] Review service account permissions

### Post-Deployment

- [ ] Verify health endpoints respond correctly
- [ ] Test rate limiting is enforced
- [ ] Verify tenant isolation (cross-tenant access blocked)
- [ ] Test circuit breaker behavior
- [ ] Verify structured logs are collected
- [ ] Test graceful shutdown behavior
- [ ] Run security scan on exposed endpoints

### Ongoing

- [ ] Rotate secrets quarterly
- [ ] Review access logs weekly
- [ ] Monitor rate limit violations
- [ ] Review circuit breaker events
- [ ] Update dependencies monthly

---

## Appendix A: API Reference

### Health Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health/live` | GET | Liveness probe |
| `/health/ready` | GET | Readiness probe with dependency checks |
| `/health` | GET | Basic health status |

### Admin Endpoints (Internal Only)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/admin/circuit-breaker/status` | GET | View circuit breaker states |
| `/admin/rate-limit/status` | GET | View rate limit status |
| `/metrics` | GET | Prometheus metrics |

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| GATRA | Generalized Anomaly Detection & Triage Response Architecture |
| TAA | Triage & Analysis Agent |
| CRA | Containment & Response Agent |
| ADA | Anomaly Detection Agent |
| CLA | Continuous Learning Agent |
| MSSP | Managed Security Service Provider |
| EPS | Events Per Second |
| Circuit Breaker | Pattern to prevent cascade failures |
| Bulkhead | Pattern to isolate resource consumption |

---

*Document generated for GATRA SOC Platform v1.0*
