# MSSP Platform Integration Guide

## Overview

The AI-Driven SOC MSSP Platform allows Managed Security Service Providers to integrate their existing systems with our advanced AI detection and response capabilities. This platform exposes a unified interface for:

1.  **Dynamic Tenant Onboarding**: Programmatically register new customers/tenants.
2.  **Event Ingestion**: Push security logs and events via a standardized REST API.
3.  **AI Agent Integration**: Connect external AI agents using the Model Context Protocol (MCP).

## Getting Started

### Prerequisites

- Python 3.11+
- Access to the `ai-driven-soc` deployment.
- A valid API key (authentication is currently handled via network security/IAM in this version).

### Running the Server

The MSSP Platform Server runs on port **8081** by default to avoid conflicts with other services.

```bash
python3 mssp_platform_server.py
```

## REST API Reference

### 1. List Tenants

**Endpoint**: `GET /api/v1/tenants`

**Response**:
```json
{
  "tenants": [
    {
      "tenant_id": "tenant_001",
      "display_name": "Pilot Bank",
      "region": "us-central1",
      "service_level": "professional"
    }
  ]
}
```

### 2. Register Tenant

**Endpoint**: `POST /api/v1/tenants`

**Request Body**:
```json
{
  "tenant_id": "new_client_x",
  "display_name": "New Client X Corp",
  "region": "us-central1",
  "service_level": "enterprise"
}
```

**Response**:
```json
{
  "status": "created",
  "tenant_id": "new_client_x"
}
```

### 3. Ingest Events

**Endpoint**: `POST /api/v1/events`

**Request Body**:
```json
{
  "tenant_id": "new_client_x",
  "events": [
    {
      "timestamp": "2025-11-23T10:00:00Z",
      "source_ip": "192.168.1.50",
      "event_type": "login_failed",
      "user": "admin"
    }
  ]
}
```

## MCP Integration

The platform exposes an MCP-compatible endpoint for AI agents to query tenant status.

**Tool**: `get_tenant_status`

**Parameters**:
- `tenant_id` (string): The ID of the tenant to query.

**Example Usage (Python MCP Client)**:

```python
result = await mcp_client.call_tool(
    "get_tenant_status",
    {"tenant_id": "new_client_x"}
)
print(result)
# Output: {'tenant_id': 'new_client_x', 'status': 'active', ...}
```

## Quantum Integration (Beta)

To enable the **Quantum Threat Detector**, you must configure the environment with your IBM Quantum API key.

### Prerequisites
1.  **IBM Quantum Account**: Get your API key from [quantum-computing.ibm.com](https://quantum-computing.ibm.com/).
2.  **Qiskit Runtime**: Install the required libraries.
    ```bash
    pip install qiskit-ibm-runtime qiskit-machine-learning
    ```

### Configuration
Set the following environment variable before running the platform server:

```bash
export IBM_QUANTUM_TOKEN="your_ibm_quantum_api_key"
export QUANTUM_BACKEND="ibmq_qasm_simulator" # or a real backend system
```

The platform will automatically detect the key and initialize the `QuantumThreatDetector` agent for "Enterprise" and "Vanguard" tenants.

