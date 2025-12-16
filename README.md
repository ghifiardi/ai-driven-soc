# AI-Driven SOC (GATRA-Aligned) Technical Overview

This repository contains the LangGraph-powered Anomaly Detection Agent (ADA) and supporting services that underpin an autonomous, multi-tenant Security Operations Center (SOC) aligned with the GATRA SaaS Technical Implementation Guide. The codebase now captures the architectural guardrails required to scale from single-tenant prototypes to a 50K events-per-second multi-tenant platform with ADA/TAA streaming, Pub/Sub orchestration, and GCP-native deployment paths.

---

## Table of Contents
1. [Platform Architecture](#platform-architecture)
2. [GATRA Blueprint Alignment](#gatra-blueprint-alignment)
3. [Configuration](#configuration)
   - [Environment Variables](#environment-variables)
   - [Multi-Tenant Registry (`config/gatra_multitenant_config.json`)](#multi-tenant-registry-configgatra_multitenant_configjson)
   - [Feature Flags](#feature-flags)
4. [End-to-End Data Flow](#end-to-end-data-flow)
5. [Deployment Modes](#deployment-modes)
   - [Mode A – Managed VM (current production)](#mode-a--managed-vm-current-production)
   - [Mode B – GKE & Terraform (preview scaffolding)](#mode-b--gke--terraform-preview-scaffolding)
6. [Monitoring & Observability](#monitoring--observability)
7. [Testing & Validation](#testing--validation)
8. [Roadmap & Next Steps](#roadmap--next-steps)
9. [References](#references)

---

## Platform Architecture

At the core of the stack are two collaborating agents orchestrated with LangGraph:

- `langgraph_ada_integration.py` – the primary ADA runtime responsible for ingesting raw events, normalising payloads, running anomaly detection (Isolation Forest + optional supervised overrides), and publishing enriched alerts downstream.
- `taa_langgraph_agent.py` – the Triage & Analysis Agent (TAA) that receives ADA alerts, enriches them with LLM-based context, makes containment/manual-review decisions, and publishes outputs to CRA, CLA, and reporting channels.

Supporting modules:

- `multi_tenant_manager.py` – multi-tenant configuration manager that resolves BigQuery datasets, Pub/Sub topics, and rate limits per tenant.
- `bigquery_client.py` – tenant-aware BigQuery interface that fetches new events and writes detection results, now parameterised by schema, region, and partition metadata.
- `docker-compose.yml` – local orchestration surface to run ADA/TAA services with an optional local LLM and enrichment workers.
- `deploy_ada.sh` – deployment helper supporting VM rollouts today and emitting Terraform/GKE scaffolding instructions for the upcoming managed cluster path.

This composition mirrors the GATRA guide: Kafka/Pub/Sub topics per tenant, ADA feeding TAA, and downstream SOAR connectors (CRA) that follow after ADA/TAA alignment.

---

## GATRA Blueprint Alignment

The repository implements the blueprint decisions captured in `../Autonomous Platform/GATRA_Technical_Implementation_Guide.md`:

| Blueprint Pillar | Implementation Touchpoints |
| --- | --- |
| Multi-tenant isolation (per-tenant schemas, Kafka topics) | `config/gatra_multitenant_config.json`, `multi_tenant_manager.py`, `BigQueryClient.for_tenant(...)` |
| Event throughput 50K eps | ADA polling loop (planned integration), partition-aware BigQuery access, future load-testing harness |
| ADA → TAA streaming pipeline | `langgraph_ada_integration.py` (producer) → `taa_langgraph_agent.py` (consumer), Pub/Sub topics defined in tenant registry |
| Deployment on GKE with Terraform | `deploy_ada.sh --mode gke` scaffolding, forthcoming Terraform modules referenced from GATRA guide |
| Monitoring via Prometheus/Grafana/GCP | Structured logging in ADA, planned Prometheus exporters, and GCP metrics instrumentation (see roadmap) |

---

## Configuration

### Environment Variables

| Variable | Required | Description |
| --- | --- | --- |
| `MULTITENANT_CONFIG_PATH` | ✅ | Filesystem path to the JSON registry (`config/gatra_multitenant_config.json` by default). |
| `DEFAULT_TENANT_ID` | ✅ | Fallback tenant when none is supplied; should match a `tenant_id` in the registry. |
| `REDIS_URL` | Optional | Enables Redis-backed caching/session management when supplied. Leave empty to disable. |
| `CONFIDENCE_THRESHOLD` | Optional | ADA anomaly confidence cutoff (default `0.8`). |
| `POLLING_INTERVAL` | Optional | ADA polling interval (seconds) between BigQuery fetches (default `30`). |
| `ENABLE_EMBEDDINGS` | Optional | Feature flag enabling embedding generation pipelines (default disabled). |
| `GCP_*` credential vars | ✅ runtime | Ensure the workload identity or service account used by ADA/TAA has BigQuery read/write, Pub/Sub publish/subscribe, and Vertex AI access where applicable. |

> Tip: When running inside Docker Compose, declare these in an `.env` file and reference them via the `environment:` sections.

### Multi-Tenant Registry (`config/gatra_multitenant_config.json`)

The registry is the single source of truth for tenant metadata. Example excerpt:

```json
{
  "defaults": {
    "project_id": "chronicle-dev-2be9",
    "location": "us-central1",
    "dataset_template": "gatra_{tenant_id}",
    "results_dataset_template": "gatra_{tenant_id}_results",
    "metrics_namespace": "gatra_multi_tenant"
  },
  "tenants": [
    {
      "tenant_id": "tenant_001",
      "display_name": "Pilot Bank - Professional Tier",
      "region": "us-central1",
      "dataset": "gatra_tenant_001",
      "results_dataset": "gatra_tenant_001_results",
      "tables": {
        "events": "events",
        "alerts": "alerts",
        "results": "events_results"
      },
      "pubsub_topics": {
        "ingest": "events-tenant_001",
        "alerts": "alerts-tenant_001",
        "priority": "priority-tenant_001"
      },
      "rate_limits": {
        "ingest_eps": 10000,
        "alerts_per_min": 500
      },
      "service_level": "professional"
    }
  ],
  "default_tenant_id": "tenant_001"
}
```

Validation rules enforced by `MultiTenantManager`:

- Every tenant must define `events`, `alerts`, and `results` table names.
- Pub/Sub topics (`ingest`, `alerts`, `priority`) are mandatory.
- Rate limits must be positive integers.
- `default_tenant_id` must exist in the tenants list.
- Dataset names must be unique across tenants (preventing accidental cross-tenant writes).

Runtime usage:

```python
from multi_tenant_manager import MultiTenantManager
from bigquery_client import BigQueryClient

manager = MultiTenantManager.from_file("config/gatra_multitenant_config.json")
ada_client = BigQueryClient.for_tenant(manager, tenant_id="tenant_001", partition_field="timestamp")
```

### Feature Flags

| Flag | Effect |
| --- | --- |
| `ENABLE_EMBEDDINGS=1` | Activates embedding generation and dual-publish (alerts + embeddings) pipelines. |
| `MULTITENANT_CONFIG_DISABLED=1` | (Emergency fallback) forces ADA to behave like a single-tenant deployment using legacy env vars. |

---

## End-to-End Data Flow

1. **Ingestion request** – Events arrive via `/api/v1/events` (FastAPI stub in the GATRA guide) or scheduled BigQuery polling executed by ADA.
2. **Tenant resolution** – `MultiTenantManager` resolves the tenant’s datasets and Pub/Sub topics.
3. **BigQuery fetch** – `BigQueryClient.fetch_new_alerts` pulls raw events from the tenant’s `events` table.
4. **Detection & enrichment** – `LangGraphAnomalyDetectionAgent` (ADA) runs anomaly detection, optional embeddings, and prepares enriched payloads.
5. **Publishing** – ADA emits alerts to `alerts-{tenant}` (plus embeddings streams when enabled).
6. **TAA workflow** – `taa_langgraph_agent.py` consumes alerts, performs enrichment, and routes to containment/manual-review/reporting nodes.
7. **Downstream actions** – CRA/SOAR executors (e.g., `cra_service.py`) receive containment requests; CLA and reporting agents digest TAA feedback for continuous learning.

---

## Deployment Modes

### Mode A – Managed VM (current production)

The existing production footprint runs ADA as a `systemd` service on a hardened VM.

1. Copy repository contents to `/home/app/ai-driven-soc/`.
2. Provision Python 3.11 virtual environment (`python3.11 -m venv venv`).
3. Install dependencies: `pip install -r requirements.txt`.
4. Export environment variables (see [Configuration](#configuration)).
5. Use `deploy_ada.sh --mode vm` to:
   - (Re)build the virtualenv if absent.
   - Publish an updated `/etc/systemd/system/ada.service`.
   - Reload and restart the service (`systemctl daemon-reload && systemctl restart ada`).
6. Monitor with `journalctl -u ada -f` or the service log at `/var/log/langgraph-ada/ada_workflow.log`.

### Mode B – GKE & Terraform (preview scaffolding)

The script now prints Terraform/GKE instructions when invoked with `deploy_ada.sh --mode gke`:

- Initialises Terraform workspaces targeting the topology described in the GATRA guide (GKE cluster, Cloud SQL, Memorystore, Pub/Sub topics).
- Emits TODO markers for secrets/credentials management (e.g., Google Secret Manager).
- Provides sample `kubectl` commands to deploy ADA/TAA workloads as microservices.

> Full automation is staged for a subsequent iteration; use this mode as a guided checklist while the Terraform modules are finalised.

---

## Monitoring & Observability

Short term (VM mode):

- Systemd telemetry: `systemctl status ada`, `journalctl -u ada -f`.
- ADA runtime log: `/var/log/langgraph-ada/ada_workflow.log`.
- TAA and enrichment workers emit structured logs to stdout (visible when using Docker Compose).

Planned enhancements aligned with GATRA (week 16+ in the guide):

- Prometheus exporters for ADA/TAA throughput, detection latency, and Pub/Sub queue depth.
- Grafana dashboards with the metrics outlined under the GATRA monitoring section.
- Google Cloud Trace integration for end-to-end request latency.

---

## Testing & Validation

Current harness:

- Unit tests (planned addition): `tests/test_multi_tenant_manager.py` to validate config parsing and guard against regressions.
- Manual smoke tests: `python3 - <<'PY' ...` snippet used in development to confirm tenant-loading.
- Dashboard/agent integration scripts under `test_*` directories for broader scenario coverage.

Upcoming (per roadmap):

- Locust load testing targeting 50K eps (mirroring `GATRA_Technical_Implementation_Guide.md` §Week 11-12).
- End-to-end ADA→TAA pipeline tests executed via pytest fixtures.
- Terraform plan/apply validation tests once the GKE stack is operational.

---

## Roadmap & Next Steps

1. **Integrate multi-tenant manager with ADA runtime (in progress)** – ensure incoming REST/webhook requests route the correct tenant metadata through LangGraph workflows.
2. **Publish Docker Compose profiles** – clarify optional services (local LLM, enrichment worker) and add `ada_worker` container.
3. **Formalise GKE deployment** – implement Terraform modules, KServe model deployments, and horizontal pod autoscaling aligned with the guide.
4. **Observability instrumentation** – add Prometheus metrics, Alertmanager rules, and logging integrations.
5. **Performance benchmarking** – automate Locust test suite and capture P99 latency, throughput, and resource utilisation metrics.

---

## References

- `../Autonomous Platform/GATRA_Technical_Implementation_Guide.md` – primary blueprint (architecture, timelines, benchmarks).
- `config/gatra_multitenant_config.json` – tenant registry governing datasets, topics, and rate limits.
- `multi_tenant_manager.py` and `bigquery_client.py` – reference implementations for tenant-aware data access.
- `deploy_ada.sh`, `docker-compose.yml` – operational surfaces for VM and container-based deployments respectively.

For historical fixes (IAM permissions, table schema corrections, etc.) refer to prior commit history or preserved notes in previous README revisions.

