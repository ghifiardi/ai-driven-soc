# BigQuery Dataset Overview – `chronicle-dev-2be9:gatra_database`

## Snapshot
- **Location:** `asia-southeast2`
- **Tables (total):** 40 tables + 4 views
- **Query snapshot:** `SELECT table_id, row_count, size_bytes FROM \`chronicle-dev-2be9.gatra_database.__TABLES__\`` (run 2025-11-11)
- **Purpose:** Central repository for SOC agents (ADA, TAA, CLA, CRA) including alert pipelines, feedback, telemetry, and operational logs.

## Status Summary
| Category | Tables (row_count > 0) | Empty/Views (row_count = 0) |
| --- | --- | --- |
| **Alert/Detection** | `processed_alerts (2,900)`, `processed_ada_alerts (67,519)`, `dashboard_alerts (1,000)`, `taa_comparison (12,440)`, `ground_truth_alerts (10)` | `dashboard_alerts_fixed` (view), `dashboard_alerts_table` (view) |
| **Agent State** | `ada_state (101,024)`, `ada_state_backup (103)`, `agent_state (92,490)`, `agent_state_backup (38)`, `cra_state (36,333)`, `cra_state_ext (94)`, `taa_state (92,086)`, `taa_state_backup (291)` | `workflow_state` |
| **Feedback & Metrics** | `feedback (66,784)`, `ada_feedback (17)`, `gatra_users (17)`, `model_performance (3)`, `rl_agent_logs (35)` | `agent_metrics`, `taa_feedback_training`, `taa_learning_log`, `taa_metrics (22 rows)**`, `taa_model_versions`, `taa_performance_trend` |
| **Telemetry & Ops** | `activity_logs (285)`, `ip_country_cache (3,772)`, `vm_resource_metrics (43,132)`, `AlarmsToDelete (4,108)` | `system_events`, `incidents`, `cra_plan_queue` |
| **SIEM Ingest** | `siem_alarms (652,536)`, `siem_events (1,254,376)`, `siem_events_results (2,265,874)` | — |
| **Raw / Staging** | — | `raw_events`, `ada_features`, `ada_ml_results`, `model_metadata`, `threat_intel` |

> **Note:** Views appear with row count 0 by design; actual rows depend on underlying tables.

## Table Highlights
- **`processed_ada_alerts`** – Main enriched output for ADA (67k rows, ~2.6 MB); complements the CLA dataset.
- **`processed_alerts`** – CLA baseline copied from production (`soc_data.processed_alerts`, 2.9k rows) and serialized snapshot at `data/rft/baseline_20251111/processed_alerts.csv`.
- **`siem_*` tables** – High-volume ingestion (millions of rows) from SIEM sources feeding upstream analytics.
- **`activity_logs`** – Partitioned by `event_timestamp`, clustered by `user`, `action`; 285 recent UI interactions.
- **`ip_country_cache`** – Partitioned cache of IP → geo mapping with 90-day TTL.
- **`taa_state` / `ada_state` / `agent_state`** – Large state stores; ensure retention/archival strategy prior to RFT telemetry expansion.
- **`feedback`** – Production analyst/auto-feedback snapshot (66,784 rows) mirrored from `soc_data.feedback`; CSV export stored at `data/rft/baseline_20251111/feedback.csv`.

## Partitioning & Clustering
- **Partitioned:** `activity_logs`, `ip_country_cache`, `taa_feedback_training`, `taa_learning_log`, `taa_metrics`.
- **Clustered:** `activity_logs` (`user`, `action`), `ip_country_cache` (`ip_address`).
- **Non-partitioned high-volume:** `siem_events`, `siem_events_results`, `siem_alarms`, `ada_state`, `taa_state`. Consider partitioning if query costs spike.

## Relationships & Pipelines
1. **Ingestion:** `siem_events` → processed via ADA/TAA pipelines → results stored in `processed_ada_alerts`, future `processed_alerts`.
2. **Stateful Agents:** `ada_state`, `taa_state`, `agent_state`, `cra_state` capture workflow checkpoints; backups maintained separately.
3. **Feedback Loop:** Analyst feedback targets `feedback` (CLA), `ada_feedback`, `taa_feedback_training`; currently sparse/empty and should be prioritized for RFT reward signals.
4. **Monitoring:** `activity_logs`, `vm_resource_metrics`, `rl_agent_logs`, `model_performance` supply operational observability and can feed RFT evaluation dashboards.

## Action Items Before RFT
- Maintain refreshed snapshots (`processed_alerts`, `feedback`) as new data arrives; re-export to `data/rft/` with versioned folders.
- Validate schema consistency across state tables; document retention/archival plans.
- Define access controls (IAM + authorized views) for high-sensitivity tables (`siem_*`, `raw_events`).
- Version dataset exports under `data/rft/` for reproducible baselines.

## Appendix – Full Row Count Listing
```
AlarmsToDelete: 4,108
activity_logs: 285
ada_features: 0
ada_feedback: 17
ada_ml_results: 0
ada_state: 101,024
ada_state_backup: 103
agent_metrics: 0
agent_state: 92,490
agent_state_backup: 38
alerts_dashboard_view (VIEW): 0
cra_plan_queue: 0
cra_state: 36,333
cra_state_ext: 94
dashboard_alerts: 1,000
dashboard_alerts_fixed (VIEW): 0
dashboard_alerts_table (VIEW): 0
feedback: 66,784
gatra_users: 17
ground_truth_alerts: 10
incidents: 0
ip_country_cache: 3,772
model_metadata: 0
model_performance: 3
processed_ada_alerts: 67,519
processed_alerts: 2,900
raw_events: 0
rl_agent_logs: 35
siem_alarms: 652,536
siem_events: 1,254,376
siem_events_results: 2,265,874
system_events: 0
taa_comparison: 12,440
taa_feedback_training: 0
taa_learning_log: 0
taa_metrics: 22
taa_model_versions: 0
taa_performance_trend (VIEW): 0
taa_state: 92,086
taa_state_backup: 291
threat_intel: 0
vm_resource_metrics: 43,132
workflow_state: 0
```

