# Feedback Collection System - Production CLA

## Overview
This document explains how analyst feedback is collected, stored, processed, and used to update Production CLA accuracy.

## Architecture
```
Analyst / Agent → Feedback Ingestion → BigQuery (soc_data.feedback) → CLA Poller / API →
  - Compare to soc_data.processed_alerts → Update in-memory counters → Mark processed
```

## Data Stores
- BigQuery tables:
  - `soc_data.feedback`
    - `alert_id STRING`
    - `is_true_positive BOOLEAN`
    - `analyst_comments STRING`
    - `confidence FLOAT`
    - `timestamp TIMESTAMP`
    - `processed BOOLEAN`
  - `soc_data.processed_alerts`
    - `alert_id STRING`
    - `classification STRING` ("anomaly" | "benign")
    - `timestamp TIMESTAMP`
    - other alert fields

## Service Components
- `production_cla_service.py`
  - Background poller: `poll_feedback_and_update_accuracy()` (every ~120s)
  - Manual trigger endpoint: `POST /feedback/poll`
  - Accuracy exposure: `GET /status`
  - Status fields: `predictions_count`, `correct_predictions/ predictions_count -> accuracy`

## Endpoints
- `GET /status` – service metrics and accuracy
- `GET /health` – quick health probe
- `POST /classify` – classify an alert
- `POST /retrain` – manual retrain
- `POST /feedback/poll` – force a feedback poll (batch up to 500)

## Real-time Options
- Keep poller (pull model): reduce interval to 15–30s
- Add push path (recommended): publish feedback via Pub/Sub to CLA and update accuracy immediately
- Add synchronous ingestion: `POST /feedback/ingest` – write to BigQuery and update counters in one call

## Operations
- Check pending feedback:
```bash
bq query --use_legacy_sql=false \
'SELECT COUNT(*) FROM `chronicle-dev-2be9.soc_data.feedback` WHERE processed IS FALSE OR processed IS NULL'
```
- Trigger manual poll:
```bash
curl -X POST http://<CLA_HOST>:8080/feedback/poll
```
- Inspect latest feedback:
```bash
bq query --use_legacy_sql=false \
'SELECT * FROM `chronicle-dev-2be9.soc_data.feedback` ORDER BY timestamp DESC LIMIT 10'
```

## How Accuracy is Computed
- For each unprocessed feedback row:
  1) Fetch latest `classification` from `soc_data.processed_alerts` for the same `alert_id`
  2) Determine correctness:
     - `is_true_positive = true` → correct if `classification = 'anomaly'`
     - `is_true_positive = false` → correct if `classification != 'anomaly'`
  3) If correct → increment `correct_predictions`
  4) Mark feedback row `processed = true`
- Accuracy returned via `/status`:
```
accuracy = correct_predictions / max(predictions_count, 1)
```

## Troubleshooting
- Accuracy stays 0.0:
  - No predictions since last restart (`predictions_count = 0`)
  - No unprocessed feedback rows (all `processed = true`)
  - `processed_alerts` missing entries for those `alert_id`s
- Fixes:
  - Generate classifications (POST /classify or wait for traffic)
  - Ensure feedback producers insert with `processed = false`
  - Verify alert IDs match between feedback and processed_alerts

## Next Enhancements
- Implement `POST /feedback/ingest` for immediate updates
- Pub/Sub-based feedback stream
- Persist rolling accuracy metrics to BigQuery for dashboards
