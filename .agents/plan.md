# Goal & Non-Goals
- Goal: Define the implementation plan to adapt the Continuous Learning Agent (CLA) for Agent RFT based on `RL/agent RFT.md`.
- Non-Goals: Modify ADA/TAA/CRA agents, run actual RFT training jobs, or deploy infrastructure changes in this phase.
# Perubahan per File
- file: `production_cla_service.py`
  - Lokasi: [L47-L420], [L517-L704]
  - Perubahan: Instrument predictions/feedback logging for rollout traces; expose tool-call metadata; add hooks to emit telemetry to RFT pipeline.
  - Mengapa: Provide detailed traces and outcomes needed by tool server and reward grader (research §Peta file).
  - Dampak: Requires config toggles and BigQuery schema updates for new telemetry tables.
- file: `config/production_cla_config.json`
  - Lokasi: entire file (new keys)
  - Perubahan: Add RFT-specific settings (reward endpoint URL, tool server base URL, telemetry dataset/table, feature flags).
  - Mengapa: Centralize runtime control for RFT features; align with research identifying config dependency.
  - Dampak: Ensure defaults disabled to avoid affecting current production until RFT activated.
- file: `docs/gatra_bigquery_overview.md` (baru)
  - Lokasi: n/a
  - Perubahan: Dokumentasikan struktur dataset BigQuery (schema ringkas, status row_count aktif vs kosong, relasi utama).
  - Mengapa: Mendukung Step 1 data audit dengan referensi arsitektur tingkat tinggi.
  - Dampak: Menjadi panduan operator saat menyiapkan baseline & telemetry.
- file: `deploy/` (new module e.g., `rft_tool_server.py`, `rft_grader_endpoint.py`)
  - Lokasi: new files
  - Perubahan: Implement FastAPI service exposing CLA tool calls and reward grader per RFT doc.
  - Mengapa: `RL/agent RFT.md` mandates externally accessible tool/grader endpoints.
  - Dampak: Requires deployment scripts and authentication handling.
- file: `analytics/cla_rft_baseline.ipynb` (new) or script
  - Lokasi: new artifact
  - Perubahan: Create notebook/script computing baseline metrics and preparing train/eval datasets.
  - Mengapa: Establish baseline and dataset readiness per research steps.
  - Dampak: Generates dataset artifacts stored under `data/rft/`.
- file: `docs/cla_rft_playbook.md` (new)
  - Lokasi: new doc
  - Perubahan: Document rollout procedure, reward rubric, monitoring strategy.
  - Mengapa: Capture plan + guardrails for operators.
# Urutan Eksekusi (Step 1..n + "uji cepat" per step)
- Step 1: Data Audit & Baseline  
  - Actions: Query BigQuery `processed_alerts`/`feedback` to export representative train/eval sets; compute baseline metrics; store results under `data/rft/`; buat dokumentasi `docs/gatra_bigquery_overview.md` meringkas schema & status tabel.  
  - Uji cepat: Notebook/script outputs summary (precision/recall, volume) tanpa error; dokumen baru menjelaskan tabel aktif vs kosong.
- Step 2: Telemetry Design & Schema Updates  
  - Actions: Design new `cla_rft_rollouts` table schema; update config defaults; draft migration SQL.  
  - Uji cepat: `bq show` confirms new table after migration.
- Step 3: Tool Server Implementation  
  - Actions: Build FastAPI app exposing CLA tool functions (e.g., BigQuery fetch, feedback submission); enforce auth/latency constraints.  
  - Uji cepat: Local `pytest`/curl hits endpoints returning 200 with mocked responses.
- Step 4: Reward Grader Endpoint  
  - Actions: Implement endpoint evaluating CLA decisions with partial credit rubric; integrate with dataset labels.  
  - Uji cepat: Unit tests produce expected reward scores for sample rollouts.
- Step 5: CLA Instrumentation  
  - Actions: Modify `production_cla_service.py` to log tool interactions, send rollout payloads to tool server, and call grader as needed; guard behind feature flag.  
  - Uji cepat: Dry-run mode logs telemetry locally; feature flag off leaves behavior unchanged.
- Step 6: Training Orchestration Scripts  
  - Actions: Create CLI or workflow to package rollouts, trigger RFT API jobs, and collect metrics.  
  - Uji cepat: Script runs end-to-end in sandbox with mocked RFT API.
# Acceptance Criteria (incl. edge-cases)
- Baseline dataset and metrics documented; reproducible scripts committed.
- Tool server & grader endpoints authenticated, latency-bound, and unit-tested.
- CLA emits RFT telemetry only when feature flag enabled; no regression in legacy flow.
- Config contains safe defaults disabling RFT; errors handled if endpoints unreachable.
- Documentation (`docs/cla_rft_playbook.md`) details rollout safety, reward rubric, and monitoring.
# Rollback & Guardrails (feature flag/circuit breaker)
- Introduce `ENABLE_CLA_RFT` feature flag default `false`; wrap all new CLA code paths.  
- Add circuit breaker in telemetry pipeline: if grader/tool server fails, fallback to current CLA flow and log warning.  
- Maintain separate deployment (container/service) for tool server; can be disabled independently.
# Risiko Sisa & Mitigasi
- Data privacy/compliance for exporting rollouts → Sanitize PII, limit dataset access (use service accounts).  
- RFT API quota or cost overruns → Add config to limit compute multiplier/epochs; monitor usage.  
- Latency impact from additional telemetry → Use async/non-blocking logging; batch writes to BigQuery.  
- Operational complexity → Provide runbooks/tests; stage rollout in non-production environment first.

