# Scope
- Target change/bug/feature: Apply `RL/agent RFT.md` blueprint specifically to the Continuous Learning Agent (CLA) to prepare it for reinforcement fine-tuning (RFT).
- Components/Services: RFT guide (`RL/agent RFT.md`), CLA production implementation (`production_cla_service.py`), related data/logging pipeline (BigQuery `processed_alerts`, Pub/Sub topics, model storage).
# Peta File & Simbol (path + [Lx–Ly] + 1-line role)
- `RL/agent RFT.md` [L21-L109] – summarizes RFT prerequisites (task definition, tool server, grader, infrastructure).
- `production_cla_service.py` [L47-L119] – CLA core service: loads models, interfaces with BigQuery/PubSub, schedules retraining.
- `production_cla_service.py` [L232-L420] – data ingestion/prediction pipeline writing to BigQuery and Pub/Sub (potential telemetry for reward calculation).
- `production_cla_service.py` [L517-L704] – feedback ingestion, retraining scheduler, metrics reporting (sources for labelled rewards).
- `config/production_cla_config.json` (if present) – runtime configuration determining datasets, topics, model paths for CLA.
- BigQuery catalog `gatra_database` (`bq ls`/`__TABLES__`) – row counts confirm which tables are populated (e.g., `siem_events` ~1.25M rows, `processed_alerts` empty).
# Alur Eksekusi end-to-end (linked to lines)
1. Document CLA’s current workflow: ingest alerts → preprocess → predict → publish results (`production_cla_service.py` [L232-L420]).
2. Track how feedback is collected and retraining triggered (`production_cla_service.py` [L517-L704]) to understand existing learning loop.
3. Map data storage (BigQuery tables `processed_alerts`, `feedback`) and Pub/Sub topics defined in config to identify reward signal sources.
4. From `RL/agent RFT.md`, extract requirements for tool calls, grader endpoint, rollout infra suited to CLA scenario.
5. Align CLA’s current outputs with RFT expectations (tool traces, reward calculations) to outline necessary instrumentation.
# Tes & Observabilitas (tests, log, how-to-run)
- Need to quantify CLA baseline (accuracy, precision/recall) via BigQuery queries (`processed_alerts`, `feedback`).
- Plan to simulate rollouts capturing CLA decisions and tool interactions for future reward grading.
- Monitor RFT-specific metrics (reward curves, tool-call counts, latency) once pipeline in place per guide [RL/agent RFT.md L95-L99].
# Risiko & Asumsi
- Need confirmation of dataset availability and production parity environment for CLA rollouts.
- Tool server and grader endpoints not yet implemented; assumptions on hosting/authorization must be validated.
- RFT may demand additional logging storage and compute resources for rollouts; capacity planning required.
# Bukti (3–5 mini snippets only)
- `RL/agent RFT.md` [L21-L76]: Lays out baseline, tool, and reward design requirements.
- `production_cla_service.py` [L47-L95]: Details CLA initialization and integration points with GCP services.
- `production_cla_service.py` [L232-L420]: Illustrates prediction pipeline generating outputs suited for reward evaluation.
- `production_cla_service.py` [L517-L704]: Shows feedback loop and retraining triggers relevant for RFT data collection.
- BigQuery query `SELECT table_id, row_count FROM \`chronicle-dev-2be9.gatra_database.__TABLES__\`` – validates active vs empty tables (e.g., `ada_state` 101,024 rows; `feedback` 0 rows).
# Bukti (3–5 mini snippets only)
- `RL/agent RFT.md` [L21-L76]: Lays out baseline, tool, and reward design requirements.
- `production_cla_service.py` [L47-L95]: Details CLA initialization and integration points with GCP services.
- `production_cla_service.py` [L232-L420]: Illustrates prediction pipeline generating outputs suited for reward evaluation.
- `production_cla_service.py` [L517-L704]: Shows feedback loop and retraining triggers relevant for RFT data collection.
