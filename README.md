
# LLM
# AI-Driven SOC: LangGraph ADA Integration

This repository provides an enhanced anomaly detection and enrichment pipeline for Security Operations Centers (SOC) using LangGraph orchestration, with robust preprocessing, enrichment, and test coverage.

## Directory Structure

```
├── langgraph_ada_integration.py   # Main ADA pipeline (preprocessing, enrichment, detection, etc)
├── test_scale_ai_preprocessing.py # Test script for Scale AI preprocessing (single, batch, large batch)
├── requirements.txt               # Python dependencies
├── README.md                      # This file
```

## Key Features
- **Scale AI Preprocessing:** Deduplication, sensitive field redaction, semantic labeling, kill chain enrichment
- **Batch and Large-Scale Testing:** Easily test with synthetic or real log data
- **Modular & Extensible:** Easily integrate with SIEM/log sources and further enrich

## Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Preprocessing Tests
```bash
python test_scale_ai_preprocessing.py
```
- The script will print summary statistics and spot-check results for single, batch, and large-scale log processing.

### 3. Main ADA Pipeline
- Use `langgraph_ada_integration.py` as the core module for your SOC pipeline.
- Integrate with your SIEM/log data as needed.

---

## Persistent BigQuery-Driven ADA Agent Service

**Overview:**

The `langgraph_ada_integration.py` script has been refactored to run as a persistent, long-running service. Instead of running a one-off test or batch job, the agent now continuously polls a BigQuery table for new security alerts and processes them as they arrive. This is suitable for production SOC environments where new events may arrive at any time.

### Key Changes

- **BigQuery Integration Import:**
  - The script imports `fetch_bigquery_data` and `process_bigquery_rows` from `bigquery_integration.py` for fetching and formatting alert data from BigQuery.

- **Persistent Service Loop:**
  - Added an asynchronous function `persistent_bigquery_service()` that:
    - Continuously fetches new, unprocessed alerts from BigQuery.
    - Processes each alert using the ADA agent.
    - Sleeps for a configurable interval when there are no new alerts, or after each polling cycle.

- **Main Entry Point Update:**
  - The script now runs the persistent service loop when started (e.g., by systemd):
    ```python
    if __name__ == "__main__":
        asyncio.run(persistent_bigquery_service())
    ```

### Usage Notes
- The example query assumes a `processed` field in your BigQuery table. You should update this field after processing each alert to avoid reprocessing the same data.
- Adjust the polling interval (`await asyncio.sleep(...)`) to balance responsiveness and resource usage.
- Errors in processing individual alerts are logged, but the service continues running.
- For high alert volumes, consider batching or parallel processing.

---

## LangGraph Workflow Restoration and Refactoring

**Overview:**

The `langgraph_ada_integration.py` script recently underwent a major restoration and refactoring to resolve severe file corruption and fix critical bugs in the anomaly detection workflow. The previous implementation suffered from duplicated code blocks, syntax errors, and incorrect LangGraph wiring, which prevented the enrichment and validation steps from executing correctly.

**Key Changes and Fixes:**

1.  **File Restoration:**
    *   The original `langgraph_ada_integration.py` was deleted and replaced with a completely new, clean implementation to resolve persistent `IndentationError` and syntax errors caused by file corruption.

2.  **LangGraph Workflow Correction:**
    *   **Parallel Enrichment:** The workflow was re-architected to execute four enrichment nodes (`geo_enrichment`, `threat_intel_enrichment`, `historical_enrichment`, `asset_enrichment`) in parallel after an anomaly is detected.
    *   **Aggregation Node:** A new `aggregate_enrichment_node` was introduced to collect the results from all parallel enrichment branches before proceeding to validation.
    *   **Corrected Routing:** The conditional routing logic was fixed to ensure that enriched data is correctly passed to the `validate_detection_node`. Non-anomalous events now correctly bypass the enrichment stage.

3.  **State Management:**
    *   The `ADAState` TypedDict was refined to manage the state of individual enrichment results separately, allowing for cleaner aggregation and debugging.

4.  **Dependency Resolution:**
    *   An `ImportError` for `NotRequired` was resolved by adding `typing-extensions` to `requirements.txt` and updating the import statement, ensuring compatibility with Python 3.9.

5.  **Validation and Logging:**
    *   The restored workflow was validated by adding detailed logging to trace the flow of data through the enrichment, aggregation, and validation nodes. A temporary `ada_workflow.log` file was used to confirm that the enrichment context was correctly applied to adjust detection confidence.

**Current Status:**

The LangGraph Anomaly Detection Agent is now in a stable, runnable state. The core workflow for detection, parallel enrichment, and validation has been verified to work as intended on the local development environment.

---

## Sharing
- All code is self-contained and ready to be pushed to GitLab or shared with your team.
- Remove or update any sensitive credentials/configs before sharing.

## Next Steps
- Integrate with real SIEM/log sources for production-scale validation.
- Extend enrichment logic as needed for your use case.

---

For questions or contributions, please contact the project maintainer.
