# CLA Dashboard Enhancements: TI Lookup, Contextual Bandit, RAG, TTPs, Historical Correlation, Investigative Steps

## Overview
This document baselines the newly added features to the Comprehensive SOC Dashboard (port 8535) Alert Review tab and sets expectations for development and implementation.

## Features
- Threat Intelligence (TI) Lookup
  - Auto-extract IPs from `raw_alert` and common fields (`source_ip`, `src_ip`, `destination_ip`, `dst_ip`, `ip`, `client_ip`, `server_ip`).
  - Render direct links for each IP: VirusTotal, AbuseIPDB, Shodan.
  - Graceful fallback notice when no IPs are detected.
- Contextual Bandit Analysis
  - Bandit score and recommendations based on confidence, severity, IP count, and bytes transferred.
  - Risk-tuned actions: immediate containment for high-confidence/high-severity; enhanced monitoring for medium contexts.
- RAG-Enhanced Context
  - Knowledge retrieval stubs mapped to alert classification, network-flow traits, repeated IPs, and data transfer volume.
  - Outputs similarity context and knowledge base references.
- TTP Analysis & MITRE ATT&CK Mapping
  - Derived from classification, network-flow presence, data-transfer magnitude, and severity.
  - Provides mapped techniques (e.g., T1041, T1048, T1071, etc.).
- Historical Incident Correlation
  - Summarizes prior similar incidents with outcomes (TP/FP) to guide analyst judgment.
- Detailed Investigative Steps (8-step playbook)
  - Initial assessment, network analysis, data transfer investigation, endpoint analysis, TI correlation, impact assessment, response, follow-up.

## Data Handling
- Timezone: Timestamps are displayed in `Asia/Jakarta` where applicable.
- NA-safe booleans: `is_anomaly` handled as Yes/No/Unknown.
- Severity: Derived from `confidence_score` when not present.
- Bytes transferred coerced to integer when possible.

## UI Placement
- Alert Review → right column → sections order:
  1) Extracted Parameters
  2) Model Recommendations
  3) Detailed Analysis (Network/Data/Threat Assessment/Questions)
  4) Enhanced Recommended Actions
     - TI Lookup
     - Contextual Bandit Analysis
     - RAG-Enhanced Context
     - TTP Analysis & Mapping
     - Historical Incident Correlation
     - Detailed Investigative Steps
     - Risk-Based Immediate Actions

## Implementation Status
- Implemented in `restored_dashboard_with_funnel.py`.
- TI links visible for IPs extracted from `raw_alert` or top-level fields.
- Robust IP parsing for dicts, JSON strings, nested dicts/lists.
- Safe handling of missing data with informative messages.

## Next Steps
- Replace RAG stubs with live retrieval (e.g., vector DB + embeddings).
- Persist feedback to BigQuery and close the loop to CLA retraining.
- Add enrichment sources (Whois, GreyNoise, URLhaus, AlienVault OTX).
- Provide toggle to export the recommendation block to incident tickets.

## Testing
- Use alerts that include: single IP, multiple IPs, large data transfers, and missing fields.
- Validate TI links render and are clickable.
- Confirm NA-safe handling of `is_anomaly` and timezone formatting.

## Change Log
- 2025-09-30: Initial baseline of enhancements documented.