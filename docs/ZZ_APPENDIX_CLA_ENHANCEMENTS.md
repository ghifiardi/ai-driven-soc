# Appendix: CLA Dashboard Enhancements (TI, Bandit, RAG)

This appendix captures the latest enhancements deployed to the Alert Review tab on the Comprehensive SOC Dashboard (port 8535). It is intended as a development and implementation reference and should be updated as features mature.

## A. Threat Intelligence (TI) Lookup
- Automatic IP extraction from `raw_alert` (JSON/dict/strings), nested structures, and top-level fields (`source_ip`, `src_ip`, `destination_ip`, `dst_ip`, `ip`, `client_ip`, `server_ip`).
- Direct links per IP: VirusTotal, AbuseIPDB, Shodan.
- Graceful fallback when no IPs detected.

## B. Contextual Bandit Analysis
- Bandit score and contextual recommendations driven by: confidence, severity, number of IPs, bytes transferred.
- Action guidance tiers: immediate containment (high/high), enhanced monitoring (medium), baseline procedures (low).

## C. RAG-Enhanced Context
- Knowledge-base stubs mapped by classification, network flow, repeated IPs, and data volume.
- Similarity context and pattern references for analyst orientation.

## D. MITRE ATT&CK TTP Mapping
- Context-derived techniques: Application Layer Protocol, Exfiltration over C2, Alternative Protocols, Remote Services, Scanning, etc.
- Severity-aware additions for high-risk cases.

## E. Historical Incident Correlation
- Summaries of prior incidents with similar traits and their outcomes to calibrate analyst judgment.

## F. Eight-Step Investigative Playbook
1. Initial Assessment
2. Network Analysis
3. Data Transfer Investigation
4. Endpoint Analysis
5. Threat Intelligence Correlation
6. Impact Assessment
7. Response Actions
8. Follow-up

## G. Risk-Based Immediate Actions
- High/critical data volumes: isolation, blocking, escalation, evidence preservation.
- Medium: verification within SLA, monitoring, documentation.
- Standard: routine workflow, 24h closure target.

## Implementation Notes
- Implemented in `restored_dashboard_with_funnel.py`.
- Timezone conversions to `Asia/Jakarta`; NA-safe boolean handling for `is_anomaly`.
- Bytes transfer coercion to integer when possible.

## Roadmap
- Replace RAG stubs with live retrieval (vector DB).
- Persist analyst feedback to BigQuery and close the CLA loop.
- Add enrichment sources (Whois, GreyNoise, OTX, URLhaus) and ticket export.
