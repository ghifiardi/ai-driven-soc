# CLA Enhancements Summary (Baseline)

This baseline documents the new Alert Review enhancements deployed to the main dashboard (port 8535):

- Threat Intelligence (TI) Lookup with VirusTotal, AbuseIPDB, Shodan per IP
- Contextual Bandit analysis with bandit score and action guidance
- RAG-enhanced context from knowledge base stubs
- MITRE ATT&CK TTP mapping by context
- Historical incident correlation summary
- Detailed 8-step investigative playbook
- Risk-based immediate actions tied to data volume and severity

Implementation file: `restored_dashboard_with_funnel.py`.
Source doc: `docs/CLA_ENHANCEMENTS_TI_BANDIT_RAG.md`.

Next: wire RAG to live retrieval, enrich TI providers, persist feedback to BigQuery for CLA loop.