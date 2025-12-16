#!/usr/bin/env python3
"""
Healthcheck & Security Validator for AI-Driven SOC Dashboards
"""

import os, sys, subprocess, requests, json

PRIMARY_DASHBOARD = "http://localhost:8507"   # taa_dashboard.py
ENHANCED_DASHBOARD = "http://localhost:8506"  # ada_bigquery_dashboard.py
SIMPLE_DASHBOARD = "http://10.45.254.19:8519"
ORCHESTRATOR_URL = "http://10.45.254.19:8000/orchestrator/run"
PROJECT_ID = os.getenv("PROJECT_ID", "chronicle-dev-2be9")
DATASET_ID = os.getenv("DATASET_ID", "gatra_database")
TABLE = "raw_events"

def curl_check(url, timeout=5):
    try:
        r = requests.get(url, timeout=timeout)
        return {"url": url, "status": r.status_code, "latency_s": round(r.elapsed.total_seconds(), 3)}
    except Exception as e:
        return {"url": url, "status": f"ERR:{type(e).__name__}", "latency_s": None}

def bigquery_count():
    try:
        out = subprocess.check_output([
            "bq","query","--use_legacy_sql=false",
            f"SELECT COUNT(*) as total FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE}`"
        ], stderr=subprocess.STDOUT, text=True)
        return {"bigquery": "OK", "rows": out.strip()}
    except subprocess.CalledProcessError as e:
        return {"bigquery": f"ERR:{e.returncode}", "rows": None}

def secret_scan():
    issues = []
    # Scan known config files for hard-coded secrets
    files = [
        ".cursor/mcp.json",
        "config/taa_config.json",
        "gatra-production/.cursor/mcp.json"
    ]
    for f in files:
        if not os.path.exists(f): 
            continue
        with open(f) as fh:
            content = fh.read()
            if "sk-ant" in content or "API_KEY" in content:
                issues.append(f"⚠️ Potential secret in {f}")
    return issues or ["✅ No secrets found"]

def streamlit_config_check():
    cfg = ".streamlit/config.toml"
    if not os.path.exists(cfg):
        return ["⚠️ Missing .streamlit/config.toml (consider hardening)"]
    warnings = []
    with open(cfg) as f:
        data = f.read()
        if "enableCORS = false" not in data:
            warnings.append("⚠️ enableCORS not disabled")
        if "address = \"127.0.0.1\"" not in data:
            warnings.append("⚠️ Streamlit not bound to localhost")
    return warnings or ["✅ Streamlit config hardened"]

def main():
    mode = "--smoke" if "--smoke" in sys.argv else "--full" if "--full" in sys.argv else "--secure" if "--secure" in sys.argv else None
    
    results = {}
    if mode in ["--smoke","--full"]:
        results["dashboards"] = [
            curl_check(PRIMARY_DASHBOARD),
            curl_check(ENHANCED_DASHBOARD),
            curl_check(SIMPLE_DASHBOARD)
        ]
        results["orchestrator"] = curl_check(ORCHESTRATOR_URL)
    if mode in ["--full"]:
        results["bigquery"] = bigquery_count()
    if mode in ["--secure"]:
        results["secret_scan"] = secret_scan()
        results["streamlit_config"] = streamlit_config_check()

    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
