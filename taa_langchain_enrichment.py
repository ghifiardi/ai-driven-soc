"""
TAA LangChain Enrichment Orchestrator
=====================================

This script orchestrates enrichment of security alerts using:
- Local LLM (Mistral via llama-cpp-python) for cost-effective enrichment
- OpenAI GPT as a fallback for high-severity or low-confidence cases

Requirements:
- pip install langchain openai llama-cpp-python langchain_community
- llama.cpp server running with Mistral model (see README)
- Set OPENAI_API_KEY and LOCAL_LLM_PATH as environment variables

NOTE: If you see import errors for langchain, run:
    pip install langchain openai llama-cpp-python langchain_community
"""

import os
import json

try:
    from langchain_community.llms import LlamaCpp, OpenAI
except ImportError as e:
    print("\n[ERROR] langchain or dependencies not installed. Please run:")
    print("    pip install langchain openai llama-cpp-python langchain_community\n")
    raise e

# Config
LOCAL_LLM_PATH = os.environ.get("LOCAL_LLM_PATH", "mistral-7b-instruct-v0.2.Q4_K_M.gguf")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Local LLM enrichment function
def local_enrich(alert: dict) -> dict:
    prompt = f"""
You are a cybersecurity analyst. Analyze the following security alert and provide:
- Risk score (0-1)
- Risk level (low/medium/high)
- Key indicators
- Recommended actions

Alert:
{json.dumps(alert, indent=2)}
"""
    llm = LlamaCpp(model_path=LOCAL_LLM_PATH, n_ctx=2048, temperature=0.1)
    response = llm(prompt)
    return {"llm": "local", "enrichment": response}

# OpenAI enrichment function
def openai_enrich(alert: dict) -> dict:
    prompt = f"""
You are a cybersecurity analyst. Analyze the following security alert and provide:
- Risk score (0-1)
- Risk level (low/medium/high)
- Key indicators
- Recommended actions

Alert:
{json.dumps(alert, indent=2)}
"""
    llm = OpenAI(api_key=OPENAI_API_KEY, model="gpt-4o", temperature=0.1)
    response = llm(prompt)
    return {"llm": "openai", "enrichment": response}

# LangChain agent orchestration
def enrich_alert(alert: dict) -> dict:
    """
    Orchestrate enrichment: use local LLM by default, OpenAI for high severity or fallback.
    """
    # Use local LLM first
    local_result = local_enrich(alert)
    # Simple heuristic: escalate if alert is high severity or local LLM is uncertain
    severity = alert.get("severity", "medium").lower()
    if severity == "high" or "uncertain" in local_result["enrichment"].lower():
        if OPENAI_API_KEY:
            openai_result = openai_enrich(alert)
            return {"alert": alert, "enrichment": openai_result}
    return {"alert": alert, "enrichment": local_result}

# For testing/demo
if __name__ == "__main__":
    sample_alert = {
        "alert_id": "ALERT-123",
        "timestamp": "2025-07-14T10:00:00Z",
        "source": "google_cloud",
        "event_type": "suspicious_login",
        "username": "alice@example.com",
        "ip_address": "203.0.113.42",
        "location": "Singapore",
        "severity": "high",
        "details": {
            "device": "Chrome on Windows",
            "login_time": "2025-07-14T09:59:58Z"
        }
    }
    result = enrich_alert(sample_alert)
    print(json.dumps(result, indent=2)) 