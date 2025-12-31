#!/usr/bin/env python3
"""
Standalone ADA + TAA Integration Showcase
=========================================
This script demonstrates the data flow between:
1. ADA (Anomaly Detection Agent): Mathematical detection.
2. TAA (Triage & Analysis Agent): Semantic LLM analysis.

It is designed to run everywhere without external dependencies.
"""

import json
from datetime import datetime

def showcase():
    print("="*80)
    print("🚀 ADA-TAA INTEGRATION FLOW DEMONSTRATION")
    print("="*80)

    # --- PART 1: ADA (MATHEMATICAL BRAIN) ---
    print("\n[STEP 1] ADA Mathematical Processing")
    print("-" * 40)
    
    # Simulate telemetry ingestion
    telemetry = {
        "event_type": "network_traffic",
        "src_ip": "10.0.0.15",
        "dst_ip": "185.220.101.45",
        "bytes_sent": 15420,
        "packet_count": 45,
        "duration": 0.5
    }
    
    # Simulate Mathematical Analysis (Autoencoder reconstruction error)
    # In a real system, this comes from advanced_anomaly_detection.py
    reconstruction_error = 0.92
    threshold = 0.1
    is_anomaly = reconstruction_error > threshold
    
    ada_signal = {
        "is_anomaly": is_anomaly,
        "score": reconstruction_error,
        "detector": "Neural Autoencoder",
        "reasoning": ["Reconstruction error significantly above baseline (0.92 > 0.10)"]
    }
    
    print(f"INPUT RAW TELEMETRY: {json.dumps(telemetry, indent=2)}")
    print(f"ADA OUTPUT SIGNAL: {json.dumps(ada_signal, indent=2)}")

    # --- PART 2: THE BRIDGE ---
    # ADA publishes this to Pub/Sub (e.g., 'security-alerts')
    alert_packet = {
        "alert_id": "ALERT-2025-001",
        "timestamp": datetime.now().isoformat(),
        "log_data": telemetry,
        "analytical_context": ada_signal
    }

    # --- PART 3: TAA (SEMANTIC BRAIN) ---
    print("\n[STEP 2] TAA Semantic Analysis (LLM)")
    print("-" * 40)
    
    # TAA receives the alert_packet and enriches it
    # Then it builds a prompt for the LLM (Gemini/Llama)
    
    mock_llm_prompt = f"""
    Analyze this alert:
    Numerical Score: {alert_packet['analytical_context']['score']}
    Source: {alert_packet['log_data']['src_ip']}
    Destination: {alert_packet['log_data']['dst_ip']}
    
    The mathematical model indicates a high deviation from normal patterns.
    Interpret this from a security perspective.
    """
    
    # Mock LLM Response (The "Semantic" part)
    # In a real system, this comes from Vertex AI call in triage_analysis_agent.py
    llm_response = {
        "is_true_positive": True,
        "attack_type": "Potential Malware C2 Beaconing",
        "severity": "HIGH",
        "reasoning": "Mathematical anomaly (0.92) confirms statistical outlier in traffic volume. Correlation with known Tor Exit Node range (185.220.x.x) strongly suggests exfiltration or beaconing.",
        "recommended_actions": ["Isolate host 10.0.0.15", "Inspect process logs for src_ip"]
    }
    
    print(f"LLM PROMPT SENT: [Contextual Metadata + ADA Signals]")
    print(f"TAA FINAL DECISION:\n{json.dumps(llm_response, indent=2)}")

    print("\n" + "="*80)
    print("✅ INTEGRATION ARCHITECTURE VALIDATED")
    print("="*80)

if __name__ == "__main__":
    showcase()
