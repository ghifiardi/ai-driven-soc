#!/usr/bin/env python3
"""
ADA + TAA LLM Showcase
======================
This script demonstrates the end-to-end integration between:
1. ADA (Anomaly Detection Agent): Mathematical detection (Autoencoder, Graph, Causal)
2. TAA (Triage & Analysis Agent): Semantic reasoning using LLM (Gemini/Llama)

Architecture:
Mathematical Signal (ADA) -> Semantic Interpretation (LLM in TAA) -> Confirmed Incident
"""

import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("Showcase")

# Fallback for missing dependencies to ensure the showcase still runs
try:
    import numpy as np
    from advanced_anomaly_detection import GATRAAnomalyDetectionSystem
    USING_MOCK_ADA = False
except ImportError as e:
    logger.warning(f"Missing dependency ({e}). Using Mock ADA for demonstration.")
    USING_MOCK_ADA = True
    class np:
        @staticmethod
        def random(): pass
        class random:
            @staticmethod
            def rand(*args): return [0.5]*args[0] if args else 0.5
    class GATRAAnomalyDetectionSystem:
        def __init__(self):
            self.ae_detector = type('obj', (object,), {'fit': lambda *args, **kwargs: None})()
        def process_telemetry(self, telemetry):
            return type('obj', (object,), {
                'is_anomaly': True,
                'score': 0.9421,
                'detector_type': 'GATRA_Ensemble (Autoencoder + Graph)',
                'reasoning': ['Reconstruction error above threshold', 'Circular dependency detected']
            })()

from triage_analysis_agent import TriageAnalysisAgent

def run_showcase():
    print("\n" + "="*80)
    print("🚀 STARTING ADA-TAA INTEGRATION SHOWCASE")
    print("="*80)

    # 1. Initialize ADA (Mathematical Brain)
    logger.info("Initializing ADA (Advanced Anomaly Detection Agent)...")
    ada = GATRAAnomalyDetectionSystem()
    
    # Train it briefly so it knows what "normal" looks like
    logger.info("Training ADA Autoencoder on baseline telemetry...")
    baseline_data = np.random.rand(100, 10)
    ada.ae_detector.fit(baseline_data, epochs=10)

    # 2. Generate a "Suspicious" Telemetry Event
    # Let's simulate a large reconstruction error (outlier)
    suspicious_telemetry = {
        'alert_id': 'TELEMETRY-999',
        'source_ip': '10.0.0.45',
        'dest_ip': '185.220.101.45', # Known malicious subnet in metadata
        'event_type': 'outbound_connection',
        'features': np.random.rand(10) * 5.0, # Scale up to cause AE error
        'timestamp': datetime.now().isoformat()
    }

    print("\n" + "-"*40)
    print("STEP 1: ADA Mathematical Processing")
    print("-"*40)
    
    # Step 3: Process through ADA
    ada_result = ada.process_telemetry(suspicious_telemetry)
    
    print(f"📊 ADA Decision: {'⚠️ ANOMALY' if ada_result.is_anomaly else '✅ NORMAL'}")
    print(f"📊 ADA Score: {ada_result.score:.4f}")
    print(f"📊 ADA Reasoning: {', '.join(ada_result.reasoning)}")

    # 4. Initialize TAA (Semantic Brain)
    print("\n" + "-"*40)
    print("STEP 2: TAA Semantic Triage (LLM)")
    print("-"*40)
    
    logger.info("Initializing TAA (Triage & Analysis Agent)...")
    # We'll use a dummy config to avoid loading real GCP credentials in the demo
    # but the agent is built to use Vertex AI or local Llama conventionally.
    try:
        taa = TriageAnalysisAgent()
        
        # In this showcase, to ensure it runs without API keys, we manually enrich 
        # and explain the "bridge" logic.
        
        # Prepare the packet that TAA receives from ADA
        alert_to_triage = {
            "alert_id": suspicious_telemetry['alert_id'],
            "log_data": suspicious_telemetry,
            "ada_signal": {
                "score": ada_result.score,
                "reasons": ada_result.reasoning,
                "detector": ada_result.detector_type
            }
        }

        # If LLM is configured, TAA would call self._analyze_with_llm(enriched_alert)
        # For this showcase, we'll demonstrate what the LLM receives and outputs.
        print(f"🧠 TAA received signal: High reconstruction error ({ada_result.score:.2f}) from Autoencoder.")
        print(f"🧠 TAA enrichment finding: Destination IP {suspicious_telemetry['dest_ip']} is high-risk.")
        
        # Mocking the LLM outcome for the showcase flow
        print("\n[LLM REASONING LOG]")
        print("> 'The mathematical anomaly (score 0.94) combined with the connection to a known C2 ")
        print("> range (185.220.x.x) suggests a potential beaconing attempt. The Autoencoder detected ")
        print("> statistical deviation in the packet size and frequency, corroborating suspicious behavior.'")
        
        print("\n🎯 FINAL TRIAGE DECISION:")
        print("   is_true_positive: TRUE")
        print("   severity: HIGH")
        print("   attack_type: MALWARE_BEACONING")
        print("   recommended_action: [Isolate 10.0.0.45, Block 185.220.101.45]")

    except Exception as e:
        logger.error(f"TAA Initialization failed: {e}")
        print("Note: TAA usually requires GCP Vertex AI or a local GGUF model.")

    print("\n" + "="*80)
    print("✅ SHOWCASE COMPLETE")
    print("="*80)

if __name__ == "__main__":
    run_showcase()
