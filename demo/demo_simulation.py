#!/usr/bin/env python3
"""
AI-Driven SOC Platform - Demo Simulation Script

This script simulates realistic SOC platform activity for demo videos and presentations.
It generates fake security events, detections, and responses to showcase the platform's
capabilities without using real customer data.

Usage:
    python demo_simulation.py [--mode <mode>] [--duration <seconds>]

Modes:
    - dashboard: Continuous activity for dashboard recording
    - detection: Single threat detection scenario
    - full-demo: Complete end-to-end demo with narration cues
"""

import argparse
import json
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any


# Demo Configuration
DEMO_TENANTS = [
    {"id": "tenant_acme", "name": "ACME Corporation", "industry": "Finance"},
    {"id": "tenant_globex", "name": "Globex Industries", "industry": "Manufacturing"},
    {"id": "tenant_initech", "name": "Initech Solutions", "industry": "Technology"},
]

THREAT_SCENARIOS = [
    {
        "name": "Brute Force Attack",
        "severity": "HIGH",
        "source_ip": "185.143.223.XXX",
        "target": "auth-server-01",
        "events_count": 1247,
        "detection_time_ms": 87,
        "confidence": 0.94,
        "action": "AUTO_BLOCK",
        "description": "Multiple failed login attempts from single IP"
    },
    {
        "name": "Data Exfiltration Attempt",
        "severity": "CRITICAL",
        "source_ip": "10.0.5.XXX",
        "target": "file-server-prod",
        "events_count": 523,
        "detection_time_ms": 156,
        "confidence": 0.89,
        "action": "QUARANTINE",
        "description": "Unusual outbound data transfer to external endpoint"
    },
    {
        "name": "Lateral Movement Detected",
        "severity": "HIGH",
        "source_ip": "10.0.3.XXX",
        "target": "multiple-hosts",
        "events_count": 89,
        "detection_time_ms": 203,
        "confidence": 0.91,
        "action": "ISOLATE",
        "description": "Suspicious RDP connections across network segments"
    },
    {
        "name": "Malware Command & Control",
        "severity": "CRITICAL",
        "source_ip": "10.0.8.XXX",
        "target": "workstation-142",
        "events_count": 34,
        "detection_time_ms": 45,
        "confidence": 0.97,
        "action": "ISOLATE",
        "description": "Beacon traffic to known C2 infrastructure"
    },
    {
        "name": "Privilege Escalation",
        "severity": "HIGH",
        "source_ip": "10.0.2.XXX",
        "target": "domain-controller",
        "events_count": 12,
        "detection_time_ms": 78,
        "confidence": 0.86,
        "action": "ALERT",
        "description": "Unauthorized admin privilege assignment detected"
    },
]

EVENT_TYPES = [
    "network_flow", "authentication", "file_access", "process_execution",
    "dns_query", "http_request", "email_event", "endpoint_telemetry"
]


def print_banner():
    """Print demo banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║           AI-DRIVEN SOC PLATFORM - DEMO MODE                ║
    ║                                                              ║
    ║   This is a simulation for demonstration purposes only.     ║
    ║   No real security events or customer data are used.        ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def generate_timestamp() -> str:
    """Generate current timestamp in ISO format."""
    return datetime.utcnow().isoformat() + "Z"


def generate_event_id() -> str:
    """Generate a realistic event ID."""
    return f"evt_{random.randint(100000000, 999999999)}"


def generate_metrics() -> Dict[str, Any]:
    """Generate realistic dashboard metrics."""
    return {
        "timestamp": generate_timestamp(),
        "events_per_second": random.randint(45000, 52000),
        "total_events_24h": random.randint(3_800_000, 4_200_000),
        "alerts_generated": random.randint(120, 180),
        "alerts_auto_resolved": random.randint(80, 120),
        "alerts_escalated": random.randint(5, 15),
        "mean_detection_time_ms": random.randint(75, 150),
        "mean_response_time_ms": random.randint(200, 500),
        "ada_agent_status": "ACTIVE",
        "taa_agent_status": "ACTIVE",
        "cra_agent_status": "ACTIVE",
        "active_tenants": len(DEMO_TENANTS),
        "system_health": "HEALTHY"
    }


def simulate_event_stream(duration_seconds: int = 10):
    """Simulate continuous event stream for dashboard recording."""
    print("\n[DEMO] Starting event stream simulation...")
    print("=" * 60)

    start_time = time.time()
    event_count = 0

    while time.time() - start_time < duration_seconds:
        # Generate batch of events
        batch_size = random.randint(100, 500)
        event_count += batch_size

        tenant = random.choice(DEMO_TENANTS)
        event_type = random.choice(EVENT_TYPES)

        print(f"[{generate_timestamp()}] Processed {batch_size} {event_type} events | "
              f"Tenant: {tenant['name']} | Total: {event_count:,}")

        # Occasionally show an anomaly detection
        if random.random() < 0.1:
            print(f"  └─ [ADA] Anomaly detected! Confidence: {random.uniform(0.7, 0.95):.2f}")

        time.sleep(0.5)

    print("=" * 60)
    print(f"[DEMO] Event stream ended. Total events: {event_count:,}")


def simulate_threat_detection(scenario: Dict[str, Any] = None):
    """Simulate a complete threat detection and response scenario."""
    if scenario is None:
        scenario = random.choice(THREAT_SCENARIOS)

    tenant = random.choice(DEMO_TENANTS)

    print("\n" + "=" * 70)
    print("                    THREAT DETECTION SIMULATION")
    print("=" * 70)

    # Phase 1: Event Ingestion
    print(f"\n[{generate_timestamp()}] Phase 1: EVENT INGESTION")
    print(f"  Tenant: {tenant['name']} ({tenant['industry']})")
    print(f"  Processing incoming security events...")
    time.sleep(1)

    for i in range(5):
        print(f"  └─ Batch {i+1}: {random.randint(500, 2000)} events ingested")
        time.sleep(0.3)

    # Phase 2: ADA Detection
    print(f"\n[{generate_timestamp()}] Phase 2: ADA ANOMALY DETECTION")
    print(f"  Running ML models...")
    time.sleep(0.5)
    print(f"  └─ Isolation Forest: Anomaly score 0.{random.randint(85, 99)}")
    print(f"  └─ Behavioral baseline: Deviation detected")
    print(f"  └─ Pattern matching: {scenario['events_count']} correlated events")
    time.sleep(0.5)

    print(f"\n  ⚠️  ANOMALY DETECTED!")
    print(f"  ├─ Type: {scenario['name']}")
    print(f"  ├─ Severity: {scenario['severity']}")
    print(f"  ├─ Source: {scenario['source_ip']}")
    print(f"  ├─ Target: {scenario['target']}")
    print(f"  ├─ Confidence: {scenario['confidence']:.0%}")
    print(f"  └─ Detection Time: {scenario['detection_time_ms']}ms")

    time.sleep(1)

    # Phase 3: TAA Analysis
    print(f"\n[{generate_timestamp()}] Phase 3: TAA TRIAGE & ANALYSIS")
    print(f"  Enriching alert with context...")
    time.sleep(0.5)
    print(f"  └─ Threat Intelligence: Matched 3 indicators")
    print(f"  └─ Asset criticality: HIGH")
    print(f"  └─ Historical context: No prior incidents")
    time.sleep(0.5)

    print(f"\n  🤖 LLM Analysis (Gemini):")
    print(f"  ┌────────────────────────────────────────────────────────────┐")
    print(f"  │ {scenario['description']}")
    print(f"  │ Risk Assessment: {scenario['severity']} - Immediate action required")
    print(f"  │ Recommendation: {scenario['action']}")
    print(f"  └────────────────────────────────────────────────────────────┘")

    time.sleep(1)

    # Phase 4: Automated Response
    print(f"\n[{generate_timestamp()}] Phase 4: AUTOMATED RESPONSE")
    action_map = {
        "AUTO_BLOCK": "Blocking source IP at firewall...",
        "QUARANTINE": "Quarantining affected endpoint...",
        "ISOLATE": "Isolating network segment...",
        "ALERT": "Escalating to security team..."
    }
    print(f"  Action: {scenario['action']}")
    print(f"  └─ {action_map.get(scenario['action'], 'Executing response...')}")
    time.sleep(0.5)
    print(f"  └─ Response executed successfully")
    print(f"  └─ Incident ticket created: INC-{random.randint(10000, 99999)}")

    # Summary
    total_time = scenario['detection_time_ms'] + random.randint(200, 400)
    print(f"\n{'=' * 70}")
    print(f"  ✅ THREAT CONTAINED")
    print(f"  Total Time: {total_time}ms ({total_time/1000:.1f} seconds)")
    print(f"  Status: {scenario['action']} completed")
    print(f"{'=' * 70}\n")


def simulate_dashboard_metrics(duration_seconds: int = 30):
    """Simulate dashboard metrics updates for recording."""
    print("\n[DEMO] Starting dashboard metrics simulation...")
    print("Press Ctrl+C to stop\n")

    start_time = time.time()

    while time.time() - start_time < duration_seconds:
        metrics = generate_metrics()

        print("\033[2J\033[H")  # Clear screen
        print("=" * 70)
        print("           AI-DRIVEN SOC PLATFORM - LIVE DASHBOARD")
        print("=" * 70)
        print(f"\n  Timestamp: {metrics['timestamp']}")
        print(f"\n  📊 REAL-TIME METRICS")
        print(f"  ├─ Events/Second:     {metrics['events_per_second']:,}")
        print(f"  ├─ Events (24h):      {metrics['total_events_24h']:,}")
        print(f"  ├─ Alerts Generated:  {metrics['alerts_generated']}")
        print(f"  ├─ Auto-Resolved:     {metrics['alerts_auto_resolved']}")
        print(f"  └─ Escalated:         {metrics['alerts_escalated']}")
        print(f"\n  ⚡ PERFORMANCE")
        print(f"  ├─ Detection Time:    {metrics['mean_detection_time_ms']}ms")
        print(f"  └─ Response Time:     {metrics['mean_response_time_ms']}ms")
        print(f"\n  🤖 AGENT STATUS")
        print(f"  ├─ ADA (Detection):   {metrics['ada_agent_status']} ✓")
        print(f"  ├─ TAA (Analysis):    {metrics['taa_agent_status']} ✓")
        print(f"  └─ CRA (Response):    {metrics['cra_agent_status']} ✓")
        print(f"\n  🏢 TENANTS: {metrics['active_tenants']} active")
        print(f"  💚 SYSTEM: {metrics['system_health']}")
        print("\n" + "=" * 70)

        time.sleep(2)


def run_full_demo():
    """Run complete demo with narration cues."""
    print_banner()

    print("\n" + "=" * 70)
    print("                    FULL DEMO SEQUENCE")
    print("=" * 70)
    print("\nThis demo will walk through the complete platform capabilities.")
    print("Follow the narration cues for your video recording.\n")

    input("Press ENTER to begin... (Cue: 'Let me show you how it works')")

    # Part 1: Dashboard Overview
    print("\n[NARRATION CUE] 'Here's our main dashboard showing real-time metrics...'")
    input("Press ENTER to show dashboard...")
    metrics = generate_metrics()
    print(f"\n  Events/Second: {metrics['events_per_second']:,}")
    print(f"  Total Events (24h): {metrics['total_events_24h']:,}")
    print(f"  Active Agents: ADA ✓ | TAA ✓ | CRA ✓")

    # Part 2: Trigger Detection
    print("\n[NARRATION CUE] 'Now watch what happens when a threat is detected...'")
    input("Press ENTER to trigger threat detection...")
    simulate_threat_detection(THREAT_SCENARIOS[0])  # Brute Force

    # Part 3: Show Response
    print("\n[NARRATION CUE] 'The threat was contained in under 30 seconds...'")
    input("Press ENTER to show summary...")

    print("\n" + "=" * 70)
    print("                    DEMO COMPLETE")
    print("=" * 70)
    print("\n  Key Takeaways:")
    print("  • Autonomous detection with 95%+ accuracy")
    print("  • Sub-second threat identification")
    print("  • Automated response without human intervention")
    print("  • Full audit trail for compliance")
    print("\n[NARRATION CUE] 'Contact us for a personalized demo...'")


def main():
    parser = argparse.ArgumentParser(description="AI-Driven SOC Demo Simulation")
    parser.add_argument("--mode", choices=["dashboard", "detection", "stream", "full-demo"],
                       default="detection", help="Demo mode to run")
    parser.add_argument("--duration", type=int, default=30,
                       help="Duration in seconds (for continuous modes)")
    args = parser.parse_args()

    print_banner()

    if args.mode == "dashboard":
        simulate_dashboard_metrics(args.duration)
    elif args.mode == "detection":
        simulate_threat_detection()
    elif args.mode == "stream":
        simulate_event_stream(args.duration)
    elif args.mode == "full-demo":
        run_full_demo()


if __name__ == "__main__":
    main()
