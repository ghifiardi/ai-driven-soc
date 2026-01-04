#!/usr/bin/env python3
"""
Threat Hunting Platform - Interactive Simulation
================================================
Demonstrates the complete workflow WITHOUT requiring GCP or deployment.
Perfect for team demonstrations and understanding the platform.

This simulation shows:
1. VALHALLA fetching threat intelligence
2. THOR scanning endpoints for threats
3. ASGARD orchestrating fleet-wide campaigns
4. TAA enriching and analyzing findings
5. CRA executing automated response
6. CLA learning from detections

Run: python3 demo_simulation.py
"""

import time
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
import json


class Colors:
    """Terminal colors for better visualization"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text: str):
    """Print a section header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(70)}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.END}\n")


def print_step(step: str, description: str):
    """Print a workflow step"""
    print(f"{Colors.CYAN}{Colors.BOLD}[{step}]{Colors.END} {description}")


def print_success(message: str):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")


def print_warning(message: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")


def print_error(message: str):
    """Print error message"""
    print(f"{Colors.RED}❌ {message}{Colors.END}")


def print_info(message: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ️  {message}{Colors.END}")


def simulate_progress(task: str, duration: int = 2):
    """Simulate a task with progress indicator"""
    print(f"   {task}...", end="", flush=True)
    for _ in range(duration):
        time.sleep(0.5)
        print(".", end="", flush=True)
    print(f" {Colors.GREEN}Done!{Colors.END}")


class ThreatHuntingSimulation:
    """Main simulation class"""

    def __init__(self):
        self.timestamp = datetime.now()

        # Simulated threat intelligence data
        self.threat_intel = {
            "iocs": {
                "ips": ["192.0.2.100", "198.51.100.50", "203.0.113.75"],
                "domains": ["malicious-site.example", "evil-domain.test"],
                "hashes": {
                    "a1b2c3d4e5f6...": "Lockbit 3.0 Ransomware",
                    "f6e5d4c3b2a1...": "Emotet Trojan",
                    "9876543210ab...": "Cobalt Strike Beacon"
                }
            },
            "yara_rules": [
                "Ransomware_Lockbit_3_0",
                "Trojan_Emotet_Loader",
                "APT_CobaltStrike_Beacon",
                "Webshell_Generic_PHP",
                "Backdoor_Linux_Rootkit"
            ]
        }

        # Simulated fleet of endpoints
        self.endpoints = [
            {"id": "web-server-01", "role": "webserver", "env": "production", "os": "linux"},
            {"id": "web-server-02", "role": "webserver", "env": "production", "os": "linux"},
            {"id": "db-server-01", "role": "database", "env": "production", "os": "linux"},
            {"id": "app-server-01", "role": "application", "env": "production", "os": "linux"},
            {"id": "dev-server-01", "role": "development", "env": "development", "os": "linux"}
        ]

        # Simulated scan results
        self.scan_results = {}

        # Simulated detections
        self.detections = []

    def run_simulation(self):
        """Run the complete simulation"""
        print_header("🎯 THREAT HUNTING PLATFORM SIMULATION")
        print(f"{Colors.BOLD}Simulating a complete threat hunting workflow{Colors.END}")
        print(f"Timestamp: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")

        input(f"{Colors.CYAN}Press ENTER to start the simulation...{Colors.END}")

        # Phase 1: VALHALLA
        self.simulate_valhalla()

        # Phase 2: ASGARD
        self.simulate_asgard()

        # Phase 3: THOR
        self.simulate_thor()

        # Phase 4: TAA
        self.simulate_taa()

        # Phase 5: CRA
        self.simulate_cra()

        # Phase 6: CLA
        self.simulate_cla()

        # Summary
        self.show_summary()

    def simulate_valhalla(self):
        """Simulate VALHALLA threat intelligence feed manager"""
        print_header("Phase 1: VALHALLA - Threat Intelligence Feed Manager")

        print_step("1.1", "Initializing VALHALLA Feed Manager")
        simulate_progress("   Loading configuration")
        print_success("VALHALLA initialized")
        print()

        print_step("1.2", "Fetching IOCs from ThreatFox (abuse.ch)")
        simulate_progress("   Connecting to ThreatFox API", 2)
        print_info(f"   Fetched {len(self.threat_intel['iocs']['ips'])} malicious IPs")
        print_info(f"   Fetched {len(self.threat_intel['iocs']['domains'])} malicious domains")
        print()

        print_step("1.3", "Fetching malware hashes from MalwareBazaar")
        simulate_progress("   Connecting to MalwareBazaar API", 2)
        print_info(f"   Fetched {len(self.threat_intel['iocs']['hashes'])} malware hashes")

        for hash_val, malware_name in list(self.threat_intel['iocs']['hashes'].items())[:3]:
            print(f"      • {hash_val[:16]}... → {malware_name}")
        print()

        print_step("1.4", "Downloading YARA rules from Emerging Threats")
        simulate_progress("   Downloading YARA rule repository", 2)
        print_info(f"   Downloaded {len(self.threat_intel['yara_rules'])} YARA rule sets")

        for rule in self.threat_intel['yara_rules']:
            print(f"      • {rule}")
        print()

        print_step("1.5", "Validating and compiling YARA rules")
        simulate_progress("   Validating rule syntax", 1)
        simulate_progress("   Compiling rules for performance", 1)
        print_success("All YARA rules validated and compiled")
        print()

        print_step("1.6", "Exporting threat intelligence for THOR agents")
        simulate_progress("   Storing in Firestore", 1)
        simulate_progress("   Uploading to Cloud Storage", 1)
        print_success("Threat intelligence exported and ready for distribution")
        print()

        print_info("📊 VALHALLA Summary:")
        print(f"   Total IOCs: {len(self.threat_intel['iocs']['ips']) + len(self.threat_intel['iocs']['domains']) + len(self.threat_intel['iocs']['hashes'])}")
        print(f"   YARA Rules: {len(self.threat_intel['yara_rules'])}")
        print(f"   Last Updated: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

        input(f"\n{Colors.CYAN}Press ENTER to continue to ASGARD...{Colors.END}")

    def simulate_asgard(self):
        """Simulate ASGARD orchestration agent"""
        print_header("Phase 2: ASGARD - Fleet Orchestration")

        print_step("2.1", "Initializing ASGARD Orchestration Agent")
        simulate_progress("   Loading configuration")
        print_success("ASGARD initialized")
        print()

        print_step("2.2", "Discovering and registering endpoints")
        simulate_progress("   Scanning GCP Compute Engine instances", 2)

        print_info(f"   Discovered {len(self.endpoints)} endpoints:")
        for ep in self.endpoints:
            print(f"      • {ep['id']} (role: {ep['role']}, env: {ep['env']})")
        print()

        print_step("2.3", "Creating threat hunting campaign")
        campaign_name = "Q1 2025 Ransomware Hunt - Production Fleet"
        print_info(f"   Campaign Name: {campaign_name}")
        print_info(f"   Target Selection: env=production")
        print_info(f"   Scan Types: filesystem, process, network")
        print_info(f"   YARA Rules: ransomware, trojan, backdoor")
        print_info(f"   Priority: HIGH")
        print()

        # Select production endpoints
        production_endpoints = [ep for ep in self.endpoints if ep['env'] == 'production']

        print_step("2.4", "Selecting target endpoints")
        print_info(f"   Matched {len(production_endpoints)} endpoints:")
        for ep in production_endpoints:
            print(f"      • {ep['id']}")
        print()

        print_step("2.5", "Creating scan tasks for each endpoint")
        simulate_progress("   Generating scan configurations", 1)

        for ep in production_endpoints:
            task_id = f"task_{random.randint(1000, 9999)}"
            print_info(f"   Task {task_id} → {ep['id']}")
        print()

        print_step("2.6", "Publishing scan tasks to Pub/Sub")
        simulate_progress("   Publishing to 'asgard-scan-tasks' topic", 2)
        print_success(f"Published {len(production_endpoints)} scan tasks")
        print()

        print_info("📊 Campaign Created:")
        print(f"   Campaign ID: campaign_{random.randint(10000, 99999)}")
        print(f"   Status: RUNNING")
        print(f"   Total Targets: {len(production_endpoints)}")
        print(f"   Scans Queued: {len(production_endpoints)}")

        input(f"\n{Colors.CYAN}Press ENTER to continue to THOR scans...{Colors.END}")

    def simulate_thor(self):
        """Simulate THOR endpoint scanning"""
        print_header("Phase 3: THOR - Endpoint Scanning")

        production_endpoints = [ep for ep in self.endpoints if ep['env'] == 'production']

        for i, endpoint in enumerate(production_endpoints, 1):
            print(f"\n{Colors.BOLD}--- Scanning Endpoint {i}/{len(production_endpoints)}: {endpoint['id']} ---{Colors.END}\n")

            print_step("3.1", f"THOR agent on {endpoint['id']} received scan task")
            simulate_progress("   Loading YARA rules from VALHALLA", 1)
            simulate_progress("   Loading IOCs from VALHALLA", 1)
            print()

            print_step("3.2", "Performing filesystem scan")
            files_to_scan = random.randint(5000, 15000)

            # Simulate scanning progress
            print(f"   Scanning {files_to_scan} files...", end="", flush=True)
            for _ in range(3):
                time.sleep(0.5)
                print(".", end="", flush=True)
            print(f" {Colors.GREEN}Done!{Colors.END}")

            # Simulate finding a threat on web-server-01
            if endpoint['id'] == 'web-server-01':
                print()
                print_warning("THREAT DETECTED!")

                detection = {
                    "endpoint": endpoint['id'],
                    "rule": "Ransomware_Lockbit_3_0",
                    "file": "/var/www/uploads/document_2025.exe",
                    "severity": "CRITICAL",
                    "confidence": 0.95,
                    "hash": "a1b2c3d4e5f6...",
                    "timestamp": datetime.now().isoformat()
                }

                self.detections.append(detection)

                print_info(f"   Rule: {detection['rule']}")
                print_info(f"   File: {detection['file']}")
                print_info(f"   Severity: {detection['severity']}")
                print_info(f"   Confidence: {detection['confidence']:.0%}")
                print_info(f"   Hash: {detection['hash']}")

            print()
            print_step("3.3", "Performing process scan")
            processes = random.randint(80, 150)
            simulate_progress(f"   Scanning {processes} running processes", 1)

            # Check for suspicious process on web-server-01
            if endpoint['id'] == 'web-server-01':
                print_warning("Suspicious process detected!")
                print_info("   Process: /tmp/.hidden/miner")
                print_info("   Pattern: Cryptocurrency miner behavior")
                print_info("   CPU Usage: 95%")

            print()
            print_step("3.4", "Performing network connection scan")
            connections = random.randint(20, 80)
            simulate_progress(f"   Scanning {connections} active connections", 1)

            # Check for malicious IP on web-server-01
            if endpoint['id'] == 'web-server-01':
                print_warning("Malicious IP connection detected!")

                network_detection = {
                    "endpoint": endpoint['id'],
                    "type": "network",
                    "remote_ip": "192.0.2.100",
                    "remote_port": 4444,
                    "severity": "CRITICAL",
                    "description": "Connection to known C2 server"
                }

                self.detections.append(network_detection)

                print_info(f"   Remote IP: {network_detection['remote_ip']}")
                print_info(f"   Remote Port: {network_detection['remote_port']}")
                print_info(f"   IOC Match: Known C2 server")

            print()
            print_step("3.5", "Publishing scan results to Pub/Sub")
            simulate_progress("   Publishing to 'thor-findings' topic", 1)

            # Store results
            self.scan_results[endpoint['id']] = {
                "files_scanned": files_to_scan,
                "processes_scanned": processes,
                "connections_scanned": connections,
                "threats_found": len([d for d in self.detections if d.get('endpoint') == endpoint['id']]),
                "status": "completed"
            }

            if endpoint['id'] == 'web-server-01':
                print_error(f"Scan completed: {self.scan_results[endpoint['id']]['threats_found']} THREATS FOUND")
            else:
                print_success("Scan completed: No threats detected")

            time.sleep(1)

        print()
        print_info("📊 THOR Scan Summary:")
        print(f"   Total Endpoints Scanned: {len(production_endpoints)}")
        print(f"   Total Files Scanned: {sum(r['files_scanned'] for r in self.scan_results.values())}")
        print(f"   Total Threats Found: {len(self.detections)}")
        print(f"   Critical Threats: {len([d for d in self.detections if d.get('severity') == 'CRITICAL'])}")

        input(f"\n{Colors.CYAN}Press ENTER to continue to TAA analysis...{Colors.END}")

    def simulate_taa(self):
        """Simulate TAA (Triage and Analysis Agent)"""
        print_header("Phase 4: TAA - Triage & Analysis")

        if not self.detections:
            print_info("No threats detected, skipping TAA analysis")
            return

        for i, detection in enumerate(self.detections, 1):
            if 'file' not in detection:
                continue

            print(f"\n{Colors.BOLD}--- Analyzing Detection {i}/{len(self.detections)} ---{Colors.END}\n")

            print_step("4.1", "TAA received threat finding from THOR")
            print_info(f"   Endpoint: {detection['endpoint']}")
            print_info(f"   Threat: {detection['rule']}")
            print_info(f"   Severity: {detection['severity']}")
            print()

            print_step("4.2", "Extracting IOCs from finding")
            simulate_progress("   Extracting file hash, IPs, domains", 1)
            print_info(f"   File Hash: {detection['hash']}")
            print_info(f"   Associated IP: 192.0.2.100")
            print()

            print_step("4.3", "Enriching with VirusTotal")
            simulate_progress("   Querying VirusTotal API", 2)
            print_info("   VirusTotal Results:")
            print(f"      • Detection Rate: 58/72 (81%)")
            print(f"      • First Seen: 2024-12-15")
            print(f"      • Last Analysis: 2025-01-01")
            print(f"      • Classification: Ransomware.Lockbit.Gen")
            print()

            print_step("4.4", "Enriching with AbuseIPDB")
            simulate_progress("   Querying AbuseIPDB for 192.0.2.100", 2)
            print_info("   AbuseIPDB Results:")
            print(f"      • Abuse Confidence: 95%")
            print(f"      • Total Reports: 342")
            print(f"      • Categories: Malware Distribution, C2 Server")
            print(f"      • Country: Russia")
            print()

            print_step("4.5", "LLM-based threat analysis (Gemini)")
            simulate_progress("   Sending context to Gemini Pro", 2)

            llm_analysis = """
   LLM Analysis:

   This appears to be a Lockbit 3.0 ransomware infection with high confidence.

   Key Indicators:
   • File hash matches known Lockbit 3.0 variant
   • Active C2 communication to known malicious IP
   • Suspicious process in /tmp suggests persistence mechanism
   • Upload directory suggests initial access via web application vulnerability

   Attack Timeline:
   1. Initial access via vulnerable web upload (likely file upload bypass)
   2. Ransomware payload dropped in /tmp
   3. Persistence established via hidden process
   4. C2 communication initiated for command and control
   5. Currently in reconnaissance phase (high CPU = network scanning)

   Recommended Actions:
   1. IMMEDIATE: Isolate web-server-01 from network
   2. Block IP 192.0.2.100 at firewall
   3. Terminate suspicious process
   4. Check for lateral movement to web-server-02 and db-server-01
   5. Restore from clean backup if encryption detected
            """

            print(f"{Colors.YELLOW}{llm_analysis}{Colors.END}")
            print()

            print_step("4.6", "Calculating threat score")
            simulate_progress("   Multi-dimensional threat scoring", 1)

            threat_score = {
                "total_score": 9.2,
                "attack_category_score": 10.0,
                "severity_score": 10.0,
                "confidence_score": 9.5,
                "ip_reputation_score": 9.5,
                "behavioral_score": 8.5,
                "classification": "TRUE_POSITIVE",
                "recommended_priority": "CRITICAL"
            }

            print_info(f"   Threat Score: {threat_score['total_score']}/10")
            print_info(f"   Classification: {threat_score['classification']}")
            print_info(f"   Priority: {threat_score['recommended_priority']}")
            print()

            print_step("4.7", "Publishing enriched finding to CRA")
            simulate_progress("   Publishing to 'taa-findings' topic", 1)
            print_success("Enriched finding sent to CRA for automated response")

            # Store enrichment
            detection['taa_analysis'] = threat_score
            detection['llm_recommendation'] = "Immediate isolation and containment"

        print()
        print_info("📊 TAA Analysis Summary:")
        print(f"   Findings Analyzed: {len(self.detections)}")
        print(f"   True Positives: {len([d for d in self.detections if d.get('taa_analysis', {}).get('classification') == 'TRUE_POSITIVE'])}")
        print(f"   Critical Priority: {len([d for d in self.detections if d.get('taa_analysis', {}).get('recommended_priority') == 'CRITICAL'])}")

        input(f"\n{Colors.CYAN}Press ENTER to continue to CRA response...{Colors.END}")

    def simulate_cra(self):
        """Simulate CRA (Containment and Response Agent)"""
        print_header("Phase 5: CRA - Containment & Response")

        critical_detections = [d for d in self.detections
                              if d.get('taa_analysis', {}).get('recommended_priority') == 'CRITICAL']

        if not critical_detections:
            print_info("No critical threats requiring automated response")
            return

        for i, detection in enumerate(critical_detections, 1):
            if 'file' not in detection:
                continue

            print(f"\n{Colors.BOLD}--- Automated Response {i}/{len(critical_detections)} ---{Colors.END}\n")

            print_step("5.1", "CRA received critical threat from TAA")
            print_info(f"   Endpoint: {detection['endpoint']}")
            print_info(f"   Threat Score: {detection['taa_analysis']['total_score']}/10")
            print_info(f"   Priority: CRITICAL")
            print()

            print_step("5.2", "Selecting response playbook")
            simulate_progress("   Matching threat to playbook", 1)
            print_info("   Selected Playbook: 'Ransomware Containment'")
            print()

            print_step("5.3", "Executing automated response actions")
            print()

            # Action 1: Isolate endpoint
            print(f"   {Colors.BOLD}Action 1: Isolate Endpoint{Colors.END}")
            simulate_progress("      Creating GCP firewall deny rule", 2)
            print_success(f"      Endpoint {detection['endpoint']} isolated from network")
            print_info("      Firewall Rule: deny-web-server-01-all (priority: 1)")
            print()

            # Action 2: Block malicious IP
            print(f"   {Colors.BOLD}Action 2: Block Malicious IP{Colors.END}")
            simulate_progress("      Adding IP to firewall blocklist", 1)
            print_success("      Blocked IP 192.0.2.100 across all firewalls")
            print_info("      Applied to: GCP, AWS, Palo Alto firewalls")
            print()

            # Action 3: Terminate malicious process
            print(f"   {Colors.BOLD}Action 3: Terminate Malicious Process{Colors.END}")
            simulate_progress("      Sending kill command to endpoint", 1)
            print_success("      Process /tmp/.hidden/miner terminated")
            print()

            # Action 4: Quarantine file
            print(f"   {Colors.BOLD}Action 4: Quarantine Malicious File{Colors.END}")
            simulate_progress("      Moving file to quarantine", 1)
            print_success(f"      File quarantined: {detection['file']}")
            print_info("      Quarantine location: /quarantine/2025-01-01/ransomware/")
            print()

            # Action 5: Create incident ticket
            print(f"   {Colors.BOLD}Action 5: Create Incident Ticket{Colors.END}")
            simulate_progress("      Creating Jira ticket", 1)
            ticket_id = f"INCIDENT-{random.randint(2025, 2099)}"
            print_success(f"      Created ticket: {ticket_id}")
            print_info(f"      Assigned to: SOC Team")
            print_info(f"      Priority: Critical")
            print_info(f"      URL: https://jira.company.com/browse/{ticket_id}")
            print()

            # Action 6: Alert on-call
            print(f"   {Colors.BOLD}Action 6: Alert On-Call Analyst{Colors.END}")
            simulate_progress("      Sending PagerDuty alert", 1)
            print_success("      PagerDuty incident created")
            print_info("      Notified: soc-oncall@company.com")
            print()

            # Action 7: Collect forensics
            print(f"   {Colors.BOLD}Action 7: Collect Forensic Data{Colors.END}")
            simulate_progress("      Capturing memory dump", 2)
            simulate_progress("      Creating disk snapshot", 2)
            print_success("      Forensic data collected")
            print_info("      Memory dump: /forensics/web-server-01-mem-20250101.dump")
            print_info("      Disk snapshot: snap-web-server-01-20250101")
            print()

            print_step("5.4", "Updating incident in Firestore")
            simulate_progress("   Storing response actions", 1)
            print_success("Incident response completed and documented")

        print()
        print_info("📊 CRA Response Summary:")
        print(f"   Incidents Handled: {len(critical_detections)}")
        print(f"   Endpoints Isolated: 1 (web-server-01)")
        print(f"   IPs Blocked: 1 (192.0.2.100)")
        print(f"   Processes Terminated: 1")
        print(f"   Files Quarantined: 1")
        print(f"   Tickets Created: 1 (INCIDENT-2025)")
        print(f"   Forensics Collected: Memory dump + Disk snapshot")
        print()
        print_success("🛡️  Threat contained successfully!")

        input(f"\n{Colors.CYAN}Press ENTER to continue to CLA learning...{Colors.END}")

    def simulate_cla(self):
        """Simulate CLA (Continuous Learning Agent)"""
        print_header("Phase 6: CLA - Continuous Learning")

        print_step("6.1", "CLA collecting feedback from all agents")
        simulate_progress("   Aggregating data from THOR scans", 1)
        simulate_progress("   Aggregating data from TAA analysis", 1)
        simulate_progress("   Aggregating data from CRA actions", 1)
        print()

        print_step("6.2", "Analyzing detection patterns")
        simulate_progress("   Processing threat vectors", 2)

        print_info("   Pattern Analysis:")
        print("      • Ransomware via web upload vulnerability (NEW PATTERN)")
        print("      • C2 communication to Russian IPs (KNOWN PATTERN)")
        print("      • Persistence via /tmp hidden processes (EMERGING PATTERN)")
        print()

        print_step("6.3", "Updating anomaly detection models")
        simulate_progress("   Retraining Isolation Forest model", 2)
        simulate_progress("   Updating feature importance weights", 1)

        print_info("   Model Updates:")
        print("      • Added 'file upload to /var/www' as high-risk feature")
        print("      • Increased weight for /tmp process creation")
        print("      • Added Russian IP ranges to geo-risk scoring")
        print_success("   ADA model updated successfully")
        print()

        print_step("6.4", "Identifying gaps in YARA rules")
        simulate_progress("   Analyzing missed detections", 1)

        print_info("   Rule Recommendations:")
        print("      • Recommend new YARA rule for Lockbit 3.0 variant (Dec 2024)")
        print("      • Suggest Sigma rule for suspicious /tmp process creation")
        print("      • Propose behavioral rule for rapid file upload patterns")
        print()

        print_step("6.5", "Sending recommendations to VALHALLA")
        simulate_progress("   Publishing rule recommendations", 1)
        print_success("Recommendations sent to VALHALLA for rule update")
        print()

        print_step("6.6", "Calculating performance improvements")
        simulate_progress("   Analyzing metrics over 30-day window", 2)

        print_info("   Performance Metrics:")
        print("      • Detection Rate: 85% → 92% (+7%)")
        print("      • False Positive Rate: 12% → 8% (-4%)")
        print("      • Mean Time to Detect: 25 min → 18 min (-28%)")
        print("      • Mean Time to Respond: 45 min → 15 min (-67%)")
        print()

        print_step("6.7", "Storing learning outcomes")
        simulate_progress("   Updating model registry", 1)
        simulate_progress("   Storing in BigQuery", 1)
        print_success("Learning cycle completed")

        print()
        print_info("📊 CLA Learning Summary:")
        print(f"   New Patterns Identified: 1")
        print(f"   Model Updates: 1 (ADA Isolation Forest)")
        print(f"   Rule Recommendations: 3 (YARA + Sigma)")
        print(f"   Performance Improvement: +7% detection, -28% MTTD")
        print()
        print_success("🧠 Platform is learning and improving!")

        input(f"\n{Colors.CYAN}Press ENTER to view final summary...{Colors.END}")

    def show_summary(self):
        """Show final summary of the simulation"""
        print_header("🎯 SIMULATION COMPLETE - SUMMARY")

        print(f"{Colors.BOLD}End-to-End Threat Hunting Workflow Demonstrated:{Colors.END}\n")

        # Phase summaries
        phases = [
            {
                "phase": "1. VALHALLA",
                "description": "Threat Intelligence Feed Manager",
                "accomplishments": [
                    f"Fetched {len(self.threat_intel['iocs']['ips']) + len(self.threat_intel['iocs']['domains']) + len(self.threat_intel['iocs']['hashes'])} IOCs from free sources",
                    f"Downloaded {len(self.threat_intel['yara_rules'])} YARA rule sets",
                    "Compiled and validated all rules",
                    "Distributed to THOR agents"
                ]
            },
            {
                "phase": "2. ASGARD",
                "description": "Fleet Orchestration",
                "accomplishments": [
                    f"Discovered {len(self.endpoints)} endpoints",
                    f"Created ransomware hunting campaign",
                    f"Deployed {len([ep for ep in self.endpoints if ep['env'] == 'production'])} scan tasks",
                    "Coordinated fleet-wide threat hunt"
                ]
            },
            {
                "phase": "3. THOR",
                "description": "Endpoint Scanning",
                "accomplishments": [
                    f"Scanned {sum(r['files_scanned'] for r in self.scan_results.values())} files",
                    f"Scanned {sum(r['processes_scanned'] for r in self.scan_results.values())} processes",
                    f"Detected {len(self.detections)} threats",
                    "Identified Lockbit 3.0 ransomware + C2 communication"
                ]
            },
            {
                "phase": "4. TAA",
                "description": "Triage & Analysis",
                "accomplishments": [
                    "Enriched findings with VirusTotal + AbuseIPDB",
                    "LLM-based threat analysis (Gemini)",
                    "Calculated threat scores (9.2/10)",
                    "Classified as TRUE_POSITIVE with CRITICAL priority"
                ]
            },
            {
                "phase": "5. CRA",
                "description": "Containment & Response",
                "accomplishments": [
                    "Isolated infected endpoint",
                    "Blocked malicious C2 IP",
                    "Terminated malicious process",
                    "Quarantined ransomware file",
                    "Created incident ticket",
                    "Alerted on-call analyst",
                    "Collected forensic evidence"
                ]
            },
            {
                "phase": "6. CLA",
                "description": "Continuous Learning",
                "accomplishments": [
                    "Identified new attack pattern",
                    "Updated ML models (+7% detection rate)",
                    "Recommended 3 new detection rules",
                    "Reduced MTTD by 28%, MTTR by 67%"
                ]
            }
        ]

        for phase_info in phases:
            print(f"{Colors.CYAN}{Colors.BOLD}{phase_info['phase']}: {phase_info['description']}{Colors.END}")
            for accomplishment in phase_info['accomplishments']:
                print(f"   ✅ {accomplishment}")
            print()

        print(f"\n{Colors.BOLD}Key Metrics from This Simulation:{Colors.END}\n")

        metrics = [
            ("Total Endpoints Scanned", len([ep for ep in self.endpoints if ep['env'] == 'production'])),
            ("Total Files Scanned", sum(r['files_scanned'] for r in self.scan_results.values())),
            ("Threats Detected", len(self.detections)),
            ("Critical Threats", len([d for d in self.detections if d.get('severity') == 'CRITICAL'])),
            ("True Positives", len([d for d in self.detections if d.get('taa_analysis', {}).get('classification') == 'TRUE_POSITIVE'])),
            ("Automated Response Actions", 7),
            ("Time from Detection to Containment", "15 minutes"),
            ("Model Performance Improvement", "+7% detection rate")
        ]

        for metric, value in metrics:
            print(f"   {Colors.BOLD}{metric}:{Colors.END} {value}")

        print(f"\n{Colors.BOLD}Cost Comparison:{Colors.END}\n")

        cost_comparison = [
            ("Platform Type", "Pure Platform", "Hybrid", "Full Nextron"),
            ("3-Year Cost", "$372,000", "$427,000", "$740K-$960K"),
            ("Annual Cost", "$124,000", "$142,000", "$247K-$320K"),
            ("Savings vs Nextron", "61-72%", "63-74%", "Baseline"),
            ("YARA Rules", "~5,000", "~20,000", "~15,000"),
            ("AI/ML Capabilities", "✅ Full", "✅ Full", "❌ None"),
            ("Automation", "✅ Full", "✅ Full", "⚠️ Limited"),
        ]

        # Print table
        col_widths = [30, 15, 15, 18]

        # Header
        header = cost_comparison[0]
        print("   " + "".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(header)))
        print("   " + "-" * sum(col_widths))

        # Data rows
        for row in cost_comparison[1:]:
            print("   " + "".join(f"{cell:<{col_widths[i]}}" for i, cell in enumerate(row)))

        print(f"\n{Colors.BOLD}What Your Team Just Witnessed:{Colors.END}\n")

        witnessed = [
            "Complete automated threat hunting workflow",
            "Integration of 6 specialized agents working together",
            "AI/ML-driven threat analysis and decision-making",
            "Automated containment and response (no human intervention needed)",
            "Continuous learning and self-improvement",
            "Enterprise-grade capabilities at 61-72% cost savings"
        ]

        for item in witnessed:
            print(f"   🎯 {item}")

        print(f"\n{Colors.BOLD}Next Steps:{Colors.END}\n")

        next_steps = [
            "Review documentation (Pure_Platform_Deployment_Guide.docx)",
            "Decide deployment model (Pure vs. Hybrid vs. Full Nextron)",
            "Test on your GCP VM (follow QUICK_START_VM.md)",
            "Run POC on 10-50 production endpoints",
            "Measure detection rate and false positives",
            "Calculate ROI for your organization",
            "Deploy to full production fleet"
        ]

        for i, step in enumerate(next_steps, 1):
            print(f"   {i}. {step}")

        print(f"\n{Colors.GREEN}{Colors.BOLD}{'='*70}{Colors.END}")
        print(f"{Colors.GREEN}{Colors.BOLD}Thank you for watching the simulation!{Colors.END}".center(70))
        print(f"{Colors.GREEN}{Colors.BOLD}{'='*70}{Colors.END}\n")


def main():
    """Main entry point"""
    simulation = ThreatHuntingSimulation()

    try:
        simulation.run_simulation()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Simulation interrupted by user{Colors.END}")
        print("You can run the simulation again anytime with: python3 demo_simulation.py")
    except Exception as e:
        print(f"\n{Colors.RED}Error during simulation: {e}{Colors.END}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
