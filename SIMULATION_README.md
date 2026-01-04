# 🎬 Threat Hunting Platform - Interactive Simulation

## Overview

This simulation demonstrates the complete threat hunting workflow **WITHOUT** requiring any GCP resources, deployment, or configuration. Perfect for team presentations and understanding how the platform works.

---

## 🎯 What the Simulation Shows

The simulation walks through a realistic scenario:

1. **VALHALLA** fetches threat intelligence (IOCs, YARA rules)
2. **ASGARD** creates a ransomware hunting campaign across production servers
3. **THOR** scans each endpoint and detects Lockbit 3.0 ransomware on web-server-01
4. **TAA** enriches the finding with VirusTotal, AbuseIPDB, and LLM analysis
5. **CRA** automatically contains the threat (isolate endpoint, block IP, quarantine file)
6. **CLA** learns from the incident and improves detection models

**Total Runtime:** ~10-15 minutes (interactive) or ~3 minutes (auto mode)

---

## 🚀 How to Run

### Option 1: Interactive Mode (Recommended for Presentations)

```bash
python3 demo_simulation.py
```

**What happens:**
- Simulation pauses at each phase for you to explain
- Press ENTER to continue to next phase
- Color-coded output shows what each agent does
- Perfect for team walkthroughs

**Example flow:**
```
[Phase 1: VALHALLA]
• Shows fetching IOCs from ThreatFox, MalwareBazaar
• Displays YARA rules downloaded
Press ENTER →

[Phase 2: ASGARD]
• Shows campaign creation
• Lists target endpoints
Press ENTER →

[Phase 3: THOR]
• Simulates scanning each endpoint
• Detects ransomware on web-server-01
Press ENTER →

... and so on
```

---

### Option 2: Auto Mode (Non-Interactive)

For automated demonstrations or CI/CD:

```bash
# Run with automatic progression
python3 demo_simulation.py --auto
```

Or use input redirection:

```bash
# Simulates pressing ENTER at each pause
yes "" | python3 demo_simulation.py
```

---

## 📊 What You'll See

### Phase 1: VALHALLA - Threat Intelligence Feed Manager
```
✅ Fetching IOCs from ThreatFox...
ℹ️  Fetched 3 malicious IPs
ℹ️  Fetched 2 malicious domains
✅ Downloaded 5 YARA rule sets:
   • Ransomware_Lockbit_3_0
   • Trojan_Emotet_Loader
   • APT_CobaltStrike_Beacon
   ...
```

### Phase 2: ASGARD - Fleet Orchestration
```
✅ Discovered 5 endpoints
✅ Creating campaign: "Q1 2025 Ransomware Hunt"
ℹ️  Target: env=production (4 endpoints)
✅ Published 4 scan tasks
```

### Phase 3: THOR - Endpoint Scanning
```
--- Scanning web-server-01 ---
✅ Scanned 12,450 files
⚠️  THREAT DETECTED!
   Rule: Ransomware_Lockbit_3_0
   File: /var/www/uploads/document_2025.exe
   Severity: CRITICAL
   Confidence: 95%

⚠️  Malicious IP connection detected!
   Remote IP: 192.0.2.100
   IOC Match: Known C2 server
```

### Phase 4: TAA - Triage & Analysis
```
✅ Enriched with VirusTotal
   Detection Rate: 58/72 (81%)
   Classification: Ransomware.Lockbit.Gen

✅ Enriched with AbuseIPDB
   Abuse Confidence: 95%
   Categories: C2 Server, Malware Distribution

✅ LLM Analysis (Gemini):
   "High confidence Lockbit 3.0 infection.
    Recommend immediate isolation..."

✅ Threat Score: 9.2/10
   Classification: TRUE_POSITIVE
   Priority: CRITICAL
```

### Phase 5: CRA - Containment & Response
```
🛡️  Executing Ransomware Containment Playbook:

✅ Action 1: Isolated web-server-01 from network
✅ Action 2: Blocked IP 192.0.2.100 at firewall
✅ Action 3: Terminated malicious process
✅ Action 4: Quarantined ransomware file
✅ Action 5: Created incident ticket: INCIDENT-2025
✅ Action 6: Alerted on-call analyst (PagerDuty)
✅ Action 7: Collected forensic evidence

🛡️  Threat contained in 15 minutes!
```

### Phase 6: CLA - Continuous Learning
```
🧠 Analyzing detection patterns...
✅ Identified new attack pattern
✅ Updated ML models
   Performance: +7% detection rate
   MTTD: -28% (25 min → 18 min)
   MTTR: -67% (45 min → 15 min)

✅ Recommended 3 new detection rules to VALHALLA
```

---

## 🎭 Scenario Details

### The Simulated Attack

**Target:** Production web server (web-server-01)

**Attack Vector:** File upload vulnerability

**Malware:** Lockbit 3.0 Ransomware

**Timeline:**
1. Attacker uploads malicious .exe via vulnerable upload form
2. Ransomware drops persistence mechanism in /tmp
3. Establishes C2 connection to 192.0.2.100 (Russian IP)
4. Begins reconnaissance (network scanning)
5. **THOR detects before encryption begins** ✅

### Why This Server Was Vulnerable
- Web upload form lacked file type validation
- Running outdated web framework
- Insufficient monitoring on /var/www/uploads/

### How the Platform Stopped It
1. YARA rule detected Lockbit signature in uploaded file
2. IOC match flagged known C2 IP connection
3. Behavioral analysis caught suspicious /tmp process
4. Platform responded faster than attacker (15 min vs typical 45 min)

---

## 📈 Key Metrics Shown

| Metric | Value |
|--------|-------|
| Endpoints Scanned | 4 (production servers) |
| Files Scanned | ~45,000 total |
| Threats Detected | 2 (ransomware + C2 connection) |
| Time to Detection | 18 minutes |
| Time to Containment | 15 minutes |
| Automated Actions | 7 (no human intervention) |
| Performance Improvement | +7% detection, -28% MTTD |

---

## 💰 Cost Comparison (Shown at End)

The simulation concludes with a cost comparison:

| Deployment | 3-Year Cost | Annual Cost | Savings |
|-----------|-------------|-------------|---------|
| **Pure Platform** | $372,000 | $124,000 | 61-72% |
| **Hybrid** | $427,000 | $142,000 | 63-74% |
| **Full Nextron** | $740K-$960K | $247K-$320K | Baseline |

---

## 🎓 Best Practices for Team Presentation

### 1. Pre-Presentation Setup (2 minutes)
```bash
# Test the simulation once
python3 demo_simulation.py

# Press ENTER through each phase to familiarize yourself
```

### 2. During Presentation (15 minutes)
```bash
# Run in full terminal window for better visibility
python3 demo_simulation.py
```

**Narration Tips:**

**Phase 1 (VALHALLA):**
> "First, VALHALLA fetches the latest threat intelligence from free sources like ThreatFox and MalwareBazaar. We're getting real IOCs and YARA rules that security researchers worldwide have shared."

**Phase 2 (ASGARD):**
> "ASGARD is our orchestrator. It discovers our endpoints and creates a campaign to hunt for ransomware across all production servers."

**Phase 3 (THOR):**
> "THOR agents scan each endpoint. Watch what happens when we scan web-server-01... [pause for detection] ...and there's our threat! Lockbit 3.0 ransomware."

**Phase 4 (TAA):**
> "TAA doesn't just detect—it analyzes. It checks VirusTotal, AbuseIPDB, and even uses Gemini AI to understand the threat context."

**Phase 5 (CRA):**
> "Here's where automation shines. CRA executes our containment playbook—isolation, blocking, quarantine—all automatically, in 15 minutes."

**Phase 6 (CLA):**
> "Finally, CLA learns from this incident. It updates our models, recommends new rules, and makes the platform smarter for next time."

### 3. Q&A Points

**Common Questions:**

Q: *"Is this real or simulated?"*
A: "The workflow is real—these are the actual agent interactions. The detection is simulated so we can demonstrate safely without needing malware."

Q: *"Does it really work this fast?"*
A: "Yes. The 15-minute response time is realistic. Traditional SOCs take 45-90 minutes for the same workflow."

Q: *"What if there are false positives?"*
A: "TAA's LLM analysis and multi-source enrichment help filter false positives. Plus, CLA learns from analyst feedback to reduce them over time."

Q: *"How much does this cost?"*
A: "For 1,000 endpoints over 3 years: $372K with free threat intel, or $427K with premium Nextron VALHALLA. Compare that to $740K-$960K for full Nextron."

---

## 🔧 Customization

### Modify the Scenario

Edit `demo_simulation.py` to customize:

```python
# Change the malware detected
detection = {
    "rule": "Your_Custom_Rule_Name",
    "file": "/your/custom/path",
    "severity": "CRITICAL",
    ...
}

# Add more endpoints
self.endpoints.append({
    "id": "your-server",
    "role": "application",
    "env": "production",
    "os": "linux"
})

# Modify threat intelligence
self.threat_intel['yara_rules'].append("Your_Custom_Rule")
```

### Change Timing

Speed up or slow down the simulation:

```python
# Find this line in simulate_progress()
time.sleep(0.5)  # Change to 0.1 for faster, 1.0 for slower
```

---

## 📹 Recording the Simulation

### For Team Training Videos

```bash
# Install asciinema (terminal recorder)
brew install asciinema  # macOS
# or
sudo apt install asciinema  # Linux

# Record the simulation
asciinema rec threat-hunting-demo.cast

# Run simulation
python3 demo_simulation.py

# Press Ctrl+D when done

# Play back
asciinema play threat-hunting-demo.cast

# Or upload to asciinema.org for sharing
asciinema upload threat-hunting-demo.cast
```

---

## 🎯 Learning Objectives

After watching this simulation, your team should understand:

✅ **What each agent does:**
- VALHALLA = Threat intel aggregation
- ASGARD = Campaign orchestration
- THOR = Endpoint scanning
- TAA = Enrichment & analysis
- CRA = Automated response
- CLA = Continuous learning

✅ **How agents work together:**
- Data flows from VALHALLA → THOR → TAA → CRA → CLA
- Each agent specializes, but they're orchestrated as one platform

✅ **Why this is better than manual SOC:**
- Faster: 15 min vs 45-90 min response time
- Smarter: AI/ML analysis beats rule-based alone
- Cheaper: 61-72% cost savings vs commercial tools
- Scalable: Cloud-native, unlimited endpoints

✅ **The business value:**
- Reduced risk (faster threat containment)
- Lower costs (no licensing fees)
- Better outcomes (continuous learning improves over time)

---

## 🚀 Next Steps After Simulation

1. **Read Documentation**
   - `Pure_Platform_Deployment_Guide.docx`
   - `Hybrid_Deployment_Guide.docx`
   - `Decision_Guide.docx`

2. **Test on GCP VM**
   - Follow `QUICK_START_VM.md`
   - Run real scans on your development VM

3. **Plan POC**
   - Identify 10-50 endpoints for pilot
   - Define success metrics
   - Set 30-60 day timeline

4. **Decide Deployment Model**
   - Pure Platform (free intel)
   - Hybrid (Nextron VALHALLA + AI-SOC)
   - Based on budget and requirements

5. **Production Deployment**
   - Follow full deployment guide
   - Integrate with existing SOC
   - Train team on operation

---

## 🆘 Troubleshooting

### Issue: Colors not showing

**Fix:** Use a terminal that supports ANSI colors
```bash
# macOS/Linux: Use default Terminal or iTerm2
# Windows: Use Windows Terminal or Git Bash
```

### Issue: Simulation too fast/slow

**Fix:** Edit timing in `demo_simulation.py`
```python
def simulate_progress(task: str, duration: int = 2):
    # Change duration parameter to adjust speed
```

### Issue: Want to skip to specific phase

**Fix:** Comment out earlier phases in `run_simulation()`
```python
def run_simulation(self):
    # self.simulate_valhalla()  # Skip
    # self.simulate_asgard()     # Skip
    self.simulate_thor()         # Start here
```

---

## 📄 Related Files

- `demo_simulation.py` - The simulation script
- `threat_hunting_quickstart.py` - Real demo (requires GCP)
- `QUICK_START_VM.md` - VM testing guide
- `Pure_Platform_Deployment_Guide.docx` - Full deployment guide

---

**Ready to show your team how the platform works? Run the simulation!** 🎬

```bash
python3 demo_simulation.py
```
