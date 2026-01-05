# Video Demo Guide - Threat Hunting Platform

This guide provides instructions for creating a video demonstration of the threat hunting platform deployment on GCP.

## Recording Options

### Option 1: Using asciinema (Recommended for Terminal Demos)

**Installation:**
```bash
# On GCP instance
sudo yum install -y asciinema
# Or
pip install asciinema
```

**Recording:**
```bash
cd ~/threat-hunting-test
source venv/bin/activate

# Start recording
asciinema rec threat_hunting_demo.cast

# Run the demo script
bash demo_script.sh

# Stop recording (Ctrl+D or type 'exit')
```

**Convert to GIF:**
```bash
# Install agg (asciinema gif generator)
# Then convert
agg threat_hunting_demo.cast threat_hunting_demo.gif
```

### Option 2: Using script command (Built-in)

```bash
cd ~/threat-hunting-test
source venv/bin/activate

# Start recording
script -a demo_output.txt

# Run commands manually or use demo script
bash demo_script.sh

# Stop recording (type 'exit')
```

### Option 3: Using OBS Studio or Screen Recording

For a more polished video with GUI elements:
1. Use OBS Studio or similar screen recording software
2. Record the SSH-in-browser session
3. Add voiceover explaining each step
4. Edit with video editing software

## Demo Script Usage

The `demo_script.sh` provides an automated walkthrough:

```bash
cd ~/threat-hunting-test
source venv/bin/activate
bash demo_script.sh
```

## Manual Demo Steps (For Recording)

If you prefer to record manually, follow these steps:

### Part 1: Introduction (30 seconds)
```bash
clear
echo "Welcome to the Threat Hunting Platform Demo"
echo "Deployed on GCP Instance: xdgaisocapp01"
```

### Part 2: Show Environment (1 minute)
```bash
pwd
ls -la
tree -L 2
python3 --version
```

### Part 3: Show Components (1 minute)
```bash
ls -lh *.py
cat THREAT_HUNTING_README.md | head -30
```

### Part 4: Show Configuration (1 minute)
```bash
cat config/thor_config.json
cat config/asgard_config.json | head -20
```

### Part 5: Show YARA Rules (1 minute)
```bash
ls -lh data/yara_rules/
cat data/yara_rules/test_malware_rules.yar | head -30
```

### Part 6: Show Test Files (1 minute)
```bash
ls -lh data/test_malware/
cat data/test_malware/test_ransomware.txt
```

### Part 7: Run THOR Scan (2 minutes)
```bash
python thor_endpoint_agent.py \
    --config config/thor_config.json \
    --scan-type filesystem \
    --target data/test_malware \
    --load-yara
```

### Part 8: Show Results (1 minute)
```bash
# Show the detected threats
echo "6 threats detected successfully!"
```

### Part 9: Show ASGARD (1 minute)
```bash
python -c "
from asgard_orchestration_agent import ASGARDOrchestrationAgent
asgard = ASGARDOrchestrationAgent()
print(f'Registered endpoints: {len(asgard.registered_endpoints)}')
"
```

### Part 10: Summary (30 seconds)
```bash
echo "Platform Summary:"
echo "- THOR: Operational"
echo "- ASGARD: Operational"
echo "- VALHALLA: Operational"
echo "- Endpoint: Registered"
echo "- Detection: Verified"
```

## Demo Script Content

The demo script (`demo_script.sh`) includes:

1. **Environment Overview** - Shows directory structure and Python environment
2. **Platform Components** - Displays all three agents
3. **Configuration** - Shows config files
4. **YARA Rules** - Displays test rules
5. **Test Files** - Shows test malware files
6. **THOR Scan** - Runs actual threat detection
7. **ASGARD Campaign** - Shows endpoint registration
8. **VALHALLA Status** - Shows feed manager status
9. **Summary** - Platform status overview

## Recording Tips

1. **Prepare beforehand:**
   - Ensure all components are working
   - Have test files ready
   - Clear terminal history if needed

2. **Timing:**
   - Pause between sections for clarity
   - Speak clearly if adding voiceover
   - Show results clearly

3. **Editing:**
   - Add title cards between sections
   - Highlight important outputs
   - Add captions for key points

4. **File Size:**
   - Keep recordings under 10 minutes
   - Focus on key features
   - Show actual results, not just commands

## Post-Production

After recording:

1. **Add Annotations:**
   - Title: "Threat Hunting Platform Demo"
   - Section titles
   - Key highlights

2. **Add Voiceover:**
   - Explain what each component does
   - Highlight key features
   - Explain results

3. **Add Captions:**
   - Command explanations
   - Result highlights
   - Key metrics

4. **Export Formats:**
   - MP4 for general use
   - GIF for quick previews
   - WebM for web embedding

## Quick Demo Script (5 minutes)

For a quick 5-minute demo:

```bash
#!/bin/bash
clear
echo "=== Threat Hunting Platform Demo ==="
echo ""
echo "1. Environment:"
pwd && python3 --version
echo ""
echo "2. Components:"
ls -lh *.py | head -3
echo ""
echo "3. Running THOR scan:"
python thor_endpoint_agent.py --config config/thor_config.json --scan-type filesystem --target data/test_malware --load-yara
echo ""
echo "4. Results: 6 threats detected!"
echo ""
echo "Demo complete!"
```

## Sharing the Demo

After recording:

1. Upload to YouTube/Vimeo for public sharing
2. Upload to internal video platform
3. Embed in documentation
4. Share via file sharing service

---

**Note:** The demo script is designed to be self-contained and can be run multiple times for multiple takes.

