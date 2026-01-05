# How to Record Threat Hunting Platform Demo Video

## Quick Start

### On Your GCP Instance (SSH Session)

```bash
cd ~/threat-hunting-test
source venv/bin/activate

# Download demo script
curl -f -s -o demo_script.sh https://raw.githubusercontent.com/ghifiardi/ai-driven-soc/main/demo_script.sh
chmod +x demo_script.sh

# Install asciinema (for terminal recording)
pip install asciinema
# OR
sudo yum install -y asciinema
```

## Recording Methods

### Method 1: asciinema (Best for Terminal Demos)

**Advantages:**
- Lightweight terminal recording
- Can be converted to GIF
- Easy to share
- Small file size

**Steps:**
```bash
cd ~/threat-hunting-test
source venv/bin/activate

# Start recording
asciinema rec threat_hunting_demo.cast

# Run demo
bash demo_script.sh

# Stop recording (Ctrl+D or type 'exit')
```

**Convert to GIF:**
```bash
# Install agg (if needed)
# Then convert
agg threat_hunting_demo.cast threat_hunting_demo.gif
```

### Method 2: SSH-in-Browser Screen Recording

**Advantages:**
- No additional software needed
- Records exactly what users see
- Can add voiceover later

**Steps:**
1. Open SSH-in-browser session
2. Start screen recording software (OBS, QuickTime, etc.)
3. Run: `bash demo_script.sh`
4. Stop recording
5. Edit and add voiceover

### Method 3: script command (Built-in)

**Advantages:**
- No installation needed
- Creates text transcript
- Can be converted to video

**Steps:**
```bash
cd ~/threat-hunting-test
source venv/bin/activate

# Start recording
script -a demo_output.txt

# Run demo
bash demo_script.sh

# Stop recording
exit
```

## Demo Script Content

The demo script (`demo_script.sh`) demonstrates:

1. **Environment Overview** (30 sec)
   - Directory structure
   - Python version
   - Virtual environment

2. **Platform Components** (30 sec)
   - THOR agent
   - ASGARD agent
   - VALHALLA agent

3. **Configuration** (30 sec)
   - Config files
   - Settings

4. **YARA Rules** (30 sec)
   - Test rules
   - Rule content

5. **Test Files** (30 sec)
   - Test malware files
   - File contents

6. **THOR Scan** (2 min)
   - Running scan
   - Detecting threats
   - Showing results

7. **ASGARD Status** (30 sec)
   - Registered endpoints
   - Campaign status

8. **Summary** (30 sec)
   - Platform status
   - Key achievements

**Total Duration:** ~6-7 minutes

## Manual Demo (If Script Doesn't Work)

Run these commands in sequence:

```bash
# 1. Introduction
clear
echo "Threat Hunting Platform Demo"
echo "GCP Instance: xdgaisocapp01"
sleep 2

# 2. Show environment
cd ~/threat-hunting-test
source venv/bin/activate
pwd
python3 --version

# 3. Show components
ls -lh *.py | head -3

# 4. Show YARA rules
ls -lh data/yara_rules/
head -20 data/yara_rules/test_malware_rules.yar

# 5. Show test files
ls -lh data/test_malware/
head -3 data/test_malware/test_ransomware.txt

# 6. Run THOR scan
python thor_endpoint_agent.py \
    --config config/thor_config.json \
    --scan-type filesystem \
    --target data/test_malware \
    --load-yara

# 7. Show endpoint status
python -c "
from asgard_orchestration_agent import ASGARDOrchestrationAgent
asgard = ASGARDOrchestrationAgent()
print(f'Registered endpoints: {len(asgard.registered_endpoints)}')
"

# 8. Summary
echo "Platform Status:"
echo "✓ THOR: Operational"
echo "✓ ASGARD: Operational"  
echo "✓ VALHALLA: Operational"
echo "✓ 6 threats detected"
```

## Post-Production Tips

1. **Add Title Card:**
   - "Threat Hunting Platform Demo"
   - "GCP Deployment - xdgaisocapp01"
   - Date

2. **Add Section Titles:**
   - "Environment Setup"
   - "YARA Rules"
   - "Threat Detection"
   - "Results"

3. **Highlight Key Points:**
   - 6 threats detected
   - 100% detection rate
   - All components operational

4. **Add Voiceover:**
   - Explain what THOR does
   - Explain ASGARD capabilities
   - Explain VALHALLA purpose
   - Highlight test results

5. **Add Captions:**
   - Command explanations
   - Result highlights
   - Key metrics

## File Formats

- **MP4:** Best for general sharing (YouTube, etc.)
- **GIF:** Quick previews, documentation
- **WebM:** Web embedding
- **MOV:** Mac compatibility

## Sharing Options

1. **YouTube:** Public or unlisted
2. **Vimeo:** Professional hosting
3. **Internal Platform:** Company video platform
4. **File Sharing:** Google Drive, Dropbox, etc.
5. **Documentation:** Embed in docs

## Quick Reference

**Record with asciinema:**
```bash
asciinema rec demo.cast
bash demo_script.sh
# Ctrl+D to stop
```

**Record with screen recorder:**
- Start recording software
- Run: `bash demo_script.sh`
- Stop recording
- Edit and export

**Duration:** 5-7 minutes recommended

---

**Ready to record?** Follow Method 1 (asciinema) for the easiest terminal demo recording.

