# Complete Execution Summary - GCP Threat Hunting Setup

## Overview
This guide will help you set up the threat hunting test environment on your GCP instance `xdgaisocapp01`.

## Files Created for You

1. **`COPY_PASTE_TO_SSH.sh`** - Main setup script (copy-paste into SSH)
2. **`complete_setup_script.sh`** - Same as above (alternative)
3. **`GCP_SETUP_COMMANDS.md`** - Detailed command reference
4. **`FILES_TO_UPLOAD_LIST.txt`** - List of files to upload
5. **`EXECUTE_STEPS.md`** - Step-by-step guide

## Quick Execution Steps

### Step 1: Open SSH-in-browser Session
- Go to: `ssh.cloud.google.com/v2/ssh/projects/chronicle-dev-2be9/zones/asia-southeast2-a/instances/xdgaisocapp01`
- Or use your existing SSH-in-browser session

### Step 2: Run Setup Script
**Copy the entire content of `COPY_PASTE_TO_SSH.sh` and paste it into your SSH terminal.**

This will:
- ✅ Create directory structure at `~/threat-hunting-test`
- ✅ Create helper scripts
- ✅ Set up the environment

### Step 3: Upload Files
Use the **"UPLOAD FILE"** button in SSH-in-browser to upload:

**To `~/threat-hunting-test/`:**
- `thor_endpoint_agent.py` (36KB)
- `asgard_orchestration_agent.py` (32KB)
- `valhalla_feed_manager.py` (32KB)
- `threat_hunting_quickstart.py`
- `requirements_threat_hunting.txt` (4KB)
- `THREAT_HUNTING_README.md`

**To `~/threat-hunting-test/config/`:**
- `config/thor_config.json` (4KB)
- `config/asgard_config.json` (4KB)
- `config/valhalla_config.json` (8KB)

### Step 4: Set Up Python Environment
In your SSH session:
```bash
cd ~/threat-hunting-test
bash scripts/setup_env.sh
```

### Step 5: Update Configuration
```bash
cd ~/threat-hunting-test
bash scripts/update_config.sh
```

### Step 6: Verify Setup
```bash
cd ~/threat-hunting-test
bash scripts/quick_test.sh
bash scripts/verify_setup.sh
```

## Expected Results

After Step 2, you should see:
```
✓ Directory structure created
✓ README created
✓ Upload instructions created
✓ Environment setup script created
✓ Config update script created
✓ Quick test script created
✓ Verification script created
```

After Step 4, you should see:
```
✓ Virtual environment created
✓ Environment setup complete!
```

After Step 6, you should see:
```
✓ GCP libraries imported successfully
✓ YARA library imported successfully
✓ Requests library imported successfully
✓ Quick test completed!
```

## Troubleshooting

### If upload fails:
- Check file sizes (total ~120KB)
- Try uploading one file at a time
- Verify you're in the correct directory

### If Python setup fails:
- Check Python version: `python3 --version`
- Try: `python3 -m pip install --upgrade pip`
- Install manually: `pip install google-cloud-pubsub google-cloud-firestore google-cloud-bigquery yara-python requests`

### If config update fails:
- Manually edit config files
- Replace `your-gcp-project-id` with `chronicle-dev-2be9`

## Next Steps After Setup

1. **Initialize VALHALLA:**
   ```bash
   cd ~/threat-hunting-test
   source venv/bin/activate
   python valhalla_feed_manager.py
   ```

2. **Run a test scan:**
   ```bash
   python thor_endpoint_agent.py --config config/thor_config.json --scan-type quick
   ```

3. **Create a test campaign:**
   ```bash
   python asgard_orchestration_agent.py --help
   ```

## Directory Structure Created

```
~/threat-hunting-test/
├── config/              # Configuration files
├── logs/                # Log files
├── data/
│   ├── yara_rules/      # YARA rule files
│   ├── iocs/            # IOC files
│   └── sigma_rules/     # Sigma rule files
├── scripts/             # Helper scripts
│   ├── setup_env.sh
│   ├── update_config.sh
│   ├── quick_test.sh
│   └── verify_setup.sh
├── tests/               # Test files
├── docs/                 # Documentation
└── venv/                # Python virtual environment (created in step 4)
```

## Support

If you encounter issues:
1. Check `UPLOAD_INSTRUCTIONS.txt` in the created directory
2. Review `GCP_SETUP_COMMANDS.md` for detailed commands
3. Run `bash scripts/verify_setup.sh` to diagnose issues

