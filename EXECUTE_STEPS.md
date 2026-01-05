# Step-by-Step Execution Guide

Since direct gcloud access has permission restrictions, follow these steps using your SSH-in-browser session.

## Step 1: Run the Complete Setup Script

**Copy and paste the entire content of `complete_setup_script.sh` into your SSH-in-browser terminal.**

Or run these commands directly:

```bash
# Create directory structure
mkdir -p ~/threat-hunting-test
cd ~/threat-hunting-test
mkdir -p config logs data/yara_rules data/iocs data/sigma_rules scripts tests docs

# Create helper scripts (copy from complete_setup_script.sh)
# ... (see complete_setup_script.sh for full content)
```

## Step 2: Upload Files via SSH-in-browser

1. Click the **"UPLOAD FILE"** button in your SSH-in-browser interface
2. Upload these files to `~/threat-hunting-test/`:

### Main Directory Files:
- `thor_endpoint_agent.py`
- `asgard_orchestration_agent.py`
- `valhalla_feed_manager.py`
- `threat_hunting_quickstart.py`
- `requirements_threat_hunting.txt`
- `THREAT_HUNTING_README.md`

### Config Directory Files (upload to `~/threat-hunting-test/config/`):
- `config/thor_config.json`
- `config/asgard_config.json`
- `config/valhalla_config.json`

## Step 3: Set Up Python Environment

In your SSH session, run:

```bash
cd ~/threat-hunting-test
bash scripts/setup_env.sh
```

## Step 4: Update Configuration

```bash
cd ~/threat-hunting-test
bash scripts/update_config.sh
```

## Step 5: Verify Setup

```bash
cd ~/threat-hunting-test
bash scripts/quick_test.sh
bash scripts/verify_setup.sh
```

## Alternative: Manual Step-by-Step

If you prefer to do it manually, see `GCP_SETUP_COMMANDS.md` for detailed commands.

