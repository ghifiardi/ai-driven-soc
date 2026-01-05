# GCP Threat Hunting Test Directory Setup

## Quick Setup Commands

Copy and paste these commands directly into your SSH session on `xdgaisocapp01`:

### 1. Create Directory Structure

```bash
# Create main directory
mkdir -p ~/threat-hunting-test
cd ~/threat-hunting-test

# Create subdirectories
mkdir -p config logs data/yara_rules data/iocs data/sigma_rules scripts tests docs

# Verify structure
ls -la
tree -L 2 2>/dev/null || find . -maxdepth 2 -type d
```

### 2. Upload Setup Script (Optional)

If you want to use the automated setup script:

**Option A: Copy-paste the script content**
- Open `setup_threat_hunting_gcp.sh` from your local machine
- Copy the entire content
- In SSH session, run: `cat > ~/setup_threat_hunting_gcp.sh << 'SCRIPT_END'`
- Paste the script content
- Type `SCRIPT_END` and press Enter
- Run: `chmod +x ~/setup_threat_hunting_gcp.sh && bash ~/setup_threat_hunting_gcp.sh`

**Option B: Upload via SSH-in-browser**
- Click "UPLOAD FILE" button
- Upload `setup_threat_hunting_gcp.sh`
- Run: `chmod +x setup_threat_hunting_gcp.sh && bash setup_threat_hunting_gcp.sh`

### 3. Manual Directory Creation (Alternative)

If you prefer to create directories manually:

```bash
cd ~
mkdir -p threat-hunting-test/{config,logs,data/{yara_rules,iocs,sigma_rules},scripts,tests,docs}
cd threat-hunting-test
```

### 4. Upload Threat Hunting Files

**Method 1: Using SSH-in-browser Upload**
1. Click "UPLOAD FILE" button
2. Upload these files to `~/threat-hunting-test/`:
   - `thor_endpoint_agent.py`
   - `asgard_orchestration_agent.py`
   - `valhalla_feed_manager.py`
   - `threat_hunting_quickstart.py`
   - `requirements_threat_hunting.txt`
   - `THREAT_HUNTING_README.md`

3. Upload config files to `~/threat-hunting-test/config/`:
   - `config/thor_config.json`
   - `config/asgard_config.json`
   - `config/valhalla_config.json`

**Method 2: Using gcloud compute scp (from your local machine)**

```bash
# From your local machine terminal:
cd /Users/raditio.ghifiardigmail.com/Downloads/ai-driven-soc.backup

# Copy agent files
gcloud compute scp \
  thor_endpoint_agent.py \
  asgard_orchestration_agent.py \
  valhalla_feed_manager.py \
  threat_hunting_quickstart.py \
  app@xdgaisocapp01:~/threat-hunting-test/ \
  --zone=asia-southeast2-a --project=chronicle-dev-2be9

# Copy config files
gcloud compute scp \
  config/thor_config.json \
  config/asgard_config.json \
  config/valhalla_config.json \
  app@xdgaisocapp01:~/threat-hunting-test/config/ \
  --zone=asia-southeast2-a --project=chronicle-dev-2be9

# Copy requirements
gcloud compute scp \
  requirements_threat_hunting.txt \
  app@xdgaisocapp01:~/threat-hunting-test/ \
  --zone=asia-southeast2-a --project=chronicle-dev-2be9

# Copy documentation
gcloud compute scp \
  THREAT_HUNTING_README.md \
  app@xdgaisocapp01:~/threat-hunting-test/docs/ \
  --zone=asia-southeast2-a --project=chronicle-dev-2be9
```

**Method 3: Using git clone (if repo is accessible)**

```bash
cd ~/threat-hunting-test
git clone https://github.com/ghifiardi/ai-driven-soc.git temp_repo
cp temp_repo/thor_endpoint_agent.py .
cp temp_repo/asgard_orchestration_agent.py .
cp temp_repo/valhalla_feed_manager.py .
cp temp_repo/config/*.json config/
cp temp_repo/requirements_threat_hunting.txt .
cp temp_repo/THREAT_HUNTING_README.md docs/
rm -rf temp_repo
```

### 5. Set Up Python Environment

```bash
cd ~/threat-hunting-test

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements_threat_hunting.txt
```

### 6. Update Configuration Files

```bash
cd ~/threat-hunting-test

# Update project ID in config files
sed -i 's/your-gcp-project-id/chronicle-dev-2be9/g' config/*.json

# Verify changes
grep -r "chronicle-dev-2be9" config/
```

### 7. Quick Test

```bash
cd ~/threat-hunting-test
source venv/bin/activate

# Test Python imports
python3 -c "
from google.cloud import pubsub_v1, firestore, bigquery
print('✓ GCP libraries OK')
import yara
print('✓ YARA library OK')
"

# Check file structure
ls -la
ls -la config/
```

### 8. Verify Setup

```bash
cd ~/threat-hunting-test

# Check directory structure
find . -type d | sort

# Check files
ls -lh *.py
ls -lh config/

# Check Python environment
source venv/bin/activate
pip list | grep -E "(google-cloud|yara)"
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
├── scripts/             # Utility scripts
├── tests/               # Test files
├── docs/                 # Documentation
└── venv/                # Python virtual environment (created in step 5)
```

## Next Steps After Setup

1. **Initialize VALHALLA feed manager:**
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

## Troubleshooting

### Permission Issues
If you get permission errors:
```bash
chmod +x *.py
chmod +x scripts/*.sh
```

### Missing Dependencies
```bash
source venv/bin/activate
pip install --upgrade google-cloud-pubsub google-cloud-firestore google-cloud-bigquery yara-python requests
```

### GCP Authentication
```bash
gcloud auth application-default login
```

## Notes

- The directory is created in your home directory: `~/threat-hunting-test`
- All paths are relative to this directory
- Make sure you're in the correct directory before running commands
- Use `source venv/bin/activate` to activate the virtual environment in each new terminal session

