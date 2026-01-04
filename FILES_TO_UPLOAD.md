# Files to Upload to Your GCP VM

## 📤 Upload These Files via SSH Browser

Use the "UPLOAD FILE" button in your SSH session to upload these files from your local machine to your VM (`xdgaisocapp01`).

---

## ✅ Required Files (Must Upload)

### 1. Setup Script
```
setup_threat_hunting_test.sh
```
**Purpose:** Automated setup script that creates the test environment
**Size:** ~15 KB

### 2. Core Agent Files
```
thor_endpoint_agent.py
asgard_orchestration_agent.py
valhalla_feed_manager.py
```
**Purpose:** The three main agents (THOR, ASGARD, VALHALLA)
**Size:** ~50-70 KB each

### 3. Demo Script
```
threat_hunting_quickstart.py
```
**Purpose:** Complete demonstration of the platform
**Size:** ~40 KB

---

## 📂 Upload Location

Upload all files to your home directory on the VM:
```
/home/app@xdgaisocapp01/
```

After upload, you'll have:
```
~
├── setup_threat_hunting_test.sh
├── thor_endpoint_agent.py
├── asgard_orchestration_agent.py
├── valhalla_feed_manager.py
└── threat_hunting_quickstart.py
```

---

## 🚀 After Upload - Run This

```bash
# Make setup script executable
chmod +x setup_threat_hunting_test.sh

# Run setup
bash setup_threat_hunting_test.sh

# Follow the prompts and wait for completion (~5-10 minutes)
```

---

## 📋 Step-by-Step Upload Process

### Using SSH Browser (Shown in Your Screenshot)

1. **Click "UPLOAD FILE" button** (visible in your screenshot)
2. **Select file from local machine:**
   - Browse to: `/Users/raditio.ghifiardigmail.com/Downloads/ai-driven-soc.backup/`
   - Select: `setup_threat_hunting_test.sh`
   - Click "Upload"
3. **Wait for upload to complete** (progress bar will show)
4. **Repeat for each file** (5 files total)

**Expected time:** 2-3 minutes for all uploads

---

## 🔄 Alternative: Upload All at Once

### Option 1: Create ZIP and Upload

On your local machine:
```bash
cd /Users/raditio.ghifiardigmail.com/Downloads/ai-driven-soc.backup/

# Create ZIP with only required files
zip threat-hunting-files.zip \
  setup_threat_hunting_test.sh \
  thor_endpoint_agent.py \
  asgard_orchestration_agent.py \
  valhalla_feed_manager.py \
  threat_hunting_quickstart.py
```

Then upload `threat-hunting-files.zip` via SSH browser.

On VM:
```bash
# Extract
unzip threat-hunting-files.zip

# Run setup
chmod +x setup_threat_hunting_test.sh
bash setup_threat_hunting_test.sh
```

---

### Option 2: Use gcloud SCP (from Local Terminal)

On your local machine (not SSH):
```bash
# Upload all files at once
gcloud compute scp \
  setup_threat_hunting_test.sh \
  thor_endpoint_agent.py \
  asgard_orchestration_agent.py \
  valhalla_feed_manager.py \
  threat_hunting_quickstart.py \
  xdgaisocapp01:~/ \
  --zone=asia-southeast2-a \
  --project=chronicle-dev-2be9
```

---

## ✅ Verification After Upload

On your VM, verify files are uploaded:
```bash
cd ~
ls -lh *.py *.sh

# Expected output:
# -rw-r--r-- 1 app app  15K Jan 01 12:00 setup_threat_hunting_test.sh
# -rw-r--r-- 1 app app  65K Jan 01 12:00 thor_endpoint_agent.py
# -rw-r--r-- 1 app app  55K Jan 01 12:00 asgard_orchestration_agent.py
# -rw-r--r-- 1 app app  60K Jan 01 12:00 valhalla_feed_manager.py
# -rw-r--r-- 1 app app  40K Jan 01 12:00 threat_hunting_quickstart.py
```

---

## 🎯 Quick Command Sequence (Copy-Paste)

After files are uploaded, run these commands in sequence:

```bash
# 1. Make setup script executable
chmod +x setup_threat_hunting_test.sh

# 2. Run automated setup
bash setup_threat_hunting_test.sh

# 3. Copy files to test directory (when setup completes)
cp ~/thor_endpoint_agent.py ~/threat-hunting-test/
cp ~/asgard_orchestration_agent.py ~/threat-hunting-test/
cp ~/valhalla_feed_manager.py ~/threat-hunting-test/
cp ~/threat_hunting_quickstart.py ~/threat-hunting-test/

# 4. Navigate to test directory
cd ~/threat-hunting-test

# 5. Activate environment
source activate.sh

# 6. Setup GCP resources
bash scripts/setup_gcp_resources.sh

# 7. Run quick test
bash scripts/quick_test_scan.sh

# 8. If test passes, run full demo
python3 threat_hunting_quickstart.py
```

**Total time: ~15-20 minutes from upload to working demo**

---

## 📊 File Sizes Reference

| File | Size (approx) | Purpose |
|------|---------------|---------|
| setup_threat_hunting_test.sh | 15 KB | Setup automation |
| thor_endpoint_agent.py | 65 KB | Endpoint scanning |
| asgard_orchestration_agent.py | 55 KB | Fleet orchestration |
| valhalla_feed_manager.py | 60 KB | Threat intel feeds |
| threat_hunting_quickstart.py | 40 KB | Demo script |
| **TOTAL** | **~235 KB** | **All files** |

**Upload time:** ~30 seconds on typical connection

---

## 🆘 Troubleshooting Upload Issues

### Issue: "Upload Failed" or "Connection Lost"

**Solution:**
```bash
# 1. Check VM is running
gcloud compute instances list --filter="name=xdgaisocapp01"

# 2. Reconnect SSH
# Click "SSH" button in GCP Console again

# 3. Retry upload
```

### Issue: "Permission Denied" after upload

**Solution:**
```bash
# Make files readable/executable
chmod +x setup_threat_hunting_test.sh
chmod +r *.py
```

### Issue: Files not showing after upload

**Solution:**
```bash
# Check current directory
pwd

# List all files including hidden
ls -la

# Files should be in /home/app@xdgaisocapp01/
cd ~
ls -lh
```

---

## 📝 Notes

- **Upload via SSH Browser is easiest** (use the button in your screenshot)
- **Files are small** (~235 KB total, quick upload)
- **No compilation needed** (Python scripts, run directly)
- **Setup script handles everything** (dependencies, config, test data)

---

## ✨ What Happens After Upload + Setup

After running `setup_threat_hunting_test.sh`, you'll have:

```
~/threat-hunting-test/
├── config/
│   ├── thor_config.json          ✅ Pre-configured
│   ├── asgard_config.json         ✅ Pre-configured
│   └── valhalla_config.json       ✅ Pre-configured
├── yara_rules/
│   └── test_malware.yar           ✅ Sample rules
├── test_data/
│   ├── safe/readme.txt            ✅ Test file
│   └── suspicious/test_sample.txt ✅ Triggers detection
├── scripts/
│   ├── setup_gcp_resources.sh     ✅ GCP setup
│   ├── update_threat_intel.sh     ✅ Fetch IOCs
│   └── quick_test_scan.sh         ✅ Test scanner
├── threat-hunting-env/            ✅ Python venv
├── thor_endpoint_agent.py         ✅ Your upload
├── asgard_orchestration_agent.py  ✅ Your upload
├── valhalla_feed_manager.py       ✅ Your upload
├── threat_hunting_quickstart.py   ✅ Your upload
└── README.md                      ✅ Documentation
```

**Everything ready to test!** 🎉

---

## 🎬 Next Steps

1. ✅ Upload 5 files (this page)
2. ✅ Run `setup_threat_hunting_test.sh`
3. ✅ Follow commands in `QUICK_START_VM.md`
4. ✅ Test threat hunting platform
5. ✅ Review results in BigQuery

**Good luck with your testing!** 🚀
