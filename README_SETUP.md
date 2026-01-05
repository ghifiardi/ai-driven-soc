# Setup Instructions - Quick Guide

## Important: This Script is for GCP Instance, NOT Your Local Machine!

The `COPY_PASTE_TO_SSH.sh` file is **NOT meant to be run on your local Mac**. 

### What to Do:

1. **Open the file** `COPY_PASTE_TO_SSH.sh` in a text editor
2. **Copy ALL the content** (all 284 lines)
3. **Paste it into your SSH-in-browser session** on the GCP instance
4. **Press Enter** to execute

### Do NOT run:
```bash
# ❌ WRONG - Don't do this on your Mac
./COPY_PASTE_TO_SSH.sh
```

### Do this instead:
1. Open SSH-in-browser: `ssh.cloud.google.com/v2/ssh/.../xdgaisocapp01`
2. Copy content from `COPY_PASTE_TO_SSH.sh`
3. Paste into the SSH terminal
4. Press Enter

---

## Virtual Environment: Optional but Recommended

### Option 1: WITH Virtual Environment (Recommended)
```bash
# On GCP instance:
cd ~/threat-hunting-test
bash scripts/setup_env.sh
source venv/bin/activate  # Activate before running Python scripts
```

**Pros:**
- ✅ Isolated dependencies
- ✅ No conflicts with system packages
- ✅ Easy to remove/recreate

### Option 2: WITHOUT Virtual Environment
```bash
# On GCP instance:
cd ~/threat-hunting-test
bash scripts/setup_env.sh no  # 'no' disables virtual environment
# No need to activate - uses system Python
```

**Pros:**
- ✅ Simpler (no activation needed)
- ✅ Uses system Python directly

**Cons:**
- ⚠️ May conflict with other Python projects
- ⚠️ Harder to manage dependencies

### Recommendation:
**Use virtual environment** unless you have a specific reason not to. It's safer and cleaner.

---

## Quick Start (After Setup)

### If using virtual environment:
```bash
cd ~/threat-hunting-test
source venv/bin/activate  # Activate first!
python thor_endpoint_agent.py --help
```

### If NOT using virtual environment:
```bash
cd ~/threat-hunting-test
python3 thor_endpoint_agent.py --help
```

---

## Troubleshooting

### "Command not found" on local Mac
- ✅ This is normal! The script is for GCP, not your Mac
- Copy the content and paste into SSH session

### "Module not found" errors
- If using venv: Make sure you activated it (`source venv/bin/activate`)
- If not using venv: Check if packages installed correctly (`pip3 list`)

### Want to switch between venv and system?
- Remove venv: `rm -rf ~/threat-hunting-test/venv`
- Reinstall: `bash scripts/setup_env.sh` (with venv) or `bash scripts/setup_env.sh no` (without)

