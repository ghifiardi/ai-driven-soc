# Quick Start - Copy This Into SSH

## Method 1: Direct Copy-Paste (Easiest)

1. **On your Mac**: Open `COPY_PASTE_TO_SSH.sh` in a text editor
2. **Select ALL** (Cmd+A) and **Copy** (Cmd+C)
3. **In your SSH session**: Right-click and paste (or Cmd+V)
4. **Press Enter** to execute

The script will create everything automatically!

---

## Method 2: Upload File First

### Step 1: Upload the file
1. In SSH-in-browser, click **"UPLOAD FILE"** button
2. Select `COPY_PASTE_TO_SSH.sh` from your Mac
3. Upload to `~` (home directory)

### Step 2: Make it executable and run
```bash
chmod +x COPY_PASTE_TO_SSH.sh
bash COPY_PASTE_TO_SSH.sh
```

---

## What the Script Does

When you run it, it will:
1. ✅ Create `~/threat-hunting-test/` directory
2. ✅ Create all subdirectories (config, logs, data, scripts, etc.)
3. ✅ Create helper scripts (setup_env.sh, update_config.sh, etc.)
4. ✅ Create README and instructions

**Then you need to:**
1. Upload the threat hunting Python files
2. Run `bash scripts/setup_env.sh` to install Python packages
3. Run `bash scripts/update_config.sh` to update config files

---

## After Running the Script

You'll see output like:
```
==========================================
Complete Threat Hunting Setup
==========================================
Step 1: Creating directory structure...
✓ Directory structure created
...
```

Then follow the "Next steps" shown at the end.

