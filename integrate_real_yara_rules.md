# Integrating Real YARA Rules from Public Sources

This guide shows how to download and integrate real YARA rules from public threat intelligence sources into your threat hunting platform.

## Public YARA Rule Sources

### 1. **YARA-Rules Project** (Recommended)
- **Repository:** https://github.com/Yara-Rules/rules
- **Content:** Comprehensive collection of YARA rules for malware families
- **Categories:** Ransomware, Trojans, Backdoors, APT groups, etc.

### 2. **Emerging Threats**
- **Repository:** https://github.com/EmergingThreats/rules
- **Content:** ET Open rules (formerly Snort rules, now includes YARA)
- **Focus:** Network and endpoint threats

### 3. **ReversingLabs YARA Rules**
- **Repository:** https://github.com/reversinglabs/reversinglabs-yara-rules
- **Content:** High-quality malware detection rules

### 4. **Florian Roth's YARA Rules**
- **Repository:** https://github.com/Neo23x0/signature-base
- **Content:** APT and malware signatures

## Integration Methods

### Method 1: Direct Download and Integration

### Method 2: Automated Script Integration

### Method 3: VALHALLA Feed Manager Integration

---

## Method 1: Manual Download and Setup

### Step 1: Download YARA Rules

```bash
cd ~/threat-hunting-test/data/yara_rules

# Download YARA-Rules repository
git clone https://github.com/Yara-Rules/rules.git yara-rules-repo

# Or download specific categories
mkdir -p real_rules
cd real_rules

# Download ransomware rules
curl -L https://api.github.com/repos/Yara-Rules/rules/tarball/master | tar -xz --strip=1
# Then copy specific rule files
```

### Step 2: Organize Rules

```bash
cd ~/threat-hunting-test/data/yara_rules

# Create organized structure
mkdir -p real_rules/{ransomware,apt,trojan,backdoor,webshell}

# Copy rules to appropriate categories
# (Manual process based on rule names and content)
```

### Step 3: Update THOR Configuration

```bash
# Update config to include real rules
python3 << 'CONFIGFIX'
import json

with open('config/thor_config.json', 'r') as f:
    config = json.load(f)

# Add real rules path
config['yara']['rules_path'] = '/home/app/threat-hunting-test/data/yara_rules/real_rules'
config['yara']['rule_sets'] = {
    "ransomware": "real_rules/ransomware",
    "apt": "real_rules/apt",
    "trojan": "real_rules/trojan"
}

with open('config/thor_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("✓ Config updated with real rules paths")
CONFIGFIX
```

---

## Method 2: Automated Script Integration

### Script: Download and Organize Real YARA Rules

```bash
#!/bin/bash
# download_real_yara_rules.sh

set -e

RULES_DIR="$HOME/threat-hunting-test/data/yara_rules/real_rules"
TEMP_DIR="/tmp/yara_download"

echo "=========================================="
echo "Downloading Real YARA Rules"
echo "=========================================="
echo ""

mkdir -p "$RULES_DIR"
mkdir -p "$TEMP_DIR"
cd "$TEMP_DIR"

# Download YARA-Rules repository
echo "Step 1: Downloading YARA-Rules repository..."
if [ -d "rules" ]; then
    cd rules
    git pull
else
    git clone https://github.com/Yara-Rules/rules.git
    cd rules
fi

echo "✓ Repository downloaded"
echo ""

# Organize rules by category
echo "Step 2: Organizing rules by category..."

# Create category directories
mkdir -p "$RULES_DIR"/{ransomware,apt,trojan,backdoor,webshell,cryptominer,other}

# Copy ransomware rules
echo "  Copying ransomware rules..."
find . -name "*ransomware*" -o -name "*Ransomware*" -o -name "*lockbit*" -o -name "*wannacry*" | \
    head -20 | xargs -I {} cp {} "$RULES_DIR/ransomware/" 2>/dev/null || echo "    Some files may not exist"

# Copy APT rules
echo "  Copying APT rules..."
find . -name "*apt*" -o -name "*APT*" | \
    head -20 | xargs -I {} cp {} "$RULES_DIR/apt/" 2>/dev/null || echo "    Some files may not exist"

# Copy trojan rules
echo "  Copying trojan rules..."
find . -name "*trojan*" -o -name "*Trojan*" | \
    head -20 | xargs -I {} cp {} "$RULES_DIR/trojan/" 2>/dev/null || echo "    Some files may not exist"

# Copy backdoor rules
echo "  Copying backdoor rules..."
find . -name "*backdoor*" -o -name "*Backdoor*" | \
    head -20 | xargs -I {} cp {} "$RULES_DIR/backdoor/" 2>/dev/null || echo "    Some files may not exist"

# Copy webshell rules
echo "  Copying webshell rules..."
find . -name "*webshell*" -o -name "*Webshell*" | \
    head -20 | xargs -I {} cp {} "$RULES_DIR/webshell/" 2>/dev/null || echo "    Some files may not exist"

echo ""
echo "Step 3: Verifying rules..."
for category in ransomware apt trojan backdoor webshell; do
    count=$(find "$RULES_DIR/$category" -name "*.yar" -o -name "*.yara" 2>/dev/null | wc -l)
    echo "  $category: $count rules"
done

echo ""
echo "=========================================="
echo "Download Complete!"
echo "=========================================="
echo ""
echo "Rules location: $RULES_DIR"
echo ""
echo "Next: Update THOR config to use these rules"
echo ""
```

---

## Method 3: VALHALLA Integration

### Enhanced VALHALLA Feed Manager

The VALHALLA feed manager can be extended to automatically fetch and manage real YARA rules from public sources.

### Implementation Steps

1. **Add GitHub integration to VALHALLA**
2. **Schedule automatic rule updates**
3. **Validate and compile rules**
4. **Distribute to THOR agents**

---

## Quick Integration Guide

### On Your GCP Instance

Run these commands to quickly integrate real YARA rules:

```bash
cd ~/threat-hunting-test
source venv/bin/activate

# Create real rules directory
mkdir -p data/yara_rules/real_rules

# Download YARA-Rules (if git is available)
cd data/yara_rules/real_rules
git clone https://github.com/Yara-Rules/rules.git 2>/dev/null || {
    echo "Git not available, downloading via curl..."
    curl -L https://github.com/Yara-Rules/rules/archive/refs/heads/master.zip -o rules.zip
    unzip -q rules.zip 2>/dev/null || echo "unzip not available"
}

# Count downloaded rules
find . -name "*.yar" -o -name "*.yara" | wc -l

echo "✓ Real YARA rules downloaded"
```

### Update Configuration

```bash
cd ~/threat-hunting-test

# Update config to use real rules
python3 << 'CONFIGFIX'
import json
import os

with open('config/thor_config.json', 'r') as f:
    config = json.load(f)

# Set rules path to include real rules
real_rules_path = os.path.expanduser('~/threat-hunting-test/data/yara_rules/real_rules')
config['yara']['rules_path'] = real_rules_path

# Also keep test rules available
config['yara']['additional_paths'] = [
    os.path.expanduser('~/threat-hunting-test/data/yara_rules')
]

with open('config/thor_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"✓ Config updated")
print(f"  Real rules path: {real_rules_path}")
CONFIGFIX
```

### Test with Real Rules

```bash
cd ~/threat-hunting-test
source venv/bin/activate

# Test THOR with real rules
python thor_endpoint_agent.py \
    --config config/thor_config.json \
    --scan-type filesystem \
    --target /tmp \
    --load-yara
```

---

## Rule Quality and Validation

### Validate Rules Before Use

```bash
# Install yara command-line tool (if not already installed)
sudo yum install -y yara

# Validate a rule file
yara -w data/yara_rules/real_rules/rules/ransomware/rule.yar /path/to/test/file

# Check for syntax errors
find data/yara_rules/real_rules -name "*.yar" -exec yara -w {} /dev/null \;
```

### Rule Organization Best Practices

1. **Categorize by threat type** (ransomware, apt, trojan, etc.)
2. **Version control** rules with git
3. **Regular updates** from source repositories
4. **Quality scoring** (low false positive rate)
5. **Testing** before production deployment

---

## Security Considerations

⚠️ **Important:**

1. **Real YARA rules may have false positives** - Test thoroughly
2. **Some rules may be too broad** - Review and tune as needed
3. **Keep rules updated** - Malware evolves constantly
4. **Monitor detection rates** - Track false positives
5. **Whitelist known-good files** - Reduce false positives

---

## Next Steps

1. Download real rules from public sources
2. Organize by category
3. Validate rule syntax
4. Test on known-good files first
5. Gradually enable rules in production
6. Monitor and tune based on results

---

## Resources

- **YARA-Rules:** https://github.com/Yara-Rules/rules
- **YARA Documentation:** https://yara.readthedocs.io/
- **Rule Writing Guide:** https://yara.readthedocs.io/en/stable/writingrules.html

---

**Note:** Always test real YARA rules in a controlled environment before deploying to production systems.

