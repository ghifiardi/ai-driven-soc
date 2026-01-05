# Guide: Integrating Real YARA Rules

## Quick Start - On Your GCP Instance

### Step 1: Download Real YARA Rules

```bash
cd ~/threat-hunting-test
source venv/bin/activate

# Download the integration script
curl -f -s -o download_real_yara_rules.sh https://raw.githubusercontent.com/ghifiardi/ai-driven-soc/main/download_real_yara_rules.sh
chmod +x download_real_yara_rules.sh

# Run the download script
bash download_real_yara_rules.sh
```

### Step 2: Validate Rules

```bash
# Validate downloaded rules
bash validate_yara_rules.sh
```

### Step 3: Update Configuration

```bash
cd ~/threat-hunting-test

# Update THOR config to use real rules
python3 << 'CONFIGFIX'
import json
import os

with open('config/thor_config.json', 'r') as f:
    config = json.load(f)

# Update rules path
real_rules_path = os.path.expanduser('~/threat-hunting-test/data/yara_rules/real_rules')
config['yara']['rules_path'] = real_rules_path

# Enable all rule categories
config['yara']['rule_sets'] = {
    "ransomware": "real_rules/ransomware",
    "apt": "real_rules/apt",
    "trojan": "real_rules/trojan",
    "backdoor": "real_rules/backdoor",
    "webshell": "real_rules/webshell"
}

with open('config/thor_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"✓ Config updated with real rules path: {real_rules_path}")
CONFIGFIX
```

### Step 4: Test with Real Rules

```bash
cd ~/threat-hunting-test
source venv/bin/activate

# Test THOR with real rules (scan a safe directory)
python thor_endpoint_agent.py \
    --config config/thor_config.json \
    --scan-type filesystem \
    --target /tmp \
    --load-yara
```

---

## What Are Real YARA Rules?

### Test Rules (Current)
- Simple keyword matching
- For testing system functionality
- Example: Matches "encrypt" or "ransom" keywords

### Real YARA Rules
- Specific malware signatures
- Hex patterns and byte sequences
- File structure analysis
- Behavioral patterns
- Family-specific indicators

### Example: Real Ransomware Rule Structure

```yara
rule Ransomware_LockBit_3 {
    meta:
        description = "Detects LockBit 3.0 ransomware"
        author = "YARA-Rules"
        reference = "https://..."
        date = "2024-01-15"
    strings:
        $s1 = { 4C 6F 63 6B 42 69 74 } // "LockBit" in hex
        $s2 = ".lockbit" ascii
        $s3 = "LOCKBIT_README" ascii
        $s4 = /[a-z0-9]{32}\.lockbit/ ascii
    condition:
        2 of them and filesize < 10MB
}
```

---

## Rule Sources

### 1. YARA-Rules Repository (Best for General Use)
- **URL:** https://github.com/Yara-Rules/rules
- **Content:** 1000+ rules covering major malware families
- **Categories:** Ransomware, APT, Trojans, Backdoors, etc.
- **Update Frequency:** Regular updates

### 2. Emerging Threats
- **URL:** https://github.com/EmergingThreats/rules
- **Focus:** Network and endpoint threats
- **Format:** ET Open rules

### 3. ReversingLabs
- **URL:** https://github.com/reversinglabs/reversinglabs-yara-rules
- **Quality:** High-quality, well-tested rules

### 4. Florian Roth (Neo23x0)
- **URL:** https://github.com/Neo23x0/signature-base
- **Focus:** APT and advanced threats

---

## Integration Methods

### Method A: Direct Git Clone (Recommended)

```bash
cd ~/threat-hunting-test/data/yara_rules

# Clone the repository
git clone https://github.com/Yara-Rules/rules.git real_rules_repo

# Use specific categories
cp -r real_rules_repo/ransomware/* real_rules/ransomware/
cp -r real_rules_repo/apt/* real_rules/apt/
```

### Method B: Automated Script

Use the provided `download_real_yara_rules.sh` script which:
- Downloads rules automatically
- Organizes by category
- Validates syntax
- Reports statistics

### Method C: Manual Download

1. Visit https://github.com/Yara-Rules/rules
2. Download specific rule files
3. Copy to `~/threat-hunting-test/data/yara_rules/real_rules/`
4. Organize by category

---

## Rule Organization

```
data/yara_rules/
├── real_rules/
│   ├── ransomware/     # Real ransomware detection rules
│   ├── apt/            # APT group rules
│   ├── trojan/         # Trojan rules
│   ├── backdoor/       # Backdoor rules
│   ├── webshell/       # Webshell rules
│   └── cryptominer/    # Cryptocurrency miner rules
└── test_malware_rules.yar  # Test rules (keep for testing)
```

---

## Testing Real Rules

### Test on Known-Good Files First

```bash
# Test on system directories (should have minimal matches)
python thor_endpoint_agent.py \
    --config config/thor_config.json \
    --scan-type filesystem \
    --target /usr/bin \
    --load-yara
```

### Monitor False Positives

Real rules may trigger on legitimate files. Monitor and whitelist as needed.

---

## Updating Rules

### Manual Update

```bash
cd ~/threat-hunting-test/data/yara_rules/real_rules_repo
git pull
# Then copy updated rules to real_rules/ directories
```

### Automated Update Script

Create a cron job or scheduled task to update rules regularly.

---

## Best Practices

1. **Start with a subset** - Don't enable all rules at once
2. **Test thoroughly** - Validate on known-good files first
3. **Monitor results** - Track false positive rates
4. **Whitelist as needed** - Add legitimate files to whitelist
5. **Update regularly** - Keep rules current with latest threats
6. **Version control** - Track which rules are active

---

## Comparison: Test vs Real Rules

| Aspect | Test Rules | Real Rules |
|--------|-----------|------------|
| **Purpose** | System testing | Malware detection |
| **Complexity** | Simple keywords | Hex patterns, structures |
| **False Positives** | High (intentional) | Lower (but possible) |
| **Coverage** | Limited | Comprehensive |
| **Maintenance** | Static | Regular updates needed |

---

## Next Steps After Integration

1. ✅ Download real rules
2. ✅ Validate syntax
3. ✅ Update configuration
4. ⏭️ Test on safe directories
5. ⏭️ Monitor false positives
6. ⏭️ Tune and whitelist as needed
7. ⏭️ Enable in production gradually

---

**Ready to integrate?** Run the download script on your GCP instance!

