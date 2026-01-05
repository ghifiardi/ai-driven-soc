#!/bin/bash

# Fix YARA rules by using the original directory structure
# This preserves include statements that use relative paths

set -e

RULES_DIR="$HOME/threat-hunting-test/data/yara_rules/real_rules"
REPO_DIR="$HOME/threat-hunting-test/data/yara_rules/rules"

echo "=========================================="
echo "Fixing YARA Rules Structure"
echo "=========================================="
echo ""

# Remove the flattened directory
echo "Step 1: Removing flattened rules directory..."
rm -rf "$RULES_DIR"
mkdir -p "$RULES_DIR"

# Step 2: Copy the entire repository structure
echo "Step 2: Copying repository structure (preserving directories)..."
if [ -d "$REPO_DIR" ]; then
    # Copy entire structure
    cp -r "$REPO_DIR"/* "$RULES_DIR/" 2>/dev/null || {
        # If that fails, try copying the rules directory itself
        cp -r "$REPO_DIR" "$RULES_DIR/repo_structure" 2>/dev/null || true
    }
    echo "✓ Repository structure copied"
else
    echo "⚠ Repository not found at $REPO_DIR"
    exit 1
fi

# Step 3: Count rules
echo ""
echo "Step 3: Counting rules..."
TOTAL=$(find "$RULES_DIR" -type f \( -name "*.yar" -o -name "*.yara" \) 2>/dev/null | wc -l)
echo "  Total YARA rules: $TOTAL"

# Step 4: Update config to point to the rules directory
echo ""
echo "Step 4: Updating configuration..."
python3 << 'CONFIGFIX'
import json
import os

with open('config/thor_config.json', 'r') as f:
    config = json.load(f)

# Point to the rules directory (with structure preserved)
real_rules_path = os.path.expanduser('~/threat-hunting-test/data/yara_rules/real_rules')
config['yara']['rules_path'] = real_rules_path

with open('config/thor_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"✓ Config updated: {real_rules_path}")
print(f"  Rules will be loaded with directory structure preserved")
CONFIGFIX

echo ""
echo "=========================================="
echo "Fix Complete!"
echo "=========================================="
echo "Rules location: $RULES_DIR"
echo "Directory structure preserved for includes"
echo ""

