#!/bin/bash
# Script to download threat hunting files from GitHub
# Run this on your GCP instance via SSH

set -e

GITHUB_REPO="https://raw.githubusercontent.com/ghifiardi/ai-driven-soc/main"
TARGET_DIR="$HOME/threat-hunting-test"

echo "=========================================="
echo "Downloading Threat Hunting Files from GitHub"
echo "=========================================="
echo ""

cd "$TARGET_DIR"

# Download main Python files
echo "Step 1: Downloading main Python files..."

files=(
    "thor_endpoint_agent.py"
    "asgard_orchestration_agent.py"
    "valhalla_feed_manager.py"
    "threat_hunting_quickstart.py"
    "requirements_threat_hunting.txt"
    "THREAT_HUNTING_README.md"
)

for file in "${files[@]}"; do
    echo "  Downloading $file..."
    if curl -f -s -o "$file" "$GITHUB_REPO/$file"; then
        echo "  ✓ $file downloaded"
    else
        echo "  ⚠ Failed to download $file"
    fi
done

echo ""
echo "Step 2: Downloading config files..."

# Create config directory if it doesn't exist
mkdir -p config

config_files=(
    "config/thor_config.json"
    "config/asgard_config.json"
    "config/valhalla_config.json"
)

for file in "${config_files[@]}"; do
    filename=$(basename "$file")
    echo "  Downloading $filename..."
    if curl -f -s -o "config/$filename" "$GITHUB_REPO/$file"; then
        echo "  ✓ $filename downloaded"
    else
        echo "  ⚠ Failed to download $filename"
    fi
done

echo ""
echo "=========================================="
echo "Download Complete!"
echo "=========================================="
echo ""
echo "Verifying files..."
ls -lh *.py requirements_threat_hunting.txt 2>/dev/null | head -10
echo ""
ls -lh config/*.json 2>/dev/null
echo ""
echo "Next steps:"
echo "1. Run: bash scripts/setup_env.sh"
echo "2. Run: bash scripts/update_config.sh"
echo "3. Run: bash scripts/quick_test.sh"
echo ""

