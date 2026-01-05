#!/bin/bash

# Automated file upload script for GCP Threat Hunting
# This script attempts to upload files using gcloud compute scp

set -e

INSTANCE_NAME="xdgaisocapp01"
ZONE="asia-southeast2-a"
PROJECT_ID="chronicle-dev-2be9"
REMOTE_DIR="~/threat-hunting-test"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Uploading Threat Hunting Files to GCP"
echo "=========================================="
echo ""

# Check if gcloud is available
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found. Please install it or use manual upload."
    exit 1
fi

# Files to upload to main directory
MAIN_FILES=(
    "thor_endpoint_agent.py"
    "asgard_orchestration_agent.py"
    "valhalla_feed_manager.py"
    "threat_hunting_quickstart.py"
    "requirements_threat_hunting.txt"
    "THREAT_HUNTING_README.md"
)

# Config files
CONFIG_FILES=(
    "config/thor_config.json"
    "config/asgard_config.json"
    "config/valhalla_config.json"
)

echo "Step 1: Uploading main Python files..."
UPLOADED_MAIN=()
for file in "${MAIN_FILES[@]}"; do
    if [ -f "$LOCAL_DIR/$file" ]; then
        echo "  Uploading $file..."
        if gcloud compute scp \
            "$LOCAL_DIR/$file" \
            "app@$INSTANCE_NAME:$REMOTE_DIR/" \
            --zone="$ZONE" \
            --project="$PROJECT_ID" \
            2>&1; then
            echo "  ✓ $file uploaded"
            UPLOADED_MAIN+=("$file")
        else
            echo "  ⚠ Failed to upload $file (may need manual upload)"
        fi
    else
        echo "  ⚠ File not found: $file"
    fi
done

echo ""
echo "Step 2: Uploading config files..."
UPLOADED_CONFIG=()
for file in "${CONFIG_FILES[@]}"; do
    if [ -f "$LOCAL_DIR/$file" ]; then
        echo "  Uploading $file..."
        if gcloud compute scp \
            "$LOCAL_DIR/$file" \
            "app@$INSTANCE_NAME:$REMOTE_DIR/config/" \
            --zone="$ZONE" \
            --project="$PROJECT_ID" \
            2>&1; then
            echo "  ✓ $file uploaded"
            UPLOADED_CONFIG+=("$file")
        else
            echo "  ⚠ Failed to upload $file (may need manual upload)"
        fi
    else
        echo "  ⚠ File not found: $file"
    fi
done

echo ""
echo "=========================================="
echo "Upload Summary"
echo "=========================================="
echo ""
echo "Main files uploaded: ${#UPLOADED_MAIN[@]}/${#MAIN_FILES[@]}"
echo "Config files uploaded: ${#UPLOADED_CONFIG[@]}/${#CONFIG_FILES[@]}"
echo ""

if [ ${#UPLOADED_MAIN[@]} -eq 0 ] && [ ${#UPLOADED_CONFIG[@]} -eq 0 ]; then
    echo "⚠ All uploads failed. This is likely due to gcloud permissions."
    echo ""
    echo "Alternative: Use SSH-in-browser UPLOAD FILE button:"
    echo "1. Upload these files to ~/threat-hunting-test/:"
    for file in "${MAIN_FILES[@]}"; do
        [ -f "$LOCAL_DIR/$file" ] && echo "   - $file"
    done
    echo "2. Upload these files to ~/threat-hunting-test/config/:"
    for file in "${CONFIG_FILES[@]}"; do
        [ -f "$LOCAL_DIR/$file" ] && echo "   - $file"
    done
else
    echo "✓ Upload complete! Some files may need manual upload."
fi

