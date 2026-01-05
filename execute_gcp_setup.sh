#!/bin/bash

# Complete setup execution script for GCP Threat Hunting
# This script executes all setup steps

set -e

INSTANCE_NAME="xdgaisocapp01"
ZONE="asia-southeast2-a"
PROJECT_ID="chronicle-dev-2be9"
REMOTE_DIR="$HOME/threat-hunting-test"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Complete GCP Threat Hunting Setup"
echo "=========================================="
echo ""

# Step 1: Create directory structure on GCP
echo "Step 1: Creating directory structure on GCP instance..."
gcloud compute ssh "$INSTANCE_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT_ID" \
    --command="mkdir -p $REMOTE_DIR/{config,logs,data/{yara_rules,iocs,sigma_rules},scripts,tests,docs} && echo 'Directory structure created successfully'" \
    2>&1 || {
    echo "⚠ Direct SSH failed. Creating setup script to upload..."
    echo ""
    echo "Please run these commands in your SSH-in-browser session:"
    echo ""
    cat << 'INLINE_SCRIPT'
mkdir -p ~/threat-hunting-test
cd ~/threat-hunting-test
mkdir -p config logs data/yara_rules data/iocs data/sigma_rules scripts tests docs
echo "Directory structure created!"
INLINE_SCRIPT
    exit 1
}

echo "✓ Directory structure created"
echo ""

# Step 2: Copy threat hunting agent files
echo "Step 2: Copying threat hunting agent files..."
FILES_TO_COPY=(
    "thor_endpoint_agent.py"
    "asgard_orchestration_agent.py"
    "valhalla_feed_manager.py"
    "threat_hunting_quickstart.py"
    "requirements_threat_hunting.txt"
)

for file in "${FILES_TO_COPY[@]}"; do
    if [ -f "$LOCAL_DIR/$file" ]; then
        echo "  Copying $file..."
        gcloud compute scp \
            "$LOCAL_DIR/$file" \
            "app@$INSTANCE_NAME:$REMOTE_DIR/" \
            --zone="$ZONE" \
            --project="$PROJECT_ID" \
            2>&1 || echo "  ⚠ Failed to copy $file (may need manual upload)"
    else
        echo "  ⚠ File not found: $file"
    fi
done

echo ""

# Step 3: Copy configuration files
echo "Step 3: Copying configuration files..."
CONFIG_FILES=(
    "config/thor_config.json"
    "config/asgard_config.json"
    "config/valhalla_config.json"
)

for file in "${CONFIG_FILES[@]}"; do
    if [ -f "$LOCAL_DIR/$file" ]; then
        echo "  Copying $file..."
        gcloud compute scp \
            "$LOCAL_DIR/$file" \
            "app@$INSTANCE_NAME:$REMOTE_DIR/config/" \
            --zone="$ZONE" \
            --project="$PROJECT_ID" \
            2>&1 || echo "  ⚠ Failed to copy $file (may need manual upload)"
    else
        echo "  ⚠ File not found: $file"
    fi
done

echo ""

# Step 4: Copy documentation
echo "Step 4: Copying documentation..."
if [ -f "$LOCAL_DIR/THREAT_HUNTING_README.md" ]; then
    gcloud compute scp \
        "$LOCAL_DIR/THREAT_HUNTING_README.md" \
        "app@$INSTANCE_NAME:$REMOTE_DIR/docs/" \
        --zone="$ZONE" \
        --project="$PROJECT_ID" \
        2>&1 || echo "  ⚠ Failed to copy documentation"
fi

echo ""

# Step 5: Set up Python environment on GCP
echo "Step 5: Setting up Python environment on GCP instance..."
gcloud compute ssh "$INSTANCE_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT_ID" \
    --command="
cd $REMOTE_DIR
if [ ! -d venv ]; then
    python3 -m venv venv
    echo 'Virtual environment created'
fi
source venv/bin/activate
pip install --upgrade pip
if [ -f requirements_threat_hunting.txt ]; then
    pip install -r requirements_threat_hunting.txt
else
    pip install google-cloud-pubsub google-cloud-firestore google-cloud-bigquery yara-python requests
fi
echo 'Python environment setup complete'
" 2>&1 || echo "⚠ Python setup may need to be done manually"

echo ""

# Step 6: Update configuration files with project ID
echo "Step 6: Updating configuration files with project ID..."
gcloud compute ssh "$INSTANCE_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT_ID" \
    --command="
cd $REMOTE_DIR
if [ -f config/thor_config.json ]; then
    sed -i 's/your-gcp-project-id/$PROJECT_ID/g' config/*.json 2>/dev/null || \
    sed -i '' 's/your-gcp-project-id/$PROJECT_ID/g' config/*.json
    echo 'Configuration files updated'
fi
" 2>&1 || echo "⚠ Config update may need to be done manually"

echo ""

# Step 7: Create helper scripts
echo "Step 7: Creating helper scripts..."
gcloud compute ssh "$INSTANCE_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT_ID" \
    --command="
cd $REMOTE_DIR/scripts
cat > quick_test.sh << 'EOF'
#!/bin/bash
cd $REMOTE_DIR
source venv/bin/activate
python3 -c \"
try:
    from google.cloud import pubsub_v1, firestore, bigquery
    print('✓ GCP libraries OK')
except ImportError as e:
    print('✗ GCP libraries:', e)
try:
    import yara
    print('✓ YARA library OK')
except ImportError as e:
    print('✗ YARA library:', e)
\"
EOF
chmod +x quick_test.sh
echo 'Helper scripts created'
" 2>&1 || echo "⚠ Helper scripts may need to be created manually"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Directory: $REMOTE_DIR on $INSTANCE_NAME"
echo ""
echo "To verify setup, SSH into the instance and run:"
echo "  cd $REMOTE_DIR"
echo "  source venv/bin/activate"
echo "  bash scripts/quick_test.sh"
echo ""

