#!/bin/bash

# Deploy Threat Hunting files to GCP instance
# This script copies all necessary files from local repository to GCP VM

set -e

# Configuration
INSTANCE_NAME="xdgaisocapp01"
ZONE="asia-southeast2-a"
PROJECT_ID="chronicle-dev-2be9"
REMOTE_DIR="$HOME/threat-hunting-test"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Deploying Threat Hunting to GCP"
echo "=========================================="
echo ""
echo "Instance: $INSTANCE_NAME"
echo "Zone: $ZONE"
echo "Remote directory: $REMOTE_DIR"
echo "Local directory: $LOCAL_DIR"
echo ""

# First, run the setup script on the remote instance
echo "Step 1: Creating directory structure on GCP instance..."
gcloud compute ssh "$INSTANCE_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT_ID" \
    --command="bash -s" < "$LOCAL_DIR/setup_gcp_threat_hunting.sh"

echo ""
echo "Step 2: Copying threat hunting agent files..."
gcloud compute scp \
    "$LOCAL_DIR/thor_endpoint_agent.py" \
    "$LOCAL_DIR/asgard_orchestration_agent.py" \
    "$LOCAL_DIR/valhalla_feed_manager.py" \
    "$LOCAL_DIR/threat_hunting_quickstart.py" \
    "app@$INSTANCE_NAME:$REMOTE_DIR/" \
    --zone="$ZONE" \
    --project="$PROJECT_ID"

echo ""
echo "Step 3: Copying configuration files..."
gcloud compute scp \
    "$LOCAL_DIR/config/thor_config.json" \
    "$LOCAL_DIR/config/asgard_config.json" \
    "$LOCAL_DIR/config/valhalla_config.json" \
    "app@$INSTANCE_NAME:$REMOTE_DIR/config/" \
    --zone="$ZONE" \
    --project="$PROJECT_ID"

echo ""
echo "Step 4: Copying requirements file..."
if [ -f "$LOCAL_DIR/requirements_threat_hunting.txt" ]; then
    gcloud compute scp \
        "$LOCAL_DIR/requirements_threat_hunting.txt" \
        "app@$INSTANCE_NAME:$REMOTE_DIR/" \
        --zone="$ZONE" \
        --project="$PROJECT_ID"
else
    echo "⚠ Warning: requirements_threat_hunting.txt not found"
fi

echo ""
echo "Step 5: Copying documentation..."
gcloud compute scp \
    "$LOCAL_DIR/THREAT_HUNTING_README.md" \
    "app@$INSTANCE_NAME:$REMOTE_DIR/docs/" \
    --zone="$ZONE" \
    --project="$PROJECT_ID"

echo ""
echo "Step 6: Copying example files..."
if [ -d "$LOCAL_DIR/examples" ]; then
    gcloud compute scp \
        --recurse \
        "$LOCAL_DIR/examples/" \
        "app@$INSTANCE_NAME:$REMOTE_DIR/examples/" \
        --zone="$ZONE" \
        --project="$PROJECT_ID"
fi

echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo ""
echo "Files deployed to: $REMOTE_DIR on $INSTANCE_NAME"
echo ""
echo "Next steps (SSH into the instance):"
echo "1. SSH into instance:"
echo "   gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID"
echo ""
echo "2. Navigate to directory:"
echo "   cd $REMOTE_DIR"
echo ""
echo "3. Set up Python environment:"
echo "   bash scripts/setup_env.sh"
echo ""
echo "4. Activate virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "5. Run quick test:"
echo "   bash scripts/quick_test.sh"
echo ""
echo "6. Update config files with your GCP project ID:"
echo "   sed -i 's/your-gcp-project-id/$PROJECT_ID/g' config/*.json"
echo ""

