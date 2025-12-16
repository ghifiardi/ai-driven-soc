#!/bin/bash
#
# FOUNDATION DASHBOARD V1.0 - RESTORATION SCRIPT
# Date: October 1, 2025
# Purpose: Restore dashboard to stable foundation version
#

set -e  # Exit on error

echo "=================================================="
echo "  Foundation Dashboard V1.0 - Restoration Script"
echo "=================================================="
echo ""

# Configuration
PROJECT_DIR="/Users/raditio.ghifiardigmail.com/Downloads/ai-driven-soc"
VM_INSTANCE="app@xdgaisocapp01"
ZONE="asia-southeast2-a"
FOUNDATION_FILE="complete_operational_dashboard_FOUNDATION_V1_20251001.py"
TARGET_FILE="complete_operational_dashboard.py"

echo "📋 Configuration:"
echo "  Project Dir: $PROJECT_DIR"
echo "  VM Instance: $VM_INSTANCE"
echo "  Zone: $ZONE"
echo "  Foundation File: $FOUNDATION_FILE"
echo ""

# Step 1: Verify foundation file exists
echo "🔍 Step 1: Verifying foundation file..."
if [ ! -f "$PROJECT_DIR/$FOUNDATION_FILE" ]; then
    echo "❌ ERROR: Foundation file not found: $FOUNDATION_FILE"
    echo "   Please ensure the foundation backup exists."
    exit 1
fi
echo "✅ Foundation file found"
echo ""

# Step 2: Create backup of current file
echo "💾 Step 2: Creating backup of current file..."
BACKUP_NAME="complete_operational_dashboard_backup_$(date +%Y%m%d_%H%M%S).py"
cd "$PROJECT_DIR"
cp "$TARGET_FILE" "$BACKUP_NAME" 2>/dev/null || echo "   (No current file to backup)"
echo "✅ Backup created: $BACKUP_NAME"
echo ""

# Step 3: Restore foundation file
echo "🔄 Step 3: Restoring foundation file..."
cp "$FOUNDATION_FILE" "$TARGET_FILE"
echo "✅ Foundation file restored locally"
echo ""

# Step 4: Deploy to VM
echo "🚀 Step 4: Deploying to VM..."
echo "   (This may take a moment...)"
gcloud compute scp "$TARGET_FILE" "$VM_INSTANCE:/home/app/ai-driven-soc/" \
    --zone="$ZONE" \
    --tunnel-through-iap \
    2>&1 | grep -v "WARNING:" || true
echo "✅ Deployed to VM"
echo ""

# Step 5: Restart dashboard
echo "♻️  Step 5: Restarting dashboard..."
gcloud compute ssh "$VM_INSTANCE" \
    --zone="$ZONE" \
    --tunnel-through-iap \
    --command='cd /home/app/ai-driven-soc && ./restart_dashboard.sh' \
    2>&1 | grep -v "WARNING:" || true
echo "✅ Dashboard restarted"
echo ""

# Step 6: Verify
echo "✓ Step 6: Verifying deployment..."
VERIFY_OUTPUT=$(gcloud compute ssh "$VM_INSTANCE" \
    --zone="$ZONE" \
    --tunnel-through-iap \
    --command='ss -tlnp | grep 8535' \
    2>&1 | grep -v "WARNING:" || echo "NOT_RUNNING")

if echo "$VERIFY_OUTPUT" | grep -q "8535"; then
    echo "✅ Dashboard is running on port 8535"
else
    echo "⚠️  WARNING: Dashboard may not be running. Check manually."
fi
echo ""

echo "=================================================="
echo "  ✅ RESTORATION COMPLETE"
echo "=================================================="
echo ""
echo "📊 Dashboard URL: http://10.45.254.19:8535"
echo "📝 Documentation: docs/FOUNDATION_DASHBOARD_V1.md"
echo "📄 DOCX Documentation: docs/FOUNDATION_DASHBOARD_V1.docx"
echo "💾 Previous backup: $BACKUP_NAME"
echo ""
echo "If you encounter any issues, check the logs:"
echo "  gcloud compute ssh $VM_INSTANCE --zone=$ZONE --tunnel-through-iap \\"
echo "    --command='tail -n 100 /home/app/ai-driven-soc/main_dashboard.log'"
echo ""

