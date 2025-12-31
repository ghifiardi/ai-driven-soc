#!/usr/bin/env bash
# sync_to_vm.sh
# Pushes local code and config to the Pilot VM.

PROJECT_ID="gatra-481606"
ZONE="us-central1-a"
INSTANCE_NAME="gatra-soc-pilot-vm"

echo "🔄 Syncing code to Pilot VM..."

# Exclude large/useless dirs
EXCLUDES="--exclude '.git' --exclude 'venv' --exclude 'node_modules' --exclude '__pycache__'"

gcloud compute scp --project=$PROJECT_ID --zone=$ZONE --recurse ./ $INSTANCE_NAME:~/ai-driven-soc/

echo "✅ Sync complete."
echo "You can now start the server on the VM."
