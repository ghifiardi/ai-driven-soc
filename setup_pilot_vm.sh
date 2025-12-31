#!/usr/bin/env bash
# setup_pilot_vm.sh
# Automates the creation of a persistent Pilot VM for the AI-Driven SOC.

PROJECT_ID="gatra-481606"
REGION="us-central1"
ZONE="us-central1-a"
INSTANCE_NAME="gatra-soc-pilot-vm"
MACHINE_TYPE="e2-micro" # Eligible for Free Tier

echo "🚀 Starting Deployment of Persistent Pilot VM..."

# 1. Create the VM Instance
echo "Step 1: Provisioning GCE Instance ($INSTANCE_NAME)..."
gcloud compute instances create $INSTANCE_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --image-family=debian-11 \
    --image-project=debian-cloud \
    --tags=http-server,https-server \
    --scopes=https://www.googleapis.com/auth/cloud-platform

if [ $? -ne 0 ]; then
    echo "❌ Failed to create VM instance. Do you have Billing enabled for GCE?"
    exit 1
fi

echo "✅ VM Instance Created."

# 2. Wait for VM to be ready
echo "Step 2: Waiting for VM to initialize (30s)..."
sleep 30

# 3. Setup Remote Environment (Python, Dependencies)
echo "Step 3: Setting up the remote environment..."
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID --command="
    sudo apt-get update && sudo apt-get install -y python3-pip git curl jq
    mkdir -p ~/ai-driven-soc
"

# 4. Instructions for user
echo ""
echo "--- DEPLOYMENT INITIALIZED ---"
echo "Your 'Pilot VM' is now running in GCP!"
echo "Billing Note: Since this is an 'e2-micro' in us-central1, it is usually FREE."
echo ""
echo "NEXT STEPS:"
echo "1. Run the sync script (coming next) to push your local code to the VM."
echo "2. SSH into the VM: gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
echo "3. Start the SOC server inside the VM."
echo "------------------------------"
