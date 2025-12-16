#!/bin/bash

# Deploy Real TAA MoE Orchestrator with BigQuery and Pub/Sub Integration
# This replaces the mock orchestrator with real data integration

echo "🚀 Deploying Real TAA MoE Orchestrator with BigQuery/Pub/Sub Integration..."

# VM connection details
VM_USER="app"
VM_HOST="xdgaisocapp01"
VM_ZONE="asia-southeast2-a"
VM_PATH="/home/raditio.ghifiardigmail.com"

echo "📡 Connecting to VM: ${VM_USER}@${VM_HOST}"

# Copy the real data orchestrator to VM
echo "📁 Copying real data orchestrator to VM..."
gcloud compute scp taa_moe_orchestrator_real_data.py ${VM_HOST}:${VM_PATH}/ --zone=${VM_ZONE}

# Stop the current mock orchestrator
echo "🛑 Stopping current mock orchestrator..."
gcloud compute ssh ${VM_HOST} --zone=${VM_ZONE} --command="sudo pkill -f taa_moe_orchestrator.py"

# Wait a moment for the process to stop
sleep 3

# Start the real data orchestrator
echo "🚀 Starting real data orchestrator..."
gcloud compute ssh ${VM_HOST} --zone=${VM_ZONE} --command="cd ${VM_PATH} && nohup python3 taa_moe_orchestrator_real_data.py > taa_moe_orchestrator.log 2>&1 &"

# Wait a moment for the service to start
sleep 5

# Test the new orchestrator
echo "🧪 Testing new orchestrator..."
gcloud compute ssh ${VM_HOST} --zone=${VM_ZONE} --command="curl -s http://localhost:8000/health | python3 -m json.tool"

echo ""
echo "✅ Deployment Complete!"
echo ""
echo "🎯 Real TAA MoE Orchestrator Features:"
echo "- ✅ BigQuery integration for real alert data"
echo "- ✅ Pub/Sub integration for real-time processing"
echo "- ✅ Expert agent routing based on threat analysis"
echo "- ✅ Real-time threat statistics and analytics"
echo ""
echo "📊 Available Endpoints:"
echo "- Health Check: http://10.45.254.19:8000/health"
echo "- Real Alerts: http://10.45.254.19:8000/alerts?limit=10"
echo "- Threat Statistics: http://10.45.254.19:8000/statistics"
echo "- Expert Agents: http://10.45.254.19:8000/experts"
echo "- Process Alert: POST http://10.45.254.19:8000/orchestrator/run"
echo ""
echo "🎉 Your TAA MoE dashboard now connects to real BigQuery and Pub/Sub data!"
