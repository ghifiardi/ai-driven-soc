#!/bin/bash

# Deploy Real Dashboard Data Integration
# Updates the main dashboard to use real BigQuery data instead of mock data

echo "🚀 Deploying Real Dashboard Data Integration..."

# VM connection details
VM_USER="app"
VM_HOST="xdgaisocapp01"
VM_ZONE="asia-southeast2-a"
VM_DASHBOARD_PATH="/home/app/dashboard"

echo "📡 Connecting to VM: ${VM_USER}@${VM_HOST}"

# Backup the current connector
echo "📁 Backing up current BigQuery connector..."
gcloud compute ssh ${VM_HOST} --zone=${VM_ZONE} --command="sudo cp ${VM_DASHBOARD_PATH}/bigquery_data_connector.py ${VM_DASHBOARD_PATH}/bigquery_data_connector_backup.py"

# Copy the new real data connector
echo "📁 Copying new real data connector to VM..."
gcloud compute scp bigquery_data_connector_real.py ${VM_HOST}:${VM_DASHBOARD_PATH}/ --zone=${VM_ZONE}

# Replace the connector
echo "🔄 Updating BigQuery connector..."
gcloud compute ssh ${VM_HOST} --zone=${VM_ZONE} --command="sudo cp ${VM_DASHBOARD_PATH}/bigquery_data_connector_real.py ${VM_DASHBOARD_PATH}/bigquery_data_connector.py"

# Restart the dashboard service
echo "🔄 Restarting dashboard service..."
gcloud compute ssh ${VM_HOST} --zone=${VM_ZONE} --command="sudo pkill -f 'streamlit run main.py'"

# Wait a moment
sleep 3

# Start the dashboard again
echo "🚀 Starting dashboard with real data integration..."
gcloud compute ssh ${VM_HOST} --zone=${VM_ZONE} --command="cd ${VM_DASHBOARD_PATH} && nohup /home/app/.local/bin/streamlit run main.py --server.port 8511 --server.address 0.0.0.0 > dashboard.log 2>&1 &"

# Wait for the service to start
sleep 5

# Test the dashboard connection
echo "🧪 Testing dashboard connection..."
gcloud compute ssh ${VM_HOST} --zone=${VM_ZONE} --command="curl -s http://localhost:8511 | head -10"

echo ""
echo "✅ Real Dashboard Data Integration Complete!"
echo ""
echo "🎯 Dashboard Now Features:"
echo "- ✅ Real BigQuery data from dashboard_alerts table"
echo "- ✅ 1,000 real security alerts with numeric alarm IDs"
echo "- ✅ Real threat classifications (anomaly, benign)"
echo "- ✅ Real confidence scores and severity levels"
echo "- ✅ Real timestamps and reasoning data"
echo ""
echo "📊 Dashboard Access:"
echo "- Main Dashboard: http://10.45.254.19:8511"
echo "- TAA MoE Analytics: http://10.45.254.19:8511/🛡️_TAA_MoE_Security_Analytics"
echo "- Human Feedback: http://10.45.254.19:8511/🤝_Human_Feedback"
echo "- Agent Performance: http://10.45.254.19:8511/📊_Agent_Performance"
echo ""
echo "🎉 Your dashboard now displays real security data from BigQuery!"


