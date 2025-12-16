#!/bin/bash
# CLA Deployment Script - Compatible versions for older Python environments
# Run this after logging into the VM

set -e
echo "Starting CLA compatible deployment..."

# Clean up any previous deployment attempts
rm -rf ~/ai-driven-soc
mkdir -p ~/ai-driven-soc

# Extract the deployment package directly into the target directory
echo "Extracting files..."
mkdir -p ~/temp_extract
tar -xzvf ~/cla_deploy.tar.gz -C ~/temp_extract

# Copy only the actual files (skip the macOS resource files)
find ~/temp_extract -type f -not -name "._*" -exec cp {} ~/ai-driven-soc/ \;
rm -rf ~/temp_extract

# Create a config directory
mkdir -p ~/ai-driven-soc/config

# Create the config file
cat > ~/ai-driven-soc/config/cla_config.json << 'EOF'
{
  "project_id": "chronicle-dev-2be9",
  "bigquery_dataset": "soc_data",
  "bigquery_alerts_table": "alerts",
  "bigquery_feedback_table": "feedback",
  "bigquery_metrics_table": "metrics", 
  "bigquery_models_table": "models",
  "bigquery_evaluations_table": "evaluations",
  "bigquery_incidents_table": "incidents",
  "bigquery_patterns_table": "patterns",
  "bigquery_improvements_table": "improvements",
  "bigquery_agent_state_table": "agent_state",
  "feedback_topic": "soc-feedback",
  "feedback_subscription": "cla-feedback-sub",
  "metrics_topic": "soc-metrics",
  "metrics_subscription": "cla-metrics-sub",
  "langgraph_feedback_topic": "langgraph-feedback",
  "feedback_threshold": 10,
  "retraining_interval_days": 7,
  "model_improvement_threshold": 0.05,
  "metrics_collection_days": 30,
  "pattern_analysis_days": 30
}
EOF

echo "Setting up Python environment..."
# Set up Python virtual environment
cd ~/ai-driven-soc
python3 -m venv venv
source venv/bin/activate

# Create compatible requirements.txt file with older versions
cat > requirements.txt << 'EOF'
# Core GCP packages - use older versions compatible with the VM
google-cloud-storage==2.0.0
google-cloud-pubsub==1.7.0
google-cloud-bigquery==1.28.0

# Data processing - use older versions
numpy==1.16.6
pandas==0.25.3
scikit-learn==0.22.2

# Visualization and optimization - use older versions
matplotlib==3.0.3
seaborn==0.9.0
EOF

# Install the compatible requirements
pip install -r requirements.txt

# Make the agent script executable
chmod +x continuous-learning-agent.py

echo "Setting up systemd service..."
# Setup the systemd service
sudo cp cla.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable cla.service

# Start the service
echo "Starting CLA service..."
sudo systemctl start cla.service

echo "Checking service status..."
# Check service status
sudo systemctl status cla.service

echo "CLA deployment complete! To monitor logs in real-time, run:"
echo "sudo journalctl -u cla.service -f"
