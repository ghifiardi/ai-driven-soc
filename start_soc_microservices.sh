#!/bin/bash
# Master Launch Script for AI-Driven SOC Microservices 🚀

# Configure Environment
export GCP_PROJECT_ID="chronicle-dev-2be9"
export BIGQUERY_PROJECT_ID="chronicle-dev-2be9"
export TAA_SUBSCRIPTION_ID="ada-alerts-sub"

echo "--------------------------------------------------"
echo "🚀 Starting AI-Driven SOC Microservices Pipeline"
echo "--------------------------------------------------"

# 1. Start Anomaly Detection Agent (ADA)
echo "[1/3] Starting LangGraph ADA..."
python3 langgraph_ada_integration.py > ada_workflow.log 2>&1 &
ADA_PID=$!
echo "      ADA started (PID: $ADA_PID)"

# 2. Start Triage & Analysis Agent (TAA)
echo "[2/3] Starting LangGraph TAA Service..."
# Note: Ensure subscription exists: 
# gcloud pubsub subscriptions create ada-alerts-sub --topic=ada-alerts
python3 taa_langgraph_service.py > taa_service.log 2>&1 &
TAA_PID=$!
echo "      TAA started (PID: $TAA_PID)"

# 3. Start Continuous Learning Agent (CLA)
echo "[3/3] Starting Self-Learning Agent (CLA)..."
python3 cla_complete.py --config config/cla_config.json > cla_service.log 2>&1 &
CLA_PID=$!
echo "      CLA started (PID: $CLA_PID)"

echo "--------------------------------------------------"
echo "✅ All microservices are initializing in the background."
echo "   - View ADA logs: tail -f ada_workflow.log"
echo "   - View TAA logs: tail -f taa_service.log"
echo "   - View CLA logs: tail -f cla_service.log"
echo "--------------------------------------------------"

# Trap exit to kill background processes (optional for dev)
trap "kill $ADA_PID $TAA_PID $CLA_PID; echo 'Services stopped.'; exit" SIGINT SIGTERM
wait
