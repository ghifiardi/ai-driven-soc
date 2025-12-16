#!/bin/bash

# TAA A2A Agent Deployment Script
# ===============================
# This script deploys the TAA agent with A2A communication capabilities
# to Google Cloud Platform using Vertex AI Agent Engine.

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID="ai-driven-soc"
LOCATION="us-central1"
STAGING_BUCKET="gs://ai-driven-soc-staging"
AGENT_NAME="taa-a2a-agent"
SERVICE_ACCOUNT="taa-a2a-agent@${PROJECT_ID}.iam.gserviceaccount.com"

echo -e "${BLUE}=== TAA A2A Agent Deployment ===${NC}"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if gcloud is installed
check_gcloud() {
    print_status "Checking Google Cloud SDK installation..."
    if ! command -v gcloud &> /dev/null; then
        print_error "Google Cloud SDK is not installed. Please install it first."
        exit 1
    fi
    print_status "Google Cloud SDK is installed."
}

# Authenticate with Google Cloud
authenticate_gcloud() {
    print_status "Authenticating with Google Cloud..."
    gcloud auth login --no-launch-browser
    gcloud config set project $PROJECT_ID
    print_status "Authentication completed."
}

# Enable required APIs
enable_apis() {
    print_status "Enabling required Google Cloud APIs..."
    
    APIs=(
        "aiplatform.googleapis.com"
        "storage.googleapis.com"
        "containerregistry.googleapis.com"
        "pubsub.googleapis.com"
        "logging.googleapis.com"
        "monitoring.googleapis.com"
        "iam.googleapis.com"
    )
    
    for api in "${APIs[@]}"; do
        print_status "Enabling $api..."
        gcloud services enable $api --project=$PROJECT_ID
    done
    
    print_status "All required APIs enabled."
}

# Create staging bucket
create_staging_bucket() {
    print_status "Creating staging bucket..."
    if ! gsutil ls -b $STAGING_BUCKET &> /dev/null; then
        gsutil mb -p $PROJECT_ID -c STANDARD -l $LOCATION $STAGING_BUCKET
        print_status "Staging bucket created: $STAGING_BUCKET"
    else
        print_status "Staging bucket already exists: $STAGING_BUCKET"
    fi
}

# Create service account
create_service_account() {
    print_status "Creating service account for TAA A2A agent..."
    
    if ! gcloud iam service-accounts describe $SERVICE_ACCOUNT &> /dev/null; then
        gcloud iam service-accounts create taa-a2a-agent \
            --display-name="TAA A2A Agent Service Account" \
            --description="Service account for TAA A2A agent"
        print_status "Service account created: $SERVICE_ACCOUNT"
    else
        print_status "Service account already exists: $SERVICE_ACCOUNT"
    fi
    
    # Grant necessary roles
    ROLES=(
        "roles/aiplatform.user"
        "roles/pubsub.publisher"
        "roles/pubsub.subscriber"
        "roles/storage.objectViewer"
        "roles/logging.logWriter"
        "roles/monitoring.metricWriter"
    )
    
    for role in "${ROLES[@]}"; do
        print_status "Granting role $role..."
        gcloud projects add-iam-policy-binding $PROJECT_ID \
            --member="serviceAccount:$SERVICE_ACCOUNT" \
            --role=$role
    done
    
    print_status "Service account configured with necessary permissions."
}

# Create Pub/Sub topics and subscriptions
setup_pubsub() {
    print_status "Setting up Pub/Sub topics and subscriptions..."
    
    TOPICS=(
        "ada-alerts"
        "taa-feedback"
        "a2a-communication"
        "threat-intelligence"
        "containment-requests"
    )
    
    for topic in "${TOPICS[@]}"; do
        if ! gcloud pubsub topics describe $topic &> /dev/null; then
            gcloud pubsub topics create $topic
            print_status "Created topic: $topic"
        else
            print_status "Topic already exists: $topic"
        fi
    done
    
    # Create subscriptions
    SUBSCRIPTIONS=(
        "ada-alerts-subscription"
        "taa-feedback-subscription"
        "a2a-communication-subscription"
    )
    
    for sub in "${SUBSCRIPTIONS[@]}"; do
        if ! gcloud pubsub subscriptions describe $sub &> /dev/null; then
            gcloud pubsub subscriptions create $sub --topic=ada-alerts
            print_status "Created subscription: $sub"
        else
            print_status "Subscription already exists: $sub"
        fi
    done
    
    print_status "Pub/Sub setup completed."
}

# Install Python dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install dependencies
    print_status "Installing A2A dependencies..."
    pip install -r requirements_a2a.txt
    
    print_status "Dependencies installed successfully."
}

# Create configuration directory
setup_config() {
    print_status "Setting up configuration..."
    
    # Create config directory if it doesn't exist
    mkdir -p config
    mkdir -p logs
    
    # Create logs directory
    mkdir -p logs
    
    print_status "Configuration setup completed."
}

# Deploy to Vertex AI Agent Engine
deploy_to_vertex_ai() {
    print_status "Deploying TAA A2A agent to Vertex AI Agent Engine..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Set environment variables
    export GOOGLE_CLOUD_PROJECT=$PROJECT_ID
    export GOOGLE_CLOUD_LOCATION=$LOCATION
    export STAGING_BUCKET=$STAGING_BUCKET
    
    # Run deployment
    python3 taa_a2a_agent.py
    
    print_status "Deployment to Vertex AI completed."
}

# Test the deployment
test_deployment() {
    print_status "Testing A2A deployment..."
    
    # Wait for deployment to be ready
    sleep 30
    
    # Test health endpoint
    print_status "Testing health endpoint..."
    curl -f http://localhost:8080/health || print_warning "Health check failed"
    
    # Test agent discovery
    print_status "Testing agent discovery..."
    curl -f http://localhost:8080/.well-known/agent.json || print_warning "Agent discovery failed"
    
    print_status "Deployment testing completed."
}

# Main deployment function
main() {
    print_status "Starting TAA A2A agent deployment..."
    
    # Check prerequisites
    check_gcloud
    
    # Setup Google Cloud
    authenticate_gcloud
    enable_apis
    create_staging_bucket
    create_service_account
    setup_pubsub
    
    # Setup local environment
    install_dependencies
    setup_config
    
    # Deploy
    deploy_to_vertex_ai
    
    # Test
    test_deployment
    
    print_status "TAA A2A agent deployment completed successfully!"
    print_status ""
    print_status "Next steps:"
    print_status "1. Start the A2A server: python3 taa_a2a_server.py"
    print_status "2. Test A2A communication: python3 taa_a2a_test_client.py"
    print_status "3. Monitor logs: tail -f logs/taa_a2a.log"
    print_status ""
    print_status "Agent endpoints:"
    print_status "- Health: http://localhost:8080/health"
    print_status "- Agent Card: http://localhost:8080/.well-known/agent.json"
    print_status "- A2A Tasks: http://localhost:8080/a2a/tasks"
    print_status "- Metrics: http://localhost:8080/metrics"
}

# Run main function
main "$@" 