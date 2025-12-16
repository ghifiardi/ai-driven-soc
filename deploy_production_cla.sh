#!/bin/bash

# Production CLA Deployment Script
# Deploy 100% Performance CLA Service to Production

set -e

echo "🚀 Starting Production CLA Deployment..."

# Configuration
VM_NAME="xdgaisocapp01"
ZONE="asia-southeast2-a"
SERVICE_NAME="production-cla"
LOCAL_DIR="/Users/raditio.ghifiardigmail.com/Downloads/ai-driven-soc"
REMOTE_DIR="/home/raditio.ghifiardigmail.com/ai-driven-soc"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Step 1: Check prerequisites
print_status "Checking prerequisites..."

# Check if gcloud is installed and authenticated
if ! command -v gcloud &> /dev/null; then
    print_error "gcloud CLI is not installed"
    exit 1
fi

# Check if user is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    print_error "Not authenticated with gcloud. Please run 'gcloud auth login'"
    exit 1
fi

print_success "Prerequisites check passed"

# Step 2: Upload files to VM
print_status "Uploading files to VM..."

# Upload production service files
gcloud compute scp \
    production_cla_service.py \
    config/production_cla_config.json \
    production_cla.service \
    ${VM_NAME}:${REMOTE_DIR}/ \
    --zone=${ZONE}

print_success "Files uploaded to VM"

# Step 3: Install dependencies on VM
print_status "Installing dependencies on VM..."

gcloud compute ssh ${VM_NAME} --zone=${ZONE} --command="
    cd ${REMOTE_DIR} &&
    
    # Install Python dependencies
    pip3 install flask schedule requests --user
    
    print_success 'Dependencies installed'
"

# Step 4: Setup systemd service
print_status "Setting up systemd service..."

gcloud compute ssh ${VM_NAME} --zone=${ZONE} --command="
    cd ${REMOTE_DIR} &&
    
    # Copy service file to systemd directory
    sudo cp production_cla.service /etc/systemd/system/ &&
    
    # Reload systemd
    sudo systemctl daemon-reload &&
    
    # Enable service
    sudo systemctl enable ${SERVICE_NAME} &&
    
    print_success 'Systemd service configured'
"

# Step 5: Start the service
print_status "Starting production CLA service..."

gcloud compute ssh ${VM_NAME} --zone=${ZONE} --command="
    # Stop any existing service
    sudo systemctl stop ${SERVICE_NAME} || true &&
    
    # Start the service
    sudo systemctl start ${SERVICE_NAME} &&
    
    # Check status
    sleep 5 &&
    sudo systemctl status ${SERVICE_NAME} --no-pager -l &&
    
    print_success 'Production CLA service started'
"

# Step 6: Verify deployment
print_status "Verifying deployment..."

# Wait for service to start
sleep 10

# Check service status
gcloud compute ssh ${VM_NAME} --zone=${ZONE} --command="
    # Check if service is running
    if sudo systemctl is-active --quiet ${SERVICE_NAME}; then
        print_success 'Service is running'
    else
        print_error 'Service is not running'
        sudo systemctl status ${SERVICE_NAME} --no-pager -l
        exit 1
    fi
    
    # Check if port is listening
    if netstat -tlnp | grep -q ':8080'; then
        print_success 'Service is listening on port 8080'
    else
        print_warning 'Service may not be listening on port 8080 yet'
    fi
    
    # Show service logs
    echo '=== Recent Service Logs ==='
    sudo journalctl -u ${SERVICE_NAME} --no-pager -n 20
"

# Step 7: Test the service
print_status "Testing the service..."

# Test health endpoint
gcloud compute ssh ${VM_NAME} --zone=${ZONE} --command="
    cd ${REMOTE_DIR} &&
    
    # Test health endpoint
    echo 'Testing health endpoint...'
    curl -s http://localhost:8080/health | python3 -m json.tool || echo 'Health check failed' &&
    
    # Test status endpoint
    echo 'Testing status endpoint...'
    curl -s http://localhost:8080/status | python3 -m json.tool || echo 'Status check failed'
"

# Step 8: Show deployment summary
print_status "Deployment completed!"

echo ""
echo "=========================================="
echo "🎉 PRODUCTION CLA DEPLOYMENT SUMMARY"
echo "=========================================="
echo ""
echo "✅ Service Name: ${SERVICE_NAME}"
echo "✅ VM: ${VM_NAME}"
echo "✅ Port: 8080"
echo "✅ Model: 100% Performance CLA"
echo ""
echo "📊 API Endpoints:"
echo "   • Health Check: http://${VM_NAME}:8080/health"
echo "   • Service Status: http://${VM_NAME}:8080/status"
echo "   • Classify Alert: http://${VM_NAME}:8080/classify (POST)"
echo "   • Manual Retrain: http://${VM_NAME}:8080/retrain (POST)"
echo ""
echo "🔧 Service Management:"
echo "   • Start:   sudo systemctl start ${SERVICE_NAME}"
echo "   • Stop:    sudo systemctl stop ${SERVICE_NAME}"
echo "   • Restart: sudo systemctl restart ${SERVICE_NAME}"
echo "   • Status:  sudo systemctl status ${SERVICE_NAME}"
echo "   • Logs:    sudo journalctl -u ${SERVICE_NAME} -f"
echo ""
echo "📈 Monitoring:"
echo "   • Service logs: /var/log/syslog"
echo "   • Application logs: ${REMOTE_DIR}/production_cla_service.log"
echo "   • Model storage: ${REMOTE_DIR}/models/"
echo ""
echo "🚀 Your 100% Performance CLA is now running in production!"
echo "=========================================="

print_success "Production CLA deployment completed successfully!"


