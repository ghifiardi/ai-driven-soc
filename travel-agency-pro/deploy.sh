#!/bin/bash

# Travel Agency Pro - Deployment Script
# This script helps deploy the application to various platforms

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="travel-agency-pro"
VERSION="1.0.0"
DOCKER_IMAGE="travel-agency-pro"
DOCKER_TAG="latest"

# Functions
print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}  Travel Agency Pro Deployment${NC}"
    echo -e "${BLUE}================================${NC}"
    echo ""
}

print_step() {
    echo -e "${YELLOW}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_step "Checking prerequisites..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        exit 1
    fi
    
    # Check pip
    if ! command -v pip &> /dev/null; then
        print_error "pip is not installed"
        exit 1
    fi
    
    # Check Docker (optional)
    if command -v docker &> /dev/null; then
        DOCKER_AVAILABLE=true
        print_info "Docker is available"
    else
        DOCKER_AVAILABLE=false
        print_info "Docker is not available (some deployment options disabled)"
    fi
    
    # Check gcloud (optional)
    if command -v gcloud &> /dev/null; then
        GCLOUD_AVAILABLE=true
        print_info "Google Cloud SDK is available"
    else
        GCLOUD_AVAILABLE=false
        print_info "Google Cloud SDK is not available (some deployment options disabled)"
    fi
    
    print_success "Prerequisites check completed"
}

# Install dependencies
install_dependencies() {
    print_step "Installing Python dependencies..."
    
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_success "Python dependencies installed"
    else
        print_error "requirements.txt not found"
        exit 1
    fi
}

# Local development setup
setup_local() {
    print_step "Setting up local development environment..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_info "Virtual environment created"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install dependencies
    install_dependencies
    
    # Set up environment variables
    if [ ! -f ".env" ]; then
        cat > .env << EOF
# Travel Agency Pro Environment Variables
OPENAI_API_KEY=your-openai-api-key-here
BROWSER_USE_DEBUG=true
NODE_ENV=development
EOF
        print_info "Environment file created (.env)"
        print_info "Please update OPENAI_API_KEY in .env file"
    fi
    
    print_success "Local development environment ready"
}

# Start local server
start_local_server() {
    print_step "Starting local development server..."
    
    # Check if virtual environment exists
    if [ -d "venv" ]; then
        source venv/bin/activate
    fi
    
    # Start HTTP server
    echo "Starting server at http://localhost:8000"
    echo "Press Ctrl+C to stop"
    python3 -m http.server 8000
}

# Docker deployment
deploy_docker() {
    if [ "$DOCKER_AVAILABLE" = false ]; then
        print_error "Docker is not available"
        return 1
    fi
    
    print_step "Building Docker image..."
    
    # Create Dockerfile if it doesn't exist
    if [ ! -f "Dockerfile" ]; then
        cat > Dockerfile << EOF
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    wget \\
    gnupg \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python3 -c "import requests; requests.get('http://localhost:8000')" || exit 1

# Start application
CMD ["python3", "-m", "http.server", "8000"]
EOF
        print_info "Dockerfile created"
    fi
    
    # Build image
    docker build -t $DOCKER_IMAGE:$DOCKER_TAG .
    print_success "Docker image built"
    
    # Run container
    print_step "Starting Docker container..."
    docker run -d \
        --name $APP_NAME \
        -p 8000:8000 \
        -e OPENAI_API_KEY=\$OPENAI_API_KEY \
        $DOCKER_IMAGE:$DOCKER_TAG
    
    print_success "Docker container started"
    print_info "Application available at http://localhost:8000"
}

# Google Cloud deployment
deploy_gcloud() {
    if [ "$GCLOUD_AVAILABLE" = false ]; then
        print_error "Google Cloud SDK is not available"
        return 1
    fi
    
    print_step "Deploying to Google Cloud..."
    
    # Create app.yaml if it doesn't exist
    if [ ! -f "app.yaml" ]; then
        cat > app.yaml << EOF
runtime: python311
service: default

env_variables:
  OPENAI_API_KEY: "your-openai-api-key-here"

automatic_scaling:
  target_cpu_utilization: 0.65
  min_instances: 1
  max_instances: 10

resources:
  cpu: 1
  memory_gb: 2
  disk_size_gb: 10

handlers:
  - url: /.*
    script: auto
    secure: always
EOF
        print_info "app.yaml created"
    fi
    
    # Deploy to App Engine
    gcloud app deploy app.yaml --project=$(gcloud config get-value project)
    
    print_success "Deployed to Google Cloud App Engine"
    print_info "Application URL: https://$(gcloud config get-value project).appspot.com"
}

# Production deployment
deploy_production() {
    print_step "Setting up production deployment..."
    
    # Create production configuration
    if [ ! -f "production.py" ]; then
        cat > production.py << EOF
#!/usr/bin/env python3
"""
Production server for Travel Agency Pro
"""

import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
import ssl

class SecureHTTPRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('X-XSS-Protection', '1; mode=block')
        self.send_header('Strict-Transport-Security', 'max-age=31536000; includeSubDomains')
        super().end_headers()

def run_server():
    port = int(os.environ.get('PORT', 8000))
    
    # Create server
    httpd = HTTPServer(('0.0.0.0', port), SecureHTTPRequestHandler)
    
    print(f"Server running on port {port}")
    print(f"Visit: http://localhost:{port}")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\\nShutting down server...")
        httpd.shutdown()

if __name__ == "__main__":
    run_server()
EOF
        print_info "Production server script created"
    fi
    
    # Create systemd service file
    if [ ! -f "travel-agency-pro.service" ]; then
        cat > travel-agency-pro.service << EOF
[Unit]
Description=Travel Agency Pro
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/venv/bin
Environment=OPENAI_API_KEY=your-api-key-here
ExecStart=$(pwd)/venv/bin/python production.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF
        print_info "Systemd service file created"
        print_info "To install service: sudo cp travel-agency-pro.service /etc/systemd/system/"
        print_info "Then: sudo systemctl enable travel-agency-pro && sudo systemctl start travel-agency-pro"
    fi
    
    print_success "Production deployment setup completed"
}

# Cleanup
cleanup() {
    print_step "Cleaning up..."
    
    # Stop and remove Docker container
    if [ "$DOCKER_AVAILABLE" = true ]; then
        docker stop $APP_NAME 2>/dev/null || true
        docker rm $APP_NAME 2>/dev/null || true
        print_info "Docker container cleaned up"
    fi
    
    print_success "Cleanup completed"
}

# Main menu
show_menu() {
    echo ""
    echo -e "${BLUE}Deployment Options:${NC}"
    echo "1) Setup local development environment"
    echo "2) Start local server"
    echo "3) Deploy with Docker"
    echo "4) Deploy to Google Cloud"
    echo "5) Setup production deployment"
    echo "6) Cleanup"
    echo "7) Exit"
    echo ""
    read -p "Choose an option (1-7): " choice
    
    case $choice in
        1)
            setup_local
            ;;
        2)
            start_local_server
            ;;
        3)
            deploy_docker
            ;;
        4)
            deploy_gcloud
            ;;
        5)
            deploy_production
            ;;
        6)
            cleanup
            ;;
        7)
            print_info "Goodbye!"
            exit 0
            ;;
        *)
            print_error "Invalid option"
            show_menu
            ;;
    esac
}

# Main execution
main() {
    print_header
    
    # Check prerequisites
    check_prerequisites
    
    # Show menu
    show_menu
}

# Handle script arguments
case "${1:-}" in
    "local")
        setup_local
        ;;
    "start")
        start_local_server
        ;;
    "docker")
        deploy_docker
        ;;
    "gcloud")
        deploy_gcloud
        ;;
    "production")
        deploy_production
        ;;
    "cleanup")
        cleanup
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [option]"
        echo ""
        echo "Options:"
        echo "  local      Setup local development environment"
        echo "  start      Start local development server"
        echo "  docker     Deploy with Docker"
        echo "  gcloud     Deploy to Google Cloud"
        echo "  production Setup production deployment"
        echo "  cleanup    Clean up deployment artifacts"
        echo "  help       Show this help message"
        echo ""
        echo "If no option is provided, an interactive menu will be shown."
        exit 0
        ;;
    "")
        main
        ;;
    *)
        print_error "Unknown option: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac