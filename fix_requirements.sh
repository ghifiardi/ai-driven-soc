#!/bin/bash
# Fix requirements for older Python versions
# Run this on your GCP instance

cd ~/threat-hunting-test

echo "Fixing requirements for Python compatibility..."

# Create a compatible requirements file
cat > requirements_threat_hunting_compatible.txt << 'REQEOF'
yara-python>=4.3.0
psutil>=5.9.0
requests>=2.20.0
google-cloud-pubsub>=2.18.0
google-cloud-firestore>=2.11.0
google-cloud-bigquery>=3.11.0
REQEOF

echo "✓ Created requirements_threat_hunting_compatible.txt"

# Install with compatible requirements
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Installing compatible packages..."
    pip install -r requirements_threat_hunting_compatible.txt
    echo "✓ Installation complete"
else
    echo "⚠ Virtual environment not found. Run: bash scripts/setup_env.sh"
fi

