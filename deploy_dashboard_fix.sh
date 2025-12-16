#!/bin/bash

# Deploy Dashboard Fix to VM
# This script fixes the NameError: name 'filtered_alerts' is not defined

echo "🔧 Deploying Dashboard Fix to VM..."

# VM connection details
VM_USER="app"
VM_HOST="10.45.254.19"
VM_PATH="/home/app/ai-driven-soc"

echo "📡 Connecting to VM: ${VM_USER}@${VM_HOST}"

# Create the fix code
cat > dashboard_fix.py << 'EOF'
#!/usr/bin/env python3
"""
Dashboard Fix - Initialize filtered_alerts variable
"""

import pandas as pd
import streamlit as st

# Initialize filtered_alerts if not defined
def initialize_filtered_alerts():
    """Initialize filtered_alerts DataFrame if not defined"""
    if 'filtered_alerts' not in st.session_state:
        st.session_state.filtered_alerts = pd.DataFrame()
    return st.session_state.filtered_alerts

# Safe check for filtered_alerts
def safe_filtered_alerts_check():
    """Safely check if filtered_alerts exists and is not empty"""
    try:
        if 'filtered_alerts' in locals() or 'filtered_alerts' in globals():
            return filtered_alerts is not None and not filtered_alerts.empty
        else:
            return False
    except NameError:
        return False

# Initialize at module level
filtered_alerts = initialize_filtered_alerts()
EOF

echo "📁 Created dashboard fix code"

# Deploy to VM
echo "🚀 Deploying fix to VM..."

# Copy the fix to VM
scp dashboard_fix.py ${VM_USER}@${VM_HOST}:${VM_PATH}/

echo "✅ Dashboard fix deployed to VM"
echo ""
echo "📋 Next steps on VM:"
echo "1. SSH into VM: ssh ${VM_USER}@${VM_HOST}"
echo "2. Navigate to: cd ${VM_PATH}"
echo "3. Find feedback.py: find . -name 'feedback.py' -type f"
echo "4. Add the initialization code at the top of feedback.py"
echo "5. Restart the dashboard service"
echo ""
echo "🔧 Quick fix command for VM:"
echo "echo 'filtered_alerts = pd.DataFrame()' | cat - feedback.py > temp && mv temp feedback.py"
echo ""
echo "🔄 Restart dashboard service:"
echo "sudo systemctl restart your-dashboard-service"
echo "# or"
echo "docker restart dashboard-container"


