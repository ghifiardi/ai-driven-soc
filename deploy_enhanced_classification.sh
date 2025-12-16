#!/bin/bash

# Deploy Enhanced Classification System to GCP VM
# ===============================================

echo "🚀 Deploying Enhanced Classification System to GCP VM"
echo "======================================================"

# Configuration
VM_USER="app"
VM_HOST="xdgaisocapp01"
VM_PATH="/home/app/ai-driven-soc"
LOCAL_PATH="/Users/raditio.ghifiardigmail.com/Downloads/ai-driven-soc"

# Files to deploy
ENHANCED_FILES=(
    "enhanced_classification_engine.py"
    "enhanced_taa_agent.py"
    "threat_detection_analysis.py"
    "requirements_mcp.txt"
)

# Results to deploy (optional)
RESULT_FILES=(
    "enhanced_classification_results_20250918_131658.csv"
    "enhanced_taa_results_20250918_131852.csv"
    "threat_detection_investigation_20250918_115727.json"
)

echo "📋 Files to deploy:"
for file in "${ENHANCED_FILES[@]}"; do
    if [ -f "$LOCAL_PATH/$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file (not found)"
    fi
done

echo ""
echo "🔧 Deploying core files..."

# Deploy enhanced classification files
for file in "${ENHANCED_FILES[@]}"; do
    if [ -f "$LOCAL_PATH/$file" ]; then
        echo "📤 Uploading $file..."
        scp "$LOCAL_PATH/$file" $VM_USER@$VM_HOST:$VM_PATH/
        
        if [ $? -eq 0 ]; then
            echo "  ✅ $file deployed successfully"
        else
            echo "  ❌ Failed to deploy $file"
        fi
    fi
done

echo ""
echo "📊 Deploying analysis results (optional)..."

# Deploy result files
for file in "${RESULT_FILES[@]}"; do
    if [ -f "$LOCAL_PATH/$file" ]; then
        echo "📤 Uploading $file..."
        scp "$LOCAL_PATH/$file" $VM_USER@$VM_HOST:$VM_PATH/
        
        if [ $? -eq 0 ]; then
            echo "  ✅ $file deployed successfully"
        else
            echo "  ❌ Failed to deploy $file"
        fi
    fi
done

echo ""
echo "🔧 Setting up environment on VM..."

# Create setup commands for the VM
cat > vm_setup_commands.sh << 'EOF'
#!/bin/bash

echo "🔧 Setting up Enhanced Classification on VM..."

# Check if virtual environment exists
if [ ! -d "venv_mcp" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv_mcp
fi

# Activate virtual environment
source venv_mcp/bin/activate

# Install/upgrade requirements
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements_mcp.txt

# Test the enhanced classification engine
echo "🧪 Testing enhanced classification engine..."
python3 -c "
from enhanced_classification_engine import EnhancedClassificationEngine
engine = EnhancedClassificationEngine()
print('✅ Enhanced Classification Engine loaded successfully')
"

# Test the enhanced TAA agent
echo "🧪 Testing enhanced TAA agent..."
python3 -c "
import asyncio
from enhanced_taa_agent import EnhancedTAAgent

async def test_agent():
    agent = EnhancedTAAgent()
    print('✅ Enhanced TAA Agent loaded successfully')

asyncio.run(test_agent())
"

echo "✅ Enhanced Classification System setup complete!"
echo ""
echo "🎯 Usage Examples:"
echo "  # Test enhanced classification on existing data"
echo "  python3 enhanced_classification_engine.py"
echo ""
echo "  # Run enhanced TAA agent"
echo "  python3 enhanced_taa_agent.py"
echo ""
echo "  # Analyze threat detection patterns"
echo "  python3 threat_detection_analysis.py"
EOF

# Upload setup script to VM
scp vm_setup_commands.sh $VM_USER@$VM_HOST:$VM_PATH/
rm vm_setup_commands.sh

echo ""
echo "🚀 Executing setup on VM..."

# Execute setup on VM
ssh $VM_USER@$VM_HOST "cd $VM_PATH && chmod +x vm_setup_commands.sh && ./vm_setup_commands.sh"

echo ""
echo "✅ Enhanced Classification System Deployment Complete!"
echo ""
echo "📋 Next Steps:"
echo "1. SSH into your VM: ssh $VM_USER@$VM_HOST"
echo "2. Navigate to: cd $VM_PATH"
echo "3. Activate environment: source venv_mcp/bin/activate"
echo "4. Test the system: python3 enhanced_classification_engine.py"
echo ""
echo "🎯 Your enhanced TAA agent is now ready to replace the broken supervised_v1 model!"


