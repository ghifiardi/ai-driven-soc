#!/bin/bash
# Enhanced CLA Deployment Script
# Deploy improved ML models to achieve 94%+ accuracy

set -e
echo "🚀 Deploying Enhanced CLA with Advanced ML..."

# Install additional ML packages
echo "Installing enhanced ML dependencies..."
pip install xgboost optuna scikit-learn==1.3.0

# Backup current config
echo "Backing up current configuration..."
cp config/cla_config.json config/cla_config_backup.json

# Deploy enhanced config
echo "Deploying enhanced configuration..."
cp enhanced_cla_config.json config/cla_config.json

# Run enhanced training
echo "Starting enhanced model training..."
python3 enhanced_cla_trainer.py

# Check if training achieved target
ACCURACY=$(python3 -c "
import joblib
import json
try:
    model_data = joblib.load('enhanced_cla_model.pkl')
    metadata = json.load(open('enhanced_cla_model_metadata.json'))
    print(f'{model_data.get(\"accuracy\", 0):.3f}')
except:
    print('0.667')
")

echo "Model accuracy achieved: $ACCURACY"

if (( $(echo "$ACCURACY >= 0.94" | bc -l) )); then
    echo "🎉 SUCCESS: Target accuracy of 94%+ achieved!"
    
    # Update production model
    echo "Deploying to production..."
    cp enhanced_cla_model.pkl models/production_cla_model.pkl
    cp enhanced_cla_model_metadata.json models/production_cla_metadata.json
    
    # Restart CLA service
    echo "Restarting CLA service..."
    sudo systemctl restart cla.service
    
    echo "✅ Enhanced CLA deployed successfully!"
    echo "📊 New performance metrics:"
    echo "   - Accuracy: $(echo "$ACCURACY * 100" | bc -l)%"
    echo "   - Target: 94%"
    echo "   - Improvement: $(echo "($ACCURACY - 0.667) * 100" | bc -l)%"
    
else
    echo "⚠️  Training did not reach 94% target"
    echo "   Current: $(echo "$ACCURACY * 100" | bc -l)%"
    echo "   Target: 94%"
    echo "   Gap: $(echo "(0.94 - $ACCURACY) * 100" | bc -l)%"
    echo ""
    echo "Recommendations:"
    echo "1. Increase training data volume"
    echo "2. Add more domain-specific features"
    echo "3. Try different model architectures"
    echo "4. Collect more analyst feedback"
fi

echo ""
echo "📈 To monitor the enhanced CLA:"
echo "   sudo journalctl -u cla.service -f"
echo ""
echo "📊 To check model performance:"
echo "   python3 -c \"import joblib; print(joblib.load('enhanced_cla_model.pkl')['accuracy'])\""
