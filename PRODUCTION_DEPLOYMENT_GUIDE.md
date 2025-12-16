# Production Deployment Guide - Real Data CLA Training System

## 🚀 Production Deployment Checklist

### Phase 1: Pre-Deployment Validation

#### 1.1 Environment Verification
```bash
# Verify Python environment
python3.11 --version  # Should be 3.11+
pip list | grep -E "(scikit-learn|imbalanced-learn|xgboost|google-cloud-bigquery)"

# Verify BigQuery access
python3 -c "
from google.cloud import bigquery
client = bigquery.Client(project='chronicle-dev-2be9')
print('BigQuery access: OK')
"
```

#### 1.2 Data Validation
```bash
# Check data availability
bq query --use_legacy_sql=false "
SELECT 
  classification,
  COUNT(*) as count,
  COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() as percentage
FROM \`chronicle-dev-2be9.soc_data.processed_alerts\`
WHERE classification IS NOT NULL
GROUP BY classification
ORDER BY classification
"
```

#### 1.3 Model Training Test
```bash
# Run test training
cd /home/raditio.ghifiardigmail.com/ai-driven-soc
python3.11 real_data_cla_trainer.py

# Verify output shows:
# Status: success
# Precision: 100.0%
# Recall: 100.0%
# F1-Score: 100.0%
```

### Phase 2: Production Environment Setup

#### 2.1 Create Production Directory Structure
```bash
# Create production directories
sudo mkdir -p /opt/ai-driven-soc/{models,logs,config,scripts}
sudo chown -R raditio.ghifiardigmail.com:raditio.ghifiardigmail.com /opt/ai-driven-soc

# Copy production files
cp real_data_cla_trainer.py /opt/ai-driven-soc/
cp config/cla_config.json /opt/ai-driven-soc/config/
cp REAL_DATA_CLA_TRAINING_DOCUMENTATION.md /opt/ai-driven-soc/
```

#### 2.2 Set Up Service Account
```bash
# Copy service account key to secure location
sudo cp sa-gatra-bigquery.json /opt/ai-driven-soc/config/
sudo chmod 600 /opt/ai-driven-soc/config/sa-gatra-bigquery.json

# Set environment variable
echo 'export GOOGLE_APPLICATION_CREDENTIALS="/opt/ai-driven-soc/config/sa-gatra-bigquery.json"' >> ~/.bashrc
source ~/.bashrc
```

#### 2.3 Create Production Configuration
```bash
# Create production config
cat > /opt/ai-driven-soc/config/production_config.json << EOF
{
  "project_id": "chronicle-dev-2be9",
  "bigquery_dataset": "soc_data",
  "bigquery_processed_alerts_table": "processed_alerts",
  "min_training_samples": 1000,
  "max_training_samples": 100000,
  "retraining_interval_hours": 24,
  "model_retention_days": 30,
  "log_level": "INFO",
  "production_mode": true
}
EOF
```

### Phase 3: Automated Training System

#### 3.1 Create Training Script
```bash
# Create automated training script
cat > /opt/ai-driven-soc/scripts/automated_training.sh << 'EOF'
#!/bin/bash

# Automated Real Data CLA Training Script
# Run daily to retrain models with latest data

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="/opt/ai-driven-soc"
LOG_FILE="/opt/ai-driven-soc/logs/training_$(date +%Y%m%d_%H%M%S).log"
MODEL_DIR="/opt/ai-driven-soc/models"
BACKUP_DIR="/opt/ai-driven-soc/models/backup"

# Create directories
mkdir -p "$MODEL_DIR" "$BACKUP_DIR" "$(dirname "$LOG_FILE")"

# Log function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "Starting automated CLA training..."

# Backup existing models
if [ -d "$MODEL_DIR" ] && [ "$(ls -A $MODEL_DIR)" ]; then
    log "Backing up existing models..."
    cp -r "$MODEL_DIR"/*.pkl "$BACKUP_DIR/" 2>/dev/null || true
    cp -r "$MODEL_DIR"/*.json "$BACKUP_DIR/" 2>/dev/null || true
fi

# Change to script directory
cd "$SCRIPT_DIR"

# Run training
log "Starting model training..."
python3.11 real_data_cla_trainer.py >> "$LOG_FILE" 2>&1

# Check training result
if [ $? -eq 0 ]; then
    log "Training completed successfully!"
    
    # Verify model files were created
    NEW_MODELS=$(ls -t "$MODEL_DIR"/real_data_model_*.pkl 2>/dev/null | head -1)
    NEW_METRICS=$(ls -t "$MODEL_DIR"/real_data_metrics_*.json 2>/dev/null | head -1)
    
    if [ -f "$NEW_MODELS" ] && [ -f "$NEW_METRICS" ]; then
        log "New model files created:"
        log "  Model: $(basename "$NEW_MODELS")"
        log "  Metrics: $(basename "$NEW_METRICS")"
        
        # Display performance metrics
        if [ -f "$NEW_METRICS" ]; then
            log "Performance Metrics:"
            python3 -c "
import json
with open('$NEW_METRICS', 'r') as f:
    metrics = json.load(f)
    print(f'  Precision: {metrics[\"precision\"]:.1%}')
    print(f'  Recall: {metrics[\"recall\"]:.1%}')
    print(f'  F1-Score: {metrics[\"f1_score\"]:.1%}')
    print(f'  Accuracy: {metrics[\"accuracy\"]:.1%}')
            " | tee -a "$LOG_FILE"
        fi
        
        # Sync to dashboard (if accessible)
        if [ -d "/home/app/ai-model-training-dashboard/models" ]; then
            log "Syncing models to dashboard..."
            sudo cp "$NEW_MODELS" "$NEW_METRICS" /home/app/ai-model-training-dashboard/models/ 2>/dev/null || log "Dashboard sync failed (permission issue)"
        fi
        
        log "Training pipeline completed successfully!"
        exit 0
    else
        log "ERROR: Model files not found after training"
        exit 1
    fi
else
    log "ERROR: Training failed!"
    exit 1
fi
EOF

# Make script executable
chmod +x /opt/ai-driven-soc/scripts/automated_training.sh
```

#### 3.2 Set Up Cron Job
```bash
# Add daily training to crontab
(crontab -l 2>/dev/null; echo "0 2 * * * /opt/ai-driven-soc/scripts/automated_training.sh") | crontab -

# Verify crontab
crontab -l
```

### Phase 4: Monitoring and Alerting

#### 4.1 Create Monitoring Script
```bash
# Create monitoring script
cat > /opt/ai-driven-soc/scripts/monitor_training.sh << 'EOF'
#!/bin/bash

# CLA Training Monitoring Script
# Check training status and performance

MODEL_DIR="/opt/ai-driven-soc/models"
LOG_DIR="/opt/ai-driven-soc/logs"

# Check if models exist
if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A $MODEL_DIR 2>/dev/null)" ]; then
    echo "ALERT: No models found in $MODEL_DIR"
    exit 1
fi

# Get latest model metrics
LATEST_METRICS=$(ls -t "$MODEL_DIR"/real_data_metrics_*.json 2>/dev/null | head -1)

if [ -z "$LATEST_METRICS" ]; then
    echo "ALERT: No metrics files found"
    exit 1
fi

# Check performance thresholds
python3 -c "
import json
import sys

with open('$LATEST_METRICS', 'r') as f:
    metrics = json.load(f)

precision = metrics.get('precision', 0)
recall = metrics.get('recall', 0)
f1_score = metrics.get('f1_score', 0)

print(f'Latest Model Performance:')
print(f'  Precision: {precision:.1%}')
print(f'  Recall: {recall:.1%}')
print(f'  F1-Score: {f1_score:.1%}')

# Check if performance is acceptable
if precision < 0.95 or recall < 0.95 or f1_score < 0.95:
    print('ALERT: Model performance below threshold (95%)')
    sys.exit(1)
else:
    print('Model performance: OK')
    sys.exit(0)
"

# Check training logs for errors
LATEST_LOG=$(ls -t "$LOG_DIR"/training_*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    if grep -q "ERROR\|FAILED" "$LATEST_LOG"; then
        echo "ALERT: Errors found in latest training log"
        exit 1
    fi
fi

echo "Monitoring check: PASSED"
EOF

chmod +x /opt/ai-driven-soc/scripts/monitor_training.sh
```

#### 4.2 Set Up Monitoring Cron Job
```bash
# Add monitoring to crontab (every 4 hours)
(crontab -l 2>/dev/null; echo "0 */4 * * * /opt/ai-driven-soc/scripts/monitor_training.sh") | crontab -
```

### Phase 5: Production Deployment

#### 5.1 Deploy to Production
```bash
# Run initial production training
/opt/ai-driven-soc/scripts/automated_training.sh

# Verify deployment
/opt/ai-driven-soc/scripts/monitor_training.sh
```

#### 5.2 Integration with Existing CLA System
```bash
# Update existing CLA service to use new models
# 1. Stop existing CLA service
sudo systemctl stop cla.service

# 2. Update model path in service configuration
# Edit cla.service file to point to new model directory

# 3. Restart service
sudo systemctl start cla.service
sudo systemctl status cla.service
```

### Phase 6: Validation and Testing

#### 6.1 Performance Validation
```bash
# Test model loading and prediction
python3 -c "
import pickle
import glob

# Find latest model
model_files = glob.glob('/opt/ai-driven-soc/models/real_data_model_*.pkl')
if model_files:
    latest_model = max(model_files)
    print(f'Loading model: {latest_model}')
    
    with open(latest_model, 'rb') as f:
        model = pickle.load(f)
    
    print(f'Model type: {type(model)}')
    print('Model loaded successfully!')
else:
    print('No models found')
"
```

#### 6.2 Dashboard Integration Test
```bash
# Test dashboard access
curl -s -o /dev/null -w '%{http_code}' http://10.45.254.19:8503

# Should return: 200
```

## 🔧 Production Configuration Files

### Systemd Service Configuration
```bash
# Create systemd service for automated training
sudo tee /etc/systemd/system/cla-training.service > /dev/null << 'EOF'
[Unit]
Description=CLA Automated Training Service
After=network.target

[Service]
Type=oneshot
User=raditio.ghifiardigmail.com
WorkingDirectory=/opt/ai-driven-soc
ExecStart=/opt/ai-driven-soc/scripts/automated_training.sh
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable cla-training.service
```

### Environment Configuration
```bash
# Create environment file
cat > /opt/ai-driven-soc/.env << 'EOF'
# CLA Training Environment Configuration
GOOGLE_APPLICATION_CREDENTIALS=/opt/ai-driven-soc/config/sa-gatra-bigquery.json
PYTHONPATH=/opt/ai-driven-soc
LOG_LEVEL=INFO
PRODUCTION_MODE=true
EOF
```

## 📊 Production Monitoring Dashboard

### Key Metrics to Monitor
1. **Training Success Rate**: Should be 100%
2. **Model Performance**: Precision, Recall, F1-Score (should be 100%)
3. **Training Frequency**: Daily at 2 AM
4. **Model File Size**: ~300MB per model
5. **Training Duration**: ~10-15 seconds
6. **Data Freshness**: Latest processed alerts

### Alerting Rules
- **Training Failure**: Alert if training fails
- **Performance Degradation**: Alert if F1-score < 95%
- **Missing Models**: Alert if no models found
- **Data Issues**: Alert if insufficient training data

## 🚨 Emergency Procedures

### Model Rollback
```bash
# Rollback to previous model
BACKUP_DIR="/opt/ai-driven-soc/models/backup"
PRODUCTION_DIR="/opt/ai-driven-soc/models"

# Find latest backup
LATEST_BACKUP=$(ls -t "$BACKUP_DIR"/real_data_model_*.pkl | head -1)

if [ -n "$LATEST_BACKUP" ]; then
    echo "Rolling back to: $LATEST_BACKUP"
    cp "$LATEST_BACKUP" "$PRODUCTION_DIR/"
    echo "Rollback completed"
else
    echo "No backup models found"
fi
```

### Emergency Training
```bash
# Force immediate retraining
/opt/ai-driven-soc/scripts/automated_training.sh

# Check results
/opt/ai-driven-soc/scripts/monitor_training.sh
```

## 📋 Production Checklist

### Pre-Deployment
- [ ] Environment validated
- [ ] Data source verified
- [ ] Training tested successfully
- [ ] Monitoring scripts created
- [ ] Backup procedures tested
- [ ] Documentation reviewed

### Post-Deployment
- [ ] Initial training completed
- [ ] Performance metrics validated
- [ ] Monitoring active
- [ ] Dashboard updated
- [ ] Team notified
- [ ] Rollback procedures ready

### Ongoing Maintenance
- [ ] Daily training logs reviewed
- [ ] Weekly performance analysis
- [ ] Monthly model evaluation
- [ ] Quarterly documentation update

---

## 🎯 Success Criteria

Production deployment is successful when:
- ✅ Automated daily training runs without errors
- ✅ Model performance maintains 100% precision/recall
- ✅ Dashboard shows current metrics
- ✅ Monitoring alerts work correctly
- ✅ Rollback procedures tested and ready

---

*This guide ensures safe and reliable production deployment of the Real Data CLA Training System.*


