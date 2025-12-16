# Operations Quick Reference - Real Data CLA Training System

## 🚨 Emergency Contacts & Procedures

### Quick Commands
```bash
# Check system status
/opt/ai-driven-soc/scripts/monitor_training.sh

# Force retraining
/opt/ai-driven-soc/scripts/automated_training.sh

# Check dashboard
curl -s -o /dev/null -w '%{http_code}' http://10.45.254.19:8503

# View latest logs
tail -f /opt/ai-driven-soc/logs/training_*.log
```

### Emergency Rollback
```bash
# Rollback to previous model
BACKUP_DIR="/opt/ai-driven-soc/models/backup"
LATEST_BACKUP=$(ls -t "$BACKUP_DIR"/real_data_model_*.pkl | head -1)
cp "$LATEST_BACKUP" /opt/ai-driven-soc/models/
```

## 📊 System Status Dashboard

### Key URLs
- **Dashboard**: http://10.45.254.19:8503/
- **Logs**: `/opt/ai-driven-soc/logs/`
- **Models**: `/opt/ai-driven-soc/models/`

### Performance Thresholds
| Metric | Threshold | Action |
|--------|-----------|--------|
| Precision | < 95% | Investigate |
| Recall | < 95% | Retrain |
| F1-Score | < 95% | Emergency |
| Training Time | > 60s | Check resources |

## 🔧 Common Operations

### Daily Tasks
```bash
# 1. Check training status (automated at 2 AM)
crontab -l | grep automated_training

# 2. Verify model performance
python3 -c "
import json, glob
latest = max(glob.glob('/opt/ai-driven-soc/models/real_data_metrics_*.json'))
with open(latest) as f: m = json.load(f)
print(f'F1: {m[\"f1_score\"]:.1%}, Precision: {m[\"precision\"]:.1%}, Recall: {m[\"recall\"]:.1%}')
"

# 3. Check data freshness
bq query --use_legacy_sql=false "
SELECT COUNT(*) as recent_alerts
FROM \`chronicle-dev-2be9.soc_data.processed_alerts\`
WHERE timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
"
```

### Weekly Tasks
```bash
# 1. Model cleanup (keep last 30 days)
find /opt/ai-driven-soc/models -name "real_data_model_*.pkl" -mtime +30 -delete

# 2. Log cleanup
find /opt/ai-driven-soc/logs -name "training_*.log" -mtime +30 -delete

# 3. Performance analysis
python3 -c "
import json, glob
metrics = []
for f in glob.glob('/opt/ai-driven-soc/models/real_data_metrics_*.json'):
    with open(f) as file: metrics.append(json.load(file))
print(f'Models trained: {len(metrics)}')
print(f'Avg F1: {sum(m[\"f1_score\"] for m in metrics)/len(metrics):.1%}')
"
```

## 🛠️ Troubleshooting Guide

### Issue: Training Fails
```bash
# Check logs
tail -n 50 /opt/ai-driven-soc/logs/training_*.log

# Common causes:
# 1. BigQuery access issues
gcloud auth application-default login

# 2. Insufficient data
bq query --use_legacy_sql=false "
SELECT COUNT(*) FROM \`chronicle-dev-2be9.soc_data.processed_alerts\`
WHERE classification IS NOT NULL
"

# 3. Memory issues
free -h
df -h /opt/ai-driven-soc/
```

### Issue: Low Performance
```bash
# Check data quality
bq query --use_legacy_sql=false "
SELECT 
  classification,
  COUNT(*) as count,
  COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() as percentage
FROM \`chronicle-dev-2be9.soc_data.processed_alerts\`
WHERE classification IS NOT NULL
GROUP BY classification
"

# Expected: ~81% anomaly, ~19% benign
```

### Issue: Dashboard Not Updating
```bash
# Check dashboard process
ps aux | grep streamlit

# Restart dashboard
sudo pkill -f streamlit
sudo -u app bash -c 'cd /home/app/ai-model-training-dashboard && nohup streamlit run enhanced_training_dashboard.py --server.port 8503 --server.address 0.0.0.0 --server.headless true > /tmp/dashboard.log 2>&1 &'
```

## 📋 Maintenance Schedule

### Daily (Automated)
- [ ] Model retraining (2:00 AM)
- [ ] Performance monitoring (every 4 hours)
- [ ] Log rotation
- [ ] Health checks

### Weekly (Manual)
- [ ] Model cleanup
- [ ] Performance analysis
- [ ] Documentation review
- [ ] Backup verification

### Monthly (Manual)
- [ ] Security updates
- [ ] Performance optimization
- [ ] Capacity planning
- [ ] Disaster recovery test

## 🔍 Monitoring Commands

### System Health
```bash
# Check all services
systemctl status cla-training.service
systemctl status cla.service

# Check disk space
df -h /opt/ai-driven-soc/

# Check memory usage
free -h

# Check network connectivity
ping google.com
```

### Model Performance
```bash
# Latest model metrics
python3 -c "
import json, glob
latest = max(glob.glob('/opt/ai-driven-soc/models/real_data_metrics_*.json'))
with open(latest) as f: 
    metrics = json.load(f)
    for k, v in metrics.items():
        if k in ['precision', 'recall', 'f1_score', 'accuracy']:
            print(f'{k}: {v:.1%}')
"

# Model age
ls -la /opt/ai-driven-soc/models/real_data_model_*.pkl | tail -1
```

### Data Quality
```bash
# Check data freshness
bq query --use_legacy_sql=false "
SELECT 
  MAX(timestamp) as latest_alert,
  COUNT(*) as total_alerts,
  COUNT(DISTINCT classification) as unique_classifications
FROM \`chronicle-dev-2be9.soc_data.processed_alerts\`
WHERE classification IS NOT NULL
"

# Expected: recent timestamp, >50K alerts, 2 classifications
```

## 📞 Escalation Procedures

### Level 1: Automated Monitoring
- System monitors itself every 4 hours
- Alerts sent to operations team
- Auto-retry failed training

### Level 2: Operations Team
- Investigate alerts within 1 hour
- Manual retraining if needed
- Check system resources

### Level 3: Development Team
- Performance degradation > 24 hours
- Data quality issues
- System architecture problems

### Level 4: Emergency
- Complete system failure
- Data corruption
- Security incidents

## 📊 Performance Benchmarks

### Expected Performance
| Metric | Target | Warning | Critical |
|--------|--------|---------|----------|
| Precision | 100% | < 98% | < 95% |
| Recall | 100% | < 98% | < 95% |
| F1-Score | 100% | < 98% | < 95% |
| Training Time | < 30s | > 45s | > 60s |
| Model Size | ~300MB | > 500MB | > 1GB |

### Data Requirements
| Metric | Target | Warning | Critical |
|--------|--------|---------|----------|
| Sample Count | > 10K | < 5K | < 1K |
| Anomaly % | 75-85% | < 70% | < 50% |
| Data Age | < 24h | > 48h | > 72h |

## 🔄 Backup and Recovery

### Backup Locations
- **Models**: `/opt/ai-driven-soc/models/backup/`
- **Configs**: `/opt/ai-driven-soc/config/`
- **Scripts**: `/opt/ai-driven-soc/scripts/`
- **Logs**: `/opt/ai-driven-soc/logs/`

### Recovery Procedures
```bash
# Full system recovery
cd /opt/ai-driven-soc
git pull origin main  # If using version control
cp -r backup/* models/
/opt/ai-driven-soc/scripts/automated_training.sh
```

### Disaster Recovery
1. **Immediate**: Use latest backup model
2. **Short-term**: Retrain with available data
3. **Long-term**: Restore from offsite backup

---

## 🎯 Success Indicators

System is healthy when:
- ✅ Daily training completes successfully
- ✅ Model performance > 95%
- ✅ Dashboard accessible and updated
- ✅ No critical alerts in logs
- ✅ Data freshness < 24 hours

---

*Quick Reference v1.0 - September 20, 2025*


