# Foundation Dashboard V1.0 - Quick Reference

**Status:** ✅ **STABLE PRODUCTION VERSION**  
**Date:** October 1, 2025  
**Purpose:** Baseline version for all future enhancements

---

## 📁 Foundation Files

| File | Purpose | Location |
|------|---------|----------|
| `complete_operational_dashboard_FOUNDATION_V1_20251001.py` | **Foundation dashboard code** | Root directory |
| `docs/FOUNDATION_DASHBOARD_V1.md` | **Markdown documentation** | docs/ |
| `docs/FOUNDATION_DASHBOARD_V1.docx` | **DOCX documentation** | docs/ |
| `RESTORE_FOUNDATION.sh` | **Restoration script** | Root directory |

---

## 🚀 Quick Restore

### Method 1: Using Restoration Script (Recommended)
```bash
cd /Users/raditio.ghifiardigmail.com/Downloads/ai-driven-soc
./RESTORE_FOUNDATION.sh
```

### Method 2: Manual Restore
```bash
# 1. Restore local file
cd /Users/raditio.ghifiardigmail.com/Downloads/ai-driven-soc
cp complete_operational_dashboard_FOUNDATION_V1_20251001.py complete_operational_dashboard.py

# 2. Deploy to VM
gcloud compute scp complete_operational_dashboard.py \
  app@xdgaisocapp01:/home/app/ai-driven-soc/ \
  --zone=asia-southeast2-a

# 3. Restart
gcloud compute ssh app@xdgaisocapp01 \
  --zone=asia-southeast2-a \
  --tunnel-through-iap \
  --command='cd /home/app/ai-driven-soc && ./restart_dashboard.sh'
```

---

## ✅ Foundation Features

### Core Capabilities
- ✅ Real-time alert review from BigQuery
- ✅ Automatic alert deduplication (by alert_id)
- ✅ Stable alert selection (dropdown matches details)
- ✅ Feedback submission to BigQuery
- ✅ Automatic alert removal after feedback
- ✅ AI chat assistant (Google Gemini Flash 2.5)
- ✅ Security operations funnel visualization
- ✅ Analytics dashboard with model metrics
- ✅ Timezone conversion (Jakarta/WIB)

### Resolved Issues
- ✅ Alert selection mismatch fixed
- ✅ Duplicate alert IDs removed
- ✅ "Is Anomaly" NA handling
- ✅ Feedback loop working correctly
- ✅ Session state stability

---

## 📊 Dashboard Access

**URL:** `http://10.45.254.19:8535`

**Tabs:**
1. **Alert Review** - Review and provide feedback on alerts
2. **Security Operations Funnel** - Visualize alert processing pipeline
3. **Analytics** - Monitor AI model performance

---

## 🔧 Troubleshooting

### Dashboard Not Loading?
```bash
# Check if running
gcloud compute ssh app@xdgaisocapp01 \
  --zone=asia-southeast2-a \
  --tunnel-through-iap \
  --command='ss -tlnp | grep 8535'
```

### Check Logs
```bash
gcloud compute ssh app@xdgaisocapp01 \
  --zone=asia-southeast2-a \
  --tunnel-through-iap \
  --command='tail -n 100 /home/app/ai-driven-soc/main_dashboard.log'
```

### Force Restart
```bash
gcloud compute ssh app@xdgaisocapp01 \
  --zone=asia-southeast2-a \
  --tunnel-through-iap \
  --command='cd /home/app/ai-driven-soc && ./restart_dashboard.sh'
```

---

## 📝 Making Changes

### Before Modifying
1. **Create backup:**
   ```bash
   cp complete_operational_dashboard.py \
      complete_operational_dashboard_backup_$(date +%Y%m%d_%H%M%S).py
   ```

2. **Test changes incrementally**

3. **If something breaks, restore foundation:**
   ```bash
   ./RESTORE_FOUNDATION.sh
   ```

### Change Workflow
```
1. Backup current version
2. Make small changes
3. Test on VM
4. If successful → Document
5. If failed → Restore foundation
```

---

## 📚 Documentation

**Full Documentation:**
- **Markdown:** `docs/FOUNDATION_DASHBOARD_V1.md`
- **DOCX:** `docs/FOUNDATION_DASHBOARD_V1.docx`

**Documentation Includes:**
- Architecture overview
- Technical implementation details
- Data flow diagrams
- Code sections with explanations
- Known issues and solutions
- Future enhancement guidelines
- Database schema
- Deployment instructions

---

## 🎯 Key Technical Details

### Alert Processing Flow
```
BigQuery → get_real_alerts() → Deduplication → Display → 
  → Analyst Review → Feedback → BigQuery → CLA Training
```

### Critical Code Sections
1. **Deduplication Logic** (Lines 1106-1113)
2. **Alert Selection** (Lines 1225-1246)
3. **Feedback Submission** (Lines 142-161)
4. **BigQuery Integration** (Lines 163-206)

### Database Tables
- `soc_data.processed_alerts` - Alert data
- `soc_data.feedback` - Analyst feedback
- `soc_data.cla_metrics` - Model performance

---

## ⚡ Quick Commands

### Deploy Current File
```bash
gcloud compute scp complete_operational_dashboard.py \
  app@xdgaisocapp01:/home/app/ai-driven-soc/ --zone=asia-southeast2-a && \
gcloud compute ssh app@xdgaisocapp01 --zone=asia-southeast2-a \
  --tunnel-through-iap --command='cd /home/app/ai-driven-soc && ./restart_dashboard.sh'
```

### Check Dashboard Status
```bash
gcloud compute ssh app@xdgaisocapp01 --zone=asia-southeast2-a \
  --tunnel-through-iap --command='ss -tlnp | grep 8535 && echo "✅ Running" || echo "❌ Not running"'
```

### View Recent Logs
```bash
gcloud compute ssh app@xdgaisocapp01 --zone=asia-southeast2-a \
  --tunnel-through-iap --command='tail -n 50 /home/app/ai-driven-soc/main_dashboard.log'
```

---

## 🛡️ Stability Guarantee

This foundation version has been tested and verified to work correctly with:
- Real BigQuery data
- Alert selection and display synchronization
- Feedback submission and storage
- AI chat integration
- All dashboard tabs

**If any future changes cause instability, you can ALWAYS restore to this working version using the restoration script.**

---

## 📞 Support

For issues or questions:
1. Check `docs/FOUNDATION_DASHBOARD_V1.md` for detailed documentation
2. Review troubleshooting section above
3. Restore to foundation if needed: `./RESTORE_FOUNDATION.sh`

---

**Foundation Dashboard V1.0**  
*Your Stable Baseline for AI-Driven SOC Operations* ✅

