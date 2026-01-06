# BigQuery Integration Setup Guide

## ✅ Integration Status

The BigQuery integration code is **ready and implemented**. The dashboards will automatically:
- ✅ Connect to BigQuery when credentials are available
- ✅ Fall back to demo data if BigQuery is unavailable
- ✅ Display connection status clearly

## 🔐 Current Issue: Permissions

**Error:** `Access Denied: User does not have bigquery.jobs.create permission`

This means the integration is working, but GCP permissions need to be configured.

## 🚀 Setup Steps

### Option 1: Use Service Account (Recommended for Production)

1. **Create Service Account:**
```bash
# In GCP Console or via gcloud CLI
gcloud iam service-accounts create soc-dashboard-sa \
    --display-name="SOC Dashboard Service Account" \
    --project=chronicle-dev-2be9
```

2. **Grant BigQuery Permissions:**
```bash
# Grant BigQuery Data Viewer and Job User roles
gcloud projects add-iam-policy-binding chronicle-dev-2be9 \
    --member="serviceAccount:soc-dashboard-sa@chronicle-dev-2be9.iam.gserviceaccount.com" \
    --role="roles/bigquery.dataViewer"

gcloud projects add-iam-policy-binding chronicle-dev-2be9 \
    --member="serviceAccount:soc-dashboard-sa@chronicle-dev-2be9.iam.gserviceaccount.com" \
    --role="roles/bigquery.jobUser"
```

3. **Create and Download Key:**
```bash
gcloud iam service-accounts keys create chronicle-key.json \
    --iam-account=soc-dashboard-sa@chronicle-dev-2be9.iam.gserviceaccount.com \
    --project=chronicle-dev-2be9
```

4. **Set Environment Variable:**
```bash
export GOOGLE_APPLICATION_CREDENTIALS="path/to/chronicle-key.json"
```

### Option 2: Use Application Default Credentials (ADC)

1. **Authenticate:**
```bash
gcloud auth application-default login
```

2. **Grant Your User Account Permissions:**
   - Go to GCP Console → IAM & Admin → IAM
   - Find your user account
   - Add roles:
     - `BigQuery Data Viewer`
     - `BigQuery Job User`

### Option 3: Use Existing Service Account Key

If you already have a service account key:

1. **Place the key file:**
```bash
# Option A: In dashboards directory
cp /path/to/your-key.json dashboards/chronicle-key.json

# Option B: Set environment variable
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your-key.json"
```

2. **Verify it works:**
```bash
cd dashboards
source venv/bin/activate
python3 -c "from bigquery_config import test_bigquery_connection; print(test_bigquery_connection())"
```

## 📊 Available Tables

Based on your BigQuery project, these tables are available:

- `gatra_database.siem_events` - SIEM security events
- `gatra_database.activity_logs` - Activity logs
- `gatra_database.ada_agent_metri` - ADA agent metrics
- `gatra_database.ada_features` - ADA features
- `gatra_database.ada_feedback` - ADA feedback
- `gatra_database.ada_ml_results` - ADA ML results
- `gatra_database.ada_state` - ADA state
- `soc_data.thor_scan_results` - THOR scan results (threat hunting)
- `soc_data.threat_intel` - Threat intelligence

## 🔧 Configuration

### Update Project/Dataset (if needed)

Edit `bigquery_config.py`:

```python
GCP_PROJECT_ID = "chronicle-dev-2be9"  # Your project ID
BIGQUERY_DATASET = "gatra_database"     # Your dataset name
```

### Test Connection

```bash
cd dashboards
source venv/bin/activate
./setup_bigquery_integration.sh
```

Or manually:
```bash
python3 << 'PYTEST'
from bigquery_config import test_bigquery_connection, get_available_tables
result = test_bigquery_connection()
print(result)
if result['success']:
    print("Available tables:", get_available_tables())
PYTEST
```

## 🎯 Dashboard Behavior

### With BigQuery Connected:
- ✅ Shows "🎉 CONNECTED TO LIVE BIGQUERY DATA"
- ✅ Displays real data from your tables
- ✅ Real-time statistics and metrics
- ✅ Live threat intelligence

### Without BigQuery (Current State):
- ⚠️ Shows "⚠️ BigQuery Offline - Using Enhanced Demo Data"
- ✅ Uses high-quality mock data
- ✅ All dashboard features work normally
- ✅ Perfect for testing and demos

## 📝 Required IAM Roles

For the service account or user account, you need:

1. **BigQuery Data Viewer** (`roles/bigquery.dataViewer`)
   - Read data from tables
   - View datasets

2. **BigQuery Job User** (`roles/bigquery.jobUser`)
   - Run queries
   - Create jobs

3. **Optional: BigQuery Data Editor** (`roles/bigquery.dataEditor`)
   - If you want to write data back

## 🔍 Troubleshooting

### Error: "Access Denied"
**Solution:** Grant BigQuery permissions to your account/service account

### Error: "API not enabled"
**Solution:** Enable BigQuery API:
```bash
gcloud services enable bigquery.googleapis.com --project=chronicle-dev-2be9
```

### Error: "Table not found"
**Solution:** 
- Check table name in `bigquery_config.py`
- Verify dataset exists
- Check table permissions

### Error: "Credentials not found"
**Solution:**
- Set `GOOGLE_APPLICATION_CREDENTIALS` environment variable
- Or run `gcloud auth application-default login`

## ✅ Quick Test

After setting up permissions:

```bash
cd dashboards
source venv/bin/activate
streamlit run enhanced_soc_dashboard.py
```

The dashboard will show connection status at the top.

---

**Note:** The dashboards work perfectly with demo data even if BigQuery is not configured. BigQuery integration is optional but recommended for production use.

