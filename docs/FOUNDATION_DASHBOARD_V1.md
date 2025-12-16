# Foundation Dashboard V1.0 - Documentation

**Date Created:** October 1, 2025  
**Status:** Stable Foundation Version  
**File:** `complete_operational_dashboard.py`

---

## Executive Summary

This document serves as the **official foundation baseline** for the AI-Driven SOC Operational Dashboard. All future changes should be able to restore to this stable version.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Features](#core-features)
4. [Technical Implementation](#technical-implementation)
5. [Data Flow](#data-flow)
6. [Key Components](#key-components)
7. [Deployment](#deployment)
8. [Restoration Instructions](#restoration-instructions)
9. [Known Issues & Solutions](#known-issues--solutions)
10. [Future Enhancement Guidelines](#future-enhancement-guidelines)

---

## Overview

### Purpose
The Foundation Dashboard provides SOC analysts with a comprehensive, real-time interface for:
- Reviewing security alerts
- Providing feedback for continuous learning
- Monitoring AI model performance
- Tracking security operations metrics
- Engaging with AI-powered threat analysis

### Key Design Principles
1. **Stability First**: Every feature is tested and proven to work reliably
2. **Real-Time Data**: All data comes from BigQuery, no mock data
3. **User-Centric**: Designed for SOC analyst workflow
4. **Feedback Loop**: Enables continuous model improvement
5. **Data Integrity**: Ensures alert selection matches displayed details

---

## Architecture

### Technology Stack
```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                        │
│                (complete_operational_dashboard.py)           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Google Cloud BigQuery Backend                   │
├─────────────────────────────────────────────────────────────┤
│  • chronicle-dev-2be9.soc_data.processed_alerts             │
│  • chronicle-dev-2be9.soc_data.feedback                     │
│  • chronicle-dev-2be9.soc_data.cla_metrics                  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Google Gemini Flash 2.5 (AI Chat)              │
└─────────────────────────────────────────────────────────────┘
```

### Deployment Architecture
```
┌─────────────────────────────────────────────────────────────┐
│           GCP VM: xdgaisocapp01                             │
│           Zone: asia-southeast2-a                           │
│           Port: 8535                                        │
├─────────────────────────────────────────────────────────────┤
│  Virtual Environment: /home/app/ai-driven-soc/venv/         │
│  Service Account: soc-bigquery-access@...                  │
│  Auto-Restart: ./restart_dashboard.sh                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Features

### 1. Alert Review & Feedback
**Purpose**: Enable SOC analysts to review and provide feedback on security alerts

**Key Capabilities**:
- ✅ Display recent alerts from BigQuery (last 24 hours)
- ✅ Deduplicate alerts by alert_id (keeps most recent)
- ✅ Aggregate similar alerts by IP addresses and classification
- ✅ Detailed alert analysis with network flow, data transfer, threat assessment
- ✅ Feedback submission to BigQuery
- ✅ Automatic alert removal after feedback submission
- ✅ Real-time alert selection sync

**Workflow**:
```
1. Load alerts from BigQuery
2. Deduplicate by alert_id
3. Display in Recent Alerts table
4. Analyst selects alert from dropdown
5. Review detailed analysis
6. Submit feedback (True Positive/False Positive)
7. Alert removed from queue
8. Feedback stored in BigQuery for CLA training
```

### 2. Security Operations Funnel
**Purpose**: Visualize the security alert processing pipeline

**Metrics Displayed**:
- Total Alerts Detected
- After Initial Triage
- Requiring Investigation
- Confirmed Threats
- Successfully Mitigated

**Features**:
- Real-time funnel visualization
- Detailed stage descriptions
- Current status indicators
- Processing metrics

### 3. Analytics Dashboard
**Purpose**: Monitor AI model performance and system metrics

**Key Metrics**:
- Model Accuracy (CLA)
- False Positive Rate
- Precision & Recall
- Alert Classification Distribution
- Confidence Score Distribution
- Daily Alert Trends

### 4. AI Assistant Chat
**Purpose**: Provide intelligent threat analysis assistance

**Capabilities**:
- ✅ Powered by Google Gemini Flash 2.5
- ✅ Context-aware responses based on selected alert
- ✅ Real-time threat intelligence
- ✅ Investigation guidance
- ✅ Fallback to enhanced predefined responses

---

## Technical Implementation

### File Structure
```
/home/app/ai-driven-soc/
├── complete_operational_dashboard.py    # Main dashboard file (Foundation)
├── restart_dashboard.sh                 # Auto-restart script
├── venv/                               # Python virtual environment
├── main_dashboard.log                  # Application logs
└── Service Account BigQuery/           # GCP credentials
    └── soc-bigquery-access-*.json
```

### Key Functions

#### 1. `get_bigquery_client()`
```python
def get_bigquery_client():
    """Initialize BigQuery client with service account authentication"""
    # Returns: google.cloud.bigquery.Client or None
```

#### 2. `get_real_alerts()`
```python
@st.cache_data(ttl=60)
def get_real_alerts():
    """
    Fetch alerts from BigQuery soc_data.processed_alerts
    - Filters: Last 24 hours
    - Limit: 100 alerts
    - Returns: pandas.DataFrame with deduplication
    """
```

#### 3. `submit_real_feedback(alert_id, is_true_positive, confidence, comments)`
```python
def submit_real_feedback(alert_id: str, is_true_positive: bool, 
                        confidence: float, comments: str) -> bool:
    """
    Submit feedback to BigQuery and remove alert from review queue
    - Stores in: soc_data.feedback
    - Updates: st.session_state.reviewed_alerts
    - Clears cache to refresh display
    """
```

#### 4. `alert_review_page()`
```python
def alert_review_page():
    """
    Main alert review interface
    - Loads alerts into session state
    - Deduplicates by alert_id (keeps most recent)
    - Displays Recent Alerts table
    - Handles alert selection and display
    - Manages feedback submission
    """
```

### Critical Implementation Details

#### Alert Deduplication Logic
```python
# Line 1106-1113: Remove duplicate alert_ids
if not alerts_df.empty and 'alert_id' in alerts_df.columns:
    # Sort by timestamp descending
    if 'timestamp' in alerts_df.columns:
        alerts_df = alerts_df.sort_values('timestamp', ascending=False)
    # Drop duplicates, keep first (most recent)
    alerts_df = alerts_df.drop_duplicates(subset='alert_id', keep='first')
    # Update session state
    st.session_state.alerts_df = alerts_df.copy()
```

#### Alert Selection Mechanism
```python
# Line 1225-1246: Unique alert selection with dynamic key
seen = set()
unique_alert_ids = []
for alert_id in alert_ids:
    if alert_id not in seen:
        seen.add(alert_id)
        unique_alert_ids.append(alert_id)

selected_alert = st.selectbox(
    "Select Alert for Feedback",
    options=unique_alert_ids,
    index=default_index,
    key=f'alert_selector_{len(alert_ids)}'  # Dynamic key prevents stale state
)

st.session_state.last_selected_alert = selected_alert
```

#### Feedback Submission & Alert Removal
```python
# Line 142-161: Feedback with automatic alert removal
def submit_real_feedback(alert_id, is_true_positive, confidence, comments):
    # 1. Submit to BigQuery
    errors = client.insert_rows_json(table, [row])
    
    # 2. Mark as reviewed
    if 'reviewed_alerts' not in st.session_state:
        st.session_state.reviewed_alerts = set()
    st.session_state.reviewed_alerts.add(alert_id)
    
    # 3. Remove from display
    if 'alerts_df' in st.session_state:
        st.session_state.alerts_df = st.session_state.alerts_df[
            st.session_state.alerts_df['alert_id'] != alert_id
        ]
    
    # 4. Clear cache
    get_real_alerts.clear()
```

---

## Data Flow

### Alert Processing Flow
```
┌─────────────────────────────────────────────────────────────┐
│  1. BigQuery: soc_data.processed_alerts                     │
│     └─ Last 24 hours, max 100 alerts                        │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  2. get_real_alerts() - Cached 60 seconds                   │
│     └─ Filter reviewed alerts                               │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  3. alert_review_page() - Deduplicate                       │
│     └─ Sort by timestamp DESC                               │
│     └─ Drop duplicates (keep first)                         │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  4. Display in Streamlit                                    │
│     └─ Recent Alerts table                                  │
│     └─ Alert selection dropdown (unique IDs)                │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  5. Analyst Review & Feedback                               │
│     └─ Select alert                                         │
│     └─ Review details                                       │
│     └─ Submit feedback                                      │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  6. BigQuery: soc_data.feedback                             │
│     └─ Store feedback for CLA training                      │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  7. Alert Removed from Queue                                │
│     └─ Added to st.session_state.reviewed_alerts            │
└─────────────────────────────────────────────────────────────┘
```

### Feedback Loop for Continuous Learning
```
Analyst Feedback → BigQuery (soc_data.feedback)
                         ↓
              CLA Hourly Retrain Job
         (cla_hourly_retrain.py + start_cla_hourly_retrain.sh)
                         ↓
              Updated CLA Metrics → BigQuery (soc_data.cla_metrics)
                         ↓
                 Analytics Dashboard
                         ↓
              Improved Alert Classification
```

---

## Key Components

### 1. Recent Alerts Table
**Location**: Alert Review tab  
**Data Source**: BigQuery `soc_data.processed_alerts`  
**Columns**:
- `alert_id` (Primary Key)
- `timestamp` (Jakarta timezone: Asia/Jakarta)
- `severity` (High/Medium/Low)
- `classification` (anomaly/threat/benign)
- `confidence_score` (0.0 - 1.0)
- `is_anomaly` (Yes/No/Unknown)

**Features**:
- ✅ Deduplication by alert_id
- ✅ Sorted by timestamp (most recent first)
- ✅ Optional aggregation by IP set + classification
- ✅ Real-time refresh button

### 2. Alert Details for Review
**Purpose**: Display comprehensive alert information for analyst review

**Sections**:
1. **Basic Information**
   - Alert ID, Timestamp (Jakarta), Severity, Confidence Score, Classification, Is Anomaly

2. **Extracted Parameters**
   - IOCs (IPs, Domains, URLs, File Hashes)
   - Network Flow Details
   - Data Transfer Analysis
   - User/Asset Information

3. **Threat Assessment**
   - Confidence Analysis
   - Severity Analysis
   - Key Investigation Questions

4. **Enhanced Recommended Actions**
   - Threat Intelligence Lookup (VirusTotal, AbuseIPDB, Shodan)
   - Contextual Bandit Analysis
   - RAG-Enhanced Context
   - TTP Analysis & MITRE ATT&CK Mapping
   - Historical Incident Correlation
   - Detailed Investigative Steps
   - Risk-Based Immediate Actions

5. **Raw Alert Data** (Expandable JSON)

### 3. Feedback Submission Form
**Location**: Below Alert Details

**Fields**:
- **Alert Selection**: Dropdown with unique alert IDs
- **Is True Positive**: Radio button (Yes/No)
- **Confidence**: Slider (0.0 - 1.0)
- **Comments**: Text area for analyst notes

**Behavior**:
- ✅ Verification display showing Dropdown vs Fetched alert ID
- ✅ Automatic alert removal after submission
- ✅ Success confirmation message
- ✅ Data persisted to BigQuery `soc_data.feedback`

### 4. AI Assistant Chat
**Location**: Right sidebar in Alert Review tab

**Features**:
- Context-aware responses based on selected alert
- Real-time LLM integration (Google Gemini Flash 2.5)
- Fallback to enhanced predefined responses
- LLM status indicator
- Chat history preservation

**Model Configuration**:
```python
model_name = "gemini-1.5-flash-latest"  # Primary
fallback_models = ["gemini-1.5-flash", "gemini-pro", "gemini-pro-vision"]
```

---

## Deployment

### Prerequisites
1. GCP VM Instance: `xdgaisocapp01`
2. Service Account with BigQuery access
3. Python 3.11+ with virtual environment
4. Required Python packages (see requirements.txt)

### Deployment Steps

#### 1. Copy Foundation Dashboard to VM
```bash
gcloud compute scp complete_operational_dashboard.py \
  app@xdgaisocapp01:/home/app/ai-driven-soc/ \
  --zone=asia-southeast2-a
```

#### 2. Restart Dashboard
```bash
gcloud compute ssh app@xdgaisocapp01 \
  --zone=asia-southeast2-a \
  --tunnel-through-iap \
  --command='cd /home/app/ai-driven-soc && ./restart_dashboard.sh'
```

#### 3. Verify Deployment
```bash
gcloud compute ssh app@xdgaisocapp01 \
  --zone=asia-southeast2-a \
  --tunnel-through-iap \
  --command='ss -tlnp | grep 8535'
```

Expected output:
```
LISTEN 0 128 0.0.0.0:8535 0.0.0.0:* users:(("python3",pid=XXXXX,fd=6))
```

#### 4. Access Dashboard
```
http://10.45.254.19:8535
```

### Auto-Restart Configuration

**File**: `restart_dashboard.sh`

```bash
#!/bin/bash
# Kill existing process
pgrep -f "streamlit run complete_operational_dashboard.py" | xargs -r kill -9

# Start dashboard in background
cd /home/app/ai-driven-soc
source venv/bin/activate
nohup python3 -m streamlit run complete_operational_dashboard.py \
  --server.port 8535 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.fileWatcherType none \
  --logger.level error \
  > main_dashboard.log 2>&1 &

# Wait and verify
sleep 3
if ss -tlnp | grep -q 8535; then
    echo "✅ Dashboard is running on port 8535"
else
    echo "❌ Dashboard failed to start"
fi
```

---

## Restoration Instructions

### Quick Restore from Local
```bash
# 1. Navigate to project directory
cd /Users/raditio.ghifiardigmail.com/Downloads/ai-driven-soc

# 2. Copy foundation file
cp complete_operational_dashboard.py complete_operational_dashboard_backup_$(date +%Y%m%d_%H%M%S).py

# 3. Deploy to VM
gcloud compute scp complete_operational_dashboard.py \
  app@xdgaisocapp01:/home/app/ai-driven-soc/ \
  --zone=asia-southeast2-a

# 4. Restart
gcloud compute ssh app@xdgaisocapp01 \
  --zone=asia-southeast2-a \
  --tunnel-through-iap \
  --command='cd /home/app/ai-driven-soc && ./restart_dashboard.sh'
```

### Restore from VM Backup
```bash
# 1. List available backups on VM
gcloud compute ssh app@xdgaisocapp01 \
  --zone=asia-southeast2-a \
  --tunnel-through-iap \
  --command='ls -lh /home/app/ai-driven-soc/*dashboard*.py'

# 2. Restore specific backup
gcloud compute ssh app@xdgaisocapp01 \
  --zone=asia-southeast2-a \
  --tunnel-through-iap \
  --command='cd /home/app/ai-driven-soc && \
             cp complete_operational_dashboard_backup_YYYYMMDD_HHMMSS.py \
             complete_operational_dashboard.py && \
             ./restart_dashboard.sh'
```

### Restore from Documentation
This markdown file includes all critical code sections. If needed:

1. Extract code blocks from this document
2. Recreate `complete_operational_dashboard.py`
3. Deploy using standard deployment steps

---

## Known Issues & Solutions

### Issue 1: Alert Selection Mismatch
**Symptom**: Dropdown shows one alert_id, but details show a different alert_id

**Root Cause**: Streamlit session state with `key` parameter causing stale data

**Solution** (Implemented in Foundation):
```python
# Use dynamic key that forces refresh
selected_alert = st.selectbox(
    "Select Alert for Feedback",
    options=unique_alert_ids,
    index=default_index,
    key=f'alert_selector_{len(alert_ids)}'  # Dynamic key
)
```

### Issue 2: Duplicate Alert IDs in Dropdown
**Symptom**: Same alert_id appears multiple times in selection dropdown

**Root Cause**: BigQuery data contains duplicate alert_ids (from reprocessing/updates)

**Solution** (Implemented in Foundation):
```python
# Deduplication at data load
alerts_df = alerts_df.sort_values('timestamp', ascending=False)
alerts_df = alerts_df.drop_duplicates(subset='alert_id', keep='first')

# Additional deduplication at dropdown
seen = set()
unique_alert_ids = []
for alert_id in alert_ids:
    if alert_id not in seen:
        seen.add(alert_id)
        unique_alert_ids.append(alert_id)
```

### Issue 3: "Is Anomaly" Showing "Unknown"
**Symptom**: `TypeError: boolean value of NA is ambiguous`

**Root Cause**: Pandas NA values in boolean column

**Solution** (Implemented in Foundation):
```python
# Handle NA values explicitly
display_df['is_anomaly'] = display_df['is_anomaly'].apply(
    lambda x: "Unknown" if pd.isna(x) else ("Yes" if x else "No")
)
```

### Issue 4: Dashboard Not Auto-Refreshing After Changes
**Symptom**: Dashboard doesn't reflect updates after file changes

**Root Cause**: Streamlit file watcher disabled for stability

**Solution**: Always run `./restart_dashboard.sh` after code changes

### Issue 5: BigQuery Client Not Available
**Symptom**: Dashboard shows "BigQuery client not available"

**Root Cause**: Service account credentials not properly configured

**Solution**:
```bash
# 1. Verify service account file exists
ls -l /home/app/ai-driven-soc/Service\ Account\ BigQuery/*.json

# 2. Check environment variable
echo $GOOGLE_APPLICATION_CREDENTIALS

# 3. Ensure proper permissions
chmod 600 /home/app/ai-driven-soc/Service\ Account\ BigQuery/*.json
```

---

## Future Enhancement Guidelines

### Before Making Changes
1. **Create Backup**:
   ```bash
   cp complete_operational_dashboard.py \
      complete_operational_dashboard_backup_$(date +%Y%m%d_%H%M%S).py
   ```

2. **Test Locally** (if possible)

3. **Deploy Incrementally**: Test each change separately

4. **Document Changes**: Update this document with new features

### Recommended Change Process
```
1. Backup current working version
2. Make small, focused changes
3. Test on VM
4. If successful, update documentation
5. If failed, restore from backup
6. Repeat for next change
```

### Critical Files to Backup
- `complete_operational_dashboard.py` (Main dashboard)
- `restart_dashboard.sh` (Restart script)
- `cla_hourly_retrain.py` (CLA training job)
- `start_cla_hourly_retrain.sh` (CLA scheduler)

### Areas for Future Enhancement
1. **Alert Filtering**
   - Add severity filter
   - Add classification filter
   - Add date range picker

2. **Advanced Analytics**
   - Real-time model performance graphs
   - Analyst productivity metrics
   - Alert trend predictions

3. **Collaboration Features**
   - Multi-analyst assignment
   - Comment threads on alerts
   - Escalation workflows

4. **Threat Intelligence Integration**
   - Automated TI enrichment
   - STIX/TAXII integration
   - Custom threat feeds

5. **Automation**
   - Auto-response for low-risk alerts
   - Playbook execution
   - SOAR integration enhancements

---

## Appendix A: Critical Code Sections

### BigQuery Client Initialization
```python
def get_bigquery_client():
    """Initialize BigQuery client with service account"""
    try:
        service_account_files = glob.glob(
            '/home/app/ai-driven-soc/Service Account BigQuery/*.json'
        )
        if service_account_files:
            credentials = service_account.Credentials.from_service_account_file(
                service_account_files[0],
                scopes=["https://www.googleapis.com/auth/bigquery"]
            )
            return bigquery.Client(
                credentials=credentials,
                project=credentials.project_id
            )
    except Exception as e:
        st.error(f"BigQuery initialization error: {e}")
        return None
```

### Alert Data Extraction
```python
def extract_alert_parameters(alert_data):
    """Extract key parameters from alert for analysis"""
    params = {
        'ip_addresses': [],
        'domains': [],
        'urls': [],
        'file_hashes': [],
        'network_flow': 'N/A',
        'bytes_transferred': 0,
        'user_info': 'N/A',
        'asset_info': 'N/A'
    }
    
    if 'raw_alert' in alert_data and pd.notna(alert_data['raw_alert']):
        raw = alert_data['raw_alert']
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except:
                pass
        
        # Extract IPs, domains, etc.
        # ... (see actual implementation)
    
    return params
```

### Gemini AI Integration
```python
def get_gemini_client():
    """Initialize Google Gemini for AI chat"""
    try:
        service_account_path = '/home/app/ai-driven-soc/Service Account BigQuery/*.json'
        sa_files = glob.glob(service_account_path)
        
        if sa_files:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = sa_files[0]
        
        genai.configure(api_key=None, transport='rest')
        
        model_names = [
            "gemini-1.5-flash-latest",
            "gemini-1.5-flash",
            "gemini-pro",
            "gemini-pro-vision"
        ]
        
        for model_name in model_names:
            try:
                model = genai.GenerativeModel(model_name)
                return model
            except Exception:
                continue
        
        return None
    except Exception as e:
        return None
```

---

## Appendix B: Database Schema

### Table: soc_data.processed_alerts
```sql
CREATE TABLE `chronicle-dev-2be9.soc_data.processed_alerts` (
  alert_id STRING NOT NULL,
  timestamp TIMESTAMP NOT NULL,
  is_anomaly BOOLEAN,
  confidence_score FLOAT64,
  classification STRING,
  raw_alert JSON,
  severity STRING,
  source_ip STRING,
  destination_ip STRING,
  bytes_transferred INT64
)
```

### Table: soc_data.feedback
```sql
CREATE TABLE `chronicle-dev-2be9.soc_data.feedback` (
  alert_id STRING NOT NULL,
  is_true_positive BOOLEAN NOT NULL,
  confidence FLOAT64 NOT NULL,
  analyst_comments STRING,
  timestamp TIMESTAMP NOT NULL,
  processed BOOLEAN DEFAULT FALSE
)
```

### Table: soc_data.cla_metrics
```sql
CREATE TABLE `chronicle-dev-2be9.soc_data.cla_metrics` (
  timestamp TIMESTAMP NOT NULL,
  accuracy FLOAT64,
  false_positive_rate FLOAT64,
  precision FLOAT64,
  recall FLOAT64,
  total_alerts INT64,
  true_positives INT64,
  false_positives INT64,
  true_negatives INT64,
  false_negatives INT64
)
```

---

## Appendix C: Configuration Files

### requirements.txt (Core Dependencies)
```
streamlit>=1.28.0
google-cloud-bigquery>=3.11.0
pandas>=2.0.0
pytz>=2023.3
google-generativeai>=0.3.0
plotly>=5.17.0
```

### Virtual Environment Setup
```bash
cd /home/app/ai-driven-soc
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Contact & Support

**Project Owner**: SOC Team  
**Technical Lead**: AI-Driven SOC Development Team  
**Documentation**: Foundation Dashboard V1.0  
**Last Updated**: October 1, 2025

---

## Version History

| Version | Date | Changes | Status |
|---------|------|---------|--------|
| 1.0 | 2025-10-01 | Initial Foundation Version | ✅ Stable |

---

## Signature

This document represents the **STABLE FOUNDATION** version of the AI-Driven SOC Operational Dashboard. All future enhancements should maintain backward compatibility or provide clear restoration paths to this baseline.

**Foundation Dashboard File**: `complete_operational_dashboard.py`  
**Restoration Point**: October 1, 2025  
**Status**: Production-Ready ✅

---

*End of Documentation*

