# Port 99 Dashboard Analysis - TAA Service Identification 🔍

## 🎯 **PORT 99 DASHBOARD IDENTIFIED:**

The dashboard on `http://10.45.254.19:99` is running a **Docker container** with a different TAA service configuration.

## ✅ **DOCKER CONTAINER DETAILS:**

### **Container Information:**
- **Container Name**: `dashboard-gatra_dashboard-1`
- **Image**: `dashboard-gatra_dashboard`
- **Port Mapping**: `0.0.0.0:99->8501/tcp`
- **Status**: ✅ Running (Up 24 hours)
- **Process**: `streamlit run app.py`

### **Dashboard Structure:**
```
/app/
├── app.py (Main navigation)
├── pages/
│   ├── main.py (Overview)
│   ├── agent.py (Agent Performance)
│   ├── feedback.py (Human Feedback) ⚠️ **Has filtered_alerts error**
│   └── chatbot.py (AI Analytics)
```

## 🔧 **TAA SERVICE CONFIGURATION:**

### **Data Source:**
The Docker container dashboard uses **`taa_state`** table instead of `dashboard_alerts`:

```sql
SELECT id, alarm_id, confidence, severity, valid, run_time,
       created_at, reasoning, remarks
FROM `chronicle-dev-2be9.gatra_database.taa_state`
```

### **TAA Agent Configuration:**
- **Table**: `chronicle-dev-2be9.gatra_database.taa_state`
- **Columns**: id, alarm_id, confidence, severity, valid, run_time, created_at, reasoning, remarks
- **Agent Type**: "TAA (Triage & Analysis)"
- **Data Format**: Different structure than `dashboard_alerts`

## 🚨 **IDENTIFIED ISSUE:**

### **Human Feedback Page Error:**
The Docker container's `feedback.py` has the same `filtered_alerts` error:
- **Line 846**: `filtered_alerts = all_alerts`
- **Line 906**: `if not filtered_alerts.empty:`
- **Error**: `NameError: name 'filtered_alerts' is not defined`

## 📊 **COMPARISON: PORT 99 vs PORT 8511**

| Aspect | Port 99 (Docker) | Port 8511 (Native) |
|--------|------------------|-------------------|
| **Environment** | Docker Container | Native Streamlit |
| **TAA Table** | `taa_state` | `dashboard_alerts` |
| **Data Structure** | id, alarm_id, confidence, severity, valid, run_time, created_at, reasoning, remarks | alarm_id, ada_case_class, ada_confidence, taa_confidence, ada_score, taa_severity, ada_reasoning, taa_reasoning, taa_created |
| **Data Source** | TAA State Table | Dashboard Alerts Table |
| **Records** | Unknown count | 1,000 records |
| **Status** | Has filtered_alerts error | Working with real data |

## 🎯 **TAA SERVICE ANSWER:**

### **Port 99 Dashboard Uses:**
- **TAA Table**: `chronicle-dev-2be9.gatra_database.taa_state`
- **TAA Service**: Connected to the `taa_state` table (likely from the original TAA service)
- **Data Format**: TAA-specific columns (confidence, severity, valid, run_time, reasoning)
- **Status**: Independent of the TAA services we killed (uses different data source)

### **Key Differences:**
1. **Port 99**: Uses `taa_state` table (TAA service state data)
2. **Port 8511**: Uses `dashboard_alerts` table (Dashboard-specific data)
3. **Port 99**: Docker containerized dashboard
4. **Port 8511**: Native Streamlit dashboard

## 🚀 **RECOMMENDATIONS:**

### **For Port 99 Dashboard:**
1. **Fix filtered_alerts error** in Docker container
2. **Update to use dashboard_alerts** for consistency
3. **Or keep separate** if `taa_state` has different purpose

### **For Port 8511 Dashboard:**
1. **Already working** with real `dashboard_alerts` data
2. **Has 1,000 records** with numeric alarm IDs
3. **No TAA service dependency** (uses static data)

## 🎉 **SUMMARY:**

**Port 99 Dashboard (`http://10.45.254.19:99`) uses:**
- ✅ **TAA Table**: `chronicle-dev-2be9.gatra_database.taa_state`
- ✅ **Docker Environment**: Containerized Streamlit app
- ✅ **Independent**: Not affected by TAA services we killed
- ⚠️ **Has Error**: `filtered_alerts` NameError needs fixing
- ✅ **Different Data**: Uses TAA state data vs dashboard alerts data

**The Port 99 dashboard is using the `taa_state` table, which is independent of the TAA services we killed and represents the TAA service's internal state data.**

