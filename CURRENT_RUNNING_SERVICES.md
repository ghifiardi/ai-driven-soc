# Current Running Services in AI-Driven SOC 🚀

## 📊 **SYSTEMD SERVICES (Active):**

### **1. AI Dashboard Service**
- **Service**: `ai-dashboard.service`
- **Status**: ✅ **RUNNING**
- **Description**: AI Model Training Dashboard
- **Port**: 8503

### **2. Continuous Learning Agent (CLA)**
- **Service**: `cla.service`
- **Status**: ✅ **RUNNING**
- **Description**: Continuous Learning Agent for AI-Driven SOC
- **Multiple Instances**: 4 running processes

### **3. Dashboard Service**
- **Service**: `dashboard.service`
- **Status**: ✅ **RUNNING**
- **Description**: AI SOC Dashboard

### **4. LangGraph ADA Services**
- **Service**: `langgraph-ada.service`
- **Status**: ✅ **RUNNING**
- **Description**: LangGraph Anomaly Detection Agent

- **Service**: `langgraph_ada.service`
- **Status**: ✅ **RUNNING**
- **Description**: LangGraph ADA BigQuery Integration Service

## 🔧 **RUNNING PROCESSES:**

### **ADA (Anomaly Detection Agent) Services:**
1. **Production Tuned ADA Agent**
   - **Process**: `production_tuned_ada_agent.py`
   - **PID**: 305713
   - **Status**: ✅ **RUNNING** (1h 28m uptime)

2. **ADA with Pub/Sub**
   - **Process**: `ada_with_pubsub.py`
   - **PID**: 305994
   - **Status**: ✅ **RUNNING** (3h 16m uptime)

3. **LangGraph ADA Integration**
   - **Process**: `fixed_langgraph_ada_integration.py`
   - **PID**: 3779938
   - **Status**: ✅ **RUNNING** (15h 21m uptime)

### **CLA (Continuous Learning Agent) Services:**
1. **CLA from Local Directory**
   - **Process**: `continuous-learning-agent.py`
   - **PID**: 1321538
   - **Status**: ✅ **RUNNING** (34h 25m uptime)

2. **CLA Complete (4 instances)**
   - **Process**: `cla_complete.py`
   - **PIDs**: 1432877, 1433071, 1433135 (and one more)
   - **Status**: ✅ **RUNNING** (24h+ uptime each)

### **Dashboard Services:**
1. **VM Dashboard API**
   - **Process**: `vm_dashboard_api.py`
   - **PID**: 2819737
   - **Status**: ✅ **RUNNING** (79h 41m uptime)

2. **Comprehensive Real Data Dashboard**
   - **Process**: `comprehensive_real_data_dashboard.py`
   - **PID**: 3078122
   - **Port**: 8527
   - **Status**: ✅ **RUNNING** (21h 58m uptime)

3. **Production Dashboard Correlator**
   - **Process**: `production_dashboard_correlator.py`
   - **PID**: 3349670
   - **Port**: 8528
   - **Status**: ✅ **RUNNING** (16h 43m uptime)

4. **AI Model Training Dashboard**
   - **Process**: `ai_model_training_dashboard.py`
   - **PID**: 3835803
   - **Port**: 8503
   - **Status**: ✅ **RUNNING**

## 🐳 **DOCKER CONTAINERS:**

### **1. Dashboard Gatra (Port 99)**
- **Container**: `dashboard-gatra_dashboard-1`
- **Image**: `dashboard-gatra_dashboard`
- **Port**: `99:8501`
- **Status**: ✅ **RUNNING** (24 hours uptime)

### **2. CrewAI API FastAPI**
- **Container**: `crewai-api-fastapi-1`
- **Image**: `crewai-api-fastapi`
- **Port**: `8100:80`
- **Status**: ✅ **RUNNING** (3 days uptime)

### **3. Notebook Lab**
- **Container**: `notebook-lab-1`
- **Image**: `notebook-lab`
- **Port**: `8888:8888`
- **Status**: ✅ **RUNNING** (2 months uptime)

### **4. Metabase**
- **Container**: `metabase`
- **Image**: `metabase/metabase`
- **Port**: `3000:3000`
- **Status**: ✅ **RUNNING** (2 months uptime)

## 🌐 **ACTIVE PORTS:**

| Port | Service | Type | Status |
|------|---------|------|--------|
| **99** | Dashboard Gatra | Docker | ✅ Running |
| **3000** | Metabase | Docker | ✅ Running |
| **8100** | CrewAI API | Docker | ✅ Running |
| **8503** | AI Model Training Dashboard | Streamlit | ✅ Running |
| **8527** | Comprehensive Real Data Dashboard | Streamlit | ✅ Running |
| **8528** | Production Dashboard Correlator | Streamlit | ✅ Running |
| **8888** | Notebook Lab | Docker | ✅ Running |

## 🚨 **NOTABLE ABSENCES:**

### **TAA Services:**
- ❌ **Enhanced TAA Service**: KILLED (gradual-migration-enhanced-taa.service)
- ❌ **TAA-MoE Enhanced Dashboard**: STOPPED
- ❌ **All TAA Processes**: TERMINATED

### **Port 8511:**
- ❌ **Port 8511**: No active service (was dashboard-alerts dashboard)

## 📈 **SERVICE SUMMARY:**

### **✅ RUNNING SERVICES:**
- **ADA Services**: 3 running (production, pubsub, langgraph)
- **CLA Services**: 5 running (1 local + 4 systemd)
- **Dashboard Services**: 4 running (various Streamlit apps)
- **Docker Containers**: 4 running
- **Total Active Ports**: 7

### **❌ STOPPED SERVICES:**
- **TAA Services**: All terminated
- **Enhanced Classification**: Stopped
- **Port 8511 Dashboard**: Not running

## 🎯 **KEY INSIGHTS:**

1. **ADA is Active**: Multiple ADA services running and processing alerts
2. **CLA is Active**: Continuous learning agents running for model improvement
3. **Dashboards Available**: Multiple dashboards on different ports
4. **TAA Completely Stopped**: No TAA services feeding data to dashboards
5. **Docker Infrastructure**: Stable containerized services running

**Total Running Services: 16+ (mix of systemd, processes, and Docker containers)**

