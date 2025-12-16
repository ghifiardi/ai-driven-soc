# TAA MoE Security Analytics - Services & Scripts Analysis 🛡️

## 🎯 **TAA MoE SECURITY ANALYTICS OVERVIEW:**

The "TAA MoE Security Analytics" page on your main dashboard (`http://10.45.254.19:8511/🛡️_TAA_MoE_Security_Analytics`) refers to a **Mixture of Experts Triage & Analysis Agent Integration** system.

## 🔧 **CORE SERVICES & SCRIPTS:**

### **1. Main Orchestrator Service:**
- **Script**: `taa_moe_orchestrator.py`
- **Location**: `/home/raditio.ghifiardigmail.com/taa_moe_orchestrator.py`
- **Port**: 8000
- **Process ID**: 3083903
- **Status**: ✅ Running
- **URL**: `http://10.45.254.19:8000`

**Function**: HTTP API server that orchestrates TAA-MoE operations
- **Endpoints**:
  - `GET /` - API status
  - `GET /health` - Health check
  - `POST /orchestrator/run` - Process security alerts

### **2. TAA MoE Integration Scripts:**
Located in `/home/app/ai-driven-soc/`:

1. **`taa_moe_integration.py`** - Main integration logic
2. **`taa_moe_api.py`** - API interface
3. **`taa_moe_simple.py`** - Simplified version
4. **`taa_moe_enhanced_dashboard.py`** - Enhanced dashboard
5. **`test_taa_moe.py`** - Testing suite
6. **`taa_service.py`** - Core TAA service
7. **`config/taa_config.json`** - Configuration file

### **3. Related Services:**
- **`taa_service.log`** - Service logs
- **Enhanced TAA Agent** - `enhanced_taa_agent.cpython-311.pyc` (compiled)

## 📊 **DASHBOARD FUNCTIONALITY:**

### **TAA MoE Security Analytics Page Features:**

1. **🚨 Security Alert Processing:**
   - Submit security alerts for analysis
   - Alert types: Malware Detection, Network Intrusion, Phishing, Insider Threat, Data Breach, DDoS
   - Severity levels: Low, Medium, High, Critical
   - Source/Destination IP tracking

2. **📊 Analytics Dashboard:**
   - **Threat Analysis**: Threat type distribution (pie chart)
   - **Severity Distribution**: Alert severity bar chart
   - **Timeline**: Alert processing timeline
   - **Expert Insights**: Recent expert analyses

3. **🔧 Orchestrator Integration:**
   - Health check for orchestrator service
   - Connection testing
   - Real-time status monitoring

## 🔗 **SYSTEM ARCHITECTURE:**

### **Data Flow:**
```
Dashboard (Port 8511) → TAA MoE Orchestrator (Port 8000) → Expert Agents → Analysis Results
```

### **Integration Points:**
- **Dashboard**: `http://10.45.254.19:8511/🛡️_TAA_MoE_Security_Analytics`
- **Orchestrator API**: `http://10.45.254.19:8000`
- **Health Check**: `http://10.45.254.19:8000/health`
- **Alert Processing**: `http://10.45.254.19:8000/orchestrator/run`

## 🎯 **EXPERT AGENTS (MoE Components):**

Based on the dashboard, the TAA MoE system includes **5 Active Expert Agents**:

1. **Malware Detection Expert**
2. **Network Intrusion Expert**
3. **Phishing Detection Expert**
4. **Insider Threat Expert**
5. **Data Breach Expert**

## 📈 **CURRENT STATUS:**

### **✅ Running Services:**
- **TAA MoE Orchestrator**: ✅ Running on port 8000
- **Main Dashboard**: ✅ Running on port 8511
- **Integration Scripts**: ✅ Available in `/home/app/ai-driven-soc/`

### **📊 Dashboard Metrics:**
- **Alerts Processed**: Tracked in session state
- **Expert Analyses**: Stored and visualized
- **Threat Types**: Categorized and counted
- **Severity Distribution**: Monitored
- **Response Times**: Tracked

## 🚀 **USAGE:**

### **To Use TAA MoE Security Analytics:**

1. **Access**: `http://10.45.254.19:8511/🛡️_TAA_MoE_Security_Analytics`
2. **Submit Alert**: Use the form to submit security incidents
3. **View Analytics**: Check the analytics tabs for insights
4. **Monitor Status**: Check orchestrator health in sidebar

### **API Endpoints:**
```bash
# Health check
curl http://10.45.254.19:8000/health

# Submit alert
curl -X POST http://10.45.254.19:8000/orchestrator/run \
  -H "Content-Type: application/json" \
  -d '{"alert_type": "Malware Detection", "severity": "High", "description": "Suspicious file detected"}'
```

## 🎉 **SUMMARY:**

The **TAA MoE Security Analytics** refers to:
- ✅ **Orchestrator Service**: `taa_moe_orchestrator.py` (Port 8000)
- ✅ **Integration Scripts**: Multiple TAA MoE scripts in `/home/app/ai-driven-soc/`
- ✅ **Expert Agents**: 5 specialized AI agents for different threat types
- ✅ **Dashboard Integration**: Real-time analytics and alert processing
- ✅ **API Interface**: RESTful API for alert submission and processing

**This is your main AI-driven security analysis system with multiple expert agents working together!** 🛡️


