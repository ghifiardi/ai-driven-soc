# CrewAI API & Metabase Services Documentation 🔧

## 📋 **TABLE OF CONTENTS**

1. [Service Overview](#service-overview)
2. [CrewAI API Service (Port 8100)](#crewai-api-service-port-8100)
3. [Metabase Service (Port 3000)](#metabase-service-port-3000)
4. [Service Integration](#service-integration)
5. [Configuration & Setup](#configuration--setup)
6. [API Endpoints & Usage](#api-endpoints--usage)
7. [Data Sources & Connections](#data-sources--connections)
8. [Monitoring & Maintenance](#monitoring--maintenance)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

---

## 🎯 **SERVICE OVERVIEW**

### **Active Docker Services:**
- **CrewAI API**: Port 8100 - FastAPI-based AI agent orchestration
- **Metabase**: Port 3000 - Business intelligence and data visualization
- **Status**: Both services are running and operational

### **Service Architecture:**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CrewAI API    │    │    Metabase     │    │   BigQuery      │
│   (Port 8100)   │◄───┤   (Port 3000)   │───▶│   Database      │
│   FastAPI       │    │   Analytics     │    │   (Data Source) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  AI Agents      │    │  Dashboards     │    │  SOC Data       │
│  Orchestration  │    │  & Reports      │    │  Analytics      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## 🤖 **CREWAI API SERVICE (PORT 8100)**

### **Service Details:**
- **Container**: `crewai-api-fastapi-1`
- **Image**: `crewai-api-fastapi`
- **Port**: 8100 (external) → 80 (internal)
- **Status**: Running (3 days uptime)
- **Location**: `/home/app/crewai-api/`

### **Technical Specifications:**
- **Framework**: FastAPI (Python)
- **Python Version**: 3.11.13
- **Runtime**: Docker containerized
- **Start Command**: `/start.sh`
- **Working Directory**: `/app`

### **Key Features:**
1. **AI Agent Orchestration**: Coordinates multiple AI agents
2. **FastAPI Interface**: RESTful API for agent management
3. **Containerized Deployment**: Docker-based deployment
4. **Configuration Management**: Environment-based configuration

### **Directory Structure:**
```
/home/app/crewai-api/
├── alert_precheck/          # Alert preprocessing modules
├── config/                  # Configuration files
├── engine/                  # Core engine components
├── routes/                  # API route definitions
├── utils/                   # Utility functions
├── keys/                    # API keys and credentials
├── main.py                  # Main application entry point
├── cra.py                   # Containment Response Agent
├── requirements.txt         # Python dependencies
├── Dockerfile              # Container configuration
├── docker-compose.yaml     # Docker compose configuration
└── .env                    # Environment variables
```

### **Configuration Files:**
- **`docker-compose.yaml`**: Docker service configuration
- **`Dockerfile`**: Container build instructions
- **`.env`**: Environment variables and secrets
- **`requirements.txt`**: Python package dependencies

### **API Capabilities:**
- **Agent Management**: Create, configure, and manage AI agents
- **Task Orchestration**: Coordinate multi-agent workflows
- **Real-time Processing**: Handle streaming data and events
- **Integration**: Connect with SOC systems and data sources

---

## 📊 **METABASE SERVICE (PORT 3000)**

### **Service Details:**
- **Container**: `metabase`
- **Image**: `metabase/metabase`
- **Port**: 3000 (external) → 3000 (internal)
- **Status**: Running (2 months uptime)
- **Database**: SQLite (`/metabase-data/metabase.db`)

### **Technical Specifications:**
- **Platform**: Java-based web application
- **Java Version**: JDK 21.0.7+6
- **Database**: SQLite (embedded)
- **Runtime**: Docker containerized
- **Start Command**: `/app/run_metabase.sh`

### **Key Features:**
1. **Business Intelligence**: Data visualization and analytics
2. **Dashboard Creation**: Interactive dashboards and reports
3. **Query Interface**: SQL query builder and execution
4. **Data Source Integration**: Multiple database connections
5. **User Management**: Role-based access control

### **Supported Data Sources:**
- **BigQuery**: Google Cloud BigQuery integration
- **PostgreSQL**: PostgreSQL database connections
- **MySQL**: MySQL database connections
- **MongoDB**: MongoDB document database
- **SQL Server**: Microsoft SQL Server
- **Redshift**: Amazon Redshift
- **And many more...**

### **Metabase Configuration:**
```json
{
  "site-url": "http://10.45.254.19:3000",
  "application-name": "Metabase",
  "version": "v0.55.4",
  "instance-creation": "2025-06-23T09:46:00.95Z",
  "has-user-setup": true,
  "available-timezones": ["UTC", "GMT", "America/New_York", ...],
  "available-locales": ["en", "es", "fr", "de", ...]
}
```

### **Database Connections:**
- **Primary**: BigQuery integration for SOC data
- **Secondary**: Various database connections as needed
- **Local**: SQLite for Metabase metadata storage

---

## 🔗 **SERVICE INTEGRATION**

### **CrewAI API Integration:**
```
CrewAI API (Port 8100)
    ↓
AI Agent Orchestration
    ↓
SOC Data Processing
    ↓
BigQuery Integration
```

### **Metabase Integration:**
```
Metabase (Port 3000)
    ↓
Data Visualization
    ↓
BigQuery Analytics
    ↓
SOC Dashboards
```

### **Data Flow:**
1. **CrewAI API** orchestrates AI agents for SOC operations
2. **AI Agents** process security data and generate insights
3. **Data** is stored in BigQuery for analytics
4. **Metabase** connects to BigQuery for visualization
5. **Dashboards** provide real-time SOC analytics

---

## ⚙️ **CONFIGURATION & SETUP**

### **CrewAI API Configuration:**
```yaml
# docker-compose.yaml
version: '3.8'
services:
  fastapi:
    build: .
    ports:
      - "8100:80"
    environment:
      - PRELOAD_APP=true
      - WEB_CONCURRENCY=1
      - TIMEOUT=300
      - GRACEFUL_TIMEOUT=60
    volumes:
      - ./:/app
```

### **Environment Variables:**
```bash
# .env file
PRELOAD_APP=true
WEB_CONCURRENCY=1
TIMEOUT=300
GRACEFUL_TIMEOUT=60
PYTHONPATH=/app
```

### **Metabase Configuration:**
```bash
# Environment variables
MB_DB_FILE=/metabase-data/metabase.db
JAVA_HOME=/opt/java/openjdk
LANG=en_US.UTF-8
JAVA_VERSION=jdk-21.0.7+6
```

---

## 🔌 **API ENDPOINTS & USAGE**

### **CrewAI API Endpoints:**
- **Health Check**: `GET /health` (returns 404 - service running)
- **API Documentation**: `GET /docs` (Swagger/OpenAPI)
- **Agent Management**: Custom endpoints for agent orchestration
- **Task Processing**: Endpoints for SOC task processing

### **Metabase Endpoints:**
- **Main Interface**: `GET /` - Web-based analytics interface
- **API**: RESTful API for programmatic access
- **Database Setup**: Connection configuration interface
- **Dashboard Management**: Dashboard creation and management

### **Usage Examples:**
```bash
# Check CrewAI API status
curl http://localhost:8100/health

# Access Metabase interface
curl http://localhost:3000

# Check service logs
docker logs crewai-api-fastapi-1
docker logs metabase
```

---

## 📈 **DATA SOURCES & CONNECTIONS**

### **BigQuery Integration:**
- **Project**: `chronicle-dev-2be9`
- **Database**: `gatra_database`
- **Tables**: 
  - `siem_events` (1,008,668 records)
  - `siem_alarms` (403,970 records)
  - `taa_state` (22,607 records)
  - `dashboard_alerts` (1,000 records)

### **Data Connection Configuration:**
```json
{
  "driver-name": "BigQuery Cloud SDK",
  "project-id": "chronicle-dev-2be9",
  "service-account-json": "service_account_credentials.json",
  "dataset-filters-type": "all",
  "auto_run_queries": true
}
```

### **Supported Connection Types:**
- **BigQuery**: Primary SOC data source
- **PostgreSQL**: Additional database connections
- **MySQL**: MySQL database support
- **MongoDB**: Document database support
- **SQL Server**: Microsoft SQL Server
- **And more...**

---

## 📊 **MONITORING & MAINTENANCE**

### **Service Monitoring:**
```bash
# Check service status
docker ps | grep -E '8100|3000'

# Monitor logs
docker logs -f crewai-api-fastapi-1
docker logs -f metabase

# Check resource usage
docker stats crewai-api-fastapi-1 metabase
```

### **Health Checks:**
- **CrewAI API**: Port 8100 accessibility
- **Metabase**: Port 3000 web interface
- **Database**: BigQuery connection status
- **Container**: Docker container health

### **Maintenance Tasks:**
- **Daily**: Check service status and logs
- **Weekly**: Review performance metrics
- **Monthly**: Update dependencies and configurations
- **As Needed**: Backup configurations and data

---

## 🔧 **TROUBLESHOOTING**

### **Common Issues:**

#### **1. CrewAI API Not Responding:**
```bash
# Check container status
docker ps | grep crewai-api-fastapi-1

# Restart service
docker restart crewai-api-fastapi-1

# Check logs
docker logs crewai-api-fastapi-1
```

#### **2. Metabase Connection Issues:**
```bash
# Check container status
docker ps | grep metabase

# Restart service
docker restart metabase

# Check logs
docker logs metabase
```

#### **3. Database Connection Problems:**
- Verify BigQuery credentials
- Check network connectivity
- Validate service account permissions
- Review connection configuration

### **Debug Commands:**
```bash
# Container inspection
docker inspect crewai-api-fastapi-1
docker inspect metabase

# Resource monitoring
docker stats crewai-api-fastapi-1 metabase

# Network connectivity
curl -v http://localhost:8100/health
curl -v http://localhost:3000
```

---

## 🎯 **BEST PRACTICES**

### **Security:**
1. **API Keys**: Secure storage of credentials
2. **Network Access**: Restrict external access
3. **Authentication**: Implement proper user authentication
4. **Data Privacy**: Protect sensitive SOC data

### **Performance:**
1. **Resource Monitoring**: Track CPU, memory, and disk usage
2. **Database Optimization**: Optimize BigQuery queries
3. **Caching**: Implement appropriate caching strategies
4. **Load Balancing**: Scale services as needed

### **Reliability:**
1. **Backup**: Regular configuration backups
2. **Monitoring**: Continuous service monitoring
3. **Updates**: Regular security and feature updates
4. **Documentation**: Maintain up-to-date documentation

### **Integration:**
1. **API Design**: Consistent API patterns
2. **Data Flow**: Clear data pipeline documentation
3. **Error Handling**: Robust error handling and logging
4. **Testing**: Regular integration testing

---

## 📞 **SERVICE ACCESS**

### **URLs:**
- **CrewAI API**: `http://10.45.254.19:8100`
- **Metabase**: `http://10.45.254.19:3000`

### **Authentication:**
- **CrewAI API**: API key-based authentication
- **Metabase**: User account-based authentication

### **Documentation:**
- **CrewAI API**: Available at `/docs` endpoint
- **Metabase**: Built-in help and documentation
- **Service Logs**: Docker container logs

---

## 📋 **SERVICE SUMMARY**

### **CrewAI API (Port 8100):**
- ✅ **Status**: Active and running
- ✅ **Purpose**: AI agent orchestration
- ✅ **Framework**: FastAPI (Python)
- ✅ **Integration**: SOC data processing

### **Metabase (Port 3000):**
- ✅ **Status**: Active and running
- ✅ **Purpose**: Business intelligence and analytics
- ✅ **Platform**: Java-based web application
- ✅ **Integration**: BigQuery data visualization

### **Key Benefits:**
1. **AI Orchestration**: Coordinated multi-agent workflows
2. **Data Visualization**: Interactive SOC dashboards
3. **Real-time Analytics**: Live data processing and insights
4. **Scalable Architecture**: Containerized, cloud-ready services

---

**These services provide essential AI orchestration and business intelligence capabilities for your AI-Driven SOC system, enabling sophisticated data processing and visualization workflows.**
