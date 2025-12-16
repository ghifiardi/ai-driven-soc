# Hybrid Cyber Defense Agent - Deployment Checklist

## Pre-Deployment Phase

### System Requirements Verification
- [ ] **Hardware Requirements**
  - [ ] CPU: 2+ cores, 2.4+ GHz (4+ cores, 3.0+ GHz recommended)
  - [ ] RAM: 4+ GB (8+ GB recommended)
  - [ ] Storage: 10+ GB free space (50+ GB recommended)
  - [ ] Network: 100+ Mbps connectivity (1 Gbps recommended)

- [ ] **Software Requirements**
  - [ ] OS: Linux (Ubuntu 20.04+), macOS 10.15+, or Windows 10+
  - [ ] Python: 3.8+ (3.9+ recommended)
  - [ ] Google Cloud CLI: Latest version installed
  - [ ] Docker: Latest version (if using containerized deployment)

### Google Cloud Setup
- [ ] **Project Configuration**
  - [ ] Valid GCP project with billing enabled
  - [ ] Project ID noted and configured
  - [ ] Required APIs enabled:
    - [ ] Pub/Sub API
    - [ ] Vertex AI API
    - [ ] Cloud Storage API

- [ ] **Authentication Setup**
  - [ ] Service account created with appropriate permissions
  - [ ] Service account key downloaded and secured
  - [ ] Application Default Credentials configured
  - [ ] Permissions verified:
    - [ ] Pub/Sub Subscriber role
    - [ ] Pub/Sub Publisher role
    - [ ] Vertex AI User role

### Environment Preparation
- [ ] **Virtual Environment**
  - [ ] Virtual environment created
  - [ ] Virtual environment activated
  - [ ] Pip upgraded to latest version

- [ ] **Dependencies Installation**
  - [ ] Core dependencies installed from requirements.txt
  - [ ] All packages successfully installed
  - [ ] Import tests passed for all modules

### Configuration Setup
- [ ] **Configuration Files**
  - [ ] `config/hybrid_defense_config.json` created and configured
  - [ ] Project ID updated in configuration
  - [ ] Service account path configured
  - [ ] Log level set appropriately
  - [ ] Port numbers configured (8083 for agent, 8529 for dashboard)

- [ ] **Environment Variables**
  - [ ] `GOOGLE_APPLICATION_CREDENTIALS` set
  - [ ] `GOOGLE_CLOUD_PROJECT` set
  - [ ] `HYBRID_DEFENSE_CONFIG_PATH` set
  - [ ] `HYBRID_DEFENSE_LOG_LEVEL` set

### Pub/Sub Infrastructure
- [ ] **Topics Created**
  - [ ] `ada-alerts` topic created
  - [ ] `containment-requests` topic created
  - [ ] `taa-feedback` topic created

- [ ] **Subscriptions Created**
  - [ ] `hybrid-defense-ada-alerts-sub` subscription created
  - [ ] Subscription properly linked to `ada-alerts` topic

### Model Preparation
- [ ] **DQN Model**
  - [ ] Model file exists at specified path
  - [ ] Model loads successfully
  - [ ] Model inference test passed
  - [ ] Fallback model configured (if applicable)

## Deployment Phase

### Development Deployment (Testing)
- [ ] **Agent Deployment**
  - [ ] Agent started successfully: `python3 hybrid_cyber_defense_agent.py`
  - [ ] Agent listening on configured port (8083)
  - [ ] Health check endpoint responding: `curl http://localhost:8083/health`
  - [ ] Status endpoint responding: `curl http://localhost:8083/status`
  - [ ] No critical errors in logs

- [ ] **Dashboard Deployment**
  - [ ] Dashboard started successfully: `streamlit run hybrid_defense_dashboard.py --server.port=8529`
  - [ ] Dashboard accessible at configured port (8529)
  - [ ] Dashboard connects to agent successfully
  - [ ] All dashboard tabs functional
  - [ ] No critical errors in logs

### Production Deployment

#### Option 1: Systemd Services
- [ ] **Service Files**
  - [ ] `hybrid-defense-agent.service` created in `/etc/systemd/system/`
  - [ ] `hybrid-defense-dashboard.service` created in `/etc/systemd/system/`
  - [ ] Service files configured with correct paths and environment variables

- [ ] **Service Management**
  - [ ] Systemd daemon reloaded: `sudo systemctl daemon-reload`
  - [ ] Services enabled: `sudo systemctl enable hybrid-defense-agent hybrid-defense-dashboard`
  - [ ] Services started: `sudo systemctl start hybrid-defense-agent hybrid-defense-dashboard`
  - [ ] Service status verified: `sudo systemctl status hybrid-defense-agent hybrid-defense-dashboard`

#### Option 2: Docker Deployment
- [ ] **Container Build**
  - [ ] Dockerfile exists and is properly configured
  - [ ] Docker image built successfully: `docker build -t hybrid-defense-agent .`
  - [ ] Image size and layers verified

- [ ] **Container Deployment**
  - [ ] Docker Compose file configured (if using)
  - [ ] Containers started successfully
  - [ ] Port mappings verified (8083, 8529)
  - [ ] Volume mounts configured correctly

#### Option 3: Kubernetes Deployment
- [ ] **Kubernetes Manifests**
  - [ ] Deployment YAML files created
  - [ ] Service YAML files created
  - [ ] ConfigMap/Secret files created
  - [ ] Resource limits configured

- [ ] **Kubernetes Deployment**
  - [ ] Namespace created (if needed)
  - [ ] Manifests applied: `kubectl apply -f k8s/`
  - [ ] Pods running: `kubectl get pods`
  - [ ] Services accessible: `kubectl get services`

## Post-Deployment Phase

### System Validation
- [ ] **Health Checks**
  - [ ] Agent health endpoint: `curl http://localhost:8083/health`
  - [ ] Dashboard health endpoint: `curl http://localhost:8529/healthz`
  - [ ] All components reporting healthy status
  - [ ] Circuit breakers in CLOSED state

- [ ] **Functionality Tests**
  - [ ] Test alert processing via A2A endpoint
  - [ ] Verify DQN inference is working
  - [ ] Test Gemini integration
  - [ ] Verify Pub/Sub message flow
  - [ ] Test dashboard data refresh

### Performance Validation
- [ ] **Response Time Tests**
  - [ ] DQN inference time < 100ms
  - [ ] Gemini API response time < 10 seconds
  - [ ] Total alert processing time < 15 seconds
  - [ ] Health check response time < 1 second

- [ ] **Load Tests**
  - [ ] System handles 10 alerts/second sustained load
  - [ ] System handles 50 alerts/second burst load
  - [ ] Dashboard supports 20 concurrent users
  - [ ] No memory leaks detected

### Monitoring Setup
- [ ] **Logging Configuration**
  - [ ] Log levels set appropriately
  - [ ] Log rotation configured
  - [ ] Centralized logging setup (if applicable)
  - [ ] Log aggregation working

- [ ] **Metrics Collection**
  - [ ] System metrics being collected
  - [ ] Application metrics being collected
  - [ ] Business metrics being collected
  - [ ] Custom metrics being collected

- [ ] **Alerting Setup**
  - [ ] Critical alerts configured (service down, high error rate)
  - [ ] Warning alerts configured (performance degradation)
  - [ ] Info alerts configured (configuration changes)
  - [ ] Alert escalation procedures defined

### Security Validation
- [ ] **Authentication**
  - [ ] Service account authentication working
  - [ ] API key authentication configured (if applicable)
  - [ ] TLS/SSL configured for production

- [ ] **Network Security**
  - [ ] Firewall rules configured
  - [ ] VPC configuration verified
  - [ ] Load balancer SSL termination working
  - [ ] DDoS protection enabled

- [ ] **Data Protection**
  - [ ] Encryption at rest configured
  - [ ] Encryption in transit configured
  - [ ] Data retention policies configured
  - [ ] PII handling verified

### Documentation
- [ ] **Runbooks**
  - [ ] Operational runbooks created
  - [ ] Troubleshooting guides created
  - [ ] Emergency procedures documented
  - [ ] Contact information updated

- [ ] **Training**
  - [ ] Operations team trained
  - [ ] Support procedures documented
  - [ ] Escalation procedures defined
  - [ ] Knowledge transfer completed

## Go-Live Checklist

### Final Pre-Go-Live
- [ ] **Backup Verification**
  - [ ] Configuration backed up
  - [ ] Model files backed up
  - [ ] Database backed up (if applicable)
  - [ ] Disaster recovery procedures tested

- [ ] **Rollback Preparation**
  - [ ] Rollback procedures documented
  - [ ] Rollback scripts prepared
  - [ ] Rollback testing completed
  - [ ] Emergency contacts available

### Go-Live Execution
- [ ] **Deployment**
  - [ ] Production deployment executed
  - [ ] All services started successfully
  - [ ] Health checks passing
  - [ ] No critical errors

- [ ] **Validation**
  - [ ] End-to-end testing completed
  - [ ] Performance benchmarks met
  - [ ] Security validation passed
  - [ ] Monitoring confirmed working

### Post-Go-Live
- [ ] **Monitoring**
  - [ ] 24-hour monitoring initiated
  - [ ] Alert thresholds verified
  - [ ] Performance metrics baseline established
  - [ ] Error rates within acceptable limits

- [ ] **Documentation**
  - [ ] Go-live report completed
  - [ ] Lessons learned documented
  - [ ] Known issues documented
  - [ ] Future improvements identified

## Maintenance Schedule

### Daily Tasks
- [ ] Monitor system health and performance
- [ ] Check error logs and circuit breaker states
- [ ] Verify Pub/Sub message flow
- [ ] Review alert processing metrics

### Weekly Tasks
- [ ] Update DQN model with new training data
- [ ] Review and optimize Gemini prompts
- [ ] Analyze performance trends
- [ ] Update documentation

### Monthly Tasks
- [ ] Security audit and vulnerability assessment
- [ ] Performance tuning and optimization
- [ ] Backup and disaster recovery testing
- [ ] Capacity planning review

## Emergency Procedures

### Service Down
1. [ ] Check service status: `sudo systemctl status hybrid-defense-agent`
2. [ ] Check logs: `sudo journalctl -u hybrid-defense-agent -f`
3. [ ] Restart service: `sudo systemctl restart hybrid-defense-agent`
4. [ ] Verify health: `curl http://localhost:8083/health`
5. [ ] Escalate if issue persists

### High Error Rate
1. [ ] Check error logs for patterns
2. [ ] Verify circuit breaker states
3. [ ] Check external service dependencies
4. [ ] Scale resources if needed
5. [ ] Escalate if issue persists

### Performance Degradation
1. [ ] Check system resources (CPU, memory, disk)
2. [ ] Review performance metrics
3. [ ] Check for bottlenecks
4. [ ] Optimize configuration
5. [ ] Escalate if issue persists

---

**Checklist Version**: 1.0  
**Last Updated**: October 2025  
**Review Date**: January 2026  
**Approval**: Technical Architecture Review Board