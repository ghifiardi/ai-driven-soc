# AI-Driven SOC Platform - Deployment Runbook

**Version:** 1.0
**Last Updated:** December 31, 2024
**Owner:** DevOps Team

---

## Table of Contents

1. [Pre-Deployment Checklist](#1-pre-deployment-checklist)
2. [Deployment Procedures](#2-deployment-procedures)
3. [Post-Deployment Verification](#3-post-deployment-verification)
4. [Rollback Procedures](#4-rollback-procedures)
5. [Troubleshooting Guide](#5-troubleshooting-guide)
6. [Emergency Contacts](#6-emergency-contacts)

---

## 1. Pre-Deployment Checklist

### 1.1 GitHub Repository Verification

```bash
# Verify you're on the correct branch
git branch --show-current
# Expected: main

# Verify latest commit
git log --oneline -1

# Verify all changes are pushed
git status
# Expected: nothing to commit, working tree clean
```

### 1.2 GitHub Secrets Configuration

Navigate to: `Settings → Secrets and variables → Actions`

**Required Secrets:**

| Secret Name | Description | How to Obtain |
|-------------|-------------|---------------|
| `GCP_PROJECT_ID` | GCP Project ID | GCP Console → Project Settings |
| `GCP_SA_KEY` | Base64-encoded service account JSON | See section 1.3 |
| `GCP_REGION` | Deployment region | e.g., `asia-southeast1` |

### 1.3 Generate GCP Service Account Key

```bash
# 1. Create service account (if not exists)
gcloud iam service-accounts create gatra-soc-deployer \
    --display-name="GATRA SOC Deployer"

# 2. Grant required roles
PROJECT_ID=$(gcloud config get-value project)
SA_EMAIL="gatra-soc-deployer@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/artifactregistry.writer"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/run.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/secretmanager.secretAccessor"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/iam.serviceAccountUser"

# 3. Generate key and encode
gcloud iam service-accounts keys create sa-key.json \
    --iam-account=$SA_EMAIL

# 4. Base64 encode for GitHub secret
cat sa-key.json | base64

# 5. Copy the output to GitHub secret GCP_SA_KEY
# 6. DELETE the local key file
rm sa-key.json
```

### 1.4 GCP Secret Manager Setup

```bash
# Create secrets in GCP Secret Manager
PROJECT_ID=$(gcloud config get-value project)

# Generate secure values
JWT_SECRET=$(openssl rand -hex 32)
API_KEY=$(openssl rand -hex 32)

# Create staging secrets
echo -n "$JWT_SECRET" | gcloud secrets create gatra-jwt-secret \
    --data-file=- --project=$PROJECT_ID

echo -n "$API_KEY" | gcloud secrets create gatra-service-api-key \
    --data-file=- --project=$PROJECT_ID

# Create production secrets (use different values!)
JWT_SECRET_PROD=$(openssl rand -hex 32)
API_KEY_PROD=$(openssl rand -hex 32)

echo -n "$JWT_SECRET_PROD" | gcloud secrets create gatra-jwt-secret-prod \
    --data-file=- --project=$PROJECT_ID

echo -n "$API_KEY_PROD" | gcloud secrets create gatra-service-api-key-prod \
    --data-file=- --project=$PROJECT_ID
```

### 1.5 Enable Required GCP APIs

```bash
gcloud services enable \
    artifactregistry.googleapis.com \
    run.googleapis.com \
    secretmanager.googleapis.com \
    cloudbuild.googleapis.com
```

### 1.6 Create Artifact Registry Repository

```bash
gcloud artifacts repositories create gatra-soc \
    --repository-format=docker \
    --location=$GCP_REGION \
    --description="GATRA SOC Docker images"
```

### 1.7 Final Pre-Deployment Verification

- [ ] All GitHub secrets configured
- [ ] GCP APIs enabled
- [ ] Artifact Registry repository created
- [ ] Secret Manager secrets created
- [ ] Service account has required permissions
- [ ] CI pipeline passing on main branch
- [ ] Team notified of deployment window

---

## 2. Deployment Procedures

### 2.1 Standard Deployment (via Git Tag)

```bash
# Step 1: Ensure you're on main branch with latest code
git checkout main
git pull github main

# Step 2: Verify CI is passing
# Check: https://github.com/ghifiardi/ai-driven-soc/actions

# Step 3: Create version tag
VERSION="v1.0.0"  # Update version as appropriate
git tag $VERSION
git push github $VERSION

# Step 4: Monitor deployment
# Go to: https://github.com/ghifiardi/ai-driven-soc/actions
# Watch the CD workflow progress
```

### 2.2 Version Numbering Convention

```
v{MAJOR}.{MINOR}.{PATCH}

MAJOR: Breaking changes or major features
MINOR: New features, backward compatible
PATCH: Bug fixes, minor improvements

Examples:
v1.0.0 - Initial production release
v1.0.1 - Bug fix
v1.1.0 - New feature added
v2.0.0 - Major architecture change
```

### 2.3 Manual Deployment (Emergency)

If automated deployment fails:

```bash
# 1. Build Docker image locally
docker build -t gatra-soc:manual -f Dockerfile.gatra .

# 2. Tag for Artifact Registry
REGION="asia-southeast1"  # Update as needed
PROJECT_ID="your-project-id"
docker tag gatra-soc:manual \
    ${REGION}-docker.pkg.dev/${PROJECT_ID}/gatra-soc/mssp-platform:manual

# 3. Authenticate and push
gcloud auth configure-docker ${REGION}-docker.pkg.dev
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/gatra-soc/mssp-platform:manual

# 4. Deploy to Cloud Run
gcloud run deploy gatra-soc-production \
    --image=${REGION}-docker.pkg.dev/${PROJECT_ID}/gatra-soc/mssp-platform:manual \
    --region=$REGION \
    --platform=managed
```

---

## 3. Post-Deployment Verification

### 3.1 Health Check Verification

```bash
# Get service URL
SERVICE_URL=$(gcloud run services describe gatra-soc-staging \
    --region=$REGION --format='value(status.url)')

# Liveness check
curl -f "${SERVICE_URL}/health/live"
# Expected: {"status": "ok"}

# Readiness check
curl -f "${SERVICE_URL}/health/ready"
# Expected: {"status": "ready", "checks": {...}}
```

### 3.2 Functional Verification

```bash
# Test authentication endpoint
curl -X POST "${SERVICE_URL}/api/auth/login" \
    -H "Content-Type: application/json" \
    -d '{"username": "test", "password": "test"}'

# Test API endpoint (with auth token)
TOKEN="<token-from-login>"
curl "${SERVICE_URL}/api/tenants" \
    -H "Authorization: Bearer ${TOKEN}"
```

### 3.3 Log Verification

```bash
# View Cloud Run logs
gcloud logging read "resource.type=cloud_run_revision \
    AND resource.labels.service_name=gatra-soc-staging" \
    --limit=50 \
    --format="table(timestamp, textPayload)"
```

### 3.4 Metrics Verification

Check in GCP Console:
1. Cloud Run → gatra-soc-staging → Metrics
2. Verify:
   - Request count increasing
   - No 5xx errors
   - Latency within acceptable range (<2s p99)
   - Instance count stable

### 3.5 Post-Deployment Checklist

- [ ] Health endpoints responding
- [ ] No errors in logs
- [ ] Metrics showing normal patterns
- [ ] Authentication working
- [ ] Core API endpoints functional
- [ ] Alert processing working (if applicable)
- [ ] Stakeholders notified of successful deployment

---

## 4. Rollback Procedures

### 4.1 Automatic Rollback

The CD pipeline includes automatic rollback on deployment failure. No action required.

### 4.2 Manual Rollback via GitHub Actions

1. Go to: https://github.com/ghifiardi/ai-driven-soc/actions
2. Select "CD" workflow
3. Click "Run workflow"
4. Select environment: `staging` or `production`
5. The workflow will detect failure and trigger rollback

### 4.3 Manual Rollback via gcloud

```bash
# List recent revisions
gcloud run revisions list \
    --service=gatra-soc-production \
    --region=$REGION \
    --format="table(name, active, created)"

# Identify the previous working revision (e.g., gatra-soc-production-00002)
PREV_REVISION="gatra-soc-production-00002"

# Route all traffic to previous revision
gcloud run services update-traffic gatra-soc-production \
    --region=$REGION \
    --to-revisions=${PREV_REVISION}=100

# Verify rollback
curl -f "$(gcloud run services describe gatra-soc-production \
    --region=$REGION --format='value(status.url)')/health/live"
```

### 4.4 Rollback Verification

After rollback:
- [ ] Health checks passing
- [ ] No new errors in logs
- [ ] Previous functionality restored
- [ ] Incident documented
- [ ] Root cause analysis scheduled

---

## 5. Troubleshooting Guide

### 5.1 Common Issues and Solutions

#### Issue: Docker build fails

**Symptoms:** CI pipeline fails at "Build Docker Image" step

**Solutions:**
```bash
# Check for syntax errors in Dockerfile
docker build -t test -f Dockerfile.gatra . 2>&1 | head -50

# Verify requirements.txt is valid
pip install -r requirements.txt --dry-run
```

#### Issue: GCP authentication fails

**Symptoms:** "failed to fetch oauth token" error

**Solutions:**
1. Verify `GCP_SA_KEY` secret is correctly base64 encoded
2. Verify service account has required permissions
3. Regenerate service account key if needed

```bash
# Test service account locally
gcloud auth activate-service-account --key-file=sa-key.json
gcloud run services list  # Should work if permissions correct
```

#### Issue: Container fails to start

**Symptoms:** Cloud Run shows "Container failed to start"

**Solutions:**
```bash
# Check logs for startup errors
gcloud logging read "resource.type=cloud_run_revision" --limit=100

# Common causes:
# - Missing environment variables
# - Port mismatch (should be 8080)
# - Memory limit too low
```

#### Issue: Health check fails

**Symptoms:** Deployment succeeds but health checks fail

**Solutions:**
```bash
# Verify health endpoint locally
docker run -p 8080:8080 gatra-soc:latest
curl http://localhost:8080/health/live

# Check if Redis/other dependencies are available
# Verify REDIS_URL environment variable is set
```

### 5.2 Log Analysis Commands

```bash
# View all logs for a service
gcloud logging read "resource.type=cloud_run_revision \
    AND resource.labels.service_name=gatra-soc-production" \
    --limit=100

# View only errors
gcloud logging read "resource.type=cloud_run_revision \
    AND resource.labels.service_name=gatra-soc-production \
    AND severity>=ERROR" \
    --limit=50

# View logs from specific time range
gcloud logging read "resource.type=cloud_run_revision \
    AND resource.labels.service_name=gatra-soc-production \
    AND timestamp>=\"2024-12-31T00:00:00Z\"" \
    --limit=100
```

### 5.3 Performance Issues

```bash
# Check instance scaling
gcloud run services describe gatra-soc-production \
    --region=$REGION \
    --format="yaml(spec.template.spec.containerConcurrency, \
                   spec.template.metadata.annotations)"

# Increase resources if needed
gcloud run services update gatra-soc-production \
    --region=$REGION \
    --memory=4Gi \
    --cpu=4
```

---

## 6. Emergency Contacts

### 6.1 Escalation Matrix

| Level | Contact | When to Escalate |
|-------|---------|------------------|
| L1 | DevOps On-Call | Any deployment issue |
| L2 | Technical Lead | Issue unresolved >30 min |
| L3 | Engineering Manager | Critical production impact |
| L4 | CTO | Major incident, data breach |

### 6.2 Contact Information

| Role | Name | Phone | Email |
|------|------|-------|-------|
| DevOps Lead | [TBD] | [TBD] | [TBD] |
| Technical Lead | [TBD] | [TBD] | [TBD] |
| Security Lead | [TBD] | [TBD] | [TBD] |
| Project Manager | [TBD] | [TBD] | [TBD] |

### 6.3 External Support

| Service | Support Channel | SLA |
|---------|-----------------|-----|
| GCP Support | console.cloud.google.com/support | Based on plan |
| GitHub Support | support.github.com | Based on plan |

---

## Appendix A: Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ENVIRONMENT` | Yes | - | `staging` or `production` |
| `JWT_SECRET` | Yes (prod) | ephemeral | JWT signing secret |
| `SERVICE_API_KEY` | Yes (prod) | - | Service auth key |
| `GEMINI_API_KEY` | No | - | Google Gemini API |
| `REDIS_URL` | Yes | - | Redis connection string |
| `LOG_LEVEL` | No | `INFO` | Logging verbosity |
| `LOG_FORMAT` | No | `json` | Log format |

---

## Appendix B: Useful Commands Quick Reference

```bash
# View deployment status
gcloud run services describe gatra-soc-production --region=$REGION

# View all revisions
gcloud run revisions list --service=gatra-soc-production --region=$REGION

# Update traffic split
gcloud run services update-traffic gatra-soc-production \
    --region=$REGION --to-latest

# View logs
gcloud logging read "resource.labels.service_name=gatra-soc-production" --limit=50

# Delete old revisions (cleanup)
gcloud run revisions delete REVISION_NAME --region=$REGION
```

---

**Document Revision History:**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | Dec 31, 2024 | Claude Code | Initial version |

