# SOC Dashboard Suite - Deployment Guide

## 🚀 Quick Deployment

### Local Development

1. **Install Dependencies:**
```bash
cd dashboards
pip install -r requirements.txt
```

2. **Run Dashboard:**
```bash
# Enhanced SOC Dashboard (Recommended)
streamlit run enhanced_soc_dashboard.py

# Executive Dashboard
streamlit run executive_dashboard.py

# Compliance Dashboard
streamlit run compliance_dashboard.py

# Original SOC Dashboard
streamlit run streamlit_soc_dashboard.py
```

3. **Access Dashboard:**
- Open browser to `http://localhost:8501`

### GCP VM Deployment

1. **SSH into your GCP instance:**
```bash
gcloud compute ssh xdgaisocapp01 --zone=asia-southeast2-a
```

2. **Upload dashboard files:**
```bash
# From local machine
gcloud compute scp dashboards/* app@xdgaisocapp01:~/dashboards/ \
  --zone=asia-southeast2-a --recurse
```

3. **On GCP VM, install dependencies:**
```bash
cd ~/dashboards
pip3 install -r requirements.txt --user
```

4. **Run dashboard:**
```bash
streamlit run enhanced_soc_dashboard.py --server.port 8501 --server.address 0.0.0.0
```

5. **Access via browser:**
- `http://YOUR_VM_IP:8501`

### Docker Deployment

1. **Create Dockerfile:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY dashboards/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY dashboards/ .

EXPOSE 8501

CMD ["streamlit", "run", "enhanced_soc_dashboard.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
```

2. **Build and Run:**
```bash
docker build -t ai-soc-dashboard .
docker run -p 8501:8501 ai-soc-dashboard
```

### Streamlit Cloud Deployment

1. **Push to GitHub:**
```bash
git add dashboards/
git commit -m "Add SOC dashboard suite from PR #2"
git push github main
```

2. **Deploy on Streamlit Cloud:**
   - Go to [streamlit.io/cloud](https://streamlit.io/cloud)
   - Connect your GitHub repository
   - Set main file: `dashboards/enhanced_soc_dashboard.py`
   - Deploy

## 🔧 Configuration

### BigQuery Integration

1. **Set up credentials:**
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

2. **Update project ID in dashboard files:**
```python
# In dashboard files, update:
client = bigquery.Client(project="chronicle-dev-2be9")
```

3. **Configure dataset:**
```python
# Update dataset and table names
dataset_id = "soc_data"
table_id = "security_events"
```

### Environment Variables

```bash
export GCP_PROJECT_ID="chronicle-dev-2be9"
export BIGQUERY_DATASET="soc_data"
export STREAMLIT_SERVER_PORT=8501
```

## 📊 Dashboard Features Integration

### With Threat Hunting Platform

The dashboards can integrate with:
- **THOR Endpoint Agent** - Threat detection data
- **ASGARD Orchestration Agent** - Campaign management
- **VALHALLA Feed Manager** - Threat intelligence

### Data Sources

- BigQuery tables from `soc_data` dataset
- Real-time threat hunting results
- YARA rule detection data
- IOC matching results

## 🔐 Security Considerations

1. **Authentication:**
   - Add Streamlit authentication for production
   - Use environment variables for secrets

2. **Access Control:**
   - Implement RBAC if needed
   - Restrict dashboard access by IP

3. **HTTPS:**
   - Use reverse proxy (nginx) for HTTPS
   - Configure SSL certificates

## 🐛 Troubleshooting

### Port Already in Use
```bash
streamlit run enhanced_soc_dashboard.py --server.port 8502
```

### BigQuery Connection Issues
- Verify credentials are set correctly
- Check project ID and dataset names
- Ensure BigQuery API is enabled
- Dashboard will fallback to mock data

### Performance Issues
- Reduce auto-refresh interval
- Limit time range for queries
- Disable unused dashboard sections

---

**For detailed dashboard documentation, see:** [README.md](./README.md)

