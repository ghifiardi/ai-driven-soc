# Quick Start Guide - SOC Dashboards

## 🚀 Running Dashboards

### Method 1: Using the Launcher Script (Easiest)

```bash
cd dashboards

# Run Enhanced SOC Dashboard (default, port 8501)
./run_dashboard.sh

# Run Executive Dashboard (port 8502)
./run_dashboard.sh executive_dashboard.py 8502

# Run Compliance Dashboard (port 8503)
./run_dashboard.sh compliance_dashboard.py 8503

# Run Original SOC Dashboard (port 8504)
./run_dashboard.sh streamlit_soc_dashboard.py 8504
```

### Method 2: Manual Activation

```bash
cd dashboards

# Activate virtual environment
source venv/bin/activate

# Run any dashboard
streamlit run enhanced_soc_dashboard.py --server.port 8501
streamlit run executive_dashboard.py --server.port 8502
streamlit run compliance_dashboard.py --server.port 8503
streamlit run streamlit_soc_dashboard.py --server.port 8504
```

## 📊 Available Dashboards

1. **Enhanced SOC Dashboard** - Full-featured operational dashboard
2. **Executive Dashboard** - C-suite reporting and KPIs
3. **Compliance Dashboard** - Multi-framework compliance tracking
4. **Original SOC Dashboard** - BigQuery-integrated dashboard

## 🌐 Access URLs

- Enhanced SOC: http://localhost:8501
- Executive: http://localhost:8502
- Compliance: http://localhost:8503
- Original SOC: http://localhost:8504

## ⚠️ Troubleshooting

### "command not found: streamlit"
**Solution:** Activate the virtual environment first:
```bash
cd dashboards
source venv/bin/activate
```

### Port Already in Use
**Solution:** Use a different port:
```bash
streamlit run enhanced_soc_dashboard.py --server.port 8505
```

### Dependencies Missing
**Solution:** Reinstall dependencies:
```bash
cd dashboards
source venv/bin/activate
pip install -r requirements.txt
```

---

**Quick Tip:** Use the launcher script (`./run_dashboard.sh`) for easiest access!

