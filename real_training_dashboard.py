#!/usr/bin/env python3
"""
Real AI Training Dashboard - Shows Actual Training Data
Displays real metrics from the CLA service and model training
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import subprocess
import json
import os
import glob
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Real AI Training Dashboard",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .training-active {
        background: linear-gradient(90deg, #28a745, #20c997);
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        margin: 10px 0;
    }
    .training-inactive {
        background: linear-gradient(90deg, #dc3545, #fd7e14);
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        margin: 10px 0;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🤖 Real AI Model Training Dashboard")
st.subheader("Live Training Data from AI-Driven SOC System")

# Get real training data
@st.cache_data(ttl=60)  # Cache for 1 minute
def get_real_training_data():
    """Get real training data from CLA service"""
    try:
        # Get BigQuery data
        query = """
        SELECT 
            (SELECT COUNT(*) FROM `chronicle-dev-2be9.gatra_database.siem_events`) as total_events,
            (SELECT COUNT(*) FROM `chronicle-dev-2be9.gatra_database.siem_alarms`) as total_alarms,
            (SELECT COUNT(*) FROM `chronicle-dev-2be9.gatra_database.taa_state`) as taa_processed,
            (SELECT COUNT(*) FROM `chronicle-dev-2be9.gatra_database.dashboard_alerts`) as dashboard_alerts
        """
        
        result = subprocess.run([
            "bq", "query", "--use_legacy_sql=false", "--max_rows=1", "--format=json", query
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        
        bigquery_data = {}
        if result.returncode == 0:
            data = json.loads(result.stdout.strip())[0]
            bigquery_data = {
                "total_events": int(data["total_events"]),
                "total_alarms": int(data["total_alarms"]),
                "taa_processed": int(data["taa_processed"]),
                "dashboard_alerts": int(data["dashboard_alerts"])
            }
        
        # Get model training data
        models_dir = "/home/raditio.ghifiardigmail.com/ai-driven-soc/models/"
        training_data = {
            "total_models_trained": 0,
            "latest_model_metrics": None,
            "training_active": False,
            "last_training_time": None
        }
        
        try:
            # Count model files
            model_files = glob.glob(os.path.join(models_dir, "trained_model_*.pkl"))
            training_data["total_models_trained"] = len(model_files)
            training_data["training_active"] = len(model_files) > 0
            
            # Get latest metrics
            metrics_files = glob.glob(os.path.join(models_dir, "model_metrics_*.json"))
            if metrics_files:
                latest_file = max(metrics_files, key=os.path.getctime)
                with open(latest_file, 'r') as f:
                    training_data["latest_model_metrics"] = json.load(f)
                training_data["last_training_time"] = datetime.fromtimestamp(os.path.getctime(latest_file)).isoformat()
                
        except Exception as e:
            st.error(f"Error reading training data: {e}")
        
        # Check CLA service status
        cla_status = subprocess.run([
            "systemctl", "is-active", "cla.service"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        
        training_data["cla_service_active"] = cla_status.stdout.strip() == "active"
        
        return {**bigquery_data, **training_data}
        
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return {}

# Get the data
data = get_real_training_data()

if not data:
    st.error("Unable to fetch training data. Please check the system status.")
    st.stop()

# Training Status Banner
if data.get("training_active", False) and data.get("cla_service_active", False):
    st.markdown('<div class="training-active">🚀 LEARNING MODE ACTIVE - Real AI Training in Progress!</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="training-inactive">⚠️ STATISTICS MODE - No Active Training</div>', unsafe_allow_html=True)

# Main metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Events", 
        f"{data.get('total_events', 0):,}",
        help="Total SIEM events in the system"
    )

with col2:
    st.metric(
        "Models Trained", 
        f"{data.get('total_models_trained', 0)}",
        help="Total number of models trained by CLA service"
    )

with col3:
    st.metric(
        "CLA Service", 
        "✅ Active" if data.get('cla_service_active', False) else "❌ Inactive",
        help="Continuous Learning Agent service status"
    )

with col4:
    if data.get('last_training_time'):
        last_training = datetime.fromisoformat(data['last_training_time'].replace('Z', '+00:00'))
        time_diff = datetime.now().replace(tzinfo=last_training.tzinfo) - last_training
        minutes_ago = int(time_diff.total_seconds() / 60)
        st.metric(
            "Last Training", 
            f"{minutes_ago}m ago",
            help="Time since last model training"
        )
    else:
        st.metric("Last Training", "Never", help="No training has occurred")

st.markdown("---")

# Training Details
if data.get("latest_model_metrics"):
    st.header("📊 Latest Model Performance")
    
    metrics = data["latest_model_metrics"]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Training Samples</h4>
            <h2>{metrics.get('training_samples', 0):,}</h2>
            <p>Test Samples: {metrics.get('test_samples', 0):,}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Model Accuracy</h4>
            <h2>{metrics.get('accuracy', 0):.1%}</h2>
            <p>Overall Performance</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>F1 Score</h4>
            <h2>{metrics.get('f1_score', 0):.3f}</h2>
            <p>Balanced Performance</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Performance Metrics")
        perf_data = {
            "Metric": ["Precision", "Recall", "F1 Score", "Accuracy"],
            "Value": [
                metrics.get('precision', 0),
                metrics.get('recall', 0), 
                metrics.get('f1_score', 0),
                metrics.get('accuracy', 0)
            ]
        }
        df = pd.DataFrame(perf_data)
        fig = px.bar(df, x="Metric", y="Value", title="Model Performance Metrics")
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Confusion Matrix")
        cm_data = {
            "True Positives": metrics.get('true_positives', 0),
            "False Positives": metrics.get('false_positives', 0),
            "True Negatives": metrics.get('true_negatives', 0),
            "False Negatives": metrics.get('false_negatives', 0)
        }
        df_cm = pd.DataFrame(list(cm_data.items()), columns=["Type", "Count"])
        fig_cm = px.pie(df_cm, values="Count", names="Type", title="Classification Results")
        st.plotly_chart(fig_cm, use_container_width=True)

else:
    st.info("No model training data available. The CLA service may not be actively training.")

# System Status
st.header("🔧 System Status")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Data Sources")
    st.write(f"**SIEM Events:** {data.get('total_events', 0):,}")
    st.write(f"**SIEM Alarms:** {data.get('total_alarms', 0):,}")
    st.write(f"**TAA Processed:** {data.get('taa_processed', 0):,}")
    st.write(f"**Dashboard Alerts:** {data.get('dashboard_alerts', 0):,}")

with col2:
    st.subheader("Training Status")
    st.write(f"**CLA Service:** {'✅ Active' if data.get('cla_service_active', False) else '❌ Inactive'}")
    st.write(f"**Models Trained:** {data.get('total_models_trained', 0)}")
    st.write(f"**Training Active:** {'✅ Yes' if data.get('training_active', False) else '❌ No'}")
    st.write(f"**Mode:** {'🚀 LEARNING' if data.get('training_active', False) else '📊 STATISTICS'}")

with col3:
    st.subheader("Last Update")
    st.write(f"**Dashboard:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if data.get('last_training_time'):
        last_training = datetime.fromisoformat(data['last_training_time'].replace('Z', '+00:00'))
        st.write(f"**Last Training:** {last_training.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.write("**Last Training:** Never")

# Footer
st.markdown("---")
st.markdown(f"""
**Real-time AI Training Dashboard** | 
Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
Training Mode: {'🚀 LEARNING MODE' if data.get('training_active', False) else '📊 STATISTICS MODE'}
""")


