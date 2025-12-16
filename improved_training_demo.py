#!/usr/bin/env python3
"""
Improved Training Demo Dashboard
Demonstrates the difference between baseline and improved CLA training
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import time

# Page configuration
st.set_page_config(
    page_title="Improved CLA Training Demo",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .improvement-box {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .baseline-box {
        background: linear-gradient(135deg, #dc3545 0%, #fd7e14 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🚀 Improved CLA Training Demo</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Demonstrating Imbalance-Aware Training vs Baseline</p>', unsafe_allow_html=True)

# Sidebar controls
st.sidebar.header("🎛️ Training Parameters")
n_samples = st.sidebar.slider("Training Samples", 500, 2000, 1000)
class_imbalance = st.sidebar.slider("Class Imbalance (%)", 80, 99, 90)
noise_level = st.sidebar.slider("Noise Level", 0.0, 0.5, 0.1)

if st.sidebar.button("🔄 Generate New Data"):
    st.rerun()

# Generate synthetic data
@st.cache_data
def generate_synthetic_data(n_samples, class_imbalance, noise_level):
    np.random.seed(42)
    
    # Calculate class proportions
    neg_prop = class_imbalance / 100
    pos_prop = 1 - neg_prop
    
    # Generate features
    data = {
        'confidence': np.random.beta(2, 5, n_samples),
        'comment_length': np.random.poisson(50, n_samples),
        'hour': np.random.randint(0, 24, n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples),
        'is_weekend': np.random.randint(0, 2, n_samples),
        'has_comments': np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    }
    
    # Create labels with specified imbalance
    labels = np.random.choice([0, 1], n_samples, p=[neg_prop, pos_prop])
    
    # Add signal to make it learnable
    data['confidence'][labels == 1] += 0.3
    data['comment_length'][labels == 1] += 20
    
    # Add noise
    if noise_level > 0:
        noise_indices = np.random.choice(n_samples, int(n_samples * noise_level), replace=False)
        labels[noise_indices] = 1 - labels[noise_indices]
    
    return pd.DataFrame(data), labels

# Generate data
X, y = generate_synthetic_data(n_samples, class_imbalance, noise_level)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Display data summary
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Samples", len(X))
with col2:
    st.metric("True Positives", f"{y.sum()} ({y.sum()/len(y)*100:.1f}%)")
with col3:
    st.metric("Training Set", len(X_train))
with col4:
    st.metric("Test Set", len(X_test))

# Train models
st.header("🎯 Model Training Results")

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="baseline-box"><h3>📉 Baseline Model</h3><p>No Class Balancing</p></div>', unsafe_allow_html=True)
    
    # Baseline model
    rf_baseline = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_baseline.fit(X_train, y_train)
    y_pred_baseline = rf_baseline.predict(X_test)
    
    # Calculate metrics
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    baseline_precision = precision_score(y_test, y_pred_baseline, zero_division=0)
    baseline_recall = recall_score(y_test, y_pred_baseline, zero_division=0)
    baseline_f1 = f1_score(y_test, y_pred_baseline, zero_division=0)
    baseline_accuracy = accuracy_score(y_test, y_pred_baseline)
    
    st.metric("Precision", f"{baseline_precision:.3f}")
    st.metric("Recall", f"{baseline_recall:.3f}")
    st.metric("F1-Score", f"{baseline_f1:.3f}")
    st.metric("Accuracy", f"{baseline_accuracy:.3f}")

with col2:
    st.markdown('<div class="improvement-box"><h3>🚀 Improved Model</h3><p>Class Weight Balancing</p></div>', unsafe_allow_html=True)
    
    # Improved model with class weights
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_map = {c: w for c, w in zip(classes, weights)}
    
    rf_improved = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        random_state=42, 
        class_weight=class_weight_map
    )
    rf_improved.fit(X_train, y_train)
    y_pred_improved = rf_improved.predict(X_test)
    
    # Calculate metrics
    improved_precision = precision_score(y_test, y_pred_improved, zero_division=0)
    improved_recall = recall_score(y_test, y_pred_improved, zero_division=0)
    improved_f1 = f1_score(y_test, y_pred_improved, zero_division=0)
    improved_accuracy = accuracy_score(y_test, y_pred_improved)
    
    st.metric("Precision", f"{improved_precision:.3f}")
    st.metric("Recall", f"{improved_recall:.3f}")
    st.metric("F1-Score", f"{improved_f1:.3f}")
    st.metric("Accuracy", f"{improved_accuracy:.3f}")

# Comparison visualization
st.header("📊 Performance Comparison")

# Create comparison data
comparison_data = pd.DataFrame({
    'Model': ['Baseline', 'Improved', 'Baseline', 'Improved', 'Baseline', 'Improved'],
    'Metric': ['Precision', 'Precision', 'Recall', 'Recall', 'F1-Score', 'F1-Score'],
    'Value': [baseline_precision, improved_precision, baseline_recall, improved_recall, baseline_f1, improved_f1]
})

fig = px.bar(
    comparison_data, 
    x='Metric', 
    y='Value', 
    color='Model',
    title='Model Performance Comparison',
    color_discrete_map={'Baseline': '#dc3545', 'Improved': '#28a745'},
    height=400
)

fig.update_layout(
    yaxis_range=[0, 1],
    showlegend=True,
    template='plotly_white'
)

st.plotly_chart(fig, use_container_width=True)

# Improvement analysis
st.header("📈 Improvement Analysis")

col1, col2, col3 = st.columns(3)

precision_improvement = ((improved_precision - baseline_precision) / baseline_precision * 100) if baseline_precision > 0 else 0
recall_improvement = ((improved_recall - baseline_recall) / baseline_recall * 100) if baseline_recall > 0 else 0
f1_improvement = ((improved_f1 - baseline_f1) / baseline_f1 * 100) if baseline_f1 > 0 else 0

with col1:
    st.metric(
        "Precision Improvement", 
        f"{precision_improvement:+.1f}%",
        delta=f"{improved_precision - baseline_precision:+.3f}"
    )
with col2:
    st.metric(
        "Recall Improvement", 
        f"{recall_improvement:+.1f}%",
        delta=f"{improved_recall - baseline_recall:+.3f}"
    )
with col3:
    st.metric(
        "F1-Score Improvement", 
        f"{f1_improvement:+.1f}%",
        delta=f"{improved_f1 - baseline_f1:+.3f}"
    )

# Confusion matrices
st.header("🔍 Confusion Matrices")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Baseline Model")
    cm_baseline = confusion_matrix(y_test, y_pred_baseline)
    fig_cm_baseline = px.imshow(
        cm_baseline, 
        text_auto=True, 
        aspect="auto",
        title="Baseline Confusion Matrix",
        labels=dict(x="Predicted", y="Actual"),
        color_continuous_scale="Reds"
    )
    st.plotly_chart(fig_cm_baseline, use_container_width=True)

with col2:
    st.subheader("Improved Model")
    cm_improved = confusion_matrix(y_test, y_pred_improved)
    fig_cm_improved = px.imshow(
        cm_improved, 
        text_auto=True, 
        aspect="auto",
        title="Improved Confusion Matrix",
        labels=dict(x="Predicted", y="Actual"),
        color_continuous_scale="Greens"
    )
    st.plotly_chart(fig_cm_improved, use_container_width=True)

# Key insights
st.header("💡 Key Insights")

insights = []

if improved_recall > baseline_recall:
    insights.append(f"✅ **Recall improved by {recall_improvement:+.1f}%** - Better at detecting true threats")
else:
    insights.append("❌ **Recall decreased** - May miss some threats")

if improved_precision > baseline_precision:
    insights.append(f"✅ **Precision improved by {precision_improvement:+.1f}%** - Fewer false alarms")
else:
    insights.append("❌ **Precision decreased** - More false alarms")

if improved_f1 > baseline_f1:
    insights.append(f"✅ **F1-Score improved by {f1_improvement:+.1f}%** - Better overall balance")
else:
    insights.append("❌ **F1-Score decreased** - Overall performance declined")

for insight in insights:
    st.markdown(insight)

# Recommendations
st.header("🎯 Recommendations")

if improved_f1 > baseline_f1:
    st.success("🚀 **Class weight balancing is working!** The improved model shows better performance on the minority class (true positives) while maintaining good overall accuracy.")
    st.info("💡 **Next steps:** Implement SMOTE oversampling and XGBoost with scale_pos_weight for even better results.")
else:
    st.warning("⚠️ **Class imbalance is severe.** Consider using SMOTE oversampling or collecting more positive examples.")
    st.info("💡 **Alternative approaches:** Use anomaly detection methods or active learning to identify more true positives.")

# Footer
st.markdown("---")
st.markdown("**🔧 This demo shows the difference between baseline and improved CLA training techniques.**")
st.markdown("**📚 The improved model uses class weight balancing to handle the severe class imbalance in SOC feedback data.**")


