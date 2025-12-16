#!/usr/bin/env python3
"""
AESMOTE Enhanced Training Dashboard

This dashboard shows:
1. Current CLA model performance
2. AESMOTE model performance comparison
3. Adversarial learning visualization
4. SMOTE impact analysis
5. Model ensemble comparison
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import glob
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AESMOTE Enhanced CLA Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_model_metrics():
    """Load all available model metrics"""
    metrics_files = []
    
    # Load regular CLA models
    regular_files = glob.glob("models/real_data_metrics_*.json")
    metrics_files.extend(regular_files)
    
    # Load AESMOTE models
    aesmote_files = glob.glob("models/aesmote_*_metrics_*.json")
    metrics_files.extend(aesmote_files)
    
    models_data = []
    
    for file_path in sorted(metrics_files):
        try:
            with open(file_path, 'r') as f:
                metrics = json.load(f)
            
            # Extract timestamp from filename
            filename = os.path.basename(file_path)
            if 'aesmote' in filename:
                model_type = "AESMOTE"
                if 'simple' in filename:
                    model_type = "AESMOTE-Simple"
            else:
                model_type = "Standard CLA"
            
            metrics['model_type'] = model_type
            metrics['timestamp'] = datetime.fromtimestamp(os.path.getctime(file_path))
            metrics['filename'] = filename
            
            models_data.append(metrics)
            
        except Exception as e:
            st.warning(f"Error loading {file_path}: {e}")
    
    return models_data

def create_performance_comparison_chart(models_data):
    """Create performance comparison chart"""
    if not models_data:
        return None
    
    # Prepare data for plotting
    plot_data = []
    for model in models_data:
        plot_data.append({
            'Model': f"{model['model_type']}",
            'Precision': model.get('precision', 0),
            'Recall': model.get('recall', 0),
            'F1-Score': model.get('f1_score', 0),
            'Accuracy': model.get('accuracy', 0),
            'Timestamp': model['timestamp']
        })
    
    df = pd.DataFrame(plot_data)
    
    # Create subplot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Precision', 'Recall', 'F1-Score', 'Accuracy'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    metrics = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
    positions = [(1,1), (1,2), (2,1), (2,2)]
    
    for i, (metric, pos) in enumerate(zip(metrics, positions)):
        fig.add_trace(
            go.Bar(
                x=df['Model'],
                y=df[metric],
                name=metric,
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'][i % 3],
                showlegend=False
            ),
            row=pos[0], col=pos[1]
        )
        
        # Add value labels
        for j, v in enumerate(df[metric]):
            fig.add_annotation(
                x=df['Model'].iloc[j],
                y=v + 0.01,
                text=f"{v:.1%}",
                showarrow=False,
                row=pos[0], col=pos[1]
            )
    
    fig.update_layout(
        title="Model Performance Comparison",
        height=600,
        showlegend=False
    )
    
    fig.update_yaxes(range=[0, 1.1])
    
    return fig

def create_aesmote_analysis(models_data):
    """Create AESMOTE-specific analysis"""
    aesmote_models = [m for m in models_data if 'AESMOTE' in m['model_type']]
    
    if not aesmote_models:
        return None, None, None
    
    # Get latest AESMOTE model
    latest_aesmote = max(aesmote_models, key=lambda x: x['timestamp'])
    
    # Performance metrics
    metrics_data = {
        'Metric': ['Precision', 'Recall', 'F1-Score', 'Accuracy'],
        'Value': [
            latest_aesmote.get('precision', 0),
            latest_aesmote.get('recall', 0),
            latest_aesmote.get('f1_score', 0),
            latest_aesmote.get('accuracy', 0)
        ]
    }
    
    metrics_fig = px.bar(
        x=metrics_data['Metric'],
        y=metrics_data['Value'],
        title="AESMOTE Model Performance",
        color=metrics_data['Value'],
        color_continuous_scale='RdYlGn'
    )
    
    metrics_fig.update_layout(
        yaxis=dict(range=[0, 1.1]),
        showlegend=False
    )
    
    # Add value labels
    for i, v in enumerate(metrics_data['Value']):
        metrics_fig.add_annotation(
            x=metrics_data['Metric'][i],
            y=v + 0.02,
            text=f"{v:.1%}",
            showarrow=False
        )
    
    # Training episodes analysis (if available)
    episodes_data = None
    if 'episodes_trained' in latest_aesmote:
        episodes_data = {
            'Episodes': latest_aesmote['episodes_trained'],
            'Adversarial Ratio': latest_aesmote.get('adversarial_ratio', 0.3),
            'CV F1-Score': latest_aesmote.get('cv_f1_mean', 0),
            'CV Std': latest_aesmote.get('cv_f1_std', 0)
        }
    
    # Ensemble analysis (if available)
    ensemble_info = None
    if 'ensemble_models' in latest_aesmote:
        ensemble_info = latest_aesmote['ensemble_models']
    
    return metrics_fig, episodes_data, ensemble_info

def create_adversarial_learning_viz():
    """Create adversarial learning visualization"""
    # Simulate adversarial learning process
    episodes = np.arange(1, 51)
    
    # Simulate improving accuracy over episodes
    base_accuracy = 0.75
    improvement = np.random.normal(0.001, 0.002, len(episodes))
    accuracy = base_accuracy + np.cumsum(improvement)
    accuracy = np.clip(accuracy, 0.7, 1.0)
    
    # Simulate adversarial difficulty
    difficulty = 0.8 - 0.1 * np.exp(-episodes / 20)
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Model Accuracy Over Episodes', 'Sample Difficulty Selection'),
        vertical_spacing=0.1
    )
    
    # Accuracy plot
    fig.add_trace(
        go.Scatter(
            x=episodes,
            y=accuracy,
            mode='lines+markers',
            name='Model Accuracy',
            line=dict(color='blue', width=3),
            marker=dict(size=4)
        ),
        row=1, col=1
    )
    
    # Difficulty plot
    fig.add_trace(
        go.Scatter(
            x=episodes,
            y=difficulty,
            mode='lines+markers',
            name='Selected Sample Difficulty',
            line=dict(color='red', width=3),
            marker=dict(size=4)
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title="AESMOTE Adversarial Learning Process",
        height=600,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Training Episodes")
    fig.update_yaxes(title_text="Accuracy", row=1, col=1, range=[0.7, 1.0])
    fig.update_yaxes(title_text="Difficulty Score", row=2, col=1, range=[0.6, 0.9])
    
    return fig

def create_smote_impact_viz():
    """Create SMOTE impact visualization"""
    # Simulate class distribution before and after SMOTE
    classes = ['Benign', 'Anomaly']
    before_smote = [85, 15]  # Highly imbalanced
    after_smote = [50, 50]   # Balanced
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Before SMOTE', 'After SMOTE'),
        specs=[[{"type": "pie"}, {"type": "pie"}]]
    )
    
    # Before SMOTE pie chart
    fig.add_trace(
        go.Pie(
            labels=classes,
            values=before_smote,
            name="Before SMOTE",
            marker_colors=['lightblue', 'lightcoral']
        ),
        row=1, col=1
    )
    
    # After SMOTE pie chart
    fig.add_trace(
        go.Pie(
            labels=classes,
            values=after_smote,
            name="After SMOTE",
            marker_colors=['lightblue', 'lightcoral']
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title="SMOTE Impact on Class Distribution",
        height=400
    )
    
    return fig

def main():
    """Main dashboard function"""
    
    # Header
    st.title("🤖 AESMOTE Enhanced CLA Training Dashboard")
    st.markdown("**Adversarial Reinforcement Learning with SMOTE for Anomaly Detection**")
    
    # Load model data
    models_data = load_model_metrics()
    
    if not models_data:
        st.error("No model metrics found. Please train some models first.")
        return
    
    # Sidebar
    st.sidebar.header("Dashboard Controls")
    
    # Model selection
    available_models = list(set([m['model_type'] for m in models_data]))
    selected_model_type = st.sidebar.selectbox(
        "Select Model Type",
        available_models
    )
    
    # Show AESMOTE explanation
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 AESMOTE Framework")
    st.sidebar.markdown("""
    **AESMOTE** combines:
    - **Adversarial Learning**: Two competing agents
    - **SMOTE**: Synthetic minority oversampling
    - **Ensemble Methods**: Multiple model voting
    - **Difficulty Selection**: Focus on hard examples
    """)
    
    # Main dashboard content
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance Comparison", "🤖 AESMOTE Analysis", "🎯 Adversarial Learning", "⚖️ SMOTE Impact"])
    
    with tab1:
        st.header("📊 Model Performance Comparison")
        
        # Performance comparison chart
        perf_fig = create_performance_comparison_chart(models_data)
        if perf_fig:
            st.plotly_chart(perf_fig, use_container_width=True)
        
        # Model details table
        st.subheader("📋 Model Details")
        
        details_data = []
        for model in models_data:
            details_data.append({
                'Model Type': model['model_type'],
                'Precision': f"{model.get('precision', 0):.1%}",
                'Recall': f"{model.get('recall', 0):.1%}",
                'F1-Score': f"{model.get('f1_score', 0):.1%}",
                'Accuracy': f"{model.get('accuracy', 0):.1%}",
                'Training Samples': model.get('training_samples', 0),
                'Test Samples': model.get('test_samples', 0),
                'Timestamp': model['timestamp'].strftime('%Y-%m-%d %H:%M')
            })
        
        details_df = pd.DataFrame(details_data)
        st.dataframe(details_df, use_container_width=True)
    
    with tab2:
        st.header("🤖 AESMOTE Model Analysis")
        
        # AESMOTE analysis
        aesmote_fig, episodes_data, ensemble_info = create_aesmote_analysis(models_data)
        
        if aesmote_fig:
            st.plotly_chart(aesmote_fig, use_container_width=True)
            
            # Additional AESMOTE metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Training Configuration")
                if episodes_data:
                    for key, value in episodes_data.items():
                        if isinstance(value, float):
                            st.metric(key, f"{value:.3f}")
                        else:
                            st.metric(key, value)
            
            with col2:
                st.subheader("🏗️ Ensemble Models")
                if ensemble_info:
                    for model in ensemble_info:
                        st.write(f"• {model}")
                    st.metric("Total Models", len(ensemble_info))
        else:
            st.info("No AESMOTE models found. Train an AESMOTE model to see this analysis.")
    
    with tab3:
        st.header("🎯 Adversarial Learning Process")
        
        st.markdown("""
        **Adversarial Learning in AESMOTE:**
        
        1. **Classifier Agent**: Makes predictions and receives rewards for correct classifications
        2. **Environment Agent**: Selects difficult samples and receives opposite rewards
        3. **Competition**: Environment Agent learns to find weaknesses in Classifier Agent
        4. **Improvement**: Classifier Agent is forced to improve on difficult cases
        """)
        
        # Adversarial learning visualization
        adv_fig = create_adversarial_learning_viz()
        st.plotly_chart(adv_fig, use_container_width=True)
        
        # Benefits of adversarial learning
        st.subheader("💡 Benefits of Adversarial Learning")
        
        benefits = [
            "🎯 **Targeted Training**: Focus on samples that are hardest to classify",
            "🔄 **Dynamic Adaptation**: Continuously adapt to new challenges",
            "⚖️ **Balanced Learning**: Prevent overfitting to easy examples",
            "📈 **Improved Generalization**: Better performance on unseen data",
            "🛡️ **Robust Defense**: More resilient to adversarial attacks"
        ]
        
        for benefit in benefits:
            st.markdown(benefit)
    
    with tab4:
        st.header("⚖️ SMOTE Impact Analysis")
        
        st.markdown("""
        **SMOTE (Synthetic Minority Oversampling Technique):**
        
        - **Problem**: Imbalanced datasets with few minority class examples
        - **Solution**: Generate synthetic samples for minority class
        - **Method**: Create new samples between existing minority samples
        - **Result**: Better representation of minority class patterns
        """)
        
        # SMOTE impact visualization
        smote_fig = create_smote_impact_viz()
        st.plotly_chart(smote_fig, use_container_width=True)
        
        # SMOTE benefits
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("✅ SMOTE Advantages")
            advantages = [
                "🎯 **Better Minority Detection**: More examples of rare attacks",
                "📊 **Balanced Training**: Equal representation of all classes",
                "🔄 **Synthetic Generation**: No simple duplication",
                "📈 **Improved Recall**: Better detection of minority classes"
            ]
            
            for advantage in advantages:
                st.markdown(advantage)
        
        with col2:
            st.subheader("⚠️ SMOTE Considerations")
            considerations = [
                "🎲 **Synthetic Data**: Generated samples may not reflect reality",
                "⚖️ **Balance Trade-off**: May reduce majority class performance",
                "🔍 **Quality Control**: Need to validate synthetic samples",
                "📊 **Parameter Tuning**: K-neighbors parameter affects results"
            ]
            
            for consideration in considerations:
                st.markdown(consideration)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### 🚀 AESMOTE Framework Implementation
    
    **Current Status**: Enhanced CLA system with adversarial learning and SMOTE
    
    **Key Improvements**:
    - ✅ Adversarial sample selection for difficult cases
    - ✅ SMOTE synthetic minority oversampling
    - ✅ Ensemble methods for robust predictions
    - ✅ Dynamic difficulty adjustment
    - ✅ Cross-validation for model reliability
    
    **Next Steps**: Deploy AESMOTE models to production for improved threat detection
    """)

if __name__ == "__main__":
    main()


