# AI Model Training Dashboard

A comprehensive Streamlit dashboard for monitoring AI model training metrics, performance improvements, and continuous learning progress for the AI-driven SOC system.

## 🎯 Dashboard Overview

This dashboard provides real-time visibility into:

- **Model Training Progress**: Accuracy, loss, and training metrics over time
- **Performance Improvements**: Automation rate, false positive reduction, response time optimization
- **Resource Utilization**: GPU, memory, and computational resource monitoring
- **A/B Testing Results**: Model version comparison and performance analysis
- **Data Quality Metrics**: Training data quality, completeness, and diversity scores
- **Training Sessions**: Recent training activities and status tracking

## 🚀 Features

### 📊 Key Metrics
- Current model version and training status
- Total training samples processed
- Model accuracy and performance indicators
- Real-time training progress monitoring

### 📈 Visualizations
- Training accuracy trends over time
- Performance improvement charts
- Resource utilization graphs
- A/B testing comparison charts
- Data quality metrics dashboard

### ⚙️ Interactive Controls
- Time range selection (24h, 7d, 30d, 90d, all time)
- Model type filtering
- Real-time data updates
- Responsive design for all devices

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Google Cloud Platform access
- BigQuery dataset with training metrics
- Service account credentials

### 1. Clone/Download Files
```bash
# Download the dashboard files to your project directory
ai_model_training_dashboard.py
ai_dashboard_config.py
ai_dashboard_requirements.txt
```

### 2. Install Dependencies
```bash
pip install -r ai_dashboard_requirements.txt
```

### 3. Configure Environment Variables
```bash
# Set your GCP project details
export GCP_PROJECT_ID="your-gcp-project-id"
export BIGQUERY_DATASET_ID="soc_analytics"
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account-key.json"
```

### 4. Update Configuration
Edit `ai_dashboard_config.py` with your specific:
- GCP project ID
- BigQuery dataset ID
- Table names for your training metrics
- Service account credentials

## 🗄️ Database Schema Requirements

### Required BigQuery Tables

#### 1. `model_training_history`
```sql
CREATE TABLE `model_training_history` (
  training_date DATE,
  model_version STRING,
  training_samples INT64,
  training_accuracy FLOAT64,
  validation_accuracy FLOAT64,
  training_loss FLOAT64,
  validation_loss FLOAT64,
  training_time_minutes INT64,
  gpu_utilization FLOAT64,
  memory_usage_gb FLOAT64,
  training_status STRING
);
```

#### 2. `model_performance_metrics`
```sql
CREATE TABLE `model_performance_metrics` (
  metric_date DATE,
  automation_rate FLOAT64,
  detection_accuracy FLOAT64,
  false_positive_rate FLOAT64,
  response_time_minutes FLOAT64
);
```

#### 3. `training_data_quality`
```sql
CREATE TABLE `training_data_quality` (
  metric_date DATE,
  data_quality_score FLOAT64,
  label_accuracy FLOAT64,
  data_completeness FLOAT64,
  feature_coverage FLOAT64,
  data_freshness_days FLOAT64,
  outlier_detection_rate FLOAT64,
  data_diversity_score FLOAT64,
  class_balance_score FLOAT64,
  validation_split_ratio FLOAT64
);
```

#### 4. `model_ab_testing_results`
```sql
CREATE TABLE `model_ab_testing_results` (
  model_version STRING,
  accuracy FLOAT64,
  automation_rate FLOAT64,
  false_positive_rate FLOAT64,
  response_time_minutes FLOAT64,
  training_time_hours FLOAT64
);
```

#### 5. `training_resource_utilization`
```sql
CREATE TABLE `training_resource_utilization` (
  training_date DATE,
  gpu_utilization FLOAT64,
  memory_usage_gb FLOAT64,
  cpu_utilization FLOAT64,
  network_bandwidth_mbps FLOAT64
);
```

## 🚀 Running the Dashboard

### Development Mode
```bash
streamlit run ai_model_training_dashboard.py
```

### Production Mode
```bash
streamlit run ai_model_training_dashboard.py --server.port 8501 --server.address 0.0.0.0
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "ai_model_training_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 📊 Dashboard Sections

### 1. Executive Summary
- **Current Model Version**: Latest stable release
- **Training Status**: Active/inactive training status
- **Total Training Samples**: Volume of data processed
- **Model Accuracy**: Current best performance

### 2. Training Progress & Performance
- Training vs. validation accuracy trends
- Current training metrics (accuracy, loss, status)
- Real-time training progress monitoring

### 3. Performance Improvements
- AI automation rate improvements over time
- False positive rate reduction trends
- Response time optimization progress

### 4. Resource Utilization
- GPU utilization during training
- Memory usage patterns
- Computational resource efficiency

### 5. Model Comparison & A/B Testing
- Version-to-version performance comparison
- Accuracy, automation, and efficiency metrics
- Training time and resource requirements

### 6. Training Data Quality
- Data quality scores and trends
- Label accuracy and completeness
- Feature coverage and diversity metrics

### 7. Recent Training Sessions
- Latest training activities
- Session status and duration
- Performance results summary

## 🔧 Customization

### Adding New Metrics
1. Update the configuration file with new table/column mappings
2. Add new visualization components to the dashboard
3. Update the data generation functions for demo mode

### Modifying Visualizations
- Edit Plotly chart configurations
- Adjust chart colors and styling
- Modify chart layouts and dimensions

### Data Source Integration
- Connect to different databases (PostgreSQL, MySQL, etc.)
- Integrate with real-time data streams
- Add API endpoints for external data sources

## 📈 Performance Optimization

### For Large Datasets
- Implement data pagination
- Add data caching layers
- Use database indexing for faster queries
- Implement data aggregation for historical views

### For Real-time Updates
- Use WebSocket connections
- Implement server-sent events
- Add auto-refresh functionality
- Cache frequently accessed data

## 🚨 Troubleshooting

### Common Issues

#### 1. BigQuery Connection Failed
```bash
# Check credentials
echo $GOOGLE_APPLICATION_CREDENTIALS
gcloud auth application-default login
```

#### 2. Missing Dependencies
```bash
# Reinstall requirements
pip install --upgrade -r ai_dashboard_requirements.txt
```

#### 3. Dashboard Not Loading
```bash
# Check Streamlit logs
streamlit run ai_model_training_dashboard.py --logger.level debug
```

#### 4. Data Not Displaying
- Verify BigQuery table schemas match requirements
- Check data permissions and access
- Validate query syntax and table names

## 🔒 Security Considerations

- Use service account keys with minimal required permissions
- Implement authentication for dashboard access
- Secure database connections
- Monitor access logs and usage patterns

## 📞 Support & Maintenance

### Regular Maintenance Tasks
- Monitor dashboard performance
- Update dependencies regularly
- Review and optimize database queries
- Backup configuration and customization files

### Performance Monitoring
- Track dashboard load times
- Monitor BigQuery query costs
- Analyze user interaction patterns
- Optimize data refresh frequencies

## 🎯 Next Steps

1. **Deploy the dashboard** to your development environment
2. **Connect to your BigQuery datasets** with real training data
3. **Customize visualizations** for your specific use cases
4. **Integrate with your CI/CD pipeline** for automated deployments
5. **Add real-time data streaming** for live monitoring
6. **Implement user authentication** and access controls

---

**Note**: This dashboard is designed to complement your existing cybersecurity executive dashboard by providing detailed AI model training insights. It helps bridge the gap between high-level security metrics and the underlying AI system performance.
