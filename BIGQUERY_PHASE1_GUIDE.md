# BigQuery Phase 1 Success Measurement Guide
## Real-time Insights into AI-Driven SOC Performance

### 🎯 **Overview**
This guide shows you how to use BigQuery to measure Phase 1 success and get real-time insights into your AI-Driven SOC's performance.

### 📊 **Current Phase 1 Status** (Based on Demo Data)
- **Infrastructure**: ✅ All BigQuery views created successfully
- **Data**: ✅ 1,500 alerts across 30 days generated
- **Success Rate**: 1/3 criteria met (Coherence Improvement achieved)
- **Key Metrics**: 26% redundancy (target <20%), 0.300 entropy index, 33.1s avg processing time

---

## 🔍 **Step 1: Access Your BigQuery Console**

### **Option A: Google Cloud Console**
1. Go to [BigQuery Console](https://console.cloud.google.com/bigquery)
2. Select project: `chronicle-dev-2be9`
3. Navigate to `soc_data` dataset

### **Option B: Command Line**
```bash
# Install Google Cloud CLI if not already installed
gcloud auth login
gcloud config set project chronicle-dev-2be9
bq query --use_legacy_sql=false "SELECT COUNT(*) FROM soc_data.processed_alerts"
```

---

## 📈 **Step 2: Key Queries for Phase 1 Success Measurement**

### **2.1 Alert Quality Metrics**
```sql
-- Get current alert quality status
SELECT 
    analysis_date,
    total_alerts,
    redundancy_percentage,
    uniqueness_percentage,
    coherence_rating,
    entropy_index,
    -- Success indicators
    CASE 
        WHEN redundancy_percentage < 20 THEN '✅ ACHIEVED'
        ELSE '❌ NEEDS WORK'
    END as redundancy_target,
    CASE 
        WHEN coherence_rating IN ('Excellent', 'Good') THEN '✅ ACHIEVED'
        ELSE '❌ NEEDS WORK'
    END as coherence_target
FROM `chronicle-dev-2be9.soc_data.kpi_summary`
WHERE analysis_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
ORDER BY analysis_date DESC;
```

### **2.2 Entropy Index Trend Analysis**
```sql
-- Track SOC coherence over time
SELECT 
    analysis_date,
    entropy_index,
    coherence_level,
    entropy_change,
    -- Trend interpretation
    CASE 
        WHEN entropy_change < -0.05 THEN '📈 Improving'
        WHEN entropy_change > 0.05 THEN '📉 Worsening'
        ELSE '➡️ Stable'
    END as trend_direction
FROM `chronicle-dev-2be9.soc_data.entropy_index_view`
WHERE analysis_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
ORDER BY analysis_date ASC;
```

### **2.3 Triage Efficiency Analysis**
```sql
-- Measure processing efficiency by similarity group
SELECT 
    similarity_group,
    AVG(avg_processing_time_seconds) as avg_processing_time,
    AVG(fast_processing_ratio) as fast_processing_ratio,
    SUM(estimated_time_savings) as total_time_savings,
    COUNT(*) as days_analyzed
FROM `chronicle-dev-2be9.soc_data.triage_time_analysis`
WHERE analysis_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
GROUP BY similarity_group
ORDER BY avg_processing_time ASC;
```

### **2.4 Real-time SOC Health Check**
```sql
-- Today's SOC performance vs yesterday
WITH today_metrics AS (
    SELECT 
        analysis_date,
        total_alerts,
        redundancy_percentage,
        entropy_index,
        coherence_rating
    FROM `chronicle-dev-2be9.soc_data.kpi_summary`
    WHERE analysis_date = CURRENT_DATE()
),
yesterday_metrics AS (
    SELECT 
        total_alerts as yesterday_alerts,
        redundancy_percentage as yesterday_redundancy,
        entropy_index as yesterday_entropy
    FROM `chronicle-dev-2be9.soc_data.kpi_summary`
    WHERE analysis_date = DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)
)
SELECT 
    t.*,
    y.yesterday_alerts,
    y.yesterday_redundancy,
    y.yesterday_entropy,
    (t.total_alerts - y.yesterday_alerts) as alert_change,
    (t.redundancy_percentage - y.yesterday_redundancy) as redundancy_change,
    (t.entropy_index - y.yesterday_entropy) as entropy_change,
    -- Health indicators
    CASE 
        WHEN (t.total_alerts - y.yesterday_alerts) > 20 THEN '⚠️ High Alert Volume'
        WHEN (t.total_alerts - y.yesterday_alerts) < -10 THEN '✅ Volume Decreasing'
        ELSE '➡️ Normal Volume'
    END as volume_status,
    CASE 
        WHEN (t.redundancy_percentage - y.yesterday_redundancy) > 5 THEN '⚠️ Redundancy Increasing'
        WHEN (t.redundancy_percentage - y.yesterday_redundancy) < -2 THEN '✅ Redundancy Improving'
        ELSE '➡️ Redundancy Stable'
    END as redundancy_status
FROM today_metrics t
CROSS JOIN yesterday_metrics y;
```

---

## 🚀 **Step 3: Automated Monitoring Queries**

### **3.1 Daily KPI Dashboard Query**
```sql
-- Create a daily KPI summary for dashboard
CREATE OR REPLACE VIEW `chronicle-dev-2be9.soc_data.daily_kpi_dashboard` AS
SELECT 
    analysis_date,
    total_alerts,
    redundancy_percentage,
    entropy_index,
    coherence_rating,
    processing_speed_rating,
    -- Success scoring
    CASE 
        WHEN redundancy_percentage < 20 AND coherence_rating IN ('Excellent', 'Good') 
        THEN '🎉 EXCELLENT'
        WHEN redundancy_percentage < 25 AND coherence_rating IN ('Good', 'Fair')
        THEN '✅ GOOD'
        WHEN redundancy_percentage < 30
        THEN '⚠️ FAIR'
        ELSE '❌ POOR'
    END as overall_rating,
    -- Trend indicators
    LAG(redundancy_percentage, 1) OVER (ORDER BY analysis_date) as prev_redundancy,
    LAG(entropy_index, 1) OVER (ORDER BY analysis_date) as prev_entropy
FROM `chronicle-dev-2be9.soc_data.kpi_summary`
ORDER BY analysis_date DESC;
```

### **3.2 Alert Clustering Analysis**
```sql
-- Analyze embedding clustering effectiveness
SELECT 
    similarity_group,
    COUNT(*) as cluster_count,
    AVG(embedding_similarity) as avg_similarity,
    AVG(rl_reward_score) as avg_reward,
    COUNT(DISTINCT classification) as unique_classifications,
    -- Clustering quality indicators
    CASE 
        WHEN AVG(embedding_similarity) > 0.8 THEN '🟢 High Quality'
        WHEN AVG(embedding_similarity) > 0.6 THEN '🟡 Medium Quality'
        WHEN AVG(embedding_similarity) > 0.4 THEN '🟠 Low Quality'
        ELSE '🔴 Poor Quality'
    END as clustering_quality
FROM `chronicle-dev-2be9.soc_data.processed_alerts`
WHERE embedding_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
GROUP BY similarity_group
ORDER BY avg_similarity DESC;
```

---

## 📊 **Step 4: Real-time Monitoring Setup**

### **4.1 Create Monitoring Alerts**
```sql
-- Set up BigQuery scheduled queries for monitoring
-- This query runs every hour to check for anomalies
SELECT 
    CURRENT_TIMESTAMP() as check_time,
    'HIGH_REDUNDANCY' as alert_type,
    redundancy_percentage as metric_value,
    'Redundancy above 25% threshold' as message
FROM `chronicle-dev-2be9.soc_data.kpi_summary`
WHERE analysis_date = CURRENT_DATE()
AND redundancy_percentage > 25

UNION ALL

SELECT 
    CURRENT_TIMESTAMP() as check_time,
    'HIGH_ENTROPY' as alert_type,
    entropy_index as metric_value,
    'Entropy index above 0.7 threshold' as message
FROM `chronicle-dev-2be9.soc_data.kpi_summary`
WHERE analysis_date = CURRENT_DATE()
AND entropy_index > 0.7;
```

### **4.2 Performance Optimization Queries**
```sql
-- Check query performance and optimize
SELECT 
    job_id,
    creation_time,
    total_bytes_processed,
    total_bytes_billed,
    total_slot_ms,
    ROUND(total_bytes_processed / POW(2, 30), 2) as gb_processed,
    ROUND(total_slot_ms / 1000, 2) as slot_seconds
FROM `chronicle-dev-2be9.region-us`.INFORMATION_SCHEMA.JOBS_BY_PROJECT
WHERE creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
AND job_type = 'QUERY'
ORDER BY total_bytes_processed DESC
LIMIT 10;
```

---

## 🎯 **Step 5: Success Criteria Monitoring**

### **5.1 Phase 1 Success Dashboard**
```sql
-- Comprehensive Phase 1 success assessment
WITH success_metrics AS (
    SELECT 
        analysis_date,
        total_alerts,
        redundancy_percentage,
        entropy_index,
        coherence_rating,
        -- Success criteria
        CASE WHEN redundancy_percentage < 20 THEN 1 ELSE 0 END as redundancy_success,
        CASE WHEN coherence_rating IN ('Excellent', 'Good') THEN 1 ELSE 0 END as coherence_success,
        CASE WHEN entropy_index < 0.5 THEN 1 ELSE 0 END as entropy_success
    FROM `chronicle-dev-2be9.soc_data.kpi_summary`
    WHERE analysis_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
),
daily_success AS (
    SELECT 
        analysis_date,
        redundancy_success + coherence_success + entropy_success as success_count,
        CASE 
            WHEN redundancy_success + coherence_success + entropy_success >= 2 
            THEN '✅ SUCCESS'
            ELSE '❌ NEEDS WORK'
        END as daily_status
    FROM success_metrics
)
SELECT 
    analysis_date,
    daily_status,
    success_count,
    -- Overall trend
    AVG(success_count) OVER (ORDER BY analysis_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) as weekly_avg_success
FROM daily_success
ORDER BY analysis_date DESC;
```

### **5.2 Improvement Recommendations**
```sql
-- Generate improvement recommendations based on current metrics
WITH current_status AS (
    SELECT 
        redundancy_percentage,
        entropy_index,
        coherence_rating,
        processing_speed_rating
    FROM `chronicle-dev-2be9.soc_data.kpi_summary`
    WHERE analysis_date = CURRENT_DATE()
)
SELECT 
    'Redundancy Reduction' as focus_area,
    CASE 
        WHEN redundancy_percentage > 25 THEN '🔴 CRITICAL: Focus on embedding clustering'
        WHEN redundancy_percentage > 20 THEN '🟡 MODERATE: Fine-tune similarity thresholds'
        ELSE '🟢 GOOD: Maintain current approach'
    END as recommendation
FROM current_status

UNION ALL

SELECT 
    'SOC Coherence' as focus_area,
    CASE 
        WHEN entropy_index > 0.7 THEN '🔴 CRITICAL: Analyze alert patterns for better organization'
        WHEN entropy_index > 0.5 THEN '🟡 MODERATE: Review clustering algorithms'
        ELSE '🟢 GOOD: SOC is well organized'
    END as recommendation
FROM current_status

UNION ALL

SELECT 
    'Processing Efficiency' as focus_area,
    CASE 
        WHEN processing_speed_rating = 'Very Slow' THEN '🔴 CRITICAL: Optimize embedding generation'
        WHEN processing_speed_rating = 'Slow' THEN '🟡 MODERATE: Review processing pipeline'
        ELSE '🟢 GOOD: Processing is efficient'
    END as recommendation
FROM current_status;
```

---

## 🔧 **Step 6: Troubleshooting Common Issues**

### **6.1 Data Quality Checks**
```sql
-- Check for missing or invalid data
SELECT 
    'Missing Embeddings' as issue_type,
    COUNT(*) as count
FROM `chronicle-dev-2be9.soc_data.processed_alerts`
WHERE embedding IS NULL OR ARRAY_LENGTH(embedding) != 768

UNION ALL

SELECT 
    'Invalid Timestamps' as issue_type,
    COUNT(*) as count
FROM `chronicle-dev-2be9.soc_data.processed_alerts`
WHERE timestamp IS NULL OR embedding_timestamp IS NULL

UNION ALL

SELECT 
    'Invalid Similarity Scores' as issue_type,
    COUNT(*) as count
FROM `chronicle-dev-2be9.soc_data.processed_alerts`
WHERE embedding_similarity < 0 OR embedding_similarity > 1;
```

### **6.2 Performance Monitoring**
```sql
-- Monitor BigQuery performance
SELECT 
    table_name,
    row_count,
    size_bytes,
    ROUND(size_bytes / POW(2, 30), 2) as size_gb,
    last_modified_time
FROM `chronicle-dev-2be9.soc_data.INFORMATION_SCHEMA.TABLES`
WHERE table_name IN ('processed_alerts', 'embedding_analysis_view', 'entropy_index_view')
ORDER BY size_bytes DESC;
```

---

## 📱 **Step 7: Dashboard Integration**

### **7.1 Access Your Dashboards**
- **KPI Dashboard**: http://localhost:8528
- **SOC Dashboard**: http://localhost:8527

### **7.2 Real-time Monitoring Commands**
```bash
# Check dashboard status
curl -s http://localhost:8528/health || echo "KPI Dashboard not running"
curl -s http://localhost:8527/health || echo "SOC Dashboard not running"

# Run Phase 1 measurement
python bigquery_phase1_guide.py

# Generate new demo data
python generate_phase1_demo_data.py
```

---

## 🎉 **Success Metrics Summary**

### **Current Status** (Based on Demo Data)
- ✅ **Infrastructure**: All BigQuery views operational
- ✅ **Coherence**: SOC is well organized (Good rating)
- ⚠️ **Redundancy**: 26% (target <20%)
- ⚠️ **Efficiency**: 41.5% fast processing ratio (target >60%)

### **Next Steps**
1. **Monitor daily** using the KPI dashboard
2. **Optimize clustering** to reduce redundancy below 20%
3. **Improve processing** to achieve >60% fast processing ratio
4. **Prepare for Phase 2** once Phase 1 criteria are met

### **Key Success Indicators**
- 🎯 Redundancy < 20%
- 🧠 Entropy Index < 0.5
- ⏱️ Fast Processing Ratio > 60%
- 📈 Coherence Rating: Excellent or Good

---

## 📚 **Additional Resources**

- [BigQuery Documentation](https://cloud.google.com/bigquery/docs)
- [Phase 1 Implementation Guide](./BIGQUERY_KPI_IMPLEMENTATION.md)
- [AI-Driven SOC Playbook](./Post%20Human%20SOC%20/AI_SoC_Implementation_Playbook.md)
- [Progress Tracking](./.agents/progress.md)

**Happy monitoring! 🚀**







