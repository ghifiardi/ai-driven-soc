-- Fixed Dashboard Alerts Table Query
-- This extracts data from the taa_comparison table which contains the enhanced TAA results

CREATE OR REPLACE VIEW `chronicle-dev-2be9.gatra_database.dashboard_alerts_table` AS
SELECT
  -- Alarm ID (from JSON)
  JSON_VALUE(raw, '$.alarm_id') AS alarm_id,
  
  -- ADA Case Class (Original classification)
  model_a AS ada_case_class,
  
  -- CRA Action Type (from JSON if available)
  JSON_VALUE(raw, '$.cra_action_type') AS cra_action_type,
  
  -- ADA Confidence (Original confidence)
  JSON_VALUE(raw, '$.original_confidence') AS ada_confidence,
  
  -- TAA Confidence (Enhanced/Calibrated confidence)
  JSON_VALUE(raw, '$.enhanced_confidence') AS taa_confidence,
  
  -- ADA Score (from JSON if available)
  JSON_VALUE(raw, '$.ada_score') AS ada_score,
  
  -- TAA Severity (Enhanced severity score)
  JSON_VALUE(raw, '$.enhanced_severity') AS taa_severity,
  
  -- ADA Valid (Original anomaly flag)
  CASE 
    WHEN JSON_VALUE(raw, '$.original_is_anomaly') = 'true' THEN TRUE
    ELSE FALSE
  END AS ada_valid,
  
  -- TAA Valid (Enhanced anomaly flag)
  CASE 
    WHEN JSON_VALUE(raw, '$.enhanced_is_anomaly') = 'true' THEN TRUE
    ELSE FALSE
  END AS taa_valid,
  
  -- CRA Success (from JSON if available)
  CASE 
    WHEN JSON_VALUE(raw, '$.cra_success') = 'true' THEN TRUE
    ELSE FALSE
  END AS cra_success,
  
  -- ADA Reasoning (from JSON if available)
  JSON_VALUE(raw, '$.ada_reasoning') AS ada_reasoning,
  
  -- TAA Reasoning (Enhanced reasoning)
  JSON_VALUE(raw, '$.classification_reasoning') AS taa_reasoning,
  
  -- CRA Reasoning (from JSON if available)
  JSON_VALUE(raw, '$.cra_reasoning') AS cra_reasoning,
  
  -- Variable of Importance (from JSON if available)
  JSON_VALUE(raw, '$.variable_of_importance') AS variable_of_importance,
  
  -- ADA Detected (from JSON if available, default to '2' as shown in dashboard)
  COALESCE(JSON_VALUE(raw, '$.ada_detected'), '2') AS ada_detected,
  
  -- Timestamp
  created_at AS timestamp,
  
  -- Additional metadata
  request_id,
  verdict,
  confidence,
  
  -- Enhanced classification details
  JSON_VALUE(raw, '$.enhanced_classification') AS enhanced_classification,
  JSON_VALUE(raw, '$.original_classification') AS original_classification,
  JSON_VALUE(raw, '$.classification_changed') AS classification_changed,
  JSON_VALUE(raw, '$.confidence_difference') AS confidence_difference,
  JSON_VALUE(raw, '$.enhanced_threat_score') AS threat_score,
  
  -- Migration info
  JSON_VALUE(raw, '$.full_migration_mode') AS full_migration_mode,
  JSON_VALUE(raw, '$.traffic_percentage') AS traffic_percentage,
  JSON_VALUE(raw, '$.migration_phase') AS migration_phase,
  
  -- Dashboard improvement metrics
  JSON_VALUE(raw, '$.dashboard_improvement_metrics.confidence_improvement') AS confidence_improvement,
  JSON_VALUE(raw, '$.dashboard_improvement_metrics.false_positive_reduction') AS false_positive_reduction,
  JSON_VALUE(raw, '$.dashboard_improvement_metrics.severity_alignment') AS severity_alignment,
  JSON_VALUE(raw, '$.dashboard_improvement_metrics.threat_detection_improvement') AS threat_detection_improvement

FROM `chronicle-dev-2be9.gatra_database.taa_comparison`
WHERE created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
ORDER BY created_at DESC;


