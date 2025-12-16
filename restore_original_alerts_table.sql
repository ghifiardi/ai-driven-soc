-- Restore Original Alerts Table Structure
-- This creates/updates the original taa_enhanced_results table with the enhanced data

-- 1) Create the original table structure if it doesn't exist
CREATE TABLE IF NOT EXISTS `chronicle-dev-2be9.gatra_database.taa_enhanced_results` (
  alarm_id STRING,
  ada_case_class STRING,
  cra_action_type STRING,
  ada_confidence FLOAT64,
  taa_confidence FLOAT64,
  ada_score FLOAT64,
  taa_severity FLOAT64,
  ada_valid BOOL,
  taa_valid BOOL,
  cra_success BOOL,
  ada_reasoning STRING,
  taa_reasoning STRING,
  cra_reasoning STRING,
  variable_of_importance STRING,
  ada_detected STRING,
  event_time TIMESTAMP,
  request_id STRING,
  rule STRING,
  source STRING,
  processed_by STRING,
  enhanced_classification STRING,
  calibrated BOOL,
  suppression_recommended BOOL,
  threat_score FLOAT64,
  raw_json JSON
)
PARTITION BY DATE(event_time)
CLUSTER BY alarm_id, processed_by;

-- 2) Insert/Update data from taa_comparison into the original structure
INSERT INTO `chronicle-dev-2be9.gatra_database.taa_enhanced_results` (
  alarm_id,
  ada_case_class,
  cra_action_type,
  ada_confidence,
  taa_confidence,
  ada_score,
  taa_severity,
  ada_valid,
  taa_valid,
  cra_success,
  ada_reasoning,
  taa_reasoning,
  cra_reasoning,
  variable_of_importance,
  ada_detected,
  event_time,
  request_id,
  rule,
  source,
  processed_by,
  enhanced_classification,
  calibrated,
  suppression_recommended,
  threat_score,
  raw_json
)
SELECT
  -- Extract alarm_id from JSON
  JSON_VALUE(raw, '$.alarm_id') AS alarm_id,
  
  -- ADA Case Class (Original classification)
  model_a AS ada_case_class,
  
  -- CRA Action Type (from JSON if available)
  JSON_VALUE(raw, '$.cra_action_type') AS cra_action_type,
  
  -- ADA Confidence (Original confidence)
  SAFE_CAST(JSON_VALUE(raw, '$.original_confidence') AS FLOAT64) AS ada_confidence,
  
  -- TAA Confidence (Enhanced/Calibrated confidence)
  SAFE_CAST(JSON_VALUE(raw, '$.enhanced_confidence') AS FLOAT64) AS taa_confidence,
  
  -- ADA Score (from JSON if available)
  SAFE_CAST(JSON_VALUE(raw, '$.ada_score') AS FLOAT64) AS ada_score,
  
  -- TAA Severity (Enhanced severity score)
  SAFE_CAST(JSON_VALUE(raw, '$.enhanced_severity') AS FLOAT64) AS taa_severity,
  
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
  created_at AS event_time,
  
  -- Request ID
  request_id,
  
  -- Additional metadata
  JSON_VALUE(raw, '$.rule') AS rule,
  JSON_VALUE(raw, '$.source') AS source,
  'enhanced_taa_calibrated' AS processed_by,
  
  -- Enhanced classification details
  JSON_VALUE(raw, '$.enhanced_classification') AS enhanced_classification,
  TRUE AS calibrated,
  CASE 
    WHEN JSON_VALUE(raw, '$.suppression_recommended') = 'true' THEN TRUE
    ELSE FALSE
  END AS suppression_recommended,
  
  -- Threat score
  SAFE_CAST(JSON_VALUE(raw, '$.enhanced_threat_score') AS FLOAT64) AS threat_score,
  
  -- Keep raw JSON for reference
  raw AS raw_json

FROM `chronicle-dev-2be9.gatra_database.taa_comparison` tc
WHERE tc.created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
  AND NOT EXISTS (
    -- Avoid duplicates by checking if record already exists
    SELECT 1 FROM `chronicle-dev-2be9.gatra_database.taa_enhanced_results` t2
    WHERE t2.request_id = tc.request_id
  );
