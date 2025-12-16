-- Test Dashboard Query - Verify the data extraction works
SELECT
  -- Alarm ID
  JSON_VALUE(raw, '$.alarm_id') AS alarm_id,
  
  -- ADA Case Class (Original)
  model_a AS ada_case_class,
  
  -- TAA Confidence (Enhanced/Calibrated)
  JSON_VALUE(raw, '$.enhanced_confidence') AS taa_confidence,
  
  -- TAA Severity (Enhanced)
  JSON_VALUE(raw, '$.enhanced_severity') AS taa_severity,
  
  -- Enhanced Classification
  JSON_VALUE(raw, '$.enhanced_classification') AS enhanced_classification,
  
  -- Original Classification
  JSON_VALUE(raw, '$.original_classification') AS original_classification,
  
  -- Classification Changed
  JSON_VALUE(raw, '$.classification_changed') AS classification_changed,
  
  -- Confidence Difference
  JSON_VALUE(raw, '$.confidence_difference') AS confidence_difference,
  
  -- Threat Score
  JSON_VALUE(raw, '$.enhanced_threat_score') AS threat_score,
  
  -- Migration Info
  JSON_VALUE(raw, '$.traffic_percentage') AS traffic_percentage,
  
  -- Timestamp
  created_at AS timestamp

FROM `chronicle-dev-2be9.gatra_database.taa_comparison`
WHERE created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 2 HOUR)
ORDER BY created_at DESC
LIMIT 20;


