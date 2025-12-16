-- Quick Dashboard Test Query
-- Run this to verify the dashboard will populate correctly

SELECT
  event_time,
  request_id,
  
  -- Alarm ID (robust extraction)
  COALESCE(
    alarm_id,
    JSON_VALUE(raw_json, '$.alarm_id'),
    JSON_VALUE(raw_json, '$.alarmId'),
    JSON_VALUE(raw_json, '$.alarmID')
  ) AS alarm_id,
  
  -- ADA Case Class (robust extraction)
  COALESCE(
    ada_case_class,
    JSON_VALUE(raw_json, '$.enhanced_classification'),
    JSON_VALUE(raw_json, '$.ADA.case_class'),
    JSON_VALUE(raw_json, '$.ada.case_class'),
    JSON_VALUE(raw_json, '$.case_class')
  ) AS ada_case_class,
  
  -- CRA Action Type (robust extraction)
  COALESCE(
    cra_action_type,
    JSON_VALUE(raw_json, '$.CRA.action_type'),
    JSON_VALUE(raw_json, '$.cra.action_type'),
    JSON_VALUE(raw_json, '$.action_type')
  ) AS cra_action_type,
  
  -- ADA Confidence
  COALESCE(
    ada_confidence,
    SAFE_CAST(JSON_VALUE(raw_json, '$.confidence') AS FLOAT64),
    confidence
  ) AS ada_confidence,
  
  -- TAA Confidence (enhanced/calibrated)
  COALESCE(
    taa_confidence,
    SAFE_CAST(JSON_VALUE(raw_json, '$.enhanced_confidence') AS FLOAT64),
    SAFE_CAST(JSON_VALUE(raw_json, '$.calibrated_confidence') AS FLOAT64),
    confidence
  ) AS taa_confidence,
  
  -- ADA Score
  COALESCE(
    ada_score,
    SAFE_CAST(JSON_VALUE(raw_json, '$.ADA.score') AS FLOAT64),
    SAFE_CAST(JSON_VALUE(raw_json, '$.score') AS FLOAT64)
  ) AS ada_score,
  
  -- TAA Severity (enhanced/calibrated)
  COALESCE(
    taa_severity,
    SAFE_CAST(JSON_VALUE(raw_json, '$.severity_score') AS FLOAT64),
    SAFE_CAST(JSON_VALUE(raw_json, '$.enhanced_severity') AS FLOAT64),
    severity
  ) AS taa_severity,
  
  -- ADA Valid
  CASE 
    WHEN COALESCE(
      ada_valid,
      SAFE_CAST(JSON_VALUE(raw_json, '$.ADA.valid') AS BOOL),
      SAFE_CAST(JSON_VALUE(raw_json, '$.valid') AS BOOL)
    ) THEN TRUE
    ELSE FALSE
  END AS ada_valid,
  
  -- TAA Valid
  CASE 
    WHEN COALESCE(
      taa_valid,
      SAFE_CAST(JSON_VALUE(raw_json, '$.enhanced_is_anomaly') AS BOOL),
      SAFE_CAST(JSON_VALUE(raw_json, '$.valid') AS BOOL)
    ) THEN TRUE
    ELSE FALSE
  END AS taa_valid,
  
  -- CRA Success
  CASE 
    WHEN COALESCE(
      cra_success,
      SAFE_CAST(JSON_VALUE(raw_json, '$.CRA.success') AS BOOL),
      SAFE_CAST(JSON_VALUE(raw_json, '$.success') AS BOOL)
    ) THEN TRUE
    ELSE FALSE
  END AS cra_success,
  
  -- Reasoning fields
  COALESCE(
    ada_reasoning,
    JSON_VALUE(raw_json, '$.ADA.reasoning'),
    JSON_VALUE(raw_json, '$.reasoning')
  ) AS ada_reasoning,
  
  COALESCE(
    taa_reasoning,
    JSON_VALUE(raw_json, '$.classification_reasoning'),
    JSON_VALUE(raw_json, '$.enhanced_reasoning'),
    JSON_VALUE(raw_json, '$.reasoning')
  ) AS taa_reasoning,
  
  COALESCE(
    cra_reasoning,
    JSON_VALUE(raw_json, '$.CRA.reasoning'),
    JSON_VALUE(raw_json, '$.reasoning')
  ) AS cra_reasoning,
  
  -- Variable of Importance
  COALESCE(
    variable_of_importance,
    JSON_VALUE(raw_json, '$.variable_of_importance'),
    JSON_VALUE(raw_json, '$.key_variable')
  ) AS variable_of_importance,
  
  -- ADA Detected
  COALESCE(
    ada_detected,
    JSON_VALUE(raw_json, '$.ADA.detected'),
    JSON_VALUE(raw_json, '$.detected'),
    '2'  -- Default value from your dashboard
  ) AS ada_detected,
  
  -- Metadata
  rule,
  source,
  processed_by,
  enhanced_classification,
  calibrated,
  suppression_recommended,
  threat_score

FROM `chronicle-dev-2be9.gatra_database.taa_enhanced_results`
WHERE event_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
ORDER BY event_time DESC
LIMIT 200;


