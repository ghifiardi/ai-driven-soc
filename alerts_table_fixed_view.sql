-- Fixed Alerts Table View with Robust Field Extraction
-- This view handles both top-level columns and JSON field extraction

CREATE OR REPLACE VIEW `chronicle-dev-2be9.gatra_database.alerts_table_fixed` AS
WITH base AS (
  SELECT
    e.request_id,
    e.event_time,
    
    -- Alarm ID from multiple locations (case-insensitive)
    COALESCE(
      e.alarm_id,
      JSON_VALUE(e.raw_json, '$.alarm_id'),
      JSON_VALUE(e.raw_json, '$.alarmId'),
      JSON_VALUE(e.raw_json, '$.alarmID'),
      JSON_VALUE(e.raw_json, '$.alarm_id'),
      JSON_VALUE(e.raw_json, '$.ALARM_ID')
    ) AS alarm_id,

    -- ADA Case Class from multiple locations
    COALESCE(
      e.ada_case_class,
      JSON_VALUE(e.raw_json, '$.ADA.case_class'),
      JSON_VALUE(e.raw_json, '$.ada.case_class'),
      JSON_VALUE(e.raw_json, '$.case_class'),
      JSON_VALUE(e.raw_json, '$.ada_case_class'),
      JSON_VALUE(e.raw_json, '$.enhanced_classification')
    ) AS ada_case_class,

    -- CRA Action Type from multiple locations
    COALESCE(
      e.cra_action_type,
      JSON_VALUE(e.raw_json, '$.CRA.action_type'),
      JSON_VALUE(e.raw_json, '$.cra.action_type'),
      JSON_VALUE(e.raw_json, '$.action_type'),
      JSON_VALUE(e.raw_json, '$.cra_action_type')
    ) AS cra_action_type,

    -- ADA Confidence (from multiple sources)
    COALESCE(
      e.ada_confidence,
      JSON_VALUE(e.raw_json, '$.ADA.confidence'),
      JSON_VALUE(e.raw_json, '$.ada.confidence'),
      JSON_VALUE(e.raw_json, '$.confidence'),
      e.confidence
    ) AS ada_confidence,

    -- TAA Confidence (enhanced/calibrated)
    COALESCE(
      e.taa_confidence,
      JSON_VALUE(e.raw_json, '$.enhanced_confidence'),
      JSON_VALUE(e.raw_json, '$.taa_confidence'),
      JSON_VALUE(e.raw_json, '$.calibrated_confidence'),
      e.confidence
    ) AS taa_confidence,

    -- ADA Score
    COALESCE(
      e.ada_score,
      JSON_VALUE(e.raw_json, '$.ADA.score'),
      JSON_VALUE(e.raw_json, '$.ada.score'),
      JSON_VALUE(e.raw_json, '$.score')
    ) AS ada_score,

    -- TAA Severity (enhanced/calibrated)
    COALESCE(
      e.taa_severity,
      JSON_VALUE(e.raw_json, '$.severity_score'),
      JSON_VALUE(e.raw_json, '$.enhanced_severity'),
      JSON_VALUE(e.raw_json, '$.taa_severity'),
      e.severity
    ) AS taa_severity,

    -- ADA Valid (boolean conversion)
    CASE 
      WHEN COALESCE(
        e.ada_valid,
        JSON_VALUE(e.raw_json, '$.ADA.valid'),
        JSON_VALUE(e.raw_json, '$.ada.valid'),
        JSON_VALUE(e.raw_json, '$.valid')
      ) IN ('true', '1', 'yes', 'True', 'TRUE') THEN TRUE
      ELSE FALSE
    END AS ada_valid,

    -- TAA Valid (boolean conversion)
    CASE 
      WHEN COALESCE(
        e.taa_valid,
        JSON_VALUE(e.raw_json, '$.taa_valid'),
        JSON_VALUE(e.raw_json, '$.enhanced_valid'),
        JSON_VALUE(e.raw_json, '$.valid')
      ) IN ('true', '1', 'yes', 'True', 'TRUE') THEN TRUE
      ELSE FALSE
    END AS taa_valid,

    -- CRA Success (boolean conversion)
    CASE 
      WHEN COALESCE(
        e.cra_success,
        JSON_VALUE(e.raw_json, '$.CRA.success'),
        JSON_VALUE(e.raw_json, '$.cra.success'),
        JSON_VALUE(e.raw_json, '$.success')
      ) IN ('true', '1', 'yes', 'True', 'TRUE') THEN TRUE
      ELSE FALSE
    END AS cra_success,

    -- Reasoning fields
    COALESCE(
      e.ada_reasoning,
      JSON_VALUE(e.raw_json, '$.ADA.reasoning'),
      JSON_VALUE(e.raw_json, '$.ada.reasoning'),
      JSON_VALUE(e.raw_json, '$.reasoning')
    ) AS ada_reasoning,

    COALESCE(
      e.taa_reasoning,
      JSON_VALUE(e.raw_json, '$.taa_reasoning'),
      JSON_VALUE(e.raw_json, '$.enhanced_reasoning'),
      JSON_VALUE(e.raw_json, '$.classification_reasoning'),
      JSON_VALUE(e.raw_json, '$.reasoning')
    ) AS taa_reasoning,

    COALESCE(
      e.cra_reasoning,
      JSON_VALUE(e.raw_json, '$.CRA.reasoning'),
      JSON_VALUE(e.raw_json, '$.cra.reasoning'),
      JSON_VALUE(e.raw_json, '$.reasoning')
    ) AS cra_reasoning,

    -- Variable of Importance
    COALESCE(
      e.variable_of_importance,
      JSON_VALUE(e.raw_json, '$.variable_of_importance'),
      JSON_VALUE(e.raw_json, '$.key_variable'),
      JSON_VALUE(e.raw_json, '$.important_field')
    ) AS variable_of_importance,

    -- ADA Detected
    COALESCE(
      e.ada_detected,
      JSON_VALUE(e.raw_json, '$.ADA.detected'),
      JSON_VALUE(e.raw_json, '$.ada.detected'),
      JSON_VALUE(e.raw_json, '$.detected')
    ) AS ada_detected,

    -- Timestamp
    e.event_time AS timestamp,

    -- Additional metadata
    e.rule,
    e.source,
    e.processed_by,
    e.enhanced_classification,
    e.calibrated,
    e.suppression_recommended,
    e.threat_score

  FROM `chronicle-dev-2be9.gatra_database.taa_enhanced_results` e
  WHERE e.event_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
)

SELECT 
  *,
  -- Add computed fields
  CASE 
    WHEN ada_valid = TRUE AND taa_valid = TRUE THEN TRUE
    ELSE FALSE
  END AS both_valid,
  
  CASE 
    WHEN taa_confidence > 0.7 THEN 'High'
    WHEN taa_confidence > 0.4 THEN 'Medium'
    ELSE 'Low'
  END AS confidence_level,
  
  CASE 
    WHEN taa_severity > 0.7 THEN 'High'
    WHEN taa_severity > 0.4 THEN 'Medium'
    ELSE 'Low'
  END AS severity_level

FROM base
-- Do NOT filter out nulls here - let the UI handle filtering
ORDER BY event_time DESC;


