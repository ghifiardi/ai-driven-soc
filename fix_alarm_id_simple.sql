-- Fix Alarm ID Format - Convert UUID-style IDs to numeric format
-- Simple approach using SQL functions

-- Create a view with numeric alarm IDs for the dashboard
CREATE OR REPLACE VIEW `chronicle-dev-2be9.gatra_database.dashboard_alerts_numeric` AS
SELECT
  -- Convert UUID to numeric alarm ID
  CAST(
    CONCAT(
      CAST(CAST(SUBSTR(REGEXP_REPLACE(alarm_id, '-', ''), 1, 4) AS INT64) % 100 AS STRING),
      CAST(CAST(SUBSTR(REGEXP_REPLACE(alarm_id, '-', ''), 5, 4) AS INT64) % 100 AS STRING),
      CAST(CAST(SUBSTR(REGEXP_REPLACE(alarm_id, '-', ''), 9, 4) AS INT64) % 100 AS STRING),
      CAST(CAST(SUBSTR(REGEXP_REPLACE(alarm_id, '-', ''), 13, 4) AS INT64) % 100 AS STRING)
    ) AS INT64
  ) AS alarm_id,
  
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
  event_time AS taa_created,
  event_time AS cra_created,
  enhanced_classification,
  calibrated,
  suppression_recommended,
  threat_score,
  request_id,
  rule,
  source,
  processed_by

FROM `chronicle-dev-2be9.gatra_database.taa_enhanced_results`
WHERE event_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
  AND alarm_id IS NOT NULL
ORDER BY event_time DESC;


