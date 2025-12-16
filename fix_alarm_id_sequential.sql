-- Fix Alarm ID Format - Generate sequential numeric alarm IDs
-- Simple approach that generates numeric IDs in the format expected by dashboard

-- Create a view with sequential numeric alarm IDs
CREATE OR REPLACE VIEW `chronicle-dev-2be9.gatra_database.dashboard_alerts_numeric` AS
SELECT
  -- Generate numeric alarm ID starting from 10800000
  CAST(10800000 + ROW_NUMBER() OVER (ORDER BY event_time DESC) AS STRING) AS alarm_id,
  
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
ORDER BY event_time DESC;

-- Also update the main dashboard_alerts table
UPDATE `chronicle-dev-2be9.gatra_database.dashboard_alerts`
SET alarm_id = CAST(10800000 + ROW_NUMBER() OVER (ORDER BY taa_created DESC) AS STRING)
WHERE alarm_id IS NOT NULL;

-- Update the taa_enhanced_results table as well
UPDATE `chronicle-dev-2be9.gatra_database.taa_enhanced_results`
SET alarm_id = CAST(10800000 + ROW_NUMBER() OVER (ORDER BY event_time DESC) AS STRING)
WHERE alarm_id IS NOT NULL;


