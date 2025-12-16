-- Fix processed_alerts table to include all dashboard columns
-- The dashboard might be using this table instead of taa_enhanced_results

-- First, let's see what's currently in processed_alerts
SELECT COUNT(*) as current_count FROM `chronicle-dev-2be9.gatra_database.processed_alerts`;

-- Add all the dashboard columns to processed_alerts
ALTER TABLE `chronicle-dev-2be9.gatra_database.processed_alerts`
ADD COLUMN IF NOT EXISTS ada_case_class STRING,
ADD COLUMN IF NOT EXISTS cra_action_type STRING,
ADD COLUMN IF NOT EXISTS ada_confidence FLOAT64,
ADD COLUMN IF NOT EXISTS taa_confidence FLOAT64,
ADD COLUMN IF NOT EXISTS ada_score FLOAT64,
ADD COLUMN IF NOT EXISTS taa_severity FLOAT64,
ADD COLUMN IF NOT EXISTS ada_valid BOOL,
ADD COLUMN IF NOT EXISTS taa_valid BOOL,
ADD COLUMN IF NOT EXISTS cra_success BOOL,
ADD COLUMN IF NOT EXISTS ada_reasoning STRING,
ADD COLUMN IF NOT EXISTS taa_reasoning STRING,
ADD COLUMN IF NOT EXISTS cra_reasoning STRING,
ADD COLUMN IF NOT EXISTS variable_of_importance STRING,
ADD COLUMN IF NOT EXISTS ada_detected STRING,
ADD COLUMN IF NOT EXISTS taa_created TIMESTAMP,
ADD COLUMN IF NOT EXISTS cra_created TIMESTAMP,
ADD COLUMN IF NOT EXISTS enhanced_classification STRING,
ADD COLUMN IF NOT EXISTS calibrated BOOL,
ADD COLUMN IF NOT EXISTS suppression_recommended BOOL,
ADD COLUMN IF NOT EXISTS threat_score FLOAT64,
ADD COLUMN IF NOT EXISTS raw_json JSON;

-- Now populate the processed_alerts table with data from taa_enhanced_results
INSERT INTO `chronicle-dev-2be9.gatra_database.processed_alerts` (
  alarmId,
  processed_timestamp,
  processed_by,
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
  taa_created,
  cra_created,
  enhanced_classification,
  calibrated,
  suppression_recommended,
  threat_score,
  raw_json
)
SELECT
  alarm_id AS alarmId,
  event_time AS processed_timestamp,
  processed_by,
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
  raw_json
FROM `chronicle-dev-2be9.gatra_database.taa_enhanced_results` ter
WHERE ter.event_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
  AND NOT EXISTS (
    -- Avoid duplicates
    SELECT 1 FROM `chronicle-dev-2be9.gatra_database.processed_alerts` p2
    WHERE p2.alarmId = ter.alarm_id
  );

-- Create a comprehensive view that the dashboard can use
CREATE OR REPLACE VIEW `chronicle-dev-2be9.gatra_database.dashboard_complete` AS
SELECT
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
  event_time AS taa_created,
  event_time AS cra_created,
  enhanced_classification,
  calibrated,
  suppression_recommended,
  threat_score,
  request_id,
  rule,
  source,
  processed_by,
  raw_json
FROM `chronicle-dev-2be9.gatra_database.taa_enhanced_results`
WHERE event_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)

UNION ALL

SELECT
  alarmId AS alarm_id,
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
  taa_created,
  cra_created,
  enhanced_classification,
  calibrated,
  suppression_recommended,
  threat_score,
  alarmId AS request_id,
  NULL AS rule,
  NULL AS source,
  processed_by,
  raw_json
FROM `chronicle-dev-2be9.gatra_database.processed_alerts`
WHERE processed_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
  AND ada_case_class IS NOT NULL

ORDER BY taa_created DESC;
