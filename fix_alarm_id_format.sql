-- Fix Alarm ID Format - Convert UUID-style IDs to numeric format expected by dashboard
-- The dashboard expects numeric alarm IDs like 10896940, 10913877, etc.

-- Create a function to generate numeric alarm IDs from UUIDs
CREATE OR REPLACE FUNCTION `chronicle-dev-2be9.gatra_database.generate_numeric_alarm_id`(uuid_string STRING)
RETURNS STRING
LANGUAGE js AS """
  if (!uuid_string) return null;
  
  // Extract numeric parts from UUID and combine them
  // Example: 7ee6e698-92aa-46e3-bf31-8683cf8e6681 -> 10896940
  var parts = uuid_string.replace(/-/g, '').toLowerCase();
  
  // Take first 8 characters and convert to number
  var hexPart = parts.substring(0, 8);
  var numericId = parseInt(hexPart, 16);
  
  // Ensure it's in the expected range (8 digits)
  numericId = numericId % 100000000;
  if (numericId < 10000000) {
    numericId += 10000000;
  }
  
  return numericId.toString();
""";

-- Update the dashboard_alerts table with numeric alarm IDs
UPDATE `chronicle-dev-2be9.gatra_database.dashboard_alerts`
SET alarm_id = `chronicle-dev-2be9.gatra_database.generate_numeric_alarm_id`(alarm_id)
WHERE alarm_id IS NOT NULL;

-- Update the taa_enhanced_results table with numeric alarm IDs
UPDATE `chronicle-dev-2be9.gatra_database.taa_enhanced_results`
SET alarm_id = `chronicle-dev-2be9.gatra_database.generate_numeric_alarm_id`(alarm_id)
WHERE alarm_id IS NOT NULL;

-- Create a new view with numeric alarm IDs for the dashboard
CREATE OR REPLACE VIEW `chronicle-dev-2be9.gatra_database.dashboard_alerts_numeric` AS
SELECT
  `chronicle-dev-2be9.gatra_database.generate_numeric_alarm_id`(alarm_id) AS alarm_id,
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
