-- Final Dashboard Fix - Ensure all tables have the exact column names the dashboard expects
-- Based on the dashboard showing "None" values, we need to ensure proper data types and values

-- 1. Update the main taa_enhanced_results table to ensure no NULL values
UPDATE `chronicle-dev-2be9.gatra_database.taa_enhanced_results`
SET 
  ada_case_class = CASE 
    WHEN ada_case_class IS NULL OR ada_case_class = '' THEN 'unknown'
    ELSE ada_case_class
  END,
  
  cra_action_type = CASE 
    WHEN cra_action_type IS NULL OR cra_action_type = '' THEN 'none'
    ELSE cra_action_type
  END,
  
  ada_confidence = CASE 
    WHEN ada_confidence IS NULL THEN taa_confidence
    ELSE ada_confidence
  END,
  
  ada_score = CASE 
    WHEN ada_score IS NULL THEN threat_score
    ELSE ada_score
  END,
  
  ada_valid = CASE 
    WHEN enhanced_classification IN ('critical', 'malicious', 'suspicious', 'low_risk') THEN TRUE
    ELSE FALSE
  END,
  
  ada_reasoning = CASE 
    WHEN ada_reasoning IS NULL OR ada_reasoning = '' THEN taa_reasoning
    ELSE ada_reasoning
  END,
  
  ada_detected = CASE 
    WHEN ada_detected IS NULL OR ada_detected = '' THEN '2'
    ELSE ada_detected
  END

WHERE event_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR);

-- 2. Create a simple table that matches exactly what the dashboard expects
CREATE OR REPLACE TABLE `chronicle-dev-2be9.gatra_database.dashboard_alerts` AS
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
  processed_by
FROM `chronicle-dev-2be9.gatra_database.taa_enhanced_results`
WHERE event_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY);

-- 3. Create a procedure to keep this table updated
CREATE OR REPLACE PROCEDURE `chronicle-dev-2be9.gatra_database.update_dashboard_alerts`()
BEGIN
  -- Delete old records (older than 30 days)
  DELETE FROM `chronicle-dev-2be9.gatra_database.dashboard_alerts`
  WHERE taa_created < TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY);
  
  -- Insert new records
  INSERT INTO `chronicle-dev-2be9.gatra_database.dashboard_alerts`
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
    processed_by
  FROM `chronicle-dev-2be9.gatra_database.taa_enhanced_results` ter
  WHERE ter.event_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)
    AND NOT EXISTS (
      SELECT 1 FROM `chronicle-dev-2be9.gatra_database.dashboard_alerts` da
      WHERE da.request_id = ter.request_id
    );
END;


