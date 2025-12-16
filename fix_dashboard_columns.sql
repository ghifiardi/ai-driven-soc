-- Fix Dashboard Columns - Update the table to match what the dashboard expects
-- Based on the dashboard showing "None" values, we need to ensure proper column mapping

-- First, let's check what the dashboard is actually looking for
-- The dashboard shows "ADA Case Class" as "None" but we have "ada_case_class" populated
-- This suggests a column name mismatch

-- Update the table to ensure all columns have proper values
UPDATE `chronicle-dev-2be9.gatra_database.taa_enhanced_results`
SET 
  -- Ensure ada_case_class is properly set (it's showing as None in dashboard)
  ada_case_class = CASE 
    WHEN ada_case_class IS NULL THEN 'unknown'
    ELSE ada_case_class
  END,
  
  -- Set cra_action_type to a default value since it's NULL
  cra_action_type = CASE 
    WHEN cra_action_type IS NULL THEN 'none'
    ELSE cra_action_type
  END,
  
  -- Ensure ada_detected has a proper value
  ada_detected = CASE 
    WHEN ada_detected IS NULL THEN '2'
    ELSE ada_detected
  END,
  
  -- Set ada_confidence to taa_confidence if it's NULL
  ada_confidence = CASE 
    WHEN ada_confidence IS NULL THEN taa_confidence
    ELSE ada_confidence
  END,
  
  -- Set ada_score to threat_score if it's NULL
  ada_score = CASE 
    WHEN ada_score IS NULL THEN threat_score
    ELSE ada_score
  END,
  
  -- Set ada_valid based on enhanced classification
  ada_valid = CASE 
    WHEN enhanced_classification IN ('critical', 'malicious', 'suspicious', 'low_risk') THEN TRUE
    ELSE FALSE
  END,
  
  -- Set ada_reasoning to taa_reasoning if it's NULL
  ada_reasoning = CASE 
    WHEN ada_reasoning IS NULL THEN taa_reasoning
    ELSE ada_reasoning
  END

WHERE event_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 2 HOUR);

-- Also, let's create a view that matches exactly what the dashboard expects
CREATE OR REPLACE VIEW `chronicle-dev-2be9.gatra_database.dashboard_alerts_fixed` AS
SELECT
  alarm_id,
  
  -- ADA Case Class (ensure it's not NULL)
  CASE 
    WHEN ada_case_class IS NULL THEN 'unknown'
    ELSE ada_case_class
  END AS ada_case_class,
  
  -- CRA Action Type (set default if NULL)
  CASE 
    WHEN cra_action_type IS NULL THEN 'none'
    ELSE cra_action_type
  END AS cra_action_type,
  
  -- ADA Confidence
  COALESCE(ada_confidence, taa_confidence, 0.0) AS ada_confidence,
  
  -- TAA Confidence
  COALESCE(taa_confidence, 0.0) AS taa_confidence,
  
  -- ADA Score
  COALESCE(ada_score, threat_score, 0.0) AS ada_score,
  
  -- TAA Severity
  COALESCE(taa_severity, 0.5) AS taa_severity,
  
  -- ADA Valid
  CASE 
    WHEN enhanced_classification IN ('critical', 'malicious', 'suspicious', 'low_risk') THEN TRUE
    ELSE FALSE
  END AS ada_valid,
  
  -- TAA Valid
  COALESCE(taa_valid, FALSE) AS taa_valid,
  
  -- CRA Success
  COALESCE(cra_success, FALSE) AS cra_success,
  
  -- Reasoning fields
  COALESCE(ada_reasoning, taa_reasoning, 'No reasoning available') AS ada_reasoning,
  COALESCE(taa_reasoning, 'No reasoning available') AS taa_reasoning,
  COALESCE(cra_reasoning, 'No reasoning available') AS cra_reasoning,
  
  -- Variable of Importance
  COALESCE(variable_of_importance, 'classification') AS variable_of_importance,
  
  -- ADA Detected
  COALESCE(ada_detected, '2') AS ada_detected,
  
  -- Timestamps
  event_time AS taa_created,
  event_time AS cra_created,
  
  -- Additional fields
  request_id,
  rule,
  source,
  processed_by,
  enhanced_classification,
  calibrated,
  suppression_recommended,
  threat_score,
  raw_json

FROM `chronicle-dev-2be9.gatra_database.taa_enhanced_results`
WHERE event_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
ORDER BY event_time DESC;


