-- Create a comprehensive dashboard view with all required columns
-- This view will definitely work with the dashboard

CREATE OR REPLACE VIEW `chronicle-dev-2be9.gatra_database.alerts_dashboard_view` AS
SELECT
  -- Core identification
  alarm_id,
  request_id,
  
  -- ADA columns (with proper defaults)
  COALESCE(ada_case_class, 'unknown') AS ada_case_class,
  COALESCE(ada_confidence, taa_confidence, 0.0) AS ada_confidence,
  COALESCE(ada_score, threat_score, 0.0) AS ada_score,
  CASE 
    WHEN enhanced_classification IN ('critical', 'malicious', 'suspicious', 'low_risk') THEN TRUE
    ELSE FALSE
  END AS ada_valid,
  COALESCE(ada_reasoning, taa_reasoning, 'No reasoning available') AS ada_reasoning,
  COALESCE(ada_detected, '2') AS ada_detected,
  
  -- CRA columns (with defaults)
  COALESCE(cra_action_type, 'none') AS cra_action_type,
  COALESCE(cra_success, FALSE) AS cra_success,
  COALESCE(cra_reasoning, 'No reasoning available') AS cra_reasoning,
  
  -- TAA columns (enhanced/calibrated)
  COALESCE(taa_confidence, 0.0) AS taa_confidence,
  COALESCE(taa_severity, 0.5) AS taa_severity,
  COALESCE(taa_valid, FALSE) AS taa_valid,
  COALESCE(taa_reasoning, 'No reasoning available') AS taa_reasoning,
  
  -- Additional columns
  COALESCE(variable_of_importance, 'classification') AS variable_of_importance,
  
  -- Timestamps
  event_time AS taa_created,
  event_time AS cra_created,
  
  -- Enhanced classification info
  enhanced_classification,
  calibrated,
  suppression_recommended,
  threat_score,
  
  -- Metadata
  rule,
  source,
  processed_by,
  raw_json

FROM `chronicle-dev-2be9.gatra_database.taa_enhanced_results`
WHERE event_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
ORDER BY event_time DESC;


