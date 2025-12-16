-- Fix Schema Columns - Add missing columns if needed
-- Run this if the diagnostics show columns are missing at schema level

-- 1) Add missing columns to the table
ALTER TABLE `chronicle-dev-2be9.gatra_database.taa_enhanced_results`
ADD COLUMN IF NOT EXISTS alarm_id STRING,
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
ADD COLUMN IF NOT EXISTS ada_detected STRING;

-- 2) Populate columns from JSON for recent data
UPDATE `chronicle-dev-2be9.gatra_database.taa_enhanced_results`
SET 
  alarm_id = COALESCE(alarm_id, JSON_VALUE(raw_json, '$.alarm_id')),
  ada_case_class = COALESCE(ada_case_class, JSON_VALUE(raw_json, '$.enhanced_classification')),
  cra_action_type = COALESCE(cra_action_type, JSON_VALUE(raw_json, '$.CRA.action_type')),
  ada_confidence = COALESCE(ada_confidence, SAFE_CAST(JSON_VALUE(raw_json, '$.confidence') AS FLOAT64)),
  taa_confidence = COALESCE(taa_confidence, SAFE_CAST(JSON_VALUE(raw_json, '$.enhanced_confidence') AS FLOAT64)),
  ada_score = COALESCE(ada_score, SAFE_CAST(JSON_VALUE(raw_json, '$.ADA.score') AS FLOAT64)),
  taa_severity = COALESCE(taa_severity, SAFE_CAST(JSON_VALUE(raw_json, '$.severity_score') AS FLOAT64)),
  ada_valid = COALESCE(ada_valid, SAFE_CAST(JSON_VALUE(raw_json, '$.ADA.valid') AS BOOL)),
  taa_valid = COALESCE(taa_valid, SAFE_CAST(JSON_VALUE(raw_json, '$.enhanced_is_anomaly') AS BOOL)),
  cra_success = COALESCE(cra_success, SAFE_CAST(JSON_VALUE(raw_json, '$.CRA.success') AS BOOL)),
  ada_reasoning = COALESCE(ada_reasoning, JSON_VALUE(raw_json, '$.ADA.reasoning')),
  taa_reasoning = COALESCE(taa_reasoning, JSON_VALUE(raw_json, '$.classification_reasoning')),
  cra_reasoning = COALESCE(cra_reasoning, JSON_VALUE(raw_json, '$.CRA.reasoning')),
  variable_of_importance = COALESCE(variable_of_importance, JSON_VALUE(raw_json, '$.variable_of_importance')),
  ada_detected = COALESCE(ada_detected, JSON_VALUE(raw_json, '$.ADA.detected'))
WHERE event_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
  AND raw_json IS NOT NULL;

-- 3) Create a scheduled query to keep columns updated (optional)
-- This can be run daily to sync JSON data to columns
CREATE OR REPLACE PROCEDURE `chronicle-dev-2be9.gatra_database.sync_json_to_columns`()
BEGIN
  UPDATE `chronicle-dev-2be9.gatra_database.taa_enhanced_results`
  SET 
    alarm_id = COALESCE(alarm_id, JSON_VALUE(raw_json, '$.alarm_id')),
    ada_case_class = COALESCE(ada_case_class, JSON_VALUE(raw_json, '$.enhanced_classification')),
    cra_action_type = COALESCE(cra_action_type, JSON_VALUE(raw_json, '$.CRA.action_type')),
    ada_confidence = COALESCE(ada_confidence, SAFE_CAST(JSON_VALUE(raw_json, '$.confidence') AS FLOAT64)),
    taa_confidence = COALESCE(taa_confidence, SAFE_CAST(JSON_VALUE(raw_json, '$.enhanced_confidence') AS FLOAT64)),
    ada_score = COALESCE(ada_score, SAFE_CAST(JSON_VALUE(raw_json, '$.ADA.score') AS FLOAT64)),
    taa_severity = COALESCE(taa_severity, SAFE_CAST(JSON_VALUE(raw_json, '$.severity_score') AS FLOAT64)),
    ada_valid = COALESCE(ada_valid, SAFE_CAST(JSON_VALUE(raw_json, '$.ADA.valid') AS BOOL)),
    taa_valid = COALESCE(taa_valid, SAFE_CAST(JSON_VALUE(raw_json, '$.enhanced_is_anomaly') AS BOOL)),
    cra_success = COALESCE(cra_success, SAFE_CAST(JSON_VALUE(raw_json, '$.CRA.success') AS BOOL)),
    ada_reasoning = COALESCE(ada_reasoning, JSON_VALUE(raw_json, '$.ADA.reasoning')),
    taa_reasoning = COALESCE(taa_reasoning, JSON_VALUE(raw_json, '$.classification_reasoning')),
    cra_reasoning = COALESCE(cra_reasoning, JSON_VALUE(raw_json, '$.CRA.reasoning')),
    variable_of_importance = COALESCE(variable_of_importance, JSON_VALUE(raw_json, '$.variable_of_importance')),
    ada_detected = COALESCE(ada_detected, JSON_VALUE(raw_json, '$.ADA.detected'))
  WHERE event_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
    AND raw_json IS NOT NULL;
END;


