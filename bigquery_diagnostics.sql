-- BigQuery Diagnostics for Dashboard Column Issues
-- Run these queries to diagnose why Alerts Table columns are empty

-- 1) Check if columns exist at top level
SELECT
  ANY_VALUE(alarm_id) IS NOT NULL AS has_alarm_id_top,
  ANY_VALUE(ada_case_class) IS NOT NULL AS has_case_top,
  ANY_VALUE(cra_action_type) IS NOT NULL AS has_action_top,
  COUNT(*) as total_rows,
  COUNT(alarm_id) as alarm_id_count,
  COUNT(ada_case_class) as case_class_count,
  COUNT(cra_action_type) as action_type_count
FROM `chronicle-dev-2be9.gatra_database.taa_enhanced_results`
WHERE event_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR);

-- 2) Check if data exists inside JSON
SELECT
  COUNTIF(JSON_VALUE(raw_json, '$.alarm_id') IS NOT NULL) AS has_alarm_in_json,
  COUNTIF(JSON_VALUE(raw_json, '$.alarmId') IS NOT NULL) AS has_alarmId_in_json,
  COUNTIF(JSON_VALUE(raw_json, '$.alarmID') IS NOT NULL) AS has_alarmID_in_json,
  COUNTIF(JSON_VALUE(raw_json, '$.ADA.case_class') IS NOT NULL) AS has_case_in_json,
  COUNTIF(JSON_VALUE(raw_json, '$.ada.case_class') IS NOT NULL) AS has_case_lower_in_json,
  COUNTIF(JSON_VALUE(raw_json, '$.CRA.action_type') IS NOT NULL) AS has_action_in_json,
  COUNTIF(JSON_VALUE(raw_json, '$.cra.action_type') IS NOT NULL) AS has_action_lower_in_json,
  COUNT(*) as total_rows
FROM `chronicle-dev-2be9.gatra_database.taa_enhanced_results`
WHERE event_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR);

-- 3) Sample the actual JSON structure
SELECT
  raw_json,
  event_time,
  request_id
FROM `chronicle-dev-2be9.gatra_database.taa_enhanced_results`
WHERE event_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
ORDER BY event_time DESC
LIMIT 5;

-- 4) Check table schema
SELECT
  column_name,
  data_type,
  is_nullable
FROM `chronicle-dev-2be9.gatra_database.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name = 'taa_enhanced_results'
ORDER BY ordinal_position;

-- 5) Check if there's an alerts table view
SELECT
  table_schema,
  table_name,
  view_definition
FROM `chronicle-dev-2be9.INFORMATION_SCHEMA.VIEWS`
WHERE table_name LIKE '%alert%' OR table_name LIKE '%taa%'
ORDER BY table_name;

-- 6) Recent data sample with robust extraction
SELECT
  event_time,
  request_id,
  -- Alarm ID from multiple locations
  COALESCE(
    alarm_id,
    JSON_VALUE(raw_json, '$.alarm_id'),
    JSON_VALUE(raw_json, '$.alarmId'),
    JSON_VALUE(raw_json, '$.alarmID')
  ) AS alarm_id_extracted,
  
  -- ADA Case Class
  COALESCE(
    ada_case_class,
    JSON_VALUE(raw_json, '$.ADA.case_class'),
    JSON_VALUE(raw_json, '$.ada.case_class'),
    JSON_VALUE(raw_json, '$.case_class')
  ) AS ada_case_class_extracted,
  
  -- CRA Action Type
  COALESCE(
    cra_action_type,
    JSON_VALUE(raw_json, '$.CRA.action_type'),
    JSON_VALUE(raw_json, '$.cra.action_type'),
    JSON_VALUE(raw_json, '$.action_type')
  ) AS cra_action_type_extracted,
  
  -- Other fields
  confidence,
  severity,
  rule,
  source,
  processed_by
FROM `chronicle-dev-2be9.gatra_database.taa_enhanced_results`
WHERE event_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
ORDER BY event_time DESC
LIMIT 20;


