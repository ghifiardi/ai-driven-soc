#!/bin/bash

# BigQuery Diagnostics Runner
# This script helps you run the BigQuery diagnostics to fix the dashboard

echo "🔍 BigQuery Dashboard Diagnostics Runner"
echo "========================================"
echo ""

# Check if gcloud is configured
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n 1 > /dev/null; then
    echo "❌ Error: gcloud not authenticated. Please run:"
    echo "   gcloud auth login"
    echo "   gcloud config set project chronicle-dev-2be9"
    exit 1
fi

echo "✅ gcloud authenticated: $(gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n 1)"
echo "✅ Project: $(gcloud config get-value project)"
echo ""

# Create temporary files for queries
TEMP_DIR=$(mktemp -d)
echo "📁 Using temporary directory: $TEMP_DIR"
echo ""

# Query 1: Check top-level columns
echo "🔍 Running Query 1: Check if columns exist at top level..."
cat > "$TEMP_DIR/query1.sql" << 'EOF'
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
EOF

echo "Running: bq query --use_legacy_sql=false --format=pretty < $TEMP_DIR/query1.sql"
bq query --use_legacy_sql=false --format=pretty < "$TEMP_DIR/query1.sql"
echo ""

# Query 2: Check JSON data
echo "🔍 Running Query 2: Check if data exists inside JSON..."
cat > "$TEMP_DIR/query2.sql" << 'EOF'
SELECT
  COUNTIF(JSON_VALUE(raw_json, '$.alarm_id') IS NOT NULL) AS has_alarm_in_json,
  COUNTIF(JSON_VALUE(raw_json, '$.alarmId') IS NOT NULL) AS has_alarmId_in_json,
  COUNTIF(JSON_VALUE(raw_json, '$.enhanced_classification') IS NOT NULL) AS has_classification_in_json,
  COUNTIF(JSON_VALUE(raw_json, '$.enhanced_confidence') IS NOT NULL) AS has_confidence_in_json,
  COUNTIF(JSON_VALUE(raw_json, '$.severity_score') IS NOT NULL) AS has_severity_in_json,
  COUNT(*) as total_rows
FROM `chronicle-dev-2be9.gatra_database.taa_enhanced_results`
WHERE event_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR);
EOF

echo "Running: bq query --use_legacy_sql=false --format=pretty < $TEMP_DIR/query2.sql"
bq query --use_legacy_sql=false --format=pretty < "$TEMP_DIR/query2.sql"
echo ""

# Query 3: Sample JSON structure
echo "🔍 Running Query 3: Sample the actual JSON structure..."
cat > "$TEMP_DIR/query3.sql" << 'EOF'
SELECT
  raw_json,
  event_time,
  request_id,
  processed_by
FROM `chronicle-dev-2be9.gatra_database.taa_enhanced_results`
WHERE event_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
ORDER BY event_time DESC
LIMIT 3;
EOF

echo "Running: bq query --use_legacy_sql=false --format=pretty < $TEMP_DIR/query3.sql"
bq query --use_legacy_sql=false --format=pretty < "$TEMP_DIR/query3.sql"
echo ""

# Query 4: Test robust extraction
echo "🔍 Running Query 4: Test robust field extraction..."
cat > "$TEMP_DIR/query4.sql" << 'EOF'
SELECT
  event_time,
  request_id,
  
  -- Alarm ID (robust extraction)
  COALESCE(
    alarm_id,
    JSON_VALUE(raw_json, '$.alarm_id'),
    JSON_VALUE(raw_json, '$.alarmId'),
    JSON_VALUE(raw_json, '$.alarmID')
  ) AS alarm_id_extracted,
  
  -- Enhanced Classification
  COALESCE(
    ada_case_class,
    JSON_VALUE(raw_json, '$.enhanced_classification'),
    JSON_VALUE(raw_json, '$.ADA.case_class'),
    JSON_VALUE(raw_json, '$.case_class')
  ) AS ada_case_class_extracted,
  
  -- Enhanced Confidence
  COALESCE(
    taa_confidence,
    SAFE_CAST(JSON_VALUE(raw_json, '$.enhanced_confidence') AS FLOAT64),
    SAFE_CAST(JSON_VALUE(raw_json, '$.calibrated_confidence') AS FLOAT64),
    confidence
  ) AS taa_confidence_extracted,
  
  -- Enhanced Severity
  COALESCE(
    taa_severity,
    SAFE_CAST(JSON_VALUE(raw_json, '$.severity_score') AS FLOAT64),
    SAFE_CAST(JSON_VALUE(raw_json, '$.enhanced_severity') AS FLOAT64),
    severity
  ) AS taa_severity_extracted,
  
  processed_by,
  enhanced_classification,
  calibrated

FROM `chronicle-dev-2be9.gatra_database.taa_enhanced_results`
WHERE event_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
ORDER BY event_time DESC
LIMIT 10;
EOF

echo "Running: bq query --use_legacy_sql=false --format=pretty < $TEMP_DIR/query4.sql"
bq query --use_legacy_sql=false --format=pretty < "$TEMP_DIR/query4.sql"
echo ""

# Cleanup
rm -rf "$TEMP_DIR"

echo "✅ BigQuery diagnostics completed!"
echo ""
echo "📊 NEXT STEPS:"
echo "1. Review the results above"
echo "2. If JSON data exists, update your dashboard view with robust extraction"
echo "3. If columns are missing, add them to the schema"
echo "4. Test the dashboard to ensure it populates correctly"
echo ""
echo "🔧 If you need help with the next steps, let me know!"
