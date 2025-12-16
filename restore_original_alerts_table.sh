#!/bin/bash

# Restore Original Alerts Table Structure
# This script restores the original taa_enhanced_results table with enhanced data

echo "🔄 Restoring Original Alerts Table Structure"
echo "============================================="
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

# Step 1: Create the original table structure
echo "📋 Step 1: Creating original table structure..."
bq query --use_legacy_sql=false < restore_original_alerts_table.sql

if [ $? -eq 0 ]; then
    echo "✅ Original table structure created/updated successfully"
else
    echo "❌ Error creating table structure"
    exit 1
fi

echo ""

# Step 2: Create the sync procedure
echo "📋 Step 2: Creating sync procedure..."
bq query --use_legacy_sql=false < sync_original_table.sql

if [ $? -eq 0 ]; then
    echo "✅ Sync procedure created successfully"
else
    echo "❌ Error creating sync procedure"
    exit 1
fi

echo ""

# Step 3: Test the restored table
echo "📋 Step 3: Testing restored table..."
bq query --use_legacy_sql=false --format=pretty "
SELECT 
  COUNT(*) as total_rows,
  COUNT(alarm_id) as alarm_id_count,
  COUNT(ada_case_class) as case_class_count,
  COUNT(taa_confidence) as confidence_count,
  MAX(event_time) as latest_record
FROM \`chronicle-dev-2be9.gatra_database.taa_enhanced_results\`
WHERE event_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR);"

if [ $? -eq 0 ]; then
    echo "✅ Table test completed successfully"
else
    echo "❌ Error testing table"
    exit 1
fi

echo ""

# Step 4: Show sample data
echo "📋 Step 4: Sample data from restored table..."
bq query --use_legacy_sql=false --format=pretty "
SELECT 
  alarm_id,
  ada_case_class,
  taa_confidence,
  enhanced_classification,
  taa_severity,
  calibrated,
  event_time
FROM \`chronicle-dev-2be9.gatra_database.taa_enhanced_results\`
WHERE event_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 2 HOUR)
ORDER BY event_time DESC
LIMIT 5;"

echo ""
echo "🎉 ORIGINAL ALERTS TABLE RESTORED!"
echo "=================================="
echo ""
echo "✅ The original taa_enhanced_results table has been restored"
echo "✅ Data is now available in the original structure your dashboard expects"
echo "✅ Enhanced data (calibrated confidence, improved classifications) is included"
echo ""
echo "📊 Your dashboard should now work with the original data structure!"
echo ""
echo "🔄 To keep the table updated, run the sync procedure periodically:"
echo "   CALL \`chronicle-dev-2be9.gatra_database.sync_to_original_table\`();"
echo ""
echo "⏰ You can set up a scheduled query to run this every hour for automatic updates."


