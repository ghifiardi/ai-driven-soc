#!/bin/bash

# Setup GCP Resources for Threat Hunting Platform
# Creates Pub/Sub topics, BigQuery datasets/tables, and GCS buckets

set -e

PROJECT_ID="chronicle-dev-2be9"
DATASET="soc_data"

echo "=========================================="
echo "Setting up GCP Resources for Threat Hunting"
echo "=========================================="
echo ""
echo "Project ID: $PROJECT_ID"
echo ""

# Check if gcloud is available
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found. Please install it first."
    exit 1
fi

# Check if bq is available
if ! command -v bq &> /dev/null; then
    echo "⚠ bq CLI not found. BigQuery setup will be skipped."
    BQ_AVAILABLE=false
else
    BQ_AVAILABLE=true
fi

# Check if gsutil is available
if ! command -v gsutil &> /dev/null; then
    echo "⚠ gsutil not found. GCS setup will be skipped."
    GCS_AVAILABLE=false
else
    GCS_AVAILABLE=true
fi

echo "Step 1: Creating Pub/Sub Topics..."
echo ""

TOPICS=(
    "thor-scan-requests"
    "thor-findings"
    "asgard-campaigns"
    "asgard-scan-tasks"
    "asgard-scan-results"
    "thor-heartbeat"
    "valhalla-rule-updates"
    "valhalla-ioc-updates"
)

for topic in "${TOPICS[@]}"; do
    echo "  Creating topic: $topic"
    if gcloud pubsub topics create "$topic" --project="$PROJECT_ID" 2>&1 | grep -q "already exists"; then
        echo "    ⚠ Topic already exists"
    else
        echo "    ✓ Topic created"
    fi
done

echo ""
echo "Step 2: Creating Pub/Sub Subscriptions..."
echo ""

SUBSCRIPTIONS=(
    "thor-scan-requests-sub:thor-scan-requests"
    "thor-findings-sub:thor-findings"
    "asgard-campaigns-sub:asgard-campaigns"
    "asgard-scan-tasks-sub:asgard-scan-tasks"
)

for sub_info in "${SUBSCRIPTIONS[@]}"; do
    IFS=':' read -r sub_name topic_name <<< "$sub_info"
    echo "  Creating subscription: $sub_name"
    if gcloud pubsub subscriptions create "$sub_name" \
        --topic="$topic_name" \
        --project="$PROJECT_ID" 2>&1 | grep -q "already exists"; then
        echo "    ⚠ Subscription already exists"
    else
        echo "    ✓ Subscription created"
    fi
done

if [ "$BQ_AVAILABLE" = true ]; then
    echo ""
    echo "Step 3: Creating BigQuery Dataset..."
    echo ""
    
    echo "  Creating dataset: $DATASET"
    if bq mk --dataset "$PROJECT_ID:$DATASET" 2>&1 | grep -q "already exists"; then
        echo "    ⚠ Dataset already exists"
    else
        echo "    ✓ Dataset created"
    fi
    
    echo ""
    echo "Step 4: Creating BigQuery Tables..."
    echo ""
    
    # THOR scan results table
    echo "  Creating table: thor_scan_results"
    bq mk --table "$PROJECT_ID:$DATASET.thor_scan_results" \
        scan_id:STRING,hostname:STRING,start_time:TIMESTAMP,end_time:TIMESTAMP,threats_detected:INTEGER,files_scanned:INTEGER,processes_scanned:INTEGER,network_connections:INTEGER,matches:JSON 2>&1 | grep -v "already exists" || echo "    ⚠ Table may already exist"
    
    # ASGARD campaign reports table
    echo "  Creating table: asgard_campaign_reports"
    bq mk --table "$PROJECT_ID:$DATASET.asgard_campaign_reports" \
        campaign_id:STRING,campaign_name:STRING,created_at:TIMESTAMP,status:STRING,total_targets:INTEGER,successfully_scanned:INTEGER,total_threats:INTEGER,critical_threats:INTEGER 2>&1 | grep -v "already exists" || echo "    ⚠ Table may already exist"
    
    # VALHALLA feed stats table
    echo "  Creating table: valhalla_feed_stats"
    bq mk --table "$PROJECT_ID:$DATASET.valhalla_feed_stats" \
        feed_name:STRING,update_time:TIMESTAMP,iocs_count:INTEGER,yara_rules_count:INTEGER,sigma_rules_count:INTEGER 2>&1 | grep -v "already exists" || echo "    ⚠ Table may already exist"
else
    echo ""
    echo "Step 3-4: Skipping BigQuery setup (bq CLI not available)"
    echo ""
fi

if [ "$GCS_AVAILABLE" = true ]; then
    echo ""
    echo "Step 5: Creating GCS Buckets..."
    echo ""
    
    BUCKET="valhalla-threat-intel-$PROJECT_ID"
    echo "  Creating bucket: gs://$BUCKET"
    if gsutil mb -p "$PROJECT_ID" "gs://$BUCKET" 2>&1 | grep -q "already exists"; then
        echo "    ⚠ Bucket already exists"
    else
        echo "    ✓ Bucket created"
    fi
else
    echo ""
    echo "Step 5: Skipping GCS setup (gsutil not available)"
    echo ""
fi

echo ""
echo "=========================================="
echo "GCP Resources Setup Complete!"
echo "=========================================="
echo ""
echo "Created resources:"
echo "  ✓ Pub/Sub Topics: ${#TOPICS[@]} topics"
echo "  ✓ Pub/Sub Subscriptions: ${#SUBSCRIPTIONS[@]} subscriptions"
if [ "$BQ_AVAILABLE" = true ]; then
    echo "  ✓ BigQuery Dataset: $DATASET"
    echo "  ✓ BigQuery Tables: 3 tables"
fi
if [ "$GCS_AVAILABLE" = true ]; then
    echo "  ✓ GCS Bucket: valhalla-threat-intel"
fi
echo ""
echo "Next steps:"
echo "1. Update config files to use these resources"
echo "2. Test Pub/Sub message publishing"
echo "3. Verify BigQuery table writes"
echo ""

