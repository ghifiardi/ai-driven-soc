#!/bin/bash

# Setup BigQuery Integration for SOC Dashboards
# This script helps configure BigQuery connection

set -e

echo "=========================================="
echo "BigQuery Integration Setup"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "bigquery_config.py" ]; then
    echo "⚠ Error: bigquery_config.py not found"
    echo "  Please run this script from the dashboards directory"
    exit 1
fi

echo "Step 1: Checking GCP authentication..."
echo ""

# Check for service account key
if [ -f "../chronicle-key.json" ]; then
    echo "✓ Found service account key: ../chronicle-key.json"
    export GOOGLE_APPLICATION_CREDENTIALS="../chronicle-key.json"
    echo "  Set GOOGLE_APPLICATION_CREDENTIALS=$GOOGLE_APPLICATION_CREDENTIALS"
elif [ -f "chronicle-key.json" ]; then
    echo "✓ Found service account key: chronicle-key.json"
    export GOOGLE_APPLICATION_CREDENTIALS="chronicle-key.json"
    echo "  Set GOOGLE_APPLICATION_CREDENTIALS=$GOOGLE_APPLICATION_CREDENTIALS"
elif [ -n "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "✓ GOOGLE_APPLICATION_CREDENTIALS already set: $GOOGLE_APPLICATION_CREDENTIALS"
else
    echo "⚠ No service account key found"
    echo "  Using Application Default Credentials (ADC)"
    echo "  Make sure you've run: gcloud auth application-default login"
fi

echo ""
echo "Step 2: Testing BigQuery connection..."
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Test connection
python3 << 'PYTEST'
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from bigquery_config import test_bigquery_connection, get_available_tables
    
    print("Testing BigQuery connection...")
    result = test_bigquery_connection()
    
    if result['success']:
        print(f"✓ Connection successful!")
        print(f"  Project: {result['project_id']}")
        print(f"  Dataset: {result['dataset']}")
        print(f"  Tables found: {result['tables_found']}")
        print("")
        
        print("Available tables:")
        tables = get_available_tables()
        if tables:
            for table in tables:
                print(f"  ✓ {table}")
        else:
            print("  ⚠ No tables found or error accessing tables")
    else:
        print(f"✗ Connection failed: {result.get('error', 'Unknown error')}")
        print("")
        print("Troubleshooting:")
        print("  1. Check GCP credentials")
        print("  2. Verify project ID: chronicle-dev-2be9")
        print("  3. Ensure BigQuery API is enabled")
        print("  4. Check dataset name: gatra_database")
        sys.exit(1)
        
except ImportError as e:
    print(f"✗ Error importing bigquery_config: {e}")
    print("  Make sure bigquery_config.py exists")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

PYTEST

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ BigQuery Integration Ready!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "1. Run dashboard: streamlit run enhanced_soc_dashboard.py"
    echo "2. Dashboard will automatically connect to BigQuery"
    echo "3. If connection fails, dashboard will use demo data"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "✗ Setup Failed"
    echo "=========================================="
    echo ""
    echo "Please check:"
    echo "1. GCP credentials are configured"
    echo "2. BigQuery API is enabled"
    echo "3. Service account has BigQuery permissions"
    echo ""
    exit 1
fi

