#!/bin/bash
# fetch_data.sh - Self-logging diagnostic version

# --- Configuration ---
VM_BASE_DIR="/home/raditio.ghifiardigmail.com/ai-driven-soc"
LOG_FILE="$VM_BASE_DIR/fetch_data.log"
OUTPUT_FILE="$VM_BASE_DIR/results.json"
BQ_COMMAND="/usr/bin/bq"  # Corrected path for Linux VM
QUERY="$1"

# --- Self-Logging Setup ---
# Redirect all output (stdout and stderr) to the log file
exec &> "$LOG_FILE"

echo "--- Starting fetch_data.sh at $(date) ---"

echo "Query to execute: $QUERY"

# --- Execution ---
echo "Executing bq command..."
$BQ_COMMAND query \
    --location asia-southeast2 \
    --project_id chronicle-dev-2be9 \
    --format=json \
    --use_legacy_sql=false \
    "$QUERY" > "$OUTPUT_FILE"

EXIT_CODE=$?
echo "'bq' command finished with exit code: $EXIT_CODE"

# --- Validation ---
if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: 'bq' command failed. Deleting potentially incomplete output file."
    rm -f "$OUTPUT_FILE"
    echo "--- Script finished with an error. ---"
    exit 1
fi

if [ ! -s "$OUTPUT_FILE" ]; then
    echo "ERROR: 'bq' command succeeded but the output file is empty. This can happen with queries that return no rows."
    rm -f "$OUTPUT_FILE"
    echo "--- Script finished with an error (empty result). ---"
    exit 1
fi

echo "SUCCESS: Data successfully fetched to $OUTPUT_FILE"
echo "--- Script finished successfully. ---"
exit 0
