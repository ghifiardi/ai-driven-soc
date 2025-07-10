#!/bin/bash
# permission_test.sh

VM_BASE_DIR="/home/raditio.ghifiardigmail.com/ai-driven-soc"
LOG_FILE="$VM_BASE_DIR/permission_test.log"
TEST_FILE="$VM_BASE_DIR/permission_test.success"

# Redirect all output to a log file to capture everything
exec &> "$LOG_FILE"

echo "--- Starting permission_test.sh at $(date) ---"
echo "Attempting to create file: $TEST_FILE"

# The actual test command
touch "$TEST_FILE"

if [ $? -eq 0 ]; then
  echo "SUCCESS: Test file was created."
  exit 0
else
  echo "ERROR: Failed to create test file. Check directory permissions." >&2
  exit 1
fi
