#!/bin/bash

# Validate YARA Rules
# Checks syntax and compiles rules to ensure they're valid

set -e

RULES_DIR="$HOME/threat-hunting-test/data/yara_rules/real_rules"

echo "=========================================="
echo "Validating YARA Rules"
echo "=========================================="
echo ""

# Check if yara command is available
if ! command -v yara &> /dev/null; then
    echo "⚠ yara command not found"
    echo "  Install with: sudo yum install -y yara"
    echo ""
    echo "  Or validate using Python:"
    python3 << 'PYVALIDATE'
import yara
import os
import sys

rules_dir = os.path.expanduser('~/threat-hunting-test/data/yara_rules/real_rules')
errors = []
valid = 0

for root, dirs, files in os.walk(rules_dir):
    for file in files:
        if file.endswith(('.yar', '.yara')):
            filepath = os.path.join(root, file)
            try:
                yara.compile(filepath)
                valid += 1
            except yara.SyntaxError as e:
                errors.append(f"{filepath}: {e}")

print(f"✓ Valid rules: {valid}")
if errors:
    print(f"✗ Errors: {len(errors)}")
    for error in errors[:10]:
        print(f"  {error}")
else:
    print("✓ All rules are valid!")
PYVALIDATE
    exit 0
fi

echo "Validating rules in: $RULES_DIR"
echo ""

VALID=0
INVALID=0
ERRORS=()

# Validate each rule file
find "$RULES_DIR" -type f \( -name "*.yar" -o -name "*.yara" \) | while read -r rulefile; do
    if yara -w "$rulefile" /dev/null 2>&1 | grep -q "error\|warning"; then
        INVALID=$((INVALID + 1))
        ERRORS+=("$rulefile")
        echo "✗ $(basename $rulefile)"
    else
        VALID=$((VALID + 1))
        if [ $((VALID % 10)) -eq 0 ]; then
            echo -n "."
        fi
    fi
done

echo ""
echo ""
echo "Validation Summary:"
echo "  Valid rules: $VALID"
echo "  Invalid rules: $INVALID"

if [ ${#ERRORS[@]} -gt 0 ]; then
    echo ""
    echo "Rules with errors:"
    for error in "${ERRORS[@]:0:10}"; do
        echo "  - $error"
    done
fi

echo ""
echo "=========================================="
echo "Validation Complete"
echo "=========================================="

