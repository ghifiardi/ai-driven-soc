#!/bin/bash

# Fixed script to properly download and organize YARA rules
# This version explores the repository structure first

set -e

RULES_DIR="$HOME/threat-hunting-test/data/yara_rules/real_rules"
REPO_DIR="$HOME/threat-hunting-test/data/yara_rules/rules"

echo "=========================================="
echo "Fixing YARA Rules Download"
echo "=========================================="
echo ""

# Check if repository exists
if [ ! -d "$REPO_DIR" ]; then
    echo "⚠ Repository not found at $REPO_DIR"
    echo "  Run the download script first"
    exit 1
fi

echo "Step 1: Exploring repository structure..."
cd "$REPO_DIR"

# Show directory structure
echo "Repository contents:"
ls -la | head -20
echo ""

# Find all YARA files
echo "Step 2: Finding all YARA rule files..."
YARA_FILES=$(find . -type f \( -name "*.yar" -o -name "*.yara" \) 2>/dev/null)
TOTAL=$(echo "$YARA_FILES" | grep -v "^$" | wc -l)
echo "  Found $TOTAL YARA rule files"
echo ""

# Create category directories
mkdir -p "$RULES_DIR"/{ransomware,apt,trojan,backdoor,webshell,cryptominer,other}

echo "Step 3: Organizing rules by category..."

# Copy ransomware rules
echo "  Copying ransomware rules..."
echo "$YARA_FILES" | grep -iE "(ransomware|lockbit|wannacry|ryuk|revil|conti|maze|sodinokibi|phobos|gandcrab)" | \
    head -30 | while read -r file; do
        if [ -f "$file" ]; then
            cp "$file" "$RULES_DIR/ransomware/" 2>/dev/null && echo "    ✓ $(basename $file)"
        fi
    done

# Copy APT rules
echo "  Copying APT rules..."
echo "$YARA_FILES" | grep -iE "(apt|apt29|apt28|lazarus|fancybear|cozybear|apt1|apt10|apt12|apt17|apt19|apt32|apt33|apt34|apt35|apt37|apt38|apt39|apt40|apt41)" | \
    head -30 | while read -r file; do
        if [ -f "$file" ]; then
            cp "$file" "$RULES_DIR/apt/" 2>/dev/null && echo "    ✓ $(basename $file)"
        fi
    done

# Copy trojan rules
echo "  Copying trojan rules..."
echo "$YARA_FILES" | grep -iE "(trojan|zeus|emotet|trickbot|banker|stealer)" | \
    head -30 | while read -r file; do
        if [ -f "$file" ]; then
            cp "$file" "$RULES_DIR/trojan/" 2>/dev/null && echo "    ✓ $(basename $file)"
        fi
    done

# Copy backdoor rules
echo "  Copying backdoor rules..."
echo "$YARA_FILES" | grep -iE "(backdoor|cobalt|metasploit|shell)" | \
    head -30 | while read -r file; do
        if [ -f "$file" ]; then
            cp "$file" "$RULES_DIR/backdoor/" 2>/dev/null && echo "    ✓ $(basename $file)"
        fi
    done

# Copy webshell rules
echo "  Copying webshell rules..."
echo "$YARA_FILES" | grep -iE "(webshell|php.*shell|jsp.*shell|asp.*shell)" | \
    head -30 | while read -r file; do
        if [ -f "$file" ]; then
            cp "$file" "$RULES_DIR/webshell/" 2>/dev/null && echo "    ✓ $(basename $file)"
        fi
    done

# Copy cryptominer rules
echo "  Copying cryptominer rules..."
echo "$YARA_FILES" | grep -iE "(miner|coinminer|cryptocurrency|monero|bitcoin.*miner)" | \
    head -30 | while read -r file; do
        if [ -f "$file" ]; then
            cp "$file" "$RULES_DIR/cryptominer/" 2>/dev/null && echo "    ✓ $(basename $file)"
        fi
    done

# Copy remaining rules to 'other'
echo "  Copying other rules..."
COPIED=$(find "$RULES_DIR" -type f -name "*.yar" -o -name "*.yara" 2>/dev/null | wc -l)
REMAINING=$((TOTAL - COPIED))
if [ "$REMAINING" -gt 0 ]; then
    echo "$YARA_FILES" | head -50 | while read -r file; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            # Only copy if not already copied
            if ! find "$RULES_DIR" -name "$filename" | grep -q .; then
                cp "$file" "$RULES_DIR/other/" 2>/dev/null || true
            fi
        fi
    done
fi

echo ""
echo "Step 4: Summary..."
TOTAL_COPIED=0
for category in ransomware apt trojan backdoor webshell cryptominer other; do
    count=$(find "$RULES_DIR/$category" -type f \( -name "*.yar" -o -name "*.yara" \) 2>/dev/null | wc -l)
    if [ "$count" -gt 0 ]; then
        echo "  ✓ $category: $count rules"
        TOTAL_COPIED=$((TOTAL_COPIED + count))
    fi
done

echo ""
echo "=========================================="
echo "Organization Complete!"
echo "=========================================="
echo "Total rules organized: $TOTAL_COPIED"
echo "Rules location: $RULES_DIR"
echo ""

