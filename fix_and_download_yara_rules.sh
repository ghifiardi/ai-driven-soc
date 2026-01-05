#!/bin/bash

# Complete fix: Find existing rules or re-download them properly

set -e

RULES_DIR="$HOME/threat-hunting-test/data/yara_rules/real_rules"
REPO_DIR="$HOME/threat-hunting-test/data/yara_rules/rules"

echo "=========================================="
echo "Finding/Downloading YARA Rules"
echo "=========================================="
echo ""

# Step 1: Check if rules exist anywhere
echo "Step 1: Checking for existing rules..."
if [ -d "$REPO_DIR" ] && [ -n "$(find "$REPO_DIR" -name "*.yar" -o -name "*.yara" 2>/dev/null | head -1)" ]; then
    echo "✓ Found existing repository at $REPO_DIR"
    RULES_SOURCE="$REPO_DIR"
elif [ -d "/tmp/yara_download" ] && [ -n "$(find /tmp/yara_download -name "*.yar" -o -name "*.yara" 2>/dev/null | head -1)" ]; then
    echo "✓ Found rules in /tmp/yara_download"
    RULES_SOURCE="/tmp/yara_download/rules"
else
    echo "⚠ No existing rules found, downloading..."
    
    # Create repository directory
    mkdir -p "$REPO_DIR"
    cd "$REPO_DIR"
    
    # Download using git
    if command -v git &> /dev/null; then
        echo "  Downloading with git..."
        git clone --depth 1 https://github.com/Yara-Rules/rules.git . 2>/dev/null || {
            # If clone fails, try into subdirectory
            rm -rf rules
            git clone --depth 1 https://github.com/Yara-Rules/rules.git
            RULES_SOURCE="$REPO_DIR/rules"
        }
        RULES_SOURCE="$REPO_DIR"
    else
        echo "  Git not available, using curl..."
        curl -L https://api.github.com/repos/Yara-Rules/rules/tarball/master | tar -xz --strip=1 || {
            curl -L https://api.github.com/repos/Yara-Rules/rules/tarball/master | tar -xz
            RULES_SOURCE=$(ls -d Yara-Rules-rules-* | head -1)
        }
        RULES_SOURCE="$REPO_DIR"
    fi
    
    echo "✓ Download complete"
fi

echo ""
echo "Step 2: Finding YARA files..."
cd "$RULES_SOURCE" 2>/dev/null || cd "$(dirname "$RULES_SOURCE")"

# Find all YARA files
YARA_FILES=$(find . -type f \( -name "*.yar" -o -name "*.yara" \) 2>/dev/null)
TOTAL=$(echo "$YARA_FILES" | grep -v "^$" | wc -l)

if [ "$TOTAL" -eq 0 ]; then
    echo "⚠ No YARA files found in $RULES_SOURCE"
    echo "  Checking subdirectories..."
    find . -type d | head -10
    exit 1
fi

echo "  Found $TOTAL YARA rule files"
echo ""

# Step 3: Organize rules
echo "Step 3: Organizing rules..."
mkdir -p "$RULES_DIR"/{ransomware,apt,trojan,backdoor,webshell,cryptominer,other}

# Function to copy rules
copy_rules() {
    local category=$1
    shift
    local patterns=("$@")
    local count=0
    
    for pattern in "${patterns[@]}"; do
        echo "$YARA_FILES" | grep -iE "$pattern" | while read -r file; do
            if [ -f "$file" ]; then
                cp "$file" "$RULES_DIR/$category/" 2>/dev/null && count=$((count + 1)) || true
            fi
        done
    done
}

# Copy by category
echo "  Copying ransomware rules..."
echo "$YARA_FILES" | grep -iE "(ransomware|lockbit|wannacry|ryuk|revil|conti|maze|sodinokibi|phobos|gandcrab)" | \
    while read -r file; do
        [ -f "$file" ] && cp "$file" "$RULES_DIR/ransomware/" 2>/dev/null || true
    done

echo "  Copying APT rules..."
echo "$YARA_FILES" | grep -iE "(apt|apt29|apt28|lazarus|fancybear|cozybear)" | \
    while read -r file; do
        [ -f "$file" ] && cp "$file" "$RULES_DIR/apt/" 2>/dev/null || true
    done

echo "  Copying trojan rules..."
echo "$YARA_FILES" | grep -iE "(trojan|zeus|emotet|trickbot|banker)" | \
    while read -r file; do
        [ -f "$file" ] && cp "$file" "$RULES_DIR/trojan/" 2>/dev/null || true
    done

echo "  Copying backdoor rules..."
echo "$YARA_FILES" | grep -iE "(backdoor|cobalt|metasploit)" | \
    while read -r file; do
        [ -f "$file" ] && cp "$file" "$RULES_DIR/backdoor/" 2>/dev/null || true
    done

echo "  Copying webshell rules..."
echo "$YARA_FILES" | grep -iE "(webshell|php.*shell|jsp.*shell)" | \
    while read -r file; do
        [ -f "$file" ] && cp "$file" "$RULES_DIR/webshell/" 2>/dev/null || true
    done

# Copy all remaining files to 'other' (simple approach)
echo "  Copying all remaining rules to 'other'..."
COPIED_COUNT=0
echo "$YARA_FILES" | head -200 | while read -r file; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        # Check if already copied
        if ! find "$RULES_DIR" -name "$filename" 2>/dev/null | grep -q .; then
            cp "$file" "$RULES_DIR/other/" 2>/dev/null && COPIED_COUNT=$((COPIED_COUNT + 1)) || true
        fi
    fi
done

# Summary
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
echo "Complete!"
echo "=========================================="
echo "Total rules organized: $TOTAL_COPIED"
echo "Rules location: $RULES_DIR"
echo ""

