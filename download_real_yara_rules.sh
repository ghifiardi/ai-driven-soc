#!/bin/bash

# Download Real YARA Rules from Public Sources
# Integrates real malware detection rules into the threat hunting platform

set -e

RULES_DIR="$HOME/threat-hunting-test/data/yara_rules/real_rules"
TEMP_DIR="/tmp/yara_download_$$"

echo "=========================================="
echo "Downloading Real YARA Rules"
echo "=========================================="
echo ""

# Create directories
mkdir -p "$RULES_DIR"
mkdir -p "$TEMP_DIR"
cd "$TEMP_DIR"

# Check if git is available
if command -v git &> /dev/null; then
    echo "Step 1: Downloading YARA-Rules repository (using git)..."
    if [ -d "rules" ]; then
        cd rules
        git pull
    else
        git clone --depth 1 https://github.com/Yara-Rules/rules.git
        cd rules
    fi
    echo "✓ Repository downloaded"
else
    echo "Step 1: Downloading YARA-Rules (using curl)..."
    echo "  Note: Git not available, downloading archive..."
    
    # Download as zip/tarball
    curl -L -o rules.zip https://github.com/Yara-Rules/rules/archive/refs/heads/master.zip 2>/dev/null || \
    curl -L https://api.github.com/repos/Yara-Rules/rules/tarball/master | tar -xz
    
    if [ -f "rules.zip" ]; then
        unzip -q rules.zip 2>/dev/null || {
            echo "⚠ unzip not available, trying alternative method..."
            # Alternative: download specific rule files
            mkdir -p rules
            cd rules
        }
    else
        mkdir -p rules
        cd rules
    fi
    
    echo "✓ Archive downloaded"
fi

echo ""
echo "Step 2: Organizing rules by category..."

# Find the rules directory
if [ -d "rules" ]; then
    RULES_SOURCE="rules"
elif [ -d "Yara-Rules-rules-"* ]; then
    RULES_SOURCE=$(ls -d Yara-Rules-rules-* | head -1)
else
    RULES_SOURCE="."
fi

# Create category directories
mkdir -p "$RULES_DIR"/{ransomware,apt,trojan,backdoor,webshell,cryptominer,other}

# Function to copy rules by pattern
copy_rules() {
    local category=$1
    shift
    local patterns=("$@")
    
    for pattern in "${patterns[@]}"; do
        find "$RULES_SOURCE" -type f \( -name "*${pattern}*" -o -name "*${pattern^}*" -o -name "*${pattern^^}*" \) \
            -name "*.yar" -o -name "*.yara" 2>/dev/null | \
            head -50 | while read -r file; do
                if [ -f "$file" ]; then
                    cp "$file" "$RULES_DIR/$category/" 2>/dev/null || true
                fi
            done
    done
}

# Copy ransomware rules
echo "  Copying ransomware rules..."
copy_rules ransomware "ransomware" "lockbit" "wannacry" "ryuk" "revil" "conti" "maze"

# Copy APT rules
echo "  Copying APT rules..."
copy_rules apt "apt" "apt29" "apt28" "lazarus" "fancybear" "cozybear"

# Copy trojan rules
echo "  Copying trojan rules..."
copy_rules trojan "trojan" "zeus" "emotet" "trickbot"

# Copy backdoor rules
echo "  Copying backdoor rules..."
copy_rules backdoor "backdoor" "cobalt" "metasploit"

# Copy webshell rules
echo "  Copying webshell rules..."
copy_rules webshell "webshell" "shell" "php" "jsp"

# Copy cryptominer rules
echo "  Copying cryptominer rules..."
copy_rules cryptominer "miner" "coinminer" "cryptocurrency"

# Copy any remaining rules to 'other'
echo "  Copying other rules..."
find "$RULES_SOURCE" -type f -name "*.yar" -o -name "*.yara" 2>/dev/null | \
    head -100 | while read -r file; do
        if [ -f "$file" ]; then
            # Check if not already copied
            filename=$(basename "$file")
            if ! find "$RULES_DIR" -name "$filename" | grep -q .; then
                cp "$file" "$RULES_DIR/other/" 2>/dev/null || true
            fi
        fi
    done

echo ""
echo "Step 3: Verifying and counting rules..."
TOTAL=0
for category in ransomware apt trojan backdoor webshell cryptominer other; do
    count=$(find "$RULES_DIR/$category" -type f \( -name "*.yar" -o -name "*.yara" \) 2>/dev/null | wc -l)
    if [ "$count" -gt 0 ]; then
        echo "  ✓ $category: $count rules"
        TOTAL=$((TOTAL + count))
    fi
done

echo ""
echo "Total rules downloaded: $TOTAL"

# Cleanup
cd ~
rm -rf "$TEMP_DIR"

echo ""
echo "=========================================="
echo "Download Complete!"
echo "=========================================="
echo ""
echo "Rules location: $RULES_DIR"
echo ""
echo "Next steps:"
echo "1. Validate rules: yara -w $RULES_DIR/ransomware/*.yar /dev/null"
echo "2. Update config: See integrate_real_yara_rules.md"
echo "3. Test THOR with real rules"
echo ""

