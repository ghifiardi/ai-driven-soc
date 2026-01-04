#!/bin/bash
#
# Automated Demo Recording Script
# Records the threat hunting simulation and converts to GIF
#
# Usage: bash record_demo.sh
#

set -e

# Configuration
RECORDINGS_DIR="demo/recordings"
DEMO_NAME="threat_hunting_demo"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CAST_FILE="${RECORDINGS_DIR}/${DEMO_NAME}_${TIMESTAMP}.cast"
GIF_FILE="${RECORDINGS_DIR}/${DEMO_NAME}_${TIMESTAMP}.gif"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo ""
echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                                                            ║${NC}"
echo -e "${CYAN}║        Threat Hunting Demo - Recording Script             ║${NC}"
echo -e "${CYAN}║                                                            ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check prerequisites
echo -e "${BLUE}🔍 Checking prerequisites...${NC}"

if ! command -v asciinema &> /dev/null; then
    echo -e "${RED}❌ asciinema is not installed${NC}"
    echo -e "${YELLOW}   Install with: brew install asciinema${NC}"
    exit 1
fi
echo -e "${GREEN}   ✅ asciinema found${NC}"

if ! command -v agg &> /dev/null; then
    echo -e "${YELLOW}   ⚠️  agg not found (optional, for GIF conversion)${NC}"
    echo -e "${YELLOW}   Install with: brew install agg${NC}"
    AGG_AVAILABLE=false
else
    echo -e "${GREEN}   ✅ agg found${NC}"
    AGG_AVAILABLE=true
fi

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ python3 is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}   ✅ python3 found${NC}"

if [ ! -f "demo_simulation.py" ]; then
    echo -e "${RED}❌ demo_simulation.py not found${NC}"
    echo -e "${YELLOW}   Make sure you're in the ai-driven-soc.backup directory${NC}"
    exit 1
fi
echo -e "${GREEN}   ✅ demo_simulation.py found${NC}"

echo ""

# Create directories
mkdir -p "$RECORDINGS_DIR"
echo -e "${GREEN}✅ Created recordings directory: ${RECORDINGS_DIR}${NC}"
echo ""

# Display instructions
echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}📋 Recording Instructions:${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo "   The simulation will run with these prompts:"
echo "   1. Press ENTER to start"
echo "   2. Press ENTER after VALHALLA phase"
echo "   3. Press ENTER after ASGARD phase"
echo "   4. Press ENTER after THOR phase"
echo "   5. Press ENTER after TAA phase"
echo "   6. Press ENTER after CRA phase"
echo "   7. Press ENTER after CLA phase"
echo "   8. Press ENTER after Summary"
echo ""
echo -e "${YELLOW}   Total: 8 prompts - just press ENTER at each${NC}"
echo ""
echo "   Recording will be saved to:"
echo "   📁 $CAST_FILE"
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
echo ""

read -p "Press ENTER to start recording (or Ctrl+C to cancel)..."

echo ""
echo -e "${GREEN}🎬 Starting recording...${NC}"
echo ""

# Start recording with asciinema
asciinema rec "$CAST_FILE" -c "python3 demo_simulation.py"

echo ""
echo -e "${GREEN}✅ Recording complete!${NC}"
echo ""

# Show recording info
if [ -f "$CAST_FILE" ]; then
    CAST_SIZE=$(du -h "$CAST_FILE" | cut -f1)
    echo -e "${BLUE}📊 Recording Details:${NC}"
    echo "   File: $CAST_FILE"
    echo "   Size: $CAST_SIZE"
    echo ""
fi

# Ask if user wants to preview
echo -e "${CYAN}─────────────────────────────────────────────────────────${NC}"
read -p "Do you want to preview the recording? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}▶️  Playing recording...${NC}"
    echo ""
    asciinema play "$CAST_FILE"
fi

echo ""
echo -e "${CYAN}─────────────────────────────────────────────────────────${NC}"

# Convert to GIF if agg is available
if [ "$AGG_AVAILABLE" = true ]; then
    read -p "Do you want to convert to GIF? (y/n) " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo -e "${BLUE}🎨 Converting to GIF...${NC}"
        echo ""

        # Show conversion options
        echo "Select GIF quality:"
        echo "  1) Standard (120x40, 1.5x speed, ~5-15 MB) - Recommended"
        echo "  2) Compact (100x30, 2.0x speed, ~3-8 MB) - For README"
        echo "  3) High Quality (120x40, 1.0x speed, ~15-30 MB) - For presentations"
        echo ""
        read -p "Choice [1-3]: " -n 1 -r
        echo ""

        case $REPLY in
            1)
                echo "Creating standard GIF..."
                agg \
                  --cols 120 \
                  --rows 40 \
                  --speed 1.5 \
                  --font-size 14 \
                  --theme monokai \
                  "$CAST_FILE" \
                  "$GIF_FILE"
                ;;
            2)
                echo "Creating compact GIF..."
                agg \
                  --cols 100 \
                  --rows 30 \
                  --speed 2.0 \
                  --font-size 12 \
                  --theme monokai \
                  "$CAST_FILE" \
                  "$GIF_FILE"
                ;;
            3)
                echo "Creating high quality GIF..."
                agg \
                  --cols 120 \
                  --rows 40 \
                  --speed 1.0 \
                  --font-size 16 \
                  --theme monokai \
                  "$CAST_FILE" \
                  "$GIF_FILE"
                ;;
            *)
                echo "Invalid choice, using standard settings..."
                agg \
                  --cols 120 \
                  --rows 40 \
                  --speed 1.5 \
                  --font-size 14 \
                  --theme monokai \
                  "$CAST_FILE" \
                  "$GIF_FILE"
                ;;
        esac

        echo ""
        if [ -f "$GIF_FILE" ]; then
            GIF_SIZE=$(du -h "$GIF_FILE" | cut -f1)
            echo -e "${GREEN}✅ GIF created successfully!${NC}"
            echo "   File: $GIF_FILE"
            echo "   Size: $GIF_SIZE"
            echo ""

            # Check if file is too large
            GIF_SIZE_MB=$(du -m "$GIF_FILE" | cut -f1)
            if [ "$GIF_SIZE_MB" -gt 20 ]; then
                echo -e "${YELLOW}⚠️  Warning: GIF is larger than 20 MB${NC}"
                echo "   Consider using a more compact setting or optimizing with gifsicle"

                if command -v gifsicle &> /dev/null; then
                    read -p "   Optimize with gifsicle? (y/n) " -n 1 -r
                    echo ""
                    if [[ $REPLY =~ ^[Yy]$ ]]; then
                        OPTIMIZED_FILE="${GIF_FILE%.gif}_optimized.gif"
                        echo "   Optimizing..."
                        gifsicle --optimize=3 --colors 256 "$GIF_FILE" -o "$OPTIMIZED_FILE"
                        OPTIMIZED_SIZE=$(du -h "$OPTIMIZED_FILE" | cut -f1)
                        echo -e "${GREEN}   ✅ Optimized GIF created${NC}"
                        echo "      File: $OPTIMIZED_FILE"
                        echo "      Size: $OPTIMIZED_SIZE"
                    fi
                fi
            fi
        else
            echo -e "${RED}❌ GIF conversion failed${NC}"
        fi
    fi
else
    echo ""
    echo -e "${YELLOW}ℹ️  Install 'agg' to convert to GIF:${NC}"
    echo "   brew install agg"
    echo ""
    echo "   Then run:"
    echo "   agg --speed 1.5 --theme monokai $CAST_FILE ${GIF_FILE}"
fi

echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}🎉 All done!${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BLUE}📁 Files created:${NC}"
echo "   .cast file: $CAST_FILE"
if [ -f "$GIF_FILE" ]; then
    echo "   .gif file:  $GIF_FILE"
fi
echo ""

echo -e "${BLUE}📤 Next steps:${NC}"
echo "   • Add to README.md:"
echo "     ![Threat Hunting Demo]($GIF_FILE)"
echo ""
echo "   • Upload to asciinema.org:"
echo "     asciinema upload $CAST_FILE"
echo ""
echo "   • Open GIF in browser:"
if [ -f "$GIF_FILE" ]; then
    echo "     open $GIF_FILE"
fi
echo ""
