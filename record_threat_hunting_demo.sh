#!/bin/bash
# Script to record the threat hunting demo using asciinema
# This creates a high-quality terminal recording that can be converted to GIF or video

set -e

# Configuration
DEMO_NAME="threat_hunting_demo"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${DEMO_NAME}_${TIMESTAMP}.cast"
GIF_FILE="${DEMO_NAME}_${TIMESTAMP}.gif"
INSTANCE_NAME="xdgaisocapp01"
ZONE="asia-southeast2-a"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}Threat Hunting Platform - Demo Recorder${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# Check if running on GCP instance or need to connect
if [ "$(hostname)" = "$INSTANCE_NAME" ]; then
    echo -e "${GREEN}✓ Running on GCP instance${NC}"
    LOCAL_MODE=true
else
    echo -e "${YELLOW}⚠ Not on GCP instance - will connect via SSH${NC}"
    LOCAL_MODE=false
fi

# Check for asciinema
if ! command -v asciinema &> /dev/null; then
    echo -e "${YELLOW}Installing asciinema...${NC}"
    if [ "$LOCAL_MODE" = true ]; then
        pip3 install asciinema
    else
        echo "Please install asciinema locally: pip install asciinema"
        echo "Or: brew install asciinema (on macOS)"
        exit 1
    fi
fi

# Check for agg (asciinema to GIF converter)
AGG_AVAILABLE=false
if command -v agg &> /dev/null; then
    AGG_AVAILABLE=true
    echo -e "${GREEN}✓ agg (GIF converter) available${NC}"
else
    echo -e "${YELLOW}⚠ agg not found - install with: cargo install --git https://github.com/asciinema/agg${NC}"
    echo -e "${YELLOW}  (GIF conversion will be skipped)${NC}"
fi

echo ""
echo -e "${CYAN}Recording options:${NC}"
echo "1. Record on local GCP instance"
echo "2. Record via SSH (from local machine)"
echo "3. Just run demo without recording"
echo ""
read -p "Choose option (1-3): " OPTION

case $OPTION in
    1)
        echo -e "${GREEN}Starting recording on local instance...${NC}"
        echo ""
        echo "Recording will start in 3 seconds..."
        sleep 3

        asciinema rec "$OUTPUT_FILE" \
            --title "AI-Driven SOC - Threat Hunting Platform Demo" \
            --command "bash demo_threat_hunting.sh" \
            --overwrite

        echo ""
        echo -e "${GREEN}✓ Recording saved to: $OUTPUT_FILE${NC}"

        # Convert to GIF if agg is available
        if [ "$AGG_AVAILABLE" = true ]; then
            echo -e "${CYAN}Converting to GIF...${NC}"
            agg "$OUTPUT_FILE" "$GIF_FILE" \
                --font-size 14 \
                --theme monokai \
                --speed 1.5 \
                --cols 120 \
                --rows 36
            echo -e "${GREEN}✓ GIF saved to: $GIF_FILE${NC}"
        fi
        ;;

    2)
        echo -e "${GREEN}Connecting to GCP instance via SSH...${NC}"
        echo ""

        # Upload demo script to GCP instance
        echo "Uploading demo script..."
        gcloud compute scp demo_threat_hunting.sh app@${INSTANCE_NAME}:~/threat-hunting-test/ \
            --zone=$ZONE

        # Record via SSH
        echo "Starting recording via SSH..."
        echo ""
        sleep 2

        asciinema rec "$OUTPUT_FILE" \
            --title "AI-Driven SOC - Threat Hunting Platform Demo" \
            --command "gcloud compute ssh app@${INSTANCE_NAME} --zone=${ZONE} --command='cd ~/threat-hunting-test && bash demo_threat_hunting.sh'" \
            --overwrite

        echo ""
        echo -e "${GREEN}✓ Recording saved to: $OUTPUT_FILE${NC}"

        # Convert to GIF if agg is available
        if [ "$AGG_AVAILABLE" = true ]; then
            echo -e "${CYAN}Converting to GIF...${NC}"
            agg "$OUTPUT_FILE" "$GIF_FILE" \
                --font-size 14 \
                --theme monokai \
                --speed 1.5 \
                --cols 120 \
                --rows 36
            echo -e "${GREEN}✓ GIF saved to: $GIF_FILE${NC}"
        fi
        ;;

    3)
        echo -e "${GREEN}Running demo without recording...${NC}"
        echo ""
        if [ "$LOCAL_MODE" = true ]; then
            bash demo_threat_hunting.sh
        else
            gcloud compute ssh app@${INSTANCE_NAME} --zone=${ZONE} \
                --command="cd ~/threat-hunting-test && bash demo_threat_hunting.sh"
        fi
        ;;

    *)
        echo "Invalid option"
        exit 1
        ;;
esac

echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}Recording Complete!${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

if [ -f "$OUTPUT_FILE" ]; then
    echo -e "${GREEN}Output files:${NC}"
    echo "  • Terminal recording: $OUTPUT_FILE"
    if [ -f "$GIF_FILE" ]; then
        echo "  • Animated GIF: $GIF_FILE"
    fi
    echo ""
    echo -e "${CYAN}Next steps:${NC}"
    echo "  • Play recording: asciinema play $OUTPUT_FILE"
    echo "  • Upload to asciinema.org: asciinema upload $OUTPUT_FILE"
    if [ -f "$GIF_FILE" ]; then
        echo "  • Share GIF: Upload $GIF_FILE to GitHub/documentation"
    fi
fi

echo ""
