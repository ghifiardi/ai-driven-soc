#!/bin/bash
set -euo pipefail
cd /home/app/ai-driven-soc
source venv/bin/activate
# Run once now
python3 cla_hourly_retrain.py || true
# Then run hourly
while true; do
  sleep 3600
  python3 cla_hourly_retrain.py || true
done
