#!/bin/bash

set -e

# === CONFIGURATION ===
PROJECT_ID="chronicle-dev-2be9"
WORKDIR="/home/app/ai-driven-soc"
VENV_DIR="/home/app/py311-venv"
SERVICE_ACCOUNT_JSON="$WORKDIR/your-service-account.json"   # <-- Update if needed
PYTHON_MAIN="langgraph_ada_integration.py"
SYSTEMD_SERVICE="/etc/systemd/system/ada.service"

# === 1. PYTHON VENV SETUP ===
echo "[1/5] Setting up Python venv and installing dependencies..."
cd "$WORKDIR"
if [ ! -d "$VENV_DIR" ]; then
    python3.11 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt

# === 2. SYSTEMD SERVICE FILE ===
echo "[2/5] Creating systemd service file..."
sudo tee "$SYSTEMD_SERVICE" > /dev/null <<EOF
[Unit]
Description=ADA Anomaly Detection Agent
After=network.target

[Service]
User=app
WorkingDirectory=$WORKDIR
Environment="PROJECT_ID=$PROJECT_ID"
Environment="GOOGLE_APPLICATION_CREDENTIALS=$SERVICE_ACCOUNT_JSON"
Environment="VERTEX_AI_LOCATION=us-central1"
# Add more Environment= lines as needed
ExecStart=$VENV_DIR/bin/python $PYTHON_MAIN
Restart=always
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# === 3. RELOAD AND ENABLE SERVICE ===
echo "[3/5] Reloading systemd and enabling ada service..."
sudo systemctl daemon-reload
sudo systemctl enable ada

# === 4. RESTART SERVICE ===
echo "[4/5] Restarting ada service..."
sudo systemctl restart ada

# === 5. STATUS & LOGS ===
echo "[5/5] Service status:"
sudo systemctl status ada --no-pager

echo ""
echo "To monitor logs live, run:"
echo "  journalctl -u ada -f"

echo ""
echo "Deployment complete!"Description=ADA Anomaly Detection Agent
After=network.target

[Service]
User=app
WorkingDirectory=$WORKDIR
Environment="PROJECT_ID=$PROJECT_ID"
Environment="GOOGLE_APPLICATION_CREDENTIALS=$SERVICE_ACCOUNT_JSON"
Environment="VERTEX_AI_LOCATION=us-central1"
# Add more Environment= lines as needed
ExecStart=$VENV_DIR/bin/python $PYTHON_MAIN
Restart=always
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# === 3. RELOAD AND ENABLE SERVICE ===
echo "[3/5] Reloading systemd and enabling ada service..."
sudo systemctl daemon-reload
sudo systemctl enable ada

# === 4. RESTART SERVICE ===
echo "[4/5] Restarting ada service..."
sudo systemctl restart ada

# === 5. STATUS & LOGS ===
echo "[5/5] Service status:"
sudo systemctl status ada --no-pager

echo ""
echo "To monitor logs live, run:"
echo "  journalctl -u ada -f"

echo ""
echo "Deployment complete!"
"deploy_ada.sh" 65L, 1621C                                                                   65,27         Bot
