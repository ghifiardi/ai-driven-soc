#!/bin/bash
# run_pilot_public.sh
# ===================
# Launch the SOC Platform and expose it to the internet via Cloudflare Tunnel.

# 1. Check for cloudflared
if ! command -v cloudflared &> /dev/null
then
    echo "❌ cloudflared not found."
    echo "   Please install it: brew install cloudflare/cloudflare/cloudflared"
    exit 1
fi

# 2. Start the SOC Server in the background
echo "🚀 Starting MSSP Platform Server..."
python3 mssp_platform_server.py &
SERVER_PID=$!

# Wait for server to start
sleep 2

# 3. Launch the Tunnel
echo "🛡️  Creating Secure Tunnel to http://localhost:8081..."
echo "--------------------------------------------------------"
echo "Look for a URL ending in '.trycloudflare.com' below."
echo "Use that as your 'Base URL' in the Integration Guide."
echo "--------------------------------------------------------"

cloudflared tunnel --url http://localhost:8081

# Cleanup on exit
kill $SERVER_PID
