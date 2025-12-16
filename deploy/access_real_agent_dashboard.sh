#!/bin/bash

echo "🤖 Real AI Agent Monitoring Dashboard is Ready!"
echo "==============================================="
echo ""
echo "🌐 Dashboard URL: http://10.45.254.19:8520"
echo ""
echo "🚨 This dashboard shows REAL agent processing, not static data:"
echo "   • Live monitoring of ADA, TAA, and CRA agents"
echo "   • Real-time processing rates and status"
echo "   • Live process monitoring (CPU, Memory)"
echo "   • BigQuery integration for actual agent data"
echo "   • Auto-refresh with configurable intervals"
echo ""
echo "🔧 Features:"
echo "   • Real agent processes detection"
echo "   • Live processing activity charts"
echo "   • Agent status (Active/Slow/Inactive)"
echo "   • Process resource usage monitoring"
echo ""
echo "🚀 Opening dashboard..."
echo ""

# Open in browser
if command -v open >/dev/null 2>&1; then
    open "http://10.45.254.19:8520"
elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "http://10.45.254.19:8520"
elif command -v start >/dev/null 2>&1; then
    start "http://10.45.254.19:8520"
else
    echo "Please manually open: http://10.45.254.19:8520"
fi

echo ""
echo "✅ Real Agent Monitoring Dashboard is now accessible!"
echo "📊 This will show your AI agents actually processing, not just static numbers"
echo "🔄 Use the sidebar to control refresh rates and monitoring options"
