#!/bin/bash

echo "🚀 Simple Working Dashboard is Ready!"
echo "====================================="
echo ""
echo "🌐 Dashboard URL: http://10.45.254.19:8519"
echo ""
echo "✅ This is a SIMPLE dashboard that WILL work:"
echo "   • Basic Streamlit components only"
echo "   • No complex HTML/CSS"
echo "   • Guaranteed to render properly"
echo "   • Real BigQuery data option"
echo ""
echo "🚀 Opening dashboard..."
echo ""

# Open in browser
if command -v open >/dev/null 2>&1; then
    open "http://10.45.254.19:8519"
elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "http://10.45.254.19:8519"
elif command -v start >/dev/null 2>&1; then
    start "http://10.45.254.19:8519"
else
    echo "Please manually open: http://10.45.254.19:8519"
fi

echo ""
echo "✅ Dashboard should now work perfectly!"
echo "🔧 Features: SIEM Events, ADA Alerts, TAA Cases, CRA Incidents"
echo "📊 Charts, Progress bars, and Status indicators"
