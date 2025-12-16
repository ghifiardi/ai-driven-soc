#!/bin/bash

echo "🚀 Accessing Native Streamlit Dashboard"
echo "========================================"
echo ""
echo "🌐 Dashboard URL: http://10.45.254.19:8518"
echo ""
echo "📊 This dashboard uses ONLY native Streamlit components"
echo "✅ No HTML/CSS rendering issues"
echo "🎨 Clean, professional enterprise look"
echo ""
echo "🔧 Features:"
echo "   • Real-time BigQuery data connection"
echo "   • Executive summary metrics"
echo "   • Threat intelligence overview"
echo "   • AI Agent performance (ADA, TAA, CRA)"
echo "   • Real-time activity timeline"
echo ""
echo "🚀 Opening dashboard in browser..."
echo ""

# Try to open in default browser (macOS)
if command -v open >/dev/null 2>&1; then
    open "http://10.45.254.19:8518"
elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "http://10.45.254.19:8518"
elif command -v start >/dev/null 2>&1; then
    start "http://10.45.254.19:8518"
else
    echo "Please manually open: http://10.45.254.19:8518"
fi

echo ""
echo "✅ Dashboard should now be accessible!"
echo "🔄 If you need to restart: ./restart_native_dashboard.sh"
