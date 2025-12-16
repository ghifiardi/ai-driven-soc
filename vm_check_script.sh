#!/bin/bash

echo "🔍 VM Directory Analysis for Enhanced Classification System"
echo "=========================================================="

# Current directory
echo "📁 Current directory: $(pwd)"
echo "📁 Directory contents (first 20 files):"
ls -la | head -20

echo ""
echo "🎯 TAA Agent Files:"
if ls *taa* 2>/dev/null; then
    echo "✅ TAA files found"
else
    echo "❌ No TAA files found"
fi

echo ""
echo "🚀 Enhanced Classification Files:"
if ls *enhanced* 2>/dev/null; then
    echo "✅ Enhanced files found:"
    ls -la *enhanced*
else
    echo "❌ No enhanced files found - NEEDS DEPLOYMENT!"
fi

echo ""
echo "📊 Python Files Count:"
python_count=$(ls *.py 2>/dev/null | wc -l)
echo "Python files: $python_count"

echo ""
echo "🔧 Virtual Environments:"
if ls -d venv* 2>/dev/null; then
    echo "✅ Virtual environments found:"
    ls -la venv*
else
    echo "❌ No virtual environments found"
fi

echo ""
echo "📋 Requirements Files:"
if ls requirements*.txt 2>/dev/null; then
    echo "✅ Requirements files found:"
    ls -la requirements*.txt
else
    echo "❌ No requirements files found"
fi

echo ""
echo "📈 Recent Files (last 24 hours):"
recent_files=$(find . -maxdepth 1 -type f -mtime -1 2>/dev/null | wc -l)
echo "Recent files: $recent_files"
if [ $recent_files -gt 0 ]; then
    find . -maxdepth 1 -type f -mtime -1 -exec ls -la {} \; 2>/dev/null
fi

echo ""
echo "🎯 Services Status:"
if ps aux | grep -E "(python|taa|ada)" | grep -v grep; then
    echo "✅ Relevant services running"
else
    echo "❌ No relevant services running"
fi

echo ""
echo "🔍 Critical Files Check:"
critical_files=(
    "taa_a2a_mcp_agent.py"
    "enhanced_taa_agent.py" 
    "enhanced_classification_engine.py"
    "threat_detection_analysis.py"
)

for file in "${critical_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file - Present"
    else
        echo "❌ $file - MISSING"
    fi
done

echo ""
echo "📊 Directory Size:"
du -sh . 2>/dev/null || echo "Could not determine directory size"

echo ""
echo "🎯 DEPLOYMENT STATUS:"
if [ -f "enhanced_classification_engine.py" ] && [ -f "enhanced_taa_agent.py" ]; then
    echo "✅ Enhanced Classification System DEPLOYED"
    echo "🧪 Ready for testing: python3 enhanced_classification_engine.py"
else
    echo "❌ Enhanced Classification System NOT DEPLOYED"
    echo "🚨 URGENT: Need to deploy enhanced files from local machine"
    echo "📤 Recommended: Run deployment script from local machine"
fi

echo ""
echo "✅ VM Analysis Complete"


