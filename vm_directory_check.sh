#!/bin/bash

echo "🔍 VM Directory Analysis"
echo "======================="

# Current directory
echo "📁 Current directory: $(pwd)"
echo "📁 Directory contents:"
ls -la | head -20

echo ""
echo "🎯 TAA Agent Files:"
ls -la *taa* 2>/dev/null || echo "No TAA files found"

echo ""
echo "🚀 Enhanced Classification Files:"
ls -la *enhanced* 2>/dev/null || echo "No enhanced files found"

echo ""
echo "📊 Python Files:"
ls -la *.py | wc -l
echo "Python files found"

echo ""
echo "🔧 Virtual Environments:"
ls -la venv* 2>/dev/null || echo "No virtual environments found"

echo ""
echo "📋 Requirements Files:"
ls -la requirements*.txt 2>/dev/null || echo "No requirements files found"

echo ""
echo "📈 Recent Files (last 24 hours):"
find . -maxdepth 1 -type f -mtime -1 -exec ls -la {} \; 2>/dev/null || echo "No recent files"

echo ""
echo "🎯 Services Status:"
ps aux | grep -E "(python|taa|ada)" | grep -v grep || echo "No relevant services running"

echo ""
echo "✅ VM Analysis Complete"
