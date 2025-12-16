#!/bin/bash
# Setup script for Gemini API integration

echo "🔑 Setting up Gemini API Key..."
echo "================================="

# Check if API key is provided as argument
if [ -z "$1" ]; then
    echo "❌ Please provide your Gemini API key as an argument:"
    echo "   ./setup_gemini.sh AIzaSyCB1jWcqRrHUueOUxLA2kt0gJUuY7ng8Ac"
    exit 1
fi

API_KEY="$1"

# Export the API key
export GEMINI_API_KEY="$API_KEY"
echo "✅ GEMINI_API_KEY environment variable set"

# Add to current shell session
echo "export GEMINI_API_KEY=\"$API_KEY\"" >> ~/.bashrc
echo "export GEMINI_API_KEY=\"$API_KEY\"" >> ~/.zshrc

echo ""
echo "🧪 Testing Gemini integration..."
python3 test_gemini_integration.py

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 SUCCESS! Gemini integration is working perfectly!"
    echo ""
    echo "💡 To make this permanent, restart your terminal or run:"
    echo "   source ~/.bashrc    # if using bash"
    echo "   source ~/.zshrc     # if using zsh"
    echo ""
    echo "🚀 Your dashboard is ready to use Gemini LLM features!"
else
    echo ""
    echo "❌ Setup failed. Please check your API key and try again."
fi






















