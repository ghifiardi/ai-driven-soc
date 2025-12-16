#!/usr/bin/env python3
"""
Test script to verify Gemini API integration is working
Run this after setting up your GEMINI_API_KEY
"""

import os
import sys

def test_gemini_integration():
    """Test Gemini API integration"""
    print("🧪 Testing Gemini API Integration...")
    print("=" * 40)

    # Check if API key is set
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY environment variable not set")
        print("💡 Please set it with: export GEMINI_API_KEY='your-api-key'")
        return False

    print(f"✅ API Key found (length: {len(api_key)})")

    try:
        import google.generativeai as genai

        # Configure Gemini
        genai.configure(api_key=api_key)
        print("✅ Gemini configured successfully")

        # Test model creation (use working model)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        print("✅ Model created successfully")

        # Test API call
        response = model.generate_content(
            "Hello! This is a test message. Please respond with 'Gemini API integration test successful' if you can read this."
        )

        response_text = response.text.strip()
        print(f"✅ API call successful! Response: {response_text}")

        if "test successful" in response_text.lower():
            print("🎉 GEMINI INTEGRATION IS WORKING PERFECTLY!")
            return True
        else:
            print("⚠️ API call worked but response was unexpected")
            return False

    except Exception as e:
        print(f"❌ Gemini API test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_gemini_integration()
    if success:
        print("\n🚀 Your dashboard should now work with Gemini LLM integration!")
    else:
        print("\n💡 Need help? Check the setup guide above.")
