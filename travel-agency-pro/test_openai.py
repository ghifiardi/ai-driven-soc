#!/usr/bin/env python3
"""
Test OpenAI API integration for Travel Agency Pro
"""

import os
import asyncio
from openai import OpenAI
from langchain_openai import ChatOpenAI

def test_openai_direct():
    """Test direct OpenAI API"""
    print("🧪 Testing OpenAI API directly...")
    
    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": "Hello! Can you help me plan a trip from Jakarta to Paris?"}
            ],
            max_tokens=100
        )
        
        print("✅ OpenAI API working!")
        print(f"Response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI API failed: {e}")
        return False

def test_langchain_openai():
    """Test LangChain OpenAI integration"""
    print("\n🧪 Testing LangChain OpenAI integration...")
    
    try:
        llm = ChatOpenAI(
            model="gpt-4o",
            api_key=os.getenv('OPENAI_API_KEY'),
            temperature=0.1
        )
        
        response = llm.invoke("What's the best time to visit Paris?")
        
        print("✅ LangChain OpenAI working!")
        print(f"Response: {response.content}")
        return True
        
    except Exception as e:
        print(f"❌ LangChain OpenAI failed: {e}")
        return False

async def test_ai_automation():
    """Test AI automation capabilities"""
    print("\n🧪 Testing AI automation capabilities...")
    
    try:
        from langchain_openai import ChatOpenAI
        
        llm = ChatOpenAI(
            model="gpt-4o",
            api_key=os.getenv('OPENAI_API_KEY'),
            temperature=0.1
        )
        
        # Test travel planning prompt
        prompt = """
        You are a travel agent AI. A client wants to travel from Jakarta to Paris for a honeymoon.
        Budget: $5000
        Dates: December 15-22, 2024
        Preferences: Luxury, romantic, 5-star hotels
        
        Please provide:
        1. Flight recommendations
        2. Hotel recommendations  
        3. Estimated costs
        4. Special honeymoon suggestions
        """
        
        response = await llm.ainvoke(prompt)
        
        print("✅ AI automation working!")
        print("AI Response:")
        print(response.content)
        return True
        
    except Exception as e:
        print(f"❌ AI automation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Travel Agency Pro - OpenAI API Integration Test")
    print("=" * 60)
    
    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return False
    
    print(f"✅ API Key found: {api_key[:20]}...")
    
    # Run tests
    tests = [
        ("Direct OpenAI API", test_openai_direct),
        ("LangChain OpenAI", test_langchain_openai),
        ("AI Automation", lambda: asyncio.run(test_ai_automation()))
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! OpenAI integration is working.")
        print("🚀 Ready for REAL AI automation!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)