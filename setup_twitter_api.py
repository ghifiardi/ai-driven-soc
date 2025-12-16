#!/usr/bin/env python3
"""
Twitter API Setup Script
Helps configure Twitter API credentials for real-time monitoring
"""

import os
import sys

def setup_twitter_api():
    """Setup Twitter API credentials"""
    print("🐦 Twitter API Setup for SOC Monitoring")
    print("=" * 50)
    
    print("\n📋 Prerequisites:")
    print("1. Twitter Developer Account (free)")
    print("2. Twitter App with API v2 access")
    print("3. Bearer Token from your Twitter App")
    
    print("\n🔗 Steps to get Twitter API access:")
    print("1. Go to: https://developer.twitter.com/en/portal/dashboard")
    print("2. Create a new App or use existing one")
    print("3. Go to 'Keys and Tokens' tab")
    print("4. Generate 'Bearer Token' (not Access Token)")
    print("5. Copy the Bearer Token")
    
    print("\n🔐 Enter your Twitter Bearer Token:")
    print("(The token will be saved to environment variables)")
    
    bearer_token = input("\nBearer Token: ").strip()
    
    if not bearer_token:
        print("❌ No token provided. Setup cancelled.")
        return False
    
    if not bearer_token.startswith(('Bearer ', 'AAAAAAAAAAAAAAAAAAAA')):
        print("⚠️  Warning: Bearer token should start with 'Bearer ' or 'AAAAAAAAAAAAAAAAAAAA'")
        proceed = input("Continue anyway? (y/N): ").strip().lower()
        if proceed != 'y':
            return False
    
    # Save to environment file
    env_file = ".env"
    env_content = f"TWITTER_BEARER_TOKEN={bearer_token}\n"
    
    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        print(f"✅ Bearer token saved to {env_file}")
    except Exception as e:
        print(f"❌ Failed to save token: {e}")
        return False
    
    # Set environment variable for current session
    os.environ['TWITTER_BEARER_TOKEN'] = bearer_token
    print("✅ Environment variable set for current session")
    
    print("\n🚀 Next Steps:")
    print("1. Run: python twitter_real_monitor.py")
    print("2. Check BigQuery for stored tweets")
    print("3. Update dashboard to show real Twitter data")
    
    print("\n📊 Test the connection:")
    test_connection = input("Test Twitter API connection now? (y/N): ").strip().lower()
    
    if test_connection == 'y':
        try:
            import tweepy
            client = tweepy.Client(bearer_token=bearer_token)
            user = client.get_me()
            print(f"✅ Twitter API connection successful!")
            print(f"📱 Connected as: @{user.data.username}")
        except Exception as e:
            print(f"❌ Twitter API test failed: {e}")
            print("Please check your Bearer Token and try again.")
            return False
    
    return True

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    
    packages = [
        'tweepy>=4.0.0',
        'python-dotenv>=0.19.0'
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        os.system(f"pip install {package}")

if __name__ == "__main__":
    print("🔧 Twitter API Setup for AI-Driven SOC")
    print("This script will help you configure Twitter API access for real-time monitoring.")
    
    install_req = input("\nInstall required packages first? (y/N): ").strip().lower()
    if install_req == 'y':
        install_requirements()
    
    setup_twitter_api()
























