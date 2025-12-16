#!/usr/bin/env python3
"""
Gemini LLM Integration using REST API (Python 3.6 compatible)
Alternative to google-generativeai package
"""

import requests
import json
import os
from typing import Dict, List, Optional

class GeminiRESTClient:
    """Gemini LLM client using REST API for Python 3.6 compatibility"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        self.headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key
        }
    
    def analyze_alert(self, alert_text: str, alert_type: str = "security") -> Dict:
        """
        Analyze security alert using Gemini LLM
        """
        try:
            prompt = f"""
            Analyze this {alert_type} alert for threat level and provide insights:
            
            Alert: {alert_text}
            
            Please provide:
            1. Threat Level (Low/Medium/High)
            2. Key Risk Factors
            3. Recommended Actions
            4. Confidence Score (0-1)
            
            Format as JSON.
            """
            
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 500
                }
            }
            
            # Try gemini-2.0-flash-exp first, fallback to gemini-1.5-flash
            models_to_try = ["gemini-2.0-flash-exp", "gemini-1.5-flash"]
            
            for model in models_to_try:
                try:
                    url = f"{self.base_url}/{model}:generateContent"
                    response = requests.post(url, headers=self.headers, json=payload, timeout=30)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        if 'candidates' in result and len(result['candidates']) > 0:
                            content = result['candidates'][0]['content']['parts'][0]['text']
                            
                            # Try to parse as JSON, fallback to text
                            try:
                                analysis = json.loads(content)
                            except:
                                analysis = {
                                    "threat_level": "Medium",
                                    "risk_factors": ["Unable to parse AI response"],
                                    "recommended_actions": ["Review manually"],
                                    "confidence_score": 0.5,
                                    "raw_response": content
                                }
                            
                            return {
                                "success": True,
                                "model_used": model,
                                "analysis": analysis
                            }
                    
                except Exception as e:
                    print(f"Model {model} failed: {e}")
                    continue
            
            return {
                "success": False,
                "error": "All Gemini models failed",
                "analysis": {
                    "threat_level": "Unknown",
                    "risk_factors": ["AI analysis unavailable"],
                    "recommended_actions": ["Manual review required"],
                    "confidence_score": 0.0
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "analysis": {
                    "threat_level": "Unknown",
                    "risk_factors": ["AI analysis failed"],
                    "recommended_actions": ["Manual review required"],
                    "confidence_score": 0.0
                }
            }
    
    def enrich_social_media_alert(self, tweet_text: str, user_info: Dict) -> Dict:
        """
        Enrich social media alert with AI analysis
        """
        try:
            prompt = f"""
            Analyze this social media post for potential security threats or brand reputation issues:
            
            Post: {tweet_text}
            User: {user_info.get('user_screen_name', 'Unknown')}
            Followers: {user_info.get('user_followers', 0)}
            
            Assess:
            1. Threat Level (Low/Medium/High)
            2. Brand Impact (Positive/Neutral/Negative)
            3. Key Concerns (e.g., service outage, security issue, competitor mention)
            4. Urgency Level (Low/Medium/High)
            5. Recommended Response
            
            Format as JSON.
            """
            
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 400
                }
            }
            
            # Try gemini-2.0-flash-exp first
            url = f"{self.base_url}/gemini-2.0-flash-exp:generateContent"
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                if 'candidates' in result and len(result['candidates']) > 0:
                    content = result['candidates'][0]['content']['parts'][0]['text']
                    
                    try:
                        analysis = json.loads(content)
                    except:
                        analysis = {
                            "threat_level": "Medium",
                            "brand_impact": "Neutral",
                            "key_concerns": ["Unable to parse AI response"],
                            "urgency_level": "Medium",
                            "recommended_response": "Monitor situation",
                            "raw_response": content
                        }
                    
                    return {
                        "success": True,
                        "enrichment": analysis
                    }
            
            return {
                "success": False,
                "error": "Gemini API call failed",
                "enrichment": {
                    "threat_level": "Unknown",
                    "brand_impact": "Unknown",
                    "key_concerns": ["AI analysis unavailable"],
                    "urgency_level": "Low",
                    "recommended_response": "Manual review"
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "enrichment": {
                    "threat_level": "Unknown",
                    "brand_impact": "Unknown", 
                    "key_concerns": ["AI analysis failed"],
                    "urgency_level": "Low",
                    "recommended_response": "Manual review"
                }
            }

def get_gemini_client() -> Optional[GeminiRESTClient]:
    """Get Gemini client instance"""
    api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
    
    if not api_key:
        print("Warning: No Gemini API key found. Set GEMINI_API_KEY environment variable.")
        return None
    
    return GeminiRESTClient(api_key)

def test_gemini_integration():
    """Test Gemini integration"""
    client = get_gemini_client()
    
    if not client:
        print("❌ Gemini client not available")
        return False
    
    print("🧪 Testing Gemini LLM integration...")
    
    # Test alert analysis
    test_alert = "Suspicious login attempt detected from unknown IP address"
    result = client.analyze_alert(test_alert)
    
    if result['success']:
        print("✅ Alert analysis working!")
        print(f"Model used: {result['model_used']}")
        print(f"Analysis: {result['analysis']}")
    else:
        print("❌ Alert analysis failed")
        print(f"Error: {result['error']}")
    
    # Test social media enrichment
    test_tweet = "Indosat jaringan down lagi nih, gak bisa internet!"
    test_user = {"user_screen_name": "test_user", "user_followers": 1000}
    
    result = client.enrich_social_media_alert(test_tweet, test_user)
    
    if result['success']:
        print("✅ Social media enrichment working!")
        print(f"Enrichment: {result['enrichment']}")
    else:
        print("❌ Social media enrichment failed")
        print(f"Error: {result['error']}")
    
    return result['success']

if __name__ == "__main__":
    test_gemini_integration()
























