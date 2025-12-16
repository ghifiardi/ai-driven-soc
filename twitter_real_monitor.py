#!/usr/bin/env python3
"""
Real Twitter SOC Monitor for Telecom Sector
Monitors Twitter for real-time threats related to Indosat, IOH, IM3, Tri
"""

import os
import time
import json
import requests
import pandas as pd
from datetime import datetime, timedelta
from google.cloud import bigquery
from typing import List, Dict, Optional
import tweepy
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTwitterSOCMonitor:
    def __init__(self):
        self.bigquery_client = self._get_bigquery_client()
        self.twitter_client = self._get_twitter_client()
        self.keywords = self._setup_keywords()
        
    def _get_bigquery_client(self):
        """Initialize BigQuery client"""
        try:
            # Try multiple service account paths
            service_account_paths = [
                'Service Account BigQuery/gatra-user-bigquery.json',
                'Service Account BigQuery/gatra-user-gemini.json',
                'Service Account BigQuery/sa-gatra-bigquery.json'
            ]
            
            for path in service_account_paths:
                if os.path.exists(path):
                    return bigquery.Client.from_service_account_json(path)
            
            # Fallback to default client
            return bigquery.Client()
        except Exception as e:
            logger.error(f"BigQuery client initialization failed: {e}")
            return None
    
    def _get_twitter_client(self):
        """Initialize Twitter API client"""
        try:
            # Twitter API v2 credentials
            bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
            
            if not bearer_token:
                logger.error("TWITTER_BEARER_TOKEN environment variable not set")
                return None
            
            # Initialize Tweepy client
            client = tweepy.Client(bearer_token=bearer_token)
            
            # Test the connection
            try:
                client.get_me()
                logger.info("Twitter API connection successful")
                return client
            except Exception as e:
                logger.error(f"Twitter API test failed: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Twitter client initialization failed: {e}")
            return None
    
    def _setup_keywords(self) -> List[str]:
        """Setup monitoring keywords"""
        return [
            # Brand names
            "indosat", "ioh", "im3", "tri", "indosat ooredoo",
            # Service issues
            "indosat bermasalah", "ioh down", "im3 error", "tri gangguan",
            "indosat lambat", "ioh tidak bisa", "im3 tidak aktif",
            # Customer complaints
            "indosat buruk", "ioh jelek", "im3 mengecewakan", "tri tidak puas",
            "indosat komplain", "ioh masalah", "im3 keluhan",
            # Security threats
            "indosat hack", "ioh scam", "im3 phishing", "tri fraud",
            "indosat data breach", "ioh penipuan", "im3 keamanan",
            # Competitors
            "telkomsel", "xl axiata", "smartfren",
            # General telecom
            "operator seluler", "provider indonesia", "telekomunikasi"
        ]
    
    def search_tweets(self, query: str, max_results: int = 100) -> List[Dict]:
        """Search for tweets using Twitter API v2"""
        if not self.twitter_client:
            logger.error("Twitter client not initialized")
            return []
        
        try:
            # Search for tweets
            tweets = tweepy.Paginator(
                self.twitter_client.search_recent_tweets,
                query=query,
                tweet_fields=['created_at', 'author_id', 'public_metrics', 'context_annotations', 'lang'],
                user_fields=['username', 'name', 'public_metrics', 'verified'],
                expansions=['author_id'],
                max_results=min(max_results, 100)  # API limit
            ).flatten(limit=max_results)
            
            results = []
            for tweet in tweets:
                # Get user info
                user = tweet.author if hasattr(tweet, 'author') else None
                
                tweet_data = {
                    'tweet_id': tweet.id,
                    'text': tweet.text,
                    'created_at': tweet.created_at.isoformat() if tweet.created_at else None,
                    'author_id': tweet.author_id,
                    'username': user.username if user else 'unknown',
                    'user_name': user.name if user else 'unknown',
                    'user_followers': user.public_metrics['followers_count'] if user and user.public_metrics else 0,
                    'user_verified': user.verified if user else False,
                    'retweet_count': tweet.public_metrics['retweet_count'] if tweet.public_metrics else 0,
                    'like_count': tweet.public_metrics['like_count'] if tweet.public_metrics else 0,
                    'reply_count': tweet.public_metrics['reply_count'] if tweet.public_metrics else 0,
                    'quote_count': tweet.public_metrics['quote_count'] if tweet.public_metrics else 0,
                    'language': tweet.lang or 'unknown',
                    'query_used': query
                }
                results.append(tweet_data)
            
            logger.info(f"Found {len(results)} tweets for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Twitter search failed for query '{query}': {e}")
            return []
    
    def analyze_tweet(self, tweet: Dict) -> Dict:
        """Analyze tweet for threat level and categorization"""
        text = tweet.get('text', '').lower()
        username = tweet.get('username', '').lower()
        user_followers = tweet.get('user_followers', 0)
        retweet_count = tweet.get('retweet_count', 0)
        like_count = tweet.get('like_count', 0)
        
        # Threat keywords
        threat_keywords = [
            'hack', 'scam', 'phishing', 'fraud', 'penipuan', 'keamanan',
            'data breach', 'curi data', 'akun palsu', 'fake account'
        ]
        
        # Complaint keywords
        complaint_keywords = [
            'buruk', 'jelek', 'mengecewakan', 'tidak puas', 'komplain',
            'masalah', 'keluhan', 'bermasalah', 'error', 'down', 'gangguan'
        ]
        
        # Brand keywords
        brand_keywords = ['indosat', 'ioh', 'im3', 'tri']
        
        # Calculate threat score
        threat_score = 0.0
        risk_factors = []
        keywords_found = []
        
        # Check for threat keywords
        for keyword in threat_keywords:
            if keyword in text:
                threat_score += 0.3
                risk_factors.append('security_threat')
                keywords_found.append(keyword)
        
        # Check for complaint keywords
        for keyword in complaint_keywords:
            if keyword in text:
                threat_score += 0.1
                risk_factors.append('customer_complaint')
                keywords_found.append(keyword)
        
        # Check for brand mentions
        for keyword in brand_keywords:
            if keyword in text:
                threat_score += 0.05
                keywords_found.append(keyword)
        
        # Influence factor (follower count)
        if user_followers > 100000:
            threat_score += 0.2
            risk_factors.append('high_influence')
        elif user_followers > 10000:
            threat_score += 0.1
            risk_factors.append('medium_influence')
        
        # Engagement factor
        total_engagement = retweet_count + like_count
        if total_engagement > 1000:
            threat_score += 0.15
            risk_factors.append('high_engagement')
        elif total_engagement > 100:
            threat_score += 0.05
            risk_factors.append('medium_engagement')
        
        # Verified account factor
        if tweet.get('user_verified', False):
            threat_score += 0.1
            risk_factors.append('verified_account')
        
        # Cap the threat score
        threat_score = min(threat_score, 1.0)
        
        # Determine severity
        if threat_score >= 0.7:
            severity = 'High'
        elif threat_score >= 0.4:
            severity = 'Medium'
        else:
            severity = 'Low'
        
        # Determine category
        if any('security_threat' in factor for factor in risk_factors):
            category = 'Cyber Threat'
        elif any('customer_complaint' in factor for factor in risk_factors):
            category = 'Customer Complaint'
        elif any('high_influence' in factor for factor in risk_factors):
            category = 'Influential User'
        else:
            category = 'General Mention'
        
        return {
            'threat_score': threat_score,
            'severity': severity,
            'category': category,
            'risk_factors': ','.join(risk_factors),
            'keywords_found': ','.join(keywords_found),
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def store_tweet_alert(self, tweet: Dict, analysis: Dict):
        """Store tweet alert in BigQuery"""
        if not self.bigquery_client:
            logger.error("BigQuery client not available")
            return False
        
        try:
            table_id = "chronicle-dev-2be9.soc_data.twitter_real_alerts"
            
            # Prepare the row
            row = {
                'tweet_id': str(tweet['tweet_id']),
                'text': tweet['text'],
                'created_at': tweet['created_at'],
                'author_id': str(tweet['author_id']),
                'username': tweet['username'],
                'user_name': tweet['user_name'],
                'user_followers': tweet['user_followers'],
                'user_verified': tweet['user_verified'],
                'retweet_count': tweet['retweet_count'],
                'like_count': tweet['like_count'],
                'reply_count': tweet['reply_count'],
                'quote_count': tweet['quote_count'],
                'language': tweet['language'],
                'query_used': tweet['query_used'],
                'threat_score': analysis['threat_score'],
                'severity': analysis['severity'],
                'category': analysis['category'],
                'risk_factors': analysis['risk_factors'],
                'keywords_found': analysis['keywords_found'],
                'analysis_timestamp': analysis['analysis_timestamp']
            }
            
            # Insert into BigQuery
            errors = self.bigquery_client.insert_rows_json(table_id, [row])
            if errors:
                logger.error(f"BigQuery insert errors: {errors}")
                return False
            else:
                logger.info(f"Stored tweet alert: {tweet['tweet_id']}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to store tweet alert: {e}")
            return False
    
    def create_twitter_alerts_table(self):
        """Create Twitter alerts table in BigQuery"""
        if not self.bigquery_client:
            logger.error("BigQuery client not available")
            return False
        
        try:
            query = """
            CREATE TABLE IF NOT EXISTS `chronicle-dev-2be9.soc_data.twitter_real_alerts` (
                tweet_id STRING,
                text STRING,
                created_at TIMESTAMP,
                author_id STRING,
                username STRING,
                user_name STRING,
                user_followers INT64,
                user_verified BOOLEAN,
                retweet_count INT64,
                like_count INT64,
                reply_count INT64,
                quote_count INT64,
                language STRING,
                query_used STRING,
                threat_score FLOAT64,
                severity STRING,
                category STRING,
                risk_factors STRING,
                keywords_found STRING,
                analysis_timestamp TIMESTAMP
            )
            """
            
            self.bigquery_client.query(query).result()
            logger.info("Twitter alerts table created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create Twitter alerts table: {e}")
            return False
    
    def run_monitoring_cycle(self):
        """Run one monitoring cycle"""
        logger.info("Starting Twitter monitoring cycle...")
        
        # Create table if it doesn't exist
        self.create_twitter_alerts_table()
        
        total_tweets = 0
        total_alerts = 0
        
        for keyword in self.keywords[:5]:  # Limit to first 5 keywords to avoid rate limits
            logger.info(f"Searching for: {keyword}")
            
            # Search for tweets
            tweets = self.search_tweets(f"{keyword} -is:retweet lang:id", max_results=20)
            total_tweets += len(tweets)
            
            # Analyze and store tweets
            for tweet in tweets:
                analysis = self.analyze_tweet(tweet)
                
                # Only store tweets with threat score > 0.1
                if analysis['threat_score'] > 0.1:
                    if self.store_tweet_alert(tweet, analysis):
                        total_alerts += 1
            
            # Rate limiting - wait between requests
            time.sleep(2)
        
        logger.info(f"Monitoring cycle complete: {total_tweets} tweets processed, {total_alerts} alerts stored")
        return total_alerts

def run_twitter_monitoring():
    """Main function to run Twitter monitoring"""
    logger.info("Starting Real Twitter SOC Monitor...")
    
    # Check for Twitter API credentials
    if not os.getenv('TWITTER_BEARER_TOKEN'):
        logger.error("Please set TWITTER_BEARER_TOKEN environment variable")
        logger.info("Get your Bearer Token from: https://developer.twitter.com/en/portal/dashboard")
        return
    
    monitor = RealTwitterSOCMonitor()
    
    if not monitor.twitter_client:
        logger.error("Failed to initialize Twitter client")
        return
    
    # Run monitoring cycle
    alerts_stored = monitor.run_monitoring_cycle()
    
    logger.info(f"Twitter monitoring completed. {alerts_stored} alerts stored in BigQuery.")

if __name__ == "__main__":
    run_twitter_monitoring()
























