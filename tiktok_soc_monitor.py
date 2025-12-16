#!/usr/bin/env python3
"""
TikTok SOC Monitor - Monitor TikTok for telecom-related threats and mentions
"""

import os
import json
import time
import requests
from datetime import datetime, timedelta
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd

class TikTokSOCMonitor:
    """TikTok Social Media Intelligence Monitor for SOC"""
    
    def __init__(self):
        self.project_id = "chronicle-dev-2be9"
        self.dataset_id = "soc_data"
        self.table_id = "tiktok_soc_alerts"
        self.client = self._get_bigquery_client()
        self.setup_keywords()
        
    def _get_bigquery_client(self):
        """Initialize BigQuery client"""
        try:
            service_account_paths = [
                'Service Account BigQuery/chronicle-dev-2be-a-driven-soc.json',
                'Service Account BigQuery/sa-gatra-bigquery.json',
                'Service Account BigQuery/gatra-user-bigquery.json'
            ]
            
            for sa_path in service_account_paths:
                if os.path.exists(sa_path):
                    credentials = service_account.Credentials.from_service_account_file(sa_path)
                    return bigquery.Client(credentials=credentials)
            
            return bigquery.Client()
        except Exception as e:
            print(f"BigQuery client initialization failed: {e}")
            return None
    
    def setup_keywords(self):
        """Set up comprehensive keyword monitoring for Indosat/telecom on TikTok"""
        self.keywords = {
            'brand_primary': [
                'Indosat', 'IOH', 'IM3', 'Tri', '3 Indonesia', 'Hutchison',
                'indosatooredoo', 'indosat ooredoo', '@indosatooredoo'
            ],
            'brand_secondary': [
                'IM3 Ooredoo', 'Indosat IM3', 'Tri Indonesia', 'Hutchison 3',
                'indosat_ooredoo', 'indosat_im3', 'tri_indonesia'
            ],
            'service_issues': [
                'outage', 'down', 'gangguan', 'masalah', 'error', 'tidak bisa',
                'maintenance', 'perbaikan', 'pemeliharaan', 'mati', 'offline'
            ],
            'customer_complaints': [
                'komplain', 'keluhan', 'buruk', 'jelek', 'lambat', 'lemot',
                'mahal', 'paket mahal', 'kuota habis', 'nelpon gagal', 'sms gagal'
            ],
            'security_threats': [
                'hack', 'phishing', 'scam', 'penipuan', 'virus', 'malware',
                'bocor', 'data bocor', 'password', 'akun', 'curi'
            ],
            'products_services': [
                'paket', 'kuota', 'internet', 'nelpon', 'sms', 'roaming',
                '4G', '5G', 'jaringan', 'signal', 'sinyal'
            ],
            'competitors': [
                'Telkomsel', 'XL', 'Axis', 'Smartfren', 'By.U', 'telkomsel',
                'xl axiata', 'axis', 'smartfren', 'byu'
            ],
            'tiktok_specific': [
                'indosat tiktok', 'im3 tiktok', 'tri tiktok', 'ioh tiktok',
                'indosat viral', 'im3 viral', 'tri viral', 'telekomunikasi viral'
            ]
        }
        
        # Combine all keywords for monitoring
        self.all_keywords = []
        for category, keywords in self.keywords.items():
            self.all_keywords.extend(keywords)
    
    def create_tiktok_alerts_table(self):
        """Create TikTok SOC alerts table in BigQuery"""
        try:
            query = f"""
            CREATE TABLE IF NOT EXISTS `{self.project_id}.{self.dataset_id}.{self.table_id}` (
                video_id STRING,
                text STRING,
                created_at TIMESTAMP,
                user_screen_name STRING,
                user_followers INT64,
                view_count INT64,
                like_count INT64,
                comment_count INT64,
                share_count INT64,
                threat_score FLOAT64,
                severity STRING,
                risk_factors STRING,
                keywords_found STRING,
                analysis_timestamp TIMESTAMP,
                language STRING,
                video_url STRING,
                hashtags STRING,
                is_cyber_threat BOOLEAN,
                engagement_rate FLOAT64,
                virality_score FLOAT64
            )
            """
            
            self.client.query(query).result()
            print(f"✅ TikTok SOC alerts table created/verified: {self.table_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating TikTok table: {e}")
            return False
    
    def simulate_tiktok_search(self, keywords, limit=50):
        """Simulate TikTok search results (replace with actual TikTok API)"""
        # This is a simulation - in real implementation, you'd use TikTok API
        simulated_videos = []
        
        sample_videos = [
            {
                'video_id': f"tiktok_{int(time.time())}_{i}",
                'text': f"IOH paket internet mahal banget! Mending pindah ke kompetitor aja 😤 #indosat #telekomunikasi #viral",
                'user_screen_name': f"user_{i}",
                'user_followers': 15000 + (i * 500),
                'view_count': 50000 + (i * 10000),
                'like_count': 2500 + (i * 500),
                'comment_count': 150 + (i * 25),
                'share_count': 300 + (i * 50),
                'hashtags': "#indosat #telekomunikasi #viral #internet",
                'video_url': f"https://tiktok.com/@user_{i}/video/{int(time.time())}_{i}",
                'created_at': datetime.now() - timedelta(hours=i),
                'language': 'id'
            }
            for i in range(min(limit, 20))
        ]
        
        # Add some cyber threat examples
        cyber_threat_videos = [
            {
                'video_id': f"tiktok_cyber_{int(time.time())}_{i}",
                'text': f"WARNING! Fake Indosat account sedang mencuri data pengguna! Hati-hati dengan link ini! 🚨 #scam #indosat #cybersecurity",
                'user_screen_name': f"security_user_{i}",
                'user_followers': 50000 + (i * 10000),
                'view_count': 100000 + (i * 20000),
                'like_count': 8000 + (i * 1000),
                'comment_count': 500 + (i * 100),
                'share_count': 1000 + (i * 200),
                'hashtags': "#scam #indosat #cybersecurity #warning",
                'video_url': f"https://tiktok.com/@security_user_{i}/video/{int(time.time())}_{i}",
                'created_at': datetime.now() - timedelta(hours=i),
                'language': 'id'
            }
            for i in range(min(5, limit))
        ]
        
        simulated_videos.extend(sample_videos)
        simulated_videos.extend(cyber_threat_videos)
        
        return simulated_videos[:limit]
    
    def analyze_tiktok_video(self, video):
        """Analyze TikTok video for threats and risk factors"""
        text_lower = video['text'].lower()
        hashtags_lower = video.get('hashtags', '').lower()
        full_text = f"{text_lower} {hashtags_lower}"
        
        # Calculate threat score
        threat_score = 0.0
        risk_factors = []
        keywords_found = []
        
        # Check for cyber threats
        cyber_keywords = self.keywords['security_threats']
        if any(keyword in full_text for keyword in cyber_keywords):
            threat_score += 0.6
            risk_factors.append('security_threat')
            keywords_found.extend([kw for kw in cyber_keywords if kw in full_text])
        
        # Check for brand mentions
        brand_keywords = self.keywords['brand_primary'] + self.keywords['brand_secondary']
        if any(keyword in full_text for keyword in brand_keywords):
            threat_score += 0.2
            risk_factors.append('brand_mention')
            keywords_found.extend([kw for kw in brand_keywords if kw in full_text])
        
        # Check for complaints
        complaint_keywords = self.keywords['customer_complaints']
        if any(keyword in full_text for keyword in complaint_keywords):
            threat_score += 0.3
            risk_factors.append('customer_complaint')
            keywords_found.extend([kw for kw in complaint_keywords if kw in full_text])
        
        # Check for service issues
        service_keywords = self.keywords['service_issues']
        if any(keyword in full_text for keyword in service_keywords):
            threat_score += 0.4
            risk_factors.append('service_issue')
            keywords_factors.extend([kw for kw in service_keywords if kw in full_text])
        
        # Calculate engagement metrics
        view_count = video.get('view_count', 0)
        like_count = video.get('like_count', 0)
        comment_count = video.get('comment_count', 0)
        share_count = video.get('share_count', 0)
        followers = video.get('user_followers', 1)
        
        # Engagement rate
        engagement_rate = (like_count + comment_count + share_count) / max(view_count, 1)
        
        # Virality score
        virality_score = min(1.0, (view_count / 100000) + (engagement_rate * 10))
        
        # Boost threat score for high engagement
        if engagement_rate > 0.1:  # High engagement
            threat_score += 0.2
        
        # Boost for viral content
        if virality_score > 0.5:
            threat_score += 0.1
        
        # Boost for influential users
        if followers > 100000:
            threat_score += 0.1
        
        # Determine severity
        if threat_score >= 0.7:
            severity = "High"
        elif threat_score >= 0.4:
            severity = "Medium"
        else:
            severity = "Low"
        
        # Determine if it's a cyber threat
        is_cyber_threat = any(factor in risk_factors for factor in ['security_threat'])
        
        return {
            'threat_score': min(1.0, threat_score),
            'severity': severity,
            'risk_factors': ','.join(risk_factors),
            'keywords_found': ','.join(set(keywords_found)),
            'engagement_rate': engagement_rate,
            'virality_score': virality_score,
            'is_cyber_threat': is_cyber_threat
        }
    
    def store_tiktok_alert(self, video, analysis):
        """Store TikTok alert in BigQuery"""
        try:
            row = {
                'video_id': video['video_id'],
                'text': video['text'],
                'created_at': video['created_at'].isoformat() if isinstance(video['created_at'], datetime) else str(video['created_at']),
                'user_screen_name': video['user_screen_name'],
                'user_followers': video['user_followers'],
                'view_count': video['view_count'],
                'like_count': video['like_count'],
                'comment_count': video['comment_count'],
                'share_count': video['share_count'],
                'threat_score': analysis['threat_score'],
                'severity': analysis['severity'],
                'risk_factors': analysis['risk_factors'],
                'keywords_found': analysis['keywords_found'],
                'analysis_timestamp': datetime.now().isoformat(),
                'language': video['language'],
                'video_url': video['video_url'],
                'hashtags': video.get('hashtags', ''),
                'is_cyber_threat': analysis['is_cyber_threat'],
                'engagement_rate': analysis['engagement_rate'],
                'virality_score': analysis['virality_score']
            }
            
            table_ref = self.client.dataset(self.dataset_id).table(self.table_id)
            table = self.client.get_table(table_ref)
            
            errors = self.client.insert_rows_json(table, [row])
            if not errors:
                print(f"✅ TikTok alert stored: {video['video_id']}")
                return True
            else:
                print(f"❌ Error storing TikTok alert: {errors}")
                return False
                
        except Exception as e:
            print(f"❌ Error storing TikTok alert: {e}")
            return False
    
    def run_tiktok_monitoring_cycle(self):
        """Run one cycle of TikTok monitoring"""
        print(f"🔍 Starting TikTok monitoring cycle at {datetime.now()}")
        
        # Create table if needed
        if not self.create_tiktok_alerts_table():
            return False
        
        # Search for videos
        videos = self.simulate_tiktok_search(self.all_keywords, limit=25)
        print(f"📱 Found {len(videos)} TikTok videos to analyze")
        
        alerts_stored = 0
        for video in videos:
            # Analyze video
            analysis = self.analyze_tiktok_video(video)
            
            # Store alert
            if self.store_tiktok_alert(video, analysis):
                alerts_stored += 1
        
        print(f"✅ TikTok monitoring cycle completed: {alerts_stored} alerts stored")
        return True
    
    def run_continuous_tiktok_monitoring(self, interval_minutes=30):
        """Run continuous TikTok monitoring"""
        print(f"🚀 Starting continuous TikTok monitoring (every {interval_minutes} minutes)")
        
        while True:
            try:
                self.run_tiktok_monitoring_cycle()
                print(f"⏰ Waiting {interval_minutes} minutes until next cycle...")
                time.sleep(interval_minutes * 60)
            except KeyboardInterrupt:
                print("🛑 TikTok monitoring stopped by user")
                break
            except Exception as e:
                print(f"❌ Error in TikTok monitoring cycle: {e}")
                time.sleep(60)  # Wait 1 minute before retrying

def run_tiktok_monitoring():
    """Main function to run TikTok monitoring"""
    monitor = TikTokSOCMonitor()
    
    # Run single cycle
    monitor.run_tiktok_monitoring_cycle()
    
    # Uncomment to run continuously
    # monitor.run_continuous_tiktok_monitoring(interval_minutes=30)

if __name__ == "__main__":
    run_tiktok_monitoring()
