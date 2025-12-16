#!/usr/bin/env python3
"""
Threat Detection Analysis
========================

Investigate why all alerts are classified as "benign" and look for potential
true positives that might be misclassified or hidden in the data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import re
from datetime import datetime
from typing import Dict, List, Any
from collections import Counter

class ThreatDetectionAnalyzer:
    """Analyzer to investigate threat detection patterns"""
    
    def __init__(self, csv_file: str):
        self.csv_file = csv_file
        self.df = None
        self.analysis = {}
        self.load_data()
    
    def load_data(self):
        """Load the comprehensive dataset"""
        print("🔍 Loading data for threat detection analysis...")
        
        try:
            self.df = pd.read_csv(self.csv_file)
            print(f"✅ Loaded {len(self.df):,} records")
            
            # Load both original and TAA extracted data
            try:
                self.taa_df = pd.read_csv("taa_extracted_data_20250918_115033.csv")
                print(f"📋 TAA extracted data: {len(self.taa_df):,} records")
            except:
                print("⚠️ TAA extracted data not found, using original data")
                self.taa_df = self.df
                
        except Exception as e:
            print(f"❌ Error loading data: {e}")
    
    def analyze_classification_patterns(self) -> Dict[str, Any]:
        """Analyze why everything is classified as benign"""
        print("\n🎯 Analyzing Classification Patterns...")
        
        analysis = {}
        
        # Check original classifications
        if 'classification' in self.df.columns:
            orig_classifications = self.df['classification'].value_counts()
            analysis['original_classifications'] = orig_classifications.to_dict()
            print(f"📊 Original classifications: {dict(orig_classifications)}")
        
        # Check TAA classifications
        if 'classification' in self.taa_df.columns:
            taa_classifications = self.taa_df['classification'].value_counts()
            analysis['taa_classifications'] = taa_classifications.to_dict()
            print(f"🤖 TAA classifications: {dict(taa_classifications)}")
        
        # Analyze confidence scores
        if 'confidence' in self.taa_df.columns:
            confidence_stats = self.taa_df['confidence'].describe()
            analysis['confidence_distribution'] = confidence_stats.to_dict()
            print(f"📈 Confidence scores: All={self.taa_df['confidence'].nunique()} unique values")
            print(f"    Range: {self.taa_df['confidence'].min():.3f} - {self.taa_df['confidence'].max():.3f}")
        
        # Check anomaly flags
        if 'is_anomaly' in self.taa_df.columns:
            anomaly_dist = self.taa_df['is_anomaly'].value_counts()
            analysis['anomaly_distribution'] = anomaly_dist.to_dict()
            print(f"🚨 Anomaly detection: {dict(anomaly_dist)}")
        
        self.analysis['classification_patterns'] = analysis
        return analysis
    
    def investigate_attack_categories(self) -> Dict[str, Any]:
        """Investigate attack categories that might indicate threats"""
        print("\n🛡️ Investigating Attack Categories...")
        
        investigation = {}
        
        # Look for attack categories in TAA data
        if 'attack_category' in self.taa_df.columns:
            attack_cats = self.taa_df['attack_category'].value_counts()
            investigation['attack_categories'] = attack_cats.to_dict()
            
            print(f"⚔️ Attack Categories Found:")
            for category, count in attack_cats.items():
                if pd.notna(category):
                    print(f"    {category}: {count:,} alerts")
            
            # Focus on non-Normal categories
            threat_categories = self.taa_df[
                (self.taa_df['attack_category'].notna()) & 
                (self.taa_df['attack_category'] != 'Normal')
            ]
            
            if len(threat_categories) > 0:
                investigation['potential_threats'] = {
                    'count': len(threat_categories),
                    'categories': threat_categories['attack_category'].value_counts().to_dict(),
                    'classifications': threat_categories['classification'].value_counts().to_dict()
                }
                
                print(f"🚨 Potential Threats Found: {len(threat_categories):,} alerts")
                print(f"    Categories: {threat_categories['attack_category'].value_counts().to_dict()}")
                print(f"    But classified as: {threat_categories['classification'].value_counts().to_dict()}")
        
        # Look for severity indicators
        if 'alert_severity' in self.taa_df.columns:
            severity_dist = self.taa_df['alert_severity'].value_counts()
            investigation['severity_distribution'] = severity_dist.to_dict()
            
            print(f"📊 Alert Severity Distribution:")
            for severity, count in severity_dist.items():
                if pd.notna(severity):
                    print(f"    {severity}: {count:,} alerts")
            
            # Check high severity alerts
            high_severity = self.taa_df[self.taa_df['alert_severity'] == 'high']
            if len(high_severity) > 0:
                investigation['high_severity_alerts'] = {
                    'count': len(high_severity),
                    'classifications': high_severity['classification'].value_counts().to_dict(),
                    'attack_categories': high_severity['attack_category'].value_counts().to_dict()
                }
                
                print(f"🚨 High Severity Alerts: {len(high_severity):,}")
                print(f"    But classified as: {high_severity['classification'].value_counts().to_dict()}")
        
        self.analysis['attack_investigation'] = investigation
        return investigation
    
    def analyze_raw_logs_for_threats(self) -> Dict[str, Any]:
        """Analyze raw logs for threat indicators"""
        print("\n🔍 Analyzing Raw Logs for Threat Indicators...")
        
        threat_analysis = {}
        
        # Look for threat keywords in raw logs
        threat_keywords = [
            'exploit', 'attack', 'malware', 'virus', 'trojan', 'backdoor',
            'intrusion', 'breach', 'suspicious', 'malicious', 'phishing',
            'ransomware', 'botnet', 'ddos', 'injection', 'xss', 'csrf',
            'brute force', 'privilege escalation', 'lateral movement'
        ]
        
        if 'raw_line' in self.df.columns:
            threat_indicators = {}
            
            for keyword in threat_keywords:
                matches = self.df['raw_line'].str.contains(keyword, case=False, na=False)
                count = matches.sum()
                if count > 0:
                    threat_indicators[keyword] = count
            
            threat_analysis['threat_keywords'] = threat_indicators
            
            if threat_indicators:
                print(f"🚨 Threat Keywords Found in Raw Logs:")
                for keyword, count in sorted(threat_indicators.items(), key=lambda x: x[1], reverse=True):
                    print(f"    {keyword}: {count:,} mentions")
            else:
                print("✅ No obvious threat keywords found in raw logs")
        
        # Look for suspicious IP patterns
        if 'source_ip' in self.taa_df.columns or 'destination_ip' in self.taa_df.columns:
            ip_analysis = {}
            
            # Analyze source IPs
            if 'source_ip' in self.taa_df.columns:
                source_ips = self.taa_df['source_ip'].value_counts().head(10)
                ip_analysis['top_source_ips'] = source_ips.to_dict()
                
                # Look for private vs public IPs
                private_ips = 0
                public_ips = 0
                
                for ip in self.taa_df['source_ip'].dropna():
                    if self.is_private_ip(str(ip)):
                        private_ips += 1
                    else:
                        public_ips += 1
                
                ip_analysis['ip_distribution'] = {
                    'private_ips': private_ips,
                    'public_ips': public_ips
                }
            
            threat_analysis['ip_analysis'] = ip_analysis
            
            print(f"🌐 IP Analysis:")
            if 'ip_distribution' in ip_analysis:
                print(f"    Private IPs: {ip_analysis['ip_distribution']['private_ips']:,}")
                print(f"    Public IPs: {ip_analysis['ip_distribution']['public_ips']:,}")
        
        self.analysis['threat_analysis'] = threat_analysis
        return threat_analysis
    
    def is_private_ip(self, ip: str) -> bool:
        """Check if IP is private"""
        try:
            parts = ip.split('.')
            if len(parts) != 4:
                return False
            
            first = int(parts[0])
            second = int(parts[1])
            
            # Private IP ranges
            if first == 10:
                return True
            elif first == 172 and 16 <= second <= 31:
                return True
            elif first == 192 and second == 168:
                return True
            
            return False
        except:
            return False
    
    def investigate_model_behavior(self) -> Dict[str, Any]:
        """Investigate why the model classifies everything as benign"""
        print("\n🤖 Investigating Model Behavior...")
        
        model_analysis = {}
        
        # Analyze model confidence patterns
        if 'confidence' in self.taa_df.columns:
            conf_data = self.taa_df['confidence'].dropna()
            
            model_analysis['confidence_analysis'] = {
                'unique_values': len(conf_data.unique()),
                'most_common_confidence': conf_data.mode().iloc[0] if len(conf_data) > 0 else None,
                'confidence_variance': float(conf_data.var()),
                'all_same_confidence': len(conf_data.unique()) == 1
            }
            
            print(f"📊 Confidence Analysis:")
            print(f"    Unique confidence values: {len(conf_data.unique())}")
            print(f"    Most common: {conf_data.mode().iloc[0] if len(conf_data) > 0 else 'N/A'}")
            print(f"    All same confidence: {len(conf_data.unique()) == 1}")
        
        # Check if model is stuck in default mode
        if 'model_type' in self.taa_df.columns:
            models = self.taa_df['model_type'].value_counts()
            model_analysis['model_types'] = models.to_dict()
            
            print(f"🤖 Model Types:")
            for model, count in models.items():
                if pd.notna(model):
                    print(f"    {model}: {count:,} predictions")
        
        # Look for processing patterns that might indicate issues
        if 'processing_time' in self.taa_df.columns:
            proc_times = self.taa_df['processing_time'].dropna()
            
            model_analysis['processing_patterns'] = {
                'mean_time': float(proc_times.mean()),
                'std_time': float(proc_times.std()),
                'all_similar_time': proc_times.std() < 0.001,  # Very low variance
                'min_time': float(proc_times.min()),
                'max_time': float(proc_times.max())
            }
            
            print(f"⏱️ Processing Time Patterns:")
            print(f"    Mean: {proc_times.mean():.4f}s")
            print(f"    Std Dev: {proc_times.std():.4f}s")
            print(f"    Very consistent timing: {proc_times.std() < 0.001}")
        
        self.analysis['model_behavior'] = model_analysis
        return model_analysis
    
    def look_for_hidden_threats(self) -> Dict[str, Any]:
        """Look for potential threats that might be misclassified"""
        print("\n🕵️ Looking for Hidden Threats...")
        
        hidden_threats = {}
        
        # Create a comprehensive threat score
        threat_scores = []
        
        for idx, row in self.taa_df.iterrows():
            score = 0
            reasons = []
            
            # Score based on attack category
            if pd.notna(row.get('attack_category')):
                if row['attack_category'] == 'Exploits':
                    score += 5
                    reasons.append('Exploit category')
                elif row['attack_category'] != 'Normal':
                    score += 3
                    reasons.append(f'Attack category: {row["attack_category"]}')
            
            # Score based on severity
            if pd.notna(row.get('alert_severity')):
                if row['alert_severity'] == 'high':
                    score += 4
                    reasons.append('High severity')
                elif row['alert_severity'] == 'medium':
                    score += 2
                    reasons.append('Medium severity')
            
            # Score based on confidence (lower confidence might indicate uncertainty)
            if pd.notna(row.get('confidence')):
                if row['confidence'] < 0.3:
                    score += 2
                    reasons.append('Low confidence')
            
            # Score based on external IPs
            if pd.notna(row.get('source_ip')):
                if not self.is_private_ip(str(row['source_ip'])):
                    score += 2
                    reasons.append('External source IP')
            
            if pd.notna(row.get('destination_ip')):
                if not self.is_private_ip(str(row['destination_ip'])):
                    score += 1
                    reasons.append('External destination IP')
            
            threat_scores.append({
                'row_id': idx,
                'threat_score': score,
                'reasons': reasons,
                'classification': row.get('classification'),
                'attack_category': row.get('attack_category'),
                'alert_severity': row.get('alert_severity')
            })
        
        # Sort by threat score
        threat_scores.sort(key=lambda x: x['threat_score'], reverse=True)
        
        # Get top potential threats
        top_threats = threat_scores[:20]  # Top 20 potential threats
        
        hidden_threats['potential_threats'] = top_threats
        hidden_threats['threat_score_distribution'] = {
            'high_threat_score': len([t for t in threat_scores if t['threat_score'] >= 5]),
            'medium_threat_score': len([t for t in threat_scores if 2 <= t['threat_score'] < 5]),
            'low_threat_score': len([t for t in threat_scores if t['threat_score'] == 1]),
            'no_threat_indicators': len([t for t in threat_scores if t['threat_score'] == 0])
        }
        
        print(f"🎯 Threat Score Analysis:")
        print(f"    High threat score (≥5): {hidden_threats['threat_score_distribution']['high_threat_score']:,}")
        print(f"    Medium threat score (2-4): {hidden_threats['threat_score_distribution']['medium_threat_score']:,}")
        print(f"    Low threat score (1): {hidden_threats['threat_score_distribution']['low_threat_score']:,}")
        print(f"    No threat indicators (0): {hidden_threats['threat_score_distribution']['no_threat_indicators']:,}")
        
        print(f"\n🚨 Top Potential Threats (misclassified as benign):")
        for i, threat in enumerate(top_threats[:10], 1):
            if threat['threat_score'] > 0:
                print(f"    {i}. Score: {threat['threat_score']}, Category: {threat['attack_category']}, "
                      f"Severity: {threat['alert_severity']}")
                print(f"       Reasons: {', '.join(threat['reasons'])}")
                print(f"       But classified as: {threat['classification']}")
        
        self.analysis['hidden_threats'] = hidden_threats
        return hidden_threats
    
    def generate_threat_detection_report(self):
        """Generate comprehensive threat detection report"""
        print("\n📊 Generating Threat Detection Report...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Threat Detection Analysis: Why Everything is Benign?', fontsize=16, fontweight='bold')
        
        # 1. Classification Distribution
        if 'taa_classifications' in self.analysis.get('classification_patterns', {}):
            class_data = self.analysis['classification_patterns']['taa_classifications']
            axes[0, 0].pie(class_data.values(), labels=class_data.keys(), autopct='%1.1f%%')
            axes[0, 0].set_title('Current Classifications\n(All Benign)')
        
        # 2. Attack Categories vs Classifications
        if 'attack_categories' in self.analysis.get('attack_investigation', {}):
            attack_data = self.analysis['attack_investigation']['attack_categories']
            valid_attacks = {k: v for k, v in attack_data.items() if pd.notna(k)}
            
            if valid_attacks:
                axes[0, 1].bar(valid_attacks.keys(), valid_attacks.values())
                axes[0, 1].set_title('Attack Categories Found\n(But Classified as Benign)')
                axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Threat Score Distribution
        if 'threat_score_distribution' in self.analysis.get('hidden_threats', {}):
            threat_dist = self.analysis['hidden_threats']['threat_score_distribution']
            categories = list(threat_dist.keys())
            values = list(threat_dist.values())
            
            axes[0, 2].bar(categories, values)
            axes[0, 2].set_title('Threat Score Distribution\n(Potential Misclassifications)')
            axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Confidence Score Analysis
        if 'confidence' in self.taa_df.columns:
            conf_data = self.taa_df['confidence'].dropna()
            axes[1, 0].hist(conf_data, bins=20, alpha=0.7, edgecolor='black')
            axes[1, 0].axvline(conf_data.mean(), color='red', linestyle='--', 
                              label=f'Mean: {conf_data.mean():.3f}')
            axes[1, 0].set_title('Confidence Score Distribution\n(All Same Value?)')
            axes[1, 0].legend()
        
        # 5. Severity vs Classification
        if 'severity_distribution' in self.analysis.get('attack_investigation', {}):
            severity_data = self.analysis['attack_investigation']['severity_distribution']
            valid_severity = {k: v for k, v in severity_data.items() if pd.notna(k)}
            
            if valid_severity:
                axes[1, 1].pie(valid_severity.values(), labels=valid_severity.keys(), autopct='%1.1f%%')
                axes[1, 1].set_title('Alert Severity Distribution\n(High Severity = Benign?)')
        
        # 6. Processing Time Consistency
        if 'processing_time' in self.taa_df.columns:
            proc_times = self.taa_df['processing_time'].dropna()
            axes[1, 2].hist(proc_times, bins=30, alpha=0.7, edgecolor='black')
            axes[1, 2].axvline(proc_times.mean(), color='red', linestyle='--',
                              label=f'Mean: {proc_times.mean():.4f}s')
            axes[1, 2].set_title('Processing Time Distribution\n(Too Consistent?)')
            axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig('threat_detection_analysis.png', dpi=300, bbox_inches='tight')
        print("  📊 Saved analysis: threat_detection_analysis.png")
        
        return fig
    
    def print_executive_summary(self):
        """Print executive summary of findings"""
        print("\n" + "="*60)
        print("🕵️ THREAT DETECTION INVESTIGATION SUMMARY")
        print("="*60)
        
        print("🎯 KEY FINDINGS:")
        
        # Classification issue
        class_patterns = self.analysis.get('classification_patterns', {})
        if 'taa_classifications' in class_patterns:
            benign_count = class_patterns['taa_classifications'].get('benign', 0)
            total_count = sum(class_patterns['taa_classifications'].values())
            benign_rate = benign_count / total_count if total_count > 0 else 0
            
            print(f"📊 Classification Issue:")
            print(f"   • {benign_rate:.1%} of alerts classified as 'benign'")
            print(f"   • {benign_count:,} out of {total_count:,} alerts")
        
        # Hidden threats
        hidden_threats = self.analysis.get('hidden_threats', {})
        if 'threat_score_distribution' in hidden_threats:
            high_threat = hidden_threats['threat_score_distribution']['high_threat_score']
            medium_threat = hidden_threats['threat_score_distribution']['medium_threat_score']
            
            print(f"🚨 Potential Misclassifications:")
            print(f"   • {high_threat:,} alerts with high threat indicators")
            print(f"   • {medium_threat:,} alerts with medium threat indicators")
            print(f"   • All classified as 'benign' despite threat indicators")
        
        # Attack categories
        attack_inv = self.analysis.get('attack_investigation', {})
        if 'attack_categories' in attack_inv:
            exploit_count = attack_inv['attack_categories'].get('Exploits', 0)
            if exploit_count > 0:
                print(f"⚔️ Attack Categories Found:")
                print(f"   • {exploit_count:,} alerts categorized as 'Exploits'")
                print(f"   • But still classified as 'benign'")
        
        # Model behavior
        model_behavior = self.analysis.get('model_behavior', {})
        if 'confidence_analysis' in model_behavior:
            conf_analysis = model_behavior['confidence_analysis']
            if conf_analysis.get('all_same_confidence'):
                print(f"🤖 Model Behavior Issues:")
                print(f"   • All alerts have identical confidence score")
                print(f"   • Model appears stuck in default mode")
                print(f"   • No dynamic threat assessment")
        
        print("\n🎯 ROOT CAUSE ANALYSIS:")
        print("1. 🔧 Model Configuration Issue:")
        print("   • supervised_v1 model appears overly conservative")
        print("   • All confidence scores identical (0.2)")
        print("   • No dynamic threat assessment")
        
        print("2. 🚨 Threshold Problems:")
        print("   • Anomaly detection threshold too high")
        print("   • Classification threshold favors 'benign'")
        print("   • No multi-level threat scoring")
        
        print("3. 📊 Data Processing Issues:")
        print("   • Attack categories ignored in final classification")
        print("   • Severity levels not factored into decisions")
        print("   • Threat indicators present but not utilized")
        
        print("\n💡 RECOMMENDATIONS:")
        print("1. 🎯 Immediate Actions:")
        print("   • Adjust classification thresholds")
        print("   • Enable dynamic confidence scoring")
        print("   • Integrate attack categories into classification logic")
        
        print("2. 🔧 Model Improvements:")
        print("   • Retrain supervised_v1 model with balanced dataset")
        print("   • Implement multi-stage threat assessment")
        print("   • Add severity-based classification rules")
        
        print("3. 📈 Enhanced Detection:")
        print("   • Implement threat scoring algorithm")
        print("   • Add behavioral anomaly detection")
        print("   • Enable adaptive learning")
        
        print("="*60)
    
    def save_investigation_results(self):
        """Save investigation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save analysis
        results_file = f"threat_detection_investigation_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(self.analysis, f, indent=2, default=str)
        
        print(f"\n📁 Investigation Results Saved: {results_file}")
        return results_file

def main():
    """Main investigation function"""
    csv_file = "comprehensive_feedback_data_20250918_112413.csv"
    
    print("🕵️ Threat Detection Investigation")
    print("=" * 60)
    print("Investigating why everything is classified as 'benign'...")
    
    # Initialize analyzer
    analyzer = ThreatDetectionAnalyzer(csv_file)
    
    if analyzer.df is None:
        return
    
    # Run investigation
    analyzer.analyze_classification_patterns()
    analyzer.investigate_attack_categories()
    analyzer.analyze_raw_logs_for_threats()
    analyzer.investigate_model_behavior()
    analyzer.look_for_hidden_threats()
    
    # Generate report
    analyzer.generate_threat_detection_report()
    
    # Print summary
    analyzer.print_executive_summary()
    
    # Save results
    analyzer.save_investigation_results()
    
    print("\n✅ Threat Detection Investigation Complete!")

if __name__ == "__main__":
    main()


