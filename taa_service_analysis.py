#!/usr/bin/env python3
"""
TAA Service Deep Analysis
========================

Specialized analysis of the Triage and Analysis Agent (TAA) service
focusing on the 6,382 TAA records from the comprehensive dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class TAAServiceAnalyzer:
    """Specialized analyzer for TAA Service performance"""
    
    def __init__(self, csv_file: str):
        self.csv_file = csv_file
        self.df = None
        self.taa_df = None
        self.metrics = {}
        self.load_data()
    
    def load_data(self):
        """Load and filter TAA service data"""
        print("📊 Loading TAA Service Data...")
        
        try:
            self.df = pd.read_csv(self.csv_file)
            print(f"✅ Loaded {len(self.df):,} total records")
            
            # Filter for TAA service records (from previous analysis we know there are 6,382)
            # TAA records are those with 'unknown' source or from TAA service logs
            self.taa_df = self.df[
                (self.df['source'].isna()) | 
                (self.df['source'] == 'unknown') |
                (self.df['raw_line'].str.contains('TAAService', na=False))
            ].copy()
            
            print(f"📋 TAA Service Records: {len(self.taa_df):,}")
            
            if len(self.taa_df) == 0:
                print("⚠️ No TAA records found, using all records for analysis")
                self.taa_df = self.df.copy()
            
            self.extract_taa_data()
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
    
    def extract_taa_data(self):
        """Extract TAA-specific data from raw logs"""
        print("🔍 Extracting TAA Service Data...")
        
        taa_data = []
        
        for idx, row in self.taa_df.iterrows():
            raw_line = str(row.get('raw_line', ''))
            
            # Extract TAA-specific information
            taa_info = {
                'row_id': idx,
                'timestamp': self.extract_timestamp(raw_line),
                'alarm_id': self.extract_alarm_id(raw_line),
                'confidence': self.extract_confidence(raw_line),
                'classification': self.extract_classification(raw_line),
                'is_anomaly': self.extract_anomaly_status(raw_line),
                'model_type': self.extract_model_type(raw_line),
                'processing_time': self.extract_processing_time(raw_line),
                'enrichment_data': self.extract_enrichment_data(raw_line),
                'taa_decision': self.extract_taa_decision(raw_line),
                'alert_severity': self.extract_alert_severity(raw_line),
                'source_ip': self.extract_source_ip(raw_line),
                'destination_ip': self.extract_destination_ip(raw_line),
                'protocol': self.extract_protocol(raw_line),
                'attack_category': self.extract_attack_category(raw_line)
            }
            
            taa_data.append(taa_info)
        
        # Convert to DataFrame
        self.taa_extracted = pd.DataFrame(taa_data)
        
        print(f"✅ Extracted TAA data from {len(taa_data)} records")
        return self.taa_extracted
    
    def extract_timestamp(self, text: str) -> str:
        """Extract timestamp from TAA log"""
        patterns = [
            r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}',
            r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}'
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group()
        return None
    
    def extract_alarm_id(self, text: str) -> str:
        """Extract alarm ID"""
        pattern = r"'alarmId': '([^']+)'"
        match = re.search(pattern, text)
        return match.group(1) if match else None
    
    def extract_confidence(self, text: str) -> float:
        """Extract confidence score"""
        pattern = r"'confidence': ([0-9.]+)"
        match = re.search(pattern, text)
        return float(match.group(1)) if match else None
    
    def extract_classification(self, text: str) -> str:
        """Extract classification"""
        pattern = r"'classification': '([^']+)'"
        match = re.search(pattern, text)
        return match.group(1) if match else None
    
    def extract_anomaly_status(self, text: str) -> bool:
        """Extract anomaly status"""
        pattern = r"'is_anomaly': (True|False)"
        match = re.search(pattern, text)
        return match.group(1) == 'True' if match else None
    
    def extract_model_type(self, text: str) -> str:
        """Extract model type"""
        pattern = r"'model_type': '([^']+)'"
        match = re.search(pattern, text)
        return match.group(1) if match else None
    
    def extract_processing_time(self, text: str) -> float:
        """Extract processing time"""
        pattern = r"'processing_time': ([0-9.]+)"
        match = re.search(pattern, text)
        return float(match.group(1)) if match else None
    
    def extract_enrichment_data(self, text: str) -> str:
        """Extract enrichment data status"""
        if 'enrichment_data' in text:
            return 'present'
        elif 'processed_by_ada' in text:
            return 'ada_processed'
        return 'none'
    
    def extract_taa_decision(self, text: str) -> str:
        """Extract TAA decision from log"""
        if 'TAAService' in text:
            if 'Received alert' in text:
                return 'alert_received'
            elif 'Analysis complete' in text:
                return 'analysis_complete'
            elif 'Decision' in text:
                return 'decision_made'
        return 'unknown'
    
    def extract_alert_severity(self, text: str) -> str:
        """Extract alert severity"""
        patterns = [
            r'"label": ([01])',
            r'severity["\s]*:\s*["\s]*([^"]+)["\s]*'
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                value = match.group(1)
                if value == '1':
                    return 'high'
                elif value == '0':
                    return 'low'
                return value
        return None
    
    def extract_source_ip(self, text: str) -> str:
        """Extract source IP address"""
        pattern = r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
        matches = re.findall(pattern, text)
        return matches[0] if matches else None
    
    def extract_destination_ip(self, text: str) -> str:
        """Extract destination IP address"""
        pattern = r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
        matches = re.findall(pattern, text)
        return matches[1] if len(matches) > 1 else None
    
    def extract_protocol(self, text: str) -> str:
        """Extract network protocol"""
        protocols = ['tcp', 'udp', 'icmp', 'http', 'https', 'ssh', 'ftp', 'dns']
        text_lower = text.lower()
        for protocol in protocols:
            if f'"{protocol}"' in text_lower or f"'{protocol}'" in text_lower:
                return protocol
        return None
    
    def extract_attack_category(self, text: str) -> str:
        """Extract attack category"""
        pattern = r'"attack_cat": "([^"]+)"'
        match = re.search(pattern, text)
        return match.group(1) if match else None
    
    def compute_taa_performance_metrics(self) -> Dict[str, Any]:
        """Compute TAA-specific performance metrics"""
        print("\n🎯 Computing TAA Performance Metrics...")
        
        df = self.taa_extracted
        
        metrics = {
            'total_taa_records': len(df),
            'unique_alarms': df['alarm_id'].nunique(),
            'processing_efficiency': {},
            'classification_performance': {},
            'enrichment_analysis': {},
            'temporal_patterns': {}
        }
        
        # Processing efficiency
        proc_times = df['processing_time'].dropna()
        if len(proc_times) > 0:
            metrics['processing_efficiency'] = {
                'mean_processing_time': float(proc_times.mean()),
                'median_processing_time': float(proc_times.median()),
                'min_processing_time': float(proc_times.min()),
                'max_processing_time': float(proc_times.max()),
                'p95_processing_time': float(proc_times.quantile(0.95)),
                'p99_processing_time': float(proc_times.quantile(0.99)),
                'total_processed': len(proc_times),
                'processing_rate_per_second': float(1 / proc_times.mean()) if proc_times.mean() > 0 else 0
            }
        
        # Classification performance
        classifications = df['classification'].value_counts().to_dict()
        anomaly_dist = df['is_anomaly'].value_counts().to_dict()
        
        metrics['classification_performance'] = {
            'classification_distribution': classifications,
            'anomaly_distribution': anomaly_dist,
            'benign_rate': float(classifications.get('benign', 0) / len(df)) if len(df) > 0 else 0,
            'anomaly_detection_rate': float(anomaly_dist.get(True, 0) / len(df)) if len(df) > 0 else 0
        }
        
        # Enrichment analysis
        enrichment_dist = df['enrichment_data'].value_counts().to_dict()
        metrics['enrichment_analysis'] = {
            'enrichment_distribution': enrichment_dist,
            'enrichment_rate': float(enrichment_dist.get('present', 0) / len(df)) if len(df) > 0 else 0,
            'ada_processing_rate': float(enrichment_dist.get('ada_processed', 0) / len(df)) if len(df) > 0 else 0
        }
        
        # Temporal patterns
        timestamps = df['timestamp'].dropna()
        if len(timestamps) > 0:
            timestamp_strings = [str(ts) for ts in timestamps if str(ts) not in ['nan', 'None', '']]
            metrics['temporal_patterns'] = {
                'earliest_alert': min(timestamp_strings) if timestamp_strings else None,
                'latest_alert': max(timestamp_strings) if timestamp_strings else None,
                'total_timespan_covered': len(timestamp_strings),
                'alerts_with_timestamps': len(timestamp_strings)
            }
        
        self.metrics['taa_performance'] = metrics
        
        # Print summary
        print(f"  📈 TAA Records Analyzed: {metrics['total_taa_records']:,}")
        print(f"  🔢 Unique Alarm IDs: {metrics['unique_alarms']:,}")
        
        if 'mean_processing_time' in metrics['processing_efficiency']:
            proc_eff = metrics['processing_efficiency']
            print(f"  ⏱️ Processing: μ={proc_eff['mean_processing_time']:.4f}s, P95={proc_eff['p95_processing_time']:.4f}s")
            print(f"  🚀 Processing Rate: {proc_eff['processing_rate_per_second']:.1f} alerts/second")
        
        class_perf = metrics['classification_performance']
        print(f"  🎯 Benign Rate: {class_perf['benign_rate']:.1%}")
        print(f"  🚨 Anomaly Detection Rate: {class_perf['anomaly_detection_rate']:.1%}")
        
        enrich = metrics['enrichment_analysis']
        print(f"  📊 Enrichment Rate: {enrich['enrichment_rate']:.1%}")
        
        return metrics
    
    def analyze_taa_decision_patterns(self) -> Dict[str, Any]:
        """Analyze TAA decision-making patterns"""
        print("\n🧠 Analyzing TAA Decision Patterns...")
        
        df = self.taa_extracted
        
        # Decision flow analysis
        decision_flow = df['taa_decision'].value_counts().to_dict()
        
        # Confidence vs Classification correlation
        conf_class_analysis = {}
        df_clean = df.dropna(subset=['confidence', 'classification'])
        
        if len(df_clean) > 0:
            for classification in df_clean['classification'].unique():
                if pd.notna(classification):
                    class_data = df_clean[df_clean['classification'] == classification]
                    conf_scores = class_data['confidence'].dropna()
                    
                    if len(conf_scores) > 0:
                        conf_class_analysis[classification] = {
                            'mean_confidence': float(conf_scores.mean()),
                            'std_confidence': float(conf_scores.std()),
                            'count': len(conf_scores),
                            'min_confidence': float(conf_scores.min()),
                            'max_confidence': float(conf_scores.max())
                        }
        
        # Model performance by type
        model_performance = {}
        for model_type in df['model_type'].dropna().unique():
            model_data = df[df['model_type'] == model_type]
            
            model_performance[model_type] = {
                'total_processed': len(model_data),
                'avg_processing_time': float(model_data['processing_time'].mean()) if model_data['processing_time'].notna().sum() > 0 else 0,
                'classification_breakdown': model_data['classification'].value_counts().to_dict()
            }
        
        patterns = {
            'decision_flow': decision_flow,
            'confidence_by_classification': conf_class_analysis,
            'model_performance': model_performance
        }
        
        self.metrics['decision_patterns'] = patterns
        
        print(f"  🔄 Decision Flow Patterns: {len(decision_flow)} types")
        print(f"  📊 Confidence Analysis: {len(conf_class_analysis)} classifications")
        print(f"  🤖 Model Types Analyzed: {len(model_performance)}")
        
        return patterns
    
    def analyze_threat_intelligence_integration(self) -> Dict[str, Any]:
        """Analyze how TAA integrates with threat intelligence"""
        print("\n🛡️ Analyzing Threat Intelligence Integration...")
        
        df = self.taa_extracted
        
        # Network analysis
        network_analysis = {
            'unique_source_ips': df['source_ip'].nunique(),
            'unique_dest_ips': df['destination_ip'].nunique(),
            'protocol_distribution': df['protocol'].value_counts().to_dict(),
            'attack_category_distribution': df['attack_category'].value_counts().to_dict()
        }
        
        # Enrichment effectiveness
        enriched_data = df[df['enrichment_data'] == 'present']
        enrichment_effectiveness = {
            'total_enriched': len(enriched_data),
            'enrichment_rate': float(len(enriched_data) / len(df)) if len(df) > 0 else 0,
            'enriched_classifications': enriched_data['classification'].value_counts().to_dict() if len(enriched_data) > 0 else {},
            'enriched_anomaly_rate': float(enriched_data['is_anomaly'].sum() / len(enriched_data)) if len(enriched_data) > 0 else 0
        }
        
        # ADA integration analysis
        ada_processed = df[df['enrichment_data'] == 'ada_processed']
        ada_integration = {
            'ada_processed_count': len(ada_processed),
            'ada_processing_rate': float(len(ada_processed) / len(df)) if len(df) > 0 else 0,
            'ada_classifications': ada_processed['classification'].value_counts().to_dict() if len(ada_processed) > 0 else {},
            'ada_confidence_avg': float(ada_processed['confidence'].mean()) if len(ada_processed) > 0 and ada_processed['confidence'].notna().sum() > 0 else 0
        }
        
        threat_intel = {
            'network_analysis': network_analysis,
            'enrichment_effectiveness': enrichment_effectiveness,
            'ada_integration': ada_integration
        }
        
        self.metrics['threat_intelligence'] = threat_intel
        
        print(f"  🌐 Network Coverage: {network_analysis['unique_source_ips']} source IPs, {network_analysis['unique_dest_ips']} dest IPs")
        print(f"  📊 Enrichment Rate: {enrichment_effectiveness['enrichment_rate']:.1%}")
        print(f"  🤖 ADA Integration: {ada_integration['ada_processing_rate']:.1%}")
        
        return threat_intel
    
    def generate_taa_visualizations(self):
        """Generate TAA-specific visualizations"""
        print("\n📊 Generating TAA Visualizations...")
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('TAA Service Deep Analysis Dashboard', fontsize=16, fontweight='bold')
        
        df = self.taa_extracted
        
        # 1. Processing Time Distribution
        proc_times = df['processing_time'].dropna()
        if len(proc_times) > 0:
            axes[0, 0].hist(proc_times, bins=30, alpha=0.7, edgecolor='black')
            axes[0, 0].axvline(proc_times.mean(), color='red', linestyle='--', 
                              label=f'Mean: {proc_times.mean():.4f}s')
            axes[0, 0].set_title('TAA Processing Time Distribution')
            axes[0, 0].set_xlabel('Processing Time (seconds)')
            axes[0, 0].legend()
        
        # 2. Classification Distribution
        class_counts = df['classification'].value_counts().head(8)
        if len(class_counts) > 0:
            axes[0, 1].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%')
            axes[0, 1].set_title('TAA Classification Distribution')
        
        # 3. Confidence Scores by Classification
        df_conf = df.dropna(subset=['confidence', 'classification'])
        if len(df_conf) > 0:
            unique_classes = df_conf['classification'].unique()[:5]  # Top 5 classes
            conf_data = [df_conf[df_conf['classification'] == cls]['confidence'] for cls in unique_classes]
            conf_data = [data for data in conf_data if len(data) > 0]
            
            if conf_data:
                axes[0, 2].boxplot(conf_data, labels=unique_classes[:len(conf_data)])
                axes[0, 2].set_title('Confidence by Classification')
                axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Anomaly Detection Results
        anomaly_counts = df['is_anomaly'].value_counts()
        if len(anomaly_counts) > 0:
            axes[1, 0].bar(anomaly_counts.index.astype(str), anomaly_counts.values)
            axes[1, 0].set_title('TAA Anomaly Detection Results')
            axes[1, 0].set_xlabel('Is Anomaly')
        
        # 5. Enrichment Data Distribution
        enrich_counts = df['enrichment_data'].value_counts()
        if len(enrich_counts) > 0:
            axes[1, 1].pie(enrich_counts.values, labels=enrich_counts.index, autopct='%1.1f%%')
            axes[1, 1].set_title('Data Enrichment Distribution')
        
        # 6. Model Type Performance
        model_counts = df['model_type'].value_counts()
        if len(model_counts) > 0:
            axes[1, 2].bar(model_counts.index, model_counts.values)
            axes[1, 2].set_title('Model Type Usage')
            axes[1, 2].tick_params(axis='x', rotation=45)
        
        # 7. Protocol Distribution
        protocol_counts = df['protocol'].value_counts().head(10)
        if len(protocol_counts) > 0:
            axes[2, 0].bar(range(len(protocol_counts)), protocol_counts.values)
            axes[2, 0].set_xticks(range(len(protocol_counts)))
            axes[2, 0].set_xticklabels(protocol_counts.index, rotation=45)
            axes[2, 0].set_title('Network Protocol Distribution')
        
        # 8. Attack Category Distribution
        attack_counts = df['attack_category'].value_counts().head(8)
        if len(attack_counts) > 0:
            axes[2, 1].pie(attack_counts.values, labels=attack_counts.index, autopct='%1.1f%%')
            axes[2, 1].set_title('Attack Category Distribution')
        
        # 9. TAA Decision Flow
        decision_counts = df['taa_decision'].value_counts()
        if len(decision_counts) > 0:
            axes[2, 2].bar(decision_counts.index, decision_counts.values)
            axes[2, 2].set_title('TAA Decision Flow')
            axes[2, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('taa_service_analysis_dashboard.png', dpi=300, bbox_inches='tight')
        print("  📊 Saved TAA visualization: taa_service_analysis_dashboard.png")
        
        return fig
    
    def generate_taa_executive_summary(self):
        """Generate executive summary for TAA service"""
        print("\n" + "="*60)
        print("🎯 TAA SERVICE EXECUTIVE SUMMARY")
        print("="*60)
        
        taa_perf = self.metrics.get('taa_performance', {})
        decision_patterns = self.metrics.get('decision_patterns', {})
        threat_intel = self.metrics.get('threat_intelligence', {})
        
        print(f"📈 TAA Records Analyzed: {taa_perf.get('total_taa_records', 0):,}")
        print(f"🔢 Unique Alarm IDs: {taa_perf.get('unique_alarms', 0):,}")
        
        # Processing performance
        if 'processing_efficiency' in taa_perf:
            proc_eff = taa_perf['processing_efficiency']
            print(f"⏱️ Processing Performance:")
            print(f"   • Mean Processing Time: {proc_eff.get('mean_processing_time', 0):.4f}s")
            print(f"   • 95th Percentile: {proc_eff.get('p95_processing_time', 0):.4f}s")
            print(f"   • Processing Rate: {proc_eff.get('processing_rate_per_second', 0):.1f} alerts/sec")
        
        # Classification performance
        if 'classification_performance' in taa_perf:
            class_perf = taa_perf['classification_performance']
            print(f"🎯 Classification Performance:")
            print(f"   • Benign Rate: {class_perf.get('benign_rate', 0):.1%}")
            print(f"   • Anomaly Detection Rate: {class_perf.get('anomaly_detection_rate', 0):.1%}")
        
        # Threat intelligence integration
        if 'network_analysis' in threat_intel:
            network = threat_intel['network_analysis']
            print(f"🌐 Network Coverage:")
            print(f"   • Unique Source IPs: {network.get('unique_source_ips', 0):,}")
            print(f"   • Unique Destination IPs: {network.get('unique_dest_ips', 0):,}")
        
        if 'enrichment_effectiveness' in threat_intel:
            enrich = threat_intel['enrichment_effectiveness']
            print(f"📊 Enrichment Effectiveness:")
            print(f"   • Enrichment Rate: {enrich.get('enrichment_rate', 0):.1%}")
        
        if 'ada_integration' in threat_intel:
            ada = threat_intel['ada_integration']
            print(f"🤖 ADA Integration:")
            print(f"   • ADA Processing Rate: {ada.get('ada_processing_rate', 0):.1%}")
        
        print("="*60)
    
    def save_taa_results(self):
        """Save TAA analysis results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics
        metrics_file = f"taa_service_metrics_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        
        # Save extracted TAA data
        data_file = f"taa_extracted_data_{timestamp}.csv"
        self.taa_extracted.to_csv(data_file, index=False)
        
        print(f"\n📁 TAA Results Saved:")
        print(f"   📊 Metrics: {metrics_file}")
        print(f"   📄 Extracted Data: {data_file}")
        
        return metrics_file, data_file

def main():
    """Main TAA analysis function"""
    csv_file = "comprehensive_feedback_data_20250918_112413.csv"
    
    print("🚀 TAA Service Deep Analysis")
    print("=" * 60)
    
    # Initialize TAA analyzer
    analyzer = TAAServiceAnalyzer(csv_file)
    
    if analyzer.df is None:
        return
    
    # Compute TAA-specific metrics
    analyzer.compute_taa_performance_metrics()
    analyzer.analyze_taa_decision_patterns()
    analyzer.analyze_threat_intelligence_integration()
    
    # Generate visualizations
    analyzer.generate_taa_visualizations()
    
    # Print executive summary
    analyzer.generate_taa_executive_summary()
    
    # Save results
    analyzer.save_taa_results()
    
    print("\n✅ TAA Service Analysis Complete!")

if __name__ == "__main__":
    main()


