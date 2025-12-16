#!/usr/bin/env python3
"""
Simple SOC Analysis
==================

Robust analysis of the SOC feedback data with proper error handling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import re
from datetime import datetime
from typing import Dict, Any
from collections import Counter

class SimpleSOCAnalyzer:
    """Simple and robust SOC analyzer"""
    
    def __init__(self, csv_file: str):
        self.csv_file = csv_file
        self.df = None
        self.metrics = {}
        self.load_data()
    
    def load_data(self):
        """Load the CSV data"""
        print("📊 Loading SOC data...")
        
        try:
            self.df = pd.read_csv(self.csv_file)
            print(f"✅ Loaded {len(self.df):,} records with {len(self.df.columns)} columns")
            
            # Show column info
            print(f"📋 Columns: {', '.join(self.df.columns.tolist())}")
            
            # Basic data info
            print(f"📄 Sample data types:")
            for col in self.df.columns:
                non_null = self.df[col].notna().sum()
                print(f"  {col}: {non_null:,}/{len(self.df):,} non-null values")
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
    
    def extract_alert_data(self):
        """Extract structured data from raw_line column"""
        print("\n🔍 Extracting Alert Data from Raw Lines...")
        
        extracted_data = []
        
        for idx, row in self.df.iterrows():
            raw_line = str(row.get('raw_line', ''))
            
            # Extract structured information
            alert_info = {
                'row_id': idx,
                'timestamp': self.extract_timestamp(raw_line),
                'alarm_id': self.extract_alarm_id(raw_line),
                'confidence': self.extract_confidence(raw_line),
                'classification': self.extract_classification(raw_line),
                'is_anomaly': self.extract_anomaly_status(raw_line),
                'model_type': self.extract_model_type(raw_line),
                'processing_time': self.extract_processing_time(raw_line),
                'source': row.get('source', 'unknown')
            }
            
            extracted_data.append(alert_info)
        
        # Convert to DataFrame
        self.extracted_df = pd.DataFrame(extracted_data)
        
        print(f"✅ Extracted structured data from {len(extracted_data)} records")
        
        return self.extracted_df
    
    def extract_timestamp(self, text: str) -> str:
        """Extract timestamp from text"""
        patterns = [
            r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}',
            r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group()
        return None
    
    def extract_alarm_id(self, text: str) -> str:
        """Extract alarm ID from text"""
        pattern = r"'alarmId': '([^']+)'"
        match = re.search(pattern, text)
        return match.group(1) if match else None
    
    def extract_confidence(self, text: str) -> float:
        """Extract confidence score from text"""
        pattern = r"'confidence': ([0-9.]+)"
        match = re.search(pattern, text)
        return float(match.group(1)) if match else None
    
    def extract_classification(self, text: str) -> str:
        """Extract classification from text"""
        pattern = r"'classification': '([^']+)'"
        match = re.search(pattern, text)
        return match.group(1) if match else None
    
    def extract_anomaly_status(self, text: str) -> bool:
        """Extract anomaly status from text"""
        pattern = r"'is_anomaly': (True|False)"
        match = re.search(pattern, text)
        return match.group(1) == 'True' if match else None
    
    def extract_model_type(self, text: str) -> str:
        """Extract model type from text"""
        pattern = r"'model_type': '([^']+)'"
        match = re.search(pattern, text)
        return match.group(1) if match else None
    
    def extract_processing_time(self, text: str) -> float:
        """Extract processing time from text"""
        pattern = r"'processing_time': ([0-9.]+)"
        match = re.search(pattern, text)
        return float(match.group(1)) if match else None
    
    def compute_basic_metrics(self) -> Dict[str, Any]:
        """Compute basic metrics from extracted data"""
        print("\n📊 Computing Basic Metrics...")
        
        df = self.extracted_df
        
        metrics = {
            'total_records': len(df),
            'unique_alarms': df['alarm_id'].nunique(),
            'source_distribution': df['source'].value_counts().to_dict(),
            'classification_distribution': df['classification'].value_counts().to_dict(),
            'anomaly_distribution': df['is_anomaly'].value_counts().to_dict(),
            'model_type_distribution': df['model_type'].value_counts().to_dict()
        }
        
        # Handle timestamps safely
        valid_timestamps = df['timestamp'].dropna()
        if len(valid_timestamps) > 0:
            # Convert to strings first to handle mixed types
            timestamp_strings = [str(ts) for ts in valid_timestamps if str(ts) not in ['nan', 'None', '']]
            if timestamp_strings:
                metrics['date_range'] = {
                    'earliest': min(timestamp_strings),
                    'latest': max(timestamp_strings),
                    'valid_timestamps': len(timestamp_strings),
                    'sample_timestamps': timestamp_strings[:5]  # Show first 5 as samples
                }
            else:
                metrics['date_range'] = {'valid_timestamps': 0}
        else:
            metrics['date_range'] = {'valid_timestamps': 0}
        
        # Confidence statistics
        confidence_data = df['confidence'].dropna()
        if len(confidence_data) > 0:
            metrics['confidence_stats'] = {
                'count': len(confidence_data),
                'mean': float(confidence_data.mean()),
                'median': float(confidence_data.median()),
                'std': float(confidence_data.std()),
                'min': float(confidence_data.min()),
                'max': float(confidence_data.max()),
                'q25': float(confidence_data.quantile(0.25)),
                'q75': float(confidence_data.quantile(0.75))
            }
        
        # Processing time statistics
        proc_time_data = df['processing_time'].dropna()
        if len(proc_time_data) > 0:
            metrics['processing_time_stats'] = {
                'count': len(proc_time_data),
                'mean_seconds': float(proc_time_data.mean()),
                'median_seconds': float(proc_time_data.median()),
                'min_seconds': float(proc_time_data.min()),
                'max_seconds': float(proc_time_data.max()),
                'p95_seconds': float(proc_time_data.quantile(0.95)),
                'p99_seconds': float(proc_time_data.quantile(0.99))
            }
        
        self.metrics['basic'] = metrics
        
        # Print summary
        print(f"  📈 Total Records: {metrics['total_records']:,}")
        print(f"  🔢 Unique Alarms: {metrics['unique_alarms']:,}")
        print(f"  📅 Valid Timestamps: {metrics['date_range']['valid_timestamps']:,}")
        
        if 'confidence_stats' in metrics:
            conf = metrics['confidence_stats']
            print(f"  🎯 Confidence: μ={conf['mean']:.3f}, σ={conf['std']:.3f}")
        
        if 'processing_time_stats' in metrics:
            proc = metrics['processing_time_stats']
            print(f"  ⏱️ Processing Time: μ={proc['mean_seconds']:.4f}s, P95={proc['p95_seconds']:.4f}s")
        
        return metrics
    
    def compute_performance_metrics(self) -> Dict[str, Any]:
        """Compute SOC performance metrics"""
        print("\n🎯 Computing Performance Metrics...")
        
        df = self.extracted_df
        
        # Classification accuracy (using anomaly detection as ground truth)
        df_clean = df.dropna(subset=['is_anomaly', 'classification'])
        
        performance = {}
        
        if len(df_clean) > 0:
            # True positives: anomaly=True and classification suggests threat
            threat_classifications = ['malicious', 'suspicious', 'attack', 'exploit']
            
            df_clean['predicted_threat'] = df_clean['classification'].str.lower().isin(threat_classifications)
            
            tp = ((df_clean['is_anomaly'] == True) & (df_clean['predicted_threat'] == True)).sum()
            tn = ((df_clean['is_anomaly'] == False) & (df_clean['predicted_threat'] == False)).sum()
            fp = ((df_clean['is_anomaly'] == False) & (df_clean['predicted_threat'] == True)).sum()
            fn = ((df_clean['is_anomaly'] == True) & (df_clean['predicted_threat'] == False)).sum()
            
            total = tp + tn + fp + fn
            
            if total > 0:
                accuracy = (tp + tn) / total
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                performance['classification_metrics'] = {
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1),
                    'true_positives': int(tp),
                    'true_negatives': int(tn),
                    'false_positives': int(fp),
                    'false_negatives': int(fn),
                    'total_samples': int(total)
                }
        
        # Confidence calibration analysis
        df_conf = df.dropna(subset=['confidence', 'is_anomaly'])
        if len(df_conf) > 10:
            
            # Bin confidence scores
            df_conf['conf_bin'] = pd.cut(df_conf['confidence'], bins=10, labels=False)
            
            calibration_data = []
            for bin_idx in range(10):
                bin_data = df_conf[df_conf['conf_bin'] == bin_idx]
                if len(bin_data) > 0:
                    avg_confidence = bin_data['confidence'].mean()
                    actual_positive_rate = bin_data['is_anomaly'].mean()
                    calibration_data.append({
                        'bin': bin_idx,
                        'avg_confidence': float(avg_confidence),
                        'actual_positive_rate': float(actual_positive_rate),
                        'sample_count': len(bin_data)
                    })
            
            performance['confidence_calibration'] = calibration_data
        
        # Response time analysis (processing time as proxy)
        proc_times = df['processing_time'].dropna()
        if len(proc_times) > 0:
            performance['response_time_analysis'] = {
                'mean_response_seconds': float(proc_times.mean()),
                'median_response_seconds': float(proc_times.median()),
                'p95_response_seconds': float(proc_times.quantile(0.95)),
                'p99_response_seconds': float(proc_times.quantile(0.99)),
                'sla_compliance_1s': float((proc_times <= 1.0).mean()),
                'sla_compliance_5s': float((proc_times <= 5.0).mean()),
                'samples': len(proc_times)
            }
        
        self.metrics['performance'] = performance
        
        # Print summary
        if 'classification_metrics' in performance:
            cm = performance['classification_metrics']
            print(f"  🎯 Classification: Acc={cm['accuracy']:.3f}, P={cm['precision']:.3f}, R={cm['recall']:.3f}, F1={cm['f1_score']:.3f}")
        
        if 'response_time_analysis' in performance:
            rt = performance['response_time_analysis']
            print(f"  ⏱️ Response Time: μ={rt['mean_response_seconds']:.4f}s, P95={rt['p95_response_seconds']:.4f}s")
            print(f"  📊 SLA Compliance: <1s={rt['sla_compliance_1s']:.1%}, <5s={rt['sla_compliance_5s']:.1%}")
        
        return performance
    
    def generate_visualizations(self):
        """Generate key visualizations"""
        print("\n📊 Generating Visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('SOC Performance Analysis Dashboard', fontsize=16, fontweight='bold')
        
        df = self.extracted_df
        
        # 1. Source Distribution
        source_counts = df['source'].value_counts()
        if len(source_counts) > 0:
            axes[0, 0].pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%')
            axes[0, 0].set_title('Data Sources Distribution')
        
        # 2. Classification Distribution
        class_counts = df['classification'].value_counts().head(10)
        if len(class_counts) > 0:
            axes[0, 1].bar(range(len(class_counts)), class_counts.values)
            axes[0, 1].set_xticks(range(len(class_counts)))
            axes[0, 1].set_xticklabels(class_counts.index, rotation=45, ha='right')
            axes[0, 1].set_title('Top 10 Classifications')
        
        # 3. Confidence Distribution
        confidence_data = df['confidence'].dropna()
        if len(confidence_data) > 0:
            axes[0, 2].hist(confidence_data, bins=30, alpha=0.7, edgecolor='black')
            axes[0, 2].axvline(confidence_data.mean(), color='red', linestyle='--', 
                              label=f'Mean: {confidence_data.mean():.3f}')
            axes[0, 2].set_title('Confidence Score Distribution')
            axes[0, 2].set_xlabel('Confidence Score')
            axes[0, 2].legend()
        
        # 4. Anomaly Distribution
        anomaly_counts = df['is_anomaly'].value_counts()
        if len(anomaly_counts) > 0:
            axes[1, 0].bar(anomaly_counts.index.astype(str), anomaly_counts.values)
            axes[1, 0].set_title('Anomaly Detection Results')
            axes[1, 0].set_xlabel('Is Anomaly')
        
        # 5. Processing Time Distribution
        proc_time_data = df['processing_time'].dropna()
        if len(proc_time_data) > 0:
            axes[1, 1].hist(proc_time_data, bins=30, alpha=0.7, edgecolor='black')
            axes[1, 1].axvline(proc_time_data.mean(), color='red', linestyle='--',
                              label=f'Mean: {proc_time_data.mean():.4f}s')
            axes[1, 1].set_title('Processing Time Distribution')
            axes[1, 1].set_xlabel('Processing Time (seconds)')
            axes[1, 1].legend()
        
        # 6. Model Type Distribution
        model_counts = df['model_type'].value_counts()
        if len(model_counts) > 0:
            axes[1, 2].pie(model_counts.values, labels=model_counts.index, autopct='%1.1f%%')
            axes[1, 2].set_title('Model Type Distribution')
        
        plt.tight_layout()
        plt.savefig('soc_analysis_dashboard.png', dpi=300, bbox_inches='tight')
        print("  📊 Saved visualization: soc_analysis_dashboard.png")
        
        return fig
    
    def print_executive_summary(self):
        """Print executive summary"""
        print("\n" + "="*60)
        print("📊 SOC PERFORMANCE EXECUTIVE SUMMARY")
        print("="*60)
        
        basic = self.metrics.get('basic', {})
        performance = self.metrics.get('performance', {})
        
        print(f"📈 Total Records Analyzed: {basic.get('total_records', 0):,}")
        print(f"🔢 Unique Alarm IDs: {basic.get('unique_alarms', 0):,}")
        print(f"📅 Valid Timestamps: {basic.get('date_range', {}).get('valid_timestamps', 0):,}")
        
        if 'confidence_stats' in basic:
            conf = basic['confidence_stats']
            print(f"🎯 Confidence Score: μ={conf['mean']:.3f} (σ={conf['std']:.3f})")
        
        if 'classification_metrics' in performance:
            cm = performance['classification_metrics']
            print(f"📊 Classification Performance:")
            print(f"   • Accuracy: {cm['accuracy']:.3f}")
            print(f"   • Precision: {cm['precision']:.3f}")
            print(f"   • Recall: {cm['recall']:.3f}")
            print(f"   • F1-Score: {cm['f1_score']:.3f}")
        
        if 'response_time_analysis' in performance:
            rt = performance['response_time_analysis']
            print(f"⏱️ Response Performance:")
            print(f"   • Mean Response Time: {rt['mean_response_seconds']:.4f}s")
            print(f"   • 95th Percentile: {rt['p95_response_seconds']:.4f}s")
            print(f"   • SLA Compliance (<1s): {rt['sla_compliance_1s']:.1%}")
        
        # Top classifications
        if 'classification_distribution' in basic:
            top_classes = dict(list(basic['classification_distribution'].items())[:5])
            print(f"🏷️ Top Classifications:")
            for class_name, count in top_classes.items():
                if pd.notna(class_name):
                    print(f"   • {class_name}: {count:,} alerts")
        
        print("="*60)
    
    def save_results(self):
        """Save analysis results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics
        metrics_file = f"soc_analysis_metrics_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        
        # Save extracted data
        data_file = f"soc_extracted_data_{timestamp}.csv"
        self.extracted_df.to_csv(data_file, index=False)
        
        print(f"\n📁 Results saved:")
        print(f"   📊 Metrics: {metrics_file}")
        print(f"   📄 Extracted Data: {data_file}")
        
        return metrics_file, data_file

def main():
    """Main analysis function"""
    csv_file = "comprehensive_feedback_data_20250918_112413.csv"
    
    print("🚀 Simple SOC Analysis")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = SimpleSOCAnalyzer(csv_file)
    
    if analyzer.df is None:
        return
    
    # Extract structured data from raw logs
    analyzer.extract_alert_data()
    
    # Compute metrics
    analyzer.compute_basic_metrics()
    analyzer.compute_performance_metrics()
    
    # Generate visualizations
    analyzer.generate_visualizations()
    
    # Print summary
    analyzer.print_executive_summary()
    
    # Save results
    analyzer.save_results()
    
    print("\n✅ Analysis Complete!")

if __name__ == "__main__":
    main()
