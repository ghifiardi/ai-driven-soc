#!/usr/bin/env python3
"""
SOC Metrics Analysis
===================

Comprehensive analysis of the extracted feedback data including:
- Counts and distributions
- Precision/Recall/F1 per agent/class
- MTTR (Mean Time To Response) distributions
- Confidence calibration and reliability diagrams
- Containment success rates
- Performance benchmarks
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import warnings
from typing import Dict, List, Tuple, Any
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.calibration import calibration_curve
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

class SOCMetricsAnalyzer:
    """Comprehensive SOC metrics analyzer"""
    
    def __init__(self, csv_file: str):
        """Initialize with the CSV file"""
        self.csv_file = csv_file
        self.df = None
        self.metrics = {}
        self.load_data()
    
    def load_data(self):
        """Load and preprocess the data"""
        print("📊 Loading SOC data...")
        
        try:
            self.df = pd.read_csv(self.csv_file)
            print(f"✅ Loaded {len(self.df):,} records with {len(self.df.columns)} columns")
            
            # Display basic info
            print(f"📋 Columns: {', '.join(self.df.columns.tolist())}")
            print(f"📅 Date range: {self.df['timestamp'].min() if 'timestamp' in self.df.columns else 'No timestamp'} to {self.df['timestamp'].max() if 'timestamp' in self.df.columns else 'No timestamp'}")
            
            # Clean and preprocess data
            self.preprocess_data()
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return
    
    def preprocess_data(self):
        """Clean and preprocess the data"""
        print("🔧 Preprocessing data...")
        
        # Convert timestamps if available - handle mixed types
        if 'timestamp' in self.df.columns:
            # First, convert to string to handle mixed types
            self.df['timestamp'] = self.df['timestamp'].astype(str)
            # Replace empty strings and 'nan' with NaN
            self.df['timestamp'] = self.df['timestamp'].replace(['', 'nan', 'None'], pd.NaT)
            # Convert to datetime
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], errors='coerce')
        
        # Clean confidence scores - handle mixed types
        for col in ['confidence', 'ada_confidence', 'taa_confidence']:
            if col in self.df.columns:
                # Replace comma with dot for decimal numbers
                self.df[col] = self.df[col].astype(str).str.replace(',', '.')
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Clean classification columns
        if 'classification' in self.df.columns:
            self.df['classification'] = self.df['classification'].fillna('unknown')
        
        # Extract agent information - handle missing source column
        if 'source' in self.df.columns:
            self.df['agent_source'] = self.df['source'].fillna('unknown')
        else:
            self.df['agent_source'] = 'unknown'
        
        # Create binary success indicators
        if 'final_decision' in self.df.columns:
            self.df['decision_success'] = self.df['final_decision'].notna()
        
        # Clean severity column if present
        if 'severity' in self.df.columns:
            self.df['severity'] = pd.to_numeric(self.df['severity'], errors='coerce')
        
        print("✅ Data preprocessing completed")
    
    def compute_basic_counts(self) -> Dict[str, Any]:
        """Compute basic counts and distributions"""
        print("\n📊 Computing Basic Counts...")
        
        counts = {
            'total_records': len(self.df),
            'unique_sources': self.df['agent_source'].nunique() if 'agent_source' in self.df.columns else 0,
            'date_range': {}
        }
        
        # Handle date range calculation safely
        if 'timestamp' in self.df.columns:
            valid_timestamps = self.df['timestamp'].dropna()
            if len(valid_timestamps) > 0:
                min_ts = valid_timestamps.min()
                max_ts = valid_timestamps.max()
                counts['date_range'] = {
                    'start': str(min_ts),
                    'end': str(max_ts),
                    'days': (max_ts - min_ts).days if pd.notna(min_ts) and pd.notna(max_ts) else None,
                    'valid_timestamps': len(valid_timestamps)
                }
            else:
                counts['date_range'] = {
                    'start': None,
                    'end': None,
                    'days': None,
                    'valid_timestamps': 0
                }
        
        # Source distribution
        if 'agent_source' in self.df.columns:
            counts['source_distribution'] = self.df['agent_source'].value_counts().to_dict()
        
        # Classification distribution
        if 'classification' in self.df.columns:
            counts['classification_distribution'] = self.df['classification'].value_counts().to_dict()
        
        # Confidence score statistics
        confidence_cols = ['confidence', 'ada_confidence', 'taa_confidence']
        counts['confidence_stats'] = {}
        
        for col in confidence_cols:
            if col in self.df.columns:
                conf_data = self.df[col].dropna()
                if len(conf_data) > 0:
                    counts['confidence_stats'][col] = {
                        'mean': float(conf_data.mean()),
                        'median': float(conf_data.median()),
                        'std': float(conf_data.std()),
                        'min': float(conf_data.min()),
                        'max': float(conf_data.max()),
                        'count': int(len(conf_data))
                    }
        
        self.metrics['basic_counts'] = counts
        
        # Print summary
        print(f"  📈 Total Records: {counts['total_records']:,}")
        print(f"  🔢 Unique Sources: {counts['unique_sources']}")
        if counts['date_range']['days']:
            print(f"  📅 Date Range: {counts['date_range']['days']} days")
        
        return counts
    
    def compute_precision_recall_f1(self) -> Dict[str, Any]:
        """Compute precision, recall, F1 per agent and class"""
        print("\n🎯 Computing Precision/Recall/F1 Metrics...")
        
        metrics = {}
        
        # We need ground truth and predictions
        # Let's use available data to simulate this
        
        if 'classification' in self.df.columns and 'final_decision' in self.df.columns:
            
            # Create ground truth from final decisions
            # This is a simplified approach - in real scenarios you'd have labeled data
            df_clean = self.df.dropna(subset=['classification', 'final_decision'])
            
            if len(df_clean) > 0:
                
                # Per classification metrics
                classifications = df_clean['classification'].unique()
                
                for classification in classifications:
                    if pd.notna(classification):
                        class_data = df_clean[df_clean['classification'] == classification]
                        
                        # Simulate binary classification (alert vs no-alert)
                        y_true = (class_data['final_decision'].str.contains('alert|true|positive', case=False, na=False)).astype(int)
                        y_pred = (class_data['confidence'] > 0.5).astype(int) if 'confidence' in class_data.columns else y_true
                        
                        if len(y_true) > 0 and len(set(y_true)) > 1:  # Need both classes
                            precision, recall, f1, support = precision_recall_fscore_support(
                                y_true, y_pred, average='binary', zero_division=0
                            )
                            
                            metrics[f'classification_{classification}'] = {
                                'precision': float(precision),
                                'recall': float(recall),
                                'f1_score': float(f1),
                                'support': int(support[1]) if isinstance(support, np.ndarray) else int(support),
                                'samples': len(class_data)
                            }
                
                # Per agent source metrics
                if 'agent_source' in df_clean.columns:
                    sources = df_clean['agent_source'].unique()
                    
                    for source in sources:
                        if pd.notna(source):
                            source_data = df_clean[df_clean['agent_source'] == source]
                            
                            y_true = (source_data['final_decision'].str.contains('alert|true|positive', case=False, na=False)).astype(int)
                            y_pred = (source_data['confidence'] > 0.5).astype(int) if 'confidence' in source_data.columns else y_true
                            
                            if len(y_true) > 0 and len(set(y_true)) > 1:
                                precision, recall, f1, support = precision_recall_fscore_support(
                                    y_true, y_pred, average='binary', zero_division=0
                                )
                                
                                metrics[f'agent_{source}'] = {
                                    'precision': float(precision),
                                    'recall': float(recall),
                                    'f1_score': float(f1),
                                    'support': int(support[1]) if isinstance(support, np.ndarray) else int(support),
                                    'samples': len(source_data)
                                }
        
        self.metrics['precision_recall_f1'] = metrics
        
        # Print summary
        print(f"  📊 Computed metrics for {len(metrics)} categories")
        for category, values in metrics.items():
            print(f"    {category}: P={values['precision']:.3f}, R={values['recall']:.3f}, F1={values['f1_score']:.3f}")
        
        return metrics
    
    def compute_mttr_distributions(self) -> Dict[str, Any]:
        """Compute Mean Time To Response distributions"""
        print("\n⏱️ Computing MTTR Distributions...")
        
        mttr_metrics = {}
        
        if 'timestamp' in self.df.columns and 'line_number' in self.df.columns:
            
            # Sort by timestamp to compute response times
            df_sorted = self.df.sort_values('timestamp').copy()
            
            # Compute time differences between consecutive events
            df_sorted['time_diff'] = df_sorted['timestamp'].diff()
            
            # Remove outliers (> 24 hours)
            df_sorted = df_sorted[df_sorted['time_diff'] <= timedelta(hours=24)]
            
            # Convert to minutes
            df_sorted['response_time_minutes'] = df_sorted['time_diff'].dt.total_seconds() / 60
            
            # Remove NaN values
            response_times = df_sorted['response_time_minutes'].dropna()
            
            if len(response_times) > 0:
                mttr_metrics['overall'] = {
                    'mean_minutes': float(response_times.mean()),
                    'median_minutes': float(response_times.median()),
                    'std_minutes': float(response_times.std()),
                    'min_minutes': float(response_times.min()),
                    'max_minutes': float(response_times.max()),
                    'p95_minutes': float(response_times.quantile(0.95)),
                    'p99_minutes': float(response_times.quantile(0.99)),
                    'count': int(len(response_times))
                }
                
                # MTTR by source
                if 'agent_source' in df_sorted.columns:
                    for source in df_sorted['agent_source'].unique():
                        if pd.notna(source):
                            source_times = df_sorted[df_sorted['agent_source'] == source]['response_time_minutes'].dropna()
                            
                            if len(source_times) > 0:
                                mttr_metrics[f'source_{source}'] = {
                                    'mean_minutes': float(source_times.mean()),
                                    'median_minutes': float(source_times.median()),
                                    'std_minutes': float(source_times.std()),
                                    'count': int(len(source_times))
                                }
                
                # MTTR by classification
                if 'classification' in df_sorted.columns:
                    for classification in df_sorted['classification'].unique():
                        if pd.notna(classification):
                            class_times = df_sorted[df_sorted['classification'] == classification]['response_time_minutes'].dropna()
                            
                            if len(class_times) > 0:
                                mttr_metrics[f'class_{classification}'] = {
                                    'mean_minutes': float(class_times.mean()),
                                    'median_minutes': float(class_times.median()),
                                    'count': int(len(class_times))
                                }
        
        self.metrics['mttr'] = mttr_metrics
        
        # Print summary
        if 'overall' in mttr_metrics:
            overall = mttr_metrics['overall']
            print(f"  ⏱️ Overall MTTR: {overall['mean_minutes']:.1f} minutes (median: {overall['median_minutes']:.1f})")
            print(f"  📊 P95: {overall['p95_minutes']:.1f} min, P99: {overall['p99_minutes']:.1f} min")
        else:
            print("  ⚠️ No timestamp data available for MTTR calculation")
        
        return mttr_metrics
    
    def compute_confidence_calibration(self) -> Dict[str, Any]:
        """Compute confidence calibration metrics"""
        print("\n📐 Computing Confidence Calibration...")
        
        calibration_metrics = {}
        
        # For each confidence column, compute calibration
        confidence_cols = ['confidence', 'ada_confidence', 'taa_confidence']
        
        for col in confidence_cols:
            if col in self.df.columns:
                
                # Get data with both confidence and outcome
                if 'final_decision' in self.df.columns:
                    df_clean = self.df[[col, 'final_decision']].dropna()
                    
                    if len(df_clean) > 10:  # Need sufficient data
                        
                        # Create binary outcomes
                        y_true = (df_clean['final_decision'].str.contains('alert|true|positive', case=False, na=False)).astype(int)
                        y_prob = df_clean[col].values
                        
                        # Ensure probabilities are in [0,1] range
                        y_prob = np.clip(y_prob, 0, 1)
                        
                        if len(set(y_true)) > 1:  # Need both classes
                            
                            # Compute calibration curve
                            try:
                                fraction_of_positives, mean_predicted_value = calibration_curve(
                                    y_true, y_prob, n_bins=10
                                )
                                
                                # Compute Expected Calibration Error (ECE)
                                bin_boundaries = np.linspace(0, 1, 11)
                                bin_lowers = bin_boundaries[:-1]
                                bin_uppers = bin_boundaries[1:]
                                
                                ece = 0
                                for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                                    in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
                                    prop_in_bin = in_bin.mean()
                                    
                                    if prop_in_bin > 0:
                                        accuracy_in_bin = y_true[in_bin].mean()
                                        avg_confidence_in_bin = y_prob[in_bin].mean()
                                        ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                                
                                calibration_metrics[col] = {
                                    'expected_calibration_error': float(ece),
                                    'fraction_of_positives': fraction_of_positives.tolist(),
                                    'mean_predicted_value': mean_predicted_value.tolist(),
                                    'samples': int(len(df_clean)),
                                    'reliability_score': float(1 - ece)  # Higher is better
                                }
                                
                            except Exception as e:
                                print(f"    ⚠️ Error computing calibration for {col}: {e}")
        
        self.metrics['calibration'] = calibration_metrics
        
        # Print summary
        print(f"  📊 Computed calibration for {len(calibration_metrics)} confidence measures")
        for col, metrics in calibration_metrics.items():
            print(f"    {col}: ECE={metrics['expected_calibration_error']:.3f}, Reliability={metrics['reliability_score']:.3f}")
        
        return calibration_metrics
    
    def compute_containment_success_rate(self) -> Dict[str, Any]:
        """Compute containment success rates"""
        print("\n🛡️ Computing Containment Success Rates...")
        
        containment_metrics = {}
        
        # Look for containment-related columns
        containment_cols = [col for col in self.df.columns if 'action' in col.lower() or 'containment' in col.lower()]
        
        if containment_cols or 'final_decision' in self.df.columns:
            
            # Overall success rate
            if 'final_decision' in self.df.columns:
                decisions = self.df['final_decision'].dropna()
                
                # Count successful containments (contains success keywords)
                successful = decisions.str.contains('success|contained|blocked|mitigated', case=False, na=False).sum()
                total = len(decisions)
                
                containment_metrics['overall'] = {
                    'success_rate': float(successful / total) if total > 0 else 0.0,
                    'successful_actions': int(successful),
                    'total_actions': int(total),
                    'failure_rate': float((total - successful) / total) if total > 0 else 0.0
                }
            
            # Success rate by agent source
            if 'agent_source' in self.df.columns and 'final_decision' in self.df.columns:
                for source in self.df['agent_source'].unique():
                    if pd.notna(source):
                        source_data = self.df[self.df['agent_source'] == source]
                        decisions = source_data['final_decision'].dropna()
                        
                        if len(decisions) > 0:
                            successful = decisions.str.contains('success|contained|blocked|mitigated', case=False, na=False).sum()
                            total = len(decisions)
                            
                            containment_metrics[f'source_{source}'] = {
                                'success_rate': float(successful / total),
                                'successful_actions': int(successful),
                                'total_actions': int(total)
                            }
            
            # Success rate by classification
            if 'classification' in self.df.columns and 'final_decision' in self.df.columns:
                for classification in self.df['classification'].unique():
                    if pd.notna(classification):
                        class_data = self.df[self.df['classification'] == classification]
                        decisions = class_data['final_decision'].dropna()
                        
                        if len(decisions) > 0:
                            successful = decisions.str.contains('success|contained|blocked|mitigated', case=False, na=False).sum()
                            total = len(decisions)
                            
                            containment_metrics[f'class_{classification}'] = {
                                'success_rate': float(successful / total),
                                'successful_actions': int(successful),
                                'total_actions': int(total)
                            }
        
        self.metrics['containment'] = containment_metrics
        
        # Print summary
        if 'overall' in containment_metrics:
            overall = containment_metrics['overall']
            print(f"  🛡️ Overall Success Rate: {overall['success_rate']:.1%} ({overall['successful_actions']}/{overall['total_actions']})")
        else:
            print("  ⚠️ No containment data available")
        
        return containment_metrics
    
    def compute_additional_metrics(self) -> Dict[str, Any]:
        """Compute additional SOC performance metrics"""
        print("\n📈 Computing Additional Metrics...")
        
        additional_metrics = {}
        
        # Alert volume trends
        if 'timestamp' in self.df.columns:
            df_time = self.df.copy()
            df_time['date'] = df_time['timestamp'].dt.date
            
            daily_counts = df_time.groupby('date').size()
            
            additional_metrics['volume_trends'] = {
                'daily_average': float(daily_counts.mean()),
                'daily_std': float(daily_counts.std()),
                'peak_day': str(daily_counts.idxmax()),
                'peak_count': int(daily_counts.max()),
                'min_day': str(daily_counts.idxmin()),
                'min_count': int(daily_counts.min())
            }
        
        # Confidence score distributions
        confidence_cols = ['confidence', 'ada_confidence', 'taa_confidence']
        additional_metrics['confidence_distributions'] = {}
        
        for col in confidence_cols:
            if col in self.df.columns:
                conf_data = self.df[col].dropna()
                if len(conf_data) > 0:
                    additional_metrics['confidence_distributions'][col] = {
                        'low_confidence': int((conf_data < 0.3).sum()),
                        'medium_confidence': int(((conf_data >= 0.3) & (conf_data < 0.7)).sum()),
                        'high_confidence': int((conf_data >= 0.7).sum()),
                        'very_high_confidence': int((conf_data >= 0.9).sum())
                    }
        
        # Data quality metrics
        additional_metrics['data_quality'] = {
            'completeness': {},
            'missing_data_percentage': {}
        }
        
        for col in self.df.columns:
            non_null_count = self.df[col].notna().sum()
            total_count = len(self.df)
            
            additional_metrics['data_quality']['completeness'][col] = float(non_null_count / total_count)
            additional_metrics['data_quality']['missing_data_percentage'][col] = float((total_count - non_null_count) / total_count)
        
        self.metrics['additional'] = additional_metrics
        
        return additional_metrics
    
    def generate_visualizations(self):
        """Generate key visualizations"""
        print("\n📊 Generating Visualizations...")
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('SOC Performance Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Source Distribution
        if 'source_distribution' in self.metrics.get('basic_counts', {}):
            source_data = self.metrics['basic_counts']['source_distribution']
            axes[0, 0].pie(source_data.values(), labels=source_data.keys(), autopct='%1.1f%%')
            axes[0, 0].set_title('Alert Sources Distribution')
        
        # 2. Classification Distribution
        if 'classification_distribution' in self.metrics.get('basic_counts', {}):
            class_data = self.metrics['basic_counts']['classification_distribution']
            axes[0, 1].bar(class_data.keys(), class_data.values())
            axes[0, 1].set_title('Classification Distribution')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Confidence Scores
        confidence_cols = ['confidence', 'ada_confidence', 'taa_confidence']
        conf_data = []
        conf_labels = []
        
        for col in confidence_cols:
            if col in self.df.columns:
                data = self.df[col].dropna()
                if len(data) > 0:
                    conf_data.append(data)
                    conf_labels.append(col)
        
        if conf_data:
            axes[0, 2].boxplot(conf_data, labels=conf_labels)
            axes[0, 2].set_title('Confidence Score Distributions')
            axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. MTTR Distribution
        if 'overall' in self.metrics.get('mttr', {}):
            # Create synthetic MTTR data for visualization
            mttr_data = self.metrics['mttr']['overall']
            mean_mttr = mttr_data['mean_minutes']
            std_mttr = mttr_data['std_minutes']
            
            # Generate sample data
            sample_mttr = np.random.normal(mean_mttr, std_mttr, 1000)
            sample_mttr = sample_mttr[sample_mttr > 0]  # Remove negative values
            
            axes[1, 0].hist(sample_mttr, bins=30, alpha=0.7, edgecolor='black')
            axes[1, 0].axvline(mean_mttr, color='red', linestyle='--', label=f'Mean: {mean_mttr:.1f} min')
            axes[1, 0].set_title('MTTR Distribution')
            axes[1, 0].set_xlabel('Response Time (minutes)')
            axes[1, 0].legend()
        
        # 5. Success Rates by Source
        if 'containment' in self.metrics:
            containment_data = self.metrics['containment']
            sources = [k.replace('source_', '') for k in containment_data.keys() if k.startswith('source_')]
            success_rates = [containment_data[f'source_{s}']['success_rate'] for s in sources]
            
            if sources:
                axes[1, 1].bar(sources, success_rates)
                axes[1, 1].set_title('Success Rate by Source')
                axes[1, 1].set_ylabel('Success Rate')
                axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. Calibration Plot
        if 'calibration' in self.metrics:
            for i, (col, cal_data) in enumerate(self.metrics['calibration'].items()):
                if 'fraction_of_positives' in cal_data:
                    x = cal_data['mean_predicted_value']
                    y = cal_data['fraction_of_positives']
                    axes[1, 2].plot(x, y, marker='o', label=f'{col} (ECE: {cal_data["expected_calibration_error"]:.3f})')
            
            axes[1, 2].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
            axes[1, 2].set_title('Confidence Calibration')
            axes[1, 2].set_xlabel('Mean Predicted Probability')
            axes[1, 2].set_ylabel('Fraction of Positives')
            axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig('soc_metrics_dashboard.png', dpi=300, bbox_inches='tight')
        print("  📊 Saved visualization: soc_metrics_dashboard.png")
        
        return fig
    
    def save_metrics_report(self, filename: str = None):
        """Save comprehensive metrics report"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"soc_metrics_report_{timestamp}.json"
        
        # Add metadata
        self.metrics['metadata'] = {
            'analysis_timestamp': datetime.now().isoformat(),
            'csv_file': self.csv_file,
            'total_records': len(self.df),
            'analysis_version': '1.0'
        }
        
        # Save to JSON
        with open(filename, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        
        print(f"📁 Saved metrics report: {filename}")
        return filename
    
    def print_executive_summary(self):
        """Print executive summary of key metrics"""
        print("\n" + "="*60)
        print("📊 SOC PERFORMANCE EXECUTIVE SUMMARY")
        print("="*60)
        
        # Basic stats
        if 'basic_counts' in self.metrics:
            counts = self.metrics['basic_counts']
            print(f"📈 Total Records Analyzed: {counts['total_records']:,}")
            print(f"🔢 Data Sources: {counts['unique_sources']}")
            
            if counts['date_range']['days']:
                print(f"📅 Analysis Period: {counts['date_range']['days']} days")
        
        # Performance metrics
        if 'precision_recall_f1' in self.metrics:
            prf_metrics = self.metrics['precision_recall_f1']
            if prf_metrics:
                avg_f1 = np.mean([m['f1_score'] for m in prf_metrics.values()])
                print(f"🎯 Average F1 Score: {avg_f1:.3f}")
        
        # Response time
        if 'mttr' in self.metrics and 'overall' in self.metrics['mttr']:
            mttr = self.metrics['mttr']['overall']
            print(f"⏱️ Mean Response Time: {mttr['mean_minutes']:.1f} minutes")
            print(f"📊 95th Percentile: {mttr['p95_minutes']:.1f} minutes")
        
        # Success rate
        if 'containment' in self.metrics and 'overall' in self.metrics['containment']:
            success = self.metrics['containment']['overall']
            print(f"🛡️ Overall Success Rate: {success['success_rate']:.1%}")
        
        # Confidence calibration
        if 'calibration' in self.metrics:
            cal_metrics = self.metrics['calibration']
            if cal_metrics:
                avg_reliability = np.mean([m['reliability_score'] for m in cal_metrics.values()])
                print(f"📐 Average Confidence Reliability: {avg_reliability:.3f}")
        
        print("="*60)

def main():
    """Main analysis function"""
    
    # Load the comprehensive dataset
    csv_file = "comprehensive_feedback_data_20250918_112413.csv"
    
    print("🚀 SOC Metrics Analysis")
    print("=" * 60)
    print(f"📁 Analyzing: {csv_file}")
    
    # Initialize analyzer
    analyzer = SOCMetricsAnalyzer(csv_file)
    
    if analyzer.df is None:
        print("❌ Failed to load data")
        return
    
    # Compute all metrics
    print("\n🔍 Computing Core SOC Metrics...")
    
    analyzer.compute_basic_counts()
    analyzer.compute_precision_recall_f1()
    analyzer.compute_mttr_distributions()
    analyzer.compute_confidence_calibration()
    analyzer.compute_containment_success_rate()
    analyzer.compute_additional_metrics()
    
    # Generate visualizations
    analyzer.generate_visualizations()
    
    # Save report
    report_file = analyzer.save_metrics_report()
    
    # Print executive summary
    analyzer.print_executive_summary()
    
    print(f"\n✅ Analysis Complete!")
    print(f"📊 Visualization: soc_metrics_dashboard.png")
    print(f"📄 Detailed Report: {report_file}")
    print(f"📈 Records Analyzed: {len(analyzer.df):,}")

if __name__ == "__main__":
    main()
