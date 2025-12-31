#!/usr/bin/env python3
"""
Enhanced TAA Agent with Fixed Classification Logic
=================================================

Replaces the broken supervised_v1 model with the enhanced classification engine
that properly utilizes all available threat intelligence data.
"""

import pandas as pd
import numpy as np
import json
import logging
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import uuid
import time

# Import our enhanced classification engine
from enhanced_classification_engine import EnhancedClassificationEngine, ThreatLevel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class AlertData:
    """Standardized alert data structure"""
    alarm_id: str
    timestamp: str
    source_ip: Optional[str] = None
    destination_ip: Optional[str] = None
    protocol: Optional[str] = None
    alert_severity: Optional[str] = None
    attack_category: Optional[str] = None
    raw_alert_data: Optional[Dict[str, Any]] = None
    enrichment_data: Optional[Dict[str, Any]] = None

@dataclass
class ClassificationResult:
    """Enhanced classification result"""
    alarm_id: str
    classification: str
    confidence: float
    threat_score: float
    is_anomaly: bool
    reasoning: List[str]
    threat_breakdown: Dict[str, float]
    processing_time: float
    enhanced_classification: bool = True

class EnhancedTAAgent:
    """Enhanced Triage and Analysis Agent with fixed classification logic"""
    
    def __init__(self, config_file: str = "config/taa_a2a_config.json"):
        """Initialize the Enhanced TAA Agent"""
        self.config = self._load_config(config_file)
        self.classification_engine = EnhancedClassificationEngine()
        self.processing_stats = {
            'total_processed': 0,
            'anomalies_detected': 0,
            'critical_alerts': 0,
            'malicious_alerts': 0,
            'suspicious_alerts': 0,
            'processing_times': []
        }
        
        logger.info("🚀 Enhanced TAA Agent initialized")
        logger.info(f"📊 Classification thresholds loaded")
        logger.info(f"🎯 Enhanced classification engine ready")
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration file"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load config file {config_file}: {e}")
            return {
                "vertex_ai": {"model": "gemini-2.0-flash-exp"},
                "pubsub": {"alerts_topic": "ada-alerts", "feedback_topic": "taa-feedback"},
                "monitoring": {"enabled": True}
            }
    
    async def analyze_alert(self, alert_data: Dict[str, Any]) -> ClassificationResult:
        """Analyze a single alert with enhanced classification"""
        start_time = time.time()
        
        try:
            # Extract alert data
            alarm_id = alert_data.get('alarm_id', str(uuid.uuid4()))
            
            # Prepare alert data for classification
            alert_for_classification = {
                'alarm_id': alarm_id,
                'timestamp': alert_data.get('timestamp', datetime.now().isoformat()),
                'source_ip': alert_data.get('source_ip'),
                'destination_ip': alert_data.get('destination_ip'),
                'protocol': alert_data.get('protocol'),
                'alert_severity': alert_data.get('alert_severity'),
                'attack_category': alert_data.get('attack_category'),
                'confidence': alert_data.get('confidence', 0.5),
                'processing_time': alert_data.get('processing_time', 0.0),
                'raw_alert_data': alert_data
            }
            
            # Apply enhanced classification
            classification_result = self.classification_engine.classify_alert(alert_for_classification)
            
            processing_time = time.time() - start_time
            
            # Create result
            result = ClassificationResult(
                alarm_id=alarm_id,
                classification=classification_result['classification'],
                confidence=classification_result['confidence'],
                threat_score=classification_result['threat_score'],
                is_anomaly=classification_result.get('is_anomaly', False),
                reasoning=classification_result.get('reasoning', []),
                threat_breakdown=classification_result.get('threat_breakdown', {}),
                processing_time=processing_time
            )
            
            # Update statistics
            self._update_stats(result)
            
            logger.info(f"🎯 Alert {alarm_id}: {result.classification} "
                       f"(threat_score: {result.threat_score:.1f}, "
                       f"confidence: {result.confidence:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing alert {alarm_id}: {e}")
            
            # Return safe default
            return ClassificationResult(
                alarm_id=alarm_id,
                classification='benign',
                confidence=0.1,
                threat_score=0.0,
                is_anomaly=False,
                reasoning=[f"Error in analysis: {str(e)}"],
                threat_breakdown={},
                processing_time=time.time() - start_time,
                enhanced_classification=False
            )
    
    async def analyze_batch(self, alerts: List[Dict[str, Any]]) -> List[ClassificationResult]:
        """Analyze a batch of alerts"""
        logger.info(f"🔄 Processing batch of {len(alerts)} alerts...")
        
        results = []
        for alert in alerts:
            result = await self.analyze_alert(alert)
            results.append(result)
        
        logger.info(f"✅ Batch processing complete: {len(results)} alerts analyzed")
        return results
    
    def _update_stats(self, result: ClassificationResult):
        """Update processing statistics"""
        self.processing_stats['total_processed'] += 1
        self.processing_stats['processing_times'].append(result.processing_time)
        
        if result.is_anomaly:
            self.processing_stats['anomalies_detected'] += 1
        
        if result.classification == 'critical':
            self.processing_stats['critical_alerts'] += 1
        elif result.classification == 'malicious':
            self.processing_stats['malicious_alerts'] += 1
        elif result.classification == 'suspicious':
            self.processing_stats['suspicious_alerts'] += 1
    
    async def process_historical_data(self, csv_file: str, sample_size: Optional[int] = None) -> pd.DataFrame:
        """Process historical data with enhanced classification"""
        logger.info(f"📊 Processing historical data from {csv_file}")
        
        try:
            # Load data
            df = pd.read_csv(csv_file)
            
            if sample_size:
                df = df.head(sample_size)
                logger.info(f"📋 Processing sample of {len(df)} records")
            
            # Convert to alert format
            alerts = []
            for idx, row in df.iterrows():
                alert_data = {
                    'alarm_id': row.get('alarm_id', f"historical_{idx}"),
                    'timestamp': row.get('timestamp', ''),
                    'source_ip': row.get('source_ip'),
                    'destination_ip': row.get('destination_ip'),
                    'protocol': row.get('protocol'),
                    'alert_severity': row.get('alert_severity'),
                    'attack_category': row.get('attack_category'),
                    'confidence': row.get('confidence', 0.2),  # Use original low confidence
                    'processing_time': row.get('processing_time', 0.0),
                    'raw_alert_data': row.to_dict()
                }
                alerts.append(alert_data)
            
            # Process with enhanced classification
            results = await self.analyze_batch(alerts)
            
            # Create enhanced dataframe
            enhanced_data = []
            for idx, (original_row, result) in enumerate(zip(df.iterrows(), results)):
                enhanced_row = df.iloc[idx].copy()
                enhanced_row['enhanced_classification'] = result.classification
                enhanced_row['enhanced_confidence'] = result.confidence
                enhanced_row['threat_score'] = result.threat_score
                enhanced_row['enhanced_is_anomaly'] = result.is_anomaly
                enhanced_row['classification_reasoning'] = '; '.join(result.reasoning)
                enhanced_row['enhanced_processing_time'] = result.processing_time
                enhanced_data.append(enhanced_row)
            
            enhanced_df = pd.DataFrame(enhanced_data)
            
            # Generate processing summary
            self._generate_processing_summary(enhanced_df)
            
            return enhanced_df
            
        except Exception as e:
            logger.error(f"Error processing historical data: {e}")
            raise
    
    def _generate_processing_summary(self, df: pd.DataFrame):
        """Generate summary of processing results"""
        logger.info("📊 Enhanced TAA Processing Summary:")
        
        # Classification distribution
        if 'enhanced_classification' in df.columns:
            classification_counts = df['enhanced_classification'].value_counts()
            for classification, count in classification_counts.items():
                percentage = (count / len(df)) * 100
                logger.info(f"   {classification}: {count:,} alerts ({percentage:.1f}%)")
        
        # Anomaly detection
        if 'enhanced_is_anomaly' in df.columns:
            anomaly_count = df['enhanced_is_anomaly'].sum()
            logger.info(f"🚨 Total Anomalies Detected: {anomaly_count:,} ({anomaly_count/len(df)*100:.1f}%)")
        
        # Threat score distribution
        if 'threat_score' in df.columns:
            high_threat = df[df['threat_score'] >= 6.0]
            medium_threat = df[(df['threat_score'] >= 4.0) & (df['threat_score'] < 6.0)]
            low_threat = df[(df['threat_score'] >= 2.0) & (df['threat_score'] < 4.0)]
            
            logger.info(f"🎯 Threat Score Distribution:")
            logger.info(f"   High Threat (≥6.0): {len(high_threat):,} alerts")
            logger.info(f"   Medium Threat (4.0-5.9): {len(medium_threat):,} alerts")
            logger.info(f"   Low Threat (2.0-3.9): {len(low_threat):,} alerts")
        
        # Performance metrics
        if 'enhanced_processing_time' in df.columns:
            avg_time = df['enhanced_processing_time'].mean()
            max_time = df['enhanced_processing_time'].max()
            logger.info(f"⏱️ Performance Metrics:")
            logger.info(f"   Average Processing Time: {avg_time:.4f}s")
            logger.info(f"   Maximum Processing Time: {max_time:.4f}s")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        stats = self.processing_stats.copy()
        
        if stats['processing_times']:
            stats['avg_processing_time'] = np.mean(stats['processing_times'])
            stats['max_processing_time'] = np.max(stats['processing_times'])
        else:
            stats['avg_processing_time'] = 0.0
            stats['max_processing_time'] = 0.0
        
        # Calculate detection rates
        if stats['total_processed'] > 0:
            stats['anomaly_detection_rate'] = stats['anomalies_detected'] / stats['total_processed']
            stats['critical_detection_rate'] = stats['critical_alerts'] / stats['total_processed']
            stats['malicious_detection_rate'] = stats['malicious_alerts'] / stats['total_processed']
        else:
            stats['anomaly_detection_rate'] = 0.0
            stats['critical_detection_rate'] = 0.0
            stats['malicious_detection_rate'] = 0.0
        
        return stats
    
    async def compare_with_original(self, original_file: str, enhanced_file: str):
        """Compare original vs enhanced classification results"""
        logger.info("🔍 Comparing Original vs Enhanced Classifications...")
        
        try:
            original_df = pd.read_csv(original_file)
            enhanced_df = pd.read_csv(enhanced_file)
            
            # Ensure same number of records
            min_records = min(len(original_df), len(enhanced_df))
            original_df = original_df.head(min_records)
            enhanced_df = enhanced_df.head(min_records)
            
            print("\n" + "="*60)
            print("📊 CLASSIFICATION COMPARISON REPORT")
            print("="*60)
            
            # Original classification
            print("🔴 Original (Broken) Classification:")
            original_counts = original_df['classification'].value_counts()
            for classification, count in original_counts.items():
                percentage = (count / len(original_df)) * 100
                print(f"   {classification}: {count:,} alerts ({percentage:.1f}%)")
            
            # Enhanced classification
            print("\n🟢 Enhanced (Fixed) Classification:")
            enhanced_counts = enhanced_df['enhanced_classification'].value_counts()
            for classification, count in enhanced_counts.items():
                percentage = (count / len(enhanced_df)) * 100
                print(f"   {classification}: {count:,} alerts ({percentage:.1f}%)")
            
            # Anomaly detection comparison
            if 'is_anomaly' in original_df.columns and 'enhanced_is_anomaly' in enhanced_df.columns:
                original_anomalies = original_df['is_anomaly'].sum()
                enhanced_anomalies = enhanced_df['enhanced_is_anomaly'].sum()
                
                print(f"\n🚨 Anomaly Detection Comparison:")
                print(f"   Original: {original_anomalies:,} anomalies ({original_anomalies/len(original_df)*100:.1f}%)")
                print(f"   Enhanced: {enhanced_anomalies:,} anomalies ({enhanced_anomalies/len(enhanced_df)*100:.1f}%)")
                print(f"   Improvement: {enhanced_anomalies - original_anomalies:,} additional threats detected")
            
            # Threat detection improvement
            if 'threat_score' in enhanced_df.columns:
                high_threat = enhanced_df[enhanced_df['threat_score'] >= 6.0]
                print(f"\n🎯 New Threat Detection Capabilities:")
                print(f"   High Threat Alerts: {len(high_threat):,}")
                print(f"   Average Threat Score: {enhanced_df['threat_score'].mean():.2f}")
                print(f"   Max Threat Score: {enhanced_df['threat_score'].max():.2f}")
            
            print("\n💡 Key Improvements:")
            print("   ✅ Proper utilization of attack categories")
            print("   ✅ Severity-based threat scoring")
            print("   ✅ IP reputation analysis")
            print("   ✅ Behavioral anomaly detection")
            print("   ✅ Comprehensive threat reasoning")
            print("   ✅ Dynamic confidence scoring")
            
            print("="*60)
            
        except Exception as e:
            logger.error(f"Error comparing classifications: {e}")

async def main():
    """Main function to demonstrate Enhanced TAA Agent"""
    print("🚀 Enhanced TAA Agent Demo")
    print("=" * 60)
    
    # Initialize the agent
    agent = EnhancedTAAgent()
    
    # Process historical data
    try:
        print("📊 Processing historical TAA data with enhanced classification...")
        
        enhanced_df = await agent.process_historical_data(
            "taa_extracted_data_20250918_115033.csv", 
            sample_size=1000
        )
        
        # Save enhanced results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        enhanced_output = f"enhanced_taa_results_{timestamp}.csv"
        enhanced_df.to_csv(enhanced_output, index=False)
        
        print(f"💾 Enhanced TAA results saved to: {enhanced_output}")
        
        # Compare with original
        await agent.compare_with_original(
            "taa_extracted_data_20250918_115033.csv",
            enhanced_output
        )
        
        # Show agent statistics
        stats = agent.get_statistics()
        print(f"\n📈 Agent Performance Statistics:")
        print(f"   Total Processed: {stats['total_processed']:,}")
        print(f"   Anomalies Detected: {stats['anomalies_detected']:,}")
        print(f"   Critical Alerts: {stats['critical_alerts']:,}")
        print(f"   Malicious Alerts: {stats['malicious_alerts']:,}")
        print(f"   Suspicious Alerts: {stats['suspicious_alerts']:,}")
        print(f"   Anomaly Detection Rate: {stats['anomaly_detection_rate']:.1%}")
        print(f"   Average Processing Time: {stats['avg_processing_time']:.4f}s")
        
        print("\n✅ Enhanced TAA Agent Demo Complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Demo failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())


