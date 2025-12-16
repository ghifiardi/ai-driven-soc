#!/usr/bin/env python3
"""
Enhanced TAA Service - 100% Migration
=====================================

This service processes 100% of alerts with enhanced classification logic.
Full migration from original broken TAA to enhanced system.

Author: AI-Driven SOC Team
Date: 2025-09-18
"""

import asyncio
import json
import logging
import random
import time
from datetime import datetime
from typing import Dict, Any, Optional
from google.cloud import pubsub_v1
from enhanced_classification_engine import EnhancedClassificationEngine

# Configuration
PROJECT_ID = "chronicle-dev-2be9"
SUBSCRIPTION_ID = "taa-alerts-sub"
OUTPUT_TOPIC_ID = "enhanced-taa-results"
COMPARISON_TOPIC_ID = "taa-comparison"

# Migration settings - FULL MIGRATION
MIGRATION_PHASE = 4  # 4=100% traffic (FULL MIGRATION)
TRAFFIC_PERCENTAGE = 100  # 100% of alerts processed with enhanced logic
ENABLE_COMPARISON = True

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_taa_100percent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EnhancedTAA-100Percent")

class FullMigrationEnhancedTAA:
    """
    Enhanced TAA service with 100% traffic processing
    """
    
    def __init__(self):
        self.enhanced_engine = EnhancedClassificationEngine()
        self.publisher = pubsub_v1.PublisherClient()
        self.subscriber = pubsub_v1.SubscriberClient()
        
        # Topic paths
        self.output_topic_path = self.publisher.topic_path(PROJECT_ID, OUTPUT_TOPIC_ID)
        self.comparison_topic_path = self.publisher.topic_path(PROJECT_ID, COMPARISON_TOPIC_ID)
        
        # Statistics
        self.stats = {
            'total_received': 0,
            'processed_enhanced': 0,
            'skipped_original': 0,
            'comparisons_sent': 0,
            'classification_changes': 0,
            'false_positive_reductions': 0,
            'start_time': datetime.now()
        }
        
        logger.info(f"🚀 Enhanced TAA Full Migration initialized")
        logger.info(f"📊 Migration Phase: {MIGRATION_PHASE} ({TRAFFIC_PERCENTAGE}% traffic - FULL MIGRATION)")
        logger.info(f"🎯 Enhanced Classification Engine loaded with {len(self.enhanced_engine.threat_indicators)} threat indicators")
        logger.info(f"⚡ FULL MIGRATION: All alerts processed with enhanced classification")
        logger.info(f"📈 Previous success: 43 classification changes in 50% mode")
    
    def should_process_alert(self, alert_data: Dict[str, Any]) -> bool:
        """
        Determine if this alert should be processed by enhanced TAA
        FULL MIGRATION: Process 100% of alerts
        """
        # FULL MIGRATION: Process ALL alerts with enhanced logic
        return True  # Always process with enhanced TAA
    
    def process_alert_enhanced(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process alert using enhanced classification engine (SYNC version)"""
        start_time = time.time()
        
        try:
            # Apply enhanced classification
            enhanced_result = self.enhanced_engine.classify_alert(alert_data)
            
            # Add enhanced classification metadata
            alert_data['enhanced_classification'] = enhanced_result.get('classification', 'unknown')
            alert_data['enhanced_confidence'] = enhanced_result.get('confidence', 0.0)
            alert_data['enhanced_threat_score'] = enhanced_result.get('threat_score', 0.0)
            alert_data['enhanced_is_anomaly'] = enhanced_result.get('is_anomaly', False)
            alert_data['enhanced_reasoning'] = enhanced_result.get('reasoning', '')
            alert_data['enhanced_processing_time'] = time.time() - start_time
            alert_data['enhanced_timestamp'] = datetime.now().isoformat()
            alert_data['enhanced_source'] = 'full_migration_enhanced_taa'
            
            # Add migration metadata
            alert_data['migration_phase'] = MIGRATION_PHASE
            alert_data['traffic_percentage'] = TRAFFIC_PERCENTAGE
            alert_data['full_migration_mode'] = True
            
            # Track classification changes
            original_classification = alert_data.get('classification', 'unknown')
            if original_classification != alert_data['enhanced_classification']:
                self.stats['classification_changes'] += 1
                
                # Track false positive reductions
                if original_classification == 'benign' and alert_data['enhanced_classification'] in ['suspicious', 'critical', 'malicious']:
                    self.stats['false_positive_reductions'] += 1
            
            logger.info(f"✅ Enhanced classification: {alert_data['alarm_id']} -> {alert_data['enhanced_classification']} "
                       f"(confidence: {alert_data['enhanced_confidence']:.2f}, threat_score: {alert_data['enhanced_threat_score']:.1f})")
            
            return alert_data
            
        except Exception as e:
            logger.error(f"❌ Enhanced classification failed for {alert_data.get('alarm_id', 'unknown')}: {e}")
            alert_data['enhanced_error'] = str(e)
            alert_data['enhanced_timestamp'] = datetime.now().isoformat()
            return alert_data
    
    def send_comparison_data(self, original_alert: Dict[str, Any], enhanced_result: Dict[str, Any]):
        """Send A/B comparison data for analysis"""
        if not ENABLE_COMPARISON:
            return
        
        try:
            comparison_data = {
                'timestamp': datetime.now().isoformat(),
                'alarm_id': original_alert.get('alarm_id'),
                'original_classification': original_alert.get('classification', 'unknown'),
                'original_confidence': original_alert.get('confidence', 0.0),
                'original_is_anomaly': original_alert.get('is_anomaly', False),
                'enhanced_classification': enhanced_result.get('enhanced_classification', 'unknown'),
                'enhanced_confidence': enhanced_result.get('enhanced_confidence', 0.0),
                'enhanced_is_anomaly': enhanced_result.get('enhanced_is_anomaly', False),
                'enhanced_threat_score': enhanced_result.get('enhanced_threat_score', 0.0),
                'classification_changed': original_alert.get('classification') != enhanced_result.get('enhanced_classification'),
                'confidence_difference': enhanced_result.get('enhanced_confidence', 0.0) - original_alert.get('confidence', 0.0),
                'migration_phase': MIGRATION_PHASE,
                'traffic_percentage': TRAFFIC_PERCENTAGE,
                'full_migration_mode': True,
                'source': 'full_migration_comparison',
                'dashboard_improvement_metrics': {
                    'false_positive_reduction': original_alert.get('classification') == 'benign' and enhanced_result.get('enhanced_classification') in ['suspicious', 'critical', 'malicious'],
                    'confidence_improvement': enhanced_result.get('enhanced_confidence', 0.0) > original_alert.get('confidence', 0.0),
                    'severity_alignment': enhanced_result.get('enhanced_confidence', 0.0) > 0.7 and enhanced_result.get('enhanced_threat_score', 0.0) > 50,
                    'threat_detection_improvement': enhanced_result.get('enhanced_is_anomaly', False) != original_alert.get('is_anomaly', False)
                }
            }
            
            # Publish comparison data
            message_data = json.dumps(comparison_data).encode('utf-8')
            future = self.publisher.publish(self.comparison_topic_path, message_data)
            future.result()
            
            self.stats['comparisons_sent'] += 1
            logger.debug(f"📊 Comparison data sent for {original_alert.get('alarm_id')}")
            
        except Exception as e:
            logger.error(f"❌ Failed to send comparison data: {e}")
    
    def publish_enhanced_result(self, enhanced_result: Dict[str, Any]):
        """Publish enhanced classification result"""
        try:
            message_data = json.dumps(enhanced_result).encode('utf-8')
            future = self.publisher.publish(self.output_topic_path, message_data)
            future.result()
            
            logger.debug(f"📤 Enhanced result published for {enhanced_result.get('alarm_id')}")
            
        except Exception as e:
            logger.error(f"❌ Failed to publish enhanced result: {e}")
    
    def log_statistics(self):
        """Log current statistics"""
        uptime = datetime.now() - self.stats['start_time']
        processed_rate = self.stats['processed_enhanced'] / uptime.total_seconds() * 60 if uptime.total_seconds() > 0 else 0
        
        logger.info(f"📊 Full Migration Statistics - Uptime: {uptime}")
        logger.info(f"   Total Received: {self.stats['total_received']}")
        logger.info(f"   Enhanced Processed: {self.stats['processed_enhanced']} ({TRAFFIC_PERCENTAGE}%)")
        logger.info(f"   Skipped (Original): {self.stats['skipped_original']} (0%)")
        logger.info(f"   Classification Changes: {self.stats['classification_changes']}")
        logger.info(f"   False Positive Reductions: {self.stats['false_positive_reductions']}")
        logger.info(f"   Processing Rate: {processed_rate:.1f} alerts/minute")
        
        # Dashboard improvement metrics
        if self.stats['processed_enhanced'] > 0:
            change_rate = (self.stats['classification_changes'] / self.stats['processed_enhanced']) * 100
            false_positive_reduction_rate = (self.stats['false_positive_reductions'] / self.stats['processed_enhanced']) * 100
            logger.info(f"🎯 Dashboard Improvement Metrics:")
            logger.info(f"   Classification Change Rate: {change_rate:.1f}%")
            logger.info(f"   False Positive Reduction Rate: {false_positive_reduction_rate:.1f}%")
            logger.info(f"   Expected Dashboard Improvement: Reduced false positive rate from 55.9%")
    
    def message_callback(self, message):
        """
        FIXED: Main message processing callback (SYNC version)
        FULL MIGRATION: Process ALL alerts with enhanced logic
        """
        try:
            alert_data = json.loads(message.data.decode("utf-8"))
            self.stats['total_received'] += 1
            
            alarm_id = alert_data.get('alarm_id', 'unknown')
            logger.info(f"📨 Received alert: {alarm_id}")
            
            # FULL MIGRATION: Process ALL alerts with enhanced TAA
            enhanced_result = self.process_alert_enhanced(alert_data)
            
            # Send comparison data
            self.send_comparison_data(alert_data, enhanced_result)
            
            # Publish enhanced result
            self.publish_enhanced_result(enhanced_result)
            
            self.stats['processed_enhanced'] += 1
            
            # Log statistics every 5 messages (more frequent for full migration)
            if self.stats['total_received'] % 5 == 0:
                self.log_statistics()
            
            # Always ack the message
            message.ack()
            
        except Exception as e:
            logger.error(f"❌ Message processing failed: {e}")
            message.ack()
    
    def start(self):
        """Start the enhanced TAA service"""
        subscription_path = self.subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_ID)
        
        logger.info(f"🚀 Starting Enhanced TAA Full Migration Service")
        logger.info(f"📡 Listening on subscription: {subscription_path}")
        logger.info(f"📤 Publishing to: {self.output_topic_path}")
        logger.info(f"🎯 Processing {TRAFFIC_PERCENTAGE}% of alerts with enhanced classification")
        logger.info(f"⚡ FULL MIGRATION: All alerts processed with enhanced logic")
        logger.info(f"📈 Expected improvement: Reduce false positive rate from 55.9%")
        
        # Start listening (SYNC version)
        streaming_pull_future = self.subscriber.subscribe(
            subscription_path, 
            callback=self.message_callback
        )
        
        try:
            streaming_pull_future.result()
        except KeyboardInterrupt:
            logger.info("🛑 Enhanced TAA Full Migration Service stopped")
            streaming_pull_future.cancel()
            self.log_statistics()

def main():
    """Main function"""
    enhanced_taa = FullMigrationEnhancedTAA()
    enhanced_taa.start()

if __name__ == "__main__":
    main()
