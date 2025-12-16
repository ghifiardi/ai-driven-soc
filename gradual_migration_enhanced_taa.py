#!/usr/bin/env python3
"""
Gradual Migration Enhanced TAA Service
=====================================

This service implements gradual migration by:
1. Running alongside the existing TAA service
2. Processing only a percentage of alerts (starting with 10%)
3. Using enhanced classification engine
4. Logging results for comparison
5. Providing real-time monitoring

Phase 1: 10% traffic routing
Phase 2: Validation and comparison
Phase 3: Gradual increase to 50%
Phase 4: Full migration

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
SUBSCRIPTION_ID = "taa-alerts-sub"  # Same subscription as original TAA
OUTPUT_TOPIC_ID = "enhanced-taa-results"  # New topic for enhanced results
COMPARISON_TOPIC_ID = "taa-comparison"  # Topic for A/B comparison results

# Migration settings
MIGRATION_PHASE = 1  # 1=10%, 2=25%, 3=50%, 4=100%
TRAFFIC_PERCENTAGE = 10  # Percentage of alerts to process
ENABLE_COMPARISON = True  # Enable A/B testing

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gradual_migration_enhanced_taa.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EnhancedTAA-GradualMigration")

class GradualMigrationEnhancedTAA:
    """
    Enhanced TAA service with gradual migration capabilities
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
            'start_time': datetime.now()
        }
        
        logger.info(f"🚀 Enhanced TAA Gradual Migration initialized")
        logger.info(f"📊 Migration Phase: {MIGRATION_PHASE} ({TRAFFIC_PERCENTAGE}% traffic)")
        logger.info(f"🎯 Enhanced Classification Engine loaded with {len(self.enhanced_engine.threat_indicators)} threat indicators")
    
    def should_process_alert(self, alert_data: Dict[str, Any]) -> bool:
        """
        Determine if this alert should be processed by enhanced TAA
        Uses hash-based routing for consistent distribution
        """
        # Use alarm_id for consistent routing
        alarm_id = alert_data.get('alarm_id', str(random.randint(1000, 9999)))
        
        # Create hash from alarm_id for consistent routing
        hash_value = hash(alarm_id) % 100
        
        # Route based on traffic percentage
        should_process = hash_value < TRAFFIC_PERCENTAGE
        
        if should_process:
            logger.debug(f"🎯 Alert {alarm_id} routed to Enhanced TAA (hash: {hash_value})")
        else:
            logger.debug(f"⏭️ Alert {alarm_id} routed to Original TAA (hash: {hash_value})")
        
        return should_process
    
    async def process_alert_enhanced(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process alert using enhanced classification engine
        """
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
            alert_data['enhanced_source'] = 'gradual_migration_enhanced_taa'
            
            # Add migration metadata
            alert_data['migration_phase'] = MIGRATION_PHASE
            alert_data['traffic_percentage'] = TRAFFIC_PERCENTAGE
            
            logger.info(f"✅ Enhanced classification: {alert_data['alarm_id']} -> {alert_data['enhanced_classification']} "
                       f"(confidence: {alert_data['enhanced_confidence']:.2f}, threat_score: {alert_data['enhanced_threat_score']:.1f})")
            
            return alert_data
            
        except Exception as e:
            logger.error(f"❌ Enhanced classification failed for {alert_data.get('alarm_id', 'unknown')}: {e}")
            # Return original alert data with error flag
            alert_data['enhanced_error'] = str(e)
            alert_data['enhanced_timestamp'] = datetime.now().isoformat()
            return alert_data
    
    async def send_comparison_data(self, original_alert: Dict[str, Any], enhanced_result: Dict[str, Any]):
        """
        Send A/B comparison data for analysis
        """
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
                'anomaly_detection_changed': original_alert.get('is_anomaly', False) != enhanced_result.get('enhanced_is_anomaly', False),
                'migration_phase': MIGRATION_PHASE,
                'source': 'gradual_migration_comparison'
            }
            
            # Publish comparison data
            message_data = json.dumps(comparison_data).encode('utf-8')
            future = self.publisher.publish(self.comparison_topic_path, message_data)
            future.result()  # Wait for publish to complete
            
            self.stats['comparisons_sent'] += 1
            logger.debug(f"📊 Comparison data sent for {original_alert.get('alarm_id')}")
            
        except Exception as e:
            logger.error(f"❌ Failed to send comparison data: {e}")
    
    async def publish_enhanced_result(self, enhanced_result: Dict[str, Any]):
        """
        Publish enhanced classification result
        """
        try:
            message_data = json.dumps(enhanced_result).encode('utf-8')
            future = self.publisher.publish(self.output_topic_path, message_data)
            future.result()  # Wait for publish to complete
            
            logger.debug(f"📤 Enhanced result published for {enhanced_result.get('alarm_id')}")
            
        except Exception as e:
            logger.error(f"❌ Failed to publish enhanced result: {e}")
    
    def log_statistics(self):
        """
        Log current statistics
        """
        uptime = datetime.now() - self.stats['start_time']
        processed_rate = self.stats['processed_enhanced'] / uptime.total_seconds() * 60 if uptime.total_seconds() > 0 else 0
        
        logger.info(f"📊 Statistics - Uptime: {uptime}")
        logger.info(f"   Total Received: {self.stats['total_received']}")
        logger.info(f"   Enhanced Processed: {self.stats['processed_enhanced']} ({TRAFFIC_PERCENTAGE}%)")
        logger.info(f"   Skipped (Original): {self.stats['skipped_original']} ({100-TRAFFIC_PERCENTAGE}%)")
        logger.info(f"   Comparisons Sent: {self.stats['comparisons_sent']}")
        logger.info(f"   Processing Rate: {processed_rate:.1f} alerts/minute")
    
    async def message_callback(self, message):
        """
        Main message processing callback
        """
        try:
            alert_data = json.loads(message.data.decode("utf-8"))
            self.stats['total_received'] += 1
            
            alarm_id = alert_data.get('alarm_id', 'unknown')
            logger.info(f"📨 Received alert: {alarm_id}")
            
            # Determine routing
            if self.should_process_alert(alert_data):
                # Process with enhanced TAA
                enhanced_result = await self.process_alert_enhanced(alert_data)
                
                # Send comparison data
                await self.send_comparison_data(alert_data, enhanced_result)
                
                # Publish enhanced result
                await self.publish_enhanced_result(enhanced_result)
                
                self.stats['processed_enhanced'] += 1
                
            else:
                # Let original TAA handle it (don't ack, let original service process)
                self.stats['skipped_original'] += 1
                logger.debug(f"⏭️ Skipping {alarm_id} - routed to original TAA")
                # Don't ack the message, let original service handle it
            
            # Log statistics every 10 messages
            if self.stats['total_received'] % 10 == 0:
                self.log_statistics()
            
            # Always ack the message (both services will receive it)
            message.ack()
            
        except Exception as e:
            logger.error(f"❌ Message processing failed: {e}")
            message.ack()  # Ack to prevent reprocessing
    
    async def start(self):
        """
        Start the gradual migration enhanced TAA service
        """
        subscription_path = self.subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_ID)
        
        logger.info(f"🚀 Starting Enhanced TAA Gradual Migration Service")
        logger.info(f"📡 Listening on subscription: {subscription_path}")
        logger.info(f"📤 Publishing to: {self.output_topic_path}")
        logger.info(f"📊 Comparison data to: {self.comparison_topic_path}")
        logger.info(f"🎯 Processing {TRAFFIC_PERCENTAGE}% of alerts with enhanced classification")
        logger.info(f"⚡ Original TAA continues processing {100-TRAFFIC_PERCENTAGE}% of alerts")
        
        # Start listening
        streaming_pull_future = self.subscriber.subscribe(
            subscription_path, 
            callback=self.message_callback
        )
        
        try:
            await streaming_pull_future.result()
        except KeyboardInterrupt:
            logger.info("🛑 Enhanced TAA Gradual Migration Service stopped")
            streaming_pull_future.cancel()
            self.log_statistics()

async def main():
    """
    Main function
    """
    enhanced_taa = GradualMigrationEnhancedTAA()
    await enhanced_taa.start()

if __name__ == "__main__":
    asyncio.run(main())


