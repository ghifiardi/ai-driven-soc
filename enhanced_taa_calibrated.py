#!/usr/bin/env python3
"""
Enhanced TAA with Calibrated Classification Engine
- Fixes confidence miscalibration (83.9% -> realistic levels)
- Improves severity scoring (0.392 -> proper distribution)
- Implements suppression policies for noise reduction
"""

import os
import json
import logging
import asyncio
import random
from datetime import datetime
from google.cloud import pubsub_v1
from enhanced_classification_engine_calibrated import CalibratedEnhancedClassificationEngine

# Configuration
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "chronicle-dev-2be9")
SUBSCRIPTION_ID = os.getenv("TAA_ALERTS_SUBSCRIPTION", "taa-alerts-sub")
ENHANCED_RESULTS_TOPIC_ID = os.getenv("ENHANCED_RESULTS_TOPIC", "enhanced-taa-results")
COMPARISON_TOPIC_ID = os.getenv("COMPARISON_TOPIC", "taa-comparison")

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Enhanced-TAA-Calibrated")

class CalibratedEnhancedTAA:
    def __init__(self):
        self.project_id = PROJECT_ID
        self.publisher = pubsub_v1.PublisherClient()
        self.subscriber = pubsub_v1.SubscriberClient()
        
        # Pub/Sub paths
        self.subscription_path = self.subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_ID)
        self.enhanced_results_topic_path = self.publisher.topic_path(PROJECT_ID, ENHANCED_RESULTS_TOPIC_ID)
        self.comparison_topic_path = self.publisher.topic_path(PROJECT_ID, COMPARISON_TOPIC_ID)
        
        # Initialize calibrated classification engine
        self.classification_engine = CalibratedEnhancedClassificationEngine()
        
        # Statistics
        self.stats = {
            'alerts_received': 0,
            'alerts_processed': 0,
            'alerts_suppressed': 0,
            'classification_changes': 0,
            'false_positive_reductions': 0,
            'start_time': datetime.utcnow()
        }
        
        logger.info("🚀 Calibrated Enhanced TAA initialized")
        logger.info(f"📊 Project: {PROJECT_ID}")
        logger.info(f"📊 Calibrated Engine: Ready with confidence calibration")

    def _publish_message(self, topic_path, data: dict):
        """Publish message to Pub/Sub"""
        try:
            message_json = json.dumps(data)
            message_bytes = message_json.encode("utf-8")
            self.publisher.publish(topic_path, message_bytes).result()
            return True
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
            return False

    def _process_alert_with_calibrated_logic(self, alert_data: dict):
        """Process alert with calibrated classification engine"""
        try:
            # Apply calibrated classification
            classification_result = self.classification_engine.classify_alert(alert_data)
            
            # Check if suppression is recommended
            if classification_result.get('suppression_recommended', False):
                self.stats['alerts_suppressed'] += 1
                logger.info(f"🚫 Alert {alert_data.get('alarm_id', 'N/A')} suppressed (confidence: {classification_result.get('enhanced_confidence', 0):.2f}, severity: {classification_result.get('severity_score', 0):.2f})")
                return None  # Suppress this alert
            
            # Update alert data with calibrated results
            alert_data.update({
                'enhanced_classification': classification_result['enhanced_classification'],
                'enhanced_confidence': classification_result['enhanced_confidence'],
                'raw_confidence': classification_result.get('raw_confidence', 0),
                'threat_score': classification_result['threat_score'],
                'enhanced_is_anomaly': classification_result['enhanced_is_anomaly'],
                'severity_score': classification_result.get('severity_score', 0),
                'classification_reasoning': classification_result['classification_reasoning'],
                'processed_by': 'calibrated_enhanced_taa',
                'processing_timestamp': datetime.utcnow().isoformat() + 'Z',
                'calibrated': True
            })
            
            # Track classification changes
            original_classification = alert_data.get('classification', 'unknown')
            if original_classification != classification_result['enhanced_classification']:
                self.stats['classification_changes'] += 1
                
                # Track false positive reductions
                if original_classification == 'benign' and classification_result['enhanced_classification'] in ['malicious', 'critical']:
                    self.stats['false_positive_reductions'] += 1
            
            logger.info(f"🎯 Alert {alert_data.get('alarm_id', 'N/A')}: {classification_result['enhanced_classification']} "
                        f"(calibrated_confidence: {classification_result['enhanced_confidence']:.2f}, "
                        f"severity: {classification_result.get('severity_score', 0):.2f}, "
                        f"threat_score: {classification_result['threat_score']:.1f})")
            
            return alert_data
            
        except Exception as e:
            logger.error(f"❌ Calibrated classification failed for {alert_data.get('alarm_id', 'N/A')}: {e}")
            return None

    def _log_statistics(self):
        """Log calibrated TAA statistics"""
        uptime = datetime.utcnow() - self.stats['start_time']
        suppression_rate = (self.stats['alerts_suppressed'] / self.stats['alerts_received'] * 100) if self.stats['alerts_received'] > 0 else 0
        
        logger.info("📊 Calibrated Enhanced TAA Statistics - Uptime: {}".format(uptime))
        logger.info("   Alerts Received: {}".format(self.stats['alerts_received']))
        logger.info("   Alerts Processed: {}".format(self.stats['alerts_processed']))
        logger.info("   Alerts Suppressed: {} ({:.1f}%)".format(self.stats['alerts_suppressed'], suppression_rate))
        logger.info("   Classification Changes: {}".format(self.stats['classification_changes']))
        logger.info("   False Positive Reductions: {}".format(self.stats['false_positive_reductions']))
        logger.info("🎯 Calibrated Performance Metrics:")
        logger.info("   Expected False Positive Rate: 40-50% (down from 66.7%)")
        logger.info("   Expected Confidence Calibration: 45-65% (down from 83.9%)")
        logger.info("   Expected Severity Distribution: 0.3-0.8 (up from 0.392)")

    def callback(self, message):
        """Process incoming alert message"""
        try:
            self.stats['alerts_received'] += 1
            
            # Handle different message formats
            if hasattr(message, 'message'):
                # Standard Pub/Sub message format
                message_data = message.message.data.decode('utf-8')
                ack_id = message.ack_id
            else:
                # Direct message format
                message_data = message.data.decode('utf-8')
                ack_id = getattr(message, 'ack_id', None)
            
            alert_data = json.loads(message_data)
            
            # Process with calibrated logic
            processed_alert = self._process_alert_with_calibrated_logic(alert_data)
            
            if processed_alert:
                # Publish enhanced result
                self._publish_message(self.enhanced_results_topic_path, processed_alert)
                
                # Publish comparison data
                comparison_data = {
                    'alarm_id': processed_alert.get('alarm_id'),
                    'timestamp': processed_alert.get('timestamp'),
                    'original_classification': alert_data.get('classification', 'unknown'),
                    'original_confidence': alert_data.get('confidence', 0.0),
                    'enhanced_classification': processed_alert.get('enhanced_classification'),
                    'enhanced_confidence': processed_alert.get('enhanced_confidence'),
                    'raw_confidence': processed_alert.get('raw_confidence'),
                    'threat_score': processed_alert.get('threat_score'),
                    'severity_score': processed_alert.get('severity_score'),
                    'processed_by_calibrated': True,
                    'suppression_recommended': processed_alert.get('suppression_recommended', False),
                    'processing_timestamp': datetime.utcnow().isoformat() + 'Z'
                }
                self._publish_message(self.comparison_topic_path, comparison_data)
                
                self.stats['alerts_processed'] += 1
            
            # Acknowledge message
            if ack_id:
                self.subscriber.acknowledge(
                    request={"subscription": self.subscription_path, "ack_ids": [ack_id]}
                )
            
        except Exception as e:
            logger.error(f"❌ Error processing message: {e}")
            # Try to acknowledge even on error to avoid reprocessing
            try:
                if hasattr(message, 'ack_id') and message.ack_id:
                    self.subscriber.acknowledge(
                        request={"subscription": self.subscription_path, "ack_ids": [message.ack_id]}
                    )
            except:
                pass

    async def start(self):
        """Start the calibrated enhanced TAA"""
        logger.info("🚀 Starting Calibrated Enhanced TAA...")
        
        logger.info("✅ Calibrated Enhanced TAA started successfully")
        logger.info("📡 Listening for alerts with calibrated classification...")
        
        # Main processing loop
        try:
            while True:
                # Pull messages synchronously
                response = self.subscriber.pull(
                    request={"subscription": self.subscription_path, "max_messages": 10}
                )
                
                if response.received_messages:
                    for message in response.received_messages:
                        self.callback(message)
                
                # Statistics logging every minute
                await asyncio.sleep(60)
                self._log_statistics()
                
        except KeyboardInterrupt:
            logger.info("🛑 Calibrated Enhanced TAA stopped by user")
        except Exception as e:
            logger.error(f"❌ Error in main loop: {e}")

def main():
    """Main function"""
    taa = CalibratedEnhancedTAA()
    
    try:
        asyncio.run(taa.start())
    except KeyboardInterrupt:
        logger.info("🛑 Calibrated Enhanced TAA stopped by user")
    except Exception as e:
        logger.error(f"❌ Calibrated Enhanced TAA error: {e}")

if __name__ == "__main__":
    main()
