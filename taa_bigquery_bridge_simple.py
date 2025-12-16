#!/usr/bin/env python3
"""
Simplified TAA BigQuery Bridge Service
======================================

This service writes enhanced TAA results to BigQuery using only the existing
table schemas. It maps enhanced data to the columns that actually exist.

Author: AI-Driven SOC Team
Date: 2025-09-18
"""

import json
import logging
import time
from datetime import datetime
from typing import Dict, Any
from google.cloud import bigquery, pubsub_v1

# Configuration
PROJECT_ID = "chronicle-dev-2be9"
ENHANCED_RESULTS_TOPIC = "enhanced-taa-results"
COMPARISON_TOPIC = "taa-comparison"
ENHANCED_RESULTS_SUBSCRIPTION = "enhanced-taa-results-sub"
COMPARISON_SUBSCRIPTION = "taa-comparison-sub"

# BigQuery table IDs
TAA_STATE_TABLE = f"{PROJECT_ID}.gatra_database.taa_state"
PROCESSED_ADA_ALERTS_TABLE = f"{PROJECT_ID}.gatra_database.processed_ada_alerts"

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('taa_bigquery_bridge_simple.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TAA-BigQuery-Bridge-Simple")

class SimpleTAABigQueryBridge:
    """
    Simplified bridge service that writes to existing BigQuery table schemas
    """
    
    def __init__(self):
        self.bq_client = bigquery.Client(project=PROJECT_ID)
        self.subscriber = pubsub_v1.SubscriberClient()
        
        # Subscription paths
        self.enhanced_results_subscription_path = self.subscriber.subscription_path(
            PROJECT_ID, ENHANCED_RESULTS_SUBSCRIPTION
        )
        self.comparison_subscription_path = self.subscriber.subscription_path(
            PROJECT_ID, COMPARISON_SUBSCRIPTION
        )
        
        # Statistics
        self.stats = {
            'enhanced_results_processed': 0,
            'comparisons_processed': 0,
            'bigquery_writes_successful': 0,
            'bigquery_writes_failed': 0,
            'start_time': datetime.now()
        }
        
        logger.info(f"🚀 Simple TAA BigQuery Bridge initialized")
        logger.info(f"📊 Project: {PROJECT_ID}")
        logger.info(f"📡 Listening to enhanced results: {self.enhanced_results_subscription_path}")
        logger.info(f"📡 Listening to comparisons: {self.comparison_subscription_path}")
    
    def create_pubsub_subscriptions(self):
        """Create Pub/Sub subscriptions if they don't exist"""
        try:
            # Create enhanced results subscription
            enhanced_results_topic_path = f"projects/{PROJECT_ID}/topics/{ENHANCED_RESULTS_TOPIC}"
            try:
                self.subscriber.create_subscription(
                    request={
                        "name": self.enhanced_results_subscription_path,
                        "topic": enhanced_results_topic_path
                    }
                )
                logger.info(f"✅ Created enhanced results subscription")
            except Exception as e:
                if "already exists" in str(e):
                    logger.info(f"✅ Enhanced results subscription already exists")
                else:
                    logger.error(f"❌ Error creating enhanced results subscription: {e}")
            
            # Create comparison subscription
            comparison_topic_path = f"projects/{PROJECT_ID}/topics/{COMPARISON_TOPIC}"
            try:
                self.subscriber.create_subscription(
                    request={
                        "name": self.comparison_subscription_path,
                        "topic": comparison_topic_path
                    }
                )
                logger.info(f"✅ Created comparison subscription")
            except Exception as e:
                if "already exists" in str(e):
                    logger.info(f"✅ Comparison subscription already exists")
                else:
                    logger.error(f"❌ Error creating comparison subscription: {e}")
                    
        except Exception as e:
            logger.error(f"❌ Error creating subscriptions: {e}")
    
    def write_enhanced_result_to_bigquery(self, enhanced_result: Dict[str, Any]) -> bool:
        """Write enhanced TAA result to BigQuery using existing schema"""
        try:
            # Map enhanced data to existing taa_state table columns
            row = {
                "alarmId": enhanced_result.get('alarm_id', 'unknown'),
                "timestamp": datetime.now().isoformat(),
                "events": json.dumps({
                    "enhanced_classification": enhanced_result.get('enhanced_classification', 'unknown'),
                    "enhanced_confidence": enhanced_result.get('enhanced_confidence', 0.0),
                    "enhanced_threat_score": enhanced_result.get('enhanced_threat_score', 0.0),
                    "enhanced_is_anomaly": enhanced_result.get('enhanced_is_anomaly', False),
                    "enhanced_reasoning": enhanced_result.get('enhanced_reasoning', ''),
                    "source": enhanced_result.get('enhanced_source', 'enhanced_taa'),
                    "migration_phase": enhanced_result.get('migration_phase', 3),
                    "traffic_percentage": enhanced_result.get('traffic_percentage', 50),
                    "processing_time": enhanced_result.get('enhanced_processing_time', 0.0),
                    "alert_severity": enhanced_result.get('alert_severity', 'LOW'),
                    "source_ip": enhanced_result.get('source_ip', 'Unknown'),
                    "destination_ip": enhanced_result.get('destination_ip', 'Unknown'),
                    "protocol": enhanced_result.get('protocol', 'Unknown'),
                    "attack_category": enhanced_result.get('attack_category', 'Network Flow')
                }),
                "processed_by_taa": True
            }
            
            # Insert into taa_state table
            table = self.bq_client.get_table(TAA_STATE_TABLE)
            errors = self.bq_client.insert_rows_json(table, [row])
            
            if not errors:
                self.stats['bigquery_writes_successful'] += 1
                logger.info(f"✅ Enhanced result written to BigQuery: {enhanced_result.get('alarm_id')}")
                return True
            else:
                self.stats['bigquery_writes_failed'] += 1
                logger.error(f"❌ Error writing enhanced result to BigQuery: {errors}")
                return False
                
        except Exception as e:
            self.stats['bigquery_writes_failed'] += 1
            logger.error(f"❌ Exception writing enhanced result to BigQuery: {e}")
            return False
    
    def write_comparison_to_bigquery(self, comparison_data: Dict[str, Any]) -> bool:
        """Write comparison data to BigQuery using existing schema"""
        try:
            # Map comparison data to existing processed_ada_alerts table columns
            row = {
                "alert_id": comparison_data.get('alarm_id', 'unknown'),
                "timestamp": datetime.now().isoformat(),
                "confidence_score": float(comparison_data.get('enhanced_confidence', 0.0)),
                "classification": comparison_data.get('enhanced_classification', 'unknown'),
                "raw_alert": json.dumps({
                    "original_classification": comparison_data.get('original_classification', 'unknown'),
                    "original_confidence": comparison_data.get('original_confidence', 0.0),
                    "enhanced_classification": comparison_data.get('enhanced_classification', 'unknown'),
                    "enhanced_confidence": comparison_data.get('enhanced_confidence', 0.0),
                    "classification_changed": comparison_data.get('classification_changed', False),
                    "confidence_difference": comparison_data.get('confidence_difference', 0.0),
                    "migration_phase": comparison_data.get('migration_phase', 3),
                    "traffic_percentage": comparison_data.get('traffic_percentage', 50),
                    "source": comparison_data.get('source', 'taa_bigquery_bridge_simple')
                })
            }
            
            # Insert into processed_ada_alerts table
            table = self.bq_client.get_table(PROCESSED_ADA_ALERTS_TABLE)
            errors = self.bq_client.insert_rows_json(table, [row])
            
            if not errors:
                self.stats['bigquery_writes_successful'] += 1
                logger.info(f"✅ Comparison data written to BigQuery: {comparison_data.get('alarm_id')}")
                return True
            else:
                self.stats['bigquery_writes_failed'] += 1
                logger.error(f"❌ Error writing comparison to BigQuery: {errors}")
                return False
                
        except Exception as e:
            self.stats['bigquery_writes_failed'] += 1
            logger.error(f"❌ Exception writing comparison to BigQuery: {e}")
            return False
    
    def enhanced_results_callback(self, message):
        """Process enhanced TAA results from Pub/Sub"""
        try:
            enhanced_result = json.loads(message.data.decode("utf-8"))
            self.stats['enhanced_results_processed'] += 1
            
            alarm_id = enhanced_result.get('alarm_id', 'unknown')
            logger.info(f"📨 Processing enhanced result: {alarm_id}")
            
            # Write to BigQuery
            success = self.write_enhanced_result_to_bigquery(enhanced_result)
            
            if success:
                logger.info(f"✅ Enhanced result processed successfully: {alarm_id}")
            else:
                logger.error(f"❌ Failed to process enhanced result: {alarm_id}")
            
            # Always ack the message
            message.ack()
            
        except Exception as e:
            logger.error(f"❌ Error processing enhanced result: {e}")
            message.ack()
    
    def comparison_callback(self, message):
        """Process comparison data from Pub/Sub"""
        try:
            comparison_data = json.loads(message.data.decode("utf-8"))
            self.stats['comparisons_processed'] += 1
            
            alarm_id = comparison_data.get('alarm_id', 'unknown')
            logger.info(f"📊 Processing comparison data: {alarm_id}")
            
            # Write to BigQuery
            success = self.write_comparison_to_bigquery(comparison_data)
            
            if success:
                logger.info(f"✅ Comparison data processed successfully: {alarm_id}")
            else:
                logger.error(f"❌ Failed to process comparison data: {alarm_id}")
            
            # Always ack the message
            message.ack()
            
        except Exception as e:
            logger.error(f"❌ Error processing comparison data: {e}")
            message.ack()
    
    def log_statistics(self):
        """Log current statistics"""
        uptime = datetime.now() - self.stats['start_time']
        
        logger.info(f"📊 Simple Bridge Statistics - Uptime: {uptime}")
        logger.info(f"   Enhanced Results Processed: {self.stats['enhanced_results_processed']}")
        logger.info(f"   Comparisons Processed: {self.stats['comparisons_processed']}")
        logger.info(f"   BigQuery Writes Successful: {self.stats['bigquery_writes_successful']}")
        logger.info(f"   BigQuery Writes Failed: {self.stats['bigquery_writes_failed']}")
        
        total_writes = self.stats['bigquery_writes_successful'] + self.stats['bigquery_writes_failed']
        if total_writes > 0:
            success_rate = (self.stats['bigquery_writes_successful'] / total_writes) * 100
            logger.info(f"   Success Rate: {success_rate:.1f}%")
    
    def start(self):
        """Start the bridge service"""
        logger.info(f"🚀 Starting Simple TAA BigQuery Bridge Service")
        
        # Create subscriptions
        self.create_pubsub_subscriptions()
        
        # Start listening to enhanced results
        enhanced_results_future = self.subscriber.subscribe(
            self.enhanced_results_subscription_path,
            callback=self.enhanced_results_callback
        )
        
        # Start listening to comparisons
        comparison_future = self.subscriber.subscribe(
            self.comparison_subscription_path,
            callback=self.comparison_callback
        )
        
        logger.info(f"📡 Simple bridge service listening for messages...")
        
        try:
            # Log statistics every 30 seconds
            while True:
                time.sleep(30)
                self.log_statistics()
                
        except KeyboardInterrupt:
            logger.info("🛑 Simple TAA BigQuery Bridge Service stopped")
            enhanced_results_future.cancel()
            comparison_future.cancel()
            self.log_statistics()

def main():
    """Main function"""
    bridge = SimpleTAABigQueryBridge()
    bridge.start()

if __name__ == "__main__":
    main()


