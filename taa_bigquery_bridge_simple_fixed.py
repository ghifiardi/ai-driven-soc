#!/usr/bin/env python3
"""
Simple Fixed TAA BigQuery Bridge Service
- Uses synchronous Pub/Sub client (more reliable)
- Matches actual BigQuery schema
- Handles schema mismatches gracefully
"""

import os
import json
import logging
import time
from datetime import datetime
from google.cloud import pubsub_v1, bigquery
from google.cloud.exceptions import NotFound

# Configuration
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "chronicle-dev-2be9")
SUBSCRIPTION_ID = os.getenv("TAA_BRIDGE_SUBSCRIPTION", "enhanced-taa-results-sub")
COMPARISON_SUBSCRIPTION_ID = os.getenv("COMPARISON_SUBSCRIPTION", "taa-comparison-sub")

# BigQuery Configuration
DATASET_ID = "gatra_database"
TAA_TABLE_ID = "taa_state"
COMPARISON_TABLE_ID = "taa_comparison"

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TAA-Bridge-Simple-Fixed")

class SimpleFixedTAABridge:
    def __init__(self):
        self.project_id = PROJECT_ID
        self.subscriber = pubsub_v1.SubscriberClient()
        self.bigquery_client = bigquery.Client(project=PROJECT_ID)
        
        # Pub/Sub paths
        self.subscription_path = self.subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_ID)
        self.comparison_subscription_path = self.subscriber.subscription_path(PROJECT_ID, COMPARISON_SUBSCRIPTION_ID)
        
        # BigQuery paths
        self.dataset_ref = self.bigquery_client.dataset(DATASET_ID)
        self.taa_table_ref = self.dataset_ref.table(TAA_TABLE_ID)
        self.comparison_table_ref = self.dataset_ref.table(COMPARISON_TABLE_ID)
        
        # Statistics
        self.stats = {
            'enhanced_results_processed': 0,
            'comparisons_processed': 0,
            'bigquery_writes_successful': 0,
            'bigquery_writes_failed': 0,
            'start_time': datetime.utcnow()
        }
        
        logger.info("🚀 Simple Fixed TAA BigQuery Bridge initialized")
        logger.info(f"📊 Project: {PROJECT_ID}")
        logger.info(f"📊 Dataset: {DATASET_ID}")

    def _map_to_bigquery_schema(self, data):
        """Map incoming data to BigQuery schema"""
        mapped_data = {}
        
        # Field mappings for TAA table
        field_mappings = {
            'alarm_id': 'alarm_id',
            'alert_data': 'alert_data',
            'attack_category': 'attack_category',
            'alert_severity': 'alert_severity',
            'source_ip': 'source_ip',
            'destination_ip': 'destination_ip',
            'port': 'port',
            'protocol': 'protocol',
            'enhanced_classification': 'classification',
            'enhanced_confidence': 'confidence',
            'threat_score': 'threat_score',
            'enhanced_is_anomaly': 'is_anomaly',
            'classification_reasoning': 'analysis_reasoning',
            'processed_by': 'processed_by'
        }
        
        # Map fields
        for source_field, target_field in field_mappings.items():
            if source_field in data:
                mapped_data[target_field] = data[source_field]
        
        # Add timestamps
        mapped_data['created_at'] = datetime.utcnow()
        mapped_data['updated_at'] = datetime.utcnow()
        
        return mapped_data

    def _write_to_bigquery(self, table_ref, data, record_type="unknown"):
        """Write data to BigQuery"""
        try:
            # Map data to schema
            mapped_data = self._map_to_bigquery_schema(data)
            
            if not mapped_data:
                logger.warning(f"⚠️ No data to write for {record_type}")
                return False
            
            # Insert into BigQuery
            errors = self.bigquery_client.insert_rows_json(table_ref, [mapped_data])
            
            if errors:
                logger.error(f"❌ Error writing {record_type} to BigQuery: {errors}")
                self.stats['bigquery_writes_failed'] += 1
                return False
            else:
                logger.info(f"✅ Successfully wrote {record_type} to BigQuery")
                self.stats['bigquery_writes_successful'] += 1
                return True
                
        except Exception as e:
            logger.error(f"❌ Exception writing {record_type} to BigQuery: {e}")
            self.stats['bigquery_writes_failed'] += 1
            return False

    def _process_enhanced_result(self, message):
        """Process enhanced TAA result"""
        try:
            data = json.loads(message.data.decode('utf-8'))
            alarm_id = data.get('alarm_id', 'unknown')
            
            logger.info(f"📨 Processing enhanced result: {alarm_id}")
            
            # Write to main TAA table
            success = self._write_to_bigquery(self.taa_table_ref, data, f"enhanced_result_{alarm_id}")
            
            if success:
                self.stats['enhanced_results_processed'] += 1
            
            message.ack()
            
        except Exception as e:
            logger.error(f"❌ Error processing enhanced result: {e}")
            message.nack()

    def _process_comparison_data(self, message):
        """Process comparison data"""
        try:
            data = json.loads(message.data.decode('utf-8'))
            alarm_id = data.get('alarm_id', 'unknown')
            
            logger.info(f"📊 Processing comparison data: {alarm_id}")
            
            # Write to comparison table
            success = self._write_to_bigquery(self.comparison_table_ref, data, f"comparison_{alarm_id}")
            
            if success:
                self.stats['comparisons_processed'] += 1
            
            message.ack()
            
        except Exception as e:
            logger.error(f"❌ Error processing comparison data: {e}")
            message.nack()

    def _log_statistics(self):
        """Log bridge statistics"""
        uptime = datetime.utcnow() - self.stats['start_time']
        total_writes = self.stats['bigquery_writes_successful'] + self.stats['bigquery_writes_failed']
        success_rate = (self.stats['bigquery_writes_successful'] / total_writes * 100) if total_writes > 0 else 0
        
        logger.info("📊 Simple Fixed Bridge Statistics - Uptime: {}".format(uptime))
        logger.info("   Enhanced Results Processed: {}".format(self.stats['enhanced_results_processed']))
        logger.info("   Comparisons Processed: {}".format(self.stats['comparisons_processed']))
        logger.info("   BigQuery Writes Successful: {}".format(self.stats['bigquery_writes_successful']))
        logger.info("   BigQuery Writes Failed: {}".format(self.stats['bigquery_writes_failed']))
        logger.info("   Success Rate: {:.1f}%".format(success_rate))

    def run(self):
        """Run the bridge service"""
        logger.info("🚀 Starting Simple Fixed TAA BigQuery Bridge...")
        
        # Enhanced results callback
        def enhanced_callback(message):
            self._process_enhanced_result(message)
        
        # Comparison callback
        def comparison_callback(message):
            self._process_comparison_data(message)
        
        # Start enhanced results subscriber
        enhanced_future = self.subscriber.pull(
            request={"subscription": self.subscription_path, "max_messages": 10},
            callback=enhanced_callback,
        )
        
        # Start comparison subscriber
        comparison_future = self.subscriber.pull(
            request={"subscription": self.comparison_subscription_path, "max_messages": 10},
            callback=comparison_callback,
        )
        
        logger.info("✅ Simple Fixed TAA BigQuery Bridge started successfully")
        
        # Statistics logging loop
        try:
            while True:
                time.sleep(30)
                self._log_statistics()
        except KeyboardInterrupt:
            logger.info("🛑 Bridge service stopped by user")
            enhanced_future.cancel()
            comparison_future.cancel()
        except Exception as e:
            logger.error(f"❌ Bridge service error: {e}")
            enhanced_future.cancel()
            comparison_future.cancel()

def main():
    """Main function"""
    bridge = SimpleFixedTAABridge()
    bridge.run()

if __name__ == "__main__":
    main()


