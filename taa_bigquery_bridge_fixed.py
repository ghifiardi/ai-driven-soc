#!/usr/bin/env python3
"""
Fixed TAA BigQuery Bridge Service
- Matches actual BigQuery schema
- Handles schema mismatches gracefully
- Writes to existing tables with correct field names
"""

import os
import json
import logging
import asyncio
from datetime import datetime
from google.cloud import pubsub_v1, bigquery
from google.cloud.exceptions import NotFound

# Configuration
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "chronicle-dev-2be9")
SUBSCRIPTION_ID = os.getenv("TAA_BRIDGE_SUBSCRIPTION", "enhanced-taa-results-sub")
COMPARISON_SUBSCRIPTION_ID = os.getenv("COMPARISON_SUBSCRIPTION", "taa-comparison-sub")

# BigQuery Configuration
DATASET_ID = "gatra_database"
TAA_TABLE_ID = "taa_state"  # Main TAA results table
COMPARISON_TABLE_ID = "taa_comparison"  # Comparison data table

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TAA-BigQuery-Bridge-Fixed")

class FixedTAABigQueryBridge:
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
        
        logger.info("🚀 Fixed TAA BigQuery Bridge initialized")
        logger.info(f"📊 Project: {PROJECT_ID}")
        logger.info(f"📊 Dataset: {DATASET_ID}")
        logger.info(f"📊 TAA Table: {TAA_TABLE_ID}")
        logger.info(f"📊 Comparison Table: {COMPARISON_TABLE_ID}")

    def _get_bigquery_schema(self, table_ref):
        """Get the actual BigQuery table schema"""
        try:
            table = self.bigquery_client.get_table(table_ref)
            return {field.name: field.field_type for field in table.schema}
        except NotFound:
            logger.warning(f"⚠️ Table {table_ref} not found, creating with default schema")
            return {}

    def _map_to_bigquery_schema(self, data, target_schema):
        """Map incoming data to BigQuery schema"""
        mapped_data = {}
        
        # Common field mappings
        field_mappings = {
            # Enhanced TAA fields -> BigQuery fields
            'alarm_id': 'alarm_id',
            'alert_data': 'alert_data',
            'attack_category': 'attack_category',
            'alert_severity': 'alert_severity',
            'source_ip': 'source_ip',
            'destination_ip': 'destination_ip',
            'port': 'port',
            'protocol': 'protocol',
            'timestamp': 'created_at',  # Map timestamp to created_at
            'enhanced_classification': 'classification',  # Map enhanced_classification to classification
            'enhanced_confidence': 'confidence',
            'threat_score': 'threat_score',
            'enhanced_is_anomaly': 'is_anomaly',
            'classification_reasoning': 'analysis_reasoning',
            'processed_by': 'processed_by',
            'processing_timestamp': 'updated_at'
        }
        
        # Map fields based on available schema
        for source_field, target_field in field_mappings.items():
            if source_field in data and target_field in target_schema:
                value = data[source_field]
                
                # Handle timestamp conversion
                if target_field in ['created_at', 'updated_at']:
                    if isinstance(value, str):
                        # Try to parse ISO format
                        try:
                            mapped_data[target_field] = datetime.fromisoformat(value.replace('Z', '+00:00'))
                        except:
                            mapped_data[target_field] = datetime.utcnow()
                    else:
                        mapped_data[target_field] = datetime.utcnow()
                else:
                    mapped_data[target_field] = value
        
        # Add default values for required fields
        if 'created_at' in target_schema and 'created_at' not in mapped_data:
            mapped_data['created_at'] = datetime.utcnow()
        if 'updated_at' in target_schema and 'updated_at' not in mapped_data:
            mapped_data['updated_at'] = datetime.utcnow()
            
        return mapped_data

    def _write_to_bigquery(self, table_ref, data, record_type="unknown"):
        """Write data to BigQuery with proper error handling"""
        try:
            # Get table schema
            schema = self._get_bigquery_schema(table_ref)
            
            # Map data to schema
            mapped_data = self._map_to_bigquery_schema(data, schema)
            
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
        
        logger.info("📊 Fixed Bridge Statistics - Uptime: {}".format(uptime))
        logger.info("   Enhanced Results Processed: {}".format(self.stats['enhanced_results_processed']))
        logger.info("   Comparisons Processed: {}".format(self.stats['comparisons_processed']))
        logger.info("   BigQuery Writes Successful: {}".format(self.stats['bigquery_writes_successful']))
        logger.info("   BigQuery Writes Failed: {}".format(self.stats['bigquery_writes_failed']))
        logger.info("   Success Rate: {:.1f}%".format(success_rate))

    async def start(self):
        """Start the bridge service"""
        logger.info("🚀 Starting Fixed TAA BigQuery Bridge...")
        
        # Start enhanced results subscriber
        enhanced_future = asyncio.create_task(
            self._subscriber_loop(self.subscription_path, self._process_enhanced_result, "Enhanced Results")
        )
        
        # Start comparison subscriber  
        comparison_future = asyncio.create_task(
            self._subscriber_loop(self.comparison_subscription_path, self._process_comparison_data, "Comparison")
        )
        
        # Start statistics logging
        stats_future = asyncio.create_task(self._statistics_loop())
        
        logger.info("✅ Fixed TAA BigQuery Bridge started successfully")
        
        # Wait for all tasks
        await asyncio.gather(enhanced_future, comparison_future, stats_future)

    async def _subscriber_loop(self, subscription_path, processor, subscriber_name):
        """Generic subscriber loop"""
        logger.info(f"📡 Starting {subscriber_name} subscriber...")
        
        def callback(message):
            processor(message)
        
        try:
            streaming_pull_future = self.subscriber.pull(
                request={"subscription": subscription_path, "max_messages": 10},
                callback=callback,
            )
            
            # Keep the subscriber running
            while True:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"❌ Error in {subscriber_name} subscriber: {e}")
            if 'streaming_pull_future' in locals():
                streaming_pull_future.cancel()
                try:
                    streaming_pull_future.result()
                except:
                    pass

    async def _statistics_loop(self):
        """Log statistics every 30 seconds"""
        while True:
            await asyncio.sleep(30)
            self._log_statistics()

def main():
    """Main function"""
    bridge = FixedTAABigQueryBridge()
    
    try:
        asyncio.run(bridge.start())
    except KeyboardInterrupt:
        logger.info("🛑 Bridge service stopped by user")
    except Exception as e:
        logger.error(f"❌ Bridge service error: {e}")

if __name__ == "__main__":
    main()
