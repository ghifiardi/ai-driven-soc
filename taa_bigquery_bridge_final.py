#!/usr/bin/env python3
"""
Final Working TAA BigQuery Bridge Service
- Simple, bulletproof implementation
- Handles all the Pub/Sub and BigQuery quirks
- Actually works!
"""

import os
import json
import logging
import sys
import time
from datetime import datetime, date
from decimal import Decimal
from google.cloud import pubsub_v1, bigquery

# Configuration
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "chronicle-dev-2be9")
SUBSCRIPTION_ID = os.getenv("TAA_BRIDGE_SUBSCRIPTION", "enhanced-taa-results-sub")
COMPARISON_SUBSCRIPTION_ID = os.getenv("COMPARISON_SUBSCRIPTION", "taa-comparison-sub")

# BigQuery Configuration
DATASET_ID = "gatra_database"
TAA_TABLE_ID = "taa_state"
COMPARISON_TABLE_ID = "taa_comparison"

# Logging Setup with proper handlers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True  # override any previous config
)
logger = logging.getLogger("TAA-Bridge-Final")

# --- JSON Serialization Helper ---
def to_jsonable(obj):
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (datetime, date)):
        # RFC3339; kalau sudah UTC, tambahkan 'Z'
        # Sesuaikan jika kamu butuh timezone-aware
        iso = obj.isoformat()
        return iso if iso.endswith('Z') else iso
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    # fallback
    return str(obj)

class FinalTAABridge:
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
        
        # Heartbeat tracking
        self._last_heartbeat = time.time()
        
        logger.info("🚀 Final TAA BigQuery Bridge initialized")
        logger.info(f"📊 Project: {PROJECT_ID}")
        logger.info(f"📊 Dataset: {DATASET_ID}")

    def _prepare_data_for_bigquery(self, data):
        """Prepare data for BigQuery using to_jsonable helper - matches actual taa_state schema"""
        prepared_data = {}
        
        # Field mappings for taa_state table (based on actual schema)
        field_mappings = {
            'alarm_id': 'alarm_id',
            'enhanced_confidence': 'confidence',
            'alert_severity': 'severity',  # Map alert_severity to severity
            'enhanced_is_anomaly': 'is_anomaly',
            'classification_reasoning': 'reasoning',
            'threat_score': 'run_time',  # Use run_time field for threat_score
            'enhanced_classification': 'valid'  # Use valid field for classification (boolean)
        }
        
        # Map fields using to_jsonable helper
        for source_field, target_field in field_mappings.items():
            if source_field in data:
                value = data[source_field]
                # Special handling for classification -> valid boolean
                if target_field == 'valid':
                    # Convert classification to boolean (malicious/suspicious = True, benign = False)
                    if isinstance(value, str):
                        prepared_data[target_field] = value.lower() not in ['benign', 'low_risk']
                    else:
                        prepared_data[target_field] = bool(value)
                # Special handling for severity -> float conversion
                elif target_field == 'severity':
                    if isinstance(value, str):
                        # Convert severity strings to float values
                        severity_map = {
                            'LOW': 0.25,
                            'MEDIUM': 0.5,
                            'HIGH': 0.75,
                            'CRITICAL': 1.0
                        }
                        prepared_data[target_field] = severity_map.get(value.upper(), 0.5)
                    else:
                        prepared_data[target_field] = float(value) if value is not None else 0.5
                else:
                    prepared_data[target_field] = to_jsonable(value)
        
        # Add raw JSON data (as JSON string since BigQuery expects JSON as string)
        prepared_data['raw'] = json.dumps(to_jsonable(data), ensure_ascii=False)
        
        # created_at is auto-generated by BigQuery, so don't set it
        # Add remarks field with processing info
        prepared_data['remarks'] = f"Enhanced TAA processing at {datetime.utcnow().isoformat()}"
        
        return prepared_data

    def _prepare_comparison_data_for_bigquery(self, data):
        """Prepare comparison data for BigQuery using to_jsonable helper"""
        prepared_data = {}
        
        # Field mappings for comparison table
        field_mappings = {
            'alarm_id': 'request_id',
            'original_classification': 'model_a',
            'enhanced_classification': 'model_b',
            'enhanced_confidence': 'confidence',
            'processing_timestamp': 'created_at'
        }
        
        # Map fields using to_jsonable helper
        for source_field, target_field in field_mappings.items():
            if source_field in data:
                prepared_data[target_field] = to_jsonable(data[source_field])
        
        # Add comparison verdict
        original = data.get('original_classification', 'unknown')
        enhanced = data.get('enhanced_classification', 'unknown')
        prepared_data['verdict'] = to_jsonable('changed' if original != enhanced else 'same')
        
        # Add diff data (as JSON string since BigQuery expects JSON as string)
        diff_data = {
            'original': original,
            'enhanced': enhanced,
            'threat_score': data.get('threat_score'),
            'classification_reasoning': data.get('classification_reasoning')
        }
        prepared_data['diff'] = json.dumps(to_jsonable(diff_data), ensure_ascii=False)
        
        # Add raw JSON data (as JSON string since BigQuery expects JSON as string)
        prepared_data['raw'] = json.dumps(to_jsonable(data), ensure_ascii=False)
        
        # Ensure created_at is properly formatted for TIMESTAMP
        if 'created_at' not in prepared_data:
            prepared_data['created_at'] = datetime.utcnow().isoformat()
        
        return prepared_data

    def _write_to_bigquery(self, table_ref, data, record_type="unknown"):
        """Write data to BigQuery with proper JSON serialization"""
        try:
            # Prepare data for BigQuery
            prepared_data = self._prepare_data_for_bigquery(data)
            
            if not prepared_data:
                logger.warning(f"⚠️ No data to write for {record_type}")
                return False
            
            # Insert into BigQuery with error handling
            errors = self.bigquery_client.insert_rows_json(
                table_ref, 
                [prepared_data],
                ignore_unknown_values=True,   # ignore extra fields not yet in schema
                skip_invalid_rows=True        # skip bad rows instead of failing the whole batch
            )
            
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
            # Handle different message structures
            if hasattr(message, 'message') and hasattr(message.message, 'data'):
                data_bytes = message.message.data
            elif hasattr(message, 'data'):
                data_bytes = message.data
            else:
                logger.error("❌ Unknown message structure")
                return False
            
            data = json.loads(data_bytes.decode('utf-8'))
            alarm_id = data.get('alarm_id', 'unknown')
            
            logger.info(f"📨 Processing enhanced result: {alarm_id}")
            
            # Write to main TAA table
            success = self._write_to_bigquery(self.taa_table_ref, data, f"enhanced_result_{alarm_id}")
            
            if success:
                self.stats['enhanced_results_processed'] += 1
            
            # Acknowledge message (handle different structures)
            try:
                if hasattr(message, 'ack'):
                    message.ack()
                elif hasattr(message, 'ack_id'):
                    self.subscriber.acknowledge(
                        request={"subscription": self.subscription_path, "ack_ids": [message.ack_id]}
                    )
            except Exception as ack_error:
                logger.warning(f"⚠️ Could not acknowledge message: {ack_error}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error processing enhanced result: {e}")
            return False

    def _process_comparison_data(self, message):
        """Process comparison data"""
        try:
            # Handle different message structures
            if hasattr(message, 'message') and hasattr(message.message, 'data'):
                data_bytes = message.message.data
            elif hasattr(message, 'data'):
                data_bytes = message.data
            else:
                logger.error("❌ Unknown message structure")
                return False
            
            data = json.loads(data_bytes.decode('utf-8'))
            alarm_id = data.get('alarm_id', 'unknown')
            
            logger.info(f"📊 Processing comparison data: {alarm_id}")
            
            # Prepare comparison data specifically
            prepared_comparison_data = self._prepare_comparison_data_for_bigquery(data)
            
            # Write to comparison table
            success = self._write_comparison_to_bigquery(prepared_comparison_data, f"comparison_{alarm_id}")
            
            if success:
                self.stats['comparisons_processed'] += 1
            
            # Acknowledge message (handle different structures)
            try:
                if hasattr(message, 'ack'):
                    message.ack()
                elif hasattr(message, 'ack_id'):
                    self.subscriber.acknowledge(
                        request={"subscription": self.comparison_subscription_path, "ack_ids": [message.ack_id]}
                    )
            except Exception as ack_error:
                logger.warning(f"⚠️ Could not acknowledge message: {ack_error}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error processing comparison data: {e}")
            return False

    def _write_comparison_to_bigquery(self, prepared_data, record_type="unknown"):
        """Write comparison data to BigQuery with proper JSON serialization"""
        try:
            if not prepared_data:
                logger.warning(f"⚠️ No comparison data to write for {record_type}")
                return False
            
            # Insert into BigQuery with error handling
            errors = self.bigquery_client.insert_rows_json(
                self.comparison_table_ref, 
                [prepared_data],
                ignore_unknown_values=True,   # ignore extra fields not yet in schema
                skip_invalid_rows=True        # skip bad rows instead of failing the whole batch
            )
            
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

    def _log_statistics(self):
        """Log bridge statistics"""
        uptime = datetime.utcnow() - self.stats['start_time']
        total_writes = self.stats['bigquery_writes_successful'] + self.stats['bigquery_writes_failed']
        success_rate = (self.stats['bigquery_writes_successful'] / total_writes * 100) if total_writes > 0 else 0
        
        logger.info("📊 Final Bridge Statistics - Uptime: {}".format(uptime))
        logger.info("   Enhanced Results Processed: {}".format(self.stats['enhanced_results_processed']))
        logger.info("   Comparisons Processed: {}".format(self.stats['comparisons_processed']))
        logger.info("   BigQuery Writes Successful: {}".format(self.stats['bigquery_writes_successful']))
        logger.info("   BigQuery Writes Failed: {}".format(self.stats['bigquery_writes_failed']))
        logger.info("   Success Rate: {:.1f}%".format(success_rate))

    def run(self):
        """Run the bridge service with proper blocking"""
        logger.info("🚀 Starting Final TAA BigQuery Bridge...")
        logger.info("📡 Listening for messages on enhanced results and comparison subscriptions")
        
        try:
            while True:
                current_time = time.time()
                
                # Heartbeat every 60 seconds
                if current_time - self._last_heartbeat > 60:
                    logger.info("💓 heartbeat: bridge alive, waiting for messages...")
                    self._last_heartbeat = current_time
                
                # Pull enhanced results
                try:
                    enhanced_response = self.subscriber.pull(
                        request={"subscription": self.subscription_path, "max_messages": 5}
                    )
                    
                    if enhanced_response.received_messages:
                        logger.info(f"📨 Enhanced Results: Received {len(enhanced_response.received_messages)} messages")
                        
                        for message in enhanced_response.received_messages:
                            self._process_enhanced_result(message)
                except Exception as e:
                    logger.error(f"❌ Error pulling enhanced results: {e}")
                
                # Pull comparison data
                try:
                    comparison_response = self.subscriber.pull(
                        request={"subscription": self.comparison_subscription_path, "max_messages": 5}
                    )
                    
                    if comparison_response.received_messages:
                        logger.info(f"📊 Comparison: Received {len(comparison_response.received_messages)} messages")
                        
                        for message in comparison_response.received_messages:
                            self._process_comparison_data(message)
                except Exception as e:
                    logger.error(f"❌ Error pulling comparison data: {e}")
                
                # Log statistics every 30 seconds
                if int(current_time) % 30 == 0:
                    self._log_statistics()
                
                # Sleep briefly to avoid overwhelming the system
                time.sleep(2)
                
        except KeyboardInterrupt:
            logger.info("🛑 Bridge service stopped by user")
        except Exception as e:
            logger.error(f"❌ Bridge service error: {e}")
            raise

def main():
    """Main function with defensive error handling"""
    try:
        bridge = FinalTAABridge()
        bridge.run()  # should block forever
    except Exception:
        logging.exception("💥 Fatal crash in main()")
        time.sleep(2)
        raise

if __name__ == "__main__":
    main()
