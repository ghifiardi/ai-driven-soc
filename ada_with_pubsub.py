#!/usr/bin/env python3
"""
Enhanced ADA Agent with Pub/Sub Publishing - Connects ADA to TAA
"""

import os
import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
from google.cloud import bigquery, pubsub_v1

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionTunedBigQueryClient:
    """Production-tuned BigQuery client with enterprise-appropriate sensitivity"""
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.client = bigquery.Client(project=project_id)
        
        # Production-tuned thresholds (significantly increased)
        self.volume_thresholds = {
            'internal_to_internal': 5000000000,   # 5GB for internal transfers (backups, sync)
            'internal_to_external': 1000000000,   # 1GB for external transfers (cloud backup)
            'external_to_internal': 500000000,    # 500MB for external incoming
            'external_to_external': 100000000     # 100MB for external-external
        }
        
        # Business hours definition (reduce sensitivity during business hours)
        self.business_hours = list(range(8, 18))  # 8 AM to 6 PM
        
        # Known business processes (whitelist patterns)
        self.business_patterns = {
            'backup_hours': list(range(22, 24)) + list(range(0, 6)),  # 10 PM - 6 AM
            'sync_hours': list(range(6, 8)) + list(range(18, 20)),    # 6-8 AM, 6-8 PM
            'standard_ports': [25, 53, 80, 443, 993, 995, 587, 143, 110, 8080, 8443]
        }
        
    def fetch_real_siem_events(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Fetch real SIEM events (same as before)"""
        try:
            query = f"""
                SELECT 
                    alarmId,
                    events
                FROM `{self.project_id}.gatra_database.siem_events`
                WHERE events IS NOT NULL 
                    AND (processed_by_ada IS NULL OR processed_by_ada = FALSE)
                ORDER BY alarmId DESC
                LIMIT {limit}
            """
            
            query_job = self.client.query(query)
            results = query_job.result()
            
            alerts = []
            for row in results:
                try:
                    events_data = row.events
                    if isinstance(events_data, str):
                        try:
                            events_data = json.loads(events_data)
                        except:
                            logger.warning(f"Could not parse events as JSON for {row.alarmId}")
                            continue
                    
                    if events_data and isinstance(events_data, dict) and len(events_data) > 5:
                        alert = {
                            'alert_id': row.alarmId,
                            'timestamp': datetime.now().isoformat(),
                            'source': self._extract_source_ip(events_data),
                            'destination': self._extract_dest_ip(events_data),
                            'description': self._generate_description(events_data),
                            'severity': self._calculate_severity(events_data),
                            'bytes_transferred': self._extract_bytes(events_data),
                            'connection_count': self._extract_connections(events_data),
                            'protocol': self._extract_protocol(events_data),
                            'ports': self._extract_ports(events_data),
                            'raw_event': events_data
                        }
                        alerts.append(alert)
                        
                except Exception as e:
                    logger.error(f"Error parsing event data for alarm {row.alarmId}: {e}")
                    continue
            
            logger.info(f"✅ Fetched {len(alerts)} real SIEM network flow events")
            return alerts
            
        except Exception as e:
            logger.error(f"❌ Error fetching real SIEM events: {e}")
            return []
    
    def _extract_source_ip(self, data: dict) -> str:
        """Extract source IP from network flow data"""
        for key in data.keys():
            if isinstance(key, str) and '.' in key and len(key.split('.')) == 4:
                try:
                    parts = key.split('.')
                    if all(0 <= int(part) <= 255 for part in parts):
                        return key
                except:
                    continue
        
        ip_patterns = ['149.171.126.7', '59.166.0.9']
        for pattern in ip_patterns:
            if pattern in data:
                return pattern
        
        return "Unknown"
    
    def _extract_dest_ip(self, data: dict) -> str:
        """Extract destination IP from network flow data"""
        for value in data.values():
            if isinstance(value, str) and '.' in value and len(value.split('.')) == 4:
                try:
                    parts = value.split('.')
                    if all(0 <= int(part) <= 255 for part in parts):
                        return value
                except:
                    continue
        
        dest_patterns = ['175.45.176.0', '175.45.176.1', '149.171.126.10']
        for pattern in dest_patterns:
            if pattern in data.values():
                return pattern
        
        return "Unknown"
    
    def _extract_bytes(self, data: dict) -> int:
        """Extract bytes transferred from flow data"""
        byte_candidates = []
        for key, value in data.items():
            try:
                if isinstance(value, (int, float)) and value > 1000:
                    byte_candidates.append(value)
            except:
                continue
        
        return max(byte_candidates) if byte_candidates else 0
    
    def _extract_connections(self, data: dict) -> int:
        """Extract connection count"""
        connection_candidates = []
        for key, value in data.items():
            try:
                if isinstance(value, int) and 1 <= value <= 200:
                    connection_candidates.append(value)
            except:
                continue
        
        return max(connection_candidates) if connection_candidates else 1
    
    def _extract_protocol(self, data: dict) -> str:
        """Extract protocol information"""
        protocols = []
        protocol_indicators = ['tcp', 'udp', 'smtp', 'dns', 'FIN', 'INT']
        
        for indicator in protocol_indicators:
            if indicator in data:
                protocols.append(indicator.upper())
        
        return ','.join(set(protocols)) if protocols else 'Unknown'
    
    def _extract_ports(self, data: dict) -> List[int]:
        """Extract port numbers"""
        ports = []
        common_ports = [25, 53, 80, 443, 7045, 3380, 8080, 8443]
        
        for port in common_ports:
            if str(port) in data and data[str(port)] != 0:
                ports.append(port)
        
        return ports
    
    def _generate_description(self, data: dict) -> str:
        """Generate description from network flow data"""
        source = self._extract_source_ip(data)
        dest = self._extract_dest_ip(data)
        protocol = self._extract_protocol(data)
        bytes_transferred = self._extract_bytes(data)
        
        return f"Network flow: {source} -> {dest}, Protocol: {protocol}, Bytes: {bytes_transferred:,}"
    
    def _calculate_severity(self, data: dict) -> str:
        """Calculate severity based on network flow characteristics"""
        bytes_transferred = self._extract_bytes(data)
        connections = self._extract_connections(data)
        
        if bytes_transferred > 5000000000 or connections > 150:  # 5GB+ or 150+ connections
            return 'HIGH'
        elif bytes_transferred > 2000000000 or connections > 100:  # 2GB+ or 100+ connections
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def mark_processed(self, alarm_id: str) -> bool:
        """Mark SIEM event as processed by ADA"""
        try:
            query = f"""
                UPDATE `{self.project_id}.gatra_database.siem_events`
                SET processed_by_ada = TRUE
                WHERE alarmId = '{alarm_id}'
            """
            
            job = self.client.query(query)
            job.result()
            return True
            
        except Exception as e:
            logger.error(f"❌ Error marking alarm {alarm_id} as processed: {e}")
            return False
    
    def insert_processed_alert(self, alert_data: Dict[str, Any]) -> bool:
        """Insert processed alert result to BigQuery"""
        try:
            table_id = f"{self.project_id}.soc_data.processed_alerts"
            table = self.client.get_table(table_id)
            
            row = {
                "alert_id": alert_data.get('alert_id'),
                "timestamp": datetime.now().isoformat(),
                "confidence_score": float(alert_data.get('confidence', 0.0)),
                "classification": alert_data.get('classification', 'benign'),
                "raw_alert": json.dumps(alert_data.get('raw_alert', {}))
            }
            
            errors = self.client.insert_rows_json(table, [row])
            
            if not errors:
                logger.info(f"✅ Successfully inserted processed alert: {alert_data.get('alert_id')}")
                return True
            else:
                logger.error(f"❌ Error inserting alert: {errors}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error inserting processed alert: {e}")
            return False

class PubSubPublisher:
    """Pub/Sub publisher for ADA alerts"""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.publisher = pubsub_v1.PublisherClient()
        self.topic_path = self.publisher.topic_path(project_id, "ada-alerts")
        
        logger.info(f"📡 Pub/Sub Publisher initialized for topic: {self.topic_path}")
    
    def publish_alert(self, alert_data: Dict[str, Any]) -> bool:
        """Publish alert to Pub/Sub for TAA consumption"""
        try:
            # Prepare message for TAA
            message_data = {
                "alarm_id": alert_data.get('alert_id'),
                "timestamp": alert_data.get('timestamp', datetime.now().isoformat()),
                "confidence": float(alert_data.get('confidence', 0.0)),
                "classification": alert_data.get('classification', 'benign'),
                "is_anomaly": alert_data.get('classification') == 'anomaly',
                "alert_severity": alert_data.get('raw_alert', {}).get('severity', 'LOW'),
                "source_ip": alert_data.get('raw_alert', {}).get('source', 'Unknown'),
                "destination_ip": alert_data.get('raw_alert', {}).get('destination', 'Unknown'),
                "protocol": alert_data.get('raw_alert', {}).get('protocol', 'Unknown'),
                "attack_category": "Network Flow",
                "enriched_data": {
                    "bytes_transferred": alert_data.get('raw_alert', {}).get('bytes_transferred', 0),
                    "connection_count": alert_data.get('raw_alert', {}).get('connection_count', 0),
                    "ports": alert_data.get('raw_alert', {}).get('ports', []),
                    "detection_reasons": alert_data.get('detection_reasons', [])
                },
                "raw_alert": alert_data.get('raw_alert', {}),
                "source": "ada_agent"
            }
            
            # Convert to JSON and publish
            message_json = json.dumps(message_data)
            message_bytes = message_json.encode("utf-8")
            
            # Publish message
            future = self.publisher.publish(self.topic_path, message_bytes)
            message_id = future.result()
            
            logger.info(f"📤 Published alert {alert_data.get('alert_id')} to Pub/Sub (message_id: {message_id})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error publishing alert to Pub/Sub: {e}")
            return False

class ProductionAnomalyDetector:
    """Production-tuned anomaly detector with enterprise-appropriate sensitivity"""
    
    def __init__(self):
        # Significantly increased threshold for production environment
        self.confidence_threshold = float(os.getenv('ANOMALY_THRESHOLD', '0.7'))  # Raised from 0.4 to 0.7
        self.processed_count = 0
        self.anomaly_count = 0
        
        logger.info(f"🎯 Production detector initialized with threshold: {self.confidence_threshold}")
        
    def detect_anomaly(self, alert_data: Dict[str, Any]) -> tuple:
        """Production-tuned anomaly detection with enterprise-appropriate sensitivity"""
        base_confidence = 0.1  # Lower base confidence for production
        enhanced_confidence = base_confidence
        anomaly_reasons = []
        
        source = alert_data.get('source', '')
        destination = alert_data.get('destination', '')
        bytes_transferred = alert_data.get('bytes_transferred', 0)
        connection_count = alert_data.get('connection_count', 0)
        protocol = alert_data.get('protocol', '')
        ports = alert_data.get('ports', [])
        
        # 1. PRODUCTION Volume-based detection (much higher thresholds)
        flow_type = self._classify_flow_type(source, destination)
        volume_threshold = self._get_volume_threshold(flow_type)
        
        if bytes_transferred > volume_threshold * 5:  # 5x threshold = extremely high
            enhanced_confidence += 0.5
            anomaly_reasons.append(f"extremely_high_volume:{self._format_bytes(bytes_transferred)}")
        elif bytes_transferred > volume_threshold * 3:  # 3x threshold = very high
            enhanced_confidence += 0.3
            anomaly_reasons.append(f"very_high_volume:{self._format_bytes(bytes_transferred)}")
        elif bytes_transferred > volume_threshold * 2:  # 2x threshold = high
            enhanced_confidence += 0.2
            anomaly_reasons.append(f"high_volume:{self._format_bytes(bytes_transferred)}")
        elif bytes_transferred > volume_threshold:  # 1x threshold = elevated (minimal penalty)
            enhanced_confidence += 0.05
            anomaly_reasons.append(f"elevated_volume:{self._format_bytes(bytes_transferred)}")
        
        # 2. PRODUCTION Connection count analysis (higher thresholds)
        if connection_count > 200:
            enhanced_confidence += 0.4
            anomaly_reasons.append(f"extremely_many_connections:{connection_count}")
        elif connection_count > 150:
            enhanced_confidence += 0.3
            anomaly_reasons.append(f"very_many_connections:{connection_count}")
        elif connection_count > 100:
            enhanced_confidence += 0.2
            anomaly_reasons.append(f"many_connections:{connection_count}")
        elif connection_count > 80:
            enhanced_confidence += 0.1
            anomaly_reasons.append(f"elevated_connections:{connection_count}")
        
        # 3. PRODUCTION IP address analysis - focus on truly external flows
        if flow_type == 'external_to_external':
            enhanced_confidence += 0.4
            anomaly_reasons.append(f"external_to_external_flow")
        elif flow_type == 'internal_to_external' and not self._is_likely_business_external(destination):
            enhanced_confidence += 0.3
            anomaly_reasons.append(f"suspicious_external_dest:{destination}")
        elif flow_type == 'external_to_internal':
            enhanced_confidence += 0.3
            anomaly_reasons.append(f"external_to_internal:{source}")
        # internal_to_internal and known business externals get no penalty
        
        # 4. PRODUCTION Protocol analysis - focus on truly suspicious combinations
        if 'SMTP' in protocol and 'FIN' in protocol and bytes_transferred > 1000000000:  # 1GB+ SMTP+FIN
            enhanced_confidence += 0.4  # High suspicion for large SMTP+FIN
            anomaly_reasons.append("large_smtp_with_fin_packets")
        elif 'SMTP' in protocol and 'FIN' in protocol:
            enhanced_confidence += 0.2  # Moderate suspicion for SMTP+FIN
            anomaly_reasons.append("smtp_with_fin_packets")
        elif 'SMTP' in protocol and bytes_transferred > 2000000000:  # 2GB+ SMTP
            enhanced_confidence += 0.2
            anomaly_reasons.append("large_smtp_transfer")
        # Regular SMTP gets no penalty (business email)
        
        # 5. PRODUCTION Port analysis - focus on truly high-risk ports
        critical_risk_ports = [7045, 3380, 1433, 3389, 22, 23, 21]  # Database, RDP, SSH, FTP
        business_ports = [25, 53, 80, 443, 993, 995, 587, 143, 110, 8080, 8443]
        
        critical_port_count = sum(1 for port in ports if port in critical_risk_ports)
        business_port_count = sum(1 for port in ports if port in business_ports)
        
        if critical_port_count > 1:  # Multiple critical ports
            enhanced_confidence += 0.3
            anomaly_reasons.append(f"multiple_critical_ports:{[p for p in ports if p in critical_risk_ports]}")
        elif critical_port_count > 0:  # Single critical port
            enhanced_confidence += 0.2
            for port in ports:
                if port in critical_risk_ports:
                    anomaly_reasons.append(f"critical_port:{port}")
        # Business ports get no penalty
        
        # 6. PRODUCTION Time-based detection (business-aware)
        current_time = datetime.now()
        hour = current_time.hour
        
        # Check if it's during known business processes
        if self._is_backup_window(hour):
            enhanced_confidence -= 0.1  # Reduce suspicion during backup windows
            anomaly_reasons.append("backup_window_activity")
        elif self._is_business_hours(hour):
            enhanced_confidence -= 0.05  # Slight reduction during business hours
        elif hour >= 23 or hour <= 4:  # Very late hours
            enhanced_confidence += 0.2
            anomaly_reasons.append("very_late_hours")
        elif hour >= 20 or hour <= 7:  # Extended hours
            enhanced_confidence += 0.1
            anomaly_reasons.append("extended_hours")
        
        # Weekend detection (reduced impact)
        if current_time.weekday() >= 5:  # Weekend
            if self._is_backup_window(hour):
                pass  # Weekend backups are normal
            else:
                enhanced_confidence += 0.1
                anomaly_reasons.append("weekend_activity")
        
        # 7. PRODUCTION Severity-based boost (reduced impact)
        severity = alert_data.get('severity', '')
        if severity == 'HIGH' and bytes_transferred > 5000000000:  # Only very large high-severity
            enhanced_confidence += 0.2
            anomaly_reasons.append("high_severity_large_transfer")
        elif severity == 'HIGH':
            enhanced_confidence += 0.1
            anomaly_reasons.append("high_severity")
        
        # Ensure minimum confidence doesn't go below base
        enhanced_confidence = max(enhanced_confidence, base_confidence)
        
        # Cap confidence at 1.0
        enhanced_confidence = min(enhanced_confidence, 1.0)
        
        # Determine if anomaly
        is_anomaly = enhanced_confidence >= self.confidence_threshold
        
        # Update counters
        self.processed_count += 1
        if is_anomaly:
            self.anomaly_count += 1
        
        if anomaly_reasons:
            logger.info(f"🔍 Production detection: base={base_confidence:.3f} -> enhanced={enhanced_confidence:.3f} (threshold={self.confidence_threshold}) - {', '.join(anomaly_reasons)}")
        
        return enhanced_confidence, is_anomaly, anomaly_reasons
    
    def _classify_flow_type(self, source: str, destination: str) -> str:
        """Classify the type of network flow"""
        source_internal = self._is_internal_ip(source)
        dest_internal = self._is_internal_ip(destination)
        
        if source_internal and dest_internal:
            return 'internal_to_internal'
        elif source_internal and not dest_internal:
            return 'internal_to_external'
        elif not source_internal and dest_internal:
            return 'external_to_internal'
        else:
            return 'external_to_external'
    
    def _get_volume_threshold(self, flow_type: str) -> int:
        """Get volume threshold based on flow type (production values)"""
        thresholds = {
            'internal_to_internal': 5000000000,   # 5GB - backups, data sync, replication
            'internal_to_external': 1000000000,   # 1GB - cloud backup, uploads
            'external_to_internal': 500000000,    # 500MB - downloads, updates
            'external_to_external': 100000000     # 100MB - suspicious
        }
        return thresholds.get(flow_type, 500000000)
    
    def _is_likely_business_external(self, ip: str) -> bool:
        """Check if external IP is likely a business service"""
        # Add your known business external IPs/ranges here
        business_externals = [
            # Cloud providers (examples - adjust for your environment)
            '8.8.8.8',      # Google DNS
            '1.1.1.1',      # Cloudflare DNS
            # Add your cloud backup, CDN, or business partner IPs
        ]
        
        for business_ip in business_externals:
            if ip.startswith(business_ip.split('.')[0]):  # Simple prefix match
                return True
        
        return False
    
    def _is_backup_window(self, hour: int) -> bool:
        """Check if current hour is during backup window"""
        return hour in list(range(22, 24)) + list(range(0, 6))  # 10 PM - 6 AM
    
    def _is_business_hours(self, hour: int) -> bool:
        """Check if current hour is during business hours"""
        return 8 <= hour <= 18  # 8 AM - 6 PM
    
    def _format_bytes(self, bytes_val: int) -> str:
        """Format bytes for display"""
        if bytes_val >= 1000000000:
            return f"{bytes_val/1000000000:.1f}GB"
        elif bytes_val >= 1000000:
            return f"{bytes_val/1000000:.1f}MB"
        else:
            return f"{bytes_val/1000:.0f}KB"
    
    def _is_internal_ip(self, ip: str) -> bool:
        """Check if IP is internal/private"""
        if ip == 'Unknown' or not ip:
            return True
        try:
            parts = ip.split('.')
            if len(parts) != 4:
                return False
            
            first = int(parts[0])
            second = int(parts[1])
            
            # Private IP ranges
            if first == 10:
                return True
            if first == 172 and 16 <= second <= 31:
                return True
            if first == 192 and second == 168:
                return True
            if first == 149:  # Organization's range
                return True
            
            return False
        except:
            return True

class ProductionTunedLangGraphAnomalyDetectionAgent:
    """Enhanced ADA Agent with Pub/Sub publishing for TAA integration"""
    
    def __init__(self):
        self.project_id = os.getenv('BIGQUERY_PROJECT_ID', 'chronicle-dev-2be9')
        self.bq_client = ProductionTunedBigQueryClient(self.project_id)
        self.detector = ProductionAnomalyDetector()
        self.pubsub_publisher = PubSubPublisher(self.project_id)
        
        logger.info(f"🏢 Enhanced ADA Agent initialized with project: {self.project_id}")
        logger.info(f"⚙️ Enterprise-appropriate sensitivity for production environment")
        logger.info(f"🎯 Anomaly threshold: {self.detector.confidence_threshold} (production-grade)")
        logger.info(f"📊 Target detection rate: 20-40% (enterprise standard)")
        logger.info(f"📡 Pub/Sub integration enabled for TAA communication")
    
    async def process_real_siem_events(self):
        """Main processing loop for real SIEM data with Pub/Sub publishing"""
        logger.info("🔄 Starting enhanced SIEM event processing loop with Pub/Sub publishing...")
        
        while True:
            try:
                events = self.bq_client.fetch_real_siem_events(limit=3)
                
                if not events:
                    logger.info("ℹ️  No new network flow events to process")
                    await asyncio.sleep(30)
                    continue
                
                processed_count = 0
                anomaly_count = 0
                pubsub_published_count = 0
                
                for event in events:
                    result = await self._process_single_event(event)
                    if result:
                        processed_count += 1
                        if result.get('classification') == 'anomaly':
                            anomaly_count += 1
                        
                        # NEW: Publish to Pub/Sub for TAA consumption
                        pubsub_success = self.pubsub_publisher.publish_alert(result)
                        if pubsub_success:
                            pubsub_published_count += 1
                        
                        self.bq_client.mark_processed(event['alert_id'])
                
                detection_rate = (anomaly_count / processed_count * 100) if processed_count > 0 else 0
                logger.info(f"📊 Enhanced batch complete: {processed_count} events, {anomaly_count} anomalies ({detection_rate:.1f}% detection rate)")
                logger.info(f"📡 Pub/Sub published: {pubsub_published_count}/{processed_count} alerts to TAA")
                logger.info(f"📈 Total processed: {self.detector.processed_count}, Total anomalies: {self.detector.anomaly_count}")
                
                if self.detector.processed_count > 0:
                    overall_rate = (self.detector.anomaly_count / self.detector.processed_count) * 100
                    logger.info(f"🎯 Overall detection rate: {overall_rate:.1f}% (target: 20-40%)")
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"💥 Error in enhanced processing loop: {e}")
                await asyncio.sleep(60)
    
    async def _process_single_event(self, event: Dict[str, Any]):
        """Process a single network flow event"""
        try:
            alarm_id = event.get('alert_id')
            
            confidence, is_anomaly, reasons = self.detector.detect_anomaly(event)
            
            result = {
                'alert_id': alarm_id,
                'confidence': confidence,
                'classification': 'anomaly' if is_anomaly else 'benign',
                'raw_alert': event,
                'detection_reasons': reasons
            }
            
            status = "🚨 ANOMALY" if is_anomaly else "✅ benign"
            logger.info(f"Enhanced Network Flow {alarm_id} processed: confidence={confidence:.3f}, classification={status}")
            logger.info(f"📊 Flow details: {event.get('description', 'No description')}")
            
            if reasons:
                logger.info(f"🔍 Detection reasons: {', '.join(reasons)}")
            
            # Insert to BigQuery (existing functionality)
            success = self.bq_client.insert_processed_alert(result)
            if not success:
                logger.error(f"❌ Failed to insert result for alarm {alarm_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"💥 Error processing enhanced event {event.get('alert_id')}: {e}")
            return None

async def main():
    """Main function"""
    logger.info("🚀 Starting Enhanced ADA Agent with Pub/Sub Integration...")
    logger.info("🏢 Enterprise-appropriate sensitivity for production environment")
    logger.info("📊 Target detection rate: 20-40% (significantly reduced from 95%)")
    logger.info("🎯 Focus on genuine security threats, ignore routine business traffic")
    logger.info("💼 Optimized for enterprise backup, sync, and data transfer patterns")
    logger.info("📡 NEW: Pub/Sub publishing enabled for TAA integration")
    
    try:
        agent = ProductionTunedLangGraphAnomalyDetectionAgent()
        await agent.process_real_siem_events()
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())


