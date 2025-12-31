#!/usr/bin/env python3
"""
SOC Data Seeding Script (Cold-Start)
===================================

Seeds a new SOC project with 30 days of synthetic historical data to:
1. Provide a baseline for the AI agents (CLA/ADA).
2. Populate the dashboard with realistic trends immediately.
3. Verify that the new BigQuery schemas are working correctly.

Usage:
    export PROJECT_ID="your-new-project-id"
    python3 seed_soc_data.py --tenant_id tenant_001
"""

import os
import argparse
import random
import logging
import json
from datetime import datetime, timedelta
from google.cloud import bigquery
import pandas as pd
import numpy as np
from multi_tenant_manager import MultiTenantManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ID = os.getenv("PROJECT_ID")
CONFIG_PATH = "config/gatra_multitenant_config.json"

if not PROJECT_ID:
    logger.error("Error: PROJECT_ID environment variable is not set.")
    exit(1)

class SOCDataSeeder:
    def __init__(self, tenant_config):
        self.tenant = tenant_config
        self.client = bigquery.Client(project=PROJECT_ID)
        self.table_id = f"{PROJECT_ID}.{self.tenant.results_dataset}.{self.tenant.tables.results}"

    def generate_seed_data(self, days=30, alerts_per_day=20):
        """Generate and insert synthetic data for the tenant."""
        logger.info(f"Seeding {days} days of data for tenant {self.tenant.tenant_id}...")
        
        all_rows = []
        base_time = datetime.now() - timedelta(days=days)

        patterns = [
            ("suspicious_login", 0.8, "Brute force attempt detected from new IP"),
            ("malware_beaconing", 0.9, "Outbound connection to known C2 domain"),
            ("privilege_escalation", 0.75, "Standard user executed sudo with high-risk command"),
            ("data_exfiltration", 0.85, "Large data transfer to unauthorized cloud storage"),
            ("network_scan", 0.6, "Internal port scanning detected from workstation")
        ]

        for day in range(days):
            current_date = base_time + timedelta(days=day)
            
            for _ in range(alerts_per_day):
                # Timing
                alert_time = current_date + timedelta(seconds=random.randint(0, 86400))
                proc_time = alert_time + timedelta(seconds=random.randint(30, 300))
                
                # Pick a pattern
                pattern, confidence, reasoning = random.choice(patterns)
                conf_variance = random.uniform(-0.1, 0.1)
                
                # Mock high-quality embeddings (768-dim)
                embedding = [random.uniform(-1, 1) for _ in range(768)]
                
                row = {
                    "alarmId": f"SEED_{alert_time.strftime('%Y%m%d%H%M')}_{random.randint(100, 999)}",
                    "processed_timestamp": proc_time.isoformat(),
                    "processed_by": "System-Seeder",
                    "ada_case_class": pattern,
                    "ada_confidence": confidence + conf_variance,
                    "taa_confidence": confidence + (conf_variance * 0.5),
                    "ada_score": random.uniform(0.5, 0.95),
                    "taa_severity": random.choice([0.1, 0.4, 0.8, 1.0]),
                    "ada_valid": True,
                    "taa_valid": True,
                    "cra_success": random.choice([True, True, True, False]), # 75% success
                    "ada_reasoning": reasoning,
                    "taa_reasoning": f"Enhanced verification of {pattern} confirms threat.",
                    "enhanced_classification": pattern,
                    "calibrated": True,
                    "suppression_recommended": False,
                    "threat_score": random.uniform(60, 95),
                    "raw_json": json.dumps({"source": "seeder", "original_alert_id": f"RAW_{random.randint(1000, 9999)}"}),
                    "embedding": embedding,
                    "embedding_timestamp": proc_time.isoformat(),
                    "embedding_model": "text-embedding-004",
                    "embedding_similarity": random.uniform(0.7, 0.9),
                    "rl_reward_score": random.uniform(0.6, 1.0),
                    "taa_created": proc_time.isoformat(),
                    "cra_created": proc_time.isoformat()
                }
                all_rows.append(row)

        # Use Load Job instead of Streaming Insert for Free Tier compatibility
        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND",
            source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        )
        
        try:
            # Convert to newline-delimited JSON format for the load job
            job = self.client.load_table_from_json(all_rows, self.table_id, job_config=job_config)
            job.result()  # Wait for the job to complete
            logger.info(f"✅ Successfully seeded {len(all_rows)} rows for {self.tenant.tenant_id} via Load Job.")
        except Exception as e:
            logger.error(f"❌ Failed to seed data via Load Job: {e}")

def main():
    parser = argparse.ArgumentParser(description="Seed SOC data for a new project.")
    parser.add_argument("--tenant_id", required=True, help="The ID of the tenant to seed.")
    parser.add_argument("--days", type=int, default=30, help="Number of days to seed.")
    args = parser.parse_args()

    manager = MultiTenantManager.from_file(CONFIG_PATH)
    try:
        tenant = manager.get_tenant(args.tenant_id)
        seeder = SOCDataSeeder(tenant)
        seeder.generate_seed_data(days=args.days)
    except Exception as e:
        logger.error(f"Error seeding data: {e}")

if __name__ == "__main__":
    main()
