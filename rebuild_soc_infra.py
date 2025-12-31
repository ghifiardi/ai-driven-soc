#!/usr/bin/env python3
"""
SOC Infrastructure Initialization Script
=======================================

Automates the creation of:
1. BigQuery Datasets for each tenant.
2. BigQuery Tables (events, alerts, results) with the correct schemas.
3. Pub/Sub Topics and Subscriptions for multi-tenant ingestion.

Usage:
    export PROJECT_ID="your-new-project-id"
    python3 rebuild_soc_infra.py
"""

import os
import logging
from google.cloud import bigquery, pubsub_v1
from google.api_core.exceptions import AlreadyExists, NotFound
from multi_tenant_manager import MultiTenantManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ID = os.getenv("PROJECT_ID")
CONFIG_PATH = "config/gatra_multitenant_config.json"

if not PROJECT_ID:
    logger.error("Error: PROJECT_ID environment variable is not set.")
    exit(1)

def create_bq_resources(bq_client, tenant):
    """Create BigQuery datasets and tables for a tenant."""
    datasets = [tenant.dataset, tenant.results_dataset]
    
    for ds_id in datasets:
        ds_ref = bigquery.DatasetReference(PROJECT_ID, ds_id)
        try:
            bq_client.get_dataset(ds_ref)
            logger.info(f"Dataset {ds_id} already exists.")
        except NotFound:
            dataset = bigquery.Dataset(ds_ref)
            dataset.location = tenant.region or "us-central1"
            bq_client.create_dataset(dataset)
            logger.info(f"Created dataset {ds_id} in {dataset.location}")

    # Define common schemas based on existing SQL fixes
    schemas = {
        "events": [
            bigquery.SchemaField("alarmId", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("events", "JSON", mode="NULLABLE"),
            bigquery.SchemaField("timestamp", "TIMESTAMP", mode="NULLABLE"),
        ],
        "alerts": [
            bigquery.SchemaField("alarmId", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("classification", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("confidence_score", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("is_anomaly", "BOOL", mode="NULLABLE"),
            bigquery.SchemaField("raw_alert", "JSON", mode="NULLABLE"),
            bigquery.SchemaField("timestamp", "TIMESTAMP", mode="NULLABLE"),
        ],
        "events_results": [
            bigquery.SchemaField("alarmId", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("processed_timestamp", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("processed_by", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("ada_case_class", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("cra_action_type", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("ada_confidence", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("taa_confidence", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("ada_score", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("taa_severity", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("ada_valid", "BOOL", mode="NULLABLE"),
            bigquery.SchemaField("taa_valid", "BOOL", mode="NULLABLE"),
            bigquery.SchemaField("cra_success", "BOOL", mode="NULLABLE"),
            bigquery.SchemaField("ada_reasoning", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("taa_reasoning", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("cra_reasoning", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("variable_of_importance", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("ada_detected", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("taa_created", "TIMESTAMP", mode="NULLABLE"),
            bigquery.SchemaField("cra_created", "TIMESTAMP", mode="NULLABLE"),
            bigquery.SchemaField("enhanced_classification", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("calibrated", "BOOL", mode="NULLABLE"),
            bigquery.SchemaField("suppression_recommended", "BOOL", mode="NULLABLE"),
            bigquery.SchemaField("threat_score", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("raw_json", "JSON", mode="NULLABLE"),
            # Embedding columns for CLA support
            bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
            bigquery.SchemaField("embedding_timestamp", "TIMESTAMP", mode="NULLABLE"),
            bigquery.SchemaField("embedding_model", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("embedding_similarity", "FLOAT64", mode="NULLABLE"),
            bigquery.SchemaField("rl_reward_score", "FLOAT64", mode="NULLABLE"),
        ]
    }

    # Create Tables
    table_mappings = {
        tenant.tables.events: schemas["events"],
        tenant.tables.alerts: schemas["alerts"],
        tenant.tables.results: schemas["events_results"]
    }

    for table_id, schema in table_mappings.items():
        # results goes to results_dataset, others to primary dataset
        target_ds = tenant.results_dataset if table_id == tenant.tables.results else tenant.dataset
        table_ref = bigquery.TableReference(bigquery.DatasetReference(PROJECT_ID, target_ds), table_id)
        
        try:
            bq_client.get_table(table_ref)
            logger.info(f"Table {target_ds}.{table_id} already exists.")
        except NotFound:
            table = bigquery.Table(table_ref, schema=schema)
            # Add partitioning on timestamp if applicable
            if "timestamp" in [f.name for f in schema]:
                table.time_partitioning = bigquery.TimePartitioning(field="timestamp")
            elif "processed_timestamp" in [f.name for f in schema]:
                table.time_partitioning = bigquery.TimePartitioning(field="processed_timestamp")
                
            bq_client.create_table(table)
            logger.info(f"Created table {target_ds}.{table_id} with partitioning.")

def create_pubsub_resources(publisher, subscriber, tenant):
    """Create Pub/Sub topics and subscriptions for a tenant."""
    topics = [
        tenant.pubsub_topics.ingest,
        tenant.pubsub_topics.alerts,
        tenant.pubsub_topics.priority
    ]
    
    for topic_name in topics:
        topic_path = publisher.topic_path(PROJECT_ID, topic_name)
        try:
            publisher.create_topic(name=topic_path)
            logger.info(f"Created topic: {topic_name}")
        except AlreadyExists:
            logger.info(f"Topic {topic_name} already exists.")

    # Create default subscriptions
    sub_map = {
        f"{tenant.pubsub_topics.ingest}-sub": tenant.pubsub_topics.ingest,
        f"{tenant.pubsub_topics.alerts}-sub": tenant.pubsub_topics.alerts,
    }

    for sub_name, topic_name in sub_map.items():
        sub_path = subscriber.subscription_path(PROJECT_ID, sub_name)
        topic_path = publisher.topic_path(PROJECT_ID, topic_name)
        try:
            subscriber.create_subscription(name=sub_path, topic=topic_path)
            logger.info(f"Created subscription: {sub_name}")
        except AlreadyExists:
            logger.info(f"Subscription {sub_name} already exists.")

def main():
    logger.info(f"Starting SOC rebuild on project: {PROJECT_ID}")
    
    # Load tenant config
    manager = MultiTenantManager.from_file(CONFIG_PATH)
    
    # Initialize Clients
    bq_client = bigquery.Client(project=PROJECT_ID)
    publisher = pubsub_v1.PublisherClient()
    subscriber = pubsub_v1.SubscriberClient()

    for tenant in manager.list_tenants():
        logger.info(f"Initializing resources for tenant: {tenant.tenant_id}")
        create_bq_resources(bq_client, tenant)
        create_pubsub_resources(publisher, subscriber, tenant)

    logger.info("✅ Infrastructure initialization complete.")

if __name__ == "__main__":
    main()
