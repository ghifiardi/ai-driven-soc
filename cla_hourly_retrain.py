#!/usr/bin/env python3
import os
import time
from datetime import datetime, timedelta
import pandas as pd
from google.cloud import bigquery
from google.cloud.exceptions import NotFound

# Simple placeholder model update logic (threshold tuning)
# In production, replace with real training and model registry updates.

def get_bq_client():
    return bigquery.Client()

def fetch_feedback(hours: int = 24) -> pd.DataFrame:
    client = get_bq_client()
    query = f"""
    SELECT alert_id, is_true_positive, confidence, analyst_comments, timestamp
    FROM `soc_data.feedback`
    WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {hours} HOUR)
    ORDER BY timestamp DESC
    """
    return client.query(query).to_dataframe()

def fetch_recent_alerts(hours: int = 24) -> pd.DataFrame:
    client = get_bq_client()
    query = f"""
    SELECT alert_id, is_anomaly, confidence_score, classification, timestamp
    FROM `soc_data.processed_alerts`
    WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {hours} HOUR)
    """
    return client.query(query).to_dataframe()

def compute_metrics(feedback: pd.DataFrame) -> dict:
    if feedback.empty:
        return {
            "accuracy": None,
            "false_positive_rate": None,
            "true_positives": 0,
            "false_positives": 0,
            "total_feedback": 0,
            "avg_confidence": None,
        }
    total = len(feedback)
    tp = int((feedback["is_true_positive"] == True).sum())
    fp = int((feedback["is_true_positive"] == False).sum())
    acc = round((tp / total) * 100, 2) if total else None
    fpr = round((fp / total) * 100, 2) if total else None
    avg_conf = round(float(feedback["confidence"].mean()), 3) if not feedback["confidence"].isna().all() else None
    return {
        "accuracy": acc,
        "false_positive_rate": fpr,
        "true_positives": tp,
        "false_positives": fp,
        "total_feedback": total,
        "avg_confidence": avg_conf,
    }

def ensure_table(client: bigquery.Client, table_id: str, schema: list[bigquery.SchemaField]):
    try:
        client.get_table(table_id)
    except NotFound:
        table = bigquery.Table(table_id, schema=schema)
        client.create_table(table)

PROJECT_ID = "chronicle-dev-2be9"
DATASET_ID = "soc_data"

def fq(table: str) -> str:
    return f"{PROJECT_ID}.{DATASET_ID}.{table}"

def update_metrics_table(metrics: dict):
    client = get_bq_client()
    table_id = fq("cla_metrics")
    # Ensure table exists
    ensure_table(
        client,
        table_id,
        [
            bigquery.SchemaField("timestamp", "TIMESTAMP"),
            bigquery.SchemaField("accuracy", "FLOAT"),
            bigquery.SchemaField("false_positive_rate", "FLOAT"),
            bigquery.SchemaField("true_positives", "INTEGER"),
            bigquery.SchemaField("false_positives", "INTEGER"),
            bigquery.SchemaField("total_feedback", "INTEGER"),
            bigquery.SchemaField("avg_confidence", "FLOAT"),
        ],
    )
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        **{k: v for k, v in metrics.items()},
    }
    errors = client.insert_rows_json(table_id, [row])
    if errors:
        raise RuntimeError(f"Failed to write metrics: {errors}")

def tune_threshold(feedback: pd.DataFrame) -> float:
    # Very simple threshold tuning based on feedback distribution
    if feedback.empty:
        return 0.8
    # If too many false positives, increase threshold; if too many false negatives (not tracked here), decrease.
    fp_rate = (feedback["is_true_positive"] == False).mean()
    base = 0.8
    if fp_rate > 0.3:
        return min(0.95, base + 0.05)
    elif fp_rate < 0.1:
        return max(0.6, base - 0.05)
    return base

def persist_threshold(threshold: float):
    # Persist threshold to a small BigQuery config table for the CLA dashboard to read
    client = get_bq_client()
    table_id = fq("cla_config")
    # Ensure table exists
    ensure_table(
        client,
        table_id,
        [
            bigquery.SchemaField("timestamp", "TIMESTAMP"),
            bigquery.SchemaField("parameter", "STRING"),
            bigquery.SchemaField("value", "FLOAT"),
        ],
    )
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "parameter": "confidence_threshold",
        "value": threshold,
    }
    errors = client.insert_rows_json(table_id, [row])
    if errors:
        raise RuntimeError(f"Failed to write threshold: {errors}")

if __name__ == "__main__":
    # One-shot execution (to be called hourly by a scheduler or cron)
    try:
        fb = fetch_feedback(hours=24)
        alerts = fetch_recent_alerts(hours=24)
        metrics = compute_metrics(fb)
        update_metrics_table(metrics)
        new_threshold = tune_threshold(fb)
        persist_threshold(new_threshold)
        print("Retrain job completed:", metrics, "threshold=", new_threshold)
    except Exception as e:
        print("Retrain job failed:", e)
        raise
