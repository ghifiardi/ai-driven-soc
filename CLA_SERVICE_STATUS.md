# Production CLA Service - Access and Endpoints

## Overview
This document lists the CLA-related services running on the GCP VM, their access URLs, and available HTTP endpoints.

## Services Running

- **Production CLA Service** (`production-cla.service`)
  - Status: Active (running)
  - Port: 8080
  - Access URL: `http://10.45.254.19:8080/`
  - Description: Production CLA Service - 100% Performance AI Model

- **Continuous Learning Agent (CLA)** (`cla.service`)
  - Status: Active (running)
  - Role: Retraining, feedback processing

- **ADA Production Service** (`ada-production.service`)
  - Status: Active (running)
  - Role: Real-time anomaly detection and production processing

- **ADA BigQuery Integration Service** (`ada-bigquery-integration.service`)
  - Status: Active (running)
  - Role: BigQuery integration and data pipeline

## Production CLA Endpoints

Base URL: `http://10.45.254.19:8080`

- `GET /status` — Service status and model info
  - Example response:
    ```json
    {
      "accuracy": 0.0,
      "last_retrain": "2025-09-22T02:38:24.102280",
      "model_loaded": true,
      "model_version": "20250922_023749",
      "next_retrain": "2025-09-23T02:38:24.102286",
      "predictions_count": 303,
      "status": "running",
      "uptime": 87381.13220834732
    }
    ```

- `GET /health` — Health check and model readiness
  - Example response:
    ```json
    {
      "healthy": true,
      "model_loaded": true,
      "test_prediction": "anomaly",
      "timestamp": "2025-09-23T02:54:19.160746"
    }
    ```

- `POST /classify` — Submit an alert for classification

- `POST /retrain` — Trigger manual retraining

## Dashboards

- Production CLA Dashboard: `http://10.45.254.19:8504/`
- Additional dashboards: Ports `8503`, `8505`

## Quick Commands

- Check service status (systemd):
  ```bash
  sudo systemctl status production-cla.service --no-pager
  sudo systemctl status cla.service --no-pager
  sudo systemctl status ada-production.service --no-pager
  sudo systemctl status ada-bigquery-integration.service --no-pager
  ```

- Query endpoints:
  ```bash
  curl -s http://10.45.254.19:8080/status | jq
  curl -s http://10.45.254.19:8080/health | jq
  ```
