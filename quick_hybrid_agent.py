#!/usr/bin/env python3
"""
Quick Hybrid Agent for Dashboard Testing
=======================================

Simplified version of the hybrid agent for immediate dashboard connectivity.
"""

import json
import time
from datetime import datetime
from fastapi import FastAPI
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Quick Hybrid Cyber Defense Agent",
    description="Simplified agent for dashboard testing",
    version="1.0.0"
)

# Mock status data
mock_status = {
    "agent_id": "quick-hybrid-defense",
    "uptime_seconds": 3600,
    "stats": {
        "alerts_processed": 542,
        "actions_recommended": 387,
        "explanations_generated": 298,
        "errors": 5,
        "start_time": datetime.now().isoformat()
    },
    "metrics": {
        "alerts_processed_total": 542,
        "dqn_inference_duration": [0.042, 0.038, 0.045, 0.041, 0.039],
        "gemini_api_calls_total": 298,
        "containment_actions_total": {
            "isolate_node": 89,
            "block_traffic": 156,
            "patch_system": 98,
            "monitor": 32,
            "no_action": 12
        },
        "errors_total": {
            "DQNError": 1,
            "GeminiTimeoutError": 2,
            "PubSubError": 2
        }
    },
    "circuit_breakers": {
        "dqn_model": "CLOSED",
        "gemini_api": "CLOSED", 
        "pubsub_connection": "CLOSED"
    }
}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agent_id": "quick-hybrid-defense",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "dqn_agent": True,
            "gemini_client": True,
            "pubsub": True
        }
    }

@app.get("/status")
async def get_status():
    """Get agent status and metrics"""
    # Update uptime
    mock_status["uptime_seconds"] += 1
    return mock_status

@app.get("/agent-card")
async def get_agent_card():
    """Return A2A agent card"""
    return {
        "name": "quick-hybrid-defense",
        "version": "1.0.0",
        "description": "Quick hybrid agent for dashboard testing",
        "capabilities": [
            "analyze_security_alert",
            "recommend_containment_action",
            "explain_defense_decision"
        ]
    }

@app.post("/a2a/process_alert")
async def process_alert_a2a(task: dict):
    """A2A endpoint for processing alerts"""
    return {
        "task_id": task.get('task_id'),
        "status": "completed",
        "result": {
            "recommended_action": "monitor",
            "confidence": 0.85,
            "explanation": "Alert analyzed and monitoring recommended"
        },
        "agent_id": "quick-hybrid-defense",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    logger.info("Starting Quick Hybrid Agent on port 8083...")
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8083,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Failed to start agent: {e}")
        print(f"Error: {e}")
        print("Try running: pip install fastapi uvicorn")


















