#!/usr/bin/env python

"""
Debug script for LangGraph ADA service
"""

import os
import sys
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/langgraph_debug.log')
    ]
)
logger = logging.getLogger("ServiceDebug")

def log_environment():
    """Log environment details for debugging"""
    logger.info("=== ENVIRONMENT DETAILS ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Current directory: {os.getcwd()}")
    logger.info(f"Files in current directory: {os.listdir('.')}")
    
    # Check if the required files exist
    files_to_check = ['bigquery_integration.py', 'langgraph_ada_integration.py', 'continuous_integration.py']
    for file in files_to_check:
        logger.info(f"File {file} exists: {os.path.isfile(file)}")
    
    # Check virtual environment
    logger.info(f"VIRTUAL_ENV: {os.environ.get('VIRTUAL_ENV', 'Not set')}")
    
    # Try to import key modules
    try:
        import google.cloud.bigquery
        logger.info("Successfully imported google.cloud.bigquery")
    except ImportError as e:
        logger.error(f"Failed to import google.cloud.bigquery: {e}")
    
    try:
        import langgraph
        logger.info(f"Successfully imported langgraph (version: {langgraph.__version__ if hasattr(langgraph, '__version__') else 'unknown'})")
    except ImportError as e:
        logger.error(f"Failed to import langgraph: {e}")

# Simple heartbeat function to show service is running
def main():
    logger.info("Debug service started")
    log_environment()
    
    try:
        logger.info("Starting heartbeat loop")
        counter = 0
        while True:
            counter += 1
            logger.info(f"Service heartbeat #{counter}")
            time.sleep(60)  # Log every minute
    except Exception as e:
        logger.error(f"Error in heartbeat loop: {e}")
        return 1
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        sys.exit(1)
