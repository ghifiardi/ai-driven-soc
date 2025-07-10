#!/usr/bin/env python
# Test script to verify BigQuery integration with Python 3.11.13

import os
import sys
import logging
import time
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("BQ_TEST")

def log_environment_info():
    """Log important environment information for debugging"""
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Current directory: {os.getcwd()}")
    logger.info(f"Script location: {os.path.dirname(os.path.abspath(__file__))}")
    
    # Check if key packages are available
    packages_to_check = [
        "pandas", "numpy", "google.cloud", "joblib", 
        "typing_extensions", "sklearn", "langgraph"
    ]
    
    for package in packages_to_check:
        try:
            module = __import__(package.split('.')[0])
            version = getattr(module, '__version__', 'unknown version')
            logger.info(f"✅ {package} is available (version: {version})")
        except ImportError:
            logger.error(f"❌ {package} is NOT available")

def test_bigquery_integration():
    """Test importing and using bigquery_integration module"""
    try:
        logger.info("Attempting to import bigquery_integration module...")
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        import bigquery_integration
        logger.info("✅ Successfully imported bigquery_integration module")
        
        logger.info("Testing BigQuery data fetch function...")
        # Get the absolute path of the project directory
        project_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Sample query with a small limit
        query = "SELECT * FROM `chronicle-dev-2be9.gatra_database.siem_events` LIMIT 5"
        
        # Call the fetch_bigquery_data function with proper parameters
        result = bigquery_integration.fetch_bigquery_data(
            query=query,
            location="asia-southeast2",  # Specify the BigQuery dataset location
            max_rows=5,  # Only fetch a small number of records for the test
            logger=logger
        )
        
        if result:
            logger.info("✅ Successfully fetched data from BigQuery!")
            
            # Check if the output file exists and has content
            if os.path.exists(os.path.join(project_dir, "test_results.json")):
                file_size = os.path.getsize(os.path.join(project_dir, "test_results.json"))
                logger.info(f"Output file size: {file_size} bytes")
                if file_size > 0:
                    logger.info("✅ Output file contains data")
                else:
                    logger.warning("⚠️ Output file is empty")
            else:
                logger.error("❌ Output file was not created")
        else:
            logger.error("❌ Failed to fetch data from BigQuery")
            
    except Exception as e:
        logger.error(f"Error during testing: {e}", exc_info=True)

if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("BIGQUERY INTEGRATION TEST - STARTED")
    logger.info("=" * 50)
    
    # Log environment information
    log_environment_info()
    
    # Run the BigQuery integration test
    test_bigquery_integration()
    
    logger.info("=" * 50)
    logger.info("BIGQUERY INTEGRATION TEST - COMPLETED")
    logger.info("=" * 50)
