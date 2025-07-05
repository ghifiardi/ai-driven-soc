#!/usr/bin/env python

"""
Continuous BigQuery Integration for LangGraph ADA

This script wraps the bigquery_integration.py functionality in a continuous loop
to ensure the service runs persistently.
"""

import time
import logging
import importlib
import traceback
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ContinuousBigQueryIntegration")

# Import the original module
try:
    import bigquery_integration
    logger.info("Successfully imported bigquery_integration module")
except ImportError:
    logger.error(f"Failed to import bigquery_integration: {traceback.format_exc()}")
    sys.exit(1)

def run_continuous_integration():
    """
    Run the BigQuery integration in a continuous loop with error handling.
    """
    logger.info("Starting continuous integration service")
    
    while True:
        try:
            logger.info("Beginning alert processing cycle")
            
            # If the module has a main function, call it
            if hasattr(bigquery_integration, 'main'):
                bigquery_integration.main()
            # Otherwise run the script using importlib
            else:
                logger.info("No main() function found, reloading module")
                importlib.reload(bigquery_integration)
                
            logger.info("Completed alert processing cycle")
            
        except Exception as e:
            logger.error(f"Error in processing cycle: {e}")
            logger.error(traceback.format_exc())
        
        # Wait before the next cycle
        logger.info("Waiting for next processing cycle (300 seconds)")
        time.sleep(300)  # 5 minutes between processing cycles

if __name__ == "__main__":
    try:
        run_continuous_integration()
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
    except Exception as e:
        logger.error(f"Unhandled exception in service: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
