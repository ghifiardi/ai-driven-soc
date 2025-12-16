#!/bin/bash

# Script to run ADA system with the correct Python version
echo "Starting ADA system using Python 3.11.13..."

# Set path to include user's local Python packages
export PATH=~/.local/bin:$PATH

# Execute the main entry point with Python 3.11.13
/usr/bin/python3.11 /home/raditio.ghifiardigmail.com/ai-driven-soc/continuous_integration.py

# Note: If this is executed on the local machine, adjust the path accordingly
