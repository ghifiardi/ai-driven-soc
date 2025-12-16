#!/usr/bin/env python3
"""
Sync Models to Dashboard Directory
Copies new model files from CLA directory to dashboard directory
"""

import os
import shutil
import glob
import time
from datetime import datetime

def sync_models():
    """Sync models from CLA directory to dashboard directory"""
    source_dir = "/home/raditio.ghifiardigmail.com/ai-driven-soc/models/"
    target_dir = "/home/app/ai-model-training-dashboard/models/"
    
    try:
        # Ensure target directory exists
        os.makedirs(target_dir, exist_ok=True)
        
        # Get all model files from source
        model_files = glob.glob(os.path.join(source_dir, "trained_model_*.pkl"))
        metrics_files = glob.glob(os.path.join(source_dir, "model_metrics_*.json"))
        
        # Copy model files
        for model_file in model_files:
            filename = os.path.basename(model_file)
            target_file = os.path.join(target_dir, filename)
            
            if not os.path.exists(target_file):
                shutil.copy2(model_file, target_file)
                print(f"Copied model: {filename}")
        
        # Copy metrics files
        for metrics_file in metrics_files:
            filename = os.path.basename(metrics_file)
            target_file = os.path.join(target_dir, filename)
            
            if not os.path.exists(target_file):
                shutil.copy2(metrics_file, target_file)
                print(f"Copied metrics: {filename}")
        
        # Set proper ownership
        os.system(f"sudo chown -R app:app {target_dir}")
        
        print(f"Model sync completed at {datetime.now()}")
        
    except Exception as e:
        print(f"Error syncing models: {e}")

if __name__ == "__main__":
    sync_models()


