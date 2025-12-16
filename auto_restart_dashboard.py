#!/usr/bin/env python3
"""
Auto-restart script for the SOC dashboard
Monitors the dashboard process and automatically restarts it if it stops
"""
import subprocess
import time
import signal
import sys
import os
from datetime import datetime

def check_dashboard_running():
    """Check if dashboard is running on port 8535"""
    try:
        result = subprocess.run(['ss', '-tulpn'], capture_output=True, text=True)
        return ':8535' in result.stdout
    except:
        try:
            result = subprocess.run(['netstat', '-tlnp'], capture_output=True, text=True)
            return ':8535' in result.stdout
        except:
            return False

def start_dashboard():
    """Start the dashboard process"""
    print(f"[{datetime.now()}] Starting dashboard...")
    try:
        # Change to the correct directory and start dashboard
        cmd = [
            'bash', '-c', 
            'cd /home/app/ai-driven-soc && source venv/bin/activate && python3 -m streamlit run complete_operational_dashboard.py --server.port 8535 --server.address 0.0.0.0 --server.headless true --server.fileWatcherType none --logger.level error'
        ]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return process
    except Exception as e:
        print(f"[{datetime.now()}] Error starting dashboard: {e}")
        return None

def main():
    """Main monitoring loop"""
    print(f"[{datetime.now()}] Auto-restart dashboard monitor started")
    print(f"[{datetime.now()}] Monitoring port 8535...")
    
    dashboard_process = None
    restart_count = 0
    
    while True:
        try:
            if not check_dashboard_running():
                print(f"[{datetime.now()}] Dashboard not running on port 8535")
                
                # Kill any existing processes
                try:
                    subprocess.run(['pkill', '-f', 'streamlit run.*8535'], check=False)
                    time.sleep(2)
                except:
                    pass
                
                # Start dashboard
                dashboard_process = start_dashboard()
                restart_count += 1
                print(f"[{datetime.now()}] Dashboard restarted (attempt #{restart_count})")
                
                # Wait for dashboard to start
                time.sleep(10)
                
                if check_dashboard_running():
                    print(f"[{datetime.now()}] ✅ Dashboard successfully running on port 8535")
                else:
                    print(f"[{datetime.now()}] ❌ Dashboard failed to start")
            else:
                # Dashboard is running, check every 30 seconds
                time.sleep(30)
                
        except KeyboardInterrupt:
            print(f"\n[{datetime.now()}] Shutting down auto-restart monitor...")
            if dashboard_process:
                dashboard_process.terminate()
            sys.exit(0)
        except Exception as e:
            print(f"[{datetime.now()}] Error in monitoring loop: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()


