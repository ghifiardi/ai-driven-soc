#!/usr/bin/env python3
"""
SOC Service Manager
==================

This script helps manage SOC services and dashboards on the remote server.
Based on the service status check, it provides recommendations and commands.
"""

import json
import subprocess
import sys
from typing import Dict, List, Any

def load_service_status() -> Dict[str, Any]:
    """Load the service status from the JSON file"""
    try:
        with open("soc_services_status.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ Service status file not found. Run check_soc_services.py first.")
        return {}

def generate_startup_commands(status: Dict[str, Any]) -> List[str]:
    """Generate commands to start SOC services"""
    commands = []
    
    print("🚀 SOC Service Startup Commands")
    print("=" * 50)
    
    # Based on the files found, generate startup commands
    dashboard_files = [
        "streamlit_soc_dashboard.py",
        "streamlit_soc_dashboard_minimal.py", 
        "realtime_soc_dashboard.py",
        "dark_mode_fixed_dashboard.py",
        "fast_loading_dashboard.py",
        "fixed_streamlit_dashboard.py",
        "simple_working_dashboard.py"
    ]
    
    ports = [8501, 8502, 8503, 8504, 8505, 8506, 8507]
    
    print("📋 Dashboard Startup Commands:")
    print("-" * 30)
    
    for i, dashboard in enumerate(dashboard_files):
        port = ports[i] if i < len(ports) else 8500 + i + 1
        
        # Check if this port is already in use
        port_status = status.get("streamlit_apps", {}).get(f"port_{port}", {})
        if port_status.get("accessible"):
            print(f"  ✅ Port {port}: Already running")
        else:
            command = f"streamlit run {dashboard} --server.port {port} --server.address 0.0.0.0"
            commands.append(command)
            print(f"  🔧 Port {port}: {command}")
    
    return commands

def generate_management_commands() -> Dict[str, List[str]]:
    """Generate management commands for SOC services"""
    
    commands = {
        "start_services": [
            "# Start main SOC dashboard on port 8501",
            "streamlit run streamlit_soc_dashboard.py --server.port 8501 --server.address 0.0.0.0 &",
            "",
            "# Start minimal dashboard on port 8502", 
            "streamlit run streamlit_soc_dashboard_minimal.py --server.port 8502 --server.address 0.0.0.0 &",
            "",
            "# Start real-time dashboard on port 8504",
            "streamlit run realtime_soc_dashboard.py --server.port 8504 --server.address 0.0.0.0 &",
            "",
            "# Start BigQuery dashboard on port 8505",
            "streamlit run local_bigquery_app.py --server.port 8505 --server.address 0.0.0.0 &"
        ],
        
        "check_processes": [
            "# Check running Streamlit processes",
            "ps aux | grep streamlit",
            "",
            "# Check port usage",
            "netstat -tulpn | grep :85",
            "",
            "# Check system resources",
            "top -p $(pgrep streamlit | tr '\\n' ',' | sed 's/,$//')"
        ],
        
        "stop_services": [
            "# Stop all Streamlit processes",
            "pkill -f streamlit",
            "",
            "# Stop specific dashboard by port",
            "lsof -ti:8501 | xargs kill -9",
            "",
            "# Stop all Python processes (be careful!)",
            "# pkill -f python"
        ],
        
        "restart_services": [
            "# Stop all services first",
            "pkill -f streamlit",
            "sleep 5",
            "",
            "# Start services again",
            "streamlit run streamlit_soc_dashboard.py --server.port 8501 --server.address 0.0.0.0 &",
            "streamlit run streamlit_soc_dashboard_minimal.py --server.port 8502 --server.address 0.0.0.0 &"
        ],
        
        "monitor_services": [
            "# Monitor dashboard logs",
            "tail -f ~/.streamlit/logs/streamlit.log",
            "",
            "# Monitor system resources",
            "htop",
            "",
            "# Check disk space",
            "df -h",
            "",
            "# Check memory usage",
            "free -h"
        ],
        
        "backup_services": [
            "# Backup dashboard files",
            "tar -czf soc_dashboards_backup_$(date +%Y%m%d_%H%M%S).tar.gz *.py *.md *.txt",
            "",
            "# Backup configuration files",
            "cp -r config/ config_backup_$(date +%Y%m%d_%H%M%S)/",
            "",
            "# Backup service account keys",
            "cp *.json keys_backup_$(date +%Y%m%d_%H%M%S)/"
        ]
    }
    
    return commands

def show_current_status(status: Dict[str, Any]):
    """Show current service status"""
    if not status:
        return
    
    print("📊 Current SOC Services Status")
    print("=" * 50)
    
    # Show accessible services
    accessible_streamlit = status.get("summary", {}).get("accessible_streamlit", 0)
    accessible_services = status.get("summary", {}).get("accessible_services", 0)
    accessible_dashboards = status.get("summary", {}).get("accessible_dashboards", 0)
    
    print(f"🌐 Streamlit Apps: {accessible_streamlit} running")
    print(f"🔧 System Services: {accessible_services} running")
    print(f"📊 SOC Dashboards: {accessible_dashboards} accessible")
    
    # Show specific accessible services
    print("\n✅ Currently Accessible:")
    print("-" * 30)
    
    streamlit_apps = status.get("streamlit_apps", {})
    for name, result in streamlit_apps.items():
        if result.get("accessible"):
            port = result.get("port")
            print(f"  🌐 Streamlit App: Port {port}")
    
    dashboards = status.get("soc_dashboards", {})
    for name, result in dashboards.items():
        if result.get("accessible"):
            url = result.get("url", "")
            status_code = result.get("status_code", "")
            print(f"  📊 Dashboard: {name} ({url}) - Status: {status_code}")

def main():
    """Main function"""
    print("🛠️  SOC Service Manager")
    print("=" * 50)
    
    # Load current status
    status = load_service_status()
    
    if status:
        show_current_status(status)
        print()
    
    # Generate commands
    commands = generate_management_commands()
    
    print("📋 Available Commands:")
    print("-" * 30)
    
    for category, cmd_list in commands.items():
        print(f"\n🔧 {category.replace('_', ' ').title()}:")
        print("-" * 20)
        for cmd in cmd_list:
            if cmd.startswith("#"):
                print(f"  {cmd}")
            else:
                print(f"  {cmd}")
    
    print("\n💡 Quick Actions:")
    print("-" * 20)
    print("  1. Start main dashboard: streamlit run streamlit_soc_dashboard.py --server.port 8501 --server.address 0.0.0.0")
    print("  2. Check running processes: ps aux | grep streamlit")
    print("  3. Stop all dashboards: pkill -f streamlit")
    print("  4. Restart services: pkill -f streamlit && sleep 5 && streamlit run streamlit_soc_dashboard.py --server.port 8501 --server.address 0.0.0.0 &")
    
    print("\n🎯 Based on Current Status:")
    print("-" * 30)
    
    if status:
        accessible_dashboards = status.get("summary", {}).get("accessible_dashboards", 0)
        
        if accessible_dashboards > 0:
            print("  ✅ Some dashboards are running - you can access them")
            print("  📊 Feedback dashboard is accessible for CSV extraction")
        else:
            print("  ⚠️  No dashboards are currently running")
            print("  🔧 Consider starting the main dashboard")
    
    print("\n📁 Save Commands to File:")
    print("-" * 30)
    
    # Save commands to a shell script
    with open("start_soc_services.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# SOC Services Startup Script\n")
        f.write("# Generated by SOC Service Manager\n\n")
        
        for category, cmd_list in commands.items():
            f.write(f"\n# {category.replace('_', ' ').title()}\n")
            for cmd in cmd_list:
                if not cmd.startswith("#") and cmd.strip():
                    f.write(f"{cmd}\n")
    
    print("  ✅ Commands saved to: start_soc_services.sh")
    print("  🔧 Make executable: chmod +x start_soc_services.sh")
    print("  🚀 Run script: ./start_soc_services.sh")

if __name__ == "__main__":
    main()


