#!/usr/bin/env python3
"""
SOC Services Status Checker
==========================

This script checks the status of various SOC services and dashboards
running on the remote server.
"""

import requests
import subprocess
import time
import json
from typing import Dict, List, Any
import concurrent.futures

def check_port_status(host: str, port: int, timeout: int = 5) -> Dict[str, Any]:
    """
    Check if a port is open and responding
    
    Args:
        host: Host IP or hostname
        port: Port number
        timeout: Connection timeout in seconds
        
    Returns:
        Dict with status information
    """
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            return {
                "port": port,
                "status": "open",
                "accessible": True,
                "error": None
            }
        else:
            return {
                "port": port,
                "status": "closed",
                "accessible": False,
                "error": f"Connection refused (code: {result})"
            }
    except Exception as e:
        return {
            "port": port,
            "status": "error",
            "accessible": False,
            "error": str(e)
        }

def check_http_service(url: str, timeout: int = 10) -> Dict[str, Any]:
    """
    Check if an HTTP service is responding
    
    Args:
        url: URL to check
        timeout: Request timeout in seconds
        
    Returns:
        Dict with service status
    """
    try:
        response = requests.get(url, timeout=timeout)
        return {
            "url": url,
            "status_code": response.status_code,
            "status": "online" if response.status_code == 200 else "error",
            "accessible": response.status_code == 200,
            "response_time": response.elapsed.total_seconds(),
            "content_length": len(response.content),
            "error": None
        }
    except requests.exceptions.Timeout:
        return {
            "url": url,
            "status_code": None,
            "status": "timeout",
            "accessible": False,
            "response_time": timeout,
            "content_length": 0,
            "error": "Request timeout"
        }
    except requests.exceptions.ConnectionError:
        return {
            "url": url,
            "status_code": None,
            "status": "connection_error",
            "accessible": False,
            "response_time": None,
            "content_length": 0,
            "error": "Connection error"
        }
    except Exception as e:
        return {
            "url": url,
            "status_code": None,
            "status": "error",
            "accessible": False,
            "response_time": None,
            "content_length": 0,
            "error": str(e)
        }

def check_streamlit_apps(host: str = "10.45.254.19") -> Dict[str, Any]:
    """
    Check common Streamlit dashboard ports
    
    Args:
        host: Host IP address
        
    Returns:
        Dict with Streamlit app statuses
    """
    # Common Streamlit ports
    streamlit_ports = [8501, 8502, 8503, 8504, 8505, 8506, 8507, 8508, 8509, 8510]
    
    results = {}
    
    print(f"🔍 Checking Streamlit apps on {host}...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Submit port checks
        future_to_port = {
            executor.submit(check_port_status, host, port): port 
            for port in streamlit_ports
        }
        
        # Collect results
        for future in concurrent.futures.as_completed(future_to_port):
            port = future_to_port[future]
            result = future.result()
            results[f"port_{port}"] = result
            
            if result["accessible"]:
                print(f"  ✅ Port {port}: Open")
            else:
                print(f"  ❌ Port {port}: Closed")
    
    return results

def check_common_services(host: str = "10.45.254.19") -> Dict[str, Any]:
    """
    Check common SOC services
    
    Args:
        host: Host IP address
        
    Returns:
        Dict with service statuses
    """
    # Common SOC service ports
    services = {
        "ssh": 22,
        "http": 80,
        "https": 443,
        "mysql": 3306,
        "postgresql": 5432,
        "redis": 6379,
        "elasticsearch": 9200,
        "kibana": 5601,
        "grafana": 3000,
        "prometheus": 9090,
        "nginx": 8080,
        "apache": 8080,
        "tomcat": 8080,
        "jenkins": 8080,
        "sonarqube": 9000,
        "jira": 8080,
        "confluence": 8090,
        "bitbucket": 7990
    }
    
    results = {}
    
    print(f"🔍 Checking common services on {host}...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        # Submit port checks
        future_to_service = {
            executor.submit(check_port_status, host, port): service 
            for service, port in services.items()
        }
        
        # Collect results
        for future in concurrent.futures.as_completed(future_to_service):
            service = future_to_service[future]
            result = future.result()
            results[service] = result
            
            if result["accessible"]:
                print(f"  ✅ {service}: Open (port {result['port']})")
            else:
                print(f"  ❌ {service}: Closed (port {result['port']})")
    
    return results

def check_soc_dashboards(host: str = "10.45.254.19") -> Dict[str, Any]:
    """
    Check specific SOC dashboard URLs
    
    Args:
        host: Host IP address
        
    Returns:
        Dict with dashboard statuses
    """
    # Based on the files in the directory, these are likely SOC dashboards
    dashboards = {
        "feedback": f"http://{host}:99/feedback",
        "soc_dashboard": f"http://{host}:8501",
        "soc_dashboard_alt": f"http://{host}:8502",
        "realtime_dashboard": f"http://{host}:8503",
        "bigquery_dashboard": f"http://{host}:8504",
        "minimal_dashboard": f"http://{host}:8505"
    }
    
    results = {}
    
    print(f"🔍 Checking SOC dashboards on {host}...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        # Submit HTTP checks
        future_to_dashboard = {
            executor.submit(check_http_service, url): name 
            for name, url in dashboards.items()
        }
        
        # Collect results
        for future in concurrent.futures.as_completed(future_to_dashboard):
            dashboard = future_to_dashboard[future]
            result = future.result()
            results[dashboard] = result
            
            if result["accessible"]:
                print(f"  ✅ {dashboard}: Online (Status: {result['status_code']}, Response: {result['response_time']:.2f}s)")
            else:
                print(f"  ❌ {dashboard}: Offline ({result['error']})")
    
    return results

def analyze_dashboard_files() -> Dict[str, Any]:
    """
    Analyze the dashboard files to understand what services should be running
    
    Returns:
        Dict with file analysis
    """
    # Based on the ls -la output provided
    dashboard_files = {
        "streamlit_soc_dashboard.py": "Main SOC dashboard",
        "streamlit_soc_dashboard_minimal.py": "Minimal SOC dashboard", 
        "realtime_soc_dashboard.py": "Real-time SOC dashboard",
        "dark_mode_fixed_dashboard.py": "Dark mode dashboard",
        "fast_loading_dashboard.py": "Fast loading dashboard",
        "fixed_streamlit_dashboard.py": "Fixed Streamlit dashboard",
        "simple_working_dashboard.py": "Simple working dashboard",
        "local_bigquery_app.py": "BigQuery integration",
        "data_sources.py": "Data sources configuration"
    }
    
    print("📋 SOC Dashboard Files Analysis:")
    print("=" * 50)
    
    for filename, description in dashboard_files.items():
        print(f"  📄 {filename}: {description}")
    
    return dashboard_files

def main():
    """Main function to check all SOC services"""
    host = "10.45.254.19"  # Based on the URL provided
    
    print("🚀 SOC Services Status Checker")
    print("=" * 60)
    print(f"🎯 Target Host: {host}")
    print(f"⏰ Check Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Analyze dashboard files
    files_analysis = analyze_dashboard_files()
    print()
    
    # Check Streamlit apps
    streamlit_status = check_streamlit_apps(host)
    print()
    
    # Check common services
    services_status = check_common_services(host)
    print()
    
    # Check SOC dashboards
    dashboards_status = check_soc_dashboards(host)
    print()
    
    # Summary
    print("📊 SUMMARY")
    print("=" * 60)
    
    # Count accessible services
    accessible_streamlit = sum(1 for result in streamlit_status.values() if result["accessible"])
    accessible_services = sum(1 for result in services_status.values() if result["accessible"])
    accessible_dashboards = sum(1 for result in dashboards_status.values() if result["accessible"])
    
    print(f"🔍 Streamlit Apps: {accessible_streamlit}/{len(streamlit_status)} accessible")
    print(f"🔧 Common Services: {accessible_services}/{len(services_status)} accessible")
    print(f"📊 SOC Dashboards: {accessible_dashboards}/{len(dashboards_status)} accessible")
    
    # Show accessible services
    print("\n✅ ACCESSIBLE SERVICES:")
    print("-" * 30)
    
    for name, result in streamlit_status.items():
        if result["accessible"]:
            print(f"  🌐 Streamlit App: {name} (Port {result['port']})")
    
    for name, result in services_status.items():
        if result["accessible"]:
            print(f"  🔧 Service: {name} (Port {result['port']})")
    
    for name, result in dashboards_status.items():
        if result["accessible"]:
            print(f"  📊 Dashboard: {name} (Status: {result['status_code']})")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    print("-" * 30)
    
    if accessible_dashboards == 0:
        print("  ⚠️  No dashboards are accessible. Check if Streamlit apps are running.")
        print("  🔧 Try starting dashboards with: streamlit run <dashboard_file>.py")
    
    if accessible_streamlit == 0:
        print("  ⚠️  No Streamlit apps are running. Start the SOC dashboards.")
    
    if dashboards_status.get("feedback", {}).get("accessible"):
        print("  ✅ Feedback dashboard is accessible - you can extract CSV data")
    
    # Save results to file
    results = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "host": host,
        "streamlit_apps": streamlit_status,
        "common_services": services_status,
        "soc_dashboards": dashboards_status,
        "files_analysis": files_analysis,
        "summary": {
            "accessible_streamlit": accessible_streamlit,
            "accessible_services": accessible_services,
            "accessible_dashboards": accessible_dashboards
        }
    }
    
    with open("soc_services_status.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to: soc_services_status.json")
    
    return results

if __name__ == "__main__":
    main()


