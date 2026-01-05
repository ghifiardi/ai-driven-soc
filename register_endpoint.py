#!/usr/bin/env python3
"""
Register Endpoint with ASGARD
Registers the current GCP instance as an endpoint for threat hunting campaigns.
"""

import sys
import os
import socket
from datetime import datetime
from asgard_orchestration_agent import ASGARDOrchestrationAgent, EndpointInfo

def get_local_endpoint_info():
    """Get information about the current endpoint."""
    hostname = socket.gethostname()
    
    # Get IP address
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip_address = s.getsockname()[0]
        s.close()
    except Exception:
        ip_address = "127.0.0.1"
    
    # Detect OS
    os_type = "linux"
    os_version = "RHEL 8"
    
    # Get agent version (if available)
    agent_version = "1.0.0"
    
    return {
        "endpoint_id": f"endpoint_{hostname}",
        "hostname": hostname,
        "ip_address": ip_address,
        "os_type": os_type,
        "os_version": os_version,
        "agent_version": agent_version,
        "last_seen": datetime.utcnow().isoformat(),
        "labels": {
            "env": "production",
            "role": "threat_hunting_test",
            "zone": "asia-southeast2-a",
            "project": "chronicle-dev-2be9"
        },
        "groups": ["production", "threat_hunting"],
        "status": "online",
        "capabilities": ["yara", "ioc", "process", "network", "filesystem"]
    }

def main():
    print("=" * 60)
    print("ASGARD Endpoint Registration")
    print("=" * 60)
    print()
    
    try:
        # Initialize ASGARD
        print("Initializing ASGARD agent...")
        asgard = ASGARDOrchestrationAgent()
        print(f"✓ ASGARD initialized")
        print()
        
        # Get endpoint information
        print("Collecting endpoint information...")
        endpoint_data = get_local_endpoint_info()
        print(f"  Hostname: {endpoint_data['hostname']}")
        print(f"  IP Address: {endpoint_data['ip_address']}")
        print(f"  OS: {endpoint_data['os_type']} {endpoint_data['os_version']}")
        print(f"  Labels: {endpoint_data['labels']}")
        print()
        
        # Create EndpointInfo object
        endpoint = EndpointInfo(**endpoint_data)
        
        # Register endpoint
        print("Registering endpoint with ASGARD...")
        success = asgard.register_endpoint(endpoint)
        
        if success:
            print(f"✓ Successfully registered endpoint: {endpoint.hostname}")
            print(f"  Endpoint ID: {endpoint.endpoint_id}")
            print()
            print(f"Total registered endpoints: {len(asgard.registered_endpoints)}")
            print()
            print("=" * 60)
            print("Endpoint Registration Complete!")
            print("=" * 60)
            return 0
        else:
            print("✗ Failed to register endpoint")
            return 1
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

