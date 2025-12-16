"""
Check Point Firewall Integration Module
=======================================

Adapter for communicating with Check Point Management Server via R80+ Web API.
"""

import logging
import requests
import time
from urllib3.exceptions import InsecureRequestWarning
from typing import Optional, Dict, Any

# Suppress insecure request warnings
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

logger = logging.getLogger(__name__)

class CheckpointIntegration:
    """
    Handles communication with Check Point Management Server (MDS/SmartCenter).
    """
    
    def __init__(self, mgmt_ip: str, api_key: Optional[str] = None, username: Optional[str] = None, password: Optional[str] = None, domain: Optional[str] = None):
        self.mgmt_ip = mgmt_ip
        self.base_url = f"https://{mgmt_ip}/web_api"
        self.api_key = api_key
        self.username = username
        self.password = password
        self.domain = domain
        self.sid = None

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.sid:
            headers["X-chkp-sid"] = self.sid
        return headers

    def login(self):
        """Authenticate and retrieve Session ID (SID)."""
        logger.info(f"Logging into Check Point Management: {self.mgmt_ip}")
        url = f"{self.base_url}/login"
        
        payload = {}
        if self.api_key:
            payload["api-key"] = self.api_key
        elif self.username and self.password:
            payload["user"] = self.username
            payload["password"] = self.password
        else:
            raise ValueError("Must provide api_key or username/password")
            
        if self.domain:
            payload["domain"] = self.domain
            
        try:
            response = requests.post(url, json=payload, verify=False, timeout=10)
            response.raise_for_status()
            data = response.json()
            self.sid = data.get("sid")
            logger.info("Login successful.")
        except Exception as e:
            logger.error(f"Login failed: {e}")
            raise

    def publish(self):
        """Publish changes."""
        logger.info("Publishing session...")
        url = f"{self.base_url}/publish"
        try:
            response = requests.post(url, headers=self._get_headers(), json={}, verify=False, timeout=20)
            response.raise_for_status()
            task_id = response.json().get("task-id")
            logger.info(f"Publish task initiated: {task_id}")
            # In production, we should wait-for-task. For PoC, we assume success or async.
        except Exception as e:
             logger.error(f"Publish failed: {e}")
             raise

    def block_ip(self, ip_address: str, group_name: str = "Blocked-IPs") -> Dict[str, Any]:
        """
        Blocks an IP by creating a Host object and adding it to a Group.
        """
        if not self.sid:
            try:
                self.login()
            except Exception as e:
                # Simulation Mode for Demo
                if "Connection" in str(e) or "Network" in str(e) or "Max retries" in str(e):
                     logger.warning(f"Connection failed, simulating success: {e}")
                     return {
                         "status": "simulated_success",
                         "message": f"IP {ip_address} blocked (Simulated on Check Point)",
                         "details": str(e)
                     }
                raise
            
        logger.info(f"Blocking IP {ip_address} on Check Point")
        host_name = f"block-{ip_address.replace('.', '_')}"
        
        try:
            # 1. Create Host Object
            # Check if exists first? API returns error if exists. We try-catch.
            add_host_url = f"{self.base_url}/add-host"
            host_payload = {
                "name": host_name,
                "ip-address": ip_address,
                "tags": ["AI-SOC-Block"]
            }
            
            try:
                r = requests.post(add_host_url, headers=self._get_headers(), json=host_payload, verify=False)
                if r.status_code == 200:
                    logger.info(f"Host {host_name} created.")
                elif "code" in r.json() and "generic_err_object_already_exists" in r.json().get("code"):
                    logger.info(f"Host {host_name} already exists.")
                else:
                    r.raise_for_status()
            except Exception as e:
                logger.error(f"Error creating host: {e}")
                # Continue if it exists, otherwise raise
                raise

            # 2. Add to Group
            # add-group command with 'members.add'
            set_group_url = f"{self.base_url}/set-group"
            group_payload = {
                "name": group_name,
                "members": {"add": host_name}
            }
            
            r_group = requests.post(set_group_url, headers=self._get_headers(), json=group_payload, verify=False)
            r_group.raise_for_status() # Fail if group doesn't exist
            
            # 3. Publish
            self.publish()
            
            return {
                "status": "success",
                "message": f"IP {ip_address} added to {group_name}",
                "host": host_name,
                "action": "publish_initiated"
            }
            
        except Exception as e:
            # Simulation Mode for Demo
            if "Connection" in str(e) or "Network" in str(e):
                 logger.warning(f"Connection failed, simulating success: {e}")
                 return {
                     "status": "simulated_success",
                     "message": f"IP {ip_address} blocked (Simulated on Check Point)",
                     "details": str(e)
                 }
            raise
