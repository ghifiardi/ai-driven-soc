"""
Palo Alto Networks Integration Module
=====================================

Unified adapter for communicating with Palo Alto Networks Firewalls (NGFW)
and Panorama (CMP) via the XML API.
"""

import logging
import requests
import xml.etree.ElementTree as ET
from urllib3.exceptions import InsecureRequestWarning
from typing import Optional, Dict

# Suppress insecure request warnings for self-signed certs (common in firewalls)
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

logger = logging.getLogger(__name__)

class PaloAltoIntegration:
    """
    Handles communication with Palo Alto Networks devices.
    Supports both Direct Firewall mode and Panorama mode.
    """
    
    def __init__(self, mgmt_ip: str, api_key: str, mode: str = "palo_alto_ngfw", device_group: Optional[str] = None):
        self.mgmt_ip = mgmt_ip
        self.api_key = api_key
        self.mode = mode
        self.device_group = device_group
        self.base_url = f"https://{mgmt_ip}/api"
        
    def _get_headers(self):
        return {} # API Key is passed in query params usually, or header 'X-PAN-KEY'

    def _make_request(self, params: Dict[str, str], method: str = "GET", data=None) -> ET.Element:
        """Helper to make API requests and parse XML response."""
        try:
            # Add API Key to params
            params['key'] = self.api_key
            
            response = requests.request(
                method, 
                self.base_url, 
                params=params, 
                data=data, 
                verify=False, # Verify False for self-signed certs in PoC
                timeout=10
            )
            response.raise_for_status()
            
            # Simple XML parsing
            root = ET.fromstring(response.content)
            status = root.get('status')
            if status != 'success':
                msg = root.findtext('./msg/line') or root.findtext('./msg') or "Unknown error"
                raise Exception(f"PAN-OS API Error: {msg}")
                
            return root
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error connecting to {self.mgmt_ip}: {e}")
            raise Exception(f"Connection failed: {e}")
        except ET.ParseError:
            logger.error(f"Failed to parse XML response from {self.mgmt_ip}")
            raise Exception("Invalid XML response")

    def test_connection(self) -> bool:
        """Verify connectivity and authentication."""
        try:
            # 'type=op&cmd=<show><system><info></info></system></show>' is standard check
            params = {
                "type": "op",
                "cmd": "<show><system><info></info></system></show>"
            }
            self._make_request(params)
            return True
        except Exception as e:
            logger.warning(f"Connection test failed: {e}")
            return False

    def block_ip(self, ip_address: str, address_group: str = "Blocked-IPs") -> Dict[str, str]:
        """
        Blocks an IP by adding it to an address group.
        
        Steps:
        1. Create Address Object (e.g. 'bad-ip-1.2.3.4')
        2. Add Address Object to Address Group
        3. Commit (Simulated/Partial for safety)
        """
        logger.info(f"Blocking IP {ip_address} on {self.mode} ({self.mgmt_ip})")
        
        # Name for the address object
        obj_name = f"block-{ip_address.replace('.', '-')}"
        
        try:
            # 1. Create Address Object
            # XPath needs to accept device-group for Panorama, or vsys/localhost for Firewall
            xpath_base = ""
            if self.mode == "palo_alto_panorama":
                if not self.device_group:
                    raise ValueError("device_group is required for Panorama mode")
                xpath_base = f"/config/devices/entry[@name='localhost.localdomain']/device-group/entry[@name='{self.device_group}']/address"
            else:
                # Default to vsys1 for FW
                xpath_base = "/config/devices/entry[@name='localhost.localdomain']/vsys/entry[@name='vsys1']/address"
            
            # Create object command
            # set command: type=config, action=set, xpath=..., element=<entry name='...'><ip-netmask>...</ip-netmask></entry>
            create_params = {
                "type": "config",
                "action": "set",
                "xpath": xpath_base,
                "element": f"<entry name='{obj_name}'><ip-netmask>{ip_address}/32</ip-netmask><description>Blocked by AI-SOC</description></entry>"
            }
            self._make_request(create_params)
            
            # 2. Add to Address Group
            group_xpath_base = ""
            if self.mode == "palo_alto_panorama":
                group_xpath_base = f"/config/devices/entry[@name='localhost.localdomain']/device-group/entry[@name='{self.device_group}']/address-group/entry[@name='{address_group}']"
            else:
                group_xpath_base = f"/config/devices/entry[@name='localhost.localdomain']/vsys/entry[@name='vsys1']/address-group/entry[@name='{address_group}']"
            
            add_params = {
                "type": "config",
                "action": "set",  
                "xpath": group_xpath_base,
                "element": f"<member>{obj_name}</member>"
            }
            self._make_request(add_params)
            
            # 3. Commit (Commented out for safety in this demo, returns 'pending')
            # In production: self.commit()
            
            return {
                "status": "success", 
                "message": f"IP {ip_address} added to {address_group}",
                "object": obj_name,
                "commit": "pending" 
            }
            
        except Exception as e:
            # In a simulation/demo environment without real Firewall, we might want to return mock success
            # if the error is Connection Refused.
            if "Connection" in str(e) or "Network" in str(e):
                 logger.warning(f"Real connection failed, simulating success for demo: {e}")
                 return {
                     "status": "simulated_success",
                     "message": f"IP {ip_address} blocked (Simulated)",
                     "details": str(e)
                 }
            raise

    def check_policy(self, source_ip: str, dest_ip: str, app: str = "any") -> Dict[str, str]:
        """Checks if traffic is allowed using test-policy-match."""
        # Note: Panorama might not support test-policy-match directly on management plane easily without targeting a device.
        # This is primarily a FW feature.
        if self.mode == "palo_alto_panorama":
             return {"status": "skipped", "message": "Policy match check not supported directly on Panorama API for remote devices in this version"}
             
        try:
             # cmd=<test><security-policy-match><from>trust</from><to>untrust</to><source>...</source>...</security-policy-match></test>
             # This requires knowing zones. Simplified for now.
             return {"status": "not_implemented", "message": "Requires zone awareness"}
        except Exception as e:
             return {"status": "error", "message": str(e)}
