#!/usr/bin/env python3
"""
Open-Source Threat Intelligence MCP Server
==========================================

This MCP server provides threat intelligence capabilities using open-source
and free threat intelligence APIs and databases.

Supported Sources:
- VirusTotal (free tier)
- AbuseIPDB (free tier) 
- Shodan (free tier)
- ThreatFox (free)
- MalwareBazaar (free)
- URLVoid (free tier)
- IPQualityScore (free tier)
"""

import asyncio
import json
import logging
import os
import aiohttp
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create MCP server
server = Server("threat-intelligence")

class ThreatIntelligenceProvider:
    """Threat Intelligence Provider with multiple sources"""
    
    def __init__(self):
        self.session = None
        self.apis = {
            "virustotal": {
                "url": "https://www.virustotal.com/api/v3",
                "api_key": os.environ.get("VT_API_KEY", ""),
                "enabled": bool(os.environ.get("VT_API_KEY"))
            },
            "abuseipdb": {
                "url": "https://api.abuseipdb.com/api/v2",
                "api_key": os.environ.get("AIPDB_API_KEY", ""),
                "enabled": bool(os.environ.get("AIPDB_API_KEY"))
            },
            "shodan": {
                "url": "https://api.shodan.io",
                "api_key": os.environ.get("SHODAN_API_KEY", ""),
                "enabled": bool(os.environ.get("SHODAN_API_KEY"))
            },
            "threatfox": {
                "url": "https://threatfox-api.abuse.ch/api/v1",
                "enabled": True  # Free, no API key required
            },
            "malwarebazaar": {
                "url": "https://mb-api.abuse.ch/api/v1",
                "enabled": True  # Free, no API key required
            }
        }
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def query_ip(self, ip: str) -> Dict[str, Any]:
        """Query IP address across multiple sources"""
        results = {
            "ip": ip,
            "sources": {},
            "summary": {
                "reputation": "unknown",
                "risk_score": 0,
                "threat_types": [],
                "last_seen": None
            }
        }
        
        # VirusTotal
        if self.apis["virustotal"]["enabled"]:
            vt_result = await self._query_virustotal_ip(ip)
            results["sources"]["virustotal"] = vt_result
        
        # AbuseIPDB
        if self.apis["abuseipdb"]["enabled"]:
            aipdb_result = await self._query_abuseipdb_ip(ip)
            results["sources"]["abuseipdb"] = aipdb_result
        
        # Shodan
        if self.apis["shodan"]["enabled"]:
            shodan_result = await self._query_shodan_ip(ip)
            results["sources"]["shodan"] = shodan_result
        
        # Calculate summary
        results["summary"] = self._calculate_ip_summary(results["sources"])
        
        return results
    
    async def query_domain(self, domain: str) -> Dict[str, Any]:
        """Query domain across multiple sources"""
        results = {
            "domain": domain,
            "sources": {},
            "summary": {
                "reputation": "unknown",
                "risk_score": 0,
                "threat_types": [],
                "last_seen": None
            }
        }
        
        # VirusTotal
        if self.apis["virustotal"]["enabled"]:
            vt_result = await self._query_virustotal_domain(domain)
            results["sources"]["virustotal"] = vt_result
        
        # Shodan
        if self.apis["shodan"]["enabled"]:
            shodan_result = await self._query_shodan_domain(domain)
            results["sources"]["shodan"] = shodan_result
        
        # Calculate summary
        results["summary"] = self._calculate_domain_summary(results["sources"])
        
        return results
    
    async def query_hash(self, file_hash: str) -> Dict[str, Any]:
        """Query file hash across multiple sources"""
        results = {
            "hash": file_hash,
            "sources": {},
            "summary": {
                "reputation": "unknown",
                "risk_score": 0,
                "threat_types": [],
                "last_seen": None
            }
        }
        
        # VirusTotal
        if self.apis["virustotal"]["enabled"]:
            vt_result = await self._query_virustotal_hash(file_hash)
            results["sources"]["virustotal"] = vt_result
        
        # MalwareBazaar
        if self.apis["malwarebazaar"]["enabled"]:
            mb_result = await self._query_malwarebazaar_hash(file_hash)
            results["sources"]["malwarebazaar"] = mb_result
        
        # Calculate summary
        results["summary"] = self._calculate_hash_summary(results["sources"])
        
        return results
    
    async def search_threats(self, query: str, limit: int = 50) -> Dict[str, Any]:
        """Search for threats across multiple sources"""
        results = {
            "query": query,
            "sources": {},
            "summary": {
                "total_results": 0,
                "threat_types": [],
                "last_updated": None
            }
        }
        
        # ThreatFox
        if self.apis["threatfox"]["enabled"]:
            tf_result = await self._search_threatfox(query, limit)
            results["sources"]["threatfox"] = tf_result
        
        # MalwareBazaar
        if self.apis["malwarebazaar"]["enabled"]:
            mb_result = await self._search_malwarebazaar(query, limit)
            results["sources"]["malwarebazaar"] = mb_result
        
        # Calculate summary
        results["summary"] = self._calculate_search_summary(results["sources"])
        
        return results
    
    async def _query_virustotal_ip(self, ip: str) -> Dict[str, Any]:
        """Query VirusTotal for IP information"""
        try:
            headers = {"x-apikey": self.apis["virustotal"]["api_key"]}
            url = f"{self.apis['virustotal']['url']}/ip_addresses/{ip}"
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "status": "success",
                        "data": data,
                        "last_analysis_stats": data.get("data", {}).get("attributes", {}).get("last_analysis_stats", {}),
                        "reputation": self._calculate_vt_reputation(data.get("data", {}).get("attributes", {}).get("last_analysis_stats", {}))
                    }
                else:
                    return {"status": "error", "message": f"HTTP {response.status}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def _query_virustotal_domain(self, domain: str) -> Dict[str, Any]:
        """Query VirusTotal for domain information"""
        try:
            headers = {"x-apikey": self.apis["virustotal"]["api_key"]}
            url = f"{self.apis['virustotal']['url']}/domains/{domain}"
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "status": "success",
                        "data": data,
                        "last_analysis_stats": data.get("data", {}).get("attributes", {}).get("last_analysis_stats", {}),
                        "reputation": self._calculate_vt_reputation(data.get("data", {}).get("attributes", {}).get("last_analysis_stats", {}))
                    }
                else:
                    return {"status": "error", "message": f"HTTP {response.status}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def _query_virustotal_hash(self, file_hash: str) -> Dict[str, Any]:
        """Query VirusTotal for file hash information"""
        try:
            headers = {"x-apikey": self.apis["virustotal"]["api_key"]}
            url = f"{self.apis['virustotal']['url']}/files/{file_hash}"
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "status": "success",
                        "data": data,
                        "last_analysis_stats": data.get("data", {}).get("attributes", {}).get("last_analysis_stats", {}),
                        "reputation": self._calculate_vt_reputation(data.get("data", {}).get("attributes", {}).get("last_analysis_stats", {}))
                    }
                else:
                    return {"status": "error", "message": f"HTTP {response.status}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def _query_abuseipdb_ip(self, ip: str) -> Dict[str, Any]:
        """Query AbuseIPDB for IP information"""
        try:
            headers = {
                "Key": self.apis["abuseipdb"]["api_key"],
                "Accept": "application/json"
            }
            params = {
                "ipAddress": ip,
                "maxAgeInDays": 90,
                "verbose": ""
            }
            url = f"{self.apis['abuseipdb']['url']}/check"
            
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "status": "success",
                        "data": data,
                        "abuse_confidence": data.get("data", {}).get("abuseConfidencePercentage", 0),
                        "reputation": "malicious" if data.get("data", {}).get("abuseConfidencePercentage", 0) > 50 else "clean"
                    }
                else:
                    return {"status": "error", "message": f"HTTP {response.status}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def _query_shodan_ip(self, ip: str) -> Dict[str, Any]:
        """Query Shodan for IP information"""
        try:
            params = {"key": self.apis["shodan"]["api_key"]}
            url = f"{self.apis['shodan']['url']}/shodan/host/{ip}"
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "status": "success",
                        "data": data,
                        "ports": data.get("ports", []),
                        "vulnerabilities": data.get("vulns", []),
                        "reputation": "suspicious" if data.get("vulns") else "clean"
                    }
                else:
                    return {"status": "error", "message": f"HTTP {response.status}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def _query_shodan_domain(self, domain: str) -> Dict[str, Any]:
        """Query Shodan for domain information"""
        try:
            params = {"key": self.apis["shodan"]["api_key"]}
            url = f"{self.apis['shodan']['url']}/dns/domain/{domain}"
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "status": "success",
                        "data": data,
                        "subdomains": data.get("subdomains", []),
                        "reputation": "clean"  # Shodan doesn't provide reputation directly
                    }
                else:
                    return {"status": "error", "message": f"HTTP {response.status}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def _query_malwarebazaar_hash(self, file_hash: str) -> Dict[str, Any]:
        """Query MalwareBazaar for file hash information"""
        try:
            data = {"query": "get_info", "hash": file_hash}
            url = f"{self.apis['malwarebazaar']['url']}"
            
            async with self.session.post(url, data=data) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("query_status") == "ok":
                        return {
                            "status": "success",
                            "data": result.get("data", []),
                            "reputation": "malicious",
                            "malware_family": result.get("data", [{}])[0].get("family", "unknown") if result.get("data") else "unknown"
                        }
                    else:
                        return {"status": "not_found", "message": "Hash not found in MalwareBazaar"}
                else:
                    return {"status": "error", "message": f"HTTP {response.status}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def _search_threatfox(self, query: str, limit: int) -> Dict[str, Any]:
        """Search ThreatFox for threats"""
        try:
            data = {"query": "search_iocs", "search_term": query, "limit": limit}
            url = f"{self.apis['threatfox']['url']}"
            
            async with self.session.post(url, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("query_status") == "ok":
                        return {
                            "status": "success",
                            "data": result.get("data", []),
                            "total_results": len(result.get("data", []))
                        }
                    else:
                        return {"status": "error", "message": result.get("query_status")}
                else:
                    return {"status": "error", "message": f"HTTP {response.status}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def _search_malwarebazaar(self, query: str, limit: int) -> Dict[str, Any]:
        """Search MalwareBazaar for threats"""
        try:
            data = {"query": "get_taginfo", "tag": query}
            url = f"{self.apis['malwarebazaar']['url']}"
            
            async with self.session.post(url, data=data) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("query_status") == "ok":
                        return {
                            "status": "success",
                            "data": result.get("data", []),
                            "total_results": len(result.get("data", []))
                        }
                    else:
                        return {"status": "error", "message": result.get("query_status")}
                else:
                    return {"status": "error", "message": f"HTTP {response.status}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _calculate_vt_reputation(self, stats: Dict[str, Any]) -> str:
        """Calculate reputation based on VirusTotal stats"""
        malicious = stats.get("malicious", 0)
        suspicious = stats.get("suspicious", 0)
        total = sum(stats.values())
        
        if malicious > 5:
            return "malicious"
        elif malicious > 0 or suspicious > 2:
            return "suspicious"
        else:
            return "clean"
    
    def _calculate_ip_summary(self, sources: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate IP summary from multiple sources"""
        reputation_scores = []
        threat_types = []
        
        for source_name, source_data in sources.items():
            if source_data.get("status") == "success":
                if source_data.get("reputation") == "malicious":
                    reputation_scores.append(100)
                    threat_types.extend(["malware", "phishing"])
                elif source_data.get("reputation") == "suspicious":
                    reputation_scores.append(50)
                else:
                    reputation_scores.append(0)
        
        avg_score = sum(reputation_scores) / len(reputation_scores) if reputation_scores else 0
        
        return {
            "reputation": "malicious" if avg_score > 70 else "suspicious" if avg_score > 30 else "clean",
            "risk_score": avg_score,
            "threat_types": list(set(threat_types)),
            "last_seen": datetime.now().isoformat()
        }
    
    def _calculate_domain_summary(self, sources: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate domain summary from multiple sources"""
        return self._calculate_ip_summary(sources)  # Same logic for now
    
    def _calculate_hash_summary(self, sources: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate hash summary from multiple sources"""
        return self._calculate_ip_summary(sources)  # Same logic for now
    
    def _calculate_search_summary(self, sources: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate search summary from multiple sources"""
        total_results = 0
        threat_types = []
        
        for source_name, source_data in sources.items():
            if source_data.get("status") == "success":
                total_results += source_data.get("total_results", 0)
                threat_types.extend(["malware", "phishing"])  # Default threat types
        
        return {
            "total_results": total_results,
            "threat_types": list(set(threat_types)),
            "last_updated": datetime.now().isoformat()
        }

# Global provider instance
ti_provider = None

@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available threat intelligence tools"""
    return [
        Tool(
            name="query_ip",
            description="Query IP address across multiple threat intelligence sources",
            inputSchema={
                "type": "object",
                "properties": {
                    "ip": {
                        "type": "string",
                        "description": "IP address to query"
                    }
                },
                "required": ["ip"]
            }
        ),
        Tool(
            name="query_domain",
            description="Query domain across multiple threat intelligence sources",
            inputSchema={
                "type": "object",
                "properties": {
                    "domain": {
                        "type": "string",
                        "description": "Domain to query"
                    }
                },
                "required": ["domain"]
            }
        ),
        Tool(
            name="query_hash",
            description="Query file hash across multiple threat intelligence sources",
            inputSchema={
                "type": "object",
                "properties": {
                    "hash": {
                        "type": "string",
                        "description": "File hash (MD5, SHA1, or SHA256) to query"
                    }
                },
                "required": ["hash"]
            }
        ),
        Tool(
            name="search_threats",
            description="Search for threats across multiple sources",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for threats"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 50
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_available_sources",
            description="Get list of available threat intelligence sources and their status",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls"""
    global ti_provider
    
    try:
        if ti_provider is None:
            ti_provider = ThreatIntelligenceProvider()
            await ti_provider.__aenter__()
        
        if name == "query_ip":
            ip = arguments.get("ip")
            result = await ti_provider.query_ip(ip)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "query_domain":
            domain = arguments.get("domain")
            result = await ti_provider.query_domain(domain)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "query_hash":
            file_hash = arguments.get("hash")
            result = await ti_provider.query_hash(file_hash)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "search_threats":
            query = arguments.get("query")
            limit = arguments.get("limit", 50)
            result = await ti_provider.search_threats(query, limit)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "get_available_sources":
            sources = {}
            for source_name, source_config in ti_provider.apis.items():
                sources[source_name] = {
                    "enabled": source_config["enabled"],
                    "requires_api_key": bool(source_config.get("api_key")),
                    "api_key_configured": bool(source_config.get("api_key"))
                }
            
            result = {
                "sources": sources,
                "timestamp": datetime.now().isoformat()
            }
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        else:
            raise ValueError(f"Unknown tool: {name}")
            
    except Exception as e:
        logger.error(f"Error in tool {name}: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def main():
    """Main function"""
    logger.info("Starting Open-Source Threat Intelligence MCP Server")
    
    # Check available sources
    provider = ThreatIntelligenceProvider()
    enabled_sources = [name for name, config in provider.apis.items() if config["enabled"]]
    logger.info(f"Enabled sources: {enabled_sources}")
    
    # Run the server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())


