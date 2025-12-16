#!/usr/bin/env python3
"""
Simulated Chronicle MCP Server
==============================

This is a simplified MCP server that simulates Chronicle functionality
for testing and development purposes.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create MCP server
server = Server("simulated-chronicle")

# Simulated Chronicle data
SIMULATED_DATA = {
    "detections": [
        {
            "id": "det_001",
            "name": "Suspicious Network Connection",
            "description": "Detects unusual network connections to known malicious IPs",
            "severity": "HIGH",
            "status": "ACTIVE",
            "last_updated": "2024-01-15T10:30:00Z"
        },
        {
            "id": "det_002", 
            "name": "Malware Execution",
            "description": "Detects execution of known malware samples",
            "severity": "CRITICAL",
            "status": "ACTIVE",
            "last_updated": "2024-01-15T09:15:00Z"
        }
    ],
    "alerts": [
        {
            "id": "alert_001",
            "detection_id": "det_001",
            "timestamp": "2024-01-15T14:30:00Z",
            "severity": "HIGH",
            "source_ip": "192.168.1.100",
            "destination_ip": "8.8.8.8",
            "description": "Suspicious connection to external IP",
            "status": "NEW"
        },
        {
            "id": "alert_002",
            "detection_id": "det_002", 
            "timestamp": "2024-01-15T13:45:00Z",
            "severity": "CRITICAL",
            "source_ip": "10.0.0.50",
            "destination_ip": "malware.example.com",
            "description": "Malware execution detected",
            "status": "INVESTIGATING"
        }
    ],
    "threat_intelligence": {
        "8.8.8.8": {
            "reputation": "GOOD",
            "country": "US",
            "asn": "AS15169",
            "description": "Google DNS"
        },
        "malware.example.com": {
            "reputation": "MALICIOUS",
            "country": "RU",
            "asn": "AS12345",
            "description": "Known malware distribution site",
            "threat_types": ["malware", "phishing"]
        }
    }
}

@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available Chronicle tools"""
    return [
        Tool(
            name="chronicle_query",
            description="Query Chronicle for detection rules and alerts",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Chronicle query string"
                    },
                    "time_range": {
                        "type": "object",
                        "properties": {
                            "start_time": {"type": "string"},
                            "end_time": {"type": "string"}
                        }
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 100
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_detection_rules",
            description="Get Chronicle detection rules",
            inputSchema={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["ACTIVE", "INACTIVE", "ALL"],
                        "description": "Filter by rule status",
                        "default": "ACTIVE"
                    },
                    "severity": {
                        "type": "string", 
                        "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL", "ALL"],
                        "description": "Filter by severity",
                        "default": "ALL"
                    }
                }
            }
        ),
        Tool(
            name="get_threat_intelligence",
            description="Get threat intelligence for indicators",
            inputSchema={
                "type": "object",
                "properties": {
                    "indicators": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of indicators to query"
                    }
                },
                "required": ["indicators"]
            }
        ),
        Tool(
            name="investigate_incident",
            description="Conduct investigation for an incident",
            inputSchema={
                "type": "object",
                "properties": {
                    "incident_id": {
                        "type": "string",
                        "description": "Incident ID to investigate"
                    },
                    "scope": {
                        "type": "string",
                        "enum": ["basic", "deep", "forensic"],
                        "description": "Investigation scope",
                        "default": "basic"
                    }
                },
                "required": ["incident_id"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls"""
    try:
        if name == "chronicle_query":
            return await handle_chronicle_query(arguments)
        elif name == "get_detection_rules":
            return await handle_get_detection_rules(arguments)
        elif name == "get_threat_intelligence":
            return await handle_get_threat_intelligence(arguments)
        elif name == "investigate_incident":
            return await handle_investigate_incident(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")
    except Exception as e:
        logger.error(f"Error in tool {name}: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def handle_chronicle_query(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle Chronicle query"""
    query = arguments.get("query", "")
    max_results = arguments.get("max_results", 100)
    
    logger.info(f"Chronicle query: {query}")
    
    # Simulate query processing
    results = []
    
    # Simple keyword matching for demo
    if "network" in query.lower():
        results.extend(SIMULATED_DATA["alerts"][:1])
    if "malware" in query.lower():
        results.extend(SIMULATED_DATA["alerts"][1:])
    
    # If no specific matches, return some sample data
    if not results:
        results = SIMULATED_DATA["alerts"]
    
    # Limit results
    results = results[:max_results]
    
    response = {
        "query": query,
        "results": results,
        "total_count": len(results),
        "timestamp": datetime.now().isoformat()
    }
    
    return [TextContent(type="text", text=json.dumps(response, indent=2))]

async def handle_get_detection_rules(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle get detection rules"""
    status = arguments.get("status", "ACTIVE")
    severity = arguments.get("severity", "ALL")
    
    logger.info(f"Getting detection rules: status={status}, severity={severity}")
    
    rules = SIMULATED_DATA["detections"]
    
    # Filter by status
    if status != "ALL":
        rules = [r for r in rules if r["status"] == status]
    
    # Filter by severity
    if severity != "ALL":
        rules = [r for r in rules if r["severity"] == severity]
    
    response = {
        "rules": rules,
        "total_count": len(rules),
        "filters": {"status": status, "severity": severity},
        "timestamp": datetime.now().isoformat()
    }
    
    return [TextContent(type="text", text=json.dumps(response, indent=2))]

async def handle_get_threat_intelligence(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle get threat intelligence"""
    indicators = arguments.get("indicators", [])
    
    logger.info(f"Getting threat intelligence for indicators: {indicators}")
    
    ti_results = {}
    
    for indicator in indicators:
        if indicator in SIMULATED_DATA["threat_intelligence"]:
            ti_results[indicator] = SIMULATED_DATA["threat_intelligence"][indicator]
        else:
            ti_results[indicator] = {
                "reputation": "UNKNOWN",
                "description": "No intelligence available"
            }
    
    response = {
        "indicators": ti_results,
        "total_indicators": len(indicators),
        "timestamp": datetime.now().isoformat()
    }
    
    return [TextContent(type="text", text=json.dumps(response, indent=2))]

async def handle_investigate_incident(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle investigate incident"""
    incident_id = arguments.get("incident_id")
    scope = arguments.get("scope", "basic")
    
    logger.info(f"Investigating incident {incident_id} with scope {scope}")
    
    # Simulate investigation based on scope
    investigation = {
        "incident_id": incident_id,
        "scope": scope,
        "status": "completed",
        "findings": [],
        "timeline": [],
        "recommendations": []
    }
    
    if scope == "basic":
        investigation["findings"] = [
            "Incident confirmed",
            "Source identified",
            "Impact assessment completed"
        ]
        investigation["recommendations"] = [
            "Monitor affected systems",
            "Update detection rules"
        ]
    elif scope == "deep":
        investigation["findings"] = [
            "Incident confirmed",
            "Source identified", 
            "Impact assessment completed",
            "Attack vector analyzed",
            "IOCs extracted"
        ]
        investigation["recommendations"] = [
            "Monitor affected systems",
            "Update detection rules",
            "Block malicious IPs",
            "Scan for lateral movement"
        ]
    elif scope == "forensic":
        investigation["findings"] = [
            "Incident confirmed",
            "Source identified",
            "Impact assessment completed", 
            "Attack vector analyzed",
            "IOCs extracted",
            "Memory analysis completed",
            "Network forensics completed",
            "Timeline reconstruction completed"
        ]
        investigation["recommendations"] = [
            "Monitor affected systems",
            "Update detection rules",
            "Block malicious IPs",
            "Scan for lateral movement",
            "Implement additional controls",
            "Conduct security awareness training"
        ]
    
    # Add timeline
    investigation["timeline"] = [
        {"timestamp": "2024-01-15T10:00:00Z", "event": "Incident detected"},
        {"timestamp": "2024-01-15T10:15:00Z", "event": "Initial assessment started"},
        {"timestamp": datetime.now().isoformat(), "event": f"Investigation completed ({scope})"}
    ]
    
    return [TextContent(type="text", text=json.dumps(investigation, indent=2))]

async def main():
    """Main function"""
    logger.info("Starting Simulated Chronicle MCP Server")
    
    # Run the server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())


