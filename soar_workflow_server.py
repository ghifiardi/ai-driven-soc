#!/usr/bin/env python3
"""
SOAR Workflow Server with MCP Integration
=========================================

This server provides SOAR (Security Orchestration, Automation and Response)
capabilities with MCP integration for threat intelligence and automated
response orchestration.

Features:
- RESTful API for SOAR workflow management
- MCP integration for threat intelligence enrichment
- Real-time workflow execution monitoring
- A2A communication for multi-agent coordination
- Advanced playbook automation with conditional logic
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import structlog

from cra_soar_mcp_agent import CRASOARMCPAgent, Incident, SOARPlaybook, PlaybookStatus

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger("SOAR_Workflow_Server")

# FastAPI app
app = FastAPI(
    title="SOAR Workflow Server",
    description="Security Orchestration, Automation and Response with MCP Integration",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class IncidentRequest(BaseModel):
    """Incident request model"""
    alert_id: str
    severity: str = Field(..., pattern="^(critical|high|medium|low|info)$")
    title: str
    description: str
    affected_assets: List[str] = []
    indicators: List[str] = []
    metadata: Dict[str, Any] = {}

class PlaybookExecutionRequest(BaseModel):
    """Playbook execution request model"""
    incident_id: str
    playbook_id: str
    parameters: Dict[str, Any] = {}

class ActionApprovalRequest(BaseModel):
    """Action approval request model"""
    workflow_id: str
    action_id: str
    approved: bool
    approver: str
    comments: Optional[str] = None

class WorkflowStatusResponse(BaseModel):
    """Workflow status response model"""
    workflow_id: str
    playbook_id: str
    incident_id: str
    status: str
    started_at: str
    completed_at: Optional[str] = None
    actions_completed: List[Dict[str, Any]] = []
    actions_failed: List[Dict[str, Any]] = []
    actions_pending_approval: List[Dict[str, Any]] = []
    current_step: int
    error: Optional[str] = None

class PlaybookResponse(BaseModel):
    """Playbook response model"""
    playbook_id: str
    name: str
    description: str
    trigger_conditions: Dict[str, Any]
    actions: List[Dict[str, Any]]
    enabled: bool

# Global agent instance
soar_agent: Optional[CRASOARMCPAgent] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the SOAR agent on startup"""
    global soar_agent
    try:
        soar_agent = CRASOARMCPAgent()
        logger.info("SOAR Workflow Server initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize SOAR agent: {e}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "mcp_enabled": soar_agent.config["mcp"]["enabled"] if soar_agent else False
    }

@app.get("/playbooks", response_model=List[PlaybookResponse])
async def list_playbooks():
    """List all available SOAR playbooks"""
    if not soar_agent:
        raise HTTPException(status_code=500, detail="SOAR agent not initialized")
    
    playbooks = []
    for playbook in soar_agent.soar_engine.playbooks.values():
        playbooks.append(PlaybookResponse(
            playbook_id=playbook.playbook_id,
            name=playbook.name,
            description=playbook.description,
            trigger_conditions=playbook.trigger_conditions,
            actions=[asdict(action) for action in playbook.actions],
            enabled=playbook.enabled
        ))
    
    return playbooks

@app.get("/playbooks/{playbook_id}", response_model=PlaybookResponse)
async def get_playbook(playbook_id: str):
    """Get specific playbook details"""
    if not soar_agent:
        raise HTTPException(status_code=500, detail="SOAR agent not initialized")
    
    playbook = soar_agent.soar_engine.playbooks.get(playbook_id)
    if not playbook:
        raise HTTPException(status_code=404, detail=f"Playbook {playbook_id} not found")
    
    return PlaybookResponse(
        playbook_id=playbook.playbook_id,
        name=playbook.name,
        description=playbook.description,
        trigger_conditions=playbook.trigger_conditions,
        actions=[asdict(action) for action in playbook.actions],
        enabled=playbook.enabled
    )

@app.post("/incidents", status_code=201)
async def create_incident(
    incident_request: IncidentRequest,
    background_tasks: BackgroundTasks
):
    """Create and process a new security incident"""
    if not soar_agent:
        raise HTTPException(status_code=500, detail="SOAR agent not initialized")
    
    try:
        # Convert request to incident data
        incident_data = {
            "alert_id": incident_request.alert_id,
            "severity": incident_request.severity,
            "title": incident_request.title,
            "description": incident_request.description,
            "affected_assets": incident_request.affected_assets,
            "indicators": incident_request.indicators,
            "metadata": incident_request.metadata
        }
        
        # Process incident in background
        background_tasks.add_task(process_incident_background, incident_data)
        
        return {
            "message": "Incident processing started",
            "incident_id": incident_request.alert_id,
            "status": "processing",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error creating incident: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/playbooks/{playbook_id}/execute")
async def execute_playbook(
    playbook_id: str,
    execution_request: PlaybookExecutionRequest,
    background_tasks: BackgroundTasks
):
    """Execute a specific playbook for an incident"""
    if not soar_agent:
        raise HTTPException(status_code=500, detail="SOAR agent not initialized")
    
    playbook = soar_agent.soar_engine.playbooks.get(playbook_id)
    if not playbook:
        raise HTTPException(status_code=404, detail=f"Playbook {playbook_id} not found")
    
    if not playbook.enabled:
        raise HTTPException(status_code=400, detail=f"Playbook {playbook_id} is disabled")
    
    try:
        # Create incident object (simplified for direct execution)
        incident = Incident(
            incident_id=execution_request.incident_id,
            alert_id=execution_request.incident_id,
            severity=soar_agent.soar_engine.config.get("default_severity", "medium"),
            title="Direct Playbook Execution",
            description=f"Direct execution of playbook {playbook_id}",
            affected_assets=[],
            indicators=[],
            threat_intelligence=[],
            containment_actions=[],
            playbook_id=playbook_id
        )
        
        # Execute playbook in background
        background_tasks.add_task(execute_playbook_background, playbook, incident)
        
        return {
            "message": "Playbook execution started",
            "playbook_id": playbook_id,
            "incident_id": execution_request.incident_id,
            "status": "executing",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error executing playbook: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/workflows", response_model=List[WorkflowStatusResponse])
async def list_workflows():
    """List all running workflows"""
    if not soar_agent:
        raise HTTPException(status_code=500, detail="SOAR agent not initialized")
    
    workflows = []
    for workflow_id, workflow_state in soar_agent.soar_engine.running_workflows.items():
        workflows.append(WorkflowStatusResponse(
            workflow_id=workflow_id,
            playbook_id=workflow_state.get("playbook_id", ""),
            incident_id=workflow_state.get("incident_id", ""),
            status=workflow_state.get("status", "unknown"),
            started_at=workflow_state.get("started_at", ""),
            completed_at=workflow_state.get("completed_at"),
            actions_completed=workflow_state.get("actions_completed", []),
            actions_failed=workflow_state.get("actions_failed", []),
            actions_pending_approval=workflow_state.get("actions_pending_approval", []),
            current_step=workflow_state.get("current_step", 0),
            error=workflow_state.get("error")
        ))
    
    return workflows

@app.get("/workflows/{workflow_id}", response_model=WorkflowStatusResponse)
async def get_workflow_status(workflow_id: str):
    """Get specific workflow status"""
    if not soar_agent:
        raise HTTPException(status_code=500, detail="SOAR agent not initialized")
    
    workflow_state = soar_agent.soar_engine.running_workflows.get(workflow_id)
    if not workflow_state:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
    
    return WorkflowStatusResponse(
        workflow_id=workflow_id,
        playbook_id=workflow_state.get("playbook_id", ""),
        incident_id=workflow_state.get("incident_id", ""),
        status=workflow_state.get("status", "unknown"),
        started_at=workflow_state.get("started_at", ""),
        completed_at=workflow_state.get("completed_at"),
        actions_completed=workflow_state.get("actions_completed", []),
        actions_failed=workflow_state.get("actions_failed", []),
        actions_pending_approval=workflow_state.get("actions_pending_approval", []),
        current_step=workflow_state.get("current_step", 0),
        error=workflow_state.get("error")
    )

@app.post("/workflows/{workflow_id}/approve")
async def approve_action(
    workflow_id: str,
    approval_request: ActionApprovalRequest
):
    """Approve or reject a pending action"""
    if not soar_agent:
        raise HTTPException(status_code=500, detail="SOAR agent not initialized")
    
    try:
        result = await soar_agent.handle_a2a_task({
            "task_type": "approve_action",
            "workflow_id": workflow_id,
            "action_id": approval_request.action_id,
            "approved": approval_request.approved,
            "approver": approval_request.approver,
            "comments": approval_request.comments
        })
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "message": f"Action {approval_request.action_id} {'approved' if approval_request.approved else 'rejected'}",
            "workflow_id": workflow_id,
            "action_id": approval_request.action_id,
            "approver": approval_request.approver,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error approving action: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/workflows/{workflow_id}/cancel")
async def cancel_workflow(workflow_id: str):
    """Cancel a running workflow"""
    if not soar_agent:
        raise HTTPException(status_code=500, detail="SOAR agent not initialized")
    
    workflow_state = soar_agent.soar_engine.running_workflows.get(workflow_id)
    if not workflow_state:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
    
    if workflow_state.get("status") not in ["running", "approval_required"]:
        raise HTTPException(status_code=400, detail=f"Workflow {workflow_id} cannot be cancelled")
    
    try:
        workflow_state["status"] = PlaybookStatus.CANCELLED.value
        workflow_state["cancelled_at"] = datetime.now().isoformat()
        
        return {
            "message": f"Workflow {workflow_id} cancelled",
            "workflow_id": workflow_id,
            "status": "cancelled",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error cancelling workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get SOAR system metrics"""
    if not soar_agent:
        raise HTTPException(status_code=500, detail="SOAR agent not initialized")
    
    try:
        # Calculate metrics
        total_playbooks = len(soar_agent.soar_engine.playbooks)
        enabled_playbooks = len([p for p in soar_agent.soar_engine.playbooks.values() if p.enabled])
        running_workflows = len(soar_agent.soar_engine.running_workflows)
        
        # Count workflows by status
        status_counts = {}
        for workflow in soar_agent.soar_engine.running_workflows.values():
            status = workflow.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "timestamp": datetime.now().isoformat(),
            "playbooks": {
                "total": total_playbooks,
                "enabled": enabled_playbooks,
                "disabled": total_playbooks - enabled_playbooks
            },
            "workflows": {
                "running": running_workflows,
                "by_status": status_counts
            },
            "mcp": {
                "enabled": soar_agent.config["mcp"]["enabled"],
                "threat_intelligence_sources": len(soar_agent.mcp_client.config.get("env", {}))
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Background task functions
async def process_incident_background(incident_data: Dict[str, Any]):
    """Background task to process incident"""
    try:
        result = await soar_agent.process_incident(incident_data)
        logger.info(f"Incident processing completed: {result}")
    except Exception as e:
        logger.error(f"Error processing incident in background: {e}")

async def execute_playbook_background(playbook: SOARPlaybook, incident: Incident):
    """Background task to execute playbook"""
    try:
        result = await soar_agent.soar_engine.execute_playbook(playbook, incident)
        logger.info(f"Playbook execution completed: {result}")
    except Exception as e:
        logger.error(f"Error executing playbook in background: {e}")

# WebSocket endpoints for real-time updates (optional)
@app.websocket("/ws/workflows")
async def websocket_workflows(websocket):
    """WebSocket endpoint for real-time workflow updates"""
    await websocket.accept()
    try:
        while True:
            # Send current workflow status
            workflows = []
            for workflow_id, workflow_state in soar_agent.soar_engine.running_workflows.items():
                workflows.append({
                    "workflow_id": workflow_id,
                    "status": workflow_state.get("status"),
                    "current_step": workflow_state.get("current_step", 0),
                    "actions_completed": len(workflow_state.get("actions_completed", [])),
                    "actions_pending": len(workflow_state.get("actions_pending_approval", []))
                })
            
            await websocket.send_json({
                "type": "workflow_update",
                "timestamp": datetime.now().isoformat(),
                "workflows": workflows
            })
            
            await asyncio.sleep(5)  # Update every 5 seconds
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(
        "soar_workflow_server:app",
        host="0.0.0.0",
        port=8081,
        reload=True,
        log_level="info"
    )
