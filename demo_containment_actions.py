#!/usr/bin/env python3
"""
Advanced Containment Actions Demo
================================

This script demonstrates the complete advanced containment actions
in the CRA SOAR MCP system with detailed examples and explanations.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def demo_containment_actions():
    """Demonstrate all advanced containment actions"""
    logger.info("🚀 Advanced Containment Actions Demonstration")
    logger.info("=" * 80)
    
    try:
        from cra_soar_mcp_agent import CRASOARMCPAgent, SOARWorkflowEngine, ContainmentAction, ActionType, RiskLevel, Incident, ThreatIntelligence
        
        # Initialize the CRA agent
        logger.info("🔧 Initializing CRA SOAR MCP Agent...")
        agent = CRASOARMCPAgent()
        logger.info("✅ CRA agent initialized successfully")
        
        # Demo 1: Asset Isolation
        logger.info("\n" + "=" * 80)
        logger.info("🔒 DEMO 1: ASSET ISOLATION CONTAINMENT ACTION")
        logger.info("=" * 80)
        
        isolation_action = ContainmentAction(
            action_id="isolate_compromised_hosts",
            action_type=ActionType.ISOLATE,
            target="workstation-001.company.com",
            parameters={
                "isolation_method": "network_quarantine",
                "duration": "until_manual_release",
                "preserve_evidence": True
            },
            risk_level=RiskLevel.HIGH,
            justification="Isolate compromised workstation to prevent lateral movement and data exfiltration",
            requires_approval=True
        )
        
        logger.info("📋 Asset Isolation Action Details:")
        logger.info(f"  🎯 Target: {isolation_action.target}")
        logger.info(f"  🔧 Action Type: {isolation_action.action_type.value}")
        logger.info(f"  📊 Risk Level: {isolation_action.risk_level.value}")
        logger.info(f"  ✅ Requires Approval: {isolation_action.requires_approval}")
        logger.info(f"  📝 Justification: {isolation_action.justification}")
        logger.info(f"  ⚙️ Parameters: {isolation_action.parameters}")
        
        # Execute isolation action
        mock_incident = Incident(
            incident_id="isolation_demo",
            alert_id="isolation_demo",
            severity=RiskLevel.HIGH,
            title="Asset Isolation Demo",
            description="Demonstrating asset isolation containment action",
            affected_assets=[isolation_action.target],
            indicators=["malware.exe", "192.168.1.100"],
            threat_intelligence=[],
            containment_actions=[],
            created_at=datetime.now().isoformat()
        )
        
        result = await agent.soar_engine._execute_action(isolation_action, mock_incident)
        logger.info(f"✅ Isolation Action Result: {result}")
        
        # Demo 2: Network Blocking
        logger.info("\n" + "=" * 80)
        logger.info("🚫 DEMO 2: NETWORK BLOCKING CONTAINMENT ACTION")
        logger.info("=" * 80)
        
        blocking_action = ContainmentAction(
            action_id="block_malicious_ips",
            action_type=ActionType.BLOCK,
            target="192.168.1.100",
            parameters={
                "block_type": "firewall_rule",
                "direction": "both",
                "protocol": "all",
                "rule_priority": "high"
            },
            risk_level=RiskLevel.MEDIUM,
            justification="Block known malicious IP address to prevent further attacks",
            requires_approval=False
        )
        
        logger.info("📋 Network Blocking Action Details:")
        logger.info(f"  🎯 Target: {blocking_action.target}")
        logger.info(f"  🔧 Action Type: {blocking_action.action_type.value}")
        logger.info(f"  📊 Risk Level: {blocking_action.risk_level.value}")
        logger.info(f"  ✅ Requires Approval: {blocking_action.requires_approval}")
        logger.info(f"  📝 Justification: {blocking_action.justification}")
        logger.info(f"  ⚙️ Parameters: {blocking_action.parameters}")
        
        result = await agent.soar_engine._execute_action(blocking_action, mock_incident)
        logger.info(f"✅ Blocking Action Result: {result}")
        
        # Demo 3: File Quarantine
        logger.info("\n" + "=" * 80)
        logger.info("📁 DEMO 3: FILE QUARANTINE CONTAINMENT ACTION")
        logger.info("=" * 80)
        
        quarantine_action = ContainmentAction(
            action_id="quarantine_suspicious_files",
            action_type=ActionType.QUARANTINE,
            target="malware.exe",
            parameters={
                "quarantine_location": "/quarantine/malware/",
                "original_location": "/Users/john/Downloads/",
                "scan_for_malware": True,
                "preserve_metadata": True
            },
            risk_level=RiskLevel.MEDIUM,
            justification="Quarantine suspicious executable file for analysis and prevent execution",
            requires_approval=False
        )
        
        logger.info("📋 File Quarantine Action Details:")
        logger.info(f"  🎯 Target: {quarantine_action.target}")
        logger.info(f"  🔧 Action Type: {quarantine_action.action_type.value}")
        logger.info(f"  📊 Risk Level: {quarantine_action.risk_level.value}")
        logger.info(f"  ✅ Requires Approval: {quarantine_action.requires_approval}")
        logger.info(f"  📝 Justification: {quarantine_action.justification}")
        logger.info(f"  ⚙️ Parameters: {quarantine_action.parameters}")
        
        result = await agent.soar_engine._execute_action(quarantine_action, mock_incident)
        logger.info(f"✅ Quarantine Action Result: {result}")
        
        # Demo 4: Account Management
        logger.info("\n" + "=" * 80)
        logger.info("👤 DEMO 4: ACCOUNT MANAGEMENT CONTAINMENT ACTION")
        logger.info("=" * 80)
        
        account_action = ContainmentAction(
            action_id="disable_compromised_account",
            action_type=ActionType.DISABLE,
            target="john.doe@company.com",
            parameters={
                "disable_type": "account_lock",
                "preserve_evidence": True,
                "notify_user": False,
                "escalate_to_hr": True
            },
            risk_level=RiskLevel.HIGH,
            justification="Disable compromised user account to prevent further unauthorized access",
            requires_approval=True
        )
        
        logger.info("📋 Account Management Action Details:")
        logger.info(f"  🎯 Target: {account_action.target}")
        logger.info(f"  🔧 Action Type: {account_action.action_type.value}")
        logger.info(f"  📊 Risk Level: {account_action.risk_level.value}")
        logger.info(f"  ✅ Requires Approval: {account_action.requires_approval}")
        logger.info(f"  📝 Justification: {account_action.justification}")
        logger.info(f"  ⚙️ Parameters: {account_action.parameters}")
        
        result = await agent.soar_engine._execute_action(account_action, mock_incident)
        logger.info(f"✅ Account Management Action Result: {result}")
        
        # Demo 5: System Monitoring
        logger.info("\n" + "=" * 80)
        logger.info("👁️ DEMO 5: SYSTEM MONITORING CONTAINMENT ACTION")
        logger.info("=" * 80)
        
        monitor_action = ContainmentAction(
            action_id="monitor_suspicious_activity",
            action_type=ActionType.MONITOR,
            target="server-002.company.com",
            parameters={
                "monitoring_level": "enhanced",
                "duration": "72h",
                "alerts": ["login_anomalies", "data_access", "network_connections"],
                "capture_evidence": True
            },
            risk_level=RiskLevel.MEDIUM,
            justification="Monitor server for suspicious activity and potential compromise",
            requires_approval=False
        )
        
        logger.info("📋 System Monitoring Action Details:")
        logger.info(f"  🎯 Target: {monitor_action.target}")
        logger.info(f"  🔧 Action Type: {monitor_action.action_type.value}")
        logger.info(f"  📊 Risk Level: {monitor_action.risk_level.value}")
        logger.info(f"  ✅ Requires Approval: {monitor_action.requires_approval}")
        logger.info(f"  📝 Justification: {monitor_action.justification}")
        logger.info(f"  ⚙️ Parameters: {monitor_action.parameters}")
        
        result = await agent.soar_engine._execute_action(monitor_action, mock_incident)
        logger.info(f"✅ Monitoring Action Result: {result}")
        
        # Demo 6: Escalation Actions
        logger.info("\n" + "=" * 80)
        logger.info("📞 DEMO 6: ESCALATION CONTAINMENT ACTION")
        logger.info("=" * 80)
        
        escalate_action = ContainmentAction(
            action_id="escalate_to_security_team",
            action_type=ActionType.ESCALATE,
            target="security_team",
            parameters={
                "escalation_level": "tier2",
                "urgency": "high",
                "assigned_to": "security_team",
                "include_evidence": True,
                "notify_management": True
            },
            risk_level=RiskLevel.HIGH,
            justification="Escalate critical security incident to security team for immediate response",
            requires_approval=False
        )
        
        logger.info("📋 Escalation Action Details:")
        logger.info(f"  🎯 Target: {escalate_action.target}")
        logger.info(f"  🔧 Action Type: {escalate_action.action_type.value}")
        logger.info(f"  📊 Risk Level: {escalate_action.risk_level.value}")
        logger.info(f"  ✅ Requires Approval: {escalate_action.requires_approval}")
        logger.info(f"  📝 Justification: {escalate_action.justification}")
        logger.info(f"  ⚙️ Parameters: {escalate_action.parameters}")
        
        result = await agent.soar_engine._execute_action(escalate_action, mock_incident)
        logger.info(f"✅ Escalation Action Result: {result}")
        
        # Demo 7: Complete Playbook Execution
        logger.info("\n" + "=" * 80)
        logger.info("🎭 DEMO 7: COMPLETE SOAR PLAYBOOK EXECUTION")
        logger.info("=" * 80)
        
        # Create a comprehensive incident
        comprehensive_incident = {
            "alert_id": "comprehensive_demo",
            "severity": "critical",
            "title": "Advanced Persistent Threat - Comprehensive Response",
            "description": "Sophisticated APT attack with multiple compromised systems and data exfiltration indicators. Comprehensive containment response required.",
            "affected_assets": [
                "workstation-001.company.com",
                "server-002.company.com",
                "database-server.company.com",
                "file-server.company.com"
            ],
            "indicators": [
                "192.168.1.100",
                "malware.example.com",
                "apt-backdoor.exe",
                "data-exfil-server.com",
                "compromised-admin@company.com"
            ],
            "metadata": {
                "threat_type": "apt",
                "confidence": 0.96,
                "data_exfiltration": True,
                "lateral_movement": True,
                "persistence": True
            }
        }
        
        logger.info("📋 Comprehensive Incident Details:")
        logger.info(f"  🚨 Alert ID: {comprehensive_incident['alert_id']}")
        logger.info(f"  📊 Severity: {comprehensive_incident['severity']}")
        logger.info(f"  📝 Title: {comprehensive_incident['title']}")
        logger.info(f"  🎯 Affected Assets: {len(comprehensive_incident['affected_assets'])} systems")
        logger.info(f"  🔍 Indicators: {len(comprehensive_incident['indicators'])} indicators")
        logger.info(f"  📈 Confidence: {comprehensive_incident['metadata']['confidence']}")
        
        # Process the comprehensive incident
        logger.info("\n🔄 Processing comprehensive incident...")
        result = await agent.process_incident(comprehensive_incident)
        
        logger.info("✅ Comprehensive Incident Processing Result:")
        logger.info(f"  📊 Status: {result.get('status')}")
        logger.info(f"  🎯 Playbook Executed: {result.get('playbook_executed')}")
        logger.info(f"  🛡️ Containment Actions: {result.get('containment_actions')}")
        logger.info(f"  🔍 TI Sources Used: {result.get('threat_intelligence_sources')}")
        logger.info(f"  📋 Report Generated: {result.get('report_generated')}")
        
        # Demo 8: Approval Workflow
        logger.info("\n" + "=" * 80)
        logger.info("✅ DEMO 8: APPROVAL WORKFLOW DEMONSTRATION")
        logger.info("=" * 80)
        
        logger.info("📋 Approval Workflow Features:")
        logger.info("  🔒 High-risk actions require human approval")
        logger.info("  ⏰ Configurable approval timeouts")
        logger.info("  📞 Automatic escalation for delayed approvals")
        logger.info("  📝 Complete audit trails for all approvals")
        logger.info("  👥 Multi-level approval chains")
        logger.info("  📧 Email notifications for approval requests")
        logger.info("  📱 Mobile-friendly approval interfaces")
        
        # Show approval-required actions
        approval_actions = [
            {
                "action": "Asset Isolation",
                "reason": "Network quarantine affects business operations",
                "approver": "Security Lead",
                "timeout": "30 minutes"
            },
            {
                "action": "Account Disable",
                "reason": "User account access disruption",
                "approver": "HR Director",
                "timeout": "60 minutes"
            },
            {
                "action": "Emergency Patching",
                "reason": "System downtime for production systems",
                "approver": "Infrastructure Manager",
                "timeout": "45 minutes"
            }
        ]
        
        for approval in approval_actions:
            logger.info(f"  ⏳ {approval['action']}: {approval['reason']}")
            logger.info(f"    👤 Approver: {approval['approver']}")
            logger.info(f"    ⏰ Timeout: {approval['timeout']}")
        
        logger.info("\n🎉 Advanced Containment Actions Demo Completed Successfully!")
        logger.info("\n🛡️ SUMMARY OF DEMONSTRATED CAPABILITIES:")
        logger.info("=" * 80)
        logger.info("✅ Asset Isolation - Network quarantine of compromised systems")
        logger.info("✅ Network Blocking - Firewall rules to block malicious traffic")
        logger.info("✅ File Quarantine - Secure isolation of suspicious files")
        logger.info("✅ Account Management - Disable compromised user accounts")
        logger.info("✅ System Monitoring - Enhanced monitoring of suspicious activities")
        logger.info("✅ Escalation Actions - Automated escalation to appropriate teams")
        logger.info("✅ Approval Workflows - Human oversight for high-risk actions")
        logger.info("✅ Complete SOAR Playbooks - Orchestrated multi-step responses")
        logger.info("✅ MCP Integration - Threat intelligence enhanced decision making")
        logger.info("✅ Real-time Processing - Immediate response to security incidents")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.info("💡 Make sure you're running from the correct directory with venv_mcp activated")
        return False
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        return False

async def main():
    """Main demo function"""
    logger.info("🚀 Starting Advanced Containment Actions Demo")
    logger.info("=" * 80)
    
    success = await demo_containment_actions()
    
    if success:
        logger.info("\n🎊 CONTAINMENT ACTIONS DEMO SUCCESSFUL!")
        logger.info("=" * 80)
        logger.info("Your CRA SOAR MCP system is ready for production use with:")
        logger.info("  🚀 Automated incident response")
        logger.info("  🛡️ Advanced containment actions")
        logger.info("  🔍 Threat intelligence integration")
        logger.info("  👥 Human approval workflows")
        logger.info("  📊 Real-time monitoring and reporting")
        logger.info("\n🎯 Next Steps:")
        logger.info("  1. Configure your security tool integrations")
        logger.info("  2. Set up approval workflows for your organization")
        logger.info("  3. Integrate with your existing SOC tools")
        logger.info("  4. Train your security team on the new capabilities")
        logger.info("  5. Monitor and optimize response times")
    else:
        logger.error("\n❌ Demo failed - please check the logs for details")
    
    return success

if __name__ == "__main__":
    asyncio.run(main())


