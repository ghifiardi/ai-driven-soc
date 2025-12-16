# Hybrid Cyber Defense Agent - Deployment Package

This package contains all necessary files and documentation for deploying the Hybrid Cyber Defense Agent to a production environment.

## Package Contents

### Documentation
- `HYBRID_DEFENSE_AGENT_DEPLOYMENT_GUIDE.md` - Comprehensive deployment guide
- `HYBRID_DEFENSE_AGENT_DEPLOYMENT_GUIDE.docx` - DOCX version of deployment guide
- `HYBRID_DEFENSE_AGENT_TECHNICAL_SPEC.md` - Technical specifications
- `HYBRID_DEFENSE_AGENT_TECHNICAL_SPEC.docx` - DOCX version of technical spec
- `DEPLOYMENT_CHECKLIST.md` - Step-by-step deployment checklist

### Configuration
- `hybrid_defense_config.json` - Agent configuration
- `hybrid_agent_card.json` - A2A agent card

### Services
- `hybrid-defense.service` - Systemd service file for agent
- `hybrid-defense-dashboard.service` - Systemd service file for dashboard

### Scripts
- `deploy_hybrid_dashboard.sh` - Dashboard deployment script
- `test_dashboard_connection.py` - Connection testing script
- `simple_validation.py` - System validation script

## Quick Start

1. Review the deployment guide: `documentation/HYBRID_DEFENSE_AGENT_DEPLOYMENT_GUIDE.md`
2. Follow the deployment checklist: `DEPLOYMENT_CHECKLIST.md`
3. Configure your environment using the provided configuration files
4. Deploy using the provided service files and scripts

## Support

For technical support and questions:
- Review the technical specification document
- Check the troubleshooting section in the deployment guide
- Contact the development team

## Package Information
- Created: 2025-10-11 17:26:24
- Version: 1.0
- Author: AI-Driven SOC Development Team
