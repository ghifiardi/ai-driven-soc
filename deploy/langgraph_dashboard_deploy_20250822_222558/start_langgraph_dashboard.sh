#!/bin/bash
cd /home/app/langgraph-ada-dashboard
export DASHBOARD_PORT=8507
/home/app/.local/bin/streamlit run langgraph_ada_dashboard.py --server.port $DASHBOARD_PORT --server.address 0.0.0.0 --server.headless true
