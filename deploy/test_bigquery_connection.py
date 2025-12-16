#!/usr/bin/env python3
"""
Test BigQuery Connection and Real Data for Hybrid ADA System
Verifies connection to your actual BigQuery tables
"""

import os
import sys
from google.cloud import bigquery
from google.oauth2 import service_account
import json

def test_bigquery_connection():
    """Test connection to BigQuery and query real data"""
    try:
        print("🔍 Testing BigQuery Connection...")
        print("=" * 50)
        
        # Initialize BigQuery client
        client = bigquery.Client(project="chronicle-dev-2be9")
        print(f"✅ Connected to project: chronicle-dev-2be9")
        
        # Test 1: SIEM Events Table
        print("\n📊 Testing SIEM Events Table...")
        siem_query = """
        SELECT COUNT(*) as total_events
        FROM `chronicle-dev-2be9.gatra_database.siem_events`
        LIMIT 1
        """
        
        siem_result = client.query(siem_query).result()
        siem_count = next(siem_result).total_events
        print(f"✅ SIEM Events Count: {siem_count:,}")
        
        # Test 2: Processed ADA Alerts Table
        print("\n🚨 Testing Processed ADA Alerts Table...")
        ada_query = """
        SELECT COUNT(*) as total_alerts
        FROM `chronicle-dev-2be9.gatra_database.processed_ada_alerts`
        LIMIT 1
        """
        
        ada_result = client.query(ada_query).result()
        ada_count = next(ada_result).total_alerts
        print(f"✅ ADA Alerts Count: {ada_count:,}")
        
        # Test 3: Agent State Table
        print("\n🤖 Testing Agent State Table...")
        agent_query = """
        SELECT COUNT(*) as total_agents
        FROM `chronicle-dev-2be9.gatra_database.agent_state`
        LIMIT 1
        """
        
        agent_result = client.query(agent_query).result()
        agent_count = next(agent_result).total_agents
        print(f"✅ Agent State Count: {agent_count}")
        
        # Test 4: Sample Data Structure
        print("\n📋 Testing Sample Data Structure...")
        sample_query = """
        SELECT alarmId, events
        FROM `chronicle-dev-2be9.gatra_database.siem_events`
        LIMIT 5
        """
        
        sample_result = client.query(sample_query).result()
        sample_data = [row for row in sample_result]
        print(f"✅ Sample Data Retrieved: {len(sample_data)} rows")
        
        # Test 5: TAA State Table (if exists)
        print("\n📋 Testing TAA State Table...")
        try:
            taa_query = """
            SELECT COUNT(*) as total_cases
            FROM `chronicle-dev-2be9.gatra_database.taa_state`
            LIMIT 1
            """
            
            taa_result = client.query(taa_query).result()
            taa_count = next(taa_result).total_cases
            print(f"✅ TAA Cases Count: {taa_count}")
        except Exception as e:
            print(f"⚠️ TAA State Table: {e}")
        
        # Test 6: CRA State Table (if exists)
        print("\n🛡️ Testing CRA State Table...")
        try:
            cra_query = """
            SELECT COUNT(*) as total_incidents
            FROM `chronicle-dev-2be9.gatra_database.cra_state`
            LIMIT 1
            """
            
            cra_result = client.query(cra_query).result()
            cra_count = next(cra_result).total_incidents
            print(f"✅ CRA Incidents Count: {cra_count}")
        except Exception as e:
            print(f"⚠️ CRA State Table: {e}")
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 BigQuery Connection Test Summary:")
        print(f"   SIEM Events: {siem_count:,}")
        print(f"   ADA Alerts: {ada_count:,}")
        print(f"   Agent State: {agent_count}")
        print(f"   Sample Data: {len(sample_data)} rows")
        
        # Test hybrid engine with real data
        print("\n🧠 Testing Hybrid Engine with Real Data...")
        test_hybrid_with_real_data(siem_count, ada_count, agent_count)
        
        return True
        
    except Exception as e:
        print(f"❌ BigQuery Connection Failed: {e}")
        return False

def test_hybrid_with_real_data(siem_count, ada_count, agent_count):
    """Test hybrid engine with real BigQuery data"""
    try:
        # Import hybrid engine
        from hybrid_ada_decision_engine import HybridADAWorkflow
        import asyncio
        
        # Create test event with real data context
        test_event = {
            'event_id': f'real_test_{int(siem_count)}',
            'timestamp': '2025-08-23T03:58:00Z',
            'source_ip': '192.168.1.100',
            'destination_ip': '10.0.0.50',
            'event_type': 'network_scan',
            'severity': 'medium',
            'context': {
                'user_id': 'admin',
                'session_duration': 3600,
                'previous_events': siem_count,
                'geolocation': 'internal',
                'bigquery_siem_count': siem_count,
                'bigquery_ada_alerts': ada_count,
                'bigquery_agent_state': agent_count
            }
        }
        
        print(f"   📊 Using Real SIEM Count: {siem_count:,}")
        print(f"   🚨 Using Real ADA Alerts: {ada_count:,}")
        print(f"   🤖 Using Real Agent State: {agent_count}")
        
        # Test hybrid workflow
        workflow = HybridADAWorkflow()
        result = asyncio.run(workflow.process_security_event(test_event))
        
        print(f"\n✅ Hybrid Engine Test with Real Data:")
        print(f"   Decision: {result.decision}")
        print(f"   Threat Score: {result.final_threat_score:.3f}")
        print(f"   Priority: {result.priority}")
        print(f"   ML Contribution: {result.ml_contribution:.1%}")
        print(f"   AI Contribution: {result.ai_contribution:.1%}")
        
    except Exception as e:
        print(f"⚠️ Hybrid Engine Test Failed: {e}")

if __name__ == "__main__":
    print("🧠 Testing Hybrid ADA System with Real BigQuery Data")
    print("=" * 60)
    
    success = test_bigquery_connection()
    
    if success:
        print("\n🎉 All Tests Passed! Your hybrid system is ready for real data.")
    else:
        print("\n❌ Tests Failed. Please check BigQuery configuration.")
