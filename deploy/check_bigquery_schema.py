#!/usr/bin/env python3
"""
Check BigQuery Schema
Find out what columns are actually available in the tables
"""

import streamlit as st
from google.cloud import bigquery

st.title("🔍 BigQuery Schema Checker")
st.write("Let's see what columns are actually available in your tables")

try:
    client = bigquery.Client(project="chronicle-dev-2be9")
    
    # Check ADA table
    st.header("📊 ADA Table Schema")
    ada_query = "SELECT * FROM `chronicle-dev-2be9.gatra_database.processed_ada_alerts` LIMIT 1"
    ada_result = client.query(ada_query).result()
    
    if ada_result:
        ada_row = next(ada_result)
        st.write("**Available columns:**")
        for key in ada_row.keys():
            st.write(f"• {key}: {type(ada_row[key]).__name__}")
    
    # Check TAA table
    st.header("📊 TAA Table Schema")
    taa_query = "SELECT * FROM `chronicle-dev-2be9.gatra_database.taa_state` LIMIT 1"
    taa_result = client.query(taa_query).result()
    
    if taa_result:
        taa_row = next(taa_result)
        st.write("**Available columns:**")
        for key in taa_row.keys():
            st.write(f"• {key}: {type(taa_row[key]).__name__}")
    
    # Check CRA table
    st.header("📊 CRA Table Schema")
    cra_query = "SELECT * FROM `chronicle-dev-2be9.gatra_database.cra_state` LIMIT 1"
    cra_result = client.query(cra_query).result()
    
    if cra_result:
        cra_row = next(cra_result)
        st.write("**Available columns:**")
        for key in cra_row.keys():
            st.write(f"• {key}: {type(cra_row[key]).__name__}")
    
    # Check SIEM table
    st.header("📊 SIEM Table Schema")
    siem_query = "SELECT * FROM `chronicle-dev-2be9.gatra_database.siem_events` LIMIT 1"
    siem_result = client.query(siem_query).result()
    
    if siem_result:
        siem_row = next(siem_result)
        st.write("**Available columns:**")
        for key in siem_row.keys():
            st.write(f"• {key}: {type(siem_row[key]).__name__}")
            
except Exception as e:
    st.error(f"Error: {e}")
    st.write("This will help us understand what columns are actually available")
