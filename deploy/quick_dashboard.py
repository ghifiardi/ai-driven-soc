import streamlit as st
import pandas as pd

st.set_page_config(page_title="Quick AI Dashboard", layout="wide")

st.title("🤖 Quick AI Models Overview")
st.write("CRA + ADA + TAA - Simple View")

# CRA Section
st.header("🛡️ CRA - Containment & Response Agent")
col1, col2 = st.columns(2)
with col1:
    st.metric("Version", "v5.2.1", "🔄 Active")
    st.metric("Accuracy", "89.7%", "+2.1%")
with col2:
    st.metric("Automation", "41.6%", "+5.3%")
    st.metric("Samples", "127,842", "+1,234")

# ADA Section  
st.header("🧠 ADA - Automated Decision Analysis")
col1, col2 = st.columns(2)
with col1:
    st.metric("Alerts", "156", "+23 today")
    st.metric("Confidence", "67%", "+2.8%")
with col2:
    st.metric("Anomalies", "23", "+3 today")
    st.metric("Accuracy", "89.3%", "+1.5%")

# TAA Section
st.header("🔍 TAA - Triage Analysis & Assessment")
col1, col2 = st.columns(2)
with col1:
    st.metric("Processed", "1,273", "+89 today")
    st.metric("Processing Rate", "90.7%", "+1.2%")
with col2:
    st.metric("Accuracy", "94.1%", "+1.8%")
    st.metric("Efficiency", "87.3%", "+2.3%")

st.success("✅ Quick Dashboard Ready!")
