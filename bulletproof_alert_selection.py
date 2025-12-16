
def submit_feedback_section_bulletproof(alerts_df, agg_toggle, aggregated):
    """BULLETPROOF Submit Feedback Section - No more misalignment!"""
    
    if alerts_df.empty:
        st.warning("No alerts available for review")
        return
    
    st.header("Submit Feedback")
    
    # BULLETPROOF STEP 1: Always use clean, deduplicated data
    clean_alerts = alerts_df.drop_duplicates(subset='alert_id', keep='first').copy()
    clean_alerts = clean_alerts.reset_index(drop=True)
    
    # BULLETPROOF STEP 2: Create stable selection options
    selection_data = []
    for idx, row in clean_alerts.iterrows():
        alert_id = str(row['alert_id'])
        timestamp = str(row.get('timestamp', 'Unknown'))[:19]  # YYYY-MM-DD HH:MM:SS
        severity = str(row.get('severity', 'Unknown'))
        classification = str(row.get('classification', 'Unknown'))
        
        # Create stable, readable option
        option = {
            'alert_id': alert_id,
            'display': f"{alert_id[:12]}... | {timestamp} | {severity} | {classification}",
            'index': idx
        }
        selection_data.append(option)
    
    # BULLETPROOF STEP 3: Initialize selection state
    if 'bulletproof_selected_alert' not in st.session_state:
        st.session_state.bulletproof_selected_alert = selection_data[0]['alert_id']
    
    # BULLETPROOF STEP 4: Validate current selection
    current_alert_id = st.session_state.bulletproof_selected_alert
    valid_ids = [item['alert_id'] for item in selection_data]
    
    if current_alert_id not in valid_ids:
        current_alert_id = valid_ids[0]
        st.session_state.bulletproof_selected_alert = current_alert_id
    
    # BULLETPROOF STEP 5: Create selection interface
    st.subheader("🎯 Select Alert to Review:")
    
    # Use radio buttons for better control and stability
    display_options = [item['display'] for item in selection_data]
    current_index = valid_ids.index(current_alert_id)
    
    selected_index = st.radio(
        "Choose an alert to review:",
        options=display_options,
        index=current_index,
        key="bulletproof_alert_selector"
    )
    
    # BULLETPROOF STEP 6: Get exact alert data
    selected_item = selection_data[selected_index]
    selected_alert_id = selected_item['alert_id']
    
    # Update session state
    st.session_state.bulletproof_selected_alert = selected_alert_id
    
    # BULLETPROOF STEP 7: Fetch exact alert data
    selected_alert_data = clean_alerts[clean_alerts['alert_id'].astype(str) == selected_alert_id]
    
    if selected_alert_data.empty:
        st.error(f"Alert {selected_alert_id} not found in data!")
        return
    
    selected_alert_row = selected_alert_data.iloc[0]
    
    # BULLETPROOF STEP 8: Display verification
    st.success(f"✅ Reviewing Alert: `{selected_alert_id}`")
    st.info(f"📊 Verification: Selected=`{selected_alert_id}` | Data=`{selected_alert_row['alert_id']}` | Match={selected_alert_id == str(selected_alert_row['alert_id'])}")
    
    # BULLETPROOF STEP 9: Display alert details
    st.subheader("📋 Alert Details for Review")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Alert ID", selected_alert_id)
        st.metric("Timestamp", str(selected_alert_row.get('timestamp', 'Unknown'))[:19])
        st.metric("Severity", str(selected_alert_row.get('severity', 'Unknown')))
    
    with col2:
        st.metric("Classification", str(selected_alert_row.get('classification', 'Unknown')))
        st.metric("Confidence Score", f"{selected_alert_row.get('confidence_score', 0):.2f}")
        st.metric("Is Anomaly", str(selected_alert_row.get('is_anomaly', 'Unknown')))
    
    # BULLETPROOF STEP 10: Feedback form
    st.subheader("📝 Provide Feedback")
    
    with st.form("feedback_form", clear_on_submit=True):
        is_true_positive = st.selectbox(
            "Is this a true positive threat?",
            options=["Yes", "No"],
            key="feedback_true_positive"
        )
        
        confidence = st.slider(
            "Analyst Confidence (1-10)",
            min_value=1,
            max_value=10,
            value=7,
            key="feedback_confidence"
        )
        
        comments = st.text_area(
            "Additional Comments",
            placeholder="Enter any additional notes about this alert...",
            key="feedback_comments"
        )
        
        submitted = st.form_submit_button("Submit Feedback", type="primary")
        
        if submitted:
            # BULLETPROOF STEP 11: Submit feedback with exact alert_id
            success = submit_real_feedback(
                alert_id=selected_alert_id,  # Use exact alert_id from selection
                is_true_positive=(is_true_positive == "Yes"),
                confidence=confidence,
                comments=comments
            )
            
            if success:
                st.success(f"✅ Feedback submitted successfully for alert {selected_alert_id}")
                st.info("Feedback is stored in BigQuery table `soc_data.feedback`")
                
                # Remove this alert from the list after feedback
                if 'reviewed_alerts' not in st.session_state:
                    st.session_state.reviewed_alerts = set()
                st.session_state.reviewed_alerts.add(selected_alert_id)
                
                # Clear cache to refresh data
                st.cache_data.clear()
                st.rerun()
            else:
                st.error("❌ Failed to submit feedback. Please try again.")
