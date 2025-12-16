def alert_review_page():
    """Clean Alert Review page with better structure"""
    st.header("🚨 Alert Review & Feedback")
    st.markdown("**Historical alert review with feedback integration**")
    
    if st.button("🔄 Refresh Alerts"):
        st.rerun()
    
    # Get alerts data
    with st.spinner("Loading alerts..."):
        alerts_df = get_real_alerts()
    
    if not alerts_df.empty:
        # Show summary metrics first
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_alerts = len(alerts_df)
            st.metric("Total Alerts", total_alerts)
        
        with col2:
            anomalies = len(alerts_df[alerts_df['is_anomaly'] == True])
            st.metric("Anomalies", anomalies)
        
        with col3:
            high_confidence = len(alerts_df[alerts_df['confidence_score'] > 0.8])
            st.metric("High Confidence", high_confidence)
        
        with col4:
            avg_confidence = alerts_df['confidence_score'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")
        
        st.markdown("---")
        
        # Clean alerts table with only essential columns
        st.subheader("📊 Alert Summary")
        
        # Prepare clean display data
        display_df = alerts_df.copy()
        display_df['severity'] = display_df['confidence_score'].apply(
            lambda x: 'High' if x > 0.8 else 'Medium' if x > 0.6 else 'Low'
        )
        display_df['timestamp_short'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%m-%d %H:%M')
        
        # Select only essential columns for display
        clean_columns = ['alert_id', 'timestamp_short', 'classification', 'severity', 'confidence_score', 'is_anomaly']
        clean_df = display_df[clean_columns].rename(columns={
            'timestamp_short': 'Time',
            'classification': 'Type',
            'confidence_score': 'Confidence'
        })
        
        st.dataframe(
            clean_df,
            use_container_width=True,
            height=300,
            hide_index=True
        )
        
        st.markdown("---")
        
        # Alert selection for feedback - simplified
        st.subheader("📝 Provide Feedback")
        
        # Create a dropdown instead of radio buttons
        alert_options = {}
        for _, row in display_df.iterrows():
            option_key = f"{row['alert_id'][:12]}... | {row['timestamp_short']} | {row['severity']} | {row['classification']}"
            alert_options[option_key] = row
        
        selected_option = st.selectbox(
            "Select Alert to Review:",
            options=list(alert_options.keys()),
            key="alert_dropdown"
        )
        
        if selected_option:
            selected_alert = alert_options[selected_option]
            
            # Display selected alert details in a clean format
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("**📋 Alert Details**")
                st.markdown(f"**Alert ID:** `{selected_alert['alert_id']}`")
                st.markdown(f"**Timestamp:** {selected_alert['timestamp']}")
                st.markdown(f"**Classification:** {selected_alert['classification']}")
                st.markdown(f"**Confidence Score:** {selected_alert['confidence_score']:.3f}")
                st.markdown(f"**Is Anomaly:** {'Yes' if selected_alert['is_anomaly'] else 'No'}")
            
            with col2:
                st.markdown("**⚡ Quick Actions**")
                if st.button("✅ True Positive", key="tp_btn", use_container_width=True):
                    st.success("✅ Marked as True Positive")
                if st.button("❌ False Positive", key="fp_btn", use_container_width=True):
                    st.warning("❌ Marked as False Positive")
                if st.button("📋 Manual Review", key="mr_btn", use_container_width=True):
                    st.info("📋 Manual review requested")
        
        # Show additional insights
        if len(alerts_df) > 0:
            st.markdown("---")
            st.subheader("📈 Alert Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Classification distribution
                classification_counts = alerts_df['classification'].value_counts()
                st.markdown("**Alert Types:**")
                for classification, count in classification_counts.items():
                    percentage = (count / len(alerts_df)) * 100
                    st.markdown(f"• {classification}: {count} ({percentage:.1f}%)")
            
            with col2:
                # Time distribution
                alerts_df['hour'] = pd.to_datetime(alerts_df['timestamp']).dt.hour
                hour_counts = alerts_df['hour'].value_counts().sort_index()
                st.markdown("**Peak Hours:**")
                for hour, count in hour_counts.head(3).items():
                    st.markdown(f"• {hour:02d}:00 - {count} alerts")
    
    else:
        st.info("No alerts available for review")
