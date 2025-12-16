"""
DEMO SCRIPT ENHANCEMENTS FOR OVERVIEW & FUNNEL TAB
Based on GATRA SOC Dashboard Demo Script

This creates an enhanced Overview & Funnel tab that combines:
- Speed metrics (< 1 minute detection)
- Accuracy metrics (89.2%, targeting 94%)
- Automation & Cost Efficiency
- Real-time Funnel visualization
"""

def overview_and_funnel_page_enhanced():
    """
    Enhanced Overview & Funnel Page
    Combines Overview and Security Operations Funnel based on demo script
    """
    
    # Executive Header with GATRA Branding
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1a365d 0%, #2d3748 100%);
                padding: 2rem; border-radius: 12px; margin-bottom: 2rem;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);">
        <h1 style="color: white; font-family: 'Inter', sans-serif; font-weight: 700;
                   font-size: 2.5rem; margin: 0; text-align: center;
                   text-shadow: 0 2px 4px rgba(0,0,0,0.3);">
            🛡️ GATRA SOC Dashboard
        </h1>
        <p style="color: rgba(255,255,255,0.9); font-family: 'Inter', sans-serif;
                  font-size: 1.2rem; margin: 0.5rem 0 0 0; text-align: center; font-weight: 300;">
            AI-Driven Security Operations Center • Guardian Agent for Threat Response and Autonomy
        </p>
        <p style="color: rgba(255,255,255,0.8); font-size: 0.95rem; margin: 0.5rem 0 0 0; text-align: center;">
            Merombak landscape security di Indonesia
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # KEY METRICS SECTION - Based on Demo Script
    st.markdown("### 🎯 **Key Performance Indicators**")
    st.markdown("*Real-time security metrics - Instantly actionable intelligence*")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "⚡ Speed (Kecepatan)",
            "< 1 minute",
            delta="Real-time",
            help="Detection to containment in under 1 minute! Game-changer: super fast detection and containment."
        )
        st.caption("**End-to-end detection time**")
        st.caption("Breach yang biasanya terdeteksi setelah 5-190 hari adalah 1 menit • Bayangkan, di layar ini, terlihat metrics yang update layar ini kita – essentially instant!")
    
    with col2:
        st.metric(
            "🎯 Accuracy (Akurasi)",
            "89.2%",
            delta="+22.5% improvement",
            help="AI extremely precise - kami berhasil menurungi false positives ±83%"
        )
        st.caption("**Current CLA Performance**")
        st.caption("Artinya 89% lebih sedikit alarm palsu • >94% threat classification accuracy (vs hanya 17% industry average)")
    
    with col3:
        st.metric(
            "🤖 Automation",
            "74.3%",
            delta="Alerts automated",
            help="Truly human-like intelligence at machine speed"
        )
        st.caption("**AI-Powered Analysis**")
        st.caption("SOC team tidak akan dibombardi false alert; sekarang, they focus on real threats only")
    
    with col4:
        st.metric(
            "💰 Cost Efficiency",
            "90% reduction",
            delta="IDR 90M/month savings",
            help="From IDR 100M to IDR 10M monthly operational cost"
        )
        st.caption("**Operational Cost Savings**")
        st.caption("Relevant alerts only – no noise, no fatigue. Level of accuracy mencapai ~94% classification accuracy - something that actual tools can't achieve.")
    
    # EXECUTIVE SUMMARY
    st.markdown("---")
    st.markdown("### 📊 **Executive Summary**")
    
    summary_col1, summary_col2 = st.columns([2, 1])
    
    with summary_col1:
        st.markdown("""
        **What you see on the overview tab is clean, relevant alerts:**
        
        Di layer ini terlihat metrics yang update **real-time**. Threat detection kita – it's **essentially instant**. 
        Ini game-changer: **super fast detection** and containment. Tidak lagi tunggu berjam-jam atau berhari-hari.
        
        **All of this is enabled by our advanced AI core** – truly human-like intelligence at machine speed.
        """)
    
    with summary_col2:
        st.info("**💡 Key Insight:** Now, let me show you how we do it.")
    
    # SECURITY OPERATIONS FUNNEL
    st.markdown("---")
    st.markdown("### 📈 **Security Operations Funnel (Last 24 Hours)**")
    st.markdown("*This is where the magic happens behind the scenes*")
    
    st.markdown("""
    **Di Funnel ini, kita lihat alur deteksi incident from raw events down to contained threats.**
    
    Sekarang kita beralih ke tab Funnel. This is where the magic happens behind the scenes. 
    Di Funnel ini, kita lihat alur deteksi incident – from thousands of raw data down to contained threats.
    """)
    
    # Funnel Metrics
    funnel_col1, funnel_col2, funnel_col3, funnel_col4 = st.columns(4)
    
    with funnel_col1:
        st.metric("📊 Raw Events", "13,203", help="Total security events received in 24h")
        st.caption("Data mentah dari berbagai sumber")
    
    with funnel_col2:
        st.metric("🔍 Alerts Analyzed", "12,428", help="Events processed by AI")
        st.caption("94.1% processed by AI")
    
    with funnel_col3:
        st.metric("⚠️ Threats Triaged", "10,670", help="Confirmed threats requiring attention")
        st.caption("85.8% threat confirmation rate")
    
    with funnel_col4:
        st.metric("✅ Incidents Contained", "3,396", help="Threats successfully contained")
        st.caption("25.7% required containment")
    
    # Visual Funnel Chart
    funnel_data = {
        'Stage': ['Raw Events', 'Alerts Analyzed', 'Threats Triaged', 'Incidents Contained'],
        'Count': [13203, 12428, 10670, 3396],
        'Color': ['#ff6b6b', '#ffa726', '#42a5f5', '#66bb6a']
    }
    
    fig = go.Figure(go.Funnel(
        y=funnel_data['Stage'],
        x=funnel_data['Count'],
        marker=dict(color=funnel_data['Color']),
        textinfo="value+percent initial",
        textfont=dict(size=16, color='white'),
        connector=dict(line=dict(color='rgb(63, 63, 63)', width=2))
    ))
    
    fig.update_layout(
        title={
            'text': "Security Operations Funnel - Real-time Processing",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#1a365d'}
        },
        height=500,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # HOW IT WORKS - DEMO SCRIPT EXPLANATION
    st.markdown("---")
    st.markdown("### 🔬 **How It Works**")
    
    with st.expander("📖 **Click to understand the intelligence behind GATRA**", expanded=True):
        st.markdown("""
        **Stage 1: Raw Events (13,203 events)**
        - Kumpulan semua log security dari berbagai source: firewall, endpoint, network, cloud
        - Tanpa AI, tim SOC harus review manual 13 ribu event per hari – impossible!
        
        **Stage 2: Alerts Analyzed (12,428 alerts) - 94% Processing**
        - **ADA (Anomaly Detection Agent)** mendeteksi anomali menggunakan AI models
        - **TAA (Triage & Analysis Agent)** mengklasifikasi threat severity
        - Filter noise, focus on signal – AI eliminates 94% of false positives
        
        **Stage 3: Threats Triaged (10,670 threats) - 86% Confirmation**
        - Confirmed threats yang perlu attention
        - **CLA (Continuous Learning Agent)** terus belajar dari feedback
        - Akurasi meningkat setiap hari – currently 89.2%, target 94%
        
        **Stage 4: Incidents Contained (3,396 incidents) - 26% Auto-Response**
        - **CRA (Containment & Response Agent)** automatically responds
        - Automated containment actions: block IP, isolate host, kill process
        - Human analyst hanya perlu handle complex cases – 4x efficiency gain!
        """)
    
    # KEY TAKEAWAYS FOR MANAGEMENT
    st.markdown("---")
    st.markdown("### 🎯 **Key Takeaways for Management**")
    
    takeaway_col1, takeaway_col2, takeaway_col3 = st.columns(3)
    
    with takeaway_col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 1.5rem; border-radius: 10px; text-align: center;">
            <h2 style="color: white; margin: 0;">74.3%</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">Alert Reduction</p>
            <p style="color: rgba(255,255,255,0.8); font-size: 0.85rem; margin: 0.3rem 0 0 0;">
                Eliminasi noise – focus on real threats
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with takeaway_col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    padding: 1.5rem; border-radius: 10px; text-align: center;">
            <h2 style="color: white; margin: 0;">4x</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">Efficiency Gain</p>
            <p style="color: rgba(255,255,255,0.8); font-size: 0.85rem; margin: 0.3rem 0 0 0;">
                Analysts can focus on strategic work
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with takeaway_col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                    padding: 1.5rem; border-radius: 10px; text-align: center;">
            <h2 style="color: white; margin: 0;"><1 min</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">Detection Time</p>
            <p style="color: rgba(255,255,255,0.8); font-size: 0.85rem; margin: 0.3rem 0 0 0;">
                Instant threat detection & response
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # FINANCIAL IMPACT
    st.markdown("---")
    st.markdown("### 💰 **Financial Impact & ROI Analysis**")
    
    cost_col1, cost_col2 = st.columns([3, 2])
    
    with cost_col1:
        st.markdown("""
        **Cost Comparison: Traditional SOC vs GATRA AI SOC**
        
        | Metric | Traditional SOC | GATRA AI SOC | Savings |
        |--------|----------------|--------------|---------|
        | **Monthly Cost** | IDR 100M | IDR 10M | **IDR 90M** |
        | **Analysts Required** | 20 analysts | 2 analysts + AI | **18 analysts** |
        | **Alert Processing** | Manual (slow) | Automated (instant) | **4x faster** |
        | **False Positive Rate** | 83% | 10.8% | **72% reduction** |
        | **Annual Savings** | - | - | **IDR 1,080M** |
        
        **ROI: 900%** return on investment
        """)
    
    with cost_col2:
        # Cost breakdown pie chart
        cost_comparison = pd.DataFrame({
            'Category': ['Traditional SOC', 'AI-Enhanced SOC'],
            'Monthly Cost (IDR M)': [100, 10]
        })
        
        fig_cost = px.bar(
            cost_comparison,
            x='Category',
            y='Monthly Cost (IDR M)',
            color='Category',
            color_discrete_map={'Traditional SOC': '#ff6b6b', 'AI-Enhanced SOC': '#66bb6a'},
            title="Monthly Operational Cost Comparison"
        )
        
        fig_cost.update_layout(
            showlegend=False,
            height=300
        )
        
        st.plotly_chart(fig_cost, use_container_width=True)
    
    # CALL TO ACTION
    st.markdown("---")
    st.success("""
    **✨ The GATRA Advantage:**
    
    Using the GATRA pitch deck, kami membuktikan bahwa AI-driven SOC bukan hanya lebih cepat dan akurat,
    tapi juga **significantly more cost-effective**. Speed, Accuracy, Automation, and Cost Efficiency –
    all delivered through truly human-like intelligence at machine speed.
    
    **Next Step:** Review detailed threat analysis in the Alert Review tab →
    """)

