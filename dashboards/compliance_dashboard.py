#!/usr/bin/env python3
"""
📜 Compliance & Audit Dashboard
Comprehensive compliance tracking, audit trails, and regulatory reporting
Supports NIST, ISO 27001, SOC 2, GDPR, HIPAA, and custom frameworks
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import random

# Configure page
st.set_page_config(
    page_title="Compliance & Audit Dashboard",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Compliance CSS Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --compliance-navy: #0F172A;
        --compliance-blue: #2563EB;
        --compliance-green: #059669;
        --compliance-amber: #D97706;
        --compliance-red: #DC2626;
        --compliance-purple: #7C3AED;
    }

    .main {
        background: linear-gradient(135deg, #F1F5F9 0%, #E2E8F0 100%);
    }

    .compliance-header {
        font-size: 3.5rem;
        font-weight: 800;
        color: var(--compliance-navy);
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .compliance-subtitle {
        text-align: center;
        color: #64748B;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }

    .framework-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        margin-bottom: 1rem;
        border-top: 4px solid var(--compliance-blue);
        transition: all 0.3s ease;
    }

    .framework-card:hover {
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.12);
        transform: translateY(-2px);
    }

    .framework-name {
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--compliance-navy);
        margin-bottom: 0.5rem;
    }

    .score-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
    }

    .score-excellent {
        background: #D1FAE5;
        color: var(--compliance-green);
    }

    .score-good {
        background: #DBEAFE;
        color: var(--compliance-blue);
    }

    .score-warning {
        background: #FEF3C7;
        color: var(--compliance-amber);
    }

    .score-critical {
        background: #FEE2E2;
        color: var(--compliance-red);
    }

    .control-item {
        background: #F8FAFC;
        border-left: 3px solid var(--compliance-blue);
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }

    .control-passed {
        border-left-color: var(--compliance-green);
        background: #F0FDF4;
    }

    .control-failed {
        border-left-color: var(--compliance-red);
        background: #FEF2F2;
    }

    .audit-entry {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.8rem 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #CBD5E1;
    }

    .panel-white {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        margin: 1.5rem 0;
    }

    .section-title {
        font-size: 2rem;
        font-weight: 700;
        color: var(--compliance-navy);
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }

    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.3rem 0.8rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }

    .status-compliant {
        background: #D1FAE5;
        color: var(--compliance-green);
    }

    .status-partial {
        background: #FEF3C7;
        color: var(--compliance-amber);
    }

    .status-noncompliant {
        background: #FEE2E2;
        color: var(--compliance-red);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📜 Compliance Settings")
    st.markdown("---")

    selected_frameworks = st.multiselect(
        "Select Frameworks",
        options=['NIST CSF', 'ISO 27001', 'SOC 2', 'GDPR', 'HIPAA', 'PCI DSS', 'CIS Controls'],
        default=['NIST CSF', 'ISO 27001', 'SOC 2', 'GDPR']
    )

    st.markdown("---")
    st.markdown("### 📅 Reporting Period")

    report_period = st.selectbox(
        "Select Period",
        options=["Current Month", "Last Quarter", "Year to Date", "Custom Range"]
    )

    st.markdown("---")
    st.markdown("### 🔍 Audit Trail Filters")

    audit_severity = st.multiselect(
        "Severity",
        options=['Critical', 'High', 'Medium', 'Low', 'Info'],
        default=['Critical', 'High', 'Medium']
    )

    audit_type = st.multiselect(
        "Event Type",
        options=['Access Control', 'Data Modification', 'Configuration Change', 'Policy Update', 'Compliance Check'],
        default=['Access Control', 'Data Modification', 'Configuration Change']
    )

    st.markdown("---")
    st.markdown("### 📥 Export Options")

    if st.button("Generate Compliance Report"):
        st.success("Compliance report generated!")

    if st.button("Export Audit Log"):
        st.success("Audit log exported!")

# Framework data
frameworks_data = {
    'NIST CSF': {
        'score': 94,
        'controls_total': 98,
        'controls_passed': 92,
        'controls_failed': 6,
        'last_audit': '2025-12-15',
        'next_audit': '2026-03-15',
        'categories': {
            'Identify': 96,
            'Protect': 94,
            'Detect': 92,
            'Respond': 93,
            'Recover': 95
        }
    },
    'ISO 27001': {
        'score': 91,
        'controls_total': 114,
        'controls_passed': 104,
        'controls_failed': 10,
        'last_audit': '2025-11-20',
        'next_audit': '2026-05-20',
        'categories': {
            'Information Security Policies': 95,
            'Organization of Information Security': 89,
            'Human Resource Security': 92,
            'Asset Management': 88,
            'Access Control': 93,
            'Cryptography': 90
        }
    },
    'SOC 2': {
        'score': 96,
        'controls_total': 64,
        'controls_passed': 62,
        'controls_failed': 2,
        'last_audit': '2025-12-01',
        'next_audit': '2026-06-01',
        'categories': {
            'Security': 98,
            'Availability': 95,
            'Processing Integrity': 96,
            'Confidentiality': 97,
            'Privacy': 94
        }
    },
    'GDPR': {
        'score': 88,
        'controls_total': 42,
        'controls_passed': 37,
        'controls_failed': 5,
        'last_audit': '2025-10-10',
        'next_audit': '2026-04-10',
        'categories': {
            'Lawfulness of Processing': 92,
            'Data Subject Rights': 85,
            'Data Protection by Design': 87,
            'Data Breach Notification': 90,
            'Data Transfer': 84
        }
    },
    'HIPAA': {
        'score': 92,
        'controls_total': 78,
        'controls_passed': 72,
        'controls_failed': 6,
        'last_audit': '2025-12-05',
        'next_audit': '2026-06-05',
        'categories': {
            'Administrative Safeguards': 94,
            'Physical Safeguards': 91,
            'Technical Safeguards': 90,
            'Organizational Requirements': 93
        }
    },
    'PCI DSS': {
        'score': 89,
        'controls_total': 52,
        'controls_passed': 46,
        'controls_failed': 6,
        'last_audit': '2025-11-15',
        'next_audit': '2026-02-15',
        'categories': {
            'Build and Maintain Secure Network': 91,
            'Protect Cardholder Data': 88,
            'Maintain Vulnerability Management': 87,
            'Implement Strong Access Control': 90,
            'Monitor and Test Networks': 89
        }
    },
    'CIS Controls': {
        'score': 93,
        'controls_total': 18,
        'controls_passed': 17,
        'controls_failed': 1,
        'last_audit': '2025-12-20',
        'next_audit': '2026-06-20',
        'categories': {
            'Basic Controls': 95,
            'Foundational Controls': 92,
            'Organizational Controls': 91
        }
    }
}

def get_score_class(score):
    """Get CSS class based on score"""
    if score >= 95:
        return 'score-excellent'
    elif score >= 85:
        return 'score-good'
    elif score >= 70:
        return 'score-warning'
    else:
        return 'score-critical'

def generate_audit_trail(num_entries=50):
    """Generate audit trail entries"""
    entries = []

    event_types = ['Access Control', 'Data Modification', 'Configuration Change', 'Policy Update', 'Compliance Check']
    severities = ['Critical', 'High', 'Medium', 'Low', 'Info']
    users = ['admin@company.com', 'security.team@company.com', 'compliance.officer@company.com',
             'john.doe@company.com', 'jane.smith@company.com', 'SYSTEM']
    actions = ['Updated', 'Created', 'Deleted', 'Accessed', 'Modified', 'Reviewed', 'Approved', 'Rejected']

    for i in range(num_entries):
        entries.append({
            'timestamp': datetime.now() - timedelta(hours=random.randint(1, 720)),
            'event_type': random.choice(event_types),
            'severity': random.choice(severities),
            'user': random.choice(users),
            'action': random.choice(actions),
            'resource': f'Resource_{random.randint(100, 999)}',
            'description': f'Audit event for compliance tracking',
            'ip_address': f'192.168.{random.randint(1,255)}.{random.randint(1,255)}',
            'status': random.choice(['Success', 'Failed', 'Pending'])
        })

    return pd.DataFrame(entries)

# Header
st.markdown('<h1 class="compliance-header">📜 Compliance & Audit Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="compliance-subtitle">Comprehensive Regulatory Compliance Tracking & Audit Management</p>', unsafe_allow_html=True)

# Overall Compliance Score
st.markdown("---")

if selected_frameworks:
    avg_score = np.mean([frameworks_data[fw]['score'] for fw in selected_frameworks if fw in frameworks_data])

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Overall Compliance Score",
            f"{avg_score:.1f}%",
            delta="+2.3% vs last month"
        )

    with col2:
        total_controls = sum([frameworks_data[fw]['controls_total'] for fw in selected_frameworks if fw in frameworks_data])
        st.metric(
            "Total Controls",
            total_controls,
            delta="+5 new controls"
        )

    with col3:
        total_passed = sum([frameworks_data[fw]['controls_passed'] for fw in selected_frameworks if fw in frameworks_data])
        st.metric(
            "Controls Passed",
            total_passed,
            delta="+12 improvements"
        )

    with col4:
        total_failed = sum([frameworks_data[fw]['controls_failed'] for fw in selected_frameworks if fw in frameworks_data])
        st.metric(
            "Controls Failed",
            total_failed,
            delta="-3 resolved"
        )

    # Framework Details
    st.markdown("---")
    st.markdown('<h2 class="section-title">🎯 Framework Compliance Status</h2>', unsafe_allow_html=True)

    for framework in selected_frameworks:
        if framework in frameworks_data:
            data = frameworks_data[framework]
            score_class = get_score_class(data['score'])

            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f'''
                <div class="framework-card">
                    <div class="framework-name">{framework}</div>
                    <div style="margin: 1rem 0;">
                        <span class="score-badge {score_class}">{data["score"]}% Compliant</span>
                    </div>
                    <div style="display: flex; gap: 2rem; margin-top: 1rem; color: #64748B;">
                        <div>
                            <strong style="color: var(--compliance-navy);">{data["controls_passed"]}/{data["controls_total"]}</strong>
                            Controls Passed
                        </div>
                        <div>
                            <strong style="color: var(--compliance-red);">{data["controls_failed"]}</strong>
                            Controls Failed
                        </div>
                        <div>
                            Last Audit: <strong style="color: var(--compliance-navy);">{data["last_audit"]}</strong>
                        </div>
                        <div>
                            Next Audit: <strong style="color: var(--compliance-blue);">{data["next_audit"]}</strong>
                        </div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)

            with col2:
                # Category breakdown
                categories = data['categories']
                fig = go.Figure(data=[
                    go.Bar(
                        y=list(categories.keys()),
                        x=list(categories.values()),
                        orientation='h',
                        marker=dict(
                            color=list(categories.values()),
                            colorscale='RdYlGn',
                            showscale=False
                        ),
                        text=[f'{v}%' for v in categories.values()],
                        textposition='inside',
                        insidetextanchor='middle'
                    )
                ])

                fig.update_layout(
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    font=dict(color='#0F172A', size=10),
                    height=200,
                    margin=dict(l=0, r=0, t=0, b=0),
                    xaxis=dict(range=[0, 100], showticklabels=False, showgrid=False),
                    yaxis=dict(showgrid=False)
                )

                st.plotly_chart(fig, use_container_width=True)

    # Compliance Trends
    st.markdown("---")
    st.markdown('<h2 class="section-title">📈 Compliance Trends</h2>', unsafe_allow_html=True)

    st.markdown('<div class="panel-white">', unsafe_allow_html=True)

    # Generate trend data
    months = pd.date_range(end=datetime.now(), periods=12, freq='M')
    trend_data = {}

    for framework in selected_frameworks:
        if framework in frameworks_data:
            base_score = frameworks_data[framework]['score']
            # Generate realistic trend
            trend_data[framework] = [
                base_score + random.uniform(-5, 3) for _ in range(12)
            ]

    fig = go.Figure()

    for framework, scores in trend_data.items():
        fig.add_trace(go.Scatter(
            x=months,
            y=scores,
            mode='lines+markers',
            name=framework,
            line=dict(width=3),
            marker=dict(size=6)
        ))

    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='#0F172A'),
        xaxis=dict(
            gridcolor='#E2E8F0',
            showgrid=True,
            title='Month'
        ),
        yaxis=dict(
            gridcolor='#E2E8F0',
            showgrid=True,
            title='Compliance Score (%)',
            range=[70, 100]
        ),
        height=400,
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Audit Trail
    st.markdown("---")
    st.markdown('<h2 class="section-title">📋 Audit Trail</h2>', unsafe_allow_html=True)

    st.markdown('<div class="panel-white">', unsafe_allow_html=True)

    audit_df = generate_audit_trail(100)

    # Apply filters
    filtered_audit = audit_df[
        (audit_df['severity'].isin(audit_severity)) &
        (audit_df['event_type'].isin(audit_type))
    ].sort_values('timestamp', ascending=False)

    # Search
    search_term = st.text_input("🔍 Search audit trail (user, resource, action...)", "")

    if search_term:
        mask = filtered_audit.astype(str).apply(
            lambda x: x.str.contains(search_term, case=False, na=False)
        ).any(axis=1)
        filtered_audit = filtered_audit[mask]

    # Display audit entries
    st.markdown(f"**Showing {len(filtered_audit)} audit entries**")

    for idx, row in filtered_audit.head(20).iterrows():
        severity_color = {
            'Critical': '#DC2626',
            'High': '#D97706',
            'Medium': '#2563EB',
            'Low': '#059669',
            'Info': '#64748B'
        }.get(row['severity'], '#64748B')

        st.markdown(f'''
        <div class="audit-entry">
            <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 0.5rem;">
                <div>
                    <strong style="color: var(--compliance-navy); font-size: 1.05rem;">
                        {row['action']} - {row['event_type']}
                    </strong>
                    <div style="margin-top: 0.3rem; color: #64748B; font-size: 0.9rem;">
                        User: {row['user']} | Resource: {row['resource']} | IP: {row['ip_address']}
                    </div>
                </div>
                <div>
                    <span style="background: {severity_color}; color: white; padding: 0.3rem 0.8rem; border-radius: 12px; font-size: 0.8rem; font-weight: 600;">
                        {row['severity']}
                    </span>
                </div>
            </div>
            <div style="color: #64748B; font-size: 0.85rem;">
                {row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} | Status: {row['status']}
            </div>
        </div>
        ''', unsafe_allow_html=True)

    # Export audit trail
    col1, col2 = st.columns([1, 4])
    with col1:
        csv_audit = filtered_audit.to_csv(index=False)
        st.download_button(
            label="📥 Export Audit Trail (CSV)",
            data=csv_audit,
            file_name=f'audit_trail_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            mime='text/csv'
        )

    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.warning("Please select at least one framework from the sidebar to view compliance data.")

# Footer
st.markdown(f"""
<div style="text-align: center; margin-top: 3rem; padding: 2rem; color: #64748B; font-size: 0.9rem;">
    📜 Compliance & Audit Dashboard | Report Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} |
    Data Retention: 7 years | Audit Log Encryption: AES-256
</div>
""", unsafe_allow_html=True)
