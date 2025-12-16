# 🛡️ AI-Driven SOC Dashboard Documentation

**URL**: `http://10.45.254.19:8535/`  
**Version**: v2.1.3  
**Last Updated**: October 4, 2025

---

## 📋 Table of Contents

1. [Dashboard Overview](#dashboard-overview)
2. [Overview & Funnel Tab](#overview--funnel-tab)
3. [Alert Review Tab](#alert-review-tab)
4. [Navigation Guide](#navigation-guide)
5. [Data Sources](#data-sources)
6. [Troubleshooting](#troubleshooting)

---

## 🎯 Dashboard Overview

The AI-Driven SOC Dashboard is a comprehensive security operations center interface that provides real-time monitoring, threat analysis, and incident management for telecom organizations. The dashboard integrates multiple data sources including security logs, social media intelligence, and AI-powered analysis.

### Key Features
- **Real-time Threat Monitoring**: Live security event analysis
- **AI-Powered Analysis**: Gemini LLM integration for contextual insights
- **Social Media Intelligence**: Twitter and TikTok monitoring
- **BigQuery Integration**: Scalable data storage and analytics
- **Multi-tab Interface**: Organized workflow for different SOC functions

---

## 📊 Overview & Funnel Tab

The "Overview & Funnel" tab provides a high-level view of the security operations center with key metrics, trends, and the security operations funnel.

### 🎯 Main Components

#### 1. **Header Section**
- **Dashboard Title**: "AI-Driven SOC Dashboard"
- **Last Updated**: Real-time timestamp showing data freshness
- **Navigation**: Quick access to different dashboard sections

#### 2. **Key Performance Indicators (KPIs)**

##### **Critical Security Metrics**
```
📈 AI Model Accuracy: 89.2%
⚡ Processing Rate: 98.5%
🔍 False Positive Rate: 14.8%
📊 Total Predictions: 1,000
```

##### **Real-time Statistics**
- **Active Alerts**: Current number of open security incidents
- **Threat Level**: Overall security posture (Low/Medium/High/Critical)
- **Response Time**: Average time to acknowledge and resolve incidents
- **Coverage**: Percentage of monitored systems and networks

#### 3. **Security Operations Funnel**

The funnel visualization shows the flow of security events through different stages:

##### **Stage 1: Raw Events**
- **Volume**: Total security events detected
- **Source**: Network logs, endpoint detection, user behavior
- **Rate**: Events per minute/hour

##### **Stage 2: Filtered Events**
- **Volume**: Events after noise reduction
- **Filtering**: Duplicate removal, known good activity
- **Reduction Rate**: Percentage of events filtered out

##### **Stage 3: Prioritized Alerts**
- **Volume**: Events requiring investigation
- **Prioritization**: Based on threat score, asset criticality
- **Categories**: High, Medium, Low severity

##### **Stage 4: Investigated Cases**
- **Volume**: Alerts actively being investigated
- **SLA Compliance**: Response time metrics
- **Analyst Assignment**: Workload distribution

##### **Stage 5: Resolved Incidents**
- **Volume**: Successfully resolved cases
- **Resolution Time**: Average time to closure
- **False Positive Rate**: Accuracy of initial detection

#### 4. **Trend Analysis Charts**

##### **Threat Volume Over Time**
- **X-axis**: Time (last 24 hours, 7 days, 30 days)
- **Y-axis**: Number of threats detected
- **Visualization**: Line chart showing trends
- **Interactive**: Hover for detailed information

##### **Threat Category Distribution**
- **Visualization**: Pie chart or bar chart
- **Categories**: Malware, Phishing, Data Exfiltration, etc.
- **Colors**: Color-coded by severity level

##### **Geographic Threat Map**
- **Visualization**: World map with threat indicators
- **Data**: Source IP geolocation
- **Heat Map**: Intensity based on threat volume

#### 5. **AI Analysis Section**

##### **Gemini LLM Integration**
```
🧠 AI Context Enrichment (Gemini LLM)
├── Summary: High-level threat assessment
├── Contextual Insights: Key findings and patterns
├── AI Recommendations: Suggested actions
└── Confidence Score: AI analysis reliability
```

##### **ML Threat Scoring**
- **Model**: CLA v2.1.3 (Contextual Learning Algorithm)
- **Threat Score Range**: 0.0 - 1.0
- **Confidence Levels**: Low, Medium, High
- **Anomaly Detection**: Unusual patterns identification

#### 6. **Quick Actions Panel**

##### **Incident Response**
- **Acknowledge Alert**: Mark alerts as reviewed
- **Escalate**: Send to higher priority queue
- **Create Ticket**: Generate incident tickets
- **Whitelist**: Add to trusted sources

##### **Dashboard Controls**
- **Refresh Data**: Manual data update
- **Export Report**: Download current view
- **Filter Settings**: Adjust display parameters
- **Notification Settings**: Configure alerts

---

## 🚨 Alert Review Tab

The "Alert Review" tab provides detailed analysis and management of individual security alerts with comprehensive investigation tools.

### 🔍 Main Components

#### 1. **Alert List View**

##### **Alert Cards Layout**
Each alert is displayed as an interactive card with:

```
🚨 ALERT CARD
├── Alert ID: SM_1234567890
├── Timestamp: 2025-10-04 12:30:45 UTC
├── Severity: 🔴 HIGH
├── Category: Cyber Threat / Customer Complaint / General Mention
├── Source: Twitter / TikTok / Network Logs
├── Threat Score: 0.85 (85%)
├── Status: New / Acknowledged / Investigating / Resolved
└── Actions: [Analyze] [Acknowledge] [Escalate] [Resolve]
```

##### **Alert Information**
- **Alert ID**: Unique identifier for tracking
- **Timestamp**: When the alert was generated
- **Severity**: High, Medium, Low classification
- **Category**: Type of threat or incident
- **Source Platform**: Social media or network source
- **Threat Score**: AI-calculated risk level (0-100%)
- **Status**: Current workflow state

#### 2. **Filtering and Search**

##### **Filter Options**
- **Severity Filter**: High, Medium, Low, All
- **Category Filter**: Cyber Threat, Customer Complaint, General Mention
- **Source Filter**: Twitter, TikTok, Network Logs
- **Time Range**: Last hour, 24 hours, 7 days, custom range
- **Status Filter**: New, Acknowledged, Investigating, Resolved

##### **Search Functionality**
- **Text Search**: Search within alert content
- **User Search**: Filter by specific usernames
- **Keyword Search**: Find alerts containing specific terms
- **Advanced Search**: Complex query builder

#### 3. **Alert Detail View**

When an alert is selected, detailed information is displayed:

##### **Alert Content**
```
📱 SOCIAL MEDIA ALERT
├── Platform: Twitter
├── Username: @username
├── User Info: 1.2K followers, Verified: No
├── Post Content: "Indosat bermasalah lagi, internet lambat banget!"
├── Engagement: 45 retweets, 123 likes, 12 replies
├── Hashtags: #indosat #keluhan #internet
└── Link: [View Original Post]
```

##### **Threat Analysis**
```
🔍 THREAT ANALYSIS
├── Threat Score: 0.75 (75%)
├── Risk Factors: customer_complaint, medium_influence
├── Keywords Found: indosat, bermasalah, lambat
├── Category: Customer Complaint
├── Severity: Medium
└── Confidence: High (0.89)
```

##### **AI-Powered Analysis**
```
🧠 AI CONTEXT ENRICHMENT (GEMINI LLM)
├── Summary: Customer expressing frustration with service quality
├── Contextual Insights:
│   ├── Language: Indonesian
│   ├── Sentiment: Negative
│   ├── Intent: Complaint/Feedback
│   └── Business Impact: Medium
├── AI Recommendations:
│   ├── Monitor for escalation
│   ├── Check service status
│   └── Consider proactive response
└── Confidence Score: 0.89 (High)
```

#### 4. **Investigation Tools**

##### **Threat Intelligence Lookup**
- **IP Address Reputation**: Check against threat databases
- **Domain Analysis**: WHOIS information and reputation
- **File Hash Check**: Malware detection
- **URL Analysis**: Phishing and malware scanning

##### **Historical Correlation**
- **Similar Incidents**: Past alerts with similar characteristics
- **Pattern Recognition**: Recurring threat patterns
- **Timeline Analysis**: Chronological event correlation
- **Attack Attribution**: Known threat actor associations

##### **Contextual Information**
- **User History**: Previous activity from the same user
- **Geographic Data**: Location information
- **Network Context**: Infrastructure details
- **Business Context**: Impact on operations

#### 5. **Response Actions**

##### **Immediate Actions**
- **Acknowledge**: Mark alert as reviewed
- **Escalate**: Move to higher priority queue
- **Assign**: Assign to specific analyst
- **Whitelist**: Add to trusted sources

##### **Investigation Actions**
- **Deep Dive**: Detailed forensic analysis
- **Correlate**: Find related incidents
- **Enrich**: Gather additional intelligence
- **Document**: Record findings and actions

##### **Resolution Actions**
- **Resolve**: Mark as completed
- **False Positive**: Mark as non-threat
- **Create Ticket**: Generate incident ticket
- **Block**: Implement blocking measures

#### 6. **Analytics and Reporting**

##### **Alert Statistics**
- **Total Alerts**: Count of all alerts
- **By Severity**: Breakdown by threat level
- **By Category**: Distribution by type
- **Resolution Time**: Average time to closure

##### **Performance Metrics**
- **Response Time**: Time to first action
- **Resolution Rate**: Percentage of resolved alerts
- **False Positive Rate**: Accuracy of detection
- **Analyst Performance**: Individual metrics

##### **Export Options**
- **CSV Export**: Download alert data
- **PDF Report**: Generate formatted reports
- **API Access**: Programmatic data access
- **Integration**: Connect to external systems

---

## 🧭 Navigation Guide

### **Tab Navigation**
1. **Overview & Funnel**: High-level dashboard view
2. **Alert Review**: Detailed alert analysis and management
3. **Settings**: Dashboard configuration and preferences

### **Keyboard Shortcuts**
- **Ctrl + R**: Refresh data
- **Ctrl + F**: Search alerts
- **Ctrl + E**: Export current view
- **Esc**: Close detailed views

### **User Interface Elements**
- **🔴 Red Indicators**: High severity alerts
- **🟡 Yellow Indicators**: Medium severity alerts
- **🟢 Green Indicators**: Low severity alerts
- **⚡ Lightning Icons**: Real-time updates
- **📊 Chart Icons**: Analytics and trends

---

## 📡 Data Sources

### **Primary Sources**
1. **BigQuery Database**: `chronicle-dev-2be9.soc_data.*`
   - Social media alerts
   - Network security logs
   - Threat intelligence feeds

2. **Social Media APIs**
   - Twitter API v2: Real-time tweet monitoring
   - TikTok API: Video and comment analysis

3. **AI/ML Services**
   - Google Gemini LLM: Contextual analysis
   - Custom ML Models: Threat scoring and classification

### **Data Flow**
```
Raw Data → Filtering → Analysis → Scoring → Dashboard Display
    ↓         ↓          ↓         ↓           ↓
  APIs → BigQuery → AI/ML → Scoring → Streamlit UI
```

---

## 🔧 Troubleshooting

### **Common Issues**

#### **Dashboard Not Loading**
- **Check URL**: Ensure `http://10.45.254.19:8535/` is correct
- **Network Access**: Verify firewall and network connectivity
- **Browser Compatibility**: Use Chrome, Firefox, or Edge

#### **No Data Displayed**
- **BigQuery Connection**: Check service account credentials
- **API Keys**: Verify Twitter and Gemini API keys
- **Data Freshness**: Check if data collection is running

#### **Slow Performance**
- **Data Volume**: Reduce time range or alert limit
- **Browser Cache**: Clear browser cache and cookies
- **Network Speed**: Check internet connection

#### **AI Analysis Errors**
- **API Quota**: Check Gemini API usage limits
- **Model Availability**: Verify model endpoints
- **Authentication**: Ensure API keys are valid

### **Support Contacts**
- **Technical Issues**: Contact SOC technical team
- **Data Questions**: Reach out to data engineering team
- **Feature Requests**: Submit through IT service desk

---

## 📚 Additional Resources

### **Training Materials**
- **SOC Analyst Training**: Comprehensive guide for new analysts
- **Dashboard Tutorial**: Step-by-step walkthrough
- **Best Practices**: Recommended workflows and procedures

### **Documentation Links**
- **API Documentation**: Technical reference for integrations
- **Security Policies**: Organizational security guidelines
- **Incident Response**: Standard operating procedures

### **Updates and Maintenance**
- **Version History**: Track dashboard updates and improvements
- **Scheduled Maintenance**: Planned downtime and updates
- **Feature Roadmap**: Upcoming enhancements and capabilities

---

**Document Version**: 1.0  
**Last Updated**: October 4, 2025  
**Maintained By**: SOC Technical Team  
**Review Cycle**: Monthly
