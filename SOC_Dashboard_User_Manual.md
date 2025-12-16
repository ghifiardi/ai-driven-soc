# 📖 SOC Dashboard User Manual

**Dashboard URL**: `http://10.45.254.19:8535/`  
**For SOC Analysts, Security Engineers, and Incident Response Teams**

---

## 🎯 Getting Started

### **Accessing the Dashboard**
1. **Open your web browser** (Chrome, Firefox, or Edge recommended)
2. **Navigate to**: `http://10.45.254.19:8535/`
3. **Wait for loading**: Dashboard should load within 3-5 seconds
4. **Verify connection**: You should see the "AI-Driven SOC Dashboard" title

### **Browser Requirements**
- **JavaScript**: Must be enabled
- **Cookies**: Required for session management
- **Screen Resolution**: Minimum 1920x1080 recommended
- **Internet Connection**: Stable connection required

---

## 📊 Overview & Funnel Tab - User Guide

### **Understanding the Dashboard Layout**

#### **1. Header Section**
```
🛡️ AI-Driven SOC Dashboard                    Last Updated: 2025-10-04 12:30:45 UTC
```

**What it shows**:
- Dashboard title and security icon
- Real-time timestamp of last data update
- Navigation indicators for current section

#### **2. Critical Security Metrics Panel**

**Key Performance Indicators**:
```
📈 AI Model Accuracy: 89.2%          ⚡ Processing Rate: 98.5%
🔍 False Positive Rate: 14.8%        📊 Total Predictions: 1,000
```

**How to interpret**:
- **AI Model Accuracy (89.2%)**: How well the AI correctly identifies real threats
  - ✅ **Good**: >85%
  - ⚠️ **Needs attention**: 75-85%
  - ❌ **Poor**: <75%

- **Processing Rate (98.5%)**: Percentage of alerts processed successfully
  - ✅ **Good**: >95%
  - ⚠️ **Needs attention**: 90-95%
  - ❌ **Poor**: <90%

- **False Positive Rate (14.8%)**: Percentage of alerts that are not real threats
  - ✅ **Good**: <15%
  - ⚠️ **Needs attention**: 15-25%
  - ❌ **Poor**: >25%

#### **3. Security Operations Funnel**

**Understanding the Funnel**:
```
Raw Events (1000) → Filtered Events (450) → Prioritized Alerts (180) 
→ Investigated Cases (120) → Resolved Incidents (95)
```

**Stage-by-stage breakdown**:
1. **Raw Events (1000)**: All security events detected by monitoring systems
2. **Filtered Events (450)**: Events after removing noise and duplicates
3. **Prioritized Alerts (180)**: Events requiring investigation
4. **Investigated Cases (120)**: Alerts actively being analyzed
5. **Resolved Incidents (95)**: Successfully closed cases

**What to look for**:
- **Healthy funnel**: Gradual reduction at each stage
- **Bottlenecks**: Large drops between stages indicate processing issues
- **Volume spikes**: Sudden increases may indicate attacks

#### **4. Threat Volume Chart**

**Reading the chart**:
- **X-axis**: Time (last 24 hours)
- **Y-axis**: Number of threats detected
- **Line colors**: Different threat types
- **Hover**: Detailed information for each point

**Action items**:
- **Spike detection**: Look for sudden increases in threat volume
- **Pattern analysis**: Identify recurring attack times
- **Trend monitoring**: Track if threats are increasing/decreasing

#### **5. AI Analysis Section**

**Gemini LLM Analysis**:
```
🧠 AI Context Enrichment (Gemini LLM)
Summary: High severity anomaly alert indicates potentially malicious network activity...
```

**Understanding AI insights**:
- **Summary**: High-level threat assessment
- **Contextual Insights**: Key findings and patterns
- **AI Recommendations**: Suggested response actions
- **Confidence Score**: How certain the AI is (0-100%)

**How to use**:
- **High confidence (>80%)**: Trust AI recommendations
- **Medium confidence (50-80%)**: Use as guidance, verify manually
- **Low confidence (<50%)**: Manual investigation required

---

## 🚨 Alert Review Tab - User Guide

### **Navigating the Alert Review Interface**

#### **1. Alert List View**

**Alert Card Layout**:
```
🚨 SM_1234567890                    2025-10-04 12:30:45 UTC
🔴 HIGH | Cyber Threat | Twitter
@username • 1.2K followers • Threat Score: 85%
"Indosat bermasalah lagi, internet lambat banget!"
❤️ 123  🔄 45  💬 12
[Analyze] [Acknowledge] [Escalate] [Resolve]
```

**Understanding alert information**:
- **Alert ID (SM_1234567890)**: Unique identifier for tracking
- **Timestamp**: When the alert was generated
- **Severity**: 🔴 High, 🟡 Medium, 🟢 Low
- **Category**: Cyber Threat, Customer Complaint, General Mention
- **Source**: Twitter, TikTok, Network Logs
- **User info**: Username, follower count, verification status
- **Engagement**: Likes, retweets, replies
- **Threat Score**: AI-calculated risk level (0-100%)

#### **2. Using Filters**

**Filter Options**:
- **Severity Filter**: Dropdown to select High/Medium/Low/All
- **Category Filter**: Dropdown to select alert types
- **Time Range**: Slider to select time period
- **Threat Score**: Slider to set minimum threat score

**Step-by-step filtering**:
1. **Select severity**: Choose threat level to focus on
2. **Choose category**: Filter by alert type
3. **Set time range**: Look at recent alerts or historical data
4. **Adjust threat score**: Focus on high-confidence alerts
5. **Apply filters**: Results update automatically

**Filter combinations**:
- **High Priority**: High severity + Threat score >70%
- **Recent Critical**: High severity + Last 2 hours
- **Customer Issues**: Customer Complaint category
- **Social Media**: Twitter/TikTok sources only

#### **3. Alert Analysis Process**

**Step 1: Initial Assessment**
1. **Read the alert content**: Understand what happened
2. **Check threat score**: High scores need immediate attention
3. **Review user information**: Check follower count and verification
4. **Examine engagement**: High engagement may indicate viral threats

**Step 2: AI Analysis Review**
1. **Read AI summary**: Understand the AI's assessment
2. **Check confidence score**: High confidence = trust AI
3. **Review recommendations**: Follow AI-suggested actions
4. **Note key insights**: Important patterns or indicators

**Step 3: Manual Investigation**
1. **Click "Analyze" button**: Get detailed AI analysis
2. **Review threat intelligence**: Check IP/domain reputation
3. **Correlate with other alerts**: Look for similar incidents
4. **Check historical data**: See if this user/pattern is known

**Step 4: Response Actions**
1. **Acknowledge**: Mark alert as reviewed
2. **Escalate**: Move to higher priority if needed
3. **Assign**: Give to specific analyst if available
4. **Resolve**: Close alert when investigation complete

#### **4. Advanced Features**

**Search Functionality**:
- **Text search**: Find alerts containing specific words
- **User search**: Filter by specific usernames
- **Keyword search**: Find alerts with specific hashtags/keywords
- **Date range**: Search within specific time periods

**Export Options**:
- **CSV Export**: Download alert data for analysis
- **PDF Report**: Generate formatted reports
- **Screenshot**: Capture dashboard views

**Keyboard Shortcuts**:
- **Ctrl + R**: Refresh data
- **Ctrl + F**: Search alerts
- **Ctrl + E**: Export current view
- **Esc**: Close detailed views

---

## 🔍 Common Use Cases

### **Case 1: Daily Security Review**

**Morning routine (9:00 AM)**:
1. **Open Overview tab**: Check overnight metrics
2. **Review funnel**: Look for processing bottlenecks
3. **Check AI accuracy**: Ensure model performance is good
4. **Switch to Alert Review**: Review overnight alerts
5. **Filter for High severity**: Focus on critical issues
6. **Acknowledge reviewed alerts**: Mark as processed

**Afternoon check (2:00 PM)**:
1. **Check threat volume chart**: Look for attack patterns
2. **Review new alerts**: Process any new high-priority items
3. **Update status**: Change alert statuses as needed
4. **Export report**: Generate afternoon summary

### **Case 2: Incident Response**

**When high-severity alert appears**:
1. **Immediately acknowledge**: Mark as under investigation
2. **Click "Analyze"**: Get detailed AI analysis
3. **Review recommendations**: Follow AI guidance
4. **Escalate if needed**: Move to critical queue
5. **Document findings**: Record investigation notes
6. **Coordinate response**: Work with team members

**During ongoing incident**:
1. **Monitor new alerts**: Watch for related incidents
2. **Update status**: Keep alert status current
3. **Correlate data**: Look for attack patterns
4. **Generate reports**: Document incident details

### **Case 3: Weekly Analysis**

**Monday morning review**:
1. **Check weekly trends**: Look at threat volume over time
2. **Review AI performance**: Analyze model accuracy trends
3. **Identify patterns**: Look for recurring attack types
4. **Generate report**: Create weekly summary
5. **Plan improvements**: Identify areas for enhancement

---

## 🚨 Troubleshooting Guide

### **Dashboard Not Loading**

**Problem**: Dashboard shows "Page not found" or doesn't load
**Solutions**:
1. **Check URL**: Ensure `http://10.45.254.19:8535/` is correct
2. **Try different browser**: Switch to Chrome/Firefox/Edge
3. **Clear cache**: Clear browser cache and cookies
4. **Check network**: Verify internet connection
5. **Contact IT**: Report to technical support

### **No Data Displayed**

**Problem**: Dashboard loads but shows no alerts or metrics
**Solutions**:
1. **Check time range**: Verify filter settings
2. **Refresh page**: Click browser refresh button
3. **Check filters**: Ensure filters aren't too restrictive
4. **Wait for loading**: Data may take 30-60 seconds to load
5. **Contact data team**: Report data availability issues

### **Slow Performance**

**Problem**: Dashboard is slow or unresponsive
**Solutions**:
1. **Reduce time range**: Narrow down date filters
2. **Lower alert limit**: Reduce number of alerts displayed
3. **Close other tabs**: Free up browser memory
4. **Check internet speed**: Ensure stable connection
5. **Report performance issues**: Contact technical support

### **AI Analysis Errors**

**Problem**: AI analysis shows errors or "unavailable"
**Solutions**:
1. **Check API status**: AI services may be temporarily down
2. **Try again later**: Wait a few minutes and retry
3. **Use manual analysis**: Fall back to manual investigation
4. **Report API issues**: Contact technical support
5. **Check quota limits**: AI services may have usage limits

---

## 📞 Support and Contacts

### **Technical Support**
- **Dashboard Issues**: SOC Technical Team
- **Data Questions**: Data Engineering Team
- **AI/Analysis Issues**: ML Engineering Team
- **Network Problems**: IT Infrastructure Team

### **Emergency Contacts**
- **Critical Security Issues**: SOC Manager (24/7)
- **System Outages**: IT Service Desk
- **Data Breaches**: Incident Response Team

### **Training and Resources**
- **New User Training**: Contact SOC Training Team
- **Advanced Features**: Schedule with technical team
- **Best Practices**: Review internal documentation
- **Updates**: Check monthly training sessions

---

## 📚 Additional Resources

### **Documentation Links**
- **Technical Guide**: Detailed implementation documentation
- **API Reference**: Integration and development guides
- **Security Policies**: Organizational security guidelines
- **Incident Response**: Standard operating procedures

### **Training Materials**
- **Video Tutorials**: Step-by-step dashboard walkthrough
- **Interactive Demos**: Hands-on training sessions
- **Best Practices Guide**: Recommended workflows
- **Case Studies**: Real-world usage examples

### **Updates and Maintenance**
- **Release Notes**: Track new features and improvements
- **Scheduled Maintenance**: Planned downtime notifications
- **Feature Requests**: Submit enhancement suggestions
- **Bug Reports**: Report issues and problems

---

**User Manual Version**: 1.0  
**Last Updated**: October 4, 2025  
**Target Audience**: SOC Analysts, Security Engineers, Incident Response Teams  
**Review Cycle**: Quarterly
























