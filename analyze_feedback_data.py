#!/usr/bin/env python3
"""
Feedback Data Analysis
=====================

Analyze the extracted CSV data to understand why only 14 rows were extracted.
"""

import pandas as pd
from datetime import datetime
import re

def analyze_csv_data(csv_text: str):
    """Analyze the CSV data from the user's extraction"""
    
    print("📊 Feedback Data Analysis")
    print("=" * 60)
    
    # Split into lines and analyze
    lines = csv_text.strip().split('\n')
    
    print(f"📋 Total Rows Found: {len(lines)}")
    print()
    
    # Analyze each row
    print("🔍 Row-by-Row Analysis:")
    print("-" * 40)
    
    for i, line in enumerate(lines, 1):
        # Parse the CSV-like data
        fields = line.split('\t')  # Looks like tab-separated
        
        if len(fields) >= 10:  # Expected number of columns
            alarm_id = fields[0] if fields[0] else "N/A"
            case_class = fields[1] if fields[1] else "N/A"
            action_type = fields[2] if fields[2] else "N/A"
            ada_confidence = fields[3] if fields[3] else "N/A"
            taa_confidence = fields[4] if fields[4] else "N/A"
            
            print(f"  Row {i:2d}: AlarmID={alarm_id}, Class={case_class}, Action={action_type}")
            
            if i <= 5:  # Show details for first 5 rows
                print(f"         ADA Confidence: {ada_confidence}")
                print(f"         TAA Confidence: {taa_confidence}")
                print()
    
    # Extract unique values for analysis
    alarm_ids = []
    case_classes = []
    action_types = []
    
    for line in lines:
        fields = line.split('\t')
        if len(fields) >= 3:
            alarm_ids.append(fields[0])
            case_classes.append(fields[1])
            action_types.append(fields[2])
    
    print("\n📈 Data Summary:")
    print("-" * 30)
    print(f"🔢 Unique Alarm IDs: {len(set(alarm_ids))}")
    print(f"📊 Unique Case Classes: {set(case_classes)}")
    print(f"⚡ Unique Action Types: {set(action_types)}")
    
    # Check for patterns in alarm IDs
    print("\n🔍 Alarm ID Patterns:")
    print("-" * 30)
    for alarm_id in set(alarm_ids):
        if alarm_id:
            count = alarm_ids.count(alarm_id)
            print(f"  {alarm_id}: appears {count} time(s)")
    
    # Analyze why only 14 rows
    print("\n🤔 Why Only 14 Rows?")
    print("-" * 30)
    print("Possible reasons:")
    print("1. 📅 Data filtering: Dashboard may be showing recent data only")
    print("2. 📄 Pagination: More data might be on other pages")
    print("3. 🔍 Search/Filter: Active filters limiting results")
    print("4. 📊 Dashboard configuration: Limited display rows")
    print("5. 🗄️  Database: Only 14 records exist in the source")
    
    return {
        "total_rows": len(lines),
        "unique_alarms": len(set(alarm_ids)),
        "unique_classes": set(case_classes),
        "unique_actions": set(action_types)
    }

def check_dashboard_for_more_data():
    """Provide recommendations to check for more data"""
    
    print("\n🔍 How to Check for More Data:")
    print("-" * 40)
    
    recommendations = [
        {
            "title": "Check Dashboard Pagination",
            "steps": [
                "Look for 'Next Page' or pagination controls",
                "Check if there's a 'Show More' button",
                "Look for page numbers (1, 2, 3, etc.)"
            ]
        },
        {
            "title": "Check Date/Time Filters", 
            "steps": [
                "Look for date range selectors",
                "Check if there's a 'Last 24 hours' filter",
                "Try expanding the time range"
            ]
        },
        {
            "title": "Check Display Settings",
            "steps": [
                "Look for 'Rows per page' settings",
                "Check if there's a 'Show All' option",
                "Look for display limit controls"
            ]
        },
        {
            "title": "Check Search/Filter Box",
            "steps": [
                "Clear any active search terms",
                "Remove any applied filters",
                "Reset to default view"
            ]
        }
    ]
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. 🎯 {rec['title']}:")
        for step in rec['steps']:
            print(f"   • {step}")
    
    print("\n💡 Enhanced Extraction Script:")
    print("-" * 40)
    print("Try this enhanced JavaScript to capture more data:")
    
    enhanced_script = '''
// Enhanced extraction that looks for pagination and more data
(function() {
    console.log("🔍 Enhanced data extraction starting...");
    
    // First, try to find and click "Show All" or expand options
    const showAllButtons = document.querySelectorAll('button, a, span');
    for (let btn of showAllButtons) {
        const text = btn.innerText.toLowerCase();
        if (text.includes('show all') || text.includes('view all') || text.includes('load more')) {
            console.log("📄 Found 'Show All' button, clicking...");
            btn.click();
            setTimeout(() => extractData(), 2000); // Wait 2 seconds then extract
            return;
        }
    }
    
    // Check for pagination
    const pageButtons = document.querySelectorAll('[class*="page"], [class*="pagination"]');
    if (pageButtons.length > 0) {
        console.log("📑 Pagination detected - you may need to extract from multiple pages");
    }
    
    // Extract current data
    extractData();
    
    function extractData() {
        const table = document.querySelector("table");
        if (!table) {
            console.log("❌ No table found");
            return;
        }
        
        const rows = Array.from(table.querySelectorAll("tr"));
        console.log(`📊 Found ${rows.length} rows (including header)`);
        
        // Show row count breakdown
        const headerRows = table.querySelectorAll("thead tr, tr:first-child th").length;
        const dataRows = rows.length - (headerRows > 0 ? 1 : 0);
        console.log(`📋 Data rows: ${dataRows}, Header rows: ${headerRows > 0 ? 1 : 0}`);
        
        const csvRows = rows.map((row, index) => {
            const cells = Array.from(row.querySelectorAll("th, td"));
            return cells.map(cell => {
                let content = cell.innerText || cell.textContent || "";
                content = content.trim();
                content = content.replace(/"/g, '""');
                return `"${content}"`;
            }).join(",");
        });
        
        const csvContent = csvRows.join("\\n");
        const blob = new Blob([csvContent], {type:'text/csv;charset=utf-8;'});
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = `feedback_enhanced_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.csv`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        
        console.log(`✅ Enhanced CSV downloaded with ${rows.length} rows`);
        
        // Check for more data indicators
        if (dataRows < 50) {
            console.log("⚠️ Relatively few rows found. Check for:");
            console.log("  • Pagination controls");
            console.log("  • Date/time filters");
            console.log("  • 'Show more' buttons");
            console.log("  • Search filters that might be active");
        }
    }
})();
    '''
    
    print(enhanced_script)

def main():
    """Main analysis function"""
    
    # The user's extracted data (tab-separated format)
    user_data = """AlarmID	ADACase Class	CRAAction Type	ADAConfidence	TAAConfidence	ADAScore	TAASeverity	ADAValid	TAAValid	CRASuccess	ADAReasoning
10913665	bad_ip		0,05972222	0,05955556	0,09	0	TRUE	FALSE	FALSE	The alert was classified as 'bad_ip'. The external IP address 114.143.150 was checked using the 'BadIP Checker' tool to verify the reputation of the IP.
10913666	bad_ip		0,09		0,09	0	TRUE	FALSE	FALSE	The alert was classified as 'bad_ip' based on the alarm rule name 'AIE: IOH-TELCO: SSH Authentication'.
10913667	bad_ip		0,09	0,09	0,09	0,01	TRUE	FALSE	FALSE	The alert was classified as 'bad_ip' based on the alarm rule name 'AIE: IOH-TELCO: SSH Authentication'.
10913668	bad_ip		0,09	0,09	0,09	0,01	TRUE	FALSE	FALSE	The alert was classified as 'bad_ip' based on the alarm rule name. The Badness of 10.24.53 is in the variable name.
10913669	bad_ip		0,05972222	1	0,09	0	TRUE	FALSE	FALSE	The alert was classified as 'bad_ip' based on the alarm rule name 'AIE: IOH-TELCO: SSH Authentication'.
10913670	bad_ip		0,05972222	1	1	0,01	TRUE	FALSE	FALSE	The alert was classified as 'bad_ip' with high certainty. The IP address 10.24.53 is in the variable name.
10913671	bad_ip		0,05805556	0,09	0,05972222	0,01	TRUE	FALSE	FALSE	The alert was classified as 'bad_ip' based on the alarm rule name 'AIE: IOH-TELCO: SSH Authentication'.
10913672	bad_ip		0	0,09	0,09	0	TRUE	FALSE	FALSE	The alert was classified as 'bad_ip' based on the alarm rule name 'AIE: IOH-TELCO: SSH Authentication'.
10913673	bad_ip		0,05972222	0,05972222	0,09	0,05	TRUE	FALSE	FALSE	The alert was classified as 'bad_ip' based on the alarm rule name 'AIE: IOH-TELCO: SSH Authentication'.
10913674	bad_ip		0,09	1	0,09	0	TRUE	FALSE	FALSE	The alert was classified as 'bad_ip' based on the alarm rule name 'AIE: IOH-TELCO: SSH Authentication'.
10913675	bad_ip		1	0,09	1	0	TRUE	FALSE	FALSE	The alert was classified as 'bad_ip'. The Bad IP Checker tool was used to verify the reputation of the IP.
10913676	bad_ip		0,09	0,09	0,09	0,01	TRUE	FALSE	FALSE	The alert was classified as 'bad_ip'. The 'Bad IP Checker' tool was used to verify the reputation of the IP.
10913677	bad_ip		1	0,06875	1	0	TRUE	FALSE	FALSE	The alert was classified as 'bad_ip'. The impacted IP '114.0.80.203' was checked against known bad IP."""
    
    # Analyze the data
    analysis = analyze_csv_data(user_data)
    
    # Provide recommendations
    check_dashboard_for_more_data()
    
    print(f"\n🎯 Summary: You extracted {analysis['total_rows']} rows successfully!")
    print("This might be all the data available, or there might be more on other pages/filters.")

if __name__ == "__main__":
    main()


