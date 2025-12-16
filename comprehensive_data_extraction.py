#!/usr/bin/env python3
"""
Comprehensive Data Extraction Strategy
=====================================

Since the dashboard only shows limited records, let's explore alternative
methods to get more comprehensive feedback data.
"""

import requests
import json
import time
from datetime import datetime, timedelta
import csv
import os
from typing import Dict, List, Any, Optional

def explore_api_endpoints(base_url: str = "http://10.45.254.19:99") -> Dict[str, Any]:
    """
    Explore potential API endpoints for more comprehensive data
    
    Args:
        base_url: Base URL of the dashboard
        
    Returns:
        Dict with API exploration results
    """
    print("🔍 Exploring API Endpoints for More Data")
    print("=" * 50)
    
    # Common API endpoints to try
    endpoints = [
        "/api/feedback",
        "/api/feedback/all",
        "/api/data",
        "/api/alerts",
        "/api/export",
        "/feedback/api",
        "/feedback/data",
        "/feedback/export",
        "/data/feedback",
        "/export/feedback",
        "/feedback.json",
        "/feedback.csv",
        "/api/v1/feedback",
        "/api/v2/feedback",
        "/rest/feedback",
        "/graphql"
    ]
    
    results = {}
    
    for endpoint in endpoints:
        url = f"{base_url}{endpoint}"
        try:
            print(f"🌐 Testing: {endpoint}")
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '')
                content_length = len(response.content)
                
                results[endpoint] = {
                    "status": "success",
                    "status_code": 200,
                    "content_type": content_type,
                    "content_length": content_length,
                    "data_preview": response.text[:200] if content_length > 0 else "No content"
                }
                
                print(f"  ✅ SUCCESS: {content_type}, {content_length} bytes")
                if 'json' in content_type.lower():
                    print(f"  📊 JSON data found!")
                elif 'csv' in content_type.lower():
                    print(f"  📋 CSV data found!")
                
            else:
                results[endpoint] = {
                    "status": "failed",
                    "status_code": response.status_code,
                    "error": f"HTTP {response.status_code}"
                }
                print(f"  ❌ Failed: HTTP {response.status_code}")
                
        except requests.exceptions.Timeout:
            results[endpoint] = {"status": "timeout", "error": "Request timeout"}
            print(f"  ⏰ Timeout")
        except requests.exceptions.ConnectionError:
            results[endpoint] = {"status": "connection_error", "error": "Connection error"}
            print(f"  🔌 Connection error")
        except Exception as e:
            results[endpoint] = {"status": "error", "error": str(e)}
            print(f"  ❌ Error: {str(e)}")
    
    return results

def check_database_access() -> Dict[str, Any]:
    """
    Check for potential database access methods
    
    Returns:
        Dict with database access information
    """
    print("\n🗄️ Database Access Options")
    print("=" * 50)
    
    options = {
        "bigquery": {
            "description": "Google BigQuery integration",
            "files_to_check": [
                "bigquery_client.py",
                "bigquery_integration.py", 
                "bigquery_integration_rt.py",
                "Service Account BigQuery/sa-gatra-bigquery.json"
            ],
            "available": False
        },
        "firestore": {
            "description": "Google Firestore database",
            "files_to_check": [
                "config/cra_soar_mcp_config.json"
            ],
            "available": False
        },
        "local_files": {
            "description": "Local data files",
            "files_to_check": [
                "*.log",
                "*.json",
                "*.csv"
            ],
            "available": False
        }
    }
    
    for option_name, option_info in options.items():
        print(f"\n🔍 Checking {option_info['description']}:")
        
        for file_path in option_info['files_to_check']:
            if os.path.exists(file_path):
                option_info['available'] = True
                print(f"  ✅ Found: {file_path}")
            else:
                print(f"  ❌ Not found: {file_path}")
    
    return options

def explore_log_files() -> List[str]:
    """
    Look for log files that might contain more feedback data
    
    Returns:
        List of log files found
    """
    print("\n📋 Log Files Analysis")
    print("=" * 50)
    
    log_files = []
    
    # Common log file patterns
    log_patterns = [
        "*.log",
        "logs/*.log",
        "ada_*.log",
        "taa_*.log",
        "feedback*.log",
        "alert*.log"
    ]
    
    import glob
    
    for pattern in log_patterns:
        found_files = glob.glob(pattern)
        for file_path in found_files:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"📄 Found: {file_path} ({file_size:,} bytes)")
                log_files.append(file_path)
    
    return log_files

def analyze_existing_data_files() -> Dict[str, Any]:
    """
    Analyze existing data files in the project
    
    Returns:
        Dict with file analysis results
    """
    print("\n📊 Existing Data Files Analysis")
    print("=" * 50)
    
    data_files = {
        "json_files": [],
        "csv_files": [],
        "log_files": [],
        "config_files": []
    }
    
    import glob
    
    # Look for various data files
    file_patterns = {
        "json_files": ["*.json", "config/*.json", "*/*.json"],
        "csv_files": ["*.csv", "data/*.csv", "*/*.csv"],
        "log_files": ["*.log", "logs/*.log", "*/*.log"],
        "config_files": ["*.conf", "*.cfg", "*.toml", "*.yaml", "*.yml"]
    }
    
    for file_type, patterns in file_patterns.items():
        print(f"\n🔍 {file_type.replace('_', ' ').title()}:")
        
        for pattern in patterns:
            found_files = glob.glob(pattern, recursive=True)
            for file_path in found_files:
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    data_files[file_type].append({
                        "path": file_path,
                        "size": file_size,
                        "modified": datetime.fromtimestamp(os.path.getmtime(file_path))
                    })
                    print(f"  📄 {file_path} ({file_size:,} bytes)")
    
    return data_files

def suggest_alternative_extraction_methods() -> List[Dict[str, Any]]:
    """
    Suggest alternative methods to get more comprehensive data
    
    Returns:
        List of alternative methods
    """
    methods = [
        {
            "name": "BigQuery Direct Access",
            "description": "Access the underlying BigQuery database directly",
            "steps": [
                "Use the service account key in 'Service Account BigQuery/'",
                "Connect to BigQuery using bigquery_client.py",
                "Query the alerts/feedback tables directly",
                "Export larger datasets"
            ],
            "files_needed": ["Service Account BigQuery/sa-gatra-bigquery.json", "bigquery_client.py"],
            "complexity": "Medium"
        },
        {
            "name": "Log File Mining",
            "description": "Extract data from log files",
            "steps": [
                "Parse ada_alert_publisher.log for alert data",
                "Extract feedback from taa_service.log",
                "Parse manual_review_queue.log for reviews",
                "Combine data from multiple log sources"
            ],
            "files_needed": ["*.log files"],
            "complexity": "Low"
        },
        {
            "name": "API Endpoint Discovery",
            "description": "Find hidden API endpoints with more data",
            "steps": [
                "Test various API endpoints",
                "Check for GraphQL endpoints",
                "Try different authentication methods",
                "Look for export/download endpoints"
            ],
            "files_needed": ["Network access to server"],
            "complexity": "Medium"
        },
        {
            "name": "Database Direct Connection",
            "description": "Connect directly to the database",
            "steps": [
                "Identify the database type (Firestore, MySQL, etc.)",
                "Use database credentials from config files",
                "Query the feedback/alerts tables directly",
                "Export comprehensive datasets"
            ],
            "files_needed": ["Database credentials", "config files"],
            "complexity": "High"
        },
        {
            "name": "Scheduled Data Collection",
            "description": "Collect data over time using automation",
            "steps": [
                "Set up automated dashboard scraping",
                "Run extraction script every hour/day",
                "Accumulate data over time",
                "Deduplicate and merge datasets"
            ],
            "files_needed": ["Cron job or scheduler"],
            "complexity": "Medium"
        }
    ]
    
    return methods

def create_bigquery_extraction_script():
    """Create a script to extract data from BigQuery directly"""
    
    script_content = '''#!/usr/bin/env python3
"""
BigQuery Direct Data Extraction
===============================

Extract comprehensive feedback data directly from BigQuery.
"""

from google.cloud import bigquery
import pandas as pd
import json
import os
from datetime import datetime, timedelta

def extract_from_bigquery():
    """Extract feedback data from BigQuery"""
    
    # Set up BigQuery client
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "Service Account BigQuery/sa-gatra-bigquery.json"
    client = bigquery.Client()
    
    # Common table names to try
    possible_tables = [
        "ai-driven-soc.feedback.alerts",
        "ai-driven-soc.soc.feedback", 
        "ai-driven-soc.alerts.feedback",
        "ai-driven-soc.main.feedback",
        "gatra-project.feedback.alerts",
        "gatra-project.soc.alerts"
    ]
    
    print("🔍 Searching for feedback tables in BigQuery...")
    
    # Try to list datasets first
    try:
        datasets = list(client.list_datasets())
        print(f"📊 Found {len(datasets)} datasets:")
        for dataset in datasets:
            print(f"  • {dataset.dataset_id}")
            
            # List tables in each dataset
            tables = list(client.list_tables(dataset.dataset_id))
            for table in tables:
                print(f"    - {table.table_id}")
                
                # Try to query tables that might contain feedback
                if any(keyword in table.table_id.lower() for keyword in ['feedback', 'alert', 'review']):
                    query = f"""
                    SELECT *
                    FROM `{dataset.dataset_id}.{table.table_id}`
                    LIMIT 1000
                    """
                    
                    try:
                        result = client.query(query).to_dataframe()
                        print(f"    ✅ Successfully queried {len(result)} rows from {table.table_id}")
                        
                        # Save to CSV
                        filename = f"bigquery_{dataset.dataset_id}_{table.table_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        result.to_csv(filename, index=False)
                        print(f"    💾 Saved to {filename}")
                        
                    except Exception as e:
                        print(f"    ❌ Error querying {table.table_id}: {e}")
                        
    except Exception as e:
        print(f"❌ Error accessing BigQuery: {e}")
        print("💡 Make sure the service account key is valid and has proper permissions")

if __name__ == "__main__":
    extract_from_bigquery()
'''
    
    with open("bigquery_feedback_extraction.py", "w") as f:
        f.write(script_content)
    
    print("✅ Created bigquery_feedback_extraction.py")

def main():
    """Main function to explore comprehensive data extraction options"""
    
    print("🚀 Comprehensive Data Extraction Strategy")
    print("=" * 60)
    print("Since the dashboard shows limited records, let's explore alternatives...")
    print()
    
    # 1. Explore API endpoints
    api_results = explore_api_endpoints()
    
    # 2. Check database access options
    db_options = check_database_access()
    
    # 3. Analyze existing files
    data_files = analyze_existing_data_files()
    
    # 4. Look for log files
    log_files = explore_log_files()
    
    # 5. Suggest alternative methods
    print("\n💡 Alternative Data Extraction Methods")
    print("=" * 50)
    
    methods = suggest_alternative_extraction_methods()
    
    for i, method in enumerate(methods, 1):
        print(f"\n{i}. 🎯 {method['name']} (Complexity: {method['complexity']})")
        print(f"   📝 {method['description']}")
        print("   📋 Steps:")
        for step in method['steps']:
            print(f"      • {step}")
        print(f"   📁 Files needed: {', '.join(method['files_needed'])}")
    
    # 6. Create extraction scripts
    print("\n🔧 Creating Helper Scripts")
    print("=" * 50)
    
    create_bigquery_extraction_script()
    
    # 7. Summary and recommendations
    print("\n📊 Summary & Recommendations")
    print("=" * 50)
    
    successful_apis = [endpoint for endpoint, result in api_results.items() if result.get("status") == "success"]
    available_dbs = [name for name, info in db_options.items() if info["available"]]
    
    print(f"✅ Successful API endpoints: {len(successful_apis)}")
    print(f"✅ Available databases: {len(available_dbs)}")
    print(f"✅ Data files found: {sum(len(files) for files in data_files.values())}")
    print(f"✅ Log files found: {len(log_files)}")
    
    print("\n🎯 Recommended Next Steps:")
    
    if successful_apis:
        print("1. 🌐 Try the successful API endpoints for more data")
        for endpoint in successful_apis:
            print(f"   • {endpoint}")
    
    if available_dbs:
        print("2. 🗄️ Use direct database access:")
        for db in available_dbs:
            print(f"   • {db}")
    
    if log_files:
        print("3. 📋 Mine log files for historical data")
    
    print("4. 🔧 Run the BigQuery extraction script if you have access")
    print("5. 📊 Set up automated collection to gather more data over time")
    
    print(f"\n📁 Results saved to comprehensive_data_analysis.json")
    
    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "api_results": api_results,
        "database_options": db_options,
        "data_files": data_files,
        "log_files": log_files,
        "methods": methods
    }
    
    with open("comprehensive_data_analysis.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

if __name__ == "__main__":
    main()
