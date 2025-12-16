#!/usr/bin/env python3
"""
Log File Data Extraction
========================

Extract comprehensive feedback data from log files since the dashboard shows limited records.
"""

import re
import json
import csv
from datetime import datetime
from typing import List, Dict, Any
import os

def extract_from_taa_service_log(log_file: str = "taa_service.log") -> List[Dict[str, Any]]:
    """
    Extract feedback data from TAA service log
    
    Args:
        log_file: Path to the TAA service log file
        
    Returns:
        List of extracted feedback records
    """
    print(f"📋 Extracting from {log_file} (13.9 MB)...")
    
    if not os.path.exists(log_file):
        print(f"❌ File not found: {log_file}")
        return []
    
    feedback_records = []
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            line_count = 0
            feedback_count = 0
            
            for line in f:
                line_count += 1
                
                # Look for feedback-related patterns
                if any(keyword in line.lower() for keyword in ['feedback', 'alert', 'confidence', 'classification']):
                    
                    # Try to extract structured data
                    try:
                        # Look for JSON-like structures in the log
                        json_match = re.search(r'\{[^}]*\}', line)
                        if json_match:
                            json_str = json_match.group()
                            try:
                                data = json.loads(json_str)
                                if isinstance(data, dict):
                                    data['log_line'] = line_count
                                    data['timestamp'] = extract_timestamp_from_line(line)
                                    feedback_records.append(data)
                                    feedback_count += 1
                            except json.JSONDecodeError:
                                pass
                    except Exception:
                        pass
                    
                    # Extract key-value pairs from log lines
                    record = extract_key_values_from_line(line, line_count)
                    if record:
                        feedback_records.append(record)
                        feedback_count += 1
                
                # Progress indicator
                if line_count % 10000 == 0:
                    print(f"  📊 Processed {line_count:,} lines, found {feedback_count} feedback records")
            
            print(f"✅ Completed: {line_count:,} lines processed, {feedback_count} feedback records found")
            
    except Exception as e:
        print(f"❌ Error reading {log_file}: {e}")
    
    return feedback_records

def extract_from_manual_review_log(log_file: str = "manual_review_queue.log") -> List[Dict[str, Any]]:
    """
    Extract manual review data from log file
    
    Args:
        log_file: Path to the manual review log file
        
    Returns:
        List of extracted review records
    """
    print(f"📋 Extracting from {log_file} (7.2 MB)...")
    
    if not os.path.exists(log_file):
        print(f"❌ File not found: {log_file}")
        return []
    
    review_records = []
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            line_count = 0
            review_count = 0
            
            for line in f:
                line_count += 1
                
                # Look for review-related patterns
                if any(keyword in line.lower() for keyword in ['review', 'manual', 'analyst', 'decision']):
                    
                    record = extract_key_values_from_line(line, line_count)
                    if record:
                        record['source'] = 'manual_review'
                        review_records.append(record)
                        review_count += 1
                
                # Progress indicator
                if line_count % 5000 == 0:
                    print(f"  📊 Processed {line_count:,} lines, found {review_count} review records")
            
            print(f"✅ Completed: {line_count:,} lines processed, {review_count} review records found")
            
    except Exception as e:
        print(f"❌ Error reading {log_file}: {e}")
    
    return review_records

def extract_from_ada_publisher_log(log_file: str = "ada_alert_publisher.log") -> List[Dict[str, Any]]:
    """
    Extract alert data from ADA publisher log
    
    Args:
        log_file: Path to the ADA alert publisher log file
        
    Returns:
        List of extracted alert records
    """
    print(f"📋 Extracting from {log_file} (514 KB)...")
    
    if not os.path.exists(log_file):
        print(f"❌ File not found: {log_file}")
        return []
    
    alert_records = []
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            line_count = 0
            alert_count = 0
            
            for line in f:
                line_count += 1
                
                # Look for alert-related patterns
                if any(keyword in line.lower() for keyword in ['alert', 'publish', 'confidence', 'score']):
                    
                    record = extract_key_values_from_line(line, line_count)
                    if record:
                        record['source'] = 'ada_publisher'
                        alert_records.append(record)
                        alert_count += 1
                
                if line_count % 1000 == 0:
                    print(f"  📊 Processed {line_count:,} lines, found {alert_count} alert records")
            
            print(f"✅ Completed: {line_count:,} lines processed, {alert_count} alert records found")
            
    except Exception as e:
        print(f"❌ Error reading {log_file}: {e}")
    
    return alert_records

def extract_timestamp_from_line(line: str) -> str:
    """Extract timestamp from log line"""
    
    # Common timestamp patterns
    timestamp_patterns = [
        r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}',
        r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}',
        r'\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2}'
    ]
    
    for pattern in timestamp_patterns:
        match = re.search(pattern, line)
        if match:
            return match.group()
    
    return ""

def extract_key_values_from_line(line: str, line_number: int) -> Dict[str, Any]:
    """Extract key-value pairs from a log line"""
    
    record = {
        'line_number': line_number,
        'timestamp': extract_timestamp_from_line(line),
        'raw_line': line.strip()
    }
    
    # Extract common patterns
    patterns = {
        'alarm_id': r'alarm[_\s]*id[:\s]*(\d+)',
        'confidence': r'confidence[:\s]*([0-9.]+)',
        'score': r'score[:\s]*([0-9.]+)',
        'severity': r'severity[:\s]*(\w+)',
        'classification': r'class(?:ification)?[:\s]*(\w+)',
        'ip_address': r'(?:ip|address)[:\s]*(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})',
        'action': r'action[:\s]*(\w+)',
        'status': r'status[:\s]*(\w+)'
    }
    
    found_data = False
    
    for key, pattern in patterns.items():
        match = re.search(pattern, line.lower())
        if match:
            record[key] = match.group(1)
            found_data = True
    
    # Only return record if we found some structured data
    return record if found_data else None

def extract_from_json_files() -> List[Dict[str, Any]]:
    """Extract data from existing JSON files"""
    
    print("📄 Extracting from JSON files...")
    
    json_files = [
        "taa_false_positive_results.json",
        "taa_case_study_results.json"
    ]
    
    all_data = []
    
    for file_path in json_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                if isinstance(data, list):
                    all_data.extend(data)
                elif isinstance(data, dict):
                    all_data.append(data)
                
                print(f"  ✅ Loaded {file_path}")
                
            except Exception as e:
                print(f"  ❌ Error loading {file_path}: {e}")
        else:
            print(f"  ❌ File not found: {file_path}")
    
    return all_data

def save_comprehensive_dataset(all_records: List[Dict[str, Any]], filename: str = None):
    """Save all extracted records to CSV"""
    
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comprehensive_feedback_data_{timestamp}.csv"
    
    if not all_records:
        print("❌ No records to save")
        return
    
    print(f"💾 Saving {len(all_records)} records to {filename}...")
    
    # Get all unique keys from all records
    all_keys = set()
    for record in all_records:
        all_keys.update(record.keys())
    
    all_keys = sorted(list(all_keys))
    
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(all_records)
        
        print(f"✅ Successfully saved {len(all_records)} records to {filename}")
        print(f"📊 Columns: {len(all_keys)}")
        print(f"📋 Sample columns: {', '.join(list(all_keys)[:10])}")
        
        return filename
        
    except Exception as e:
        print(f"❌ Error saving CSV: {e}")
        return None

def main():
    """Main extraction function"""
    
    print("🚀 Comprehensive Log Data Extraction")
    print("=" * 60)
    print("Extracting feedback data from multiple log sources...")
    print()
    
    all_records = []
    
    # 1. Extract from TAA service log (largest file)
    taa_records = extract_from_taa_service_log()
    all_records.extend(taa_records)
    print(f"📊 TAA Service: {len(taa_records)} records")
    
    # 2. Extract from manual review log
    review_records = extract_from_manual_review_log()
    all_records.extend(review_records)
    print(f"📊 Manual Review: {len(review_records)} records")
    
    # 3. Extract from ADA publisher log
    ada_records = extract_from_ada_publisher_log()
    all_records.extend(ada_records)
    print(f"📊 ADA Publisher: {len(ada_records)} records")
    
    # 4. Extract from JSON files
    json_records = extract_from_json_files()
    all_records.extend(json_records)
    print(f"📊 JSON Files: {len(json_records)} records")
    
    print(f"\n📈 Total Records Extracted: {len(all_records)}")
    
    if all_records:
        # Save comprehensive dataset
        csv_file = save_comprehensive_dataset(all_records)
        
        print(f"\n🎉 Success! Comprehensive dataset created:")
        print(f"📁 File: {csv_file}")
        print(f"📊 Records: {len(all_records):,}")
        print(f"🔍 This is much more than the 14 records from the dashboard!")
        
        # Show sample of data types found
        sources = {}
        for record in all_records:
            source = record.get('source', 'unknown')
            sources[source] = sources.get(source, 0) + 1
        
        print(f"\n📋 Data Sources:")
        for source, count in sources.items():
            print(f"  • {source}: {count} records")
            
    else:
        print("❌ No records extracted from log files")
        print("💡 The log files might be empty or use different formats")

if __name__ == "__main__":
    main()


