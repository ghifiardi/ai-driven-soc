#!/usr/bin/env python3
"""
Enhanced Feedback CSV Extractor
===============================

This script provides enhanced feedback data extraction with better debugging
and support for various HTML structures.
"""

import requests
import csv
import json
from bs4 import BeautifulSoup
from typing import List, Dict, Any

def analyze_page_structure(url: str) -> Dict[str, Any]:
    """
    Analyze the page structure to understand the data format
    
    Args:
        url: URL of the feedback page
        
    Returns:
        Dict containing analysis results
    """
    try:
        print(f"🔍 Analyzing page structure at: {url}")
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        analysis = {
            "status_code": response.status_code,
            "content_length": len(response.content),
            "title": soup.title.string if soup.title else "No title",
            "tables": len(soup.find_all('table')),
            "divs": len(soup.find_all('div')),
            "lists": len(soup.find_all(['ul', 'ol'])),
            "paragraphs": len(soup.find_all('p')),
            "pre_blocks": len(soup.find_all('pre')),
            "json_scripts": [],
            "data_attributes": []
        }
        
        # Look for JSON data in script tags
        scripts = soup.find_all('script')
        for script in scripts:
            if script.string:
                content = script.string.strip()
                if content.startswith('{') or content.startswith('['):
                    try:
                        json_data = json.loads(content)
                        analysis["json_scripts"].append({
                            "type": "json",
                            "length": len(content),
                            "keys": list(json_data.keys()) if isinstance(json_data, dict) else "array"
                        })
                    except:
                        pass
        
        # Look for data attributes
        elements_with_data = soup.find_all(attrs=lambda x: x and any(k.startswith('data-') for k in x.keys()))
        analysis["data_attributes"] = len(elements_with_data)
        
        return analysis
        
    except Exception as e:
        return {"error": str(e)}

def extract_feedback_data(url: str, output_file: str = "feedback_data.csv") -> bool:
    """
    Extract feedback data using multiple methods
    
    Args:
        url: URL of the feedback page
        output_file: Output CSV file name
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"🔍 Fetching data from: {url}")
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        print(f"✅ Successfully fetched page (Status: {response.status_code})")
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Method 1: Try to find tables
        tables = soup.find_all('table')
        if tables:
            print(f"✅ Found {len(tables)} table(s)")
            
            for i, table in enumerate(tables):
                print(f"📊 Processing table {i+1}...")
                
                rows = []
                for tr in table.find_all('tr'):
                    row = []
                    for cell in tr.find_all(['th', 'td']):
                        text = cell.get_text(strip=True)
                        text = text.replace('"', '""')
                        if ',' in text or '"' in text or '\n' in text or '\r' in text:
                            text = f'"{text}"'
                        row.append(text)
                    rows.append(row)
                
                if rows:
                    table_file = f"feedback_table_{i+1}.csv"
                    with open(table_file, 'w', newline='', encoding='utf-8') as csvfile:
                        writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
                        writer.writerows(rows)
                    
                    print(f"✅ Saved table {i+1} to: {table_file} ({len(rows)} rows)")
                    return True
        
        # Method 2: Try to find structured data in divs
        print("🔍 Looking for structured data in divs...")
        
        # Look for divs with data attributes or specific classes
        data_divs = soup.find_all('div', class_=lambda x: x and any(
            keyword in x.lower() for keyword in ['row', 'item', 'feedback', 'entry', 'record']
        ))
        
        if data_divs:
            print(f"✅ Found {len(data_divs)} data divs")
            
            rows = []
            for div in data_divs:
                row = []
                # Extract text from all child elements
                for element in div.find_all(['span', 'p', 'div', 'td', 'th']):
                    text = element.get_text(strip=True)
                    if text and text not in row:  # Avoid duplicates
                        text = text.replace('"', '""')
                        if ',' in text or '"' in text or '\n' in text or '\r' in text:
                            text = f'"{text}"'
                        row.append(text)
                
                if row:
                    rows.append(row)
            
            if rows:
                with open("feedback_divs.csv", 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
                    writer.writerows(rows)
                
                print(f"✅ Saved div data to: feedback_divs.csv ({len(rows)} rows)")
                return True
        
        # Method 3: Look for JSON data
        print("🔍 Looking for JSON data...")
        
        scripts = soup.find_all('script')
        for script in scripts:
            if script.string:
                content = script.string.strip()
                if content.startswith('{') or content.startswith('['):
                    try:
                        json_data = json.loads(content)
                        
                        # Convert JSON to CSV
                        if isinstance(json_data, list):
                            if json_data and isinstance(json_data[0], dict):
                                # List of objects
                                with open("feedback_json.csv", 'w', newline='', encoding='utf-8') as csvfile:
                                    fieldnames = json_data[0].keys()
                                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                                    writer.writeheader()
                                    writer.writerows(json_data)
                                
                                print(f"✅ Saved JSON data to: feedback_json.csv ({len(json_data)} records)")
                                return True
                        elif isinstance(json_data, dict):
                            # Single object or nested structure
                            with open("feedback_json.json", 'w', encoding='utf-8') as jsonfile:
                                json.dump(json_data, jsonfile, indent=2)
                            
                            print(f"✅ Saved JSON data to: feedback_json.json")
                            return True
                            
                    except json.JSONDecodeError:
                        continue
        
        # Method 4: Extract all text content
        print("🔍 Extracting all text content...")
        
        # Get all text content
        all_text = soup.get_text(separator='\n', strip=True)
        
        if all_text:
            # Split into lines and try to identify structured data
            lines = [line.strip() for line in all_text.split('\n') if line.strip()]
            
            # Look for patterns that might indicate structured data
            structured_lines = []
            for line in lines:
                # Look for lines that might contain key-value pairs or structured data
                if any(char in line for char in [':', '|', '\t']) or ',' in line:
                    structured_lines.append(line)
            
            if structured_lines:
                with open("feedback_text.csv", 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
                    for line in structured_lines:
                        # Try to split by common delimiters
                        if ':' in line:
                            parts = line.split(':', 1)
                        elif '|' in line:
                            parts = line.split('|')
                        elif '\t' in line:
                            parts = line.split('\t')
                        else:
                            parts = [line]
                        
                        # Clean and escape parts
                        clean_parts = []
                        for part in parts:
                            text = part.strip().replace('"', '""')
                            if ',' in text or '"' in text or '\n' in text or '\r' in text:
                                text = f'"{text}"'
                            clean_parts.append(text)
                        
                        writer.writerow(clean_parts)
                
                print(f"✅ Saved text data to: feedback_text.csv ({len(structured_lines)} lines)")
                return True
        
        print("❌ No structured data found using any method")
        return False
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error extracting data: {e}")
        return False

def main():
    """Main function"""
    url = "http://10.45.254.19:99/feedback"
    
    print("🚀 Enhanced Feedback CSV Extractor")
    print("=" * 50)
    
    # First, analyze the page structure
    analysis = analyze_page_structure(url)
    
    print("\n📊 Page Analysis Results:")
    print(f"  📄 Status Code: {analysis.get('status_code', 'Unknown')}")
    print(f"  📏 Content Length: {analysis.get('content_length', 'Unknown')} bytes")
    print(f"  📑 Title: {analysis.get('title', 'Unknown')}")
    print(f"  📊 Tables: {analysis.get('tables', 0)}")
    print(f"  📦 Divs: {analysis.get('divs', 0)}")
    print(f"  📝 Lists: {analysis.get('lists', 0)}")
    print(f"  📄 Paragraphs: {analysis.get('paragraphs', 0)}")
    print(f"  💻 Pre blocks: {analysis.get('pre_blocks', 0)}")
    print(f"  🔧 Data attributes: {analysis.get('data_attributes', 0)}")
    
    if analysis.get('json_scripts'):
        print(f"  📋 JSON scripts: {len(analysis['json_scripts'])}")
        for i, script in enumerate(analysis['json_scripts']):
            print(f"    Script {i+1}: {script['type']} ({script['length']} chars)")
    
    print("\n🔄 Attempting data extraction...")
    
    success = extract_feedback_data(url)
    
    if success:
        print("\n🎉 Feedback data extraction completed successfully!")
        print("\n💡 Check the generated files:")
        print("  - feedback_table_*.csv (if tables found)")
        print("  - feedback_divs.csv (if structured divs found)")
        print("  - feedback_json.csv (if JSON data found)")
        print("  - feedback_text.csv (if text patterns found)")
    else:
        print("\n❌ Feedback data extraction failed!")
        print("\n🔧 Alternative approaches:")
        print("  1. Use the JavaScript code directly in your browser:")
        print("     - Open the URL in your browser")
        print("     - Press F12 to open Developer Tools")
        print("     - Go to Console tab")
        print("     - Paste and run your JavaScript code")
        print("  2. Check if the page requires authentication")
        print("  3. Verify the page structure has changed")
        print("  4. Try accessing the page from a different network")
    
    return success

if __name__ == "__main__":
    main()


