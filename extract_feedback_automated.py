#!/usr/bin/env python3
"""
Automated Feedback CSV Extraction
================================

This script attempts to extract CSV data from the feedback dashboard
using various methods including HTTP requests and browser automation.
"""

import requests
import json
import time
from datetime import datetime
import csv
import os
from typing import Dict, List, Any, Optional
import subprocess
import sys

def check_dashboard_accessibility(url: str) -> Dict[str, Any]:
    """
    Check if the feedback dashboard is accessible
    
    Args:
        url: Dashboard URL
        
    Returns:
        Dict with accessibility status
    """
    try:
        print(f"🔍 Checking dashboard accessibility: {url}")
        
        response = requests.get(url, timeout=30)
        
        return {
            "accessible": True,
            "status_code": response.status_code,
            "content_length": len(response.content),
            "content_type": response.headers.get("content-type", ""),
            "response_time": response.elapsed.total_seconds(),
            "error": None
        }
        
    except requests.exceptions.Timeout:
        return {
            "accessible": False,
            "status_code": None,
            "content_length": 0,
            "content_type": "",
            "response_time": 30,
            "error": "Request timeout"
        }
    except requests.exceptions.ConnectionError:
        return {
            "accessible": False,
            "status_code": None,
            "content_length": 0,
            "content_type": "",
            "response_time": None,
            "error": "Connection error"
        }
    except Exception as e:
        return {
            "accessible": False,
            "status_code": None,
            "content_length": 0,
            "content_type": "",
            "response_time": None,
            "error": str(e)
        }

def extract_data_with_requests(url: str) -> Optional[List[List[str]]]:
    """
    Attempt to extract data using HTTP requests
    
    Args:
        url: Dashboard URL
        
    Returns:
        List of rows with data, or None if extraction fails
    """
    try:
        print("🌐 Attempting data extraction with HTTP requests...")
        
        # Try different headers to mimic a browser
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code != 200:
            print(f"❌ HTTP request failed with status: {response.status_code}")
            return None
        
        # Check if it's a Streamlit app
        content = response.text
        if "streamlit" in content.lower():
            print("🎯 Detected Streamlit application")
            return extract_streamlit_data(content)
        else:
            print("📄 Non-Streamlit application detected")
            return extract_generic_data(content)
            
    except Exception as e:
        print(f"❌ Error during HTTP extraction: {e}")
        return None

def extract_streamlit_data(content: str) -> Optional[List[List[str]]]:
    """
    Extract data from Streamlit application content
    
    Args:
        content: HTML content
        
    Returns:
        List of rows with data, or None if extraction fails
    """
    try:
        from bs4 import BeautifulSoup
        
        print("🔍 Parsing Streamlit content...")
        soup = BeautifulSoup(content, 'html.parser')
        
        # Look for Streamlit tables
        tables = soup.find_all("table")
        if tables:
            print(f"📊 Found {len(tables)} tables")
            return extract_table_data(tables[0])  # Use first table
        
        # Look for Streamlit data containers
        data_containers = soup.find_all("div", {"data-testid": "stTable"})
        if data_containers:
            print(f"📦 Found {len(data_containers)} data containers")
            return extract_container_data(data_containers[0])
        
        # Look for other data structures
        print("🔍 Looking for alternative data structures...")
        return extract_alternative_data(soup)
        
    except ImportError:
        print("❌ BeautifulSoup not available. Install with: pip install beautifulsoup4")
        return None
    except Exception as e:
        print(f"❌ Error parsing Streamlit content: {e}")
        return None

def extract_table_data(table) -> List[List[str]]:
    """
    Extract data from HTML table
    
    Args:
        table: BeautifulSoup table element
        
    Returns:
        List of rows with data
    """
    rows = []
    tr_elements = table.find_all("tr")
    
    for tr in tr_elements:
        cells = tr.find_all(["th", "td"])
        row_data = [cell.get_text(strip=True) for cell in cells]
        if row_data:  # Only add non-empty rows
            rows.append(row_data)
    
    print(f"📋 Extracted {len(rows)} rows from table")
    return rows

def extract_container_data(container) -> List[List[str]]:
    """
    Extract data from Streamlit data container
    
    Args:
        container: BeautifulSoup container element
        
    Returns:
        List of rows with data
    """
    rows = []
    
    # Look for rows in the container
    row_elements = container.find_all("tr") or container.find_all("div", {"role": "row"})
    
    for row in row_elements:
        cells = row.find_all(["th", "td"]) or row.find_all("div", {"role": "cell"})
        row_data = [cell.get_text(strip=True) for cell in cells]
        if row_data:
            rows.append(row_data)
    
    print(f"📦 Extracted {len(rows)} rows from container")
    return rows

def extract_alternative_data(soup) -> Optional[List[List[str]]]:
    """
    Extract data from alternative HTML structures
    
    Args:
        soup: BeautifulSoup object
        
    Returns:
        List of rows with data, or None if extraction fails
    """
    print("🔍 Looking for alternative data structures...")
    
    # Look for JSON data in script tags
    script_tags = soup.find_all("script")
    for script in script_tags:
        if script.string and "data" in script.string.lower():
            try:
                # Try to extract JSON data
                script_content = script.string
                if "{" in script_content and "}" in script_content:
                    print("📄 Found potential JSON data in script tag")
                    # This would need more sophisticated parsing
                    return None
            except:
                continue
    
    # Look for data in divs with specific classes
    data_divs = soup.find_all("div", class_=lambda x: x and any(
        keyword in x.lower() for keyword in ["data", "table", "row", "cell"]
    ))
    
    if data_divs:
        print(f"📦 Found {len(data_divs)} potential data containers")
        # This would need more sophisticated parsing
    
    return None

def extract_generic_data(content: str) -> Optional[List[List[str]]]:
    """
    Extract data from generic HTML content
    
    Args:
        content: HTML content
        
    Returns:
        List of rows with data, or None if extraction fails
    """
    try:
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(content, 'html.parser')
        tables = soup.find_all("table")
        
        if tables:
            print(f"📊 Found {len(tables)} tables in generic content")
            return extract_table_data(tables[0])
        
        return None
        
    except ImportError:
        print("❌ BeautifulSoup not available")
        return None
    except Exception as e:
        print(f"❌ Error parsing generic content: {e}")
        return None

def save_to_csv(data: List[List[str]], filename: str) -> bool:
    """
    Save extracted data to CSV file
    
    Args:
        data: List of rows with data
        filename: Output filename
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"💾 Saving data to {filename}...")
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(data)
        
        print(f"✅ Successfully saved {len(data)} rows to {filename}")
        return True
        
    except Exception as e:
        print(f"❌ Error saving CSV: {e}")
        return False

def try_browser_automation(url: str) -> bool:
    """
    Try to use browser automation to extract data
    
    Args:
        url: Dashboard URL
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print("🤖 Attempting browser automation...")
        
        # Check if selenium is available
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
        except ImportError:
            print("❌ Selenium not available. Install with: pip install selenium")
            return False
        
        # Set up Chrome options
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        driver = webdriver.Chrome(options=chrome_options)
        
        try:
            print(f"🌐 Navigating to {url}...")
            driver.get(url)
            
            # Wait for page to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Look for tables
            tables = driver.find_elements(By.TAG_NAME, "table")
            if tables:
                print(f"📊 Found {len(tables)} tables with Selenium")
                
                # Extract data from first table
                rows = tables[0].find_elements(By.TAG_NAME, "tr")
                data = []
                
                for row in rows:
                    cells = row.find_elements(By.TAG_NAME, "td") + row.find_elements(By.TAG_NAME, "th")
                    row_data = [cell.text.strip() for cell in cells]
                    if row_data:
                        data.append(row_data)
                
                if data:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"feedback_data_selenium_{timestamp}.csv"
                    return save_to_csv(data, filename)
            
            print("❌ No tables found with Selenium")
            return False
            
        finally:
            driver.quit()
            
    except Exception as e:
        print(f"❌ Error with browser automation: {e}")
        return False

def main():
    """Main function"""
    url = "http://10.45.254.19:99/feedback"
    
    print("🚀 Automated Feedback CSV Extraction")
    print("=" * 50)
    print(f"🎯 Target URL: {url}")
    print(f"⏰ Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Step 1: Check accessibility
    print("📡 Step 1: Checking Dashboard Accessibility")
    print("-" * 40)
    accessibility = check_dashboard_accessibility(url)
    
    if not accessibility["accessible"]:
        print(f"❌ Dashboard not accessible: {accessibility['error']}")
        print("💡 Try the browser-based JavaScript extraction method instead")
        return False
    
    print(f"✅ Dashboard accessible (Status: {accessibility['status_code']})")
    print(f"📊 Response time: {accessibility['response_time']:.2f}s")
    print(f"📄 Content length: {accessibility['content_length']} bytes")
    print()
    
    # Step 2: Try HTTP extraction
    print("🌐 Step 2: HTTP Data Extraction")
    print("-" * 40)
    data = extract_data_with_requests(url)
    
    if data:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"feedback_data_http_{timestamp}.csv"
        
        if save_to_csv(data, filename):
            print(f"✅ Successfully extracted data via HTTP")
            return True
    
    # Step 3: Try browser automation
    print("\n🤖 Step 3: Browser Automation")
    print("-" * 40)
    if try_browser_automation(url):
        print("✅ Successfully extracted data via browser automation")
        return True
    
    # Step 4: Fallback to JavaScript method
    print("\n💡 Step 4: Fallback Recommendation")
    print("-" * 40)
    print("❌ Automated extraction methods failed")
    print("🎯 Use the browser-based JavaScript method:")
    print()
    print("1. Open your browser and go to: http://10.45.254.19:99/feedback")
    print("2. Wait for the page to fully load")
    print("3. Open Developer Console (F12)")
    print("4. Copy and paste the JavaScript code from extract_feedback_csv_browser.js")
    print("5. Press Enter to run the script")
    print("6. The CSV file will be automatically downloaded")
    
    return False

if __name__ == "__main__":
    success = main()
    
    if not success:
        print("\n📋 Alternative Methods:")
        print("-" * 30)
        print("1. 🌐 Browser JavaScript: Use extract_feedback_csv_browser.js")
        print("2. 🔧 Manual extraction: Contact dashboard administrator")
        print("3. 📊 API access: Check if there's a REST API endpoint")
        print("4. 🗄️  Direct database: Access the underlying data source")
    
    sys.exit(0 if success else 1)


