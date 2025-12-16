#!/usr/bin/env python3
"""
Feedback CSV Extractor
=====================

This script extracts feedback data from a web page and converts it to CSV format.
Alternative to the JavaScript approach for extracting feedback data.
"""

import requests
import csv
import sys
from bs4 import BeautifulSoup
from typing import List, List

def extract_feedback_csv(url: str, output_file: str = "feedback_table.csv") -> bool:
    """
    Extract feedback data from a web page and save as CSV
    
    Args:
        url: URL of the feedback page
        output_file: Output CSV file name
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"🔍 Fetching data from: {url}")
        
        # Fetch the webpage
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        print(f"✅ Successfully fetched page (Status: {response.status_code})")
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the table
        table = soup.find('table')
        if not table:
            print("❌ No table found on the page")
            return False
        
        print("✅ Table found, extracting data...")
        
        # Extract rows
        rows = []
        for tr in table.find_all('tr'):
            row = []
            for cell in tr.find_all(['th', 'td']):
                # Clean and escape cell content
                text = cell.get_text(strip=True)
                # Escape quotes by doubling them (CSV standard)
                text = text.replace('"', '""')
                # Wrap in quotes if it contains comma, quote, or newline
                if ',' in text or '"' in text or '\n' in text or '\r' in text:
                    text = f'"{text}"'
                row.append(text)
            rows.append(row)
        
        if not rows:
            print("❌ No data rows found in table")
            return False
        
        print(f"✅ Extracted {len(rows)} rows of data")
        
        # Write to CSV
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
            writer.writerows(rows)
        
        print(f"✅ Successfully saved data to: {output_file}")
        print(f"📊 Rows written: {len(rows)}")
        
        # Show preview of first few rows
        if rows:
            print("\n📋 Preview of extracted data:")
            for i, row in enumerate(rows[:3]):  # Show first 3 rows
                print(f"  Row {i+1}: {', '.join(row)}")
            if len(rows) > 3:
                print(f"  ... and {len(rows) - 3} more rows")
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error extracting data: {e}")
        return False

def main():
    """Main function"""
    url = "http://10.45.254.19:99/feedback"
    output_file = "feedback_table.csv"
    
    print("🚀 Feedback CSV Extractor")
    print("=" * 50)
    
    success = extract_feedback_csv(url, output_file)
    
    if success:
        print("\n🎉 Feedback data extraction completed successfully!")
        print(f"📁 Output file: {output_file}")
        print("\n💡 You can now:")
        print("  - Open the CSV file in Excel, Google Sheets, or any spreadsheet application")
        print("  - Import the data into your analysis tools")
        print("  - Process the feedback data programmatically")
    else:
        print("\n❌ Feedback data extraction failed!")
        print("\n🔧 Troubleshooting tips:")
        print("  1. Check if the URL is accessible from your network")
        print("  2. Verify the page contains a table with feedback data")
        print("  3. Try using the JavaScript approach in your browser instead")
        print("  4. Check if authentication is required to access the page")
    
    return success

if __name__ == "__main__":
    main()


