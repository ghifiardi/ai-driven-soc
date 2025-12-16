#!/usr/bin/env python3
"""
Fix for dashboard NameError: name 'filtered_alerts' is not defined
This script creates a simple fix for the feedback.py error
"""

# Common fix for the filtered_alerts error
dashboard_fix_code = '''
# Add this at the top of your feedback.py file, after imports
import pandas as pd

# Initialize filtered_alerts with empty DataFrame if not defined
if 'filtered_alerts' not in locals() and 'filtered_alerts' not in globals():
    filtered_alerts = pd.DataFrame()

# Or add this check before using filtered_alerts
if 'filtered_alerts' in locals() or 'filtered_alerts' in globals():
    if not filtered_alerts.empty:
        # Your existing code here
        pass
else:
    # Initialize with empty DataFrame
    filtered_alerts = pd.DataFrame()
'''

print("Dashboard Fix Code:")
print("=" * 50)
print(dashboard_fix_code)
print("=" * 50)

print("\nTo fix the dashboard error:")
print("1. SSH into your VM: ssh app@10.45.254.19")
print("2. Find the feedback.py file (likely in /app/pages/feedback.py)")
print("3. Add the initialization code above")
print("4. Restart the dashboard service")


