#!/usr/bin/env python3
"""
Test BigQuery JSON field handling
"""

import json
from google.cloud import bigquery
from datetime import datetime

def to_jsonable(obj):
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (datetime,)):
        iso = obj.isoformat()
        return iso if iso.endswith('Z') else iso
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    return str(obj)

def test_bigquery_insert():
    client = bigquery.Client(project='chronicle-dev-2be9')
    
    # Test data
    test_data = {
        'alarm_id': 'test-123',
        'confidence': 0.85,
        'severity': 0.9,
        'valid': True,
        'reasoning': 'Test reasoning',
        'run_time': 1.5,
        'is_anomaly': True,
        'remarks': 'Test insert',
        'raw': {'test': 'data', 'nested': {'value': 123}}
    }
    
    print("Test data:")
    print(json.dumps(test_data, indent=2))
    
    # Try different approaches for the raw field
    approaches = [
        ("Python dict", test_data['raw']),
        ("JSON string", json.dumps(test_data['raw'])),
        ("to_jsonable dict", to_jsonable(test_data['raw'])),
        ("to_jsonable string", json.dumps(to_jsonable(test_data['raw'])))
    ]
    
    table_ref = client.dataset('gatra_database').table('taa_state')
    
    for approach_name, raw_value in approaches:
        print(f"\n--- Testing approach: {approach_name} ---")
        test_row = test_data.copy()
        test_row['raw'] = raw_value
        
        try:
            errors = client.insert_rows_json(
                table_ref, 
                [test_row],
                ignore_unknown_values=True,
                skip_invalid_rows=True
            )
            
            if errors:
                print(f"❌ Errors: {errors}")
            else:
                print("✅ Success!")
                break
                
        except Exception as e:
            print(f"❌ Exception: {e}")

if __name__ == "__main__":
    test_bigquery_insert()


