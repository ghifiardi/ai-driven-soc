                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            #!/usr/bin/env python3
"""
Sample TAA Tests - Comprehensive Security Scenarios
Test your Enhanced TAA with various real-world security scenarios
"""

import json
import time
from datetime import datetime
from enhanced_taa_flash25 import EnhancedTAA

def test_brute_force_attack():
    """Test TAA with brute force attack scenario"""
    
    print("🔴 TEST 1: Brute Force Attack")
    print("=" * 50)
    
    alert = {
        "alert_id": "brute-force-001",
        "src_ip": "203.0.113.45",
        "dst_ip": "192.168.1.100",
        "port": 22,
        "protocol": "SSH",
        "bytes_transferred": 5000000,
        "connection_count": 500,
        "timestamp": datetime.now().isoformat(),
        "description": "Multiple failed SSH login attempts from external IP",
        "context": {
            "attack_type": "brute_force",
            "failed_attempts": 500,
            "time_window": "5 minutes",
            "target_system": "production_server"
        }
    }
    
    print("📥 Alert Details:")
    print(f"   Source: {alert['src_ip']} → {alert['dst_ip']}")
    print(f"   Protocol: {alert['protocol']}:{alert['port']}")
    print(f"   Connections: {alert['connection_count']}")
    print(f"   Data: {alert['bytes_transferred']:,} bytes")
    print(f"   Context: {alert['context']}")
    
    return alert

def test_data_exfiltration():
    """Test TAA with data exfiltration scenario"""
    
    print("\n🔴 TEST 2: Data Exfiltration")
    print("=" * 50)
    
    alert = {
        "alert_id": "data-exfil-002",
        "src_ip": "192.168.1.50",
        "dst_ip": "suspicious-cloud-storage.com",
        "port": 443,
        "protocol": "HTTPS",
        "bytes_transferred": 100000000,  # 100MB
        "connection_count": 20,
        "timestamp": datetime.now().isoformat(),
        "description": "Large data transfer to external cloud storage",
        "context": {
            "attack_type": "data_exfiltration",
            "data_volume": "100MB",
            "destination": "external_cloud",
            "encryption": "https",
            "time_pattern": "sustained"
        }
    }
    
    print("📥 Alert Details:")
    print(f"   Source: {alert['src_ip']} → {alert['dst_ip']}")
    print(f"   Protocol: {alert['protocol']}:{alert['port']}")
    print(f"   Data: {alert['bytes_transferred']:,} bytes")
    print(f"   Context: {alert['context']}")
    
    return alert

def test_malware_c2():
    """Test TAA with malware command & control scenario"""
    
    print("\n🔴 TEST 3: Malware Command & Control")
    print("=" * 50)
    
    alert = {
        "alert_id": "malware-c2-003",
        "src_ip": "192.168.1.75",
        "dst_ip": "malicious-c2-server.net",
        "port": 443,
        "protocol": "HTTPS",
        "bytes_transferred": 2048,
        "connection_count": 1,
        "timestamp": datetime.now().isoformat(),
        "description": "Suspicious connection to known C2 server",
        "context": {
            "attack_type": "command_control",
            "threat_intelligence": "known_malicious",
            "communication_pattern": "beacon",
            "malware_family": "unknown",
            "ioc_confidence": 0.95
        }
    }
    
    print("📥 Alert Details:")
    print(f"   Source: {alert['src_ip']} → {alert['dst_ip']}")
    print(f"   Protocol: {alert['protocol']}:{alert['port']}")
    print(f"   Data: {alert['bytes_transferred']:,} bytes")
    print(f"   Context: {alert['context']}")
    
    return alert

def test_insider_threat():
    """Test TAA with insider threat scenario"""
    
    print("\n🔴 TEST 4: Insider Threat")
    print("=" * 50)
    
    alert = {
        "alert_id": "insider-threat-004",
        "src_ip": "192.168.1.25",
        "dst_ip": "competitor-website.com",
        "port": 443,
        "protocol": "HTTPS",
        "bytes_transferred": 50000000,  # 50MB
        "connection_count": 5,
        "timestamp": datetime.now().isoformat(),
        "description": "Employee accessing competitor website with large data transfer",
        "context": {
            "attack_type": "insider_threat",
            "user_id": "employee_12345",
            "department": "engineering",
            "access_level": "high",
            "data_sensitivity": "confidential"
        }
    }
    
    print("📥 Alert Details:")
    print(f"   Source: {alert['src_ip']} → {alert['dst_ip']}")
    print(f"   Protocol: {alert['protocol']}:{alert['port']}")
    print(f"   Data: {alert['bytes_transferred']:,} bytes")
    print(f"   Context: {alert['context']}")
    
    return alert

def test_network_scanning():
    """Test TAA with network scanning scenario"""
    
    print("\n🟡 TEST 5: Network Scanning")
    print("=" * 50)
    
    alert = {
        "alert_id": "network-scan-005",
        "src_ip": "192.168.1.30",
        "dst_ip": "192.168.1.0/24",
        "port": 80,
        "protocol": "HTTP",
        "bytes_transferred": 10000,
        "connection_count": 100,
        "timestamp": datetime.now().isoformat(),
        "description": "Multiple HTTP requests to internal network range",
        "context": {
            "attack_type": "network_scanning",
            "scan_type": "port_scan",
            "target_range": "internal_network",
            "stealth_level": "low",
            "scan_duration": "30 minutes"
        }
    }
    
    print("📥 Alert Details:")
    print(f"   Source: {alert['src_ip']} → {alert['dst_ip']}")
    print(f"   Protocol: {alert['protocol']}:{alert['port']}")
    print(f"   Connections: {alert['connection_count']}")
    print(f"   Context: {alert['context']}")
    
    return alert

def test_normal_traffic():
    """Test TAA with normal business traffic"""
    
    print("\n🟢 TEST 6: Normal Business Traffic")
    print("=" * 50)
    
    alert = {
        "alert_id": "normal-traffic-006",
        "src_ip": "192.168.1.10",
        "dst_ip": "192.168.1.1",
        "port": 80,
        "protocol": "HTTP",
        "bytes_transferred": 1500,
        "connection_count": 1,
        "timestamp": datetime.now().isoformat(),
        "description": "Single HTTP request to internal server",
        "context": {
            "attack_type": "normal_traffic",
            "business_justified": True,
            "expected_behavior": True,
            "user_activity": "normal",
            "time_of_day": "business_hours"
        }
    }
    
    print("📥 Alert Details:")
    print(f"   Source: {alert['src_ip']} → {alert['dst_ip']}")
    print(f"   Protocol: {alert['protocol']}:{alert['port']}")
    print(f"   Data: {alert['bytes_transferred']:,} bytes")
    print(f"   Context: {alert['context']}")
    
    return alert

def test_phishing_attempt():
    """Test TAA with phishing attempt scenario"""
    
    print("\n🔴 TEST 7: Phishing Attempt")
    print("=" * 50)
    
    alert = {
        "alert_id": "phishing-007",
        "src_ip": "192.168.1.80",
        "dst_ip": "fake-bank-website.com",
        "port": 443,
        "protocol": "HTTPS",
        "bytes_transferred": 50000,
        "connection_count": 3,
        "timestamp": datetime.now().isoformat(),
        "description": "Employee accessing suspicious banking website",
        "context": {
            "attack_type": "phishing",
            "target_domain": "fake-bank-website.com",
            "user_behavior": "suspicious",
            "threat_intelligence": "known_phishing",
            "ioc_confidence": 0.90
        }
    }
    
    print("📥 Alert Details:")
    print(f"   Source: {alert['src_ip']} → {alert['dst_ip']}")
    print(f"   Protocol: {alert['protocol']}:{alert['port']}")
    print(f"   Data: {alert['bytes_transferred']:,} bytes")
    print(f"   Context: {alert['context']}")
    
    return alert

def test_ddos_attack():
    """Test TAA with DDoS attack scenario"""
    
    print("\n🔴 TEST 8: DDoS Attack")
    print("=" * 50)
    
    alert = {
        "alert_id": "ddos-008",
        "src_ip": "10.0.0.0/8",
        "dst_ip": "192.168.1.100",
        "port": 80,
        "protocol": "HTTP",
        "bytes_transferred": 1000000000,  # 1GB
        "connection_count": 10000,
        "timestamp": datetime.now().isoformat(),
        "description": "Massive HTTP traffic from multiple sources",
        "context": {
            "attack_type": "ddos",
            "attack_volume": "high",
            "source_diversity": "multiple_ips",
            "target_vulnerability": "web_server",
            "attack_duration": "2 hours"
        }
    }
    
    print("📥 Alert Details:")
    print(f"   Source: {alert['src_ip']} → {alert['dst_ip']}")
    print(f"   Protocol: {alert['protocol']}:{alert['port']}")
    print(f"   Data: {alert['bytes_transferred']:,} bytes")
    print(f"   Connections: {alert['connection_count']:,}")
    print(f"   Context: {alert['context']}")
    
    return alert

def run_taa_analysis(alert, test_name):
    """Run TAA analysis on an alert"""
    
    print(f"\n⚡ TAA Flash 2.5 Analysis...")
    start_time = time.time()
    
    try:
        # Initialize TAA
        config = {
            "project_id": "chronicle-dev-2be9",
            "location": "us-central1"
        }
        taa = EnhancedTAA(config)
        
        # Analyze alert
        result = taa.analyze_alert(alert)
        processing_time = (time.time() - start_time) * 1000
        
        print(f"\n📊 ANALYSIS RESULTS:")
        print(f"   🤖 Model: {result.get('llm_model', 'Unknown')}")
        print(f"   ✅ True Positive: {result.get('is_true_positive', False)}")
        print(f"   🚨 Severity: {result.get('severity', 'Unknown').upper()}")
        print(f"   🎯 Confidence: {result.get('confidence', 0):.3f}")
        print(f"   ⚠️  Risk Score: {result.get('risk_score', 0):.3f}")
        print(f"   🔥 Urgency: {result.get('urgency', 'Unknown').upper()}")
        print(f"   🎯 Threat Category: {result.get('threat_category', 'Unknown').upper()}")
        print(f"   📡 Attack Vector: {result.get('attack_vector', 'Unknown').upper()}")
        print(f"   🕒 Processing Time: {processing_time:.1f}ms")
        
        print(f"\n🧠 AI REASONING:")
        reasoning = result.get('reasoning', 'No reasoning provided')
        print(f"   {reasoning}")
        
        print(f"\n💡 RECOMMENDED ACTIONS:")
        actions = result.get('recommended_actions', [])
        for action in actions:
            print(f"   • {action}")
        
        # Determine response
        severity = result.get('severity', 'unknown').lower()
        confidence = result.get('confidence', 0)
        
        if severity == 'high' and confidence > 0.8:
            response = "🚨 IMMEDIATE CONTAINMENT REQUIRED"
        elif severity == 'medium' and confidence > 0.6:
            response = "🟡 INVESTIGATE AND MONITOR"
        elif severity == 'low' or confidence < 0.5:
            response = "🟢 CONTINUE MONITORING"
        else:
            response = "⚪ MANUAL REVIEW RECOMMENDED"
        
        print(f"\n🎯 RESPONSE: {response}")
        
        return result
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return None

def main():
    """Run all sample tests"""
    
    print("🚀 Enhanced TAA Sample Tests")
    print("=" * 70)
    print(f"Test Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    print("\n🔍 This will test your Enhanced TAA with various security scenarios:")
    print("   • Brute Force Attacks")
    print("   • Data Exfiltration")
    print("   • Malware Command & Control")
    print("   • Insider Threats")
    print("   • Network Scanning")
    print("   • Normal Traffic")
    print("   • Phishing Attempts")
    print("   • DDoS Attacks")
    
    # Test scenarios
    test_scenarios = [
        (test_brute_force_attack, "Brute Force Attack"),
        (test_data_exfiltration, "Data Exfiltration"),
        (test_malware_c2, "Malware C2"),
        (test_insider_threat, "Insider Threat"),
        (test_network_scanning, "Network Scanning"),
        (test_normal_traffic, "Normal Traffic"),
        (test_phishing_attempt, "Phishing Attempt"),
        (test_ddos_attack, "DDoS Attack")
    ]
    
    results = []
    
    for test_func, test_name in test_scenarios:
        try:
            alert = test_func()
            result = run_taa_analysis(alert, test_name)
            if result:
                results.append({
                    "test_name": test_name,
                    "alert_id": alert["alert_id"],
                    "success": True,
                    "severity": result.get('severity', 'unknown'),
                    "confidence": result.get('confidence', 0),
                    "is_true_positive": result.get('is_true_positive', False)
                })
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
            results.append({
                "test_name": test_name,
                "success": False,
                "error": str(e)
            })
    
    # Summary
    print(f"\n📋 TEST SUMMARY")
    print("=" * 50)
    
    successful_tests = [r for r in results if r['success']]
    failed_tests = [r for r in results if not r['success']]
    
    print(f"✅ Successful Tests: {len(successful_tests)}/{len(results)}")
    print(f"❌ Failed Tests: {len(failed_tests)}")
    
    if successful_tests:
        print(f"\n🎯 THREAT CLASSIFICATION:")
        high_severity = [r for r in successful_tests if r.get('severity') == 'high']
        medium_severity = [r for r in successful_tests if r.get('severity') == 'medium']
        low_severity = [r for r in successful_tests if r.get('severity') == 'low']
        
        print(f"   🔴 High Severity: {len(high_severity)}")
        print(f"   🟡 Medium Severity: {len(medium_severity)}")
        print(f"   🟢 Low Severity: {len(low_severity)}")
        
        print(f"\n📊 CONFIDENCE ANALYSIS:")
        high_confidence = [r for r in successful_tests if r.get('confidence', 0) > 0.8]
        medium_confidence = [r for r in successful_tests if 0.5 <= r.get('confidence', 0) <= 0.8]
        low_confidence = [r for r in successful_tests if r.get('confidence', 0) < 0.5]
        
        print(f"   🎯 High Confidence (>0.8): {len(high_confidence)}")
        print(f"   🎯 Medium Confidence (0.5-0.8): {len(medium_confidence)}")
        print(f"   🎯 Low Confidence (<0.5): {len(low_confidence)}")
        
        print(f"\n✅ TRUE POSITIVE DETECTION:")
        true_positives = [r for r in successful_tests if r.get('is_true_positive', False)]
        print(f"   🎯 True Positives: {len(true_positives)}/{len(successful_tests)}")
        print(f"   🎯 Detection Rate: {len(true_positives)/len(successful_tests)*100:.1f}%")
    
    print(f"\n🎉 Sample tests completed!")
    print(f"Test Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

if __name__ == "__main__":
    main()
