from gatra_super_agent import classify_threat


def test_login_anomaly_classification():
    event_data = {
        "message": "User login anomaly: impossible travel and new device. MFA failed.",
        "user": "userA",
        "mfa_failures": 2,
        "geo_from": "ID",
        "geo_to": "SG",
    }
    result = classify_threat(
        event_type="unauthorized_access",
        event_data=event_data,
        iocs=None,
        raw_log="impossible travel new device mfa failed",
    )
    assert result["threat_type"] == "login_anomaly"
    assert result["mitre_tactics"] == ["TA0006"]


def test_c2_classification():
    event_data = {
        "message": "C2 beacon detected with callback to malicious domain",
        "destination_domain": "malicious.example.com",
    }
    result = classify_threat(
        event_type="malware_detection",
        event_data=event_data,
        iocs=["malicious.example.com"],
        raw_log="command and control beaconing detected",
    )
    assert result["threat_type"] == "command_and_control"
    assert result["mitre_tactics"] == ["TA0011"]


def test_exfil_classification():
    event_data = {
        "message": "Unusual S3 upload by service account from CloudTrail",
        "bucket": "s3://sensitive-bucket",
        "service_account": "svc-data-export",
        "bytes_uploaded": 20000000,
    }
    result = classify_threat(
        event_type="data_exfiltration",
        event_data=event_data,
        iocs=[],
        raw_log="cloudtrail large transfer to s3 bucket",
    )
    assert result["threat_type"] == "data_exfiltration"
    assert result["mitre_tactics"] == ["TA0010"]


def test_all_zero_unauthorized_access_defaults_to_login():
    result = classify_threat(
        event_type="unauthorized_access",
        event_data={},
        iocs=[],
        raw_log="",
    )
    assert result["threat_type"] == "login_anomaly"


def test_all_zero_unknown_event_defaults_to_unknown():
    result = classify_threat(
        event_type="healthcheck",
        event_data={},
        iocs=None,
        raw_log="",
    )
    assert result["threat_type"] == "unknown_security_event"


def test_deterministic_output():
    event_data = {"message": "C2 beacon detected", "dest_ip": "10.0.0.5"}
    args = {
        "event_type": "malware_detection",
        "event_data": event_data,
        "iocs": ["10.0.0.5"],
        "raw_log": "c2 beacon detected",
    }
    first = classify_threat(**args)
    second = classify_threat(**args)
    assert first == second
