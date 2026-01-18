import json
from pathlib import Path

import pytest
import requests

pytestmark = pytest.mark.integration


SCENARIO_DIR = Path(__file__).resolve().parents[1] / "datasets" / "botsv3" / "scenarios"
SCENARIO_FILES = [
    "login_anomaly.json",
    "c2_beacon.json",
    "exfil_s3.json",
]


def load_scenario(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_severity(value, default=50) -> int:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        mapping = {"critical": 90, "high": 75, "medium": 55, "low": 30, "info": 10}
        value = value.strip().lower()
        if value in mapping:
            return mapping[value]
        try:
            return int(float(value))
        except ValueError:
            return default
    return default


def test_botsv3_scenario_files_present():
    missing = [name for name in SCENARIO_FILES if not (SCENARIO_DIR / name).exists()]
    assert not missing, f"Missing scenario files: {missing}"


def test_botsv3_scenario_schema():
    for filename in SCENARIO_FILES:
        scenario = load_scenario(SCENARIO_DIR / filename)
        assert scenario.get("input"), f"{filename} missing input"
        assert scenario.get("intent"), f"{filename} missing intent"
        assert isinstance(scenario.get("metadata"), dict), f"{filename} metadata must be dict"
        assert isinstance(scenario.get("iocs"), list), f"{filename} iocs must be list"
        assert scenario.get("evidence_ref"), f"{filename} missing evidence_ref"


def test_botsv3_orchestrator_post_and_classification(orchestrator_base_url):
    scenario = load_scenario(SCENARIO_DIR / "login_anomaly.json")

    endpoint = f"{orchestrator_base_url}/api/v2/orchestrate"
    metadata = dict(scenario.get("metadata") or {})
    iocs = scenario.get("iocs") or []
    if iocs:
        metadata.setdefault("indicators", list(iocs))
        metadata.setdefault("iocs", list(iocs))
        metadata.setdefault("ioc_list", list(iocs))

    payload = {
        "input": scenario.get("input"),
        "event_type": scenario.get("intent"),
        "severity": normalize_severity(metadata.get("severity")),
        "source": metadata.get("source") or "botsv3",
        "affected_assets": [metadata.get("host")] if metadata.get("host") else [],
        "raw_log": scenario.get("input"),
        "data": {"message": scenario.get("input"), **metadata},
    }

    response = requests.post(endpoint, json=payload, timeout=10)
    response.raise_for_status()

    data = response.json()
    assert "threat_classification" in data, "Missing threat_classification in response"
    assert data["threat_classification"].get("threat_type") is not None
