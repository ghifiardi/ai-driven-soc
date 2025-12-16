#!/usr/bin/env python3
"""
Lightweight AbuseIPDB client for TAA enrichment.
- Reads API key from env var ABUSEIPDB_API_KEY
- Exposes query_abuseipdb(ip: str) -> dict with normalized fields
- Safe to import even if key is missing (returns {"provider": "abuseipdb", "enabled": False})
"""
import os
import time
import json
import logging
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger("TI.AbuseIPDB")
logger.setLevel(logging.INFO)

ABUSEIPDB_API_URL = "https://api.abuseipdb.com/api/v2/check"
DEFAULT_TIMEOUT = 6

# Simple in-memory cache to avoid repeated lookups during a run
_cache: Dict[str, Dict[str, Any]] = {}
_cache_ttl_seconds = 60 * 30  # 30 minutes


def _get_api_key() -> Optional[str]:
    return os.getenv("ABUSEIPDB_API_KEY")


def _normalize(resp_json: Dict[str, Any]) -> Dict[str, Any]:
    data = resp_json.get("data", {}) if isinstance(resp_json, dict) else {}
    return {
        "provider": "abuseipdb",
        "enabled": True,
        "ipAddress": data.get("ipAddress"),
        "isWhitelisted": data.get("isWhitelisted"),
        "abuseConfidenceScore": data.get("abuseConfidenceScore"),
        "countryCode": data.get("countryCode"),
        "usageType": data.get("usageType"),
        "isp": data.get("isp"),
        "domain": data.get("domain"),
        "totalReports": data.get("totalReports"),
        "lastReportedAt": data.get("lastReportedAt"),
        "reports": data.get("reports"),  # may be large; caller can trim
        "raw": data,
    }


def query_abuseipdb(ip: str, *, max_age_days: int = 90, timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """Query AbuseIPDB for a single IPv4/IPv6 address.
    Returns a normalized dict. On error, returns {"provider": "abuseipdb", "enabled": False, "error": str}.
    """
    api_key = _get_api_key()
    if not api_key:
        return {"provider": "abuseipdb", "enabled": False, "error": "missing_api_key"}

    # cache lookup
    entry = _cache.get(ip)
    now = time.time()
    if entry and (now - entry.get("_ts", 0)) < _cache_ttl_seconds:
        return entry["payload"]

    headers = {"Key": api_key, "Accept": "application/json"}
    params = {"ipAddress": ip, "maxAgeInDays": str(max_age_days)}

    try:
        r = requests.get(ABUSEIPDB_API_URL, headers=headers, params=params, timeout=timeout)
        if r.status_code == 429:
            return {"provider": "abuseipdb", "enabled": True, "rate_limited": True}
        r.raise_for_status()
        resp_json = r.json()
        payload = _normalize(resp_json)
        _cache[ip] = {"_ts": now, "payload": payload}
        return payload
    except requests.RequestException as e:
        logger.warning("AbuseIPDB request error for %s: %s", ip, e)
        return {"provider": "abuseipdb", "enabled": True, "error": str(e)}
    except Exception as e:  # pragma: no cover
        logger.exception("Unexpected error in query_abuseipdb")
        return {"provider": "abuseipdb", "enabled": True, "error": str(e)}


if __name__ == "__main__":
    import sys
    test_ip = sys.argv[1] if len(sys.argv) > 1 else "1.1.1.1"
    print(json.dumps(query_abuseipdb(test_ip), indent=2))
