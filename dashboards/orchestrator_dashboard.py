import json
import hashlib
import os
import socket
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests
import streamlit as st
import streamlit.components.v1 as components

WIN_EVT_NS = {"ev": "http://schemas.microsoft.com/win/2004/08/events/event"}


def _safe_int(v, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def is_splunk_export_row(obj: Any) -> bool:
    # Splunk JSON export format: {"preview":..., "offset":..., "result": {...}}
    return isinstance(obj, dict) and "result" in obj and isinstance(obj["result"], dict) and "_raw" in obj["result"]


def parse_windows_eventdata_from_xml(raw_xml: str) -> Dict[str, str]:
    """
    Extracts <EventData><Data Name='...'>value</Data></EventData> into dict.
    Safe: returns {} on parse failure.
    """
    out: Dict[str, str] = {}
    try:
        root = ET.fromstring(raw_xml)
        event_data = root.find("ev:EventData", WIN_EVT_NS)
        if event_data is None:
            return out
        for data in event_data.findall("ev:Data", WIN_EVT_NS):
            name = data.attrib.get("Name")
            if name:
                out[name] = (data.text or "").strip()
    except Exception:
        return out
    return out


def infer_severity_4625(user: str, logon_type: int, status: str, sub_status: str) -> int:
    # Deterministic & simple; tune later.
    sev = 50
    if user.lower() in {"admin", "administrator"}:
        sev += 10
    if logon_type == 10:  # RDP / RemoteInteractive often
        sev += 10
    if sub_status.lower() == "0xc0000064":  # user does not exist (enum hint)
        sev += 5
    return max(10, min(100, sev))


def map_splunk_windows_security_to_orchestrator(splunk_row: Dict[str, Any]) -> Dict[str, Any]:
    r = splunk_row["result"]
    raw_xml = r.get("_raw", "") or ""
    ed = parse_windows_eventdata_from_xml(raw_xml)

    event_code = _safe_int(r.get("EventCode") or r.get("EventCode".lower()) or r.get("EventCode"), 0)
    # Many Splunk exports store EventCode as string "4625"
    if not event_code:
        event_code = _safe_int(r.get("EventCode") or r.get("EventCode".lower()) or r.get("EventCode"), 0)

    # Best-effort extraction
    time_str = r.get("_time")
    host = r.get("host") or r.get("dest") or "unknown-host"
    dest = r.get("dest") or host
    user = r.get("user") or ed.get("TargetUserName") or "unknown-user"
    src_ip = r.get("src_ip") or ed.get("IpAddress") or ""
    source = r.get("source") or "splunk"

    # For Windows Security Event 4625: treat as unauthorized_access
    event_type = "unauthorized_access"

    # Extra useful fields from XML
    logon_type = _safe_int(ed.get("LogonType"), 0)
    status = ed.get("Status", "")
    sub_status = ed.get("SubStatus", "")
    target_domain = ed.get("TargetDomainName", "")
    process_name = ed.get("ProcessName", "")

    # Deterministic event_id
    record_id = None
    try:
        root = ET.fromstring(raw_xml)
        rid_el = root.find("ev:System/ev:EventRecordID", WIN_EVT_NS)
        record_id = rid_el.text.strip() if rid_el is not None and rid_el.text else None
    except Exception:
        record_id = None

    if record_id:
        event_id = f"evt-4625-{record_id}"
    else:
        h = hashlib.sha1((str(time_str) + host + user + src_ip + raw_xml[:200]).encode("utf-8")).hexdigest()[:10]
        event_id = f"evt-4625-{h}"

    severity = infer_severity_4625(user=user, logon_type=logon_type, status=status, sub_status=sub_status)

    affected_assets = [host, f"user:{user}"]
    if src_ip:
        affected_assets.append(f"src_ip:{src_ip}")

    tags = ["windows_security", "failed_logon"]
    if logon_type == 10:
        tags.append("rdp_possible")

    data = {
        "time": time_str,
        "host": host,
        "dest": dest,
        "user": user,
        "src_ip": src_ip,
        "action": r.get("action"),
        "event_code": event_code or 4625,
        "logon_type": logon_type,
        "target_domain": target_domain,
        "status": status,
        "sub_status": sub_status,
        "process_name": process_name,
        "indicators": [x for x in [src_ip] if x],
        "tags": tags,
        "splunk": {
            "index": r.get("index"),
            "sourcetype": r.get("sourcetype"),
            "splunk_server": r.get("splunk_server"),
        },
        # keep raw splunk fields (except _raw) so you can evolve mapping later
        "splunk_fields": {k: v for k, v in r.items() if k != "_raw"},
        # XML EventData extracted (handy for enrichment)
        "eventdata": ed,
    }

    return {
        "event_id": event_id,
        "event_type": event_type,
        "severity": severity,
        "source": source,
        "affected_assets": affected_assets,
        "data": data,
        "raw_log": raw_xml,
    }


def try_parse_json(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except Exception:
        return None


def parse_csv_list(value: str) -> list:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def normalize_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def normalize_text(value) -> str:
    return "" if value is None else str(value)


def safe_get(data: dict, path: list, default=None):
    current = data
    for part in path:
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def dict_diff(a: dict, b: dict, prefix: str = "") -> list:
    """Small diff: returns list of (key, a_val, b_val) for changes."""
    changes = []
    keys = set((a or {}).keys()) | set((b or {}).keys())
    for key in sorted(keys):
        left = (a or {}).get(key, "__MISSING__")
        right = (b or {}).get(key, "__MISSING__")
        full_key = f"{prefix}{key}"
        if isinstance(left, dict) and isinstance(right, dict):
            changes.extend(dict_diff(left, right, prefix=full_key + "."))
        elif left != right:
            changes.append((full_key, left, right))
    return changes


def has_overrides(payload: dict) -> bool:
    ov = (payload or {}).get("data", {}).get("overrides", {})
    if not isinstance(ov, dict):
        return False
    return any(value is not None and value != "" and value != [] and value != {} for value in ov.values())


def detect_blocked_override_attempts(meta: dict, blocked_fields: list) -> list:
    if not isinstance(meta, dict):
        return []
    attempts = []
    for field in blocked_fields:
        if field in meta:
            attempts.append(field)
            meta.pop(field, None)
    return attempts


def compute_evidence_hash(raw_log: str, original: dict) -> str:
    payload = {
        "raw_log": normalize_text(raw_log),
        "original": original or {},
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_compact(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def get_session_id() -> str:
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    return st.session_state["session_id"]


def trigger_rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    st.experimental_rerun()


def render_badges(badges: list) -> None:
    if not badges:
        return
    badge_html = " ".join(
        [
            "<span style='padding:2px 8px; border-radius:999px; "
            "background:#1f2937; color:#f9fafb; font-size:0.75rem; "
            "margin-right:6px; display:inline-block;'>"
            f"{badge}</span>"
            for badge in badges
        ]
    )
    st.markdown(badge_html, unsafe_allow_html=True)


def render_copy_button(label: str, text: str, key: str) -> None:
    safe_text = json.dumps(text)
    button_id = f"copy-{key}"
    html = f"""
    <button id="{button_id}" style="padding:4px 10px; font-size:0.8rem;">
        {label}
    </button>
    <script>
    const btn = document.getElementById("{button_id}");
    if (btn) {{
        btn.addEventListener("click", () => {{
            navigator.clipboard.writeText({safe_text});
        }});
    }}
    </script>
    """
    components.html(html, height=36)


def get_incidents() -> list:
    if "incidents" not in st.session_state:
        st.session_state["incidents"] = []
    return st.session_state["incidents"]


def get_policy_blocks() -> list:
    if "policy_blocks" not in st.session_state:
        st.session_state["policy_blocks"] = []
    return st.session_state["policy_blocks"]


def get_watchlist() -> dict:
    if "watchlist" not in st.session_state:
        st.session_state["watchlist"] = {"assets": [], "users": [], "iocs": []}
    return st.session_state["watchlist"]


def add_incident(record: dict) -> None:
    incidents = get_incidents()
    incident_id = record.get("incident_id")
    for idx, row in enumerate(incidents):
        if row.get("incident_id") == incident_id:
            merged = {**row, **record}
            for field in ["status", "owner", "team", "tags", "ticket_id"]:
                if row.get(field) and not record.get(field):
                    merged[field] = row.get(field)
            incidents[idx] = merged
            return
    incidents.insert(0, record)


def compute_kpis(incidents: list, policy_blocks: list) -> dict:
    now = datetime.now(timezone.utc)
    open_incidents = len([row for row in incidents if row.get("status") != "closed"])
    critical_today = 0
    confidence_values = []
    for row in incidents:
        if row.get("risk_level") == "CRITICAL" and row.get("created_at"):
            try:
                created_dt = datetime.fromisoformat(row["created_at"])
                if created_dt.date() == now.date():
                    critical_today += 1
            except Exception:
                pass
        if isinstance(row.get("confidence"), (int, float)):
            confidence_values.append(row["confidence"])
    avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0
    policy_or_tamper = len([row for row in incidents if row.get("tamper_attempt")]) + len(policy_blocks)
    return {
        "open_incidents": open_incidents,
        "critical_today": critical_today,
        "avg_confidence": avg_confidence,
        "policy_or_tamper": policy_or_tamper,
    }


def extract_iocs(payload: dict) -> list:
    data_payload = (payload or {}).get("data", {}) or {}
    effective = data_payload.get("effective") or {}
    iocs = effective.get("iocs") or data_payload.get("iocs") or data_payload.get("indicators") or []
    return normalize_list(iocs)


def build_splunk_query(iocs: list) -> str:
    if not iocs:
        return "search index=* | head 50"
    tokens = [f'\"{ioc}\"' for ioc in iocs]
    or_block = " OR ".join(tokens)
    return f"search index=* ({or_block}) | head 100"


def build_incident_summary(resp: dict) -> dict:
    unified = resp.get("unified_insight") or {}
    tc = resp.get("threat_classification") or {}
    reason = tc.get("reason") or (tc.get("classification_reason") or {}).get("summary") or ""
    summary = unified.get("summary") or f"Detected {tc.get('threat_type', 'event')}."
    recs = resp.get("recommendations") or []
    next_step = recs[0] if recs else "Review incident details and confirm scope."
    return {
        "what": summary,
        "why": reason,
        "next": next_step,
    }


def build_timeline_rows(payload: dict, evidence_rows: list, resp: Optional[dict] = None) -> list:
    rows = []
    for row in evidence_rows or []:
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "time": row.get("time") or row.get("_time") or row.get("timestamp") or "",
                "source": row.get("source") or row.get("sourcetype") or "",
                "entity": row.get("user") or row.get("host") or row.get("src_ip") or "",
                "summary": row.get("message") or row.get("event") or row.get("action") or "",
            }
        )
    if rows:
        return sorted(rows, key=lambda item: item.get("time") or "")
    data_payload = (payload or {}).get("data", {}) or {}
    original = data_payload.get("original") or {}
    evidence = original.get("evidence") or {}
    base_time = original.get("time") or original.get("_time") or utc_now_iso()
    rows.append(
        {
            "time": base_time,
            "source": original.get("source") or original.get("source_label") or "",
            "entity": evidence.get("user") or evidence.get("host") or evidence.get("src_ip") or "",
            "summary": data_payload.get("message") or "Event received",
        }
    )
    if resp:
        agent_results = resp.get("agent_results") or []
        for agent in agent_results:
            domain = agent.get("domain") or "agent"
            rows.append(
                {
                    "time": base_time,
                    "source": domain,
                    "entity": agent.get("agent_id") or domain,
                    "summary": f"{domain} finished",
                }
            )
    return rows


def collect_entities(payload: dict, evidence_rows: list) -> dict:
    entities = {"users": set(), "hosts": set(), "ips": set(), "domains": set()}
    data_payload = (payload or {}).get("data", {}) or {}
    evidence = (data_payload.get("original") or {}).get("evidence") or {}
    for key in ["user", "username"]:
        if evidence.get(key):
            entities["users"].add(str(evidence.get(key)))
    for key in ["host", "hostname", "destination", "bucket"]:
        if evidence.get(key):
            entities["hosts"].add(str(evidence.get(key)))
    for key in ["src_ip", "destination_ip", "dest_ip"]:
        if evidence.get(key):
            entities["ips"].add(str(evidence.get(key)))
    for key in ["destination_domain", "domain", "url"]:
        if evidence.get(key):
            entities["domains"].add(str(evidence.get(key)))
    for row in evidence_rows or []:
        if isinstance(row, dict):
            for key, bucket in [("user", "users"), ("host", "hosts"), ("src_ip", "ips"), ("domain", "domains")]:
                if row.get(key):
                    entities[bucket].add(str(row.get(key)))
    return {key: sorted(values) for key, values in entities.items()}


def compute_triage_seconds(resp: dict) -> int:
    results = resp.get("agent_results") or []
    times = []
    for result in results:
        value = result.get("execution_time")
        if isinstance(value, (int, float)):
            times.append(value)
    if not times:
        return 0
    return int(sum(times) / len(times))


def compute_avg_agent_confidence(resp: dict) -> Optional[float]:
    results = resp.get("agent_results") or []
    values = []
    for result in results:
        value = result.get("confidence")
        if isinstance(value, (int, float)):
            values.append(value)
    if not values:
        return None
    return sum(values) / len(values)


def build_incident_record(resp: dict, payload: dict) -> dict:
    intent = resp.get("intent") or {}
    risk = resp.get("risk_assessment") or {}
    tc = resp.get("threat_classification") or {}
    unified = resp.get("unified_insight") or {}
    data_payload = (payload or {}).get("data", {}) or {}
    effective = data_payload.get("effective") or {}
    created_at = resp.get("timestamp") or utc_now_iso()
    confidence_value = unified.get("confidence_score")
    if not isinstance(confidence_value, (int, float)):
        confidence_value = compute_avg_agent_confidence(resp)
    assets = payload.get("affected_assets") or resp.get("affected_assets") or []
    tags = normalize_list(effective.get("tags") or data_payload.get("tags") or [])
    return {
        "incident_id": resp.get("orchestration_id") or resp.get("orchestrationId") or resp.get("id") or str(uuid.uuid4()),
        "event_id": resp.get("event_id") or payload.get("event_id") or "-",
        "created_at": created_at,
        "event_type": intent.get("event_type") or payload.get("event_type") or "unknown",
        "risk_level": risk.get("level") or "UNKNOWN",
        "risk_score": risk.get("final_score"),
        "severity_effective": payload.get("severity_effective"),
        "severity_original": payload.get("severity_original"),
        "confidence": confidence_value,
        "threat_type": tc.get("threat_type") or "unknown",
        "assets": assets,
        "owner": effective.get("owner") or data_payload.get("owner"),
        "team": effective.get("team") or data_payload.get("team"),
        "status": "new",
        "tags": tags,
        "evidence_hash": payload.get("evidence_hash"),
        "tamper_attempt": bool(payload.get("tamper_attempt")),
        "ticket_id": effective.get("ticket_id") or data_payload.get("ticket_id"),
        "raw_api_response": resp,
        "payload_snapshot": payload,
        "summary": unified.get("summary") or "",
    }


def format_age(iso_time: str) -> str:
    try:
        cleaned = iso_time.replace("Z", "+00:00") if isinstance(iso_time, str) else iso_time
        created_dt = datetime.fromisoformat(cleaned)
        if created_dt.tzinfo is None:
            created_dt = created_dt.replace(tzinfo=timezone.utc)
    except Exception:
        return "-"
    delta = datetime.now(timezone.utc) - created_dt
    minutes = int(delta.total_seconds() // 60)
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h"
    days = hours // 24
    return f"{days}d"


def select_active_incident(incidents: list) -> Optional[dict]:
    selected_id = st.session_state.get("selected_incident_id")
    if selected_id:
        for row in incidents:
            if row.get("incident_id") == selected_id:
                return row
    return incidents[0] if incidents else None


def incident_search_blob(incident: dict) -> str:
    pieces = []
    for key in ["event_type", "threat_type", "risk_level", "owner", "team", "summary", "event_id"]:
        value = incident.get(key)
        if value:
            pieces.append(str(value))
    for asset in incident.get("assets") or []:
        pieces.append(str(asset))
    for tag in incident.get("tags") or []:
        pieces.append(str(tag))
    payload = incident.get("payload_snapshot") or {}
    effective = (payload.get("data") or {}).get("effective") or {}
    for key in ["event_type", "owner", "team", "case_id", "ticket_id", "notes"]:
        value = effective.get(key)
        if value:
            pieces.append(str(value))
    for ioc in normalize_list(effective.get("iocs") or []):
        pieces.append(str(ioc))
    return " ".join(pieces).lower()


def filter_incidents(incidents: list, status_filter: list, risk_filter: list, type_filter: list, search_text: str) -> list:
    filtered = []
    search_lower = (search_text or "").lower().strip()
    for row in incidents:
        if status_filter and row.get("status") not in status_filter:
            continue
        if risk_filter and row.get("risk_level") not in risk_filter:
            continue
        if type_filter and row.get("event_type") not in type_filter:
            continue
        if search_lower:
            blob = incident_search_blob(row)
            if search_lower not in blob:
                continue
        filtered.append(row)
    return filtered


def incident_flags(incident: dict) -> str:
    flags = []
    if incident.get("evidence_hash"):
        flags.append("Evidence Locked")
    payload = incident.get("payload_snapshot") or {}
    overrides = (payload.get("data") or {}).get("overrides") or {}
    if overrides:
        flags.append("Override")
    if incident.get("tamper_attempt"):
        flags.append("Tamper")
    return ", ".join(flags) if flags else "-"


def assess_action_risk(action: str) -> dict:
    text = (action or "").lower()
    if any(word in text for word in ["isolate", "block", "disable", "revoke", "lock", "quarantine"]):
        return {"risk": "High", "rollback": "Restore access and remove blocks after validation."}
    if any(word in text for word in ["reset", "force", "rotate", "patch"]):
        return {"risk": "Medium", "rollback": "Revert change with approval if impact is confirmed."}
    return {"risk": "Low", "rollback": "Document action and monitor for impact."}


ORCH_URL = os.getenv("ORCH_URL", "http://localhost:8080")
ORCH_ENDPOINT = f"{ORCH_URL.rstrip('/')}/api/v2/orchestrate"
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BOTSV3_DIR = os.path.join(ROOT_DIR, "datasets", "botsv3")
SCENARIO_DIR = os.path.join(BOTSV3_DIR, "scenarios")
EVIDENCE_DIR = os.path.join(BOTSV3_DIR, "evidence")
UI_APP = "orchestrator_dashboard.py"
UI_VERSION = "v1"
POLICY_ID = "override_policy_regulated_v1"
POLICY_VERSION = "v1"

st.set_page_config(page_title="GATRA - Orchestrator Dashboard", layout="wide")
st.title("GATRA Orchestrator Dashboard (Streamlit)")

# ---------- Helpers ----------
def post_orchestrate(payload: dict) -> dict:
    r = requests.post(ORCH_ENDPOINT, json=payload, timeout=20)
    r.raise_for_status()
    return r.json()


def load_scenarios() -> dict:
    scenarios = {}
    if os.path.isdir(SCENARIO_DIR):
        for filename in sorted(os.listdir(SCENARIO_DIR)):
            if not filename.endswith(".json"):
                continue
            path = os.path.join(SCENARIO_DIR, filename)
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            name = data.get("name") or data.get("id") or filename.replace(".json", "")
            scenarios[name] = data
    return scenarios


def load_evidence(path: str) -> list:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def derive_assets(meta: dict) -> list:
    assets = []
    for key in ["affected_assets", "host", "hostname", "destination", "bucket"]:
        value = meta.get(key)
        if not value:
            continue
        if isinstance(value, list):
            assets.extend([str(item) for item in value])
        else:
            assets.append(str(value))
    return assets


def render_threat_classification(resp: dict):
    tc = resp.get("threat_classification") or {}
    if not tc:
        st.info("No threat_classification in response.")
        return

    ttype = tc.get("threat_type", "unknown")
    severity = tc.get("severity", "-")
    tactics = tc.get("mitre_tactics") or []
    sigs = tc.get("matched_signatures") or []

    st.subheader("Threat Classification")

    c1, c2, c3 = st.columns([1, 1, 2])
    c1.metric("Threat Type", ttype)
    c2.metric("Severity", severity)
    c3.write("**MITRE Tactics:** " + (", ".join(tactics) if tactics else "-"))

    classification_reason = tc.get("classification_reason") or {}
    login_score = classification_reason.get("login_score")
    c2_score = classification_reason.get("c2_score")
    login_signals = classification_reason.get("login_signals") or []
    c2_signals = classification_reason.get("c2_signals") or []

    st.markdown("**Score breakdown**")
    st.write(f"Login: {login_score} | C2: {c2_score}")

    if classification_reason:
        st.info(
            f"Why: login_score={login_score} | c2_score={c2_score} "
            f"| login_signals={', '.join(login_signals) if login_signals else '-'} "
            f"| c2_signals={', '.join(c2_signals) if c2_signals else '-'}"
        )

    st.markdown("**Signals**")
    login_chip_line = " ".join([f"`{signal}`" for signal in login_signals]) if login_signals else "-"
    c2_chip_line = " ".join([f"`{signal}`" for signal in c2_signals]) if c2_signals else "-"
    st.write("Login signals: " + login_chip_line)
    st.write("C2 signals: " + c2_chip_line)

    if classification_reason.get("tie_breaker_applied") is True:
        st.warning(f"Tie-breaker applied: {classification_reason.get('tie_breaker_rule')}")

    if tactics:
        st.markdown("**MITRE tactics**")
        st.write(" ".join([f"`{tactic}`" for tactic in tactics]))
    if sigs:
        st.markdown("**Signatures matched**")
        for sig in sigs:
            st.write(f"- {sig}")

    with st.expander("Raw threat_classification JSON"):
        st.json(tc)


def render_overview(resp: dict):
    intent = resp.get("intent") or {}
    risk = resp.get("risk_assessment") or {}
    threat_classification = resp.get("threat_classification") or {}
    unified_insight = resp.get("unified_insight") or {}
    intent_type = intent.get("event_type", "-")
    threat_type = threat_classification.get("threat_type")
    primary_label = threat_type if threat_classification else intent_type
    if not primary_label:
        primary_label = "-"

    st.markdown(
        f"<div style='font-size: 1.5rem; font-weight: 700; padding: 0.4rem 0.8rem; "
        f"border-radius: 10px; background: #0f172a; color: #f8fafc; display: inline-block;'>"
        f"Primary Threat: {primary_label}</div>",
        unsafe_allow_html=True
    )
    st.write(f"**Orchestrator Intent:** {intent_type or '-'}")
    st.write(f"**Threat Type (from TD):** {threat_type or '-'}")

    domains = intent.get("domains_engaged", []) or []
    if domains:
        domain_chips = " ".join([f"`{domain}`" for domain in domains])
        st.write("**Domains engaged:** " + domain_chips)
    else:
        st.write("**Domains engaged:** -")

    confidence_score = unified_insight.get("confidence_score")
    confidence_display = f"{confidence_score:.2f}" if isinstance(confidence_score, (int, float)) else "-"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Event", resp.get("event_id", "-"))
    c2.metric("Intent", intent_type or "-")
    c3.metric("Risk Level", risk.get("level", "-"))
    c4.metric("Final Score", risk.get("final_score", "-"))
    c5.metric("Confidence", confidence_display)

    st.write("**Response mode:** " + str(resp.get("response_mode", "-")))
    st.write("**Autonomy level:** " + str(resp.get("autonomy_level", "-")))


def render_agent_results(resp: dict):
    st.subheader("Agent Results")
    results = resp.get("agent_results") or []
    if not results:
        st.write("No agent_results.")
        return

    for ar in results:
        with st.expander(f"{ar.get('agent_id','agent')} • {ar.get('domain','-')} • success={ar.get('success')}"):
            st.write("**Confidence:**", ar.get("confidence"))
            st.write("**Execution time:**", ar.get("execution_time"))
            # findings may not be surfaced in your agent_results currently; still show if present
            if "findings" in ar and ar["findings"]:
                st.json(ar["findings"])


def render_recommendations(resp: dict):
    st.subheader("Recommendations")
    recs = resp.get("recommendations") or []
    if not recs:
        st.write("No recommendations.")
        return
    now_keywords = [
        "isolate", "block", "disable", "revoke", "lock", "contain",
        "reset", "quarantine", "step-up", "force", "disconnect"
    ]
    next_keywords = [
        "investigate", "review", "analyze", "hunt", "monitor",
        "collect", "preserve", "triage", "scope"
    ]
    later_keywords = [
        "post-incident", "hardening", "policy", "patch",
        "training", "audit", "lessons learned"
    ]

    buckets = {"now": [], "next": [], "later": []}
    for rec in recs:
        rec_lower = rec.lower()
        if any(keyword in rec_lower for keyword in now_keywords):
            buckets["now"].append(rec)
        elif any(keyword in rec_lower for keyword in later_keywords):
            buckets["later"].append(rec)
        elif any(keyword in rec_lower for keyword in next_keywords):
            buckets["next"].append(rec)
        else:
            buckets["next"].append(rec)

    st.markdown("**Now (Containment)**")
    if buckets["now"]:
        for rec in buckets["now"]:
            st.write(f"- {rec}")
    else:
        st.write("-")

    st.markdown("**Next (Investigation)**")
    if buckets["next"]:
        for rec in buckets["next"]:
            st.write(f"- {rec}")
    else:
        st.write("-")

    st.markdown("**Later (Hardening)**")
    if buckets["later"]:
        for rec in buckets["later"]:
            st.write(f"- {rec}")
    else:
        st.write("-")


# ---------- UI ----------
with st.sidebar:
    st.header("Settings")
    api = st.text_input("API Base URL", value=ORCH_URL)
    if api:
        ORCH_URL = api.rstrip("/")
    endpoint = st.text_input("API Endpoint", value=ORCH_ENDPOINT)
    if endpoint:
        ORCH_ENDPOINT = endpoint

    st.divider()
    st.header("Test Input")

    scenarios = load_scenarios()
    if not scenarios:
        st.warning("No scenario pack found under datasets/botsv3/scenarios.")
    scenario_names = list(scenarios.keys()) if scenarios else ["(no scenarios found)"]
    pending_scenario = st.session_state.pop("pending_scenario", None)
    if pending_scenario and pending_scenario in scenario_names:
        st.session_state["scenario_name"] = pending_scenario
    scenario_name = st.selectbox("Scenario", scenario_names, key="scenario_name")
    scenario = scenarios.get(scenario_name, {})

    if "scenario_last" not in st.session_state:
        st.session_state["scenario_last"] = scenario_name
        st.session_state["input_text"] = scenario.get("input", "")
        scenario_meta = dict(scenario.get("metadata", {}))
        scenario_iocs = scenario.get("iocs") or []
        if scenario_iocs:
            scenario_meta.setdefault("indicators", list(scenario_iocs))
            scenario_meta.setdefault("iocs", list(scenario_iocs))
            scenario_meta.setdefault("ioc_list", list(scenario_iocs))
        st.session_state["metadata_json"] = json.dumps(scenario_meta, indent=2)
        st.session_state["intent_input"] = scenario.get("intent", "")
    elif st.session_state["scenario_last"] != scenario_name:
        st.session_state["scenario_last"] = scenario_name
        st.session_state["input_text"] = scenario.get("input", "")
        scenario_meta = dict(scenario.get("metadata", {}))
        scenario_iocs = scenario.get("iocs") or []
        if scenario_iocs:
            scenario_meta.setdefault("indicators", list(scenario_iocs))
            scenario_meta.setdefault("iocs", list(scenario_iocs))
            scenario_meta.setdefault("ioc_list", list(scenario_iocs))
        st.session_state["metadata_json"] = json.dumps(scenario_meta, indent=2)
        st.session_state["intent_input"] = scenario.get("intent", "")

    input_text = st.text_area("Alert text", height=120, key="input_text")

    intent = st.text_input("Intent (optional)", value="", key="intent_input")  # leave blank if you want auto
    response_mode = st.selectbox("Response Mode", ["assisted", "autonomous", "manual"], index=0)

    metadata_json = st.text_area(
        "Evidence JSON (preserved in data.original)",
        height=200,
        key="metadata_json",
    )

    st.divider()
    st.header("Overrides (Ops-first)")
    st.info(
        "Evidence is immutable and hashed. Overrides require actor + reason + ticket in regulated mode. "
        "Blocked-field attempts stop execution and are recorded."
    )
    override_enable = st.checkbox("I am making an operational override", value=False)
    override_event_type = st.text_input("Override event_type", value="", disabled=not override_enable)
    override_severity = st.text_input("Override severity (int or label)", value="", disabled=not override_enable)
    override_assets = st.text_input("Override affected_assets (comma)", value="", disabled=not override_enable)
    override_iocs = st.text_input("Override IOCs/indicators (comma)", value="", disabled=not override_enable)
    override_tags = st.text_input("Override tags/labels (comma)", value="", disabled=not override_enable)
    override_owner = st.text_input("Owner", value="", disabled=not override_enable)
    override_team = st.text_input("Team", value="", disabled=not override_enable)
    override_case_id = st.text_input("Case ID", value="", disabled=not override_enable)
    override_ticket_id = st.text_input("Ticket ID", value="", disabled=not override_enable)
    override_notes = st.text_area("Notes", value="", height=80, disabled=not override_enable)
    override_reason = st.selectbox(
        "Override reason",
        [
            "",
            "FALSE_POSITIVE",
            "BUSINESS_CRITICAL",
            "KNOWN_BENIGN",
            "INCIDENT_CONTEXT",
            "ANALYST_OVERRIDE",
        ],
        disabled=not override_enable,
    )
    override_actor = st.text_input("Override actor", value="", disabled=not override_enable)

    override_mode = st.selectbox("Override mode", ["Ops/Demo", "Customer/Regulated"], index=0)

    show_payload_panel = st.checkbox("Show full audit detail", value=False)

    def _has_text(value: str) -> bool:
        return bool((value or "").strip())

    override_inputs = {
        "event_type": override_event_type,
        "severity": override_severity,
        "affected_assets": override_assets,
        "iocs": override_iocs,
        "tags": override_tags,
        "owner": override_owner,
        "team": override_team,
        "case_id": override_case_id,
        "ticket_id": override_ticket_id,
        "notes": override_notes,
    }

    has_override_inputs = any(_has_text(str(value)) for value in override_inputs.values())
    regulated_mode = override_mode == "Customer/Regulated"
    override_enabled = override_enable

    missing_required = []
    if regulated_mode and override_enabled:
        if not _has_text(override_reason):
            missing_required.append("override_reason")
        if not _has_text(override_actor):
            missing_required.append("override_actor")
        if not _has_text(override_ticket_id):
            missing_required.append("ticket_id")

    block_orchestrate = False
    block_reason = ""

    if regulated_mode and has_override_inputs and not override_enabled:
        block_orchestrate = True
        block_reason = "Overrides entered but not enabled. Check 'I am making an operational override' to proceed."

    if regulated_mode and override_enabled and missing_required:
        block_orchestrate = True
        block_reason = f"Regulated overrides require: {', '.join(missing_required)}."

    policy_block_record = None
    if block_orchestrate and regulated_mode:
        attempted_overrides = {key: value for key, value in override_inputs.items() if str(value).strip()}
        attempted_overrides_hash = hashlib.sha256(safe_compact(attempted_overrides).encode("utf-8")).hexdigest()
        evidence_meta_preview = try_parse_json(metadata_json) if metadata_json.strip() else {}
        evidence_meta_preview = evidence_meta_preview if isinstance(evidence_meta_preview, dict) else {}
        evidence_original_preview = {
            "evidence": evidence_meta_preview,
            "_time": evidence_meta_preview.get("_time"),
            "source_label": evidence_meta_preview.get("source"),
        }
        evidence_hash_preview = compute_evidence_hash(input_text, evidence_original_preview)
        policy_block_record = {
            "policy_id": POLICY_ID,
            "policy_version": POLICY_VERSION,
            "ui_app": UI_APP,
            "ui_version": UI_VERSION,
            "session_id": get_session_id(),
            "correlation_id": str(uuid.uuid4()),
            "policy_block": True,
            "blocked_at": utc_now_iso(),
            "mode": override_mode,
            "reason": block_reason,
            "override_enabled": bool(override_enabled),
            "missing_required": missing_required,
            "attempted_override_fields": sorted(list(attempted_overrides.keys())),
            "attempted_override_hash": attempted_overrides_hash,
            "override_count": len(attempted_overrides),
            "override_actor": (override_actor or "").strip(),
            "override_reason": (override_reason or "").strip(),
            "ticket_id": (override_ticket_id or "").strip(),
            "evidence_hash": evidence_hash_preview,
            "orch_endpoint": ORCH_ENDPOINT,
            "scenario_name": scenario_name,
            "selected_mode": override_mode,
            "event_id": evidence_meta_preview.get("event_id"),
            "ui_user": os.getenv("USER", ""),
            "client_host": socket.gethostname(),
            "raw_log_length": len(input_text or ""),
            "override_policy_version": POLICY_VERSION,
        }
        st.session_state["last_policy_block"] = policy_block_record
        get_policy_blocks().append(policy_block_record)
        st.error(block_reason)
        st.download_button(
            "Download Blocked Attempt JSON",
            data=safe_compact(policy_block_record),
            file_name=f"policy_block_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )

    if block_orchestrate and not regulated_mode:
        st.error(block_reason)

    run_btn = st.button("Orchestrate", disabled=block_orchestrate)

if st.session_state.pop("auto_run", False):
    run_btn = True
if st.session_state.pop("run_from_operations", False):
    run_btn = True

# ---------- Run ----------
if run_btn:
    def normalize_severity(value):
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            mapping = {"critical": 90, "high": 75, "medium": 55, "low": 30, "info": 10}
            return mapping.get(value.strip().lower())
        return None

    payload = {"input": input_text, "response_mode": response_mode}

    if intent.strip():
        payload["intent"] = intent.strip()

    # parse metadata if valid JSON
    blocked_override_fields = [
        "event_id",
        "raw_log",
        "source",
        "_time",
        "time",
        "timestamp",
        "event_time",
        "original",
        "data.original",
    ]

    try:
        meta = json.loads(metadata_json) if metadata_json.strip() else None
        if meta:
            payload["metadata"] = meta
    except Exception as exc:
        st.error(f"Metadata JSON invalid: {exc}")
        st.stop()

    scenario_defaults = scenarios.get(scenario_name, {})
    scenario_meta = scenario_defaults.get("metadata", {})
    tamper_fields = []
    meta_override = {}
    merged_meta = {**scenario_meta, **(meta or {})}
    if not merged_meta.get("indicators"):
        indicators = []
        for key in ["destination_ip", "dest_ip", "destination_domain", "dest_domain", "domain", "url"]:
            value = merged_meta.get(key)
            if value:
                indicators.append(value)
        if indicators:
            merged_meta["indicators"] = indicators
    if merged_meta.get("indicators") and not merged_meta.get("iocs"):
        merged_meta["iocs"] = list(merged_meta["indicators"])
    if merged_meta.get("indicators") and not merged_meta.get("ioc_list"):
        merged_meta["ioc_list"] = list(merged_meta["indicators"])

    severity_value = normalize_severity((meta or {}).get("severity"))
    if severity_value is None:
        severity_value = normalize_severity(scenario_defaults.get("severity")) or 50

    maybe_json = try_parse_json(input_text)
    if maybe_json and is_splunk_export_row(maybe_json):
        mapped = map_splunk_windows_security_to_orchestrator(maybe_json)
        payload.update(mapped)
        payload.setdefault("data", {})
        payload["data"]["ui_note"] = "Mapped from Splunk JSON export row"
    else:
        payload.update(
            {
                "event_type": (meta or {}).get("event_type") or scenario_defaults.get("intent") or "unknown",
                "severity": severity_value,
                "source": payload.get("source") or merged_meta.get("source") or "ui",
                "affected_assets": (meta or {}).get("affected_assets") or derive_assets(merged_meta),
                "raw_log": input_text,
                "data": {"message": input_text, **merged_meta},
            }
        )

    data_payload = payload.get("data", {}) or {}
    original_context = {
        "event_type": normalize_text(payload.get("event_type")),
        "severity": payload.get("severity") if payload.get("severity") is not None else 50,
        "affected_assets": normalize_list(payload.get("affected_assets")),
        "source": normalize_text(payload.get("source")),
        "time": normalize_text(data_payload.get("time") or data_payload.get("timestamp") or data_payload.get("event_time")),
        "_time": normalize_text(data_payload.get("_time")),
        "host": normalize_text(data_payload.get("host") or data_payload.get("hostname")),
        "sourcetype": normalize_text((data_payload.get("splunk") or {}).get("sourcetype")),
        "index": normalize_text((data_payload.get("splunk") or {}).get("index")),
    }
    if merged_meta.get("source"):
        original_context["source_label"] = normalize_text(merged_meta.get("source"))

    overrides = {}
    severity_original = payload.get("severity")

    if override_enable:
        override_event_type_value = override_event_type.strip()
        if override_event_type_value:
            overrides["event_type"] = override_event_type_value
            payload["event_type"] = override_event_type_value

        override_severity_value = override_severity.strip()
        if override_severity_value not in ("", None):
            overrides["severity_effective"] = override_severity_value
            payload["severity"] = normalize_severity(override_severity_value)

        override_assets_list = parse_csv_list(override_assets)
        if override_assets_list:
            overrides["affected_assets"] = override_assets_list
            payload["affected_assets"] = override_assets_list

        override_iocs_list = parse_csv_list(override_iocs)
        if override_iocs_list:
            overrides["iocs"] = override_iocs_list
            data_payload["iocs"] = override_iocs_list
            data_payload["indicators"] = override_iocs_list
            data_payload["ioc_list"] = override_iocs_list

        override_tags_list = parse_csv_list(override_tags)
        if override_tags_list:
            overrides["tags"] = override_tags_list
            data_payload["tags"] = override_tags_list

        override_owner_value = override_owner.strip()
        if override_owner_value:
            overrides["owner"] = override_owner_value
            data_payload["owner"] = override_owner_value

        override_team_value = override_team.strip()
        if override_team_value:
            overrides["team"] = override_team_value
            data_payload["team"] = override_team_value

        override_case_id_value = override_case_id.strip()
        if override_case_id_value:
            overrides["case_id"] = override_case_id_value
            data_payload["case_id"] = override_case_id_value

        override_ticket_id_value = override_ticket_id.strip()
        if override_ticket_id_value:
            overrides["ticket_id"] = override_ticket_id_value
            data_payload["ticket_id"] = override_ticket_id_value

        override_notes_value = override_notes.strip()
        if override_notes_value:
            overrides["notes"] = override_notes_value
            data_payload["notes"] = override_notes_value

    override_ticket_id_value = override_ticket_id.strip() if override_enable else ""

    effective_view = {
        "event_type": normalize_text(payload.get("event_type")),
        "severity": payload.get("severity") if payload.get("severity") is not None else 50,
        "affected_assets": normalize_list(payload.get("affected_assets")),
        "iocs": normalize_list(data_payload.get("iocs") or data_payload.get("indicators")),
        "tags": normalize_list(data_payload.get("tags")),
        "owner": normalize_text(data_payload.get("owner")),
        "team": normalize_text(data_payload.get("team")),
        "case_id": normalize_text(data_payload.get("case_id")),
        "ticket_id": normalize_text(data_payload.get("ticket_id")),
        "notes": normalize_text(data_payload.get("notes")),
    }

    data_payload["original"] = original_context
    data_payload["original"]["evidence"] = merged_meta
    data_payload["overrides"] = overrides
    override_reason_value = normalize_text(override_reason if override_enable else "")
    override_actor_value = normalize_text(override_actor if override_enable else "")
    data_payload["overrides_meta"] = {
        "reason_code": override_reason_value,
        "actor": override_actor_value,
        "override_time": datetime.utcnow().isoformat(),
        "tamper_attempt": bool(tamper_fields),
        "tamper_fields": tamper_fields,
        "tamper_count": len(tamper_fields),
    }
    data_payload["override_policy"] = {
        "version": "v1",
        "allowed_fields": [
            "event_type",
            "severity_effective",
            "affected_assets",
            "iocs",
            "tags",
            "owner",
            "team",
            "case_id",
            "ticket_id",
            "notes",
            "response_mode",
        ],
        "blocked_fields": [
            "event_id",
            "raw_log",
            "source",
            "_time",
            "time",
            "timestamp",
            "event_time",
            "data.original",
        ],
        "enforced_by": "ui",
    }
    data_payload["effective"] = effective_view
    data_payload["original"]["evidence_hash"] = compute_evidence_hash(payload.get("raw_log", ""), data_payload["original"])
    payload["data"] = data_payload

    payload["evidence_hash"] = data_payload["original"]["evidence_hash"]
    payload["tamper_attempt"] = bool(tamper_fields)

    try:
        payload["severity"] = int(payload.get("severity", 50))
    except Exception:
        payload["severity"] = 50

    payload["severity_original"] = int(severity_original) if isinstance(severity_original, (int, float)) else severity_original
    payload["severity_effective"] = payload.get("severity")

    if policy_block_record and block_orchestrate:
        policy_block_record["evidence_hash"] = payload.get("evidence_hash", "")
        policy_block_record["override_policy_version"] = data_payload.get("override_policy", {}).get("version", "unknown")
        preview_payload = json.loads(safe_compact(payload))
        preview_payload.setdefault("data", {}).setdefault("overrides_meta", {})["policy_block"] = policy_block_record
    else:
        preview_payload = payload

    if tamper_fields:
        tamper_message = "Override ignored: attempted to change blocked field(s): " + ", ".join(tamper_fields)
        if override_mode == "Customer/Regulated":
            st.error(tamper_message)
            st.stop()
        else:
            st.warning(tamper_message)

    if override_mode == "Customer/Regulated" and override_enable and has_overrides(payload):
        if not override_reason_value or not override_actor_value:
            st.error("Override reason and actor are required in Customer/Regulated mode.")
            st.stop()
        if not override_ticket_id_value:
            st.error("Ticket ID is required for overrides in Customer/Regulated mode.")
            st.stop()

    try:
        resp = post_orchestrate(payload)
    except requests.RequestException as exc:
        st.error(f"API call failed: {exc}")
        st.stop()

    evidence_rows = []
    evidence_ref = scenario_defaults.get("evidence_ref")
    if evidence_ref:
        evidence_path = os.path.join(EVIDENCE_DIR, evidence_ref)
        if os.path.exists(evidence_path):
            evidence_rows = load_evidence(evidence_path)

    st.session_state["last_response"] = resp
    st.session_state["last_payload"] = payload
    st.session_state["last_preview_payload"] = preview_payload
    st.session_state["last_scenario_defaults"] = scenario_defaults
    st.session_state["last_evidence_rows"] = evidence_rows
    st.session_state["last_run_at"] = utc_now_iso()
    incident_record = build_incident_record(resp, payload)
    incident_record["evidence_rows"] = evidence_rows
    add_incident(incident_record)
    st.session_state["selected_incident_id"] = incident_record.get("incident_id")
    st.success("Orchestration completed")

# ---------- Dashboard ----------
resp = st.session_state.get("last_response")
payload = st.session_state.get("last_payload")
preview_payload = st.session_state.get("last_preview_payload") or payload
scenario_defaults = st.session_state.get("last_scenario_defaults") or {}
evidence_rows = st.session_state.get("last_evidence_rows") or []
last_policy_block = st.session_state.get("last_policy_block")
incidents = get_incidents()
active_incident = select_active_incident(incidents)
active_resp = (active_incident or {}).get("raw_api_response") if active_incident else None
active_payload = (active_incident or {}).get("payload_snapshot") if active_incident else None

tabs = st.tabs(["Operations", "Investigation", "Response", "Audit"])

with tabs[0]:
    st.subheader("Operations")
    policy_blocks = get_policy_blocks()
    kpis = compute_kpis(incidents, policy_blocks)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Open incidents", kpis["open_incidents"])
    k2.metric("Critical today", kpis["critical_today"])
    avg_conf = f"{kpis['avg_confidence']:.2f}" if kpis["avg_confidence"] else "-"
    k3.metric("Avg confidence", avg_conf)
    k4.metric("Policy/Tamper", kpis["policy_or_tamper"])

    active_incident = select_active_incident(incidents)
    if active_incident:
        active_resp = active_incident.get("raw_api_response") or {}
        summary = build_incident_summary(active_resp)
        st.markdown("**Incident summary**")
        st.write(f"What happened: {summary['what']}")
        st.write(f"Why: {summary['why'] or '-'}")
        st.write(f"What next: {summary['next']}")
        summary_text = (
            f"What happened: {summary['what']}\n"
            f"Why: {summary['why'] or '-'}\n"
            f"What next: {summary['next']}"
        )
        render_copy_button("Copy incident summary", summary_text, "incident-summary")

        badge_list = ["Evidence locked"]
        if active_incident.get("tamper_attempt"):
            badge_list.append("Tamper")
        approved_store = st.session_state.get("response_approved", {})
        if approved_store.get(active_incident.get("incident_id")):
            badge_list.append("Approved")
        if last_policy_block:
            badge_list.append("Policy blocked")
        render_badges(badge_list)

        risk_score = active_incident.get("risk_score")
        if isinstance(risk_score, (int, float)):
            st.write(f"Risk score: {risk_score}")
            st.progress(min(max(float(risk_score), 0), 100) / 100.0)
        confidence_value = active_incident.get("confidence")
        if isinstance(confidence_value, (int, float)):
            st.write(f"Confidence: {confidence_value:.2f}")
            st.progress(min(max(float(confidence_value), 0), 1.0))
    else:
        st.info("Run Orchestrate to generate an incident.")

    st.subheader("Queue filters")
    status_options = sorted({row.get("status") for row in incidents if row.get("status")})
    risk_options = sorted({row.get("risk_level") for row in incidents if row.get("risk_level")})
    type_options = sorted({row.get("event_type") for row in incidents if row.get("event_type")})
    f1, f2, f3, f4 = st.columns([2, 2, 2, 4])
    status_filter = f1.multiselect("Status", status_options)
    risk_filter = f2.multiselect("Risk level", risk_options)
    type_filter = f3.multiselect("Type", type_options)
    search_text = f4.text_input("Search", value="")

    filtered = filter_incidents(incidents, status_filter, risk_filter, type_filter, search_text)

    st.subheader("Incident queue")
    if filtered:
        table_rows = []
        for row in filtered:
            risk_label = row.get("risk_level")
            if risk_label == "CRITICAL":
                risk_label = "CRITICAL ⚠️"
            table_rows.append(
                {
                    "Risk": risk_label,
                    "Type": row.get("event_type"),
                    "Threat": row.get("threat_type"),
                    "Key asset": (row.get("assets") or ["-"])[0],
                    "Confidence": row.get("confidence"),
                    "Owner/team": " / ".join([x for x in [row.get("owner"), row.get("team")] if x]) or "-",
                    "Age": format_age(row.get("created_at", "")),
                    "Status": row.get("status"),
                    "Flags": incident_flags(row),
                }
            )
        st.dataframe(table_rows, use_container_width=True)
    else:
        st.write("No incidents match the filters.")

    if filtered:
        selected_id = st.selectbox(
            "Select incident",
            [row.get("incident_id") for row in filtered],
            format_func=lambda value: value or "-",
        )
        b1, b2, b3 = st.columns(3)
        if b1.button("Investigate", key="action-investigate"):
            st.session_state["selected_incident_id"] = selected_id
            st.info("Selected incident for investigation. Open the Investigation tab.")
        if b2.button("Mark triaged", key="action-triage"):
            for row in incidents:
                if row.get("incident_id") == selected_id:
                    row["status"] = "triaged"
            st.success("Incident marked as triaged.")
        if b3.button("Create ticket", key="action-ticket"):
            for row in incidents:
                if row.get("incident_id") == selected_id:
                    ticket_value = row.get("ticket_id") or f"TICKET-{datetime.now().strftime('%H%M%S')}"
                    row["ticket_id"] = ticket_value
                    payload_snapshot = row.get("payload_snapshot") or {}
                    data_snapshot = payload_snapshot.get("data") or {}
                    effective = data_snapshot.get("effective") or {}
                    effective["ticket_id"] = ticket_value
                    data_snapshot["effective"] = effective
                    payload_snapshot["data"] = data_snapshot
                    row["payload_snapshot"] = payload_snapshot
            st.success("Ticket stub created for incident.")

    with st.expander("New analysis", expanded=False):
        st.write("Use the sidebar to configure input, then run orchestration.")
        c1, c2 = st.columns(2)
        if c1.button("Run Orchestrator", key="run-from-ops"):
            st.session_state["run_from_operations"] = True
            trigger_rerun()
        if c2.button("Save as Incident", key="save-incident"):
            if st.session_state.get("last_response") and st.session_state.get("last_payload"):
                record = build_incident_record(st.session_state["last_response"], st.session_state["last_payload"])
                add_incident(record)
                st.session_state["selected_incident_id"] = record.get("incident_id")
                st.success("Incident saved to the queue.")
            else:
                st.info("Run Orchestrate first.")

    st.subheader("Preset scenarios")
    if scenarios:
        for name in scenario_names:
            if name == "(no scenarios found)":
                continue
            if st.button(f"Run {name}", key=f"run-scenario-{name}"):
                st.session_state["pending_scenario"] = name
                st.session_state["auto_run"] = True
                trigger_rerun()
    else:
        st.caption("No scenario pack available.")

with tabs[1]:
    st.subheader("Investigation")
    if not active_incident:
        st.info("Run Orchestrate to populate investigation details.")
    else:
        incident_resp = active_incident.get("raw_api_response") or {}
        incident_payload = active_incident.get("payload_snapshot") or {}
        incident_evidence = active_incident.get("evidence_rows") or []
        incident_id = active_incident.get("incident_id")
        risk_level = active_incident.get("risk_level") or "-"
        status = active_incident.get("status") or "-"
        evidence_hash = active_incident.get("evidence_hash") or ""
        short_hash = evidence_hash[:12] if evidence_hash else "-"

        st.markdown(f"### Incident: {incident_id}")
        render_badges([f"Risk: {risk_level}", f"Status: {status}", f"Hash: {short_hash}"])

        qa1, qa2, qa3, qa4 = st.columns([2, 2, 2, 2])
        owner_input = qa1.text_input("Assign owner", value=active_incident.get("owner") or "")
        tag_input = qa2.text_input("Add tag", value="")
        if qa3.button("Generate Splunk query", key="qa-splunk"):
            iocs = extract_iocs(incident_payload) if incident_payload else []
            query = build_splunk_query(iocs)
            st.code(query, language="bash")
            render_copy_button("Copy Splunk query", query, "qa-splunk-copy")
        if qa4.button("Export audit bundle", key="qa-export"):
            audit_bundle = {
                "exported_at": utc_now_iso(),
                "payload": incident_payload,
                "response": incident_resp,
                "policy_block": last_policy_block,
            }
            st.download_button(
                "Download audit bundle (JSON)",
                data=json.dumps(audit_bundle, indent=2),
                file_name=f"audit_bundle_{incident_id}.json",
                mime="application/json",
            )
        if owner_input and owner_input != active_incident.get("owner"):
            active_incident["owner"] = owner_input
        if tag_input:
            tags = set(active_incident.get("tags") or [])
            tags.add(tag_input)
            active_incident["tags"] = sorted(tags)

        render_overview(incident_resp)
        st.divider()
        render_threat_classification(incident_resp)

        st.subheader("Summary")
        unified = incident_resp.get("unified_insight") or {}
        tc = incident_resp.get("threat_classification") or {}
        reason_summary = ""
        if isinstance(tc.get("classification_reason"), dict):
            reason_summary = tc.get("classification_reason", {}).get("summary") or ""
        if not reason_summary:
            reason_summary = tc.get("reason") or ""
        if unified.get("summary"):
            st.write(unified.get("summary"))
        if reason_summary:
            st.write(f"Why it matters: {reason_summary}")
        recs = incident_resp.get("recommendations") or []
        if recs:
            st.write("What to do next:")
            for rec in recs[:3]:
                st.write(f"- {rec}")

        st.subheader("Entities")
        entities = collect_entities(incident_payload, incident_evidence)
        iocs = extract_iocs(incident_payload) if incident_payload else []
        assets = normalize_list(active_incident.get("assets") or entities.get("hosts") or [])
        users = entities.get("users") or []
        ioc_list = iocs or entities.get("ips") or entities.get("domains") or []
        e1, e2, e3 = st.columns(3)
        with e1:
            st.write("**Assets**")
            st.write(" ".join([f"`{a}`" for a in assets]) or "-")
            if assets:
                render_copy_button("Copy assets", "\n".join(assets), "assets-copy")
                if st.button("Add to watchlist", key="watch-assets"):
                    watchlist = get_watchlist()
                    watchlist["assets"] = sorted(set(watchlist["assets"]).union(set(assets)))
                    st.success("Assets added to watchlist.")

        with e2:
            st.write("**Users**")
            st.write(" ".join([f"`{u}`" for u in users]) or "-")
            if users:
                render_copy_button("Copy users", "\n".join(users), "users-copy")
                if st.button("Add to watchlist", key="watch-users"):
                    watchlist = get_watchlist()
                    watchlist["users"] = sorted(set(watchlist["users"]).union(set(users)))
                    st.success("Users added to watchlist.")

        with e3:
            st.write("**IOCs**")
            st.write(" ".join([f"`{i}`" for i in ioc_list]) or "-")
            if ioc_list:
                render_copy_button("Copy IOCs", "\n".join(ioc_list), "iocs-copy")
                if st.button("Add to watchlist", key="watch-iocs"):
                    watchlist = get_watchlist()
                    watchlist["iocs"] = sorted(set(watchlist["iocs"]).union(set(ioc_list)))
                    st.success("IOCs added to watchlist.")

        st.subheader("Timeline")
        timeline_rows = build_timeline_rows(incident_payload, incident_evidence, incident_resp)
        st.dataframe(timeline_rows, use_container_width=True)

        st.subheader("Correlations and hypothesis")
        unified = incident_resp.get("unified_insight") or {}
        correlations = unified.get("cross_domain_correlations") or []
        if correlations:
            for item in correlations:
                st.write(f"- {item}")
        else:
            st.write("No correlations yet.")
        if unified.get("summary"):
            st.write(f"Hypothesis: {unified.get('summary')}")

        st.subheader("Actions")
        recommendations = incident_resp.get("recommendations") or []
        if recommendations:
            action_rows = []
            requires_approval = override_mode == "Customer/Regulated"
            for action in recommendations:
                risk_meta = assess_action_risk(action)
                action_rows.append(
                    {
                        "Action": action,
                        "Risk": risk_meta["risk"],
                        "Requires approval": "Yes" if requires_approval else "No",
                        "Status": "Proposed",
                    }
                )
            st.dataframe(action_rows, use_container_width=True)
        else:
            st.write("No proposed actions.")

        st.subheader("Investigation tools")
        iocs = extract_iocs(incident_payload) if incident_payload else []
        splunk_query = build_splunk_query(iocs)
        st.code(splunk_query, language="bash")
        render_copy_button("Copy Splunk query", splunk_query, "splunk-query")
        ioc_text = "\n".join(iocs) if iocs else ""
        st.code(ioc_text or "No IOCs available.", language="text")
        render_copy_button("Copy IOC list", ioc_text, "ioc-list")
        if iocs:
            st.download_button(
                "Export IOC list",
                data=ioc_text,
                file_name="ioc_list.txt",
                mime="text/plain",
            )

        st.subheader("Agent results")
        render_agent_results(incident_resp)
        with st.expander("Unified Insight"):
            st.json(unified)

        with st.expander("Audit (read-only)", expanded=False):
            st.write(f"Evidence hash: `{incident_payload.get('evidence_hash', '')}`")
            overrides_view = safe_get(incident_payload, ["data", "overrides"], {}) or {}
            if overrides_view:
                st.write("Overrides summary:")
                st.json(overrides_view)
                original = safe_get(incident_payload, ["data", "original"], {}) or {}
                effective = safe_get(incident_payload, ["data", "effective"], {}) or {}
                diff_rows = dict_diff(original, effective)
                if diff_rows:
                    st.table([{"field": field, "original": oa, "effective": ob} for field, oa, ob in diff_rows])
            else:
                st.write("No overrides applied.")

with tabs[2]:
    st.subheader("Response")
    if not active_incident:
        st.info("Run Orchestrate to populate response actions.")
    else:
        incident_resp = active_incident.get("raw_api_response") or {}
        recommendations = incident_resp.get("recommendations") or []
        if not recommendations:
            st.write("No recommendations.")
        else:
            action_status_store = st.session_state.setdefault("response_action_status", {})
            incident_key = active_incident.get("incident_id")
            action_status = action_status_store.setdefault(incident_key, {})
            action_rows = []
            requires_approval = override_mode == "Customer/Regulated"
            for action in recommendations:
                risk_meta = assess_action_risk(action)
                status = action_status.get(action, "Proposed")
                action_rows.append(
                    {
                        "Action": action,
                        "Risk": risk_meta["risk"],
                        "Requires approval": "Yes" if requires_approval else "No",
                        "Rollback": risk_meta["rollback"],
                        "Status": status,
                    }
                )
            st.dataframe(action_rows, use_container_width=True)

        st.subheader("Approval workflow")
        approval_meta_store = st.session_state.setdefault("response_approval_meta", {})
        approval_meta = approval_meta_store.get(incident_key, {})
        approval_actor = st.text_input("Approver", value=approval_meta.get("actor", ""))
        approval_reason = st.text_input("Approval reason", value=approval_meta.get("reason", ""))
        approval_ticket = st.text_input("Ticket ID", value=approval_meta.get("ticket", ""))
        if st.button("Approve actions", key="approve-actions"):
            approval_meta_store[incident_key] = {
                "actor": approval_actor,
                "reason": approval_reason,
                "ticket": approval_ticket,
            }
            approved_store = st.session_state.setdefault("response_approved", {})
            approved_store[incident_key] = True
            for action in recommendations:
                action_status[action] = "Approved"

        approved_store = st.session_state.setdefault("response_approved", {})
        if approved_store.get(incident_key):
            if st.button("Mark actions executed", key="execute-actions"):
                logs_store = st.session_state.setdefault("response_evidence_log", {})
                log = logs_store.setdefault(incident_key, [])
                for action in recommendations:
                    action_status[action] = "Executed"
                    log.append(
                        {
                            "time": utc_now_iso(),
                            "action": action,
                            "actor": approval_actor or "-",
                            "ticket": approval_ticket or "-",
                        }
                    )

        st.subheader("Execution log")
        logs_store = st.session_state.get("response_evidence_log", {})
        log_rows = logs_store.get(incident_key, [])
        if log_rows:
            st.dataframe(log_rows, use_container_width=True)
        else:
            st.write("No execution entries yet.")

with tabs[3]:
    st.subheader("Audit")
    if not active_incident:
        st.info("Run Orchestrate to populate audit data.")
    else:
        incident_payload = active_incident.get("payload_snapshot") or {}
        incident_resp = active_incident.get("raw_api_response") or {}
        st.write(f"Evidence hash: `{incident_payload.get('evidence_hash', '')}`")
        overrides_meta = (incident_payload.get("data") or {}).get("overrides_meta") or {}
        override_actor = overrides_meta.get("actor") or "-"
        override_reason = overrides_meta.get("reason_code") or "-"
        override_ticket = (incident_payload.get("data") or {}).get("effective", {}).get("ticket_id") or "-"
        st.write(f"Overrides: actor={override_actor} | reason={override_reason} | ticket={override_ticket}")
        show_panel = has_overrides(incident_payload) or show_payload_panel
        if show_panel:
            st.subheader("Evidence vs effective")
            original = safe_get(incident_payload, ["data", "original"], {}) or {}
            overrides_view = safe_get(incident_payload, ["data", "overrides"], {}) or {}
            effective = safe_get(incident_payload, ["data", "effective"], {}) or {}

            with st.expander("Evidence (data.original) - immutable", expanded=False):
                st.json(original)

            with st.expander("Overrides (data.overrides)", expanded=False):
                st.json(overrides_view)

            with st.expander("Effective (data.effective) - used for routing", expanded=True):
                st.json(effective)

            diff_rows = dict_diff(original, effective)
            if diff_rows:
                with st.expander("Diff (original -> effective)", expanded=False):
                    st.write(f"{len(diff_rows)} field(s) changed:")
                    st.table([{"field": field, "original": oa, "effective": ob} for field, oa, ob in diff_rows])
        else:
            st.caption("No overrides applied - using original evidence.")

        with st.expander("Raw Evidence (read-only)", expanded=False):
            st.json(incident_payload.get("data", {}).get("original", {}))
            st.text_area("Raw log", value=incident_payload.get("raw_log", ""), height=120, disabled=True)

        with st.expander("Raw Response (read-only)", expanded=False):
            st.json(incident_resp)

        policy_blocks = get_policy_blocks()
        if policy_blocks:
            with st.expander("Override policy block logs", expanded=False):
                st.json(policy_blocks[-1])
                if st.button("Clear blocked records"):
                    st.session_state.pop("last_policy_block", None)
                    st.session_state["policy_blocks"] = []

        audit_bundle = {
            "exported_at": utc_now_iso(),
            "payload": incident_payload,
            "response": incident_resp,
            "policy_block": last_policy_block,
        }
        st.download_button(
            "Download audit bundle (JSON)",
            data=json.dumps(audit_bundle, indent=2),
            file_name="audit_bundle.json",
            mime="application/json",
        )
        try:
            import io
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas

            buffer = io.BytesIO()
            pdf = canvas.Canvas(buffer, pagesize=letter)
            pdf.setTitle("GATRA Audit Bundle")
            pdf.drawString(72, 750, "GATRA Audit Bundle")
            pdf.drawString(72, 730, f"Exported at: {audit_bundle['exported_at']}")
            pdf.drawString(72, 710, f"Event ID: {incident_payload.get('event_id', '-')}")
            pdf.drawString(72, 690, f"Evidence hash: {incident_payload.get('evidence_hash', '')}")
            pdf.drawString(72, 670, "See JSON bundle for full details.")
            pdf.showPage()
            pdf.save()
            buffer.seek(0)
            st.download_button(
                "Download audit bundle (PDF)",
                data=buffer.getvalue(),
                file_name="audit_bundle.pdf",
                mime="application/pdf",
            )
        except Exception:
            st.caption("PDF export not available. Install reportlab to enable.")
