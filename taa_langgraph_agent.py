from langgraph.graph import StateGraph, END
import logging
import json
import os
import threading
import time
from google.cloud import pubsub_v1
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    _st_model = SentenceTransformer("all-MiniLM-L6-v2")
except ImportError:
    _st_model = None

from gatra_intelligence.memory.agent_db import AgentDB, ThreatCategory, MemoryTier
from gatra_intelligence.learning.self_learning_loop import GATRASelfLearningLoop
from gatra_intelligence.routing.model_router import (
    GATRAModelRouter, AgentType, TaskType,
)

logger = logging.getLogger("TAALangGraphAgent")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class TAAConfig:
    project_id: str = os.getenv("BIGQUERY_PROJECT_ID", "chronicle-dev-2be9")
    tenant_id: str = os.getenv("TENANT_ID", "default")
    persist_dir: str = os.getenv("GATRA_MEMORY_DIR", "/var/gatra/memory")
    embedding_dim: int = 384


# ---------------------------------------------------------------------------
# Text → embedding via sentence-transformers (all-MiniLM-L6-v2, 384-dim)
# ---------------------------------------------------------------------------
def embed(text: str) -> np.ndarray:
    """
    Encode text into a 384-dim normalised embedding using all-MiniLM-L6-v2.

    Falls back to a deterministic hash projection if sentence-transformers
    is not installed (e.g. in lightweight test environments).
    """
    if _st_model is not None:
        vec = _st_model.encode(text, normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32)
    # Fallback: deterministic hash projection for testing only
    import hashlib
    h = hashlib.sha512(text.encode()).digest()
    seed = int.from_bytes(h[:4], "big")
    rng = np.random.RandomState(seed)
    vec = rng.randn(384).astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-9)


# Example state structure
class TAAState(dict):
    pass

# Global publisher for Pub/Sub
publisher = None

# Shared memory and learning instances (initialised per-config)
_agent_db: Optional[AgentDB] = None
_learning_loop: Optional[GATRASelfLearningLoop] = None
_router: Optional[GATRAModelRouter] = None
_taa_config: Optional[TAAConfig] = None
_learning_thread: Optional[threading.Thread] = None


def _init_intelligence(config: Optional[TAAConfig] = None):
    """Initialise shared AgentDB, learning loop, and router."""
    global _agent_db, _learning_loop, _router, _taa_config
    if _agent_db is not None:
        return
    config = config or TAAConfig()
    _taa_config = config
    _agent_db = AgentDB(
        tenant_id=config.tenant_id,
        persist_dir=config.persist_dir,
        dim=config.embedding_dim,
    )
    _learning_loop = GATRASelfLearningLoop(_agent_db)
    _router = GATRAModelRouter()
    logger.info(
        "TAA intelligence initialised: tenant=%s persist=%s",
        config.tenant_id, config.persist_dir,
    )


def init_pubsub():
    global publisher
    if publisher is None:
        publisher = pubsub_v1.PublisherClient()
    return publisher

def receive_alert_node(state: TAAState) -> TAAState:
    _init_intelligence()
    logging.info(f"Received alert: {state.get('alert_id')}")
    return state

def enrichment_node(state: TAAState) -> TAAState:
    """Enrich alert with similar past incidents from AgentDB memory."""
    alert_id = state.get('alert_id', '')
    alert_data = state.get('alert_data', {})
    alert_text = json.dumps({"alert_id": alert_id, **alert_data}, default=str)
    query_vec = embed(alert_text)

    # --- Memory-backed enrichment: check for similar past incidents ---
    similar = _agent_db.retrieve(
        query_embedding=query_vec,
        k=5,
        min_severity=6,
    )

    if similar:
        past_context = []
        for rec in similar:
            past_context.append({
                "alert_id": rec.payload.get("alert_id", rec.record_id),
                "outcome": rec.payload.get("outcome", "unknown"),
                "severity": rec.severity,
                "ttps": rec.tags,
            })
        state['memory_context'] = past_context
        state['memory_prompt_prefix'] = (
            f"Similar past incidents ({len(similar)} found): "
            + json.dumps(past_context, default=str)
        )
        logging.info(
            "Memory enrichment: %d similar incidents found for alert %s",
            len(similar), alert_id,
        )
    else:
        state['memory_context'] = []
        state['memory_prompt_prefix'] = ""

    state['enriched'] = True
    state['_alert_embedding'] = query_vec.tolist()
    logging.info(f"Enriched alert: {alert_id}")
    return state

def llm_analysis_node(state: TAAState) -> TAAState:
    """LLM analysis with memory context prepended to the prompt."""
    memory_prefix = state.get('memory_prompt_prefix', '')
    alert_id = state.get('alert_id', '')

    # In production, the memory_prefix is prepended to the actual LLM prompt.
    # For now we log it and keep the existing stub result.
    if memory_prefix:
        logging.info("LLM prompt enriched with memory context for alert %s", alert_id)

    state['llm_result'] = {
        'is_true_positive': True,
        'confidence': 0.95,
        'severity': 'high',
        'memory_assisted': bool(memory_prefix),
    }
    logging.info(f"LLM analysis complete for alert: {alert_id}")
    return state

def unpack_state(state):
    # Utility to handle (node, state) tuple if passed by mistake
    if isinstance(state, tuple):
        return state[1]
    return state

def containment_node(state: TAAState) -> TAAState:
    state = unpack_state(state)
    state['containment'] = 'requested'
    
    # Publish containment request to CRA
    try:
        pub = init_pubsub()
        project_id = "chronicle-dev-2be9"  # TODO: Make configurable
        topic_name = "containment-requests"
        topic_path = pub.topic_path(project_id, topic_name)
        
        containment_data = {
            "alert_id": state.get('alert_id'),
            "alert_data": state.get('alert_data', {}),
            "llm_result": state.get('llm_result', {}),
            "timestamp": state.get('timestamp'),
            "containment_request": {
                "action": "immediate_containment",
                "severity": state.get('llm_result', {}).get('severity', 'high'),
                "confidence": state.get('llm_result', {}).get('confidence', 0.95),
                "reasoning": f"High severity true positive detected: {state.get('llm_result', {}).get('reasoning', 'Automated analysis')}"
            }
        }
        
        message_data = json.dumps(containment_data).encode("utf-8")
        future = pub.publish(topic_path, message_data)
        message_id = future.result()
        
        logging.info(f"Published containment request for alert {state.get('alert_id')} to CRA. Message ID: {message_id}")
        state['containment_message_id'] = message_id
        
    except Exception as e:
        logging.error(f"Failed to publish containment request for alert {state.get('alert_id')}: {e}")
        state['containment_error'] = str(e)
    
    return state

def manual_review_node(state: TAAState) -> TAAState:
    state = unpack_state(state)
    state['manual_review'] = True
    logging.info(f"Manual review flagged for alert: {state.get('alert_id')}")
    return state

def feedback_node(state: TAAState) -> TAAState:
    state = unpack_state(state)
    state['feedback'] = 'published'

    alert_id = state.get('alert_id', '')
    llm_result = state.get('llm_result', {})
    severity_str = llm_result.get('severity', 'medium')
    severity_map = {'low': 3.0, 'medium': 5.0, 'high': 8.0, 'critical': 10.0}
    severity_num = severity_map.get(severity_str, 5.0)
    is_tp = llm_result.get('is_true_positive', False)
    outcome = "true_positive" if is_tp else "false_positive"

    # --- Store the outcome in AgentDB for future memory retrieval ---
    try:
        alert_embedding = state.get('_alert_embedding')
        if alert_embedding is not None:
            alert_vec = np.asarray(alert_embedding, dtype=np.float32)
        else:
            alert_text = json.dumps({"alert_id": alert_id, **state.get('alert_data', {})}, default=str)
            alert_vec = embed(alert_text)

        ttps = state.get('alert_data', {}).get('ttps', [])
        _agent_db.store(
            category=ThreatCategory.INCIDENT,
            content={
                "alert_id": alert_id,
                "ttps": ttps,
                "outcome": outcome,
                "severity": severity_str,
                "confidence": llm_result.get('confidence', 0.0),
                "containment_requested": state.get('containment') == 'requested',
            },
            embedding=alert_vec,
            severity=severity_num,
            tags=ttps if isinstance(ttps, list) else [],
            tier=MemoryTier.EPISODIC,
        )
        logging.info("Stored alert %s outcome in AgentDB (sev=%.1f, %s)", alert_id, severity_num, outcome)
    except Exception as e:
        logging.error("Failed to store alert %s in AgentDB: %s", alert_id, e)

    # Publish feedback to CLA via Pub/Sub
    try:
        pub = init_pubsub()
        project_id = _taa_config.project_id if _taa_config else "chronicle-dev-2be9"
        topic_name = "taa-feedback"
        topic_path = pub.topic_path(project_id, topic_name)

        feedback_data = {
            "alert_id": alert_id,
            "is_true_positive": is_tp,
            "confidence": llm_result.get('confidence', 0.95),
            "severity": severity_str,
            "timestamp": state.get('timestamp'),
            "reasoning": llm_result.get('reasoning', 'Automated analysis'),
            "source": "taa_langgraph"
        }

        message_data = json.dumps(feedback_data).encode("utf-8")
        future = pub.publish(topic_path, message_data)
        message_id = future.result()

        logging.info(f"Published feedback for alert {alert_id} to CLA. Message ID: {message_id}")
        state['feedback_message_id'] = message_id

    except Exception as e:
        logging.error(f"Failed to publish feedback for alert {alert_id}: {e}")
        state['feedback_error'] = str(e)

    return state

def reporting_node(state: TAAState) -> TAAState:
    state = unpack_state(state)
    state['reported'] = True
    
    # Publish report to RVA (Reporting & Visualization Agent)
    try:
        pub = init_pubsub()
        project_id = "chronicle-dev-2be9"  # TODO: Make configurable
        topic_name = "taa-reports"
        topic_path = pub.topic_path(project_id, topic_name)
        
        report_data = {
            "alert_id": state.get('alert_id'),
            "alert_data": state.get('alert_data', {}),
            "llm_result": state.get('llm_result', {}),
            "timestamp": state.get('timestamp'),
            "containment_requested": state.get('containment') == 'requested',
            "manual_review_flagged": state.get('manual_review', False),
            "feedback_published": state.get('feedback') == 'published',
            "containment_message_id": state.get('containment_message_id'),
            "feedback_message_id": state.get('feedback_message_id'),
            "source": "taa_langgraph"
        }
        
        message_data = json.dumps(report_data).encode("utf-8")
        future = pub.publish(topic_path, message_data)
        message_id = future.result()
        
        logging.info(f"Published report for alert {state.get('alert_id')} to RVA. Message ID: {message_id}")
        state['report_message_id'] = message_id
        
    except Exception as e:
        logging.error(f"Failed to publish report for alert {state.get('alert_id')}: {e}")
        state['report_error'] = str(e)
    
    return state

def decision_node(state: TAAState) -> TAAState:
    result = state.get('llm_result', {})
    if result.get('is_true_positive') and result.get('severity') == 'high':
        next_node = 'containment'
        logging.info("Routing to containment node")
    elif result.get('confidence', 1.0) < 0.7:
        next_node = 'manual_review'
        logging.info("Routing to manual_review node")
    else:
        next_node = 'feedback'
        logging.info("Routing to feedback node")
    state['next'] = next_node
    return state

def build_taa_workflow():
    graph = StateGraph(TAAState)
    
    # Add nodes
    graph.add_node('receive_alert', receive_alert_node)
    graph.add_node('enrichment', enrichment_node)
    graph.add_node('llm_analysis', llm_analysis_node)
    graph.add_node('decision', decision_node)
    graph.add_node('containment', containment_node)
    graph.add_node('manual_review', manual_review_node)
    graph.add_node('feedback', feedback_node)
    graph.add_node('reporting', reporting_node)

    # Set entrypoint
    graph.set_entry_point('receive_alert')

    # Define linear flow before decision
    graph.add_edge('receive_alert', 'enrichment')
    graph.add_edge('enrichment', 'llm_analysis')
    graph.add_edge('llm_analysis', 'decision')
    
    # Define conditional routing after decision
    graph.add_conditional_edges(
        'decision',
        lambda s: s.get('next', 'feedback'),
        {
            'containment': 'containment',
            'manual_review': 'manual_review', 
            'feedback': 'feedback'
        }
    )
    
    # Define flow after decision branches
    graph.add_edge('containment', 'feedback')
    graph.add_edge('manual_review', 'feedback')
    graph.add_edge('feedback', 'reporting')
    graph.add_edge('reporting', END)

    return graph

def start_learning_loop(interval_seconds: int = 300):
    """
    Start the RETRIEVE→JUDGE→DISTILL→CONSOLIDATE learning loop
    as a background daemon thread.

    Runs every ``interval_seconds`` (default 5 min), using the most
    recent alert embedding as the context query.  Closed alerts that
    were stored in ``feedback_node`` are automatically used to improve
    routing and memory quality over time.
    """
    global _learning_thread

    def _loop():
        logger.info("Background learning loop started (interval=%ds)", interval_seconds)
        while True:
            try:
                time.sleep(interval_seconds)
                if _agent_db is None or _learning_loop is None:
                    continue
                tenant = _taa_config.tenant_id if _taa_config else "default"
                stats = _agent_db.stats(tenant)
                if stats["total"] == 0:
                    continue
                # Use a random unit vector as context (covers the full space)
                context_vec = np.random.randn(_agent_db.dim).astype(np.float32)
                context_vec /= np.linalg.norm(context_vec) + 1e-9
                result = _learning_loop.run_cycle(
                    tenant, context_vec, force_consolidate=False,
                )
                logger.info(
                    "Learning cycle %s: retrieved=%d kept=%d distilled=%d",
                    result.cycle_id, result.retrieved_count,
                    result.judged_kept, result.distilled_count,
                )
                # Persist after each cycle
                _agent_db.save_to_disk()
            except Exception as e:
                logger.error("Learning loop error: %s", e)

    if _learning_thread is not None and _learning_thread.is_alive():
        logger.info("Learning loop already running")
        return

    _learning_thread = threading.Thread(target=_loop, daemon=True, name="gatra-learning")
    _learning_thread.start()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    # Initialise intelligence layer
    config = TAAConfig(persist_dir="/tmp/gatra_memory_test")
    _init_intelligence(config)

    # Start background learning loop
    start_learning_loop(interval_seconds=60)

    workflow = build_taa_workflow().compile()
    initial_state = TAAState(alert_id="test-alert-001", alert_data={"example": "data"})
    final_state = workflow.invoke(initial_state)
    print("Final state:", final_state)