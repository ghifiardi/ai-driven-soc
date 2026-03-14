"""
GATRA AgentDB — HNSW 3-Tier Memory with MutationGuard and Ebbinghaus Forgetting

Core Responsibilities:
- Working / Episodic / Semantic memory tiers with HNSW vector index
- MutationGuard HMAC proof chains for tamper-evident memory integrity
- Ebbinghaus forgetting curves with EWC++ (Elastic Weight Consolidation)
- Tenant-isolated memory stores (tenant_id partitioning)
- Thread-safe read/write with per-tenant locking
"""

import hashlib
import hmac
import json
import logging
import math
import os
import pickle
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import hnswlib
except ImportError:
    hnswlib = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("AgentDB")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMBEDDING_DIM = 384
HNSW_EF_CONSTRUCTION = 200
HNSW_M = 16
HNSW_EF_SEARCH = 50
WORKING_TTL_SECONDS = 300          # 5 min default
EBBINGHAUS_HALF_LIFE = 3600.0      # 1 hour base half-life
EWC_LAMBDA = 400.0                 # EWC++ regularisation strength
MAX_SEMANTIC_MEMORIES = 100_000


class MemoryTier(Enum):
    """Three-tier memory classification."""
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


class ThreatCategory(Enum):
    """Threat intelligence categories for memory classification."""
    INCIDENT = "incident"
    IOC = "ioc"                      # Indicators of Compromise
    TTP = "ttp"                      # Tactics, Techniques, Procedures
    VULNERABILITY = "vulnerability"
    THREAT_ACTOR = "threat_actor"
    CAMPAIGN = "campaign"
    BENIGN = "benign"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class MemoryRecord:
    """Single memory entry across any tier."""
    record_id: str
    tenant_id: str
    tier: MemoryTier
    embedding: np.ndarray
    payload: Dict[str, Any]
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    strength: float = 1.0
    hmac_hash: str = ""
    prev_hmac: str = ""
    severity: float = 0.0
    category: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class EWCTask:
    """Snapshot of Fisher information for one consolidation window."""
    task_id: str
    fisher_diag: np.ndarray          # diagonal Fisher approximation
    param_snapshot: np.ndarray       # θ* at consolidation time
    gamma: float = 0.999             # online EWC++ decay


# ---------------------------------------------------------------------------
# MutationGuard — HMAC proof chain
# ---------------------------------------------------------------------------
class MutationGuard:
    """
    Append-only HMAC chain that makes memory tampering evident.

    Each record's HMAC is computed over its payload + the previous HMAC,
    forming a hash chain analogous to a lightweight blockchain.
    """

    def __init__(self, secret: bytes = b"gatra-memory-guard"):
        self._secret = secret

    def sign(self, payload: Dict[str, Any], prev_hmac: str = "") -> str:
        canonical = json.dumps(payload, sort_keys=True, default=str)
        message = f"{prev_hmac}|{canonical}".encode()
        return hmac.new(self._secret, message, hashlib.sha256).hexdigest()

    def verify(self, payload: Dict[str, Any], expected: str, prev_hmac: str = "") -> bool:
        return hmac.compare_digest(self.sign(payload, prev_hmac), expected)


# ---------------------------------------------------------------------------
# Ebbinghaus forgetting curve
# ---------------------------------------------------------------------------
def ebbinghaus_retention(elapsed: float, half_life: float = EBBINGHAUS_HALF_LIFE) -> float:
    """Return retention ∈ (0, 1] using exponential decay."""
    return math.exp(-0.693 * elapsed / half_life)


# ---------------------------------------------------------------------------
# TenantMemoryStore — per-tenant HNSW index + records
# ---------------------------------------------------------------------------
class _TenantMemoryStore:
    """Isolated memory store for a single tenant."""

    def __init__(self, tenant_id: str, dim: int = EMBEDDING_DIM):
        self.tenant_id = tenant_id
        self.dim = dim
        self.records: Dict[str, MemoryRecord] = {}
        self._id_to_idx: Dict[str, int] = {}
        self._idx_to_id: Dict[int, str] = {}
        self._next_idx = 0
        self.lock = threading.Lock()

        # HNSW index (lazy-init on first insert)
        self._index: Optional[Any] = None

    # -- HNSW helpers -------------------------------------------------------
    def _ensure_index(self):
        if self._index is not None:
            return
        if hnswlib is None:
            raise ImportError(
                "hnswlib is required for AgentDB. Install with: pip install hnswlib"
            )
        idx = hnswlib.Index(space="cosine", dim=self.dim)
        idx.init_index(
            max_elements=MAX_SEMANTIC_MEMORIES,
            ef_construction=HNSW_EF_CONSTRUCTION,
            M=HNSW_M,
        )
        idx.set_ef(HNSW_EF_SEARCH)
        self._index = idx

    def _add_to_index(self, record: MemoryRecord) -> int:
        self._ensure_index()
        idx = self._next_idx
        vec = record.embedding.astype(np.float32).reshape(1, -1)
        self._index.add_items(vec, np.array([idx]))
        self._id_to_idx[record.record_id] = idx
        self._idx_to_id[idx] = record.record_id
        self._next_idx += 1
        return idx

    def _query_index(self, query_vec: np.ndarray, k: int) -> List[Tuple[str, float]]:
        if self._index is None or self._index.get_current_count() == 0:
            return []
        k = min(k, self._index.get_current_count())
        labels, distances = self._index.knn_query(
            query_vec.astype(np.float32).reshape(1, -1), k=k
        )
        results = []
        for lbl, dist in zip(labels[0], distances[0]):
            rid = self._idx_to_id.get(int(lbl))
            if rid:
                results.append((rid, float(dist)))
        return results


# ---------------------------------------------------------------------------
# AgentDB — public API
# ---------------------------------------------------------------------------
class AgentDB:
    """
    3-tier HNSW memory system with HMAC proof chains, Ebbinghaus forgetting,
    and EWC++ consolidation.

    Usage::

        db = AgentDB()
        rid = db.store("tenant-1", embedding_vec, {"alert_id": "A-42"},
                       tier=MemoryTier.EPISODIC)
        results = db.recall("tenant-1", query_vec, k=5)
        db.consolidate("tenant-1")
    """

    def __init__(
        self,
        dim: int = EMBEDDING_DIM,
        hmac_secret: bytes = b"gatra-memory-guard",
        ewc_lambda: float = EWC_LAMBDA,
        working_ttl: float = WORKING_TTL_SECONDS,
        tenant_id: Optional[str] = None,
        persist_dir: Optional[str] = None,
    ):
        self.dim = dim
        self.guard = MutationGuard(secret=hmac_secret)
        self.ewc_lambda = ewc_lambda
        self.working_ttl = working_ttl
        self.default_tenant_id = tenant_id
        self.persist_dir = persist_dir

        self._stores: Dict[str, _TenantMemoryStore] = {}
        self._stores_lock = threading.Lock()

        # EWC++ task registry per tenant
        self._ewc_tasks: Dict[str, List[EWCTask]] = {}
        # Chain head per tenant (last HMAC)
        self._chain_heads: Dict[str, str] = {}

        # Load persisted state if available
        if self.persist_dir:
            self._load_from_disk()

    # -- tenant store accessor ---------------------------------------------
    def _get_store(self, tenant_id: str) -> _TenantMemoryStore:
        if tenant_id not in self._stores:
            with self._stores_lock:
                if tenant_id not in self._stores:
                    self._stores[tenant_id] = _TenantMemoryStore(tenant_id, self.dim)
        return self._stores[tenant_id]

    # -- STORE -------------------------------------------------------------
    def store(
        self,
        tenant_id: Optional[str] = None,
        embedding: Optional[np.ndarray] = None,
        payload: Optional[Dict[str, Any]] = None,
        tier: MemoryTier = MemoryTier.WORKING,
        record_id: Optional[str] = None,
        # Convenience parameters for threat intel use-cases
        category: Optional[ThreatCategory] = None,
        content: Optional[Dict[str, Any]] = None,
        severity: float = 0.0,
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        Insert a memory record and return its id.

        Accepts either the low-level ``(tenant_id, embedding, payload)`` form
        or the convenience ``(category, content, embedding, severity, tags)``
        form used by TAA threat-intel integration.
        """
        tenant_id = tenant_id or self.default_tenant_id
        if tenant_id is None:
            raise ValueError("tenant_id is required (pass it or set default_tenant_id)")

        # Allow 'content' as alias for 'payload'
        if payload is None and content is not None:
            payload = content
        if payload is None:
            payload = {}

        if embedding is None:
            raise ValueError("embedding is required")

        store = self._get_store(tenant_id)
        rid = record_id or str(uuid.uuid4())

        # Merge category into payload if provided
        if category is not None:
            payload["_category"] = category.value

        prev_hmac = self._chain_heads.get(tenant_id, "")
        hmac_hash = self.guard.sign(payload, prev_hmac)

        record = MemoryRecord(
            record_id=rid,
            tenant_id=tenant_id,
            tier=tier,
            embedding=np.asarray(embedding, dtype=np.float32),
            payload=payload,
            hmac_hash=hmac_hash,
            prev_hmac=prev_hmac,
            severity=severity,
            category=category.value if category else None,
            tags=list(tags) if tags else [],
        )

        with store.lock:
            store.records[rid] = record
            store._add_to_index(record)
            self._chain_heads[tenant_id] = hmac_hash

        logger.info(
            "Stored memory %s tier=%s cat=%s sev=%.1f tenant=%s chain=%s…",
            rid, tier.value, record.category, severity, tenant_id, hmac_hash[:12],
        )
        return rid

    # -- RETRIEVE (convenience wrapper) ------------------------------------
    def retrieve(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        min_severity: float = 0.0,
        tenant_id: Optional[str] = None,
        tier_filter: Optional[MemoryTier] = None,
        category_filter: Optional[ThreatCategory] = None,
    ) -> List[MemoryRecord]:
        """
        Convenience retrieval with severity and category filtering.

        Wraps ``recall()`` and adds post-filters for ``min_severity``
        and ``category_filter``.  Uses ``default_tenant_id`` when
        ``tenant_id`` is not provided.
        """
        tenant_id = tenant_id or self.default_tenant_id
        if tenant_id is None:
            raise ValueError("tenant_id is required")

        # Fetch more candidates so post-filters still return k results
        candidates = self.recall(
            tenant_id, query_embedding, k=k * 3, tier_filter=tier_filter,
        )

        results: List[MemoryRecord] = []
        for rec in candidates:
            if rec.severity < min_severity:
                continue
            if category_filter and rec.category != category_filter.value:
                continue
            results.append(rec)
            if len(results) >= k:
                break
        return results

    # -- RECALL ------------------------------------------------------------
    def recall(
        self,
        tenant_id: str,
        query_embedding: np.ndarray,
        k: int = 5,
        tier_filter: Optional[MemoryTier] = None,
        min_strength: float = 0.0,
    ) -> List[MemoryRecord]:
        """Retrieve top-k memories by cosine similarity with strength filtering."""
        store = self._get_store(tenant_id)
        now = time.time()

        with store.lock:
            raw_hits = store._query_index(np.asarray(query_embedding, dtype=np.float32), k * 3)

        results: List[Tuple[float, MemoryRecord]] = []
        for rid, dist in raw_hits:
            rec = store.records.get(rid)
            if rec is None:
                continue
            if tier_filter and rec.tier != tier_filter:
                continue

            # Apply Ebbinghaus decay
            elapsed = now - rec.last_accessed
            retention = ebbinghaus_retention(elapsed)
            effective_strength = rec.strength * retention

            if effective_strength < min_strength:
                continue

            # Update access stats (read-through refresh)
            rec.last_accessed = now
            rec.access_count += 1
            # Rehearsal bonus: each access extends half-life
            rec.strength = min(1.0, effective_strength + 0.05)

            results.append((dist, rec))

        # Sort by distance (cosine: lower = more similar)
        results.sort(key=lambda x: x[0])
        return [rec for _, rec in results[:k]]

    # -- WORKING MEMORY EXPIRY ---------------------------------------------
    def expire_working(self, tenant_id: str) -> int:
        """Remove stale working-memory entries past TTL. Returns count removed."""
        store = self._get_store(tenant_id)
        now = time.time()
        expired: List[str] = []

        with store.lock:
            for rid, rec in store.records.items():
                if rec.tier == MemoryTier.WORKING:
                    if now - rec.created_at > self.working_ttl:
                        expired.append(rid)
            for rid in expired:
                del store.records[rid]

        if expired:
            logger.info(
                "Expired %d working memories for tenant=%s", len(expired), tenant_id
            )
        return len(expired)

    # -- PROMOTE (working → episodic → semantic) ---------------------------
    def promote(self, tenant_id: str, record_id: str) -> MemoryTier:
        """Promote a record to the next tier. Returns the new tier."""
        store = self._get_store(tenant_id)
        promotion_map = {
            MemoryTier.WORKING: MemoryTier.EPISODIC,
            MemoryTier.EPISODIC: MemoryTier.SEMANTIC,
            MemoryTier.SEMANTIC: MemoryTier.SEMANTIC,
        }
        with store.lock:
            rec = store.records.get(record_id)
            if rec is None:
                raise KeyError(f"Record {record_id} not found for tenant {tenant_id}")
            old = rec.tier
            rec.tier = promotion_map[old]
        logger.info(
            "Promoted %s %s→%s tenant=%s", record_id, old.value, rec.tier.value, tenant_id
        )
        return rec.tier

    # -- CONSOLIDATE (EWC++) -----------------------------------------------
    def consolidate(self, tenant_id: str) -> str:
        """
        Run EWC++ consolidation: snapshot current embedding distribution as a
        'task' so future writes are regularised toward important memories.
        """
        store = self._get_store(tenant_id)
        semantic = [
            r for r in store.records.values() if r.tier == MemoryTier.SEMANTIC
        ]
        if not semantic:
            logger.info("No semantic memories to consolidate for tenant=%s", tenant_id)
            return ""

        # Build parameter snapshot: mean of all semantic embeddings
        embeddings = np.stack([r.embedding for r in semantic])
        param_snapshot = embeddings.mean(axis=0)

        # Approximate Fisher as variance of gradients (embedding spread)
        fisher_diag = embeddings.var(axis=0) + 1e-8

        # Online EWC++: decay older tasks
        existing = self._ewc_tasks.get(tenant_id, [])
        for task in existing:
            object.__setattr__(task, "fisher_diag", task.fisher_diag * task.gamma)

        task_id = f"ewc-{tenant_id}-{uuid.uuid4().hex[:8]}"
        new_task = EWCTask(
            task_id=task_id,
            fisher_diag=fisher_diag,
            param_snapshot=param_snapshot,
        )
        self._ewc_tasks.setdefault(tenant_id, []).append(new_task)

        logger.info(
            "EWC++ consolidation task=%s tenant=%s n_semantic=%d total_tasks=%d",
            task_id, tenant_id, len(semantic), len(self._ewc_tasks[tenant_id]),
        )
        return task_id

    def ewc_penalty(self, tenant_id: str, new_params: np.ndarray) -> float:
        """
        Compute the EWC++ penalty for a proposed parameter vector.
        Returns scalar penalty (lower is better).
        """
        tasks = self._ewc_tasks.get(tenant_id, [])
        if not tasks:
            return 0.0
        penalty = 0.0
        for task in tasks:
            diff = new_params - task.param_snapshot
            penalty += float(np.sum(task.fisher_diag * diff ** 2))
        return 0.5 * self.ewc_lambda * penalty

    # -- INTEGRITY ---------------------------------------------------------
    def verify_chain(self, tenant_id: str) -> bool:
        """Walk the HMAC chain for a tenant and verify every link."""
        store = self._get_store(tenant_id)
        ordered = sorted(store.records.values(), key=lambda r: r.created_at)
        prev = ""
        for rec in ordered:
            if not self.guard.verify(rec.payload, rec.hmac_hash, prev):
                logger.error(
                    "Chain broken at record=%s tenant=%s", rec.record_id, tenant_id
                )
                return False
            prev = rec.hmac_hash
        logger.info("Chain verified OK for tenant=%s (%d records)", tenant_id, len(ordered))
        return True

    # -- STATS -------------------------------------------------------------
    def stats(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Return memory statistics for a tenant."""
        tenant_id = tenant_id or self.default_tenant_id
        if tenant_id is None:
            raise ValueError("tenant_id is required")
        store = self._get_store(tenant_id)
        counts = {t.value: 0 for t in MemoryTier}
        for rec in store.records.values():
            counts[rec.tier.value] += 1
        return {
            "tenant_id": tenant_id,
            "total": len(store.records),
            "tiers": counts,
            "ewc_tasks": len(self._ewc_tasks.get(tenant_id, [])),
            "chain_head": self._chain_heads.get(tenant_id, "")[:16],
        }

    # -- PERSISTENCE -------------------------------------------------------
    def save_to_disk(self):
        """Persist all tenant stores to ``persist_dir``."""
        if not self.persist_dir:
            logger.warning("persist_dir not set, skipping save")
            return
        os.makedirs(self.persist_dir, exist_ok=True)
        state = {
            "stores": {},
            "chain_heads": self._chain_heads,
        }
        for tid, store in self._stores.items():
            records_ser = {}
            for rid, rec in store.records.items():
                records_ser[rid] = {
                    "record_id": rec.record_id,
                    "tenant_id": rec.tenant_id,
                    "tier": rec.tier.value,
                    "embedding": rec.embedding.tolist(),
                    "payload": rec.payload,
                    "created_at": rec.created_at,
                    "last_accessed": rec.last_accessed,
                    "access_count": rec.access_count,
                    "strength": rec.strength,
                    "hmac_hash": rec.hmac_hash,
                    "prev_hmac": rec.prev_hmac,
                    "severity": rec.severity,
                    "category": rec.category,
                    "tags": rec.tags,
                }
            state["stores"][tid] = records_ser

        path = os.path.join(self.persist_dir, "agentdb_state.pkl")
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info("Saved AgentDB state to %s (%d tenants)", path, len(self._stores))

    def _load_from_disk(self):
        """Restore state from ``persist_dir`` if a snapshot exists."""
        if not self.persist_dir:
            return
        path = os.path.join(self.persist_dir, "agentdb_state.pkl")
        if not os.path.exists(path):
            return
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)
            self._chain_heads = state.get("chain_heads", {})
            for tid, records_ser in state.get("stores", {}).items():
                store = self._get_store(tid)
                for rid, data in records_ser.items():
                    rec = MemoryRecord(
                        record_id=data["record_id"],
                        tenant_id=data["tenant_id"],
                        tier=MemoryTier(data["tier"]),
                        embedding=np.asarray(data["embedding"], dtype=np.float32),
                        payload=data["payload"],
                        created_at=data["created_at"],
                        last_accessed=data["last_accessed"],
                        access_count=data["access_count"],
                        strength=data["strength"],
                        hmac_hash=data["hmac_hash"],
                        prev_hmac=data["prev_hmac"],
                        severity=data.get("severity", 0.0),
                        category=data.get("category"),
                        tags=data.get("tags", []),
                    )
                    store.records[rid] = rec
                    store._add_to_index(rec)
            logger.info("Loaded AgentDB state from %s (%d tenants)", path, len(self._stores))
        except Exception as e:
            logger.error("Failed to load AgentDB state from %s: %s", path, e)
