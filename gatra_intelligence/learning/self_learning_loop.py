"""
GATRA Self-Learning Loop — RETRIEVE → JUDGE → DISTILL → CONSOLIDATE

Core Responsibilities:
- Four-phase learning cycle integrated with GATRA's PPO RL reward signal
- RETRIEVE: pull relevant episodic memories via AgentDB similarity search
- JUDGE: score retrieved memories against the current PPO reward signal
- DISTILL: compress high-value episodes into compact semantic memories
- CONSOLIDATE: run EWC++ consolidation to protect important knowledge
- Tenant-isolated cycles with configurable thresholds
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from gatra_intelligence.memory.agent_db import (
    AgentDB,
    MemoryRecord,
    MemoryTier,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("GATRASelfLearningLoop")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_RETRIEVE_K = 20
JUDGE_REWARD_THRESHOLD = 0.5
DISTILL_MIN_CLUSTER_SIZE = 3
CONSOLIDATION_INTERVAL_SECONDS = 3600  # 1 hour


class Phase(Enum):
    """Learning cycle phases."""
    RETRIEVE = "retrieve"
    JUDGE = "judge"
    DISTILL = "distill"
    CONSOLIDATE = "consolidate"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class JudgedMemory:
    """A memory scored by the JUDGE phase."""
    record: MemoryRecord
    reward: float
    keep: bool
    reasoning: str = ""


@dataclass
class DistilledKnowledge:
    """Compressed semantic knowledge from DISTILL phase."""
    knowledge_id: str
    tenant_id: str
    centroid_embedding: np.ndarray
    source_ids: List[str]
    summary_payload: Dict[str, Any]
    mean_reward: float
    created_at: float = field(default_factory=time.time)


@dataclass
class CycleResult:
    """Output of one full RETRIEVE→JUDGE→DISTILL→CONSOLIDATE cycle."""
    cycle_id: str
    tenant_id: str
    retrieved_count: int = 0
    judged_kept: int = 0
    judged_discarded: int = 0
    distilled_count: int = 0
    consolidation_task_id: str = ""
    duration_seconds: float = 0.0
    phases_completed: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# PPO Reward Interface
# ---------------------------------------------------------------------------
class PPORewardSignal:
    """
    Adapter for GATRA's PPO reinforcement learning reward.

    Wraps the SOCRLModel feedback or an external reward function so the
    learning loop can score memories against the current policy.
    """

    def __init__(self, reward_fn: Optional[Callable[[Dict[str, Any]], float]] = None):
        self._reward_fn = reward_fn or self._default_reward

    @staticmethod
    def _default_reward(payload: Dict[str, Any]) -> float:
        """
        Default reward heuristic when no PPO model is connected.

        Rewards true positives, penalises false positives, neutral otherwise.
        Maps to the SOCRLModel convention: +1.0 TP, -1.0 FP.
        """
        outcome = payload.get("outcome", "").lower()
        if outcome in ("true_positive", "tp"):
            return 1.0
        if outcome in ("false_positive", "fp"):
            return -1.0
        # Partial credit for severity-proportional signals
        severity = float(payload.get("severity", 5))
        return (severity - 5) / 5.0   # maps [0,10] → [-1, +1]

    def score(self, payload: Dict[str, Any]) -> float:
        return self._reward_fn(payload)


# ---------------------------------------------------------------------------
# GATRASelfLearningLoop — public API
# ---------------------------------------------------------------------------
class GATRASelfLearningLoop:
    """
    Runs the RETRIEVE → JUDGE → DISTILL → CONSOLIDATE learning cycle.

    Integrates with:
    - AgentDB for memory retrieval and storage
    - PPO reward signal for memory quality judgement
    - EWC++ consolidation to protect important knowledge

    Usage::

        db = AgentDB()
        loop = GATRASelfLearningLoop(db)
        result = loop.run_cycle("tenant-1", context_embedding)
    """

    def __init__(
        self,
        agent_db: AgentDB,
        reward_signal: Optional[PPORewardSignal] = None,
        retrieve_k: int = DEFAULT_RETRIEVE_K,
        judge_threshold: float = JUDGE_REWARD_THRESHOLD,
        distill_min_cluster: int = DISTILL_MIN_CLUSTER_SIZE,
    ):
        self.db = agent_db
        self.reward = reward_signal or PPORewardSignal()
        self.retrieve_k = retrieve_k
        self.judge_threshold = judge_threshold
        self.distill_min_cluster = distill_min_cluster
        self._last_consolidation: Dict[str, float] = {}

    # -- RETRIEVE ----------------------------------------------------------
    def retrieve(
        self,
        tenant_id: str,
        context_embedding: np.ndarray,
    ) -> List[MemoryRecord]:
        """Phase 1: Retrieve episodic memories relevant to current context."""
        records = self.db.recall(
            tenant_id,
            context_embedding,
            k=self.retrieve_k,
            tier_filter=MemoryTier.EPISODIC,
        )
        logger.info(
            "RETRIEVE tenant=%s found %d episodic memories", tenant_id, len(records)
        )
        return records

    # -- JUDGE -------------------------------------------------------------
    def judge(self, memories: List[MemoryRecord]) -> List[JudgedMemory]:
        """Phase 2: Score each memory against the PPO reward signal."""
        judged: List[JudgedMemory] = []
        for mem in memories:
            reward = self.reward.score(mem.payload)
            keep = reward >= self.judge_threshold
            judged.append(JudgedMemory(
                record=mem,
                reward=reward,
                keep=keep,
                reasoning=f"reward={reward:.3f} threshold={self.judge_threshold}",
            ))

        kept = sum(1 for j in judged if j.keep)
        logger.info(
            "JUDGE scored %d memories: %d kept, %d discarded",
            len(judged), kept, len(judged) - kept,
        )
        return judged

    # -- DISTILL -----------------------------------------------------------
    def distill(
        self, tenant_id: str, judged: List[JudgedMemory]
    ) -> List[DistilledKnowledge]:
        """
        Phase 3: Compress high-value episodes into semantic memories.

        Groups kept memories by payload similarity, computes centroid
        embeddings, and stores the distilled knowledge as SEMANTIC tier.
        """
        kept = [j for j in judged if j.keep]
        if len(kept) < self.distill_min_cluster:
            logger.info(
                "DISTILL skipped: only %d kept memories (need %d)",
                len(kept), self.distill_min_cluster,
            )
            return []

        # Compute centroid embedding of all kept memories
        embeddings = np.stack([j.record.embedding for j in kept])
        centroid = embeddings.mean(axis=0)

        # Aggregate payload into a summary
        source_ids = [j.record.record_id for j in kept]
        mean_reward = float(np.mean([j.reward for j in kept]))
        alert_types = set()
        severities = []
        for j in kept:
            alert_types.add(j.record.payload.get("alert_type", "unknown"))
            severities.append(float(j.record.payload.get("severity", 0)))

        summary_payload = {
            "distilled": True,
            "source_count": len(kept),
            "alert_types": sorted(alert_types),
            "mean_severity": float(np.mean(severities)) if severities else 0.0,
            "mean_reward": mean_reward,
        }

        # Store as semantic memory
        knowledge_id = str(uuid.uuid4())
        self.db.store(
            tenant_id,
            centroid,
            summary_payload,
            tier=MemoryTier.SEMANTIC,
            record_id=knowledge_id,
        )

        # Promote source episodic memories that scored highly
        for j in kept:
            if j.reward >= self.judge_threshold * 1.5:
                try:
                    self.db.promote(tenant_id, j.record.record_id)
                except KeyError:
                    pass

        distilled = DistilledKnowledge(
            knowledge_id=knowledge_id,
            tenant_id=tenant_id,
            centroid_embedding=centroid,
            source_ids=source_ids,
            summary_payload=summary_payload,
            mean_reward=mean_reward,
        )

        logger.info(
            "DISTILL produced knowledge=%s from %d sources mean_reward=%.3f tenant=%s",
            knowledge_id, len(kept), mean_reward, tenant_id,
        )
        return [distilled]

    # -- CONSOLIDATE -------------------------------------------------------
    def consolidate(self, tenant_id: str) -> str:
        """Phase 4: Run EWC++ consolidation to protect semantic knowledge."""
        task_id = self.db.consolidate(tenant_id)
        self._last_consolidation[tenant_id] = time.time()
        if task_id:
            logger.info("CONSOLIDATE task=%s tenant=%s", task_id, tenant_id)
        return task_id

    # -- FULL CYCLE --------------------------------------------------------
    def run_cycle(
        self,
        tenant_id: str,
        context_embedding: np.ndarray,
        force_consolidate: bool = False,
    ) -> CycleResult:
        """
        Execute the full RETRIEVE → JUDGE → DISTILL → CONSOLIDATE cycle.

        Consolidation runs only if the interval has elapsed or forced.
        """
        cycle_id = f"cycle-{uuid.uuid4().hex[:8]}"
        start = time.time()
        result = CycleResult(cycle_id=cycle_id, tenant_id=tenant_id)

        logger.info("=== Learning cycle %s START tenant=%s ===", cycle_id, tenant_id)

        # Phase 1: RETRIEVE
        retrieved = self.retrieve(tenant_id, context_embedding)
        result.retrieved_count = len(retrieved)
        result.phases_completed.append(Phase.RETRIEVE.value)

        # Phase 2: JUDGE
        judged = self.judge(retrieved)
        result.judged_kept = sum(1 for j in judged if j.keep)
        result.judged_discarded = len(judged) - result.judged_kept
        result.phases_completed.append(Phase.JUDGE.value)

        # Phase 3: DISTILL
        distilled = self.distill(tenant_id, judged)
        result.distilled_count = len(distilled)
        result.phases_completed.append(Phase.DISTILL.value)

        # Phase 4: CONSOLIDATE (rate-limited)
        last = self._last_consolidation.get(tenant_id, 0)
        if force_consolidate or (time.time() - last > CONSOLIDATION_INTERVAL_SECONDS):
            task_id = self.consolidate(tenant_id)
            result.consolidation_task_id = task_id
            result.phases_completed.append(Phase.CONSOLIDATE.value)

        result.duration_seconds = time.time() - start

        logger.info(
            "=== Learning cycle %s DONE in %.2fs: retrieved=%d kept=%d "
            "distilled=%d consolidated=%s ===",
            cycle_id, result.duration_seconds, result.retrieved_count,
            result.judged_kept, result.distilled_count,
            bool(result.consolidation_task_id),
        )
        return result
