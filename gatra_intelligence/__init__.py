"""
GATRA Intelligence — Adaptive Memory, Routing, and Self-Learning

Provides three core subsystems for the GATRA SOC platform:

- **AgentDB**: 3-tier HNSW memory (working/episodic/semantic) with
  MutationGuard HMAC proof chains and Ebbinghaus forgetting + EWC++.
- **GATRAModelRouter**: Claude model routing (Haiku→Sonnet→Opus) with
  Q-table reinforcement learning and per-tenant cost tracking.
- **GATRASelfLearningLoop**: RETRIEVE→JUDGE→DISTILL→CONSOLIDATE cycle
  integrated with GATRA's PPO RL reward signal.
"""

from gatra_intelligence.memory.agent_db import (
    AgentDB,
    MemoryRecord,
    MemoryTier,
    MutationGuard,
    ThreatCategory,
)
from gatra_intelligence.routing.model_router import (
    AgentType,
    ClaudeModel,
    GATRAModelRouter,
    TaskType,
)
from gatra_intelligence.learning.self_learning_loop import (
    GATRASelfLearningLoop,
    PPORewardSignal,
)

__version__ = "0.1.0"

__all__ = [
    "AgentDB",
    "MemoryRecord",
    "MemoryTier",
    "MutationGuard",
    "ThreatCategory",
    "AgentType",
    "ClaudeModel",
    "GATRAModelRouter",
    "TaskType",
    "GATRASelfLearningLoop",
    "PPORewardSignal",
]
