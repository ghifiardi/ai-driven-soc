"""
GATRA Model Router — 3-Tier Claude Routing with Q-Table Learning

Core Responsibilities:
- Route tasks to Claude Haiku / Sonnet / Opus based on agent type and severity
- Default mapping: ADA→Haiku, TAA→Sonnet, deep forensics & incident reports→Opus
- Severity escalation: any task with severity ≥ 9 escalates to Opus
- Q-table reinforcement learning for adaptive routing decisions
- Per-tenant cost tracking and budget enforcement
"""

import json
import logging
import os
import random
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

try:
    import anthropic
except ImportError:
    anthropic = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("GATRAModelRouter")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEVERITY_ESCALATION_THRESHOLD = 9


class ClaudeModel(Enum):
    """Anthropic Claude model tiers (cheapest → most capable)."""
    HAIKU = "claude-haiku-4-5-20251001"
    SONNET = "claude-sonnet-4-6"
    OPUS = "claude-opus-4-6"


class AgentType(Enum):
    """GATRA agent identifiers."""
    ADA = "ada"   # Anomaly Detection Agent
    TAA = "taa"   # Triage & Analysis Agent
    CRA = "cra"   # Containment & Response Agent
    CLA = "cla"   # Continuous Learning Agent
    RVA = "rva"   # Reporting & Visualization Agent


class TaskType(Enum):
    """Task categories that influence routing."""
    PRE_FILTER = "pre_filter"
    ALERT_TRIAGE = "alert_triage"
    ANOMALY_SCORING = "anomaly_scoring"
    LOG_ENRICHMENT = "log_enrichment"
    DEEP_FORENSICS = "deep_forensics"
    INCIDENT_REPORT = "incident_report"
    THREAT_HUNTING = "threat_hunting"
    POLICY_UPDATE = "policy_update"
    CONTAINMENT = "containment"
    METRIC_SUMMARY = "metric_summary"


# Cost per 1M tokens (input, output) — approximate pricing
MODEL_COST_PER_M: Dict[ClaudeModel, Tuple[float, float]] = {
    ClaudeModel.HAIKU:  (1.00,  5.00),
    ClaudeModel.SONNET: (3.00, 15.00),
    ClaudeModel.OPUS:   (15.00, 75.00),
}

# Default static routing table
DEFAULT_ROUTE: Dict[AgentType, ClaudeModel] = {
    AgentType.ADA: ClaudeModel.HAIKU,
    AgentType.TAA: ClaudeModel.SONNET,
    AgentType.CRA: ClaudeModel.SONNET,
    AgentType.CLA: ClaudeModel.HAIKU,
    AgentType.RVA: ClaudeModel.SONNET,
}

# Task types that always escalate to Opus
OPUS_TASK_TYPES = frozenset({TaskType.DEEP_FORENSICS, TaskType.INCIDENT_REPORT})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class RoutingDecision:
    """Immutable record of a single routing decision."""
    request_id: str
    tenant_id: str
    agent: AgentType
    task_type: TaskType
    severity: float
    selected_model: ClaudeModel
    reason: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class CostRecord:
    """Accumulated cost for a tenant."""
    tenant_id: str
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    request_count: int = 0
    cost_by_model: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Q-Table for adaptive routing
# ---------------------------------------------------------------------------
class _QTable:
    """
    Tabular Q-learning over (agent, task_type, severity_bucket) → model.

    The state is (AgentType, TaskType, severity_bucket).
    The action is the ClaudeModel selected.
    Reward comes from downstream quality signal (accuracy, latency, cost).
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount: float = 0.95,
        epsilon: float = 0.15,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.02,
    ):
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        # Q[state_key][model_name] = q_value
        self.q: Dict[str, Dict[str, float]] = {}
        self.lock = threading.Lock()

    @staticmethod
    def _severity_bucket(severity: float) -> str:
        if severity < 4:
            return "low"
        if severity < 7:
            return "medium"
        if severity < 9:
            return "high"
        return "critical"

    @staticmethod
    def _state_key(agent: AgentType, task: TaskType, severity: float) -> str:
        bucket = _QTable._severity_bucket(severity)
        return f"{agent.value}|{task.value}|{bucket}"

    def _init_state(self, key: str):
        if key not in self.q:
            # Initialise with slight preference for cheaper models
            self.q[key] = {
                ClaudeModel.HAIKU.value: 0.6,
                ClaudeModel.SONNET.value: 0.4,
                ClaudeModel.OPUS.value: 0.2,
            }

    def select(
        self, agent: AgentType, task: TaskType, severity: float
    ) -> ClaudeModel:
        key = self._state_key(agent, task, severity)
        with self.lock:
            self._init_state(key)
            if random.random() < self.epsilon:
                return random.choice(list(ClaudeModel))
            best = max(self.q[key], key=self.q[key].get)
            return ClaudeModel(best)

    def update(
        self,
        agent: AgentType,
        task: TaskType,
        severity: float,
        model: ClaudeModel,
        reward: float,
    ):
        key = self._state_key(agent, task, severity)
        with self.lock:
            self._init_state(key)
            old_q = self.q[key][model.value]
            # No next-state in contextual bandit formulation
            max_next = max(self.q[key].values())
            new_q = old_q + self.lr * (reward + self.gamma * max_next - old_q)
            self.q[key][model.value] = new_q
            # Decay exploration
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ---------------------------------------------------------------------------
# GATRAModelRouter — public API
# ---------------------------------------------------------------------------
class GATRAModelRouter:
    """
    Routes GATRA tasks to the appropriate Claude model tier.

    Static rules guarantee correctness (Opus for deep forensics, severity
    escalation), while the Q-table learns from feedback to optimise the
    cost / quality trade-off for routine tasks.

    Usage::

        router = GATRAModelRouter()
        decision = router.route("tenant-1", AgentType.ADA,
                                TaskType.ANOMALY_SCORING, severity=3.0)
        # After receiving quality feedback:
        router.feedback(decision, reward=0.9)
    """

    def __init__(
        self,
        enable_learning: bool = True,
        budget_limit_usd: Optional[float] = None,
        learning_rate: float = 0.1,
        epsilon: float = 0.15,
    ):
        self.enable_learning = enable_learning
        self.budget_limit = budget_limit_usd

        self._q = _QTable(learning_rate=learning_rate, epsilon=epsilon)
        self._costs: Dict[str, CostRecord] = {}
        self._history: List[RoutingDecision] = []
        self._lock = threading.Lock()

    # -- ROUTE -------------------------------------------------------------
    def route(
        self,
        tenant_id: str,
        agent: AgentType,
        task_type: TaskType,
        severity: float = 0.0,
        request_id: str = "",
    ) -> RoutingDecision:
        """Select a model for the given task. Returns a RoutingDecision."""

        # --- Hard rules (never overridden by Q-table) ---
        # Rule 1: Severity escalation
        if severity >= SEVERITY_ESCALATION_THRESHOLD:
            model = ClaudeModel.OPUS
            reason = f"severity_escalation (severity={severity}>=9)"

        # Rule 2: Task type escalation
        elif task_type in OPUS_TASK_TYPES:
            model = ClaudeModel.OPUS
            reason = f"task_type_escalation ({task_type.value})"

        # --- Learned routing ---
        elif self.enable_learning:
            model = self._q.select(agent, task_type, severity)
            reason = f"q_table (ε={self._q.epsilon:.3f})"

        # --- Static fallback ---
        else:
            model = DEFAULT_ROUTE.get(agent, ClaudeModel.SONNET)
            reason = "static_default"

        # Budget check — downgrade if over budget
        if self.budget_limit is not None:
            cost_rec = self._costs.get(tenant_id)
            if cost_rec and cost_rec.total_cost_usd >= self.budget_limit:
                model = ClaudeModel.HAIKU
                reason = f"budget_cap (${cost_rec.total_cost_usd:.2f}>=${self.budget_limit})"

        decision = RoutingDecision(
            request_id=request_id or f"req-{len(self._history)}",
            tenant_id=tenant_id,
            agent=agent,
            task_type=task_type,
            severity=severity,
            selected_model=model,
            reason=reason,
        )

        with self._lock:
            self._history.append(decision)

        logger.info(
            "Routed %s %s sev=%.1f → %s (%s) tenant=%s",
            agent.value, task_type.value, severity,
            model.value, reason, tenant_id,
        )
        return decision

    # -- FEEDBACK ----------------------------------------------------------
    def feedback(self, decision: RoutingDecision, reward: float):
        """Feed a quality/cost reward back to the Q-table."""
        if not self.enable_learning:
            return
        self._q.update(
            decision.agent,
            decision.task_type,
            decision.severity,
            decision.selected_model,
            reward,
        )
        logger.info(
            "Q-update %s/%s sev=%.1f model=%s reward=%.3f",
            decision.agent.value, decision.task_type.value,
            decision.severity, decision.selected_model.value, reward,
        )

    # -- COST TRACKING -----------------------------------------------------
    def record_usage(
        self,
        tenant_id: str,
        model: ClaudeModel,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Record token usage and return the cost in USD."""
        in_rate, out_rate = MODEL_COST_PER_M[model]
        cost = (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000

        with self._lock:
            if tenant_id not in self._costs:
                self._costs[tenant_id] = CostRecord(tenant_id=tenant_id)
            rec = self._costs[tenant_id]
            rec.total_input_tokens += input_tokens
            rec.total_output_tokens += output_tokens
            rec.total_cost_usd += cost
            rec.request_count += 1
            rec.cost_by_model[model.value] = (
                rec.cost_by_model.get(model.value, 0.0) + cost
            )
        return cost

    def get_cost(self, tenant_id: str) -> CostRecord:
        """Return accumulated cost record for a tenant."""
        return self._costs.get(tenant_id, CostRecord(tenant_id=tenant_id))

    # -- STATS -------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        """Return router statistics."""
        model_counts: Dict[str, int] = {}
        for d in self._history:
            model_counts[d.selected_model.value] = (
                model_counts.get(d.selected_model.value, 0) + 1
            )
        return {
            "total_decisions": len(self._history),
            "model_distribution": model_counts,
            "q_epsilon": self._q.epsilon,
            "tenants_tracked": list(self._costs.keys()),
        }

    # -- CALL (route + Anthropic API) -------------------------------------
    def call(
        self,
        tenant_id: str,
        agent: AgentType,
        task_type: TaskType,
        prompt: str,
        severity: float = 0.0,
        system_prompt: str = "",
        max_tokens: int = 1024,
        request_id: str = "",
    ) -> Dict[str, Any]:
        """
        Route a task and call the selected Claude model via the Anthropic API.

        Returns a dict with ``model``, ``content``, ``input_tokens``,
        ``output_tokens``, ``cost_usd``, and the ``decision`` object.
        """
        if anthropic is None:
            raise ImportError(
                "anthropic package is required for call(). "
                "Install with: pip install anthropic"
            )

        decision = self.route(
            tenant_id, agent, task_type, severity, request_id=request_id,
        )

        client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        )

        messages = [{"role": "user", "content": prompt}]
        kwargs: Dict[str, Any] = {
            "model": decision.selected_model.value,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = client.messages.create(**kwargs)

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = self.record_usage(
            tenant_id, decision.selected_model, input_tokens, output_tokens,
        )

        content = response.content[0].text if response.content else ""

        logger.info(
            "call() %s → %s tokens_in=%d tokens_out=%d cost=$%.4f tenant=%s",
            task_type.value, decision.selected_model.value,
            input_tokens, output_tokens, cost, tenant_id,
        )

        return {
            "model": decision.selected_model.value,
            "content": content,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost,
            "decision": decision,
        }

    def q_table_snapshot(self) -> Dict[str, Dict[str, float]]:
        """Return a copy of the current Q-table for inspection."""
        with self._q.lock:
            return {k: dict(v) for k, v in self._q.q.items()}
