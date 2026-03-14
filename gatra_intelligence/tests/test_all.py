"""
GATRA Intelligence — Full Test Suite (12 tests)

Covers:
- AgentDB: HNSW store/recall, tier promotion, HMAC chain, Ebbinghaus decay, EWC++
- GATRAModelRouter: static routing, severity escalation, Q-learning, cost tracking
- GATRASelfLearningLoop: full cycle, PPO reward scoring, distillation
"""

import sys
import time
import traceback

import numpy as np

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from gatra_intelligence.memory.agent_db import (
    AgentDB,
    MemoryTier,
    MutationGuard,
    ebbinghaus_retention,
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

DIM = 384
TENANT = "test-tenant-1"
TENANT_B = "test-tenant-2"

PASS = 0
FAIL = 0
RESULTS: list = []


def _vec(seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(DIM).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def run_test(name: str, fn):
    global PASS, FAIL
    try:
        fn()
        PASS += 1
        RESULTS.append((name, True, ""))
        print(f"  ✓ {name}")
    except Exception as exc:
        FAIL += 1
        tb = traceback.format_exc()
        RESULTS.append((name, False, tb))
        print(f"  ✗ {name}\n    {exc}")


# ===================================================================
# TEST 1 — AgentDB store and recall
# ===================================================================
def test_agentdb_store_recall():
    db = AgentDB(dim=DIM)
    rid = db.store(TENANT, _vec(1), {"alert_id": "A-1"}, tier=MemoryTier.EPISODIC)
    db.store(TENANT, _vec(2), {"alert_id": "A-2"}, tier=MemoryTier.EPISODIC)
    db.store(TENANT, _vec(3), {"alert_id": "A-3"}, tier=MemoryTier.WORKING)

    results = db.recall(TENANT, _vec(1), k=2)
    assert len(results) >= 1, f"Expected >=1 result, got {len(results)}"
    assert results[0].record_id == rid, "Closest match should be the same vector"
    stats = db.stats(TENANT)
    assert stats["total"] == 3


# ===================================================================
# TEST 2 — Tenant isolation
# ===================================================================
def test_tenant_isolation():
    db = AgentDB(dim=DIM)
    db.store(TENANT, _vec(10), {"data": "tenant-1"}, tier=MemoryTier.EPISODIC)
    db.store(TENANT_B, _vec(20), {"data": "tenant-2"}, tier=MemoryTier.EPISODIC)

    r1 = db.recall(TENANT, _vec(10), k=5)
    r2 = db.recall(TENANT_B, _vec(20), k=5)

    assert all(r.tenant_id == TENANT for r in r1), "Tenant A leaked into B"
    assert all(r.tenant_id == TENANT_B for r in r2), "Tenant B leaked into A"


# ===================================================================
# TEST 3 — HMAC proof chain integrity
# ===================================================================
def test_hmac_chain():
    db = AgentDB(dim=DIM)
    db.store(TENANT, _vec(1), {"step": 1}, tier=MemoryTier.EPISODIC)
    db.store(TENANT, _vec(2), {"step": 2}, tier=MemoryTier.EPISODIC)
    db.store(TENANT, _vec(3), {"step": 3}, tier=MemoryTier.EPISODIC)

    assert db.verify_chain(TENANT), "Chain should be valid"

    # Tamper with a record
    store = db._get_store(TENANT)
    first_rid = list(store.records.keys())[0]
    store.records[first_rid].payload["step"] = 999

    assert not db.verify_chain(TENANT), "Tampered chain should fail"


# ===================================================================
# TEST 4 — MutationGuard sign / verify
# ===================================================================
def test_mutation_guard():
    guard = MutationGuard(secret=b"test-secret")
    payload = {"alert_id": "X-42", "severity": 8}
    sig = guard.sign(payload, prev_hmac="")
    assert guard.verify(payload, sig, prev_hmac="")
    assert not guard.verify({"alert_id": "X-42", "severity": 9}, sig, prev_hmac="")


# ===================================================================
# TEST 5 — Ebbinghaus forgetting curve
# ===================================================================
def test_ebbinghaus_curve():
    r0 = ebbinghaus_retention(0)
    assert abs(r0 - 1.0) < 1e-6, "Retention at t=0 should be 1.0"

    half_life = 3600.0
    r_half = ebbinghaus_retention(half_life, half_life)
    assert abs(r_half - 0.5) < 0.01, f"Retention at half-life should be ~0.5, got {r_half}"

    r_late = ebbinghaus_retention(half_life * 10, half_life)
    assert r_late < 0.01, "Retention after 10 half-lives should be near zero"


# ===================================================================
# TEST 6 — EWC++ consolidation and penalty
# ===================================================================
def test_ewc_consolidation():
    db = AgentDB(dim=DIM, ewc_lambda=100.0)
    for i in range(5):
        db.store(TENANT, _vec(i), {"idx": i}, tier=MemoryTier.SEMANTIC)

    task_id = db.consolidate(TENANT)
    assert task_id, "Consolidation should return a task ID"

    # Penalty should be zero at the consolidated centroid
    store = db._get_store(TENANT)
    embeddings = np.stack([r.embedding for r in store.records.values()])
    centroid = embeddings.mean(axis=0)
    penalty_at_center = db.ewc_penalty(TENANT, centroid)
    assert penalty_at_center < 1.0, f"Penalty at centroid should be small, got {penalty_at_center}"

    # Penalty should increase for a distant vector
    far_vec = np.ones(DIM, dtype=np.float32) * 10.0
    penalty_far = db.ewc_penalty(TENANT, far_vec)
    assert penalty_far > penalty_at_center, "Far vector should have higher penalty"


# ===================================================================
# TEST 7 — Model router static rules + severity escalation
# ===================================================================
def test_router_static_and_escalation():
    router = GATRAModelRouter(enable_learning=False)

    # ADA → Haiku
    d1 = router.route(TENANT, AgentType.ADA, TaskType.ANOMALY_SCORING, severity=3.0)
    assert d1.selected_model == ClaudeModel.HAIKU, f"ADA should route to Haiku, got {d1.selected_model}"

    # TAA → Sonnet
    d2 = router.route(TENANT, AgentType.TAA, TaskType.ALERT_TRIAGE, severity=5.0)
    assert d2.selected_model == ClaudeModel.SONNET, f"TAA should route to Sonnet, got {d2.selected_model}"

    # Deep forensics → Opus
    d3 = router.route(TENANT, AgentType.TAA, TaskType.DEEP_FORENSICS, severity=5.0)
    assert d3.selected_model == ClaudeModel.OPUS, f"Deep forensics should route to Opus, got {d3.selected_model}"

    # Severity >= 9 → Opus regardless of agent
    d4 = router.route(TENANT, AgentType.ADA, TaskType.ANOMALY_SCORING, severity=9.5)
    assert d4.selected_model == ClaudeModel.OPUS, f"Severity 9.5 should escalate to Opus, got {d4.selected_model}"

    # Incident report → Opus
    d5 = router.route(TENANT, AgentType.RVA, TaskType.INCIDENT_REPORT, severity=2.0)
    assert d5.selected_model == ClaudeModel.OPUS, f"Incident report should route to Opus, got {d5.selected_model}"


# ===================================================================
# TEST 8 — Q-table learning updates
# ===================================================================
def test_q_learning():
    router = GATRAModelRouter(enable_learning=True, epsilon=0.0)

    # Force a decision and reward it
    decision = router.route(TENANT, AgentType.CRA, TaskType.CONTAINMENT, severity=5.0)
    initial_q = router.q_table_snapshot()

    # Positive reward should increase Q value for chosen model
    router.feedback(decision, reward=1.0)
    updated_q = router.q_table_snapshot()

    state_key = f"cra|containment|medium"
    model_key = decision.selected_model.value
    assert updated_q[state_key][model_key] > initial_q[state_key][model_key], \
        "Positive reward should increase Q value"


# ===================================================================
# TEST 9 — Cost tracking and budget cap
# ===================================================================
def test_cost_tracking():
    router = GATRAModelRouter(enable_learning=False, budget_limit_usd=0.10)

    # Record usage that exceeds budget
    cost = router.record_usage(TENANT, ClaudeModel.OPUS, 100_000, 50_000)
    assert cost > 0, "Cost should be positive"

    cost_rec = router.get_cost(TENANT)
    assert cost_rec.request_count == 1
    assert cost_rec.total_cost_usd > 0

    # Blow past budget
    router.record_usage(TENANT, ClaudeModel.OPUS, 5_000_000, 5_000_000)

    # Next route should downgrade to Haiku
    d = router.route(TENANT, AgentType.TAA, TaskType.ALERT_TRIAGE, severity=5.0)
    assert d.selected_model == ClaudeModel.HAIKU, \
        f"Over budget should force Haiku, got {d.selected_model}"
    assert "budget_cap" in d.reason


# ===================================================================
# TEST 10 — PPO reward signal scoring
# ===================================================================
def test_ppo_reward_signal():
    signal = PPORewardSignal()

    tp = signal.score({"outcome": "true_positive"})
    assert tp == 1.0, f"TP reward should be 1.0, got {tp}"

    fp = signal.score({"outcome": "false_positive"})
    assert fp == -1.0, f"FP reward should be -1.0, got {fp}"

    mid = signal.score({"severity": 5})
    assert abs(mid) < 0.01, f"Severity=5 should be near 0, got {mid}"

    high = signal.score({"severity": 10})
    assert high == 1.0, f"Severity=10 should be 1.0, got {high}"


# ===================================================================
# TEST 11 — Self-learning loop full cycle
# ===================================================================
def test_self_learning_cycle():
    db = AgentDB(dim=DIM)

    # Seed episodic memories with outcomes
    for i in range(10):
        outcome = "true_positive" if i % 3 == 0 else "unknown"
        db.store(
            TENANT, _vec(i),
            {"alert_id": f"A-{i}", "outcome": outcome, "severity": 7,
             "alert_type": "brute_force"},
            tier=MemoryTier.EPISODIC,
        )

    loop = GATRASelfLearningLoop(db, retrieve_k=10, judge_threshold=0.3)
    result = loop.run_cycle(TENANT, _vec(0), force_consolidate=True)

    assert result.retrieved_count > 0, "Should retrieve memories"
    assert result.judged_kept >= 0, "Should have judged results"
    assert "retrieve" in result.phases_completed
    assert "judge" in result.phases_completed
    assert "distill" in result.phases_completed
    assert "consolidate" in result.phases_completed
    assert result.duration_seconds >= 0


# ===================================================================
# TEST 12 — Memory tier promotion flow
# ===================================================================
def test_promotion_flow():
    db = AgentDB(dim=DIM)

    rid = db.store(TENANT, _vec(42), {"alert_id": "P-1"}, tier=MemoryTier.WORKING)
    rec = db._get_store(TENANT).records[rid]
    assert rec.tier == MemoryTier.WORKING

    new_tier = db.promote(TENANT, rid)
    assert new_tier == MemoryTier.EPISODIC

    new_tier = db.promote(TENANT, rid)
    assert new_tier == MemoryTier.SEMANTIC

    # Promoting semantic stays semantic
    new_tier = db.promote(TENANT, rid)
    assert new_tier == MemoryTier.SEMANTIC


# ===================================================================
# RUNNER
# ===================================================================
def main():
    print("\n╔══════════════════════════════════════════════╗")
    print("║   GATRA Intelligence — Test Suite (12 tests) ║")
    print("╚══════════════════════════════════════════════╝\n")

    tests = [
        ("1.  AgentDB store & recall",         test_agentdb_store_recall),
        ("2.  Tenant isolation",               test_tenant_isolation),
        ("3.  HMAC proof chain integrity",     test_hmac_chain),
        ("4.  MutationGuard sign/verify",      test_mutation_guard),
        ("5.  Ebbinghaus forgetting curve",    test_ebbinghaus_curve),
        ("6.  EWC++ consolidation & penalty",  test_ewc_consolidation),
        ("7.  Router static + escalation",     test_router_static_and_escalation),
        ("8.  Q-table learning updates",       test_q_learning),
        ("9.  Cost tracking & budget cap",     test_cost_tracking),
        ("10. PPO reward signal scoring",      test_ppo_reward_signal),
        ("11. Self-learning full cycle",       test_self_learning_cycle),
        ("12. Memory tier promotion flow",     test_promotion_flow),
    ]

    for name, fn in tests:
        run_test(name, fn)

    print(f"\n{'─' * 48}")
    if FAIL == 0:
        print(f"✓ All {PASS} tests PASSED")
    else:
        print(f"✗ {FAIL} FAILED, {PASS} passed")
        print("\nFailed tests:")
        for name, ok, tb in RESULTS:
            if not ok:
                print(f"\n--- {name} ---")
                print(tb)

    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
