"""
Unit tests for Phase 5: Genuine Bellman Q-Learning Recommendation.

Validates:
- Bellman TD update: Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
- Proper next-state bootstrapping when s' is provided.
- Graceful degradation when s' is None (gamma * max_q = 0).
- Epsilon-greedy exploration / exploitation balance.
- Epsilon decay over Q-table lifetime.
- Absence of hardcoded bias in SNNRecommender.predict().
- Q-table persistence across loads.
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pytest

_BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from snn_ai_optimizer.models.snn_recommender import SNNRecommender
from snn_ai_optimizer.optimizer import (
    ALPHA,
    EPSILON_DECAY,
    EPSILON_START,
    GAMMA,
    MIN_EPSILON,
    N_ACTIONS,
    TASKS,
    VALID_STATES,
    _current_epsilon,
    _n_updates,
    get_q_value,
    max_q,
    recommend_task,
    set_q_value,
    update_q_table,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_q_table(tmp_path, monkeypatch):
    """Redirect Q_TABLE_PATH to a fresh temp file for each test."""
    from pathlib import Path
    import snn_ai_optimizer.optimizer as opt_mod

    tmp_q = tmp_path / "q_table.json"
    monkeypatch.setattr(opt_mod, "Q_TABLE_PATH", tmp_q)
    return tmp_q


# ---------------------------------------------------------------------------
# Test 1: Bellman update with no next_state degrades to observed-reward update
# ---------------------------------------------------------------------------
def test_bellman_update_no_next_state(clean_q_table):
    info = update_q_table("Neutral", 2, reward=0.8, next_state=None)
    # With old_q=0, no next-state, expected: new_q = 0 + 0.1*(0.8 + 0 - 0) = 0.08
    assert math.isclose(info["new_q"], ALPHA * 0.8, rel_tol=1e-5), (
        f"Expected new_q ≈ {ALPHA * 0.8}, got {info['new_q']}"
    )
    assert info["old_q"] == 0.0


# ---------------------------------------------------------------------------
# Test 2: Bellman update with next_state bootstraps future returns
# ---------------------------------------------------------------------------
def test_bellman_update_with_next_state(clean_q_table):
    # First: seed Q("Focused", 0) = 0.9 by doing a manual update
    update_q_table("Focused", 0, reward=1.0, next_state=None)
    # Now: update Q("Neutral", 1) with s'="Focused"
    # max_q("Focused") should be > 0 from step above
    info = update_q_table("Neutral", 1, reward=0.0, next_state="Focused")
    # future_return = GAMMA * max_q("Focused") > 0
    assert info["new_q"] > 0.0, "Should bootstrap future return from Q('Focused', *)"
    assert info["td_error"] > 0.0


# ---------------------------------------------------------------------------
# Test 3: TD error is exactly r + gamma*max_q(s') - old_q
# ---------------------------------------------------------------------------
def test_td_error_formula(clean_q_table, monkeypatch):
    import snn_ai_optimizer.optimizer as opt_mod

    # Manually inject a known Q-value for next state
    qt = {}
    set_q_value(qt, "Focused", 3, 0.5)
    with open(clean_q_table, "w") as f:
        json.dump(qt, f)

    info = update_q_table("Neutral", 1, reward=0.2, next_state="Focused")
    expected_future = GAMMA * 0.5
    expected_td = 0.2 + expected_future - 0.0   # old_q("Neutral",1) = 0
    expected_new_q = 0.0 + ALPHA * expected_td
    assert math.isclose(info["td_error"], expected_td, rel_tol=1e-5), (
        f"TD error should be {expected_td:.6f}, got {info['td_error']}"
    )
    assert math.isclose(info["new_q"], expected_new_q, rel_tol=1e-5), (
        f"new_q should be {expected_new_q:.6f}, got {info['new_q']}"
    )


# ---------------------------------------------------------------------------
# Test 4: Q-values persist on disk after update
# ---------------------------------------------------------------------------
def test_q_table_persistence(clean_q_table):
    update_q_table("Stressed", 0, reward=0.5)
    with open(clean_q_table, "r") as f:
        data = json.load(f)
    assert "Stressed" in data
    assert "0" in data["Stressed"]
    assert data["Stressed"]["0"] != 0.0


# ---------------------------------------------------------------------------
# Test 5: get_q_value returns 0.0 for completely unseen (state, action) pairs
# ---------------------------------------------------------------------------
def test_get_q_value_unseen():
    qt = {}
    assert get_q_value(qt, "Focused", 3) == 0.0


# ---------------------------------------------------------------------------
# Test 6: max_q returns the highest Q-value across all actions for a state
# ---------------------------------------------------------------------------
def test_max_q_returns_maximum():
    qt = {}
    set_q_value(qt, "Focused", 0, 0.3)
    set_q_value(qt, "Focused", 1, 0.7)
    set_q_value(qt, "Focused", 2, 0.1)
    assert math.isclose(max_q(qt, "Focused"), 0.7)


# ---------------------------------------------------------------------------
# Test 7: Epsilon decays over updates
# ---------------------------------------------------------------------------
def test_epsilon_decay_over_updates(clean_q_table):
    # Perform 20 updates and check that epsilon decreases
    for i in range(20):
        update_q_table("Neutral", i % N_ACTIONS, reward=0.0)
    import snn_ai_optimizer.optimizer as opt_mod
    qt = {}  # fresh
    qt_loaded = json.load(open(clean_q_table))
    eps_after = _current_epsilon(qt_loaded)
    eps_before = EPSILON_START
    assert eps_after < eps_before, f"Epsilon should have decayed: {eps_before} -> {eps_after}"


# ---------------------------------------------------------------------------
# Test 8: Epsilon never drops below MIN_EPSILON
# ---------------------------------------------------------------------------
def test_epsilon_floor(clean_q_table):
    # Many updates — epsilon should converge to MIN_EPSILON
    for _ in range(5000):
        update_q_table("Focused", 0, reward=0.0)
    qt_loaded = json.load(open(clean_q_table))
    eps = _current_epsilon(qt_loaded)
    assert eps >= MIN_EPSILON - 1e-9, f"Epsilon {eps} went below MIN_EPSILON {MIN_EPSILON}"


# ---------------------------------------------------------------------------
# Test 9: recommend_task returns a valid task dict for each state
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("state", ["Focused", "Neutral", "Stressed"])
def test_recommend_task_valid_output(state, clean_q_table):
    rec = recommend_task(state)
    assert "task" in rec and "difficulty" in rec and "task_index" in rec
    assert rec["difficulty"] in range(1, 6)
    assert 0 <= rec["task_index"] < N_ACTIONS
    assert rec["task"] == TASKS[rec["task_index"]]["task"]


# ---------------------------------------------------------------------------
# Test 10: recommend_task handles unrecognised state gracefully
# ---------------------------------------------------------------------------
def test_recommend_task_invalid_state(clean_q_table):
    rec = recommend_task("CognitivelyImpaired")  # invalid per AGENTS.md
    assert rec["difficulty"] in range(1, 6)


# ---------------------------------------------------------------------------
# Test 11: SNNRecommender.predict has NO hardcoded bias injected
# ---------------------------------------------------------------------------
def test_snn_recommender_no_hardcoded_bias():
    """
    SNNRecommender.predict must determine its output from spike counts alone.
    The old code injected a hardcoded np.zeros bias array with manual index
    increments (bias[3] += 1, etc). Verify these are absent.
    """
    import inspect
    import snn_ai_optimizer.models.snn_recommender as mod
    src = inspect.getsource(mod.SNNRecommender.predict)
    assert "bias[" not in src, (
        "Hardcoded index bias (bias[N] += 1) found in SNNRecommender.predict — must be removed."
    )
    assert "np.zeros(self.output_size)" not in src or "spike_counts + bias" not in src, (
        "Old bias addition pattern found in SNNRecommender.predict — must be removed."
    )


# ---------------------------------------------------------------------------
# Test 12: SNNRecommender.predict returns int in 1–5 for all states
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("state", ["Focused", "Neutral", "Stressed"])
def test_snn_recommender_valid_output(state):
    rec = SNNRecommender()
    result = rec.predict(state)
    assert isinstance(result, int)
    assert 1 <= result <= 5, f"SNNRecommender.predict('{state}') returned {result}"


# ---------------------------------------------------------------------------
# Test 13: update_q_table n_updates increments correctly
# ---------------------------------------------------------------------------
def test_n_updates_increments(clean_q_table):
    for i in range(5):
        info = update_q_table("Neutral", 0, reward=0.1)
    assert info["n_updates"] == 5
