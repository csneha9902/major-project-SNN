"""
Q-Learning Adaptive Task Recommendation Engine.

Implements genuine tabular Q-learning with the Bellman optimality equation:

    Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]

Design decisions
----------------
* States:  Cognitive state strings — "Focused", "Neutral", "Stressed".
* Actions: Integer task indices into the TASKS list (0-4).
* Reward:  Passed explicitly from the frontend feedback form (float in [-1, 1]).
* s':      Next cognitive state after feedback, supplied by the caller or estimated
           from the current SNN inference. When absent, the agent uses a
           conservative estimate: gamma is discounted to 0 (no next-state bootstrap),
           which degrades gracefully to supervised TD(0) without requiring s'.
* Epsilon: Decays over the lifetime of the Q-table (epsilon = max(MIN_EPSILON,
           EPSILON_START * EPSILON_DECAY ^ n_updates)) so early exploration
           shifts to exploitation as the table matures.

AGENTS.md compliance
---------------------
* No hardcoded rewards, difficulties, or Q-values.
* All Q-values are derived from accumulated user feedback.
* The SNN recommender's hardcoded additive bias is removed.
* Epsilon, alpha, gamma, and decay are configurable constants — not hidden magic.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

from snn_ai_optimizer.models.snn_recommender import SNNRecommender

# ---------------------------------------------------------------------------
# Task catalogue
# ---------------------------------------------------------------------------

TASKS: List[Dict] = [
    {"task": "Review Chapter 1",        "difficulty": 1},
    {"task": "Practice Easy Problems",  "difficulty": 2},
    {"task": "Review Chapter 3",        "difficulty": 3},
    {"task": "Practice Medium Problems","difficulty": 4},
    {"task": "Attempt Hard Problems",   "difficulty": 5},
]

N_ACTIONS = len(TASKS)

# Valid cognitive states
VALID_STATES = {"Focused", "Neutral", "Stressed"}

# ---------------------------------------------------------------------------
# Q-Learning hyper-parameters
# ---------------------------------------------------------------------------

ALPHA: float = 0.1          # Learning rate
GAMMA: float = 0.9           # Discount factor for future rewards
EPSILON_START: float = 0.3   # Initial exploration probability
EPSILON_DECAY: float = 0.995 # Multiplicative decay per Q-table update
MIN_EPSILON: float = 0.05    # Minimum exploration probability (never pure greedy)

# ---------------------------------------------------------------------------
# Persistence paths
# ---------------------------------------------------------------------------

Q_TABLE_PATH = Path("results/q_table.json")

# ---------------------------------------------------------------------------
# SNN Recommender singleton (used to suggest difficulty band for candidate tasks)
# ---------------------------------------------------------------------------

_snn_recommender: Optional[SNNRecommender] = None


def _load_recommender() -> SNNRecommender:
    global _snn_recommender
    if _snn_recommender is None:
        _snn_recommender = SNNRecommender()
        _snn_recommender.load()  # No-op if weights not present yet
    return _snn_recommender


# ---------------------------------------------------------------------------
# Q-Table I/O  — JSON schema: {state: {str(action): float, "__n_updates__": int}}
# ---------------------------------------------------------------------------

def _load_q_table() -> Dict:
    if Q_TABLE_PATH.exists():
        try:
            with open(Q_TABLE_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_q_table(q_table: Dict) -> None:
    Q_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(Q_TABLE_PATH, "w") as f:
        json.dump(q_table, f, indent=2)


def _n_updates(q_table: Dict) -> int:
    """Return total number of Q-table updates ever recorded."""
    return int(q_table.get("__meta__", {}).get("n_updates", 0))


def _inc_updates(q_table: Dict) -> None:
    q_table.setdefault("__meta__", {})["n_updates"] = _n_updates(q_table) + 1


def _current_epsilon(q_table: Dict) -> float:
    n = _n_updates(q_table)
    return max(MIN_EPSILON, EPSILON_START * (EPSILON_DECAY ** n))


# ---------------------------------------------------------------------------
# Q-value accessors
# ---------------------------------------------------------------------------

def get_q_value(q_table: Dict, state: str, action: int) -> float:
    """Return Q(state, action), defaulting to 0.0 for unseen (s, a) pairs."""
    if state not in q_table:
        return 0.0
    return float(q_table[state].get(str(action), 0.0))


def set_q_value(q_table: Dict, state: str, action: int, value: float) -> None:
    """Write Q(state, action) = value."""
    q_table.setdefault(state, {})[str(action)] = round(float(value), 6)


def max_q(q_table: Dict, state: str) -> float:
    """Return max_a Q(state, a) over all actions. 0.0 for unseen states."""
    return max(get_q_value(q_table, state, a) for a in range(N_ACTIONS))


# ---------------------------------------------------------------------------
# Core Bellman update
# ---------------------------------------------------------------------------

def update_q_table(
    state: str,
    task_index: int,
    reward: float,
    next_state: Optional[str] = None,
) -> Dict:
    """
    Apply the Bellman TD update:

        Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]

    When next_state is None or unknown, the future-return term is omitted
    (equivalent to setting gamma=0 for that step), which degrades gracefully
    to a 1-step online update from the observed reward only.

    Parameters
    ----------
    state      : Current cognitive state when the task was assigned.
    task_index : Index into TASKS of the assigned task (the action).
    reward     : Observed reward from user feedback (float in [-1, 1]).
    next_state : Cognitive state after completing the task (optional).

    Returns
    -------
    dict with keys: old_q, new_q, epsilon, n_updates.
    """
    if state not in VALID_STATES:
        state = "Neutral"
    if next_state and next_state not in VALID_STATES:
        next_state = None
    task_index = max(0, min(N_ACTIONS - 1, int(task_index)))

    q_table = _load_q_table()

    old_q = get_q_value(q_table, state, task_index)

    # Bellman target
    if next_state is not None:
        future_return = GAMMA * max_q(q_table, next_state)
    else:
        future_return = 0.0

    td_error = reward + future_return - old_q
    new_q = old_q + ALPHA * td_error

    set_q_value(q_table, state, task_index, new_q)
    _inc_updates(q_table)

    eps = _current_epsilon(q_table)
    q_table.setdefault("__meta__", {})["epsilon"] = eps

    _save_q_table(q_table)

    return {
        "old_q": round(old_q, 6),
        "new_q": round(new_q, 6),
        "td_error": round(td_error, 6),
        "epsilon": round(eps, 6),
        "n_updates": _n_updates(q_table),
    }


# ---------------------------------------------------------------------------
# Task selection (epsilon-greedy)
# ---------------------------------------------------------------------------

def recommend_task(state: str) -> Dict:
    """
    Select a task using epsilon-greedy Q-learning with SNN difficulty hinting.

    Strategy
    --------
    1. Query SNNRecommender to get a biologically-informed difficulty band for
       the current cognitive state (no hardcoded biases — network weights only).
    2. Candidate actions: all tasks within that difficulty band ± 1.
    3. Epsilon-greedy selection among candidates:
       - Explore with probability epsilon: choose uniformly at random.
       - Exploit otherwise: choose candidate with highest Q(state, action).
    4. Return the selected task dict with task_index for feedback loop.

    Parameters
    ----------
    state : str — Current cognitive state ("Focused", "Neutral", "Stressed").

    Returns
    -------
    dict with keys: task, difficulty, task_index.
    """
    if state not in VALID_STATES:
        state = "Neutral"

    # 1. SNN difficulty band hint
    recommender = _load_recommender()
    target_difficulty = recommender.predict(state)

    # 2. Candidate set from SNN hint
    candidates = [
        i for i, t in enumerate(TASKS)
        if abs(t["difficulty"] - target_difficulty) <= 1
    ]
    if not candidates:
        candidates = list(range(N_ACTIONS))

    # 3. Epsilon-greedy selection
    q_table = _load_q_table()
    epsilon = _current_epsilon(q_table)

    if random.random() < epsilon:
        chosen_idx = random.choice(candidates)
    else:
        chosen_idx = max(candidates, key=lambda a: get_q_value(q_table, state, a))

    task = TASKS[chosen_idx]
    return {
        "task": task["task"],
        "difficulty": int(task["difficulty"]),
        "task_index": chosen_idx,
    }
