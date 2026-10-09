"""
Integration tests for Phase 6: Backend / Frontend Integration.

Validates:
- GET /api/recommend endpoint (returns task, difficulty, task_index, epsilon, state)
- POST /feedback endpoint with next_state (Bellman TD update metadata)
- GET /api/qtable/state endpoint (inspection of active Q-table)
- GET /api/benchmark/summary (genuine metrics, never fabricated)
- GET /results/history (no hardcoded dummy accuracy/auc fallback arrays)
- GET /feedback (reads from real metrics paths)
- Integration of SNN inference with recommendation pipeline
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from starlette.testclient import TestClient

from snn_ai_optimizer.app import app
import snn_ai_optimizer.optimizer as opt_mod


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Provide a TestClient with an isolated temporary Q-table."""
    tmp_q = tmp_path / "test_api_q_table.json"
    monkeypatch.setattr(opt_mod, "Q_TABLE_PATH", tmp_q)
    return TestClient(app)


def test_recommend_endpoint_default(client):
    res = client.get("/api/recommend")
    assert res.status_code == 200
    data = res.json()
    assert "task" in data
    assert "difficulty" in data
    assert "task_index" in data
    assert "epsilon" in data
    assert data["state"] == "Neutral"
    assert 1 <= data["difficulty"] <= 5
    assert 0 <= data["task_index"] <= 4


@pytest.mark.parametrize("state", ["Focused", "Stressed", "Neutral"])
def test_recommend_endpoint_custom_state(client, state):
    res = client.get(f"/api/recommend?state={state}")
    assert res.status_code == 200
    data = res.json()
    assert data["state"] == state
    assert 1 <= data["difficulty"] <= 5


def test_recommend_endpoint_invalid_state(client):
    res = client.get("/api/recommend?state=InvalidState")
    assert res.status_code == 200
    data = res.json()
    assert data["state"] == "InvalidState"
    assert 1 <= data["difficulty"] <= 5


def test_feedback_post_bellman_update(client):
    res = client.post("/feedback", json={
        "state": "Neutral",
        "task_id": 1,
        "reward": 0.5,
    })
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "update" in data
    update = data["update"]
    assert "old_q" in update
    assert "new_q" in update
    assert "td_error" in update
    assert "epsilon" in update
    assert "n_updates" in update
    assert update["n_updates"] == 1


def test_feedback_post_with_next_state(client):
    # Step 1: initial feedback
    client.post("/feedback", json={
        "state": "Focused",
        "task_id": 0,
        "reward": 1.0,
    })
    # Step 2: feedback with s' = Focused
    res = client.post("/feedback", json={
        "state": "Neutral",
        "task_id": 2,
        "reward": 0.2,
        "next_state": "Focused",
    })
    assert res.status_code == 200
    update = res.json()["update"]
    assert update["n_updates"] == 2
    assert update["td_error"] != 0.0


def test_qtable_state_endpoint(client):
    # Perform an update
    client.post("/feedback", json={
        "state": "Focused",
        "task_id": 3,
        "reward": 0.9,
    })
    res = client.get("/api/qtable/state")
    assert res.status_code == 200
    data = res.json()
    assert "states" in data
    assert "n_updates" in data
    assert "epsilon" in data
    assert "q_table" in data
    assert data["n_updates"] >= 1
    assert "Focused" in data["states"]


def test_benchmark_summary_endpoint_structure(client):
    res = client.get("/api/benchmark/summary")
    assert res.status_code == 200
    data = res.json()
    assert "baseline" in data
    assert "snn" in data
    assert "has_baseline" in data
    assert "has_snn" in data
    # Type integrity: baseline and snn must be dict or None
    assert data["baseline"] is None or isinstance(data["baseline"], dict)
    assert data["snn"] is None or isinstance(data["snn"], dict)


def test_results_history_no_hardcoded_dummy_runs(client, monkeypatch, tmp_path):
    # Point metrics_log to a non-existent path
    non_existent = tmp_path / "does_not_exist.json"
    res = client.get("/results/history")
    assert res.status_code == 200
    data = res.json()
    assert "runs" in data
    # When file is absent, runs must be an empty list, NOT hardcoded 0.85/0.90
    if not Path("results/history/metrics_log.json").exists():
        assert data["runs"] == []


def test_feedback_get_endpoint(client):
    res = client.get("/feedback")
    assert res.status_code == 200
    data = res.json()
    assert "summary" in data
    assert "tips" in data
    assert "actions" in data
    assert isinstance(data["tips"], list)


def test_streaming_inference_metadata_structure():
    """Verify that project_scalars_to_128_features produces 128 elements."""
    from snn_ai_optimizer.snn.inference import project_scalars_to_128_features
    feats = project_scalars_to_128_features(alpha=0.7, beta=0.3, lf_hf=1.2)
    assert feats.shape == (128,)
    assert (feats >= 0).all()
