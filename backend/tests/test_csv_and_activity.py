import io
import json
import os
import tempfile
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient

from snn_ai_optimizer.app import app
from snn_ai_optimizer.upload.csv_processor import process_csv_file
from snn_ai_optimizer.activity_history import (
    load_calendar_history,
    record_session,
    update_observation_note,
    update_date_status,
)


@pytest.fixture
def client():
    return TestClient(app)


def test_process_csv_raw_multichannel_headerless(tmp_path):
    # Create synthetic 19-channel raw EEG file (128 samples, 19 channels)
    data = np.random.normal(0, 10.0, size=(128, 19))
    csv_file = tmp_path / "raw_19ch.csv"
    np.savetxt(csv_file, data, delimiter=",")

    res = process_csv_file(csv_file, upload_id="test-19ch")
    assert res["upload_id"] == "test-19ch"
    assert res["metadata"]["n_channels"] == 19
    assert len(res["time_series"]) > 0
    assert "alpha" in res["time_series"][0]
    assert "beta" in res["time_series"][0]
    assert "cognitive_state" in res["time_series"][0]
    assert "arousal_label" in res["time_series"][0]


def test_process_csv_precomputed_features(tmp_path):
    # Create synthetic feature CSV with alpha and beta headers
    csv_file = tmp_path / "features.csv"
    with open(csv_file, "w") as f:
        f.write("timestamp,alpha,beta,heart_rate\n")
        for i in range(20):
            f.write(f"{i * 0.2},{0.5 + i*0.01},{0.6 - i*0.01},72\n")

    res = process_csv_file(csv_file, upload_id="test-feat")
    assert res["upload_id"] == "test-feat"
    assert len(res["time_series"]) == 20
    assert res["time_series"][0]["cognitive_state"] in ("Focused", "Neutral", "Stressed", "Low Arousal", "High Arousal")


def test_activity_history_persistence(tmp_path):
    # Test recording a session
    entry = record_session(
        date_str="2026-10-23",
        state="Low Arousal",
        label="Test Session",
        avg_beta=0.6,
        avg_alpha=0.7,
        avg_hr=70,
        duration_str="4m 15s",
        stress_spikes=2,
        snn_score=88,
        notes="Testing persistent activity recording",
    )
    assert entry["state"] == "Low Arousal"
    assert entry["snnScore"] == 88

    # Test observation note update
    updated = update_observation_note("2026-10-23", "Clinician note: patient displayed sustained relaxation.")
    assert "sustained relaxation" in updated["notes"]

    # Test state update
    overridden = update_date_status("2026-10-23", "High Arousal")
    assert overridden["state"] == "High Arousal"


def test_api_progress_endpoints(client):
    # GET history
    res = client.get("/api/progress/history")
    assert res.status_code == 200
    assert isinstance(res.json(), dict)

    # POST state
    res_state = client.post("/api/progress/state", json={"date": "2026-10-23", "state": "Low Arousal"})
    assert res_state.status_code == 200
    assert res_state.json()["entry"]["state"] == "Low Arousal"

    # POST note
    res_note = client.post("/api/progress/note", json={"date": "2026-10-23", "note": "Integration test note"})
    assert res_note.status_code == 200
    assert res_note.json()["entry"]["notes"] == "Integration test note"
