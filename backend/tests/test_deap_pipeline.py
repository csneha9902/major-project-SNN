"""
Unit tests for the DEAP EEG data pipeline (Task 1).

Fixture strategy
----------------
All tests use a synthetic pickle fixture that is structurally identical to a real
DEAP file:
    data   : shape (40, 40, 8064)  — float32 zeros (values are irrelevant here)
    labels : shape (40, 4)         — deterministic ratings 1–9

Synthetic data is used ONLY inside these test fixtures.
The production loader (DEAPLoader) must never fall back to synthetic data.

Trial ID convention
-------------------
Zero-based (0–39) throughout this pipeline.

Tests covered
-------------
 1. Correct DEAP file parsing from a synthetic fixture
 2. Full data dimensions  (40, 40, 8064)
 3. EEG channel extraction reduces to (40, 32, 8064)
 4. Subject ID derived from filename, not position
 5. Trial IDs are 0-based list [0 … 39]
 6. Arousal labels generated from ratings
 7. arousal ≤ 5 → LOW_AROUSAL
 8. arousal > 5 → HIGH_AROUSAL
 9. Missing DEAP_DATA_DIR raises EnvironmentError
10. Invalid data dimensions raise ValueError
11. Subject-wise split has zero subject overlap
12. Fixed random seed → reproducible split
13. Production loader does not import or call load_mock_data
"""

import importlib
import inspect
import os
import pickle
import sys
import tempfile
from typing import Dict, Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup — ensure backend/ is importable when tests run from that dir
# ---------------------------------------------------------------------------
_BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from snn_ai_optimizer.datasets.deap_config import (
    LOW_AROUSAL,
    HIGH_AROUSAL,
    DEFAULT_AROUSAL_THRESHOLD,
    EXPECTED_TRIALS,
    EXPECTED_TOTAL_CHANNELS,
    EXPECTED_EEG_CHANNELS,
    EXPECTED_SAMPLES,
)
from snn_ai_optimizer.datasets.deap_loader import DEAPLoader
from snn_ai_optimizer.datasets.deap_labels import (
    generate_arousal_labels,
    label_subject_record,
    records_to_flat_arrays,
)
from snn_ai_optimizer.datasets.deap_split import subject_wise_split

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_deap_file(
    tmp_dir: str,
    subject_id: str = "s99",
    n_trials: int = EXPECTED_TRIALS,
    n_channels: int = EXPECTED_TOTAL_CHANNELS,
    n_samples: int = EXPECTED_SAMPLES,
    arousal_vals: np.ndarray | None = None,
) -> str:
    """
    Write a structurally valid synthetic DEAP .dat file to *tmp_dir*.

    This fixture is used ONLY in unit tests.  It is not used by any production
    training loader.

    Returns the path to the written .dat file.
    """
    data = np.zeros((n_trials, n_channels, n_samples), dtype=np.float32)

    # Build ratings: valence=5, arousal=configurable, dominance=5, liking=5
    labels = np.full((n_trials, 4), 5.0, dtype=np.float32)
    if arousal_vals is not None:
        labels[:, 1] = arousal_vals[:n_trials]  # arousal column

    payload = {"data": data, "labels": labels}
    filepath = os.path.join(tmp_dir, f"{subject_id}.dat")
    with open(filepath, "wb") as f:
        pickle.dump(payload, f)
    return filepath


def _make_synthetic_records(n_subjects: int = 6) -> list[Dict[str, Any]]:
    """
    Generate *n_subjects* synthetic subject records for split tests.

    Each record has EXPECTED_TRIALS trials with alternating LOW/HIGH arousal.
    """
    records = []
    for i in range(n_subjects):
        sid = f"s{i + 1:02d}"
        # Alternate arousal above/below threshold across trials
        arousal_vals = np.where(
            np.arange(EXPECTED_TRIALS) % 2 == 0,
            3.0,   # LOW
            7.0,   # HIGH
        ).astype(np.float32)
        ratings = np.zeros((EXPECTED_TRIALS, 4), dtype=np.float32)
        ratings[:, 1] = arousal_vals
        records.append(
            {
                "subject_id": sid,
                "eeg": np.zeros((EXPECTED_TRIALS, EXPECTED_EEG_CHANNELS, EXPECTED_SAMPLES), dtype=np.float32),
                "ratings": ratings,
                "trial_ids": list(range(EXPECTED_TRIALS)),
                "n_total_channels": EXPECTED_TOTAL_CHANNELS,
            }
        )
    return records


# ===========================================================================
# TEST 1 — Correct DEAP file parsing from synthetic fixture
# ===========================================================================

def test_deap_file_parses_correctly():
    """A structurally valid synthetic fixture is loaded without errors."""
    with tempfile.TemporaryDirectory() as tmp:
        _make_synthetic_deap_file(tmp, subject_id="s01")
        loader = DEAPLoader(data_dir=tmp)
        record = loader.load_subject("s01")
    assert record is not None
    assert record["subject_id"] == "s01"


# ===========================================================================
# TEST 2 — Full data dimensions (40, 40, 8064)
# ===========================================================================

def test_raw_data_dimensions():
    """Loaded file exposes the full 40-channel data implicitly via n_total_channels."""
    with tempfile.TemporaryDirectory() as tmp:
        _make_synthetic_deap_file(tmp, subject_id="s01")
        loader = DEAPLoader(data_dir=tmp)
        record = loader.load_subject("s01")
    assert record["n_total_channels"] == EXPECTED_TOTAL_CHANNELS, (
        f"Expected {EXPECTED_TOTAL_CHANNELS} total channels, got {record['n_total_channels']}"
    )
    # EEG shape verifies trial count and samples
    assert record["eeg"].shape[0] == EXPECTED_TRIALS
    assert record["eeg"].shape[2] == EXPECTED_SAMPLES


# ===========================================================================
# TEST 3 — EEG channel extraction → (40, 32, 8064)
# ===========================================================================

def test_eeg_channel_selection():
    """Only the first 32 channels are returned in 'eeg'."""
    with tempfile.TemporaryDirectory() as tmp:
        _make_synthetic_deap_file(tmp, subject_id="s01")
        loader = DEAPLoader(data_dir=tmp)
        record = loader.load_subject("s01")
    assert record["eeg"].shape == (EXPECTED_TRIALS, EXPECTED_EEG_CHANNELS, EXPECTED_SAMPLES), (
        f"Expected EEG shape {(EXPECTED_TRIALS, EXPECTED_EEG_CHANNELS, EXPECTED_SAMPLES)}, "
        f"got {record['eeg'].shape}"
    )


# ===========================================================================
# TEST 4 — Subject ID derived from filename, not from array position
# ===========================================================================

def test_subject_id_from_filename():
    """subject_id in the record matches the filename, not any array index."""
    with tempfile.TemporaryDirectory() as tmp:
        _make_synthetic_deap_file(tmp, subject_id="s22")
        loader = DEAPLoader(data_dir=tmp)
        record = loader.load_subject("s22")
    assert record["subject_id"] == "s22", (
        f"Expected 's22', got '{record['subject_id']}'"
    )


# ===========================================================================
# TEST 5 — Trial IDs are zero-based list [0 … 39]
# ===========================================================================

def test_trial_ids_zero_based():
    """trial_ids is a zero-based list from 0 to 39 inclusive."""
    with tempfile.TemporaryDirectory() as tmp:
        _make_synthetic_deap_file(tmp, subject_id="s01")
        loader = DEAPLoader(data_dir=tmp)
        record = loader.load_subject("s01")
    assert record["trial_ids"] == list(range(EXPECTED_TRIALS)), (
        f"Expected [0..{EXPECTED_TRIALS-1}], got {record['trial_ids']}"
    )


# ===========================================================================
# TEST 6 — Arousal labels generated from ratings
# ===========================================================================

def test_arousal_labels_generated():
    """label_subject_record adds 'arousal_labels' of correct length."""
    with tempfile.TemporaryDirectory() as tmp:
        _make_synthetic_deap_file(tmp, subject_id="s01")
        loader = DEAPLoader(data_dir=tmp)
        record = loader.load_subject("s01")
    labelled = label_subject_record(record)
    assert "arousal_labels" in labelled
    assert len(labelled["arousal_labels"]) == EXPECTED_TRIALS
    # Original ratings are preserved
    assert "ratings" in labelled
    assert labelled["ratings"].shape == (EXPECTED_TRIALS, 4)


# ===========================================================================
# TEST 7 — arousal ≤ 5 → LOW_AROUSAL
# ===========================================================================

def test_low_arousal_label():
    """Arousal ratings of exactly 5 and below produce LOW_AROUSAL."""
    arousal_vals = np.array([1.0, 2.5, 5.0], dtype=np.float32)
    ratings = np.column_stack([
        np.ones(3),    # valence
        arousal_vals,  # arousal
        np.ones(3),    # dominance
        np.ones(3),    # liking
    ]).astype(np.float32)
    labels = generate_arousal_labels(ratings, threshold=5.0)
    assert all(l == LOW_AROUSAL for l in labels), (
        f"Expected all LOW_AROUSAL for arousal ≤ 5, got {labels}"
    )


# ===========================================================================
# TEST 8 — arousal > 5 → HIGH_AROUSAL
# ===========================================================================

def test_high_arousal_label():
    """Arousal ratings strictly above 5 produce HIGH_AROUSAL."""
    arousal_vals = np.array([5.01, 7.0, 9.0], dtype=np.float32)
    ratings = np.column_stack([
        np.ones(3),
        arousal_vals,
        np.ones(3),
        np.ones(3),
    ]).astype(np.float32)
    labels = generate_arousal_labels(ratings, threshold=5.0)
    assert all(l == HIGH_AROUSAL for l in labels), (
        f"Expected all HIGH_AROUSAL for arousal > 5, got {labels}"
    )


# ===========================================================================
# TEST 9 — Missing DEAP_DATA_DIR raises EnvironmentError
# ===========================================================================

def test_missing_env_var_raises_error(monkeypatch):
    """DEAPLoader raises EnvironmentError when DEAP_DATA_DIR is not set."""
    monkeypatch.delenv("DEAP_DATA_DIR", raising=False)
    with pytest.raises(EnvironmentError, match="DEAP_DATA_DIR"):
        DEAPLoader()


# ===========================================================================
# TEST 10 — Invalid data dimensions raise ValueError
# ===========================================================================

def test_invalid_dimensions_raise_value_error():
    """A malformed file (wrong shape) raises ValueError, not a silent fallback."""
    with tempfile.TemporaryDirectory() as tmp:
        # Write file with wrong number of channels (20 instead of 40)
        bad_data = {
            "data": np.zeros((40, 20, 8064), dtype=np.float32),  # BAD
            "labels": np.zeros((40, 4), dtype=np.float32),
        }
        path = os.path.join(tmp, "s01.dat")
        with open(path, "wb") as f:
            pickle.dump(bad_data, f)

        loader = DEAPLoader(data_dir=tmp)
        with pytest.raises(ValueError, match="shape mismatch"):
            loader.load_subject("s01")


# ===========================================================================
# TEST 11 — Subject-wise split has zero subject overlap
# ===========================================================================

def test_split_zero_subject_overlap():
    """Train and test subject sets must be completely disjoint."""
    records = _make_synthetic_records(n_subjects=8)
    flat = records_to_flat_arrays(records)
    result = subject_wise_split(flat, test_size=0.25, random_seed=42)

    train_set = set(result["train_subject_ids"])
    test_set = set(result["test_subject_ids"])

    assert train_set.isdisjoint(test_set), (
        f"Subject leakage! Overlap: {train_set & test_set}"
    )
    assert len(train_set) > 0, "Train set must not be empty"
    assert len(test_set) > 0, "Test set must not be empty"


# ===========================================================================
# TEST 12 — Fixed random seed produces reproducible splits
# ===========================================================================

def test_split_reproducibility():
    """The same random seed must produce identical train/test subject assignments."""
    records = _make_synthetic_records(n_subjects=8)
    flat = records_to_flat_arrays(records)

    r1 = subject_wise_split(flat, test_size=0.25, random_seed=42)
    r2 = subject_wise_split(flat, test_size=0.25, random_seed=42)

    assert r1["train_subject_ids"] == r2["train_subject_ids"], (
        "Train subjects differ between identical seeds"
    )
    assert r1["test_subject_ids"] == r2["test_subject_ids"], (
        "Test subjects differ between identical seeds"
    )


# ===========================================================================
# TEST 13 — Production loader does NOT call load_mock_data
# ===========================================================================

def test_production_loader_does_not_call_mock_data():
    """
    DEAPLoader source code must not call load_mock_data.

    The production loader must raise an error when data is unavailable rather
    than silently substituting synthetic data.
    """
    import snn_ai_optimizer.datasets.deap_loader as loader_module

    source = inspect.getsource(loader_module)
    assert "load_mock_data" not in source, (
        "deap_loader.py must NOT call load_mock_data(). "
        "The production loader must never substitute synthetic data."
    )
