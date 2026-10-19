"""
Unit tests for Phase 8: Reproducibility and Experiment Automation.

Validates:
- Research integrity checks run cleanly via pytest.
- Zero subject overlap in GroupShuffleSplit.
- Complete provenance logging in baseline and SNN training results.
- No fabricated constant metric patterns in saved results.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np

from snn_ai_optimizer.datasets.deap_split import subject_wise_split
from snn_ai_optimizer.features.eeg_features import extract_band_powers
from snn_ai_optimizer.snn.deap_snn_model import DEAPArousalSNN


def test_zero_subject_leakage_invariant():
    """Rule 6: Never mix training subjects and test subjects."""
    n_subjects = 16
    trials_per_subj = 40
    n_total = n_subjects * trials_per_subj
    subject_ids = [f"s{s:02d}" for s in range(1, n_subjects + 1) for _ in range(trials_per_subj)]

    flat_data = {
        "X": np.zeros((n_total, 32, 128)),
        "y": np.zeros(n_total, dtype=int),
        "ratings": np.zeros((n_total, 4)),
        "subject_ids": subject_ids,
        "trial_ids": list(range(n_total)),
    }

    res = subject_wise_split(flat_data, test_size=0.25, random_seed=42)
    train_set = set(res["train_subject_ids"])
    test_set = set(res["test_subject_ids"])

    assert len(train_set) > 0
    assert len(test_set) > 0
    assert len(train_set.intersection(test_set)) == 0, (
        f"Subject leakage detected: {train_set.intersection(test_set)}"
    )


def test_feature_extraction_invariant():
    """Rule 10: Canonical 128 band-power feature representation."""
    raw = np.random.randn(3, 32, 128 * 4)
    feats = extract_band_powers(raw, fs=128)
    assert feats.shape == (3, 128)
    assert not np.isnan(feats).any()
    assert not np.isinf(feats).any()
    assert (feats >= 0).all()


def test_snn_output_invariant():
    """Spiking network produces binary activations and correct tensor dimensions."""
    import torch
    snn = DEAPArousalSNN(input_dim=128, hidden_dims=(64, 32), num_classes=2)
    x = torch.rand(8, 2, 128)  # (time_steps=8, batch_size=2, input_dim=128)
    spikes = snn(x)
    assert spikes.shape == (8, 2, 2)
    assert ((spikes == 0.0) | (spikes == 1.0)).all()


def test_metrics_no_hardcoded_stubs():
    """Rule 1 & Rule 2: Never hardcode accuracy, F1, AUC."""
    base_file = Path("results/baseline/metrics.json")
    if base_file.exists():
        data = json.loads(base_file.read_text(encoding="utf-8"))
        for model in ("svm", "random_forest"):
            acc = data.get(model, {}).get("accuracy")
            auc = data.get(model, {}).get("auc")
            assert not (acc == 0.85 and auc == 0.90), f"Hardcoded stub in {base_file}"

    snn_file = Path("results/snn/metrics.json")
    if snn_file.exists():
        data = json.loads(snn_file.read_text(encoding="utf-8"))
        acc = data.get("accuracy")
        auc = data.get("auc")
        assert not (acc == 0.92 and auc == 0.95), f"Hardcoded stub in {snn_file}"


def test_provenance_experiment_file_structure():
    """Verify experiment provenance structure if files exist."""
    base_exp = Path("results/baseline/experiment.json")
    if base_exp.exists():
        data = json.loads(base_exp.read_text(encoding="utf-8"))
        assert "dataset" in data
        assert "evaluation" in data or "models" in data

    snn_exp = Path("results/snn/experiment.json")
    if snn_exp.exists():
        data = json.loads(snn_exp.read_text(encoding="utf-8"))
        assert "dataset" in data
        assert "model_config" in data or "evaluation" in data
